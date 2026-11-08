#!/usr/bin/env python3
"""Prepare full OBJ copies and create one heroleg26 actor. No physics steps.

Run in the personal heroleg-isaac environment. Original repository assets are
read-only inputs; generated files are placed alongside this script.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import struct
import sys
import time
import xml.etree.ElementTree as ET


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def user_path(path):
    home = Path.home().resolve()
    path = path.expanduser().resolve()
    if home not in path.parents:
        raise RuntimeError("Write path must be inside your own HOME: " + str(path))
    return path


def prepare(source, output):
    import numpy as np

    root = ET.parse(str(source)).getroot()
    paths = {}
    for element in root.iter("mesh"):
        name = element.attrib["filename"]
        if "://" in name:
            raise ValueError("Use heroleg26_sim/urdf/heroleg26.urdf with local mesh paths")
        path = (source.parent / name).resolve()
        if not path.is_file() or path.suffix.lower() != ".stl":
            raise ValueError("Expected an existing full-resolution STL: " + str(path))
        paths[name] = path
    if len(paths) != 7:
        raise ValueError("Expected 7 independent mesh resources")
    original_hashes = {str(p): sha256(p) for p in [source] + list(paths.values())}
    manifest_path = output / "preparation.json"
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if saved.get("original_sha256") == original_hashes:
            generated = saved.get("generated_sha256", {})
            if generated and all(Path(p).is_file() and sha256(Path(p)) == h
                                 for p, h in generated.items()):
                print("Reusing checked OBJ copies.", flush=True)
                return saved

    mesh_dir = output / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    converted = {}
    records = []
    for old_name, path in paths.items():
        print("Converting " + path.name + " (full resolution) ...", flush=True)
        with path.open("rb") as stream:
            header = stream.read(84)
        if len(header) != 84:
            raise ValueError("Invalid STL header: " + str(path))
        count = struct.unpack_from("<I", header, 80)[0]
        if path.stat().st_size != 84 + count * 50:
            raise ValueError("Expected binary STL; source was not changed: " + str(path))
        dtype = np.dtype([("normal", "<f4", (3,)),
                          ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        raw = np.fromfile(str(path), dtype=dtype, offset=84, count=count)
        triangles = raw["vertices"]
        if not np.isfinite(triangles).all():
            raise ValueError("Non-finite mesh coordinates: " + str(path))
        vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0,
                                      return_inverse=True)
        faces = inverse.reshape(-1, 3)
        np.testing.assert_array_equal(vertices[faces], triangles)
        target = mesh_dir / (path.stem + ".obj")
        with target.open("w", encoding="ascii") as stream:
            stream.write("# Full-resolution STL geometry. Coordinates in metres.\n")
            # Nine significant digits round-trip all original float32 values.
            np.savetxt(stream, vertices, fmt="v %.9g %.9g %.9g")
            np.savetxt(stream, faces + 1, fmt="f %d %d %d")
        with target.open("r", encoding="ascii") as stream:
            restored = np.loadtxt(stream, dtype=np.float32, usecols=(1, 2, 3),
                                  max_rows=len(vertices), skiprows=1)
            restored_faces = np.loadtxt(stream, dtype=np.int64, usecols=(1, 2, 3)) - 1
        np.testing.assert_array_equal(restored_faces, faces)
        np.testing.assert_array_equal(restored[restored_faces], triangles)
        converted[old_name] = "meshes/" + target.name
        records.append({"source": str(path), "obj": str(target),
                        "triangles": count, "float32_geometry_exact": True})
        print("  OK: {:,} triangles preserved".format(count), flush=True)

    generated_root = copy.deepcopy(root)
    for element in generated_root.iter("mesh"):
        element.set("filename", converted[element.attrib["filename"]])
    physical_a, physical_b = copy.deepcopy(root), copy.deepcopy(generated_root)
    for tree in (physical_a, physical_b):
        for element in tree.iter("mesh"):
            element.set("filename", "")
    if ET.tostring(physical_a) != ET.tostring(physical_b):
        raise RuntimeError("Unexpected non-path XML change")
    urdf = output / "heroleg26_obj.urdf"
    ET.ElementTree(generated_root).write(str(urdf), encoding="utf-8", xml_declaration=True)
    for path, old_hash in original_hashes.items():
        if sha256(Path(path)) != old_hash:
            raise RuntimeError("Original asset changed: " + path)
    manifest = {"urdf": str(urdf), "full_resolution": True,
                "physical_xml_unchanged": True, "original_sha256": original_hashes,
                "resources": records,
                "generated_sha256": {str(p): sha256(p)
                                     for p in [urdf] + [Path(r["obj"]) for r in records]}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def show_static_viewer(gym, sim, env, gymapi, fps):
    camera = gymapi.CameraProperties()
    camera.width, camera.height = 960, 540
    viewer = gym.create_viewer(sim, camera)
    if viewer is None:
        raise RuntimeError("Cannot open Viewer in this desktop session; save the SDK error output")
    frames = 0
    try:
        gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(1.5, -1.5, 1.4),
                                  gymapi.Vec3(0, 0, 1.1))
        print("Static Viewer: close the window or press Ctrl+C to finish.", flush=True)
        try:
            while not gym.query_viewer_has_closed(viewer):
                start = time.monotonic()
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, False)
                frames += 1
                if frames == 1:
                    print("Viewer rendering started; physics remains paused.", flush=True)
                time.sleep(max(0, 1.0 / fps - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass
    finally:
        gym.destroy_viewer(viewer)
    if frames == 0:
        raise RuntimeError("Viewer closed before a frame could be drawn")
    return frames


def check_actor(manifest, output, viewer=False, compute_device=0,
                graphics_device=0, fps=10):
    from isaacgym import gymapi

    source = Path(manifest["urdf"])
    root = ET.parse(str(source)).getroot()
    masses = {link.attrib["name"]: float(link.find("inertial/mass").attrib["value"])
              for link in root.findall("link")}
    expected_dofs = {j.attrib["name"] for j in root.findall("joint")
                     if j.attrib["type"] != "fixed"}
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.physx.use_gpu = True
    params.use_gpu_pipeline = False
    sim = gym.create_sim(compute_device, graphics_device if viewer else -1,
                         gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Failed to create PhysX simulation")
    try:
        options = gymapi.AssetOptions()
        options.fix_base_link = False
        options.collapse_fixed_joints = False
        options.override_com = False
        options.override_inertia = False
        options.vhacd_enabled = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        print("Loading your robot into Isaac Gym ...", flush=True)
        asset = gym.load_asset(sim, str(source.parent), source.name, options)
        if asset is None:
            raise RuntimeError("load_asset failed")
        names = list(gym.get_asset_rigid_body_names(asset))
        dofs = list(gym.get_asset_dof_names(asset))
        shapes = gym.get_asset_rigid_shape_count(asset)
        if len(names) != 7 or set(names) != set(masses):
            raise RuntimeError("Rigid body mismatch: " + repr(names))
        if len(dofs) != 6 or set(dofs) != expected_dofs:
            raise RuntimeError("DOF mismatch: " + repr(dofs))
        if shapes != 7:
            raise RuntimeError("Expected 7 collision shapes; got " + str(shapes))
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
        if env is None:
            raise RuntimeError("create_env failed")
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1)
        actor = gym.create_actor(env, asset, pose, "heroleg26", 0, -1)
        if actor < 0:
            raise RuntimeError("create_actor failed")
        gym.prepare_sim(sim)
        properties = gym.get_actor_rigid_body_properties(env, actor)
        if len(properties) != len(names):
            raise RuntimeError("Actor body count mismatch")
        actual = {name: float(prop.mass) for name, prop in zip(names, properties)}
        for name, mass in actual.items():
            if not math.isclose(mass, masses[name], rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError("Imported mass mismatch for " + name)
        print("Robot loaded: 7 rigid bodies, 6 DOFs, 7 collision shapes.", flush=True)
        frames = show_static_viewer(gym, sim, env, gymapi, fps) if viewer else 0
        for path, old_hash in manifest["original_sha256"].items():
            if sha256(Path(path)) != old_hash:
                raise RuntimeError("Original asset changed: " + path)
        report = {"status": "SUCCESS", "rigid_bodies": names, "dofs": dofs,
                  "collision_shapes": shapes, "mass_by_link_kg": actual,
                  "total_mass_kg": sum(actual.values()), "physics_steps": 0,
                  "original_files_unchanged": True,
                  "viewer_requested": viewer, "rendered_frames": frames,
                  "compute_device": compute_device,
                  "graphics_device": graphics_device if viewer else -1,
                  "override_com": False, "override_inertia": False,
                  "vhacd_enabled": False}
        target = user_path(output / "load_report.json")
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("SUCCESS: your robot actor was created.")
        print("Rigid bodies:", len(names), names)
        print("DOFs:", len(dofs), dofs)
        print("Collision shapes:", shapes)
        print("Imported mass: {:.8f} kg".format(sum(actual.values())))
        print("Physics steps: 0; original files unchanged.")
        print("Report:", target)
    finally:
        gym.destroy_sim(sim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--viewer", action="store_true",
                      help="Show a static window in an existing desktop session")
    parser.add_argument("--compute-device", type=int, default=0,
                        help="CUDA ordinal within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--graphics-device", type=int, default=0,
                        help="Vulkan graphics ordinal; independent of CUDA numbering")
    parser.add_argument("--fps", type=int, default=10,
                        help="Limit Viewer refresh to reduce remote desktop traffic")
    parser.add_argument("--repo", type=Path,
                        default=Path.home() / "projects/RM-wheelleg-rl")
    args = parser.parse_args()
    if args.compute_device < 0 or args.graphics_device < 0 or not 1 <= args.fps <= 60:
        parser.error("Device indices must be nonnegative; fps must be between 1 and 60")
    if args.viewer and not os.environ.get("DISPLAY"):
        raise RuntimeError("Run --viewer from your remote desktop terminal; DISPLAY is unset")
    if os.geteuid() == 0:
        raise RuntimeError("Use your ordinary account, not root")
    if Path.home().resolve() != Path(pwd.getpwuid(os.getuid()).pw_dir).resolve():
        raise RuntimeError("HOME must be your own account home")
    source = args.repo.expanduser().resolve() / "heroleg26_sim/urdf/heroleg26.urdf"
    if not source.is_file():
        raise FileNotFoundError("Missing source URDF: " + str(source))
    output = user_path(Path(__file__).resolve().parent / "generated")
    output.mkdir(parents=True, exist_ok=True)
    for variable, suffix in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"),
                             ("CUDA_CACHE_PATH", "cuda"),
                             ("TORCH_EXTENSIONS_DIR", "torch_extensions")):
        path = user_path(Path.home() / ".cache/heroleg26" / suffix)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    sys.dont_write_bytecode = True
    os.chdir(str(output))
    manifest = prepare(source, output)
    print("Preparation OK: 7 full OBJ meshes; physical XML unchanged.", flush=True)
    if not args.prepare_only:
        check_actor(manifest, output, viewer=args.viewer,
                    compute_device=args.compute_device,
                    graphics_device=args.graphics_device, fps=args.fps)


if __name__ == "__main__":
    main()
