#!/usr/bin/env python3
"""Kinematic MuJoCo inspection: no mj_step, policy, or source-model edits."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from queue import SimpleQueue, Empty
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ('LJ1', 'LJ2', 'LJ3', 'RJ1', 'RJ2', 'RJ3')
ROLES = ('left hip', 'left knee', 'left wheel', 'right hip', 'right knee', 'right wheel')
HELP = '''
KINEMATIC INSPECTION: no physics integration, no ONNX, base pose held fixed.
Click the MuJoCo window before using keys:
  1..6       Select LJ1, LJ2, LJ3, RJ1, RJ2, RJ3
  Right / =  Increase selected angle by 5 degrees
  Left / -   Decrease selected angle by 5 degrees
  Up / Down  Increase / decrease selected angle by 1 degree
  Z          Reset selected joint to source zero
  0          Reset all six joints to source zero
  J          Toggle joint axes and selected positive-axis arrow
  C          Switch visual / original collision geometry
  P          Print current joint angles and axes to the terminal
  Esc        Quit
Mouse: use native MuJoCo camera controls; right sidebar Joint sliders also work.
The red selected arrow denotes POSITIVE rotation axis (right-hand rule).
Joint roles are provisional labels; confirm against CAD / real hardware.
No motor torque, velocity, or mechanical-angle limits are inferred here.
'''


def compile_model(path: Path):
    """Resolve resources and add a temporary floating root for inspection only."""
    root = ET.parse(path).getroot()
    links = {x.get('name') for x in root.findall('link')}
    children = {x.find('child').get('link') for x in root.findall('joint')}
    roots = links - children
    if len(roots) != 1:
        raise ValueError(f'Expected one root link, found {sorted(roots)}')
    root_link = roots.pop()
    prefix = '__inspection_world__'
    if prefix in links:
        raise ValueError('Reserved inspection link name already exists')
    resources = [path]
    # Unique names avoid importer warnings for unnamed CAD visual/collision geoms.
    for link in root.findall('link'):
        for kind in ('visual', 'collision'):
            for i, geom in enumerate(link.findall(kind)):
                geom.set('name', f"{link.get('name')}_{kind}_{i}")
    for mesh in root.findall('.//mesh'):
        name = mesh.get('filename', '')
        if '://' in name:
            raise ValueError(f'Unresolved URI {name}; use heroleg26_visual.urdf')
        file = (path.parent / name).resolve()
        if not file.is_file():
            raise FileNotFoundError(file)
        mesh.set('filename', str(file))
        resources.append(file)
    # Kept entirely in memory. All robot link/joint/inertial definitions retained.
    ET.SubElement(root, 'link', name=prefix)
    joint = ET.SubElement(root, 'joint', name='__inspection_free__', type='floating')
    ET.SubElement(joint, 'parent', link=prefix)
    ET.SubElement(joint, 'child', link=root_link)
    extension = root.find('mujoco')
    if extension is None:
        extension = ET.SubElement(root, 'mujoco')
    compiler = extension.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(extension, 'compiler')
    compiler.attrib.update(strippath='false', discardvisual='false', fusestatic='true')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    # Viewing illumination only. No geom colours or physical parameters altered.
    model.vis.headlight.ambient[:] = .35
    model.vis.headlight.diffuse[:] = .6
    model.vis.scale.jointlength = .12
    model.vis.scale.jointwidth = .015
    return model, sorted(set(resources))


def fingerprints(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Inspector:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        if min(self.ids) < 0:
            raise ValueError('Expected six joints: ' + ', '.join(JOINTS))
        self.adrs = [int(model.jnt_qposadr[j]) for j in self.ids]
        self.zero = model.qpos0.copy()
        self.selected = 0
        self.axes = True
        self.collision = False
        self.running = True
        self.forward()

    def forward(self):
        # Hold floating base, cancel velocities; this tool NEVER integrates dynamics.
        free = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, '__inspection_free__')
        adr = int(self.model.jnt_qposadr[free])
        self.data.qpos[adr:adr+7] = self.zero[adr:adr+7]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def adjust(self, degrees):
        j = self.ids[self.selected]
        q = self.adrs[self.selected]
        value = self.data.qpos[q] + math.radians(degrees)
        if self.model.jnt_limited[j]:
            value = np.clip(value, *self.model.jnt_range[j])
        self.data.qpos[q] = value
        self.forward()

    def key(self, code):
        if 49 <= code <= 54:
            self.selected = code - 49
        elif code in (262, ord('='), ord('+')):
            self.adjust(5)
        elif code in (263, ord('-')):
            self.adjust(-5)
        elif code == 265:
            self.adjust(1)
        elif code == 264:
            self.adjust(-1)
        elif code in (ord('z'), ord('Z')):
            q = self.adrs[self.selected]
            self.data.qpos[q] = self.zero[q]
        elif code == ord('0'):
            self.data.qpos[self.adrs] = self.zero[self.adrs]
        elif code in (ord('j'), ord('J')):
            self.axes = not self.axes
        elif code in (ord('c'), ord('C')):
            self.collision = not self.collision
        elif code == 256:
            self.running = False
        elif code not in (ord('p'), ord('P')):
            return
        self.forward()
        self.print_status()

    def layout(self):
        return [dict(key=i+1, name=name, role=ROLES[i],
                     angle_deg=float(np.degrees(self.data.qpos[self.adrs[i]])),
                     axis_local=self.model.jnt_axis[j].tolist(),
                     axis_world=self.data.xaxis[j].tolist(),
                     pivot_world_m=self.data.xanchor[j].tolist(),
                     limits_rad=self.model.jnt_range[j].tolist() if self.model.jnt_limited[j] else None)
                for i, (name, j) in enumerate(zip(JOINTS, self.ids))]

    def print_status(self):
        row = self.layout()[self.selected]
        print(f"[JOINT] {row['key']} {row['name']} ({row['role']}): "
              f"{row['angle_deg']:+.1f} deg | world axis={np.round(row['axis_world'], 4).tolist()} | "
              f"view={'COLLISION' if self.collision else 'VISUAL'} | axes={self.axes}", flush=True)

    def options(self, opt):
        # Native numeric shortcuts also toggle geom groups; enforce the chosen view.
        opt.geomgroup[:] = 0
        opt.geomgroup[0 if self.collision else 1] = 1
        opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self.axes
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    def annotate(self, scene, clear=False):
        if clear:
            scene.ngeom = 0
        if not self.axes:
            return
        j = self.ids[self.selected]
        start = self.data.xanchor[j].copy()
        end = start + .18 * self.data.xaxis[j]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                           np.zeros(3), start, np.eye(3).ravel(), np.array([1., .12, .08, 1.]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .008, start, end)
        scene.ngeom += 1
        label = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(label, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                           end + np.array([0, 0, .025]), np.eye(3).ravel(), np.ones(4))
        label.label = f'{JOINTS[self.selected]} {np.degrees(self.data.qpos[self.adrs[self.selected]]):+.1f} deg'
        scene.ngeom += 1


def checks(inspector):
    """Check keyboard routing and kinematic transforms without any desktop."""
    q0 = inspector.data.qpos.copy()
    for i in range(6):
        inspector.key(ord(str(i+1)))
        before = inspector.data.qpos.copy()
        body = inspector.model.jnt_bodyid[inspector.ids[i]]
        rotation = inspector.data.xmat[body].reshape(3, 3).copy()
        inspector.adjust(5)
        expected = before.copy()
        expected[inspector.adrs[i]] += math.radians(5)
        np.testing.assert_allclose(inspector.data.qpos, expected, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(inspector.data.xaxis[inspector.ids[i]]), 1)
        axis = inspector.model.jnt_axis[inspector.ids[i]]
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        theta = math.radians(5)
        expected_rotation = np.eye(3) + math.sin(theta)*skew + (1-math.cos(theta))*(skew@skew)
        np.testing.assert_allclose(rotation.T @ inspector.data.xmat[body].reshape(3, 3),
                                   expected_rotation, atol=1e-12)
        inspector.key(ord('z'))
    inspector.key(ord('0'))
    np.testing.assert_array_equal(inspector.data.qpos, q0)
    assert inspector.data.time == 0
    inspector.selected = 0
    return dict(status='passed', joints_checked=6, physics_steps=0,
                total_mass_kg=float(inspector.model.body_mass.sum()), joint_layout=inspector.layout())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf', type=Path, default=ROOT/'urdf/heroleg26_visual.urdf')
    p.add_argument('--check', action='store_true', help='Run kinematic checks without a desktop window')
    p.add_argument('--snapshot', type=Path, help='Render a static preview without opening a desktop window')
    args = p.parse_args()
    print('[LOAD] ' + str(args.urdf.resolve()), flush=True)
    print('[LOAD] Full collision mesh preserved; initial compilation may take several seconds.', flush=True)
    model, resources = compile_model(args.urdf.resolve())
    before = fingerprints(resources)
    inspector = Inspector(model)
    print(HELP, flush=True)
    if args.check:
        print(json.dumps(checks(inspector), indent=2), flush=True)
    if args.snapshot:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .2]
        camera.distance = 1.65
        camera.azimuth = 145
        camera.elevation = -18
        opt = mujoco.MjvOption()
        inspector.options(opt)
        with mujoco.Renderer(model, height=720, width=960) as renderer:
            renderer.update_scene(inspector.data, camera, scene_option=opt)
            inspector.annotate(renderer.scene)
            # Simple PNG encoding keeps the entry point free of extra dependencies.
            import struct, zlib
            rgb = renderer.render()
            def chunk(t, b):
                return struct.pack('!I', len(b)) + t + b + struct.pack('!I', zlib.crc32(t+b)&0xffffffff)
            payload = b''.join(b'\0'+row.tobytes() for row in rgb)
            png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 960, 720, 8, 2, 0, 0, 0))
            png += chunk(b'IDAT', zlib.compress(payload)) + chunk(b'IEND', b'')
            args.snapshot.write_bytes(png)
        print('[SNAPSHOT] ' + str(args.snapshot.resolve()), flush=True)
    if not args.check and not args.snapshot:
        from mujoco import viewer as native_viewer
        events = SimpleQueue()
        try:
            with native_viewer.launch_passive(model, inspector.data, key_callback=events.put) as viewer:
                viewer.cam.lookat[:] = [0, 0, .2]
                viewer.cam.distance = 1.65
                viewer.cam.azimuth = 145
                viewer.cam.elevation = -18
                inspector.print_status()
                while viewer.is_running() and inspector.running:
                    with viewer.lock():
                        while True:
                            try:
                                inspector.key(events.get_nowait())
                            except Empty:
                                break
                        inspector.forward()
                        inspector.options(viewer.opt)
                        inspector.annotate(viewer.user_scn, clear=True)
                    viewer.sync()
                    time.sleep(1/60)
        finally:
            if fingerprints(resources) != before:
                raise RuntimeError('Source resource changed during inspection')
    assert fingerprints(resources) == before, 'Source resource changed'
    print('[OK] Source URDF and mesh files unchanged.', flush=True)


if __name__ == '__main__':
    main()
