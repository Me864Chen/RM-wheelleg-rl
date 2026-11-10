#!/usr/bin/env python3
"""Headless free-base ground-contact diagnostic; no balance policy or training."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import pwd
import sys
import xml.etree.ElementTree as ET

from check_isaacgym import prepare, sha256, user_path
from check_pd import load_config, diagnostic_config, load_helper, verify_no_self_collision

ENTRY_PATH = Path(__file__).resolve()


def tilt_degrees(quaternion):
    x, y, z, w = [float(v) for v in quaternion]
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(norm) or not .95 <= norm <= 1.05:
        raise ValueError('Invalid root quaternion norm: '+str(norm))
    return math.degrees(math.acos(max(-1., min(1., 1.-2.*(x*x+y*y)/(norm*norm)))))


def vectors(array):
    """Read classic Vec3 arrays without assuming an unstructured memory layout."""
    import numpy as np
    fields = getattr(getattr(array, 'dtype', None), 'names', None)
    if fields and {'x', 'y', 'z'} <= set(fields):
        result = np.column_stack([array[key] for key in ('x', 'y', 'z')])
    else:
        try:
            result = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError):
            result = np.array([[row.x, row.y, row.z] for row in array], dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError('Expected finite N-by-3 vectors from SDK')
    return result


def contact_summary(contacts, body_names):
    """Use named SDK fields only. Never guess tuple offsets for body indices."""
    result = {'ground_links': [], 'self_pairs': [], 'parsed': True,
              'unknown_body_indices': [], 'maximum_initial_overlap_m': None}
    if len(contacts) == 0:
        return result
    fields = getattr(getattr(contacts, 'dtype', None), 'names', None)
    if not fields or not {'body0', 'body1'} <= set(fields):
        result['parsed'] = False
        return result
    links = set()
    for row in contacts:
        a, b = int(row['body0']), int(row['body1'])
        if 'initial_overlap' in fields:
            overlap = float(row['initial_overlap'])
            if math.isfinite(overlap):
                previous = result['maximum_initial_overlap_m']
                result['maximum_initial_overlap_m'] = overlap if previous is None else max(previous, overlap)
        if a == -1 or b == -1:
            index = b if a == -1 else a
            if 0 <= index < len(body_names):
                links.add(body_names[index])
            else:
                result['unknown_body_indices'].append(index)
        elif a >= 0 and b >= 0:
            result['self_pairs'].append([a, b])
    result['ground_links'] = sorted(links)
    return result


def classify_result(metrics):
    """A fall is an observation, not failure of this unbalanced contact test."""
    reasons = []
    if not metrics['completed']:
        reasons.append('Simulation did not complete the requested duration')
    if not metrics['ground_contact_seen']:
        reasons.append('No ground contact was observed')
    if not metrics['loaded_ground_contact_seen']:
        reasons.append('No upward ground contact force above 1 N was observed')
    for name in ('L_Link3', 'R_Link3'):
        if name not in metrics['wheel_ground_contact_seen']:
            reasons.append('No wheel-ground contact observed for '+name)
        if name not in metrics['wheel_loaded_contact_seen']:
            reasons.append('No upward wheel contact force above 1 N observed for '+name)
    if metrics['minimum_base_height_m'] is None:
        reasons.append('No finite base height was recorded')
    elif metrics['minimum_base_height_m'] < -.05:
        reasons.append('Base origin moved more than 5 cm below the ground plane')
    if metrics['maximum_base_speed_m_s'] > 20.:
        reasons.append('Base speed exceeded the 20 m/s diagnostic guard')
    if metrics['maximum_joint_speed_ratio'] > 4.:
        reasons.append('A joint exceeded 4 times its operating speed limit')
    if metrics['maximum_knee_bound_violation_rad'] > .035:
        reasons.append('A knee exceeded the source bound by more than 0.035 rad')
    if metrics['self_contact_count']:
        reasons.append('Self contacts were reported despite the intended mask')
    if metrics['maximum_initial_overlap_m'] is not None and metrics['maximum_initial_overlap_m'] > .05:
        reasons.append('SDK reported contact initial overlap exceeding 5 cm')
    return ('PASS' if not reasons else 'NEEDS_REVIEW'), reasons


def validate_setup(cfg, seconds, lift):
    if not math.isfinite(seconds) or not .2 <= seconds <= 30.:
        raise ValueError('--seconds must be between 0.2 and 30')
    if not math.isfinite(lift) or not 0. <= lift <= .2:
        raise ValueError('--lift must be between 0 and 0.2 metres')
    if cfg['sim']['dt'] > .001000001:
        raise ValueError('Synchronize the validated 1 ms heroleg26 config, or use --dt 0.001')
    if len(cfg['init_state']['pos']) != 3 or not all(math.isfinite(v) for v in cfg['init_state']['pos']):
        raise ValueError('Expected a finite three-component initial position')
    for key, value in cfg['terrain'].items():
        if not math.isfinite(value) or value < 0 or (key == 'restitution' and value > 1):
            raise ValueError('Invalid ground material parameter: '+key)


def run_check(repo, cfg, output, device, seconds, lift, control):
    # Keep SDK before Torch; classic CPU state API avoids building gymtorch.
    from isaacgym import gymapi
    import numpy as np
    import torch

    helper, helper_path = load_helper(repo, cfg['branch'])
    cache = user_path(repo/'heroleg26_isaac_check/generated')
    manifest = prepare(repo/'heroleg26_sim/urdf/heroleg26.urdf', cache)
    source = Path(manifest['urdf'])
    xml = ET.parse(source).getroot()
    joints = {j.get('name'): j for j in xml.findall('joint')}
    masses = {link.get('name'): float(link.find('inertial/mass').get('value')) for link in xml.findall('link')}
    protected = dict(manifest['original_sha256'])
    for path in cfg['source_files']+[str(helper_path), str(ENTRY_PATH),
                                       str(ENTRY_PATH.with_name('check_pd.py'))]:
        protected[path] = sha256(Path(path))
    report = {'status': 'RUNNING', 'physics_steps': 0, 'branch': cfg['branch'],
              'configuration': cfg, 'seconds_requested': seconds, 'additional_lift_m': lift,
              'compute_device': device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'fixed_base': False, 'ground_present': True, 'ground_height_m': 0.,
              'ground_material': cfg['terrain'], 'control_mode': control,
              'joint_velocity_source': 'SDK STATE_ALL raw velocities, as in the validated fixed-base check',
              'collision_filter_override': 1, 'randomization_enabled': False,
              'source_sha256': protected, 'fall_observed': False,
              'first_non_wheel_loaded_contact_time_s': None,
              'first_fall_time_s': None, 'first_contact_time_s': None,
              'first_wheel_contact_time_s': {},
              'limitations': 'Basic free-base contact and numerical checks only. Not a balance policy, measured friction validation, contact-mesh fidelity proof, full motor model or training.'}
    metrics = {'completed': False, 'ground_contact_seen': False, 'wheel_ground_contact_seen': set(),
               'loaded_ground_contact_seen': False, 'wheel_loaded_contact_seen': set(),
               'minimum_base_height_m': None, 'maximum_base_speed_m_s': 0.,
               'maximum_joint_speed_ratio': 0., 'maximum_knee_bound_violation_rad': 0.,
               'self_contact_count': 0, 'maximum_initial_overlap_m': None}
    non_wheel_loaded = set()
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.dt, params.substeps = cfg['sim']['dt'], cfg['sim']['substeps']
    params.gravity = gymapi.Vec3(*cfg['sim']['gravity'])
    params.use_gpu_pipeline = False
    params.physx.use_gpu = True
    for key, value in cfg['sim.physx'].items():
        setattr(params.physx, key, value)
    sim = None
    try:
        sim = gym.create_sim(device, -1, gymapi.SIM_PHYSX, params)
        if sim is None:
            raise RuntimeError('Could not create GPU PhysX simulation')
        ground = gymapi.PlaneParams()
        ground.normal = gymapi.Vec3(0., 0., 1.)
        ground.distance = 0.
        for key, value in cfg['terrain'].items():
            setattr(ground, key, value)
        gym.add_ground(sim, ground)
        options = gymapi.AssetOptions()
        options.fix_base_link = False
        options.collapse_fixed_joints = False
        options.override_com = False
        options.override_inertia = False
        options.vhacd_enabled = False
        options.disable_gravity = False
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        for key in ('angular_damping', 'linear_damping', 'max_angular_velocity',
                    'max_linear_velocity', 'armature', 'thickness'):
            setattr(options, key, cfg['asset'][key])
        asset = gym.load_asset(sim, str(source.parent), source.name, options)
        if asset is None:
            raise RuntimeError('Could not load the robot asset')
        names = list(gym.get_asset_dof_names(asset))
        body_names = list(gym.get_asset_rigid_body_names(asset))
        if len(names) != 6 or set(names) != set(cfg['asset']['policy_joint_names']):
            raise RuntimeError('Unexpected joint names: '+repr(names))
        if len(body_names) != 7 or set(body_names) != set(masses):
            raise RuntimeError('Unexpected body names: '+repr(body_names))
        report['native_joint_order'], report['body_order'] = names, body_names
        env = gym.create_env(sim, gymapi.Vec3(-3, -3, 0), gymapi.Vec3(3, 3, 3), 1)
        if env is None:
            raise RuntimeError('Could not create environment')
        pose = gymapi.Transform()
        initial_pos = list(cfg['init_state']['pos'])
        initial_pos[2] += lift
        pose.p = gymapi.Vec3(*initial_pos)
        pose.r = gymapi.Quat(*cfg['init_state']['rot'])
        actor = gym.create_actor(env, asset, pose, 'heroleg26_contact_check', 0, 1)
        if actor < 0:
            raise RuntimeError('Could not create actor')
        report['initial_base_position_m'] = initial_pos
        report['rigid_shape_filters'] = verify_no_self_collision(gym, env, actor)
        report['self_collisions_enabled'] = False
        print('Verified shape collision filters:', report['rigid_shape_filters'], flush=True)
        shape_props = gym.get_actor_rigid_shape_properties(env, actor)
        material_keys = ('friction', 'rolling_friction', 'torsion_friction', 'restitution',
                         'contact_offset', 'rest_offset')
        report['imported_shape_materials'] = [
            {key: float(getattr(shape, key)) for key in material_keys if hasattr(shape, key)}
            for shape in shape_props]
        actual_masses = gym.get_actor_rigid_body_properties(env, actor)
        if len(actual_masses) != 7:
            raise RuntimeError('Expected seven body mass records')
        report['mass_by_link_kg'] = {n: float(p.mass) for n, p in zip(body_names, actual_masses)}
        for name, mass in report['mass_by_link_kg'].items():
            if not math.isclose(mass, masses[name], rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError('Imported mass mismatch for '+name)
        report['total_mass_kg'] = sum(report['mass_by_link_kg'].values())
        props = gym.get_asset_dof_properties(asset)
        helper.apply_actuator_limits(props, names, cfg['asset']['actuator_limits'])
        props['driveMode'].fill(int(gymapi.DOF_MODE_EFFORT))
        props['stiffness'].fill(0.)
        props['damping'].fill(0.)
        for i, name in enumerate(names):
            props['armature'][i] = next((value for key, value in cfg['asset']['dof_armature'].items()
                                          if key in name), 0.)
        if not gym.set_actor_dof_properties(env, actor, props):
            raise RuntimeError('Could not set external PD properties')
        actual_props = gym.get_actor_dof_properties(env, actor)
        for key in ('effort', 'velocity'):
            np.testing.assert_allclose(actual_props[key], props[key], rtol=1e-5, atol=1e-6)
        default = np.array([cfg['init_state']['default_joint_angles'][n] for n in names], dtype=np.float32)
        limited = [i for i, n in enumerate(names) if joints[n].get('type') == 'revolute']
        lower, upper = props['lower'].copy(), props['upper'].copy()
        if np.any(default[limited] < lower[limited]) or np.any(default[limited] > upper[limited]):
            raise ValueError('Initial pose is outside joint bounds')
        initial = np.zeros(6, dtype=gymapi.DofState.dtype)
        initial['pos'] = default
        if not gym.set_actor_dof_states(env, actor, initial, gymapi.STATE_ALL):
            raise RuntimeError('Could not set initial joint pose')
        # Finalize only after the desired stance is set, before any physics step.
        gym.prepare_sim(sim)
        kp = torch.tensor([[cfg['control']['stiffness'][n] for n in names]], dtype=torch.float32)
        kd = torch.tensor([[cfg['control']['damping'][n] for n in names]], dtype=torch.float32)
        effort, speed = torch.tensor(props['effort'].copy()), torch.tensor(props['velocity'].copy())
        bounds = torch.tensor(np.stack((lower, upper), axis=-1))
        indices = torch.tensor(limited, dtype=torch.long)
        base = body_names.index('base_link')
        body_indices = {name: gym.find_actor_rigid_body_index(env, actor, name, gymapi.DOMAIN_ENV)
                        for name in body_names}
        if set(body_indices.values()) != set(range(7)):
            raise RuntimeError('Expected a single actor in this environment')
        env_body_names = [next(n for n, i in body_indices.items() if i == index) for index in range(7)]
        max_fz = np.zeros(7)
        saturated = np.zeros(6, dtype=np.int64)
        maximum_joint_speed = np.zeros(6)
        count = max(1, round(seconds/params.dt))
        with (output/'trace.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            header = ['time_s', 'base_x_m', 'base_y_m', 'base_z_m',
                      'base_qx', 'base_qy', 'base_qz', 'base_qw',
                      'base_vx_m_s', 'base_vy_m_s', 'base_vz_m_s',
                      'base_wx_rad_s', 'base_wy_rad_s', 'base_wz_rad_s',
                      'tilt_deg', 'contact_count', 'ground_links']
            for prefix in ('q_rad_', 'v_rad_s_', 'commanded_tau_pre_step_Nm_'):
                header.extend(prefix+n for n in names)
            header.extend('contact_fz_N_'+n for n in env_body_names)
            writer.writerow(header)
            for step in range(count):
                state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
                q, v = state['pos'].copy(), state['vel'].copy()
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    raise RuntimeError('Non-finite state before PD')
                torque = np.zeros(6, dtype=np.float32)
                if control == 'hold':
                    torque = helper.compute_pd_torques(
                        torch.from_numpy(default[None]), torch.zeros((1, 6)),
                        torch.from_numpy(q[None]), torch.from_numpy(v[None]), kp, kd, effort,
                        velocity_limits=speed if cfg['control']['enforce_operating_speed'] else None,
                        position_indices=indices if cfg['control']['clip_position_targets'] else None,
                        position_bounds=bounds,
                    )[0].numpy().astype(np.float32, copy=False)
                if not np.isfinite(torque).all() or np.any(np.abs(torque) > props['effort']+1e-5):
                    raise RuntimeError('Invalid torque command')
                if not gym.apply_actor_dof_efforts(env, actor, np.ascontiguousarray(torque)):
                    raise RuntimeError('Could not apply efforts')
                gym.simulate(sim)
                fetch_status = gym.fetch_results(sim, True)
                report['physics_steps'] += 1
                if step == 0:
                    report['first_fetch_results_return'] = repr(fetch_status)
                elapsed = report['physics_steps']*params.dt
                after = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
                bodies = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
                body = bodies[base]
                p = np.array([float(body['pose']['p'][key]) for key in ('x', 'y', 'z')])
                quaternion = [float(body['pose']['r'][key]) for key in ('x', 'y', 'z', 'w')]
                linear = np.array([float(body['vel']['linear'][key]) for key in ('x', 'y', 'z')])
                angular = [float(body['vel']['angular'][key]) for key in ('x', 'y', 'z')]
                if not all(np.isfinite(a).all() for a in (p, quaternion, linear, angular, after['pos'], after['vel'])):
                    raise RuntimeError('Non-finite post-step body or joint state')
                tilt = tilt_degrees(quaternion)
                contacts = gym.get_env_rigid_contacts(env)
                forces = vectors(gym.get_env_rigid_contact_forces(env))
                if forces.shape != (7, 3):
                    raise RuntimeError('Unexpected contact-force array shape: '+repr(forces.shape))
                summary = contact_summary(contacts, env_body_names)
                if len(contacts) and 'contact_format' not in report:
                    report['contact_format'] = {'dtype': str(getattr(contacts, 'dtype', None)),
                                                'named_fields_available': summary['parsed'],
                                                'first_record_preview': repr(contacts[0])}
                touched = set(summary['ground_links'])
                # With one actor, verified self filters and a ground plane only,
                # nonzero upward body contact force also identifies ground load.
                loaded = {n for n, i in body_indices.items() if forces[i, 2] > 1.}
                unexpected_load = loaded - {'L_Link3', 'R_Link3'}
                non_wheel_loaded.update(unexpected_load)
                if unexpected_load and report['first_non_wheel_loaded_contact_time_s'] is None:
                    report['first_non_wheel_loaded_contact_time_s'] = elapsed
                    print('Non-wheel ground load >1 N:', sorted(unexpected_load),
                          'at {:.4f}s; do not use this as a standing-height calibration.'.format(elapsed),
                          flush=True)
                touched.update(loaded)
                if loaded:
                    metrics['loaded_ground_contact_seen'] = True
                metrics['self_contact_count'] += len(summary['self_pairs'])
                overlap = summary['maximum_initial_overlap_m']
                if overlap is not None:
                    prev = metrics['maximum_initial_overlap_m']
                    metrics['maximum_initial_overlap_m'] = overlap if prev is None else max(prev, overlap)
                if touched:
                    metrics['ground_contact_seen'] = True
                    if report['first_contact_time_s'] is None:
                        report['first_contact_time_s'] = elapsed
                        print('First ground contact at {:.4f}s: {}'.format(elapsed, sorted(touched)), flush=True)
                for name in ('L_Link3', 'R_Link3'):
                    if name in touched:
                        metrics['wheel_ground_contact_seen'].add(name)
                        report['first_wheel_contact_time_s'].setdefault(name, elapsed)
                    if name in loaded:
                        metrics['wheel_loaded_contact_seen'].add(name)
                if tilt >= 60. and not report['fall_observed']:
                    report['fall_observed'] = True
                    report['first_fall_time_s'] = elapsed
                    print('Tilt reached 60 degrees at {:.3f}s; recording a fall, not a balance failure.'.format(elapsed), flush=True)
                previous_height = metrics['minimum_base_height_m']
                metrics['minimum_base_height_m'] = float(p[2]) if previous_height is None else min(previous_height, float(p[2]))
                metrics['maximum_base_speed_m_s'] = max(metrics['maximum_base_speed_m_s'], float(np.linalg.norm(linear)))
                metrics['maximum_joint_speed_ratio'] = max(metrics['maximum_joint_speed_ratio'],
                                                           float(np.max(np.abs(after['vel'])/props['velocity'])))
                violation = np.maximum(lower[limited]-after['pos'][limited], after['pos'][limited]-upper[limited])
                metrics['maximum_knee_bound_violation_rad'] = max(metrics['maximum_knee_bound_violation_rad'],
                                                                  float(np.maximum(violation, 0).max()))
                max_fz = np.maximum(max_fz, np.abs(forces[:, 2]))
                maximum_joint_speed = np.maximum(maximum_joint_speed, np.abs(after['vel']))
                saturated += np.abs(torque) >= .99*props['effort']
                writer.writerow([format(elapsed, '.6f')]+p.tolist()+quaternion+linear.tolist()+angular+
                                [tilt, len(contacts), ';'.join(sorted(touched))]+
                                after['pos'].tolist()+after['vel'].tolist()+torque.tolist()+forces[:, 2].tolist())
                if p[2] < -.5 or np.linalg.norm(linear) > 100:
                    raise RuntimeError('Severe fall-through or numerical divergence; see trace')
            metrics['completed'] = True
        report['maximum_abs_contact_fz_N'] = dict(zip(env_body_names, max_fz.tolist()))
        report['maximum_abs_joint_speed_rad_s'] = dict(zip(names, maximum_joint_speed.tolist()))
        report['torque_saturation_fraction'] = dict(zip(names, (saturated/report['physics_steps']).tolist()))
        report['status'], report['review_reasons'] = classify_result(metrics)
        report['final_base_position_m'] = p.tolist()
        report['final_tilt_deg'] = tilt
    except BaseException as error:
        report['status'] = 'ERROR'
        report['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        try:
            if sim is not None:
                gym.destroy_sim(sim)
        finally:
            metrics['wheel_ground_contact_seen'] = sorted(metrics['wheel_ground_contact_seen'])
            metrics['wheel_loaded_contact_seen'] = sorted(metrics['wheel_loaded_contact_seen'])
            report['metrics'] = metrics
            report['non_wheel_loaded_contact_bodies'] = sorted(non_wheel_loaded)
            report['wheel_only_support_status'] = (
                'NEEDS_REVIEW' if non_wheel_loaded else 'NO_NON_WHEEL_LOAD_OBSERVED')
            changed = [path for path, digest in protected.items()
                       if not Path(path).is_file() or sha256(Path(path)) != digest]
            report['source_files_unchanged'] = not changed
            if changed:
                report['status'], report['changed_source_files'] = 'ERROR', changed
            (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('Ground contact check status:', report['status'], flush=True)
            print('Physics steps:', report['physics_steps'], flush=True)
            print('Wheel-ground contacts observed:', metrics['wheel_ground_contact_seen'], flush=True)
            print('Fall observed:', report['fall_observed'], '(balance is not tested)', flush=True)
            print('Wheel-only support:', report['wheel_only_support_status'],
                  sorted(non_wheel_loaded), '(standing height is not calibrated)', flush=True)
            for reason in report.get('review_reasons', []):
                print('Review:', reason, flush=True)
            print('Report:', output/'report.json', flush=True)
            print('Trace:', output/'trace.csv', flush=True)
    return 0 if report['status'] == 'PASS' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--branch', choices=('plane', 'jump'), default='plane')
    parser.add_argument('--compute-device', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--lift', type=float, default=0., help='Additional diagnostic release height, metres')
    parser.add_argument('--control', choices=('hold', 'passive'), default='hold')
    parser.add_argument('--dt', type=float, help='Diagnostic override; 1 ms or smaller required')
    parser.add_argument('--check-config', action='store_true', help='Read settings only; no SDK/GPU or writes')
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    cfg = diagnostic_config(load_config(repo, args.branch), args.dt)
    validate_setup(cfg, args.seconds, args.lift)
    if args.check_config:
        print(json.dumps({'configuration': cfg, 'seconds': args.seconds,
                          'additional_lift_m': args.lift, 'control': args.control}, indent=2))
        return 0
    if args.compute_device < 0:
        parser.error('--compute-device must be nonnegative')
    if os.geteuid() == 0 or Path.home().resolve() != Path(pwd.getpwuid(os.getuid()).pw_dir).resolve():
        raise RuntimeError('Use an ordinary account and its own HOME')
    for key, suffix in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                        ('CUDA_CACHE_PATH', 'cuda'), ('TORCH_EXTENSIONS_DIR', 'torch_extensions')):
        directory = user_path(Path.home()/'.cache/heroleg26'/suffix)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    output = user_path(Path.home()/'.local/state/heroleg26/isaacgym'/(
        'contact-'+args.branch+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    output.mkdir(parents=True)
    user_path(repo/'heroleg26_isaac_check/generated').mkdir(parents=True, exist_ok=True)
    os.chdir(str(output))
    sys.dont_write_bytecode = True
    print('Headless free-base ground-contact check; no balance policy, no training.', flush=True)
    print('Output:', output, flush=True)
    return run_check(repo, cfg, output, args.compute_device, args.seconds, args.lift, args.control)


if __name__ == '__main__':
    sys.exit(main())
