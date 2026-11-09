#!/usr/bin/env python3
"""Headless, fixed-base heroleg26 PD response check. No policy or training."""
import argparse
import ast
import copy
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
import sys
import xml.etree.ElementTree as ET

from check_isaacgym import prepare, sha256, user_path


def literal_section(path, outer, section, keys):
    """Read only supported literal config fields without importing RL packages.

    Never execute the config. Reject expressions rather than guessing their value.
    Inheritance order is explicitly defined in load_config below.
    """
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == outer)
    parts = section.split('.')
    for part in parts:
        cls = next((n for n in cls.body if isinstance(n, ast.ClassDef) and n.name == part), None)
        if cls is None:
            return {}
    values = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0]
            if isinstance(name, ast.Name) and name.id in keys:
                try:
                    values[name.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError) as error:
                    raise ValueError('PD check requires literal field: ' + str(path) + ':' + name.id) from error
    return values


def load_config(repo, branch):
    pkg = repo / branch / 'wheel_legged_gym'
    sources = [(pkg/'envs/base/legged_robot_config.py', 'LeggedRobotCfg'),
               (pkg/'envs/wheel_legged/wheel_legged_config.py', 'WheelLeggedCfg'),
               (pkg/'envs/heroleg26/heroleg26_config.py', 'Heroleg26Cfg')]
    fields = {
        'init_state': ('default_joint_angles', 'rot'),
        'asset': ('policy_joint_names', 'actuator_limits', 'dof_armature'),
        'control': ('stiffness', 'damping', 'decimation', 'clip_position_targets', 'enforce_operating_speed'),
        'sim': ('dt', 'substeps', 'gravity'),
        'sim.physx': ('solver_type', 'num_position_iterations', 'num_velocity_iterations',
                      'contact_offset', 'rest_offset', 'bounce_threshold_velocity',
                      'max_depenetration_velocity'),
    }
    cfg = {key: {} for key in fields}
    for path, cls in sources:
        for section, keys in fields.items():
            cfg[section].update(literal_section(path, cls, section, keys))
    names = cfg['asset']['policy_joint_names']
    if len(names) != 6 or set(names) != {'RJ1', 'RJ2', 'RJ3', 'LJ1', 'LJ2', 'LJ3'}:
        raise ValueError('Expected exactly the six heroleg26 policy joints')
    for row in (cfg['init_state']['default_joint_angles'], cfg['asset']['actuator_limits'],
                cfg['control']['stiffness'], cfg['control']['damping']):
        if set(row) != set(names):
            raise ValueError('Pose, actuator and PD maps must cover all six named joints')
    if not 0 < cfg['sim']['dt'] <= .01:
        raise ValueError('Unsupported timestep for PD check')
    for name in names:
        values = [cfg['init_state']['default_joint_angles'][name],
                  cfg['control']['stiffness'][name], cfg['control']['damping'][name]]
        if not all(math.isfinite(v) for v in values) or min(values[1:]) < 0:
            raise ValueError('Invalid pose or PD values for ' + name)
        limits = cfg['asset']['actuator_limits'][name]
        if set(limits) != {'effort', 'velocity'} or not all(
                math.isfinite(v) and v > 0 for v in limits.values()):
            raise ValueError('Invalid motor limits for ' + name)
    cfg['source_files'] = [str(path) for path, _ in sources]
    cfg['branch'] = branch
    return cfg


def load_helper(repo, branch):
    path = repo / branch / 'wheel_legged_gym/utils/motor_limits.py'
    spec = importlib.util.spec_from_file_location('pd_motor_limits', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def diagnostic_config(cfg, dt=None):
    """Optional control-period experiment; never write training configuration."""
    result = copy.deepcopy(cfg)
    if dt is not None:
        if not math.isfinite(dt) or not .0001 <= dt <= .01:
            raise ValueError('Diagnostic --dt must be between 0.0001 and 0.01 seconds')
        result['diagnostic_overrides'] = {'training_dt_s': cfg['sim']['dt'], 'check_dt_s': dt}
        result['sim']['dt'] = dt
    return result


def verify_no_self_collision(gym, env, actor):
    """Read back actual shape masks instead of trusting a report label."""
    filters = [int(shape.filter) for shape in gym.get_actor_rigid_shape_properties(env, actor)]
    if not filters or not all(mask & 1 for mask in filters):
        raise RuntimeError('Expected common collision-filter bit 1 on every check shape; got '+repr(filters))
    return filters


def phase_plan(names):
    phases = [{'label': 'initial_hold', 'seconds': 2., 'joint': None, 'sign': 0}]
    for name in names:
        for sign in (1, -1):
            suffix = 'positive' if sign > 0 else 'negative'
            phases.append({'label': name+'_'+suffix+'_settle', 'seconds': 1., 'joint': None, 'sign': 0})
            phases.append({'label': name+'_'+suffix, 'seconds': 2., 'joint': name, 'sign': sign})
    phases.append({'label': 'final_stop', 'seconds': 2., 'joint': None, 'sign': 0})
    return phases


def run_check(repo, cfg, output, device):
    # Isaac Gym must be imported before Torch, even though all control is on CPU.
    from isaacgym import gymapi
    import numpy as np
    import torch

    helper, helper_path = load_helper(repo, cfg['branch'])
    manifest = prepare(repo/'heroleg26_sim/urdf/heroleg26.urdf',
                       user_path(repo/'heroleg26_isaac_check/generated'))
    source = Path(manifest['urdf'])
    xml = ET.parse(source).getroot()
    joints = {j.get('name'): j for j in xml.findall('joint')}
    protected = dict(manifest['original_sha256'])
    for p in cfg['source_files'] + [str(helper_path)]:
        protected[p] = sha256(Path(p))
    report = {'status': 'RUNNING', 'branch': cfg['branch'], 'physics_steps': 0,
              'compute_device': device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'fixed_base': True, 'base_height_m': 1., 'gravity_enabled': True,
              'ground_present': False, 'self_collisions_enabled': False,
              'collision_filter_override': 1, 'maximum_sampled_contact_count': 0,
              'controller': 'shared compute_pd_torques from selected branch motor_limits.py',
              'configuration': cfg, 'source_sha256': protected,
              'checks': [], 'tracking_tolerance_deg': 3.,
              'wheel_tracking_tolerance_rad_s': .2,
              'limitations': 'Numerical joint response only. Does not validate real hardware direction, ground contact, self-collision, balance, low-voltage motor performance or training.'}
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.dt = cfg['sim']['dt']
    params.substeps = cfg['sim']['substeps']
    params.gravity = gymapi.Vec3(*cfg['sim']['gravity'])
    params.use_gpu_pipeline = False
    params.physx.use_gpu = True
    for key, value in cfg['sim.physx'].items():
        setattr(params.physx, key, value)
    sim = None
    try:
        sim = gym.create_sim(device, -1, gymapi.SIM_PHYSX, params)
        if sim is None:
            raise RuntimeError('Failed to create GPU PhysX simulation')
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.collapse_fixed_joints = False
        options.override_com = False
        options.override_inertia = False
        options.vhacd_enabled = False
        options.disable_gravity = False
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        asset = gym.load_asset(sim, str(source.parent), source.name, options)
        if asset is None:
            raise RuntimeError('Could not load the robot asset')
        names = list(gym.get_asset_dof_names(asset))
        if len(names) != 6 or set(names) != set(cfg['asset']['policy_joint_names']):
            raise RuntimeError('Unexpected native joint list: ' + repr(names))
        if gym.get_asset_rigid_body_count(asset) != 7:
            raise RuntimeError('Expected seven rigid bodies')
        report['native_joint_order'] = names
        props = gym.get_asset_dof_properties(asset)
        helper.apply_actuator_limits(props, names, cfg['asset']['actuator_limits'])
        props['driveMode'].fill(int(gymapi.DOF_MODE_EFFORT))
        # External PD only; no hidden SDK position/velocity servo.
        props['stiffness'].fill(0.)
        props['damping'].fill(0.)
        for i, name in enumerate(names):
            props['armature'][i] = next((v for key, v in cfg['asset']['dof_armature'].items()
                                          if key in name), 0.)
        env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
        if env is None:
            raise RuntimeError('Failed to create environment')
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0., 0., 1.)
        pose.r = gymapi.Quat(*cfg['init_state']['rot'])
        # Isolate joint/controller dynamics before testing contact and self-collision.
        # -1 preserves asset collision settings; it does NOT explicitly disable
        # self-collision. A shared bit 1 filters all internal shape pairs here.
        actor = gym.create_actor(env, asset, pose, 'heroleg26_pd_check', 0, 1)
        if actor < 0:
            raise RuntimeError('Failed to create actor')
        report['rigid_shape_filters'] = verify_no_self_collision(gym, env, actor)
        print('Verified shape collision filters:', report['rigid_shape_filters'], flush=True)
        if not gym.set_actor_dof_properties(env, actor, props):
            raise RuntimeError('Could not set actor DOF properties')
        gym.prepare_sim(sim)
        actual_props = gym.get_actor_dof_properties(env, actor)
        for key in ('effort', 'velocity'):
            np.testing.assert_allclose(actual_props[key], props[key], rtol=1e-5, atol=1e-6)
        default = np.array([cfg['init_state']['default_joint_angles'][n] for n in names], dtype=np.float32)
        lower, upper = props['lower'].copy(), props['upper'].copy()
        limited = [i for i, n in enumerate(names) if joints[n].get('type') == 'revolute']
        wheels = [i for i, n in enumerate(names) if n.endswith('3')]
        legs = [i for i in range(6) if i not in wheels]
        if np.any(default[limited] < lower[limited]) or np.any(default[limited] > upper[limited]):
            raise ValueError('Default pose lies outside physical joint bounds')
        initial = np.zeros(6, dtype=gymapi.DofState.dtype)
        initial['pos'] = default
        if not gym.set_actor_dof_states(env, actor, initial, gymapi.STATE_ALL):
            raise RuntimeError('Could not set initial standing pose')
        kp = torch.tensor([[cfg['control']['stiffness'][n] for n in names]], dtype=torch.float32)
        kd = torch.tensor([[cfg['control']['damping'][n] for n in names]], dtype=torch.float32)
        effort = torch.tensor(props['effort'].copy())
        speed = torch.tensor(props['velocity'].copy())
        bounds = torch.tensor(np.stack((lower, upper), axis=-1))
        indices = torch.tensor(limited, dtype=torch.long)
        max_velocity = np.zeros(6)
        saturated = np.zeros(6, dtype=np.int64)
        dt = params.dt
        trace_path = output/'trace.csv'
        with trace_path.open('w') as trace:
            columns = ['time_s', 'phase']
            for prefix in ('q_rad_', 'v_rad_s_', 'commanded_tau_Nm_'):
                columns.extend(prefix+n for n in names)
            trace.write(','.join(columns)+'\n')
            for phase in phase_plan(cfg['asset']['policy_joint_names']):
                q_target = default.copy()
                v_target = np.zeros(6, dtype=np.float32)
                i = names.index(phase['joint']) if phase['joint'] else None
                if i is not None:
                    if i in wheels:
                        v_target[i] = phase['sign'] * 1.  # rad/s
                    else:
                        q_target[i] += phase['sign'] * math.radians(5.)
                        if i in limited and not lower[i] <= q_target[i] <= upper[i]:
                            raise ValueError('5-degree check target outside joint bounds: '+names[i])
                before = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
                tail = []
                count = round(phase['seconds']/dt)
                for step in range(count):
                    state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
                    q, v = state['pos'].copy(), state['vel'].copy()
                    if not (np.isfinite(q).all() and np.isfinite(v).all()):
                        raise RuntimeError('Non-finite joint state in '+phase['label'])
                    if np.any(np.abs(v) > 4*props['velocity']):
                        raise RuntimeError('Velocity exceeded 4x check operating limit in '+phase['label'])
                    if np.any(q[limited] < lower[limited]-.035) or np.any(q[limited] > upper[limited]+.035):
                        raise RuntimeError('Knee exceeded source bound by more than 0.035 rad')
                    torque = helper.compute_pd_torques(
                        torch.from_numpy(q_target[None]), torch.from_numpy(v_target[None]),
                        torch.from_numpy(q[None]), torch.from_numpy(v[None]), kp, kd, effort,
                        velocity_limits=speed if cfg['control']['enforce_operating_speed'] else None,
                        position_indices=indices if cfg['control']['clip_position_targets'] else None,
                        position_bounds=bounds,
                    )[0].numpy().astype(np.float32, copy=False)
                    if not np.isfinite(torque).all() or np.any(np.abs(torque) > props['effort']+1e-5):
                        raise RuntimeError('Invalid commanded torque')
                    if not gym.apply_actor_dof_efforts(env, actor, np.ascontiguousarray(torque)):
                        raise RuntimeError('Could not apply joint efforts')
                    gym.simulate(sim)
                    # Use blocking synchronization as in IsaacGymEnvs. Its
                    # return value is not a portable success predicate across
                    # bindings/backends; validate states and responses instead.
                    fetch_status = gym.fetch_results(sim, True)
                    if report['physics_steps'] == 0:
                        report['first_fetch_results_return'] = repr(fetch_status)
                    report['physics_steps'] += 1
                    if report['physics_steps'] <= 10 or report['physics_steps'] % 100 == 0:
                        contacts = len(gym.get_env_rigid_contacts(env))
                        report['maximum_sampled_contact_count'] = max(
                            report['maximum_sampled_contact_count'], contacts)
                        if contacts:
                            raise RuntimeError('Unexpected contact in the isolated fixed-base PD check')
                    max_velocity = np.maximum(max_velocity, np.abs(v))
                    saturated += np.abs(torque) >= .99*props['effort']
                    # q/v were sampled before applying this step's command.
                    row = [format((report['physics_steps']-1)*dt, '.6f'), phase['label']]
                    row += [format(float(x), '.8g') for arr in (q, v, torque) for x in arr]
                    trace.write(','.join(row)+'\n')
                    if step >= count-max(1, round(.25/dt)):
                        tail.append((q.copy(), v.copy()))
                if i is not None:
                    q_mean = np.mean([sample[0] for sample in tail], axis=0)
                    v_mean = np.mean([sample[1] for sample in tail], axis=0)
                    wheel = i in wheels
                    response = float(v_mean[i]-before['vel'][i] if wheel else q_mean[i]-before['pos'][i])
                    error = float(v_mean[i]-v_target[i] if wheel else q_mean[i]-q_target[i])
                    tolerance = .2 if wheel else math.radians(3.)
                    moved = phase['sign']*response > (.1 if wheel else math.radians(.5))
                    result = {'phase': phase['label'], 'joint': names[i], 'sign': phase['sign'],
                              'target': float(v_target[i] if wheel else q_target[i]),
                              'response_delta': response, 'tracking_error': error,
                              'unit': 'rad/s' if wheel else 'rad',
                              'direction_ok': bool(moved), 'tracking_ok': bool(abs(error) <= tolerance)}
                    report['checks'].append(result)
                    print('{}: response={:.5f}, error={:.5f} {}, {}'.format(
                        phase['label'], response, error, result['unit'],
                        'OK' if result['direction_ok'] and result['tracking_ok'] else 'NEEDS_REVIEW'), flush=True)
        # Re-read final state after the last completed physics step.
        final = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
        if not np.isfinite(final['pos']).all() or not np.isfinite(final['vel']).all():
            raise RuntimeError('Non-finite final state')
        report['maximum_abs_speed_rad_s'] = dict(zip(names, max_velocity.tolist()))
        report['torque_saturation_fraction'] = dict(zip(names, (saturated/report['physics_steps']).tolist()))
        report['final_angles_rad'] = dict(zip(names, final['pos'].tolist()))
        report['final_velocities_rad_s'] = dict(zip(names, final['vel'].tolist()))
        report['final_stop_ok'] = bool(np.all(np.abs(final['vel'][wheels]) <= .2))
        report['final_pose_ok'] = bool(np.all(np.abs(final['pos'][legs]-default[legs]) <= math.radians(3.)))
        report['status'] = 'PASS' if (len(report['checks']) == 12 and report['final_stop_ok'] and report['final_pose_ok'] and all(
            r['direction_ok'] and r['tracking_ok'] for r in report['checks'])) else 'NEEDS_REVIEW'
    except BaseException as error:
        report['status'] = 'ERROR'
        report['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        try:
            if sim is not None:
                gym.destroy_sim(sim)
        finally:
            changed = [p for p, digest in protected.items() if sha256(Path(p)) != digest]
            report['source_files_unchanged'] = not changed
            if changed:
                report['changed_source_files'] = changed
                report['status'] = 'ERROR'
            (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print('PD check status:', report['status'], flush=True)
            print('Physics steps:', report['physics_steps'], flush=True)
            print('Report:', output/'report.json', flush=True)
            print('Trace:', output/'trace.csv', flush=True)
    return 0 if report['status'] == 'PASS' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--branch', choices=('plane', 'jump'), default='plane')
    parser.add_argument('--compute-device', type=int, default=0)
    parser.add_argument('--dt', type=float, help='Diagnostic control/physics period only; leave source config unchanged')
    parser.add_argument('--check-config', action='store_true', help='Read config only; no SDK, GPU or writes')
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    cfg = diagnostic_config(load_config(repo, args.branch), args.dt)
    if args.check_config:
        print(json.dumps(cfg, indent=2))
        return 0
    if args.compute_device < 0:
        parser.error('--compute-device must be nonnegative')
    if os.geteuid() == 0 or Path.home().resolve() != Path(pwd.getpwuid(os.getuid()).pw_dir).resolve():
        raise RuntimeError('Use an ordinary account with its own HOME')
    for key, suffix in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                        ('CUDA_CACHE_PATH', 'cuda'), ('TORCH_EXTENSIONS_DIR', 'torch_extensions')):
        path = user_path(Path.home()/'.cache/heroleg26'/suffix)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    output = user_path(Path.home()/'.local/state/heroleg26/isaacgym'/(
        'pd-fixed-'+args.branch+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    output.mkdir(parents=True)
    user_path(repo/'heroleg26_isaac_check/generated').mkdir(parents=True, exist_ok=True)
    os.chdir(str(output))
    sys.dont_write_bytecode = True
    print('Headless fixed-base PD check; no policy, no training.', flush=True)
    print('Output:', output, flush=True)
    return run_check(repo, cfg, output, args.compute_device)


if __name__ == '__main__':
    sys.exit(main())
