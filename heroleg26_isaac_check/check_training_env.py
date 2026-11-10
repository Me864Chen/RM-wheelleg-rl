#!/usr/bin/env python3
"""Exercise the real heroleg26 task and its base-contact reset; no PPO training."""
import argparse
import copy
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import pwd
import resource
import sys
import xml.etree.ElementTree as ET

from check_isaacgym import prepare, sha256, user_path
from check_pd import load_config, literal_section

ENTRY_PATH = Path(__file__).resolve()


def training_rules(repo, branch):
    pkg = repo / branch / 'wheel_legged_gym'
    sources = [(pkg/'envs/base/legged_robot_config.py', 'LeggedRobotCfg'),
               (pkg/'envs/wheel_legged/wheel_legged_config.py', 'WheelLeggedCfg'),
               (pkg/'envs/heroleg26/heroleg26_config.py', 'Heroleg26Cfg')]
    fields = {'env': ('fail_to_terminal_time_s',),
              'asset': ('penalize_contacts_on', 'terminate_after_contacts_on')}
    result = {section: {} for section in fields}
    for path, cls in sources:
        for section, keys in fields.items():
            result[section].update(literal_section(path, cls, section, keys))
    if result['asset']['penalize_contacts_on'] != ['base_link']:
        raise ValueError('Expected base_link collision penalty; synchronize the task config.')
    if result['asset']['terminate_after_contacts_on'] != ['base_link']:
        raise ValueError('Expected base_link contact termination; synchronize the task config.')
    if result['env']['fail_to_terminal_time_s'] != .05:
        raise ValueError('Expected 0.05 s failure debounce; synchronize the task config.')
    return result


def configure_check(cfg, count, kp_scale, kd_scale):
    """Temporary diagnostic overrides. Preserve the asset, stance and limits."""
    cfg.env.num_envs = count
    cfg.env.episode_length_s = 2.
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.commands.ranges.lin_vel_x = [0., 0.]
    cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    cfg.noise.add_noise = False
    for name in dir(cfg.domain_rand):
        if name.startswith('randomize_') and isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    cfg.domain_rand.push_robots = False
    # Do not mutate dictionaries shared with the original task in this process.
    cfg.control.stiffness = copy.deepcopy(cfg.control.stiffness)
    cfg.control.damping = copy.deepcopy(cfg.control.damping)
    for name in ('RJ1', 'RJ2', 'LJ1', 'LJ2'):
        cfg.control.stiffness[name] *= kp_scale
        cfg.control.damping[name] *= kd_scale


def body_names_by_handle(env, link_names):
    """Resolve tensor body order through the same SDK lookup used by the task."""
    names = [None] * env.num_bodies
    for name in link_names:
        index = env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
        if index < 0 or index >= len(names) or names[index] is not None:
            raise RuntimeError('Invalid or duplicate body handle for '+name)
        names[index] = name
    if any(name is None for name in names):
        raise RuntimeError('URDF links do not cover every task body.')
    return names


def observe_termination(env):
    """Snapshot real task states before reset_idx replaces terminal states."""
    check = env.check_termination
    captured = {}

    def wrapped_check():
        check()
        captured.clear()
        for name in ('root_states', 'dof_pos', 'dof_vel', 'torques',
                     'projected_gravity', 'contact_forces', 'reset_buf',
                     'time_out_buf', 'fail_buf', 'episode_length_buf'):
            captured[name] = getattr(env, name).detach().clone()
        captured['collision_penalty_raw'] = env._reward_collision().detach().clone()

    env.check_termination = wrapped_check
    return captured


def run_check(repo, args, output, protected):
    import isaacgym  # Must precede torch.
    import torch
    import wheel_legged_gym
    import wheel_legged_gym.envs
    from wheel_legged_gym.utils import get_args, task_registry

    expected = repo / args.branch / 'wheel_legged_gym'
    if Path(wheel_legged_gym.__file__).resolve().parent != expected:
        raise RuntimeError('Unexpected wheel_legged_gym import: '+wheel_legged_gym.__file__)
    sys.argv = ['environment_check', '--task', 'heroleg26', '--num_envs', str(args.num_envs),
                '--seed', '1', '--headless', '--sim_device', 'cuda:0', '--pipeline', 'gpu']
    gym_args = get_args()
    cfg, _ = task_registry.get_cfgs('heroleg26')
    configure_check(cfg, args.num_envs, args.leg_kp_scale, args.leg_kd_scale)
    report = {'status': 'RUNNING', 'branch': args.branch, 'physical_gpu': args.gpu,
              'steps_requested': args.steps, 'steps_completed': 0,
              'num_envs': args.num_envs, 'source_sha256': protected,
              'rules': training_rules(repo, args.branch),
              'leg_kp_scale': args.leg_kp_scale, 'leg_kd_scale': args.leg_kd_scale,
              'reset_events': 0, 'timeout_events': 0, 'base_contact_reset_events': 0,
              'base_contact_samples': 0, 'collision_penalty_samples': 0,
              'height_reference_status': 'NOT_CALIBRATED',
              'limitations': 'Zero actions, no policy or training. Domain randomization is disabled; '
                             'the real task still randomizes root velocity on reset. '
                             'Pre-reset snapshots are diagnostic; sparse upright samples are not a height calibration.'}
    env = None
    try:
        print('GPU cuda:0:', torch.cuda.get_device_name(0), flush=True)
        env, _ = task_registry.make_env('heroleg26', args=gym_args, env_cfg=cfg)
        source = ET.parse(repo/'heroleg26_sim/urdf/heroleg26.urdf').getroot()
        body_names = body_names_by_handle(env, [link.get('name') for link in source.findall('link')])
        base_index = body_names.index('base_link')
        expected_indices = [base_index]
        if env.termination_contact_indices.tolist() != expected_indices:
            raise RuntimeError('The task did not bind base_link as its termination body.')
        if env.penalised_contact_indices.tolist() != expected_indices:
            raise RuntimeError('The task did not bind base_link as its penalty body.')
        report['native_joint_order'] = list(env.dof_names)
        report['body_order'] = body_names
        report['physics_dt_s'], report['action_dt_s'] = env.sim_params.dt, env.dt
        report['leg_kp'] = dict(cfg.control.stiffness)
        report['leg_kd'] = dict(cfg.control.damping)
        snapshot = observe_termination(env)
        fields = ['action_step', 'env', 'episode_time_s', 'base_z_m', 'tilt_deg',
                  'base_contact_norm_N', 'left_wheel_fz_N', 'right_wheel_fz_N',
                  'failure_steps', 'done', 'timeout', 'collision_penalty_raw']
        fields += ['q_rad_'+n for n in env.dof_names]
        fields += ['v_rad_s_'+n for n in env.dof_names]
        candidate_heights = []
        consecutive = [0] * env.num_envs
        longest = 0
        left, right = [body_names.index(n) for n in ('L_Link3', 'R_Link3')]
        print('DOFs:', env.dof_names, flush=True)
        print('Base contact resets after consecutive >10 N samples for >0.05 s.', flush=True)
        with (output/'trace.csv').open('w', newline='') as stream, torch.no_grad():
            writer = csv.writer(stream)
            writer.writerow(fields)
            env.reset()
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            for step in range(args.steps):
                obs, privileged, reward, done, info, history = env.step(actions)
                values = dict(snapshot, obs=obs, privileged=privileged, reward=reward, history=history)
                for name, value in values.items():
                    if value is not None and not torch.isfinite(value).all().item():
                        raise RuntimeError('Non-finite values in '+name)
                if tuple(obs.shape) != (env.num_envs, env.num_obs):
                    raise RuntimeError('Unexpected observation shape')
                if tuple(history.shape) != (env.num_envs, env.num_obs * env.obs_history_length):
                    raise RuntimeError('Unexpected history shape')
                if privileged is not None and tuple(privileged.shape) != (env.num_envs, env.num_privileged_obs):
                    raise RuntimeError('Unexpected privileged observation shape')
                data = {k: v.cpu().tolist() for k, v in snapshot.items()}
                for i in range(env.num_envs):
                    root = data['root_states'][i]
                    force = data['contact_forces'][i]
                    base_norm = math.sqrt(sum(v*v for v in force[base_index]))
                    tilt = math.degrees(math.acos(max(-1., min(1., -data['projected_gravity'][i][2]))))
                    ended, timeout = bool(data['reset_buf'][i]), bool(data['time_out_buf'][i])
                    report['reset_events'] += int(ended)
                    report['timeout_events'] += int(timeout)
                    report['base_contact_samples'] += int(base_norm > 10.)
                    report['collision_penalty_samples'] += int(data['collision_penalty_raw'][i] > 0.)
                    # Exclude the inherited near-inversion termination and timeouts.
                    report['base_contact_reset_events'] += int(
                        ended and not timeout and base_norm > 10. and tilt < 60.)
                    writer.writerow([step+1, i, data['episode_length_buf'][i]*env.dt,
                                     root[2], tilt, base_norm, force[left][2], force[right][2],
                                     data['fail_buf'][i], int(ended), int(timeout),
                                     data['collision_penalty_raw'][i]] + data['dof_pos'][i] + data['dof_vel'][i])
                    pose_error = max(abs(data['dof_pos'][i][j]-cfg.init_state.default_joint_angles[n])
                                     for j, n in enumerate(env.dof_names) if n in ('LJ1', 'LJ2', 'RJ1', 'RJ2'))
                    candidate = (not ended and tilt < 5. and base_norm < 1.
                                 and force[left][2] > 1. and force[right][2] > 1.
                                 and pose_error < math.radians(5.)
                                 and math.sqrt(sum(v*v for v in root[7:10])) < .05)
                    consecutive[i] = consecutive[i]+1 if candidate else 0
                    longest = max(longest, consecutive[i])
                    if candidate:
                        candidate_heights.append(root[2])
                report['steps_completed'] = step+1
                if (step+1) % 100 == 0:
                    print('Step:', step+1, 'base-contact resets:', report['base_contact_reset_events'], flush=True)
        torch.cuda.synchronize()
        report['observation_shape'], report['history_shape'] = list(obs.shape), list(history.shape)
        report['upright_wheel_support_candidates'] = {
            'samples': len(candidate_heights), 'longest_continuous_s': longest*env.dt,
            'height_min_m': min(candidate_heights) if candidate_heights else None,
            'height_max_m': max(candidate_heights) if candidate_heights else None}
        report['status'] = 'PASS' if report['base_contact_reset_events'] else 'NEEDS_REVIEW'
        if report['collision_penalty_samples'] == 0:
            report['status'] = 'NEEDS_REVIEW'
        if report['status'] != 'PASS':
            report['review_reason'] = 'Expected base-contact penalty/reset was not exercised; inspect the trace.'
    except BaseException as error:
        report['status'], report['error'] = 'ERROR', type(error).__name__+': '+str(error)
        raise
    finally:
        try:
            if env is not None:
                env.gym.destroy_sim(env.sim)
        finally:
            changed = [p for p, digest in protected.items()
                       if not Path(p).is_file() or sha256(Path(p)) != digest]
            report['source_files_unchanged'] = not changed
            if changed:
                report['status'], report['changed_source_files'] = 'ERROR', changed
            (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('Environment contact-rule check:', report['status'], flush=True)
            print('Base-contact resets:', report['base_contact_reset_events'], flush=True)
            print('Height reference: NOT_CALIBRATED; no balance policy was tested.', flush=True)
            print('Report:', output/'report.json', flush=True)
            print('Trace:', output/'trace.csv', flush=True)
    return 0 if report['status'] == 'PASS' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ENTRY_PATH.parents[1])
    parser.add_argument('--branch', choices=('plane', 'jump'), default='plane')
    parser.add_argument('--gpu', type=int, default=1, help='Physical GPU in PCI bus order')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--leg-kp-scale', type=float, default=1., help='Diagnostic only; wheel gains unchanged')
    parser.add_argument('--leg-kd-scale', type=float, default=1., help='Diagnostic only; wheel gains unchanged')
    parser.add_argument('--check-config', action='store_true', help='Read configuration only; no SDK/GPU or writes')
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    if not 1 <= args.num_envs <= 16 or not 200 <= args.steps <= 5000 or args.gpu < 0:
        parser.error('Require 1..16 environments, 200..5000 steps and a nonnegative GPU index.')
    for value in (args.leg_kp_scale, args.leg_kd_scale):
        if not math.isfinite(value) or not .25 <= value <= 4.:
            parser.error('Diagnostic gain scales must be within 0.25..4.')
    rules = training_rules(repo, args.branch)
    cfg = load_config(repo, args.branch)
    if args.check_config:
        print(json.dumps({'rules': rules, 'configuration': cfg,
                          'leg_kp_scale': args.leg_kp_scale, 'leg_kd_scale': args.leg_kd_scale}, indent=2))
        return 0
    if os.geteuid() == 0 or Path.home().resolve() != Path(pwd.getpwuid(os.getuid()).pw_dir).resolve():
        raise RuntimeError('Use an ordinary account and its own HOME.')
    user_path(repo)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ['CUDA_DEVICE_ORDER'], os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID', str(args.gpu)
    for name, suffix in (('TORCH_EXTENSIONS_DIR', 'torch_extensions'), ('MPLCONFIGDIR', 'matplotlib'),
                         ('XDG_CACHE_HOME', 'xdg'), ('CUDA_CACHE_PATH', 'cuda'), ('TMPDIR', 'tmp')):
        path = user_path(Path.home()/'.cache/heroleg26'/suffix)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['MAX_JOBS'] = '2'
    sys.dont_write_bytecode = True
    output = user_path(Path.home()/'.local/state/heroleg26/isaacgym'/(
        'env-contacts-'+args.branch+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    output.mkdir(parents=True)
    manifest = prepare(repo/'heroleg26_sim/urdf/heroleg26.urdf',
                       user_path(repo/'heroleg26_isaac_check/generated'))
    protected = dict(manifest['original_sha256'])
    files = [Path(p) for p in cfg['source_files']]+[
        ENTRY_PATH, ENTRY_PATH.with_name('check_pd.py'), ENTRY_PATH.with_name('check_isaacgym.py'),
        repo/args.branch/'wheel_legged_gym/envs/base/legged_robot.py',
        repo/args.branch/'wheel_legged_gym/utils/motor_limits.py']
    protected.update({str(p): sha256(p) for p in files})
    sys.path.insert(0, str(repo/args.branch))
    os.chdir(str(output))
    print('Real task contact-rule check; zero actions, no policy or training.', flush=True)
    print('Output:', output, flush=True)
    return run_check(repo, args, output, protected)


if __name__ == '__main__':
    sys.exit(main())
