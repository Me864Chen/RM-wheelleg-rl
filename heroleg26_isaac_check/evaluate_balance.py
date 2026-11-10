#!/usr/bin/env python3
"""Headless deterministic policy evaluation against a matched zero-action baseline."""
import argparse
import ast
import copy
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import pwd
import random
import resource
import sys
import xml.etree.ElementTree as ET

from check_isaacgym import prepare, sha256, user_path
from check_training_env import body_names_by_handle, observe_termination, training_rules

ENTRY_PATH = Path(__file__).resolve()
RESET_FIX = ('self.dof_state.view(self.num_envs, self.num_dof, 2)'
             '[env_ids, :, 1] = 0.0')


def restore_config(obj, values):
    """Restore saved runtime values without replacing nested config objects by dicts."""
    for name, value in values.items():
        if name.startswith('_') or not hasattr(obj, name):
            continue
        current = getattr(obj, name)
        if callable(current):
            continue
        if isinstance(value, dict) and not isinstance(current, dict) and hasattr(current, '__dict__'):
            restore_config(current, value)
        else:
            setattr(obj, name, copy.deepcopy(value))


def check_reset_fix(current_path, snapshot_path, trained_digest):
    current = ast.parse(current_path.read_text())
    cls = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == 'LeggedRobot')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_reset_dofs')
    expected = ast.dump(ast.parse(RESET_FIX).body[0], include_attributes=False)
    fixes = [n for n in fn.body if ast.dump(n, include_attributes=False) == expected]
    if len(fixes) != 1:
        raise RuntimeError('The native SDK joint-velocity reset fix is missing.')
    if sha256(current_path) == trained_digest:
        return 'Already fixed in training source'
    if not snapshot_path.is_file() or sha256(snapshot_path) != trained_digest:
        raise RuntimeError('Original training source snapshot is missing or changed.')
    fn.body.remove(fixes[0])
    original = ast.parse(snapshot_path.read_text())
    if ast.dump(current, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise RuntimeError('Environment changes beyond the velocity reset fix require review.')
    return 'Only the native joint-velocity reset differs from training'


def initial_cases(count, seed):
    rng = random.Random(seed)
    return [('zero_velocity', 0, [0.]*6)] + [
        ('training_velocity_range', i, [rng.uniform(-.5, .5) for _ in range(6)])
        for i in range(count)]


def select_actions(policy, obs, history, torch):
    if policy is None:
        return torch.zeros((obs.shape[0], 6), device=obs.device)
    actions, _ = policy.act_inference(obs, history)
    if actions.shape != (obs.shape[0], 6) or not torch.isfinite(actions).all().item():
        raise RuntimeError('Invalid deterministic policy actions.')
    return actions


def clean_balance(episode):
    # Explicit preliminary evaluation tolerances, not physical/mechanical limits.
    return (episode['end_reason'] == 'time_limit' and episode['non_wheel_loaded_samples'] == 0
            and episode['both_wheels_loaded_fraction'] >= .95
            and episode['maximum_tilt_deg'] <= 10.
            and episode['maximum_xy_displacement_m'] <= .25)


def run_case(repo, saved, checkpoint, controller, cases, seconds, output, args):
    import isaacgym
    from isaacgym import gymtorch
    import torch
    import wheel_legged_gym
    import wheel_legged_gym.envs
    from wheel_legged_gym.utils import get_args, task_registry
    from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence

    if Path(wheel_legged_gym.__file__).resolve().parent != repo/'plane/wheel_legged_gym':
        raise RuntimeError('Unexpected framework import path.')
    sys.argv = ['evaluate_balance', '--task', 'heroleg26', '--num_envs', '1', '--seed', str(args.seed),
                '--headless', '--sim_device', 'cuda:0', '--pipeline', 'gpu']
    gym_args = get_args()
    cfg, _ = task_registry.get_cfgs('heroleg26')
    restore_config(cfg, saved['env'])
    cfg.env.num_envs, cfg.env.episode_length_s = 1, seconds
    cfg.commands.curriculum = cfg.terrain.curriculum = False
    if cfg.terrain.mesh_type != 'plane' or cfg.asset.fix_base_link:
        raise RuntimeError('Evaluation requires the original free-base plane setup.')
    if cfg.commands.ranges.lin_vel_x != [0., 0.] or cfg.commands.ranges.ang_vel_yaw != [0., 0.]:
        raise RuntimeError('Expected the saved zero-speed training profile.')
    if cfg.noise.add_noise or any(getattr(cfg.domain_rand, n) for n in dir(cfg.domain_rand)
                                 if n.startswith('randomize_') and isinstance(getattr(cfg.domain_rand, n), bool)):
        raise RuntimeError('Unexpected randomization in the saved smoke configuration.')
    policy = None
    if controller == 'policy_mean':
        options = saved['train']['policy']
        if saved['train']['runner']['policy_class_name'] != 'ActorCriticSequence':
            raise RuntimeError('This entry expects the trained sequence policy.')
        critic_obs = cfg.env.num_privileged_obs + options['latent_dim']
        policy = ActorCriticSequence(cfg.env.num_observations, critic_obs,
                                    cfg.env.num_actions, **options).to('cuda:0')
        weights = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        policy.load_state_dict(weights['model_state_dict'], strict=True)
        policy.eval()
    env = None
    try:
        env, _ = task_registry.make_env('heroleg26', args=gym_args, env_cfg=cfg)
        xml = ET.parse(repo/'heroleg26_sim/urdf/heroleg26.urdf').getroot()
        bodies = body_names_by_handle(env, [n.get('name') for n in xml.findall('link')])
        base, left, right = [bodies.index(n) for n in ('base_link', 'L_Link3', 'R_Link3')]
        knees = {i: (float(j.find('limit').get('lower')), float(j.find('limit').get('upper')))
                 for i, name in enumerate(env.dof_names) for j in xml.findall('joint')
                 if j.get('name') == name and name in ('LJ2', 'RJ2')}
        snapshot = observe_termination(env)
        capture = env.check_termination

        def capture_full():
            capture()
            snapshot['sdk_joint_velocity'] = env.dof_state.view(1, env.num_dof, 2)[:, :, 1].detach().clone()
            snapshot['actions'] = env.actions.detach().clone()

        env.check_termination = capture_full
        planned_velocity = torch.zeros(6, device=env.device)
        reset_root = env._reset_root_states

        def reset_with_planned_velocity(ids):
            reset_root(ids)
            native_vel = env.dof_state.view(1, env.num_dof, 2)[ids, :, 1]
            if native_vel.abs().max().item() > 1.e-8:
                raise RuntimeError('A reset retained SDK joint velocity.')
            env.root_states[ids, 7:13] = planned_velocity
            ids32 = ids.to(dtype=torch.int32)
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim, gymtorch.unwrap_tensor(env.root_states), gymtorch.unwrap_tensor(ids32), len(ids32))

        env._reset_root_states = reset_with_planned_velocity
        # Exact timeout boundary for both controllers, with the original failure logic unchanged.
        steps = max(1, int(math.ceil(seconds/env.dt)))
        env.max_episode_length = steps-1
        episodes = []
        fieldnames = ['scenario', 'trial', 'time_s', 'done', 'timeout', 'base_x_m', 'base_y_m',
                      'base_z_m', 'tilt_deg', 'base_contact_norm_N', 'base_contact_fz_N',
                      'left_wheel_fz_N', 'right_wheel_fz_N', 'torque_saturation_fraction']
        fieldnames += ['q_rad_'+n for n in env.dof_names]
        fieldnames += ['sdk_v_rad_s_'+n for n in env.dof_names]
        fieldnames += ['tau_Nm_'+n for n in env.dof_names]
        fieldnames += ['policy_action_'+str(i) for i in range(6)]
        fieldnames += ['raw_policy_action_'+str(i) for i in range(6)]
        with (output/(controller+'.csv')).open('w', newline='') as stream, torch.no_grad():
            writer = csv.writer(stream)
            writer.writerow(fieldnames)
            for scenario, trial, velocity in cases:
                planned_velocity[:] = torch.tensor(velocity, device=env.device)
                episode = {'controller': controller, 'scenario': scenario, 'trial': trial,
                           'initial_root_velocity': velocity, 'duration_s': 0., 'end_reason': None,
                           'maximum_tilt_deg': 0., 'maximum_xy_displacement_m': 0.,
                           'base_loaded_samples': 0, 'base_contact_over_10N_samples': 0,
                           'non_wheel_loaded_samples': 0, 'both_wheels_loaded_samples': 0,
                           'first_base_load_time_s': None, 'torque_saturated_values': 0,
                           'action_clipped_values': 0, 'knee_near_limit_samples': 0}
                tilt_sum = force_sum = 0.
                unexpected_bodies = set()
                for step in range(steps):
                    if step == 0:
                        # Standard task reset includes one shared zero-action warmup step.
                        env.reset()
                        # Independent trials start with current observations, not
                        # terminal history left over from the preceding trial.
                        env.obs_history[:] = env.obs_buf.repeat(1, env.obs_history_length)
                        raw_actions = torch.zeros((1, 6), device=env.device)
                    else:
                        obs, history = env.get_observations()
                        raw_actions = select_actions(policy, obs, history, torch)
                        env.step(raw_actions)
                    if any(not torch.isfinite(v).all().item() for v in snapshot.values()):
                        raise RuntimeError('Non-finite pre-reset evaluation state.')
                    data = {k: v.cpu().tolist()[0] for k, v in snapshot.items()}
                    p, forces, q = data['root_states'], data['contact_forces'], data['dof_pos']
                    base_norm = math.sqrt(sum(v*v for v in forces[base]))
                    tilt = math.degrees(math.acos(max(-1., min(1., -data['projected_gravity'][2]))))
                    displacement = math.hypot(p[0]-float(env.env_origins[0, 0]),
                                              p[1]-float(env.env_origins[0, 1]))
                    saturated = int((snapshot['torques'].abs() >= .99*env.torque_limits).sum().item())
                    elapsed = (step+1)*env.dt
                    episode['duration_s'] = elapsed
                    episode['maximum_tilt_deg'] = max(episode['maximum_tilt_deg'], tilt)
                    episode['maximum_xy_displacement_m'] = max(episode['maximum_xy_displacement_m'], displacement)
                    episode['base_loaded_samples'] += int(forces[base][2] > 1.)
                    non_wheel_load = {bodies[i] for i, f in enumerate(forces)
                                      if i not in (left, right) and f[2] > 1.}
                    unexpected_bodies.update(non_wheel_load)
                    episode['non_wheel_loaded_samples'] += int(bool(non_wheel_load))
                    episode['both_wheels_loaded_samples'] += int(forces[left][2] > 1. and forces[right][2] > 1.)
                    episode['base_contact_over_10N_samples'] += int(base_norm > 10.)
                    if forces[base][2] > 1. and episode['first_base_load_time_s'] is None:
                        episode['first_base_load_time_s'] = elapsed
                    episode['torque_saturated_values'] += saturated
                    episode['action_clipped_values'] += int((raw_actions.abs() > cfg.normalization.clip_actions).sum().item())
                    episode['knee_near_limit_samples'] += int(any(
                        q[i] < low+math.radians(1.) or q[i] > high-math.radians(1.)
                        for i, (low, high) in knees.items()))
                    tilt_sum += tilt
                    force_sum += forces[base][2]
                    writer.writerow([scenario, trial, elapsed, int(data['reset_buf']), int(data['time_out_buf'])]
                                    + p[:3] + [tilt, base_norm, forces[base][2], forces[left][2], forces[right][2],
                                               saturated/6.] + q + data['sdk_joint_velocity'] + data['torques']
                                    + data['actions'] + raw_actions.cpu().tolist()[0])
                    if data['reset_buf']:
                        if base_norm > 10. and data['fail_buf'] > cfg.env.fail_to_terminal_time_s/env.dt:
                            episode['end_reason'] = 'base_contact'
                        elif data['projected_gravity'][2] > -.1:
                            episode['end_reason'] = 'near_inversion'
                        elif data['time_out_buf']:
                            episode['end_reason'] = 'time_limit'
                        else:
                            episode['end_reason'] = 'other_task_failure'
                        break
                if episode['end_reason'] is None:
                    raise RuntimeError('The expected timeout or failure was not observed.')
                samples = step+1
                episode['mean_tilt_deg'] = tilt_sum/samples
                episode['mean_base_fz_N'] = force_sum/samples
                episode['base_loaded_fraction'] = episode['base_loaded_samples']/samples
                episode['both_wheels_loaded_fraction'] = episode['both_wheels_loaded_samples']/samples
                episode['non_wheel_loaded_bodies'] = sorted(unexpected_bodies)
                episode['torque_saturation_fraction'] = episode.pop('torque_saturated_values')/(samples*6)
                episode['action_clipping_fraction'] = episode.pop('action_clipped_values')/(samples*6)
                episode['clean_balance'] = clean_balance(episode)
                episodes.append(episode)
                print(controller, scenario, trial, 'duration={:.3f}s'.format(episode['duration_s']),
                      episode['end_reason'], 'max_tilt={:.2f}deg'.format(episode['maximum_tilt_deg']), flush=True)
                (output/(controller+'_episodes.json')).write_text(json.dumps(episodes, indent=2)+'\n')
        return episodes
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ENTRY_PATH.parents[1])
    parser.add_argument('--run', type=Path, required=True, help='Original PPO run directory, containing checkpoint and effective_config.json')
    parser.add_argument('--checkpoint', default='model_100.pt', help='Checkpoint filename inside --run')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=8, help='Matched random-velocity trials, plus one stationary trial per controller')
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 32 or not 2 <= args.seconds <= 10 or args.gpu < 0:
        parser.error('Require 1..32 random trials, a 2..10 s horizon and nonnegative GPU index.')
    if Path(args.checkpoint).name != args.checkpoint:
        parser.error('--checkpoint must be a filename.')
    repo, run = args.repo.expanduser().resolve(), args.run.expanduser().resolve()
    training_rules(repo, 'plane')
    saved = json.loads((run/'effective_config.json').read_text())
    trained = json.loads((run/'report.json').read_text())
    source = repo/'plane/wheel_legged_gym/envs/base/legged_robot.py'
    recorded = trained['source_sha256']
    suffix = 'plane/wheel_legged_gym/envs/base/legged_robot.py'
    old_path = next(p for p in recorded if p.endswith('/'+suffix))
    compatibility = check_reset_fix(source, run/'source_snapshot'/suffix, recorded[old_path])
    protected = {}
    for old, digest in recorded.items():
        marker = '/RM-wheelleg-rl/'
        if marker not in old:
            continue
        relative = old.split(marker, 1)[1]
        path = repo/relative
        if path != source and (not path.is_file() or sha256(path) != digest):
            raise RuntimeError('Source/model differs from training: '+relative)
        protected[str(path)] = sha256(path)
    checkpoint = run/args.checkpoint
    for path in (checkpoint, run/'effective_config.json', run/'report.json', ENTRY_PATH):
        protected[str(path)] = sha256(path)
    cases = initial_cases(args.episodes, args.seed)
    plan = {'controllers': ['zero_actions', 'policy_mean'], 'cases_per_controller': len(cases),
            'maximum_episode_s': args.seconds, 'physical_gpu': args.gpu, 'seed': args.seed,
            'checkpoint': str(checkpoint), 'checkpoint_sha256': protected[str(checkpoint)],
            'root_velocity_cases': cases, 'reset_source_compatibility': compatibility,
            'clean_balance_criteria': 'Entire horizon; no non-wheel upward force >1 N; both wheels loaded >1 N '
                                     'for >=95% of samples; tilt <=10 degrees; xy drift <=0.25 m.',
            'limitations': 'Matched initial velocities, not guaranteed bitwise GPU determinism. One shared zero-action '
                           'warmup per reset, followed by filling history with the current observation. '
                           'No policy sampling, PPO update, external push, terrain or real-robot test. '
                           'Evaluation uses the corrected zero-joint-velocity reset; training used the old reset.'}
    if args.check_config:
        print(json.dumps(plan, indent=2))
        return 0
    if os.geteuid() == 0 or Path.home().resolve() != Path(pwd.getpwuid(os.getuid()).pw_dir).resolve():
        raise RuntimeError('Use an ordinary account and its own HOME.')
    user_path(repo)
    user_path(run)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.dont_write_bytecode = True
    os.environ['CUDA_DEVICE_ORDER'], os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID', str(args.gpu)
    for key, suffix in (('TORCH_EXTENSIONS_DIR', 'torch_extensions'), ('MPLCONFIGDIR', 'matplotlib'),
                        ('XDG_CACHE_HOME', 'xdg'), ('CUDA_CACHE_PATH', 'cuda'), ('TMPDIR', 'tmp')):
        directory = user_path(Path.home()/'.cache/heroleg26'/suffix)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ['MAX_JOBS'] = '2'
    manifest = prepare(repo/'heroleg26_sim/urdf/heroleg26.urdf', user_path(repo/'heroleg26_isaac_check/generated'))
    protected.update(manifest['original_sha256'])
    output = user_path(Path.home()/'.local/state/heroleg26/isaacgym'/(
        'eval-balance-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    output.mkdir(parents=True)
    sys.path.insert(0, str(repo/'plane'))
    os.chdir(str(output))
    report = {'status': 'RUNNING', 'plan': plan, 'source_sha256': protected, 'results': {}}
    print('Deterministic evaluation; no training. Output:', output, flush=True)
    try:
        for controller in plan['controllers']:
            episodes = run_case(repo, saved, checkpoint, controller, cases, args.seconds, output, args)
            summary = {}
            for scenario in ('zero_velocity', 'training_velocity_range'):
                selected = [e for e in episodes if e['scenario'] == scenario]
                summary[scenario] = {'trials': len(selected),
                    'mean_duration_s': sum(e['duration_s'] for e in selected)/len(selected),
                    'clean_balance_trials': sum(e['clean_balance'] for e in selected),
                    'time_limit_trials': sum(e['end_reason'] == 'time_limit' for e in selected)}
            report['results'][controller] = {'episodes': episodes, 'summary': summary}
        report['status'] = 'COMPLETED'
    except BaseException as error:
        report['status'], report['error'] = 'ERROR', type(error).__name__+': '+str(error)
        raise
    finally:
        changed = [p for p, digest in protected.items() if not Path(p).is_file() or sha256(Path(p)) != digest]
        report['source_files_unchanged'] = not changed
        if changed:
            report['status'], report['changed_source_files'] = 'ERROR', changed
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('Evaluation:', report['status'], flush=True)
        for name, data in report['results'].items():
            print(name, json.dumps(data['summary']), flush=True)
        print('Report:', output/'report.json', flush=True)
    return 0 if report['status'] == 'COMPLETED' else 2


if __name__ == '__main__':
    sys.exit(main())
