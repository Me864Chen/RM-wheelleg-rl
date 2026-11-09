"""CPU tests for the actual env tensor methods, without importing Isaac Gym.

Select pure-Torch methods from the env AST to avoid importing unavailable native
SDK bindings. No substitute controller or observation implementation is tested.
"""
import ast
import contextlib
import importlib.util
import io
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ('plane', 'jump')
POLICY = ['RJ1', 'RJ2', 'RJ3', 'LJ1', 'LJ2', 'LJ3']
LEGACY = ['lf0_Joint', 'lf1_Joint', 'l_wheel_Joint',
          'rf0_Joint', 'rf1_Joint', 'r_wheel_Joint']
ALIASES = dict(zip(LEGACY, POLICY))
HEROCFG = {}
METHODS = {'_process_dof_props', '_init_joint_mapping', '_get_leg_joint_angles', '_compute_torques',
           'compute_proprioception_observations', 'compute_observations',
           '_reward_dof_vel', '_reward_dof_pos_limits', '_body_to_policy',
           '_policy_to_body', 'forward_kinematics', '_update_leg_kinematics',
           '_init_height_points', '_reward_tracking_lin_vel'}


def load_code(branch):
    pkg = ROOT / branch / 'wheel_legged_gym'
    spec = importlib.util.spec_from_file_location(
        'joint_mapping_' + branch, pkg / 'utils/joint_mapping.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    spec = importlib.util.spec_from_file_location(
        'robot_kinematics_' + branch, pkg / 'utils/robot_kinematics.py')
    kinematics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kinematics)
    spec = importlib.util.spec_from_file_location(
        'motor_limits_' + branch, pkg / 'utils/motor_limits.py')
    motor_limits = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(motor_limits)
    namespace = {'torch': torch, 'math': math, 'np': np,
                 'apply_actuator_limits': motor_limits.apply_actuator_limits,
                 'compute_pd_torques': motor_limits.compute_pd_torques,
                 'WHEEL_LEGGED_GYM_ROOT_DIR': str(pkg.parent),
                 'resolve_joint_mapping': helper.resolve_joint_mapping,
                 'body_frame_rotation': kinematics.body_frame_rotation,
                 'SerialLegKinematics': kinematics.SerialLegKinematics}
    # Execute just the class definitions to inspect real configs without SDK imports.
    namespace['BaseConfig'] = object
    for path in (pkg / 'envs/base/legged_robot_config.py',
                 pkg / 'envs/wheel_legged/wheel_legged_config.py',
                 pkg / 'envs/heroleg26/heroleg26_config.py'):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        exec(compile(tree, str(path), 'exec'), namespace)
    path = pkg / 'envs/base/legged_robot.py'
    tree = ast.parse(path.read_text())
    robot = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LeggedRobot')
    selected = [node for node in robot.body if isinstance(node, ast.FunctionDef)
                and node.name in METHODS]
    # The source retains an old shadowed callback definition; Python uses the last.
    selected = list({node.name: node for node in selected}.values())
    assert len(selected) == len(METHODS)
    # Also exercise the real default-angle/gain initialization loop in isolation.
    init = next(n for n in robot.body if isinstance(n, ast.FunctionDef) and n.name == '_init_buffers')
    lookup = next(n for n in init.body if isinstance(n, ast.For) and
                  ast.dump(n.iter) == ast.dump(ast.parse('range(self.num_dofs)', mode='eval').body))
    init_test = ast.parse('def _initialize_joint_parameters(self):\n    pass').body[0]
    init_test.body = [lookup]
    selected.append(init_test)
    robot.body = selected
    robot.bases = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[robot], type_ignores=[])),
                 str(path), 'exec'), namespace)
    HEROCFG[branch] = namespace['Heroleg26Cfg']
    return helper.resolve_joint_mapping, namespace['WheelLeggedCfg'], namespace['LeggedRobot']


CODE = {branch: load_code(branch) for branch in BRANCHES}


def make_env(branch, native_names, cfg=None, urdf=None):
    _, legacy_cfg, cls = CODE[branch]
    cfg = legacy_cfg if cfg is None else cfg
    env = cls()
    asset = SimpleNamespace(**{k: getattr(cfg.asset, k) for k in dir(cfg.asset)
                               if not k.startswith('_')})
    if urdf is not None:
        asset.file = str(urdf)
    env.cfg = SimpleNamespace(asset=asset, control=cfg.control, init_state=cfg.init_state,
                              env=cfg.env, rewards=cfg.rewards, terrain=cfg.terrain)
    env.dof_names = list(native_names)
    env.device = 'cpu'
    env.num_actions = env.num_dofs = env.num_dof = 6
    env.num_envs = 2
    env.num_obs = 25
    env.pi = torch.tensor([torch.pi])
    with contextlib.redirect_stdout(io.StringIO()):
        env._init_joint_mapping()
    env.default_dof_pos = torch.zeros(2, 6)
    env.raw_default_dof_pos = torch.zeros(6)
    env.p_gains = torch.zeros(2, 6)
    env.d_gains = torch.zeros(2, 6)
    env._initialize_joint_parameters()
    env.dof_pos = env.default_dof_pos.clone()
    env.dof_vel = torch.zeros(2, 6)
    env.torques_scale = torch.ones(2, 6)
    env.torque_limits = torch.full((6,), 1000.0)  # Synthetic legacy test values only.
    if getattr(cfg.asset, 'actuator_limits', None) is not None:
        source = ET.parse(urdf).getroot()
        joints = {j.get('name'): j for j in source.findall('joint')}
        props = np.zeros(6, dtype=[(name, 'f4') for name in
                                  ('lower', 'upper', 'effort', 'velocity', 'armature')])
        for i, name in enumerate(native_names):
            limit = joints[name].find('limit')
            if limit is not None:
                for field in ('lower', 'upper', 'effort', 'velocity'):
                    props[field][i] = float(limit.get(field, 0.))
        with contextlib.redirect_stdout(io.StringIO()):
            env._process_dof_props(props, 0)
    return env


def native_tensor(env, policy_tensor):
    result = torch.empty_like(policy_tensor)
    result[..., env.policy_dof_indices] = policy_tensor
    return result


class JointMappingTests(unittest.TestCase):
    def test_config_and_urdf_names(self):
        for branch in BRANCHES:
            _, cfg, _ = CODE[branch]
            self.assertEqual(cfg.asset.policy_joint_names, POLICY)
            self.assertEqual(cfg.asset.joint_name_mapping, ALIASES)
        for branch in BRANCHES:
            path = ROOT / branch / 'resources/robots/infantry_V4/urdf/infantry_V4_increase.urdf'
            names = [j.attrib['name'] for j in ET.parse(path).getroot().findall('joint')
                     if j.attrib['type'] != 'fixed']
            self.assertEqual(set(names), set(LEGACY))
        for path in ('heroleg26/urdf/4.SLDASM.urdf',
                     'heroleg26_sim/urdf/heroleg26.urdf',
                     'heroleg26_sim/urdf/heroleg26_visual.urdf'):
            names = [j.attrib['name'] for j in ET.parse(ROOT / path).getroot().findall('joint')
                     if j.attrib['type'] != 'fixed']
            self.assertEqual(set(names), set(POLICY))
            self.assertEqual(len(names), 6)

    def test_all_native_orders_and_legacy_names(self):
        for branch in BRANCHES:
            resolve, cfg, _ = CODE[branch]
            for names in (POLICY, LEGACY):
                for order in itertools.permutations(names):
                    config_names, indices = resolve(order, cfg.asset.policy_joint_names,
                                                    cfg.asset.joint_name_mapping)
                    self.assertEqual([config_names[i] for i in indices], POLICY)
                    self.assertEqual([order[i] for i in indices], names)

    def test_invalid_mapping_fails(self):
        bad_orders = [POLICY[:-1], POLICY + ['extra'], POLICY[:-1] + ['RJ1'],
                      POLICY[:-1] + ['bad_joint'], POLICY[:-1] + ['lf0_Joint']]
        for branch in BRANCHES:
            resolve, cfg, _ = CODE[branch]
            for order in bad_orders:
                with self.assertRaises(ValueError):
                    resolve(order, cfg.asset.policy_joint_names, cfg.asset.joint_name_mapping)
            with self.assertRaises(ValueError):
                resolve(POLICY, ['RJ1'] * 6, ALIASES)

    def test_each_action_drives_only_the_named_joint(self):
        orders = (POLICY, POLICY[3:] + POLICY[:3], list(reversed(POLICY)), LEGACY)
        for branch in BRANCHES:
            for order in orders:
                env = make_env(branch, order)
                expected_pose = torch.tensor([.2, .4, 0, -.2, -.4, 0])
                torch.testing.assert_close(env.raw_default_dof_pos[env.policy_dof_indices], expected_pose)
                for channel, name in enumerate(POLICY):
                    actions = torch.zeros(2, 6)
                    actions[:, channel] = 1.0
                    actual = env._compute_torques(actions)
                    index = env.policy_dof_indices[channel].item()
                    expected = torch.zeros(2, 6)
                    gain = (env.cfg.control.damping[name] * env.cfg.control.vel_action_scale
                            if channel in (2, 5) else
                            env.cfg.control.stiffness[name] * env.cfg.control.pos_action_scale)
                    expected[:, index] = gain
                    torch.testing.assert_close(actual, expected)

    def test_control_clipping_and_original_equations(self):
        for branch in BRANCHES:
            reference = make_env(branch, LEGACY)
            # Distinct values expose both incorrect state gathers and limit reordering.
            q = torch.tensor([[.1, .3, .7, -.4, -.2, -.9], [.4, -.2, .5, .3, .8, -.1]])
            v = q * 2.3
            actions = q * 4
            reference.dof_pos, reference.dof_vel = q, v
            reference.torque_limits = torch.tensor([.3, .5, .7, .9, 1.1, 1.3])
            reference.torques_scale = q.abs() + .5
            pos_ref = actions * reference.cfg.control.pos_action_scale
            pos_ref[:, [2, 5]] = 0
            vel_ref = actions * reference.cfg.control.vel_action_scale
            vel_ref[:, [0, 1, 3, 4]] = 0
            expected = reference.p_gains * (pos_ref + reference.default_dof_pos - q)
            expected += reference.d_gains * (vel_ref - v)
            expected = torch.clip(expected * reference.torques_scale,
                                  -reference.torque_limits, reference.torque_limits)
            for order in (POLICY, POLICY[3:] + POLICY[:3], list(reversed(POLICY)), LEGACY):
                env = make_env(branch, order)
                env.dof_pos, env.dof_vel = native_tensor(env, q), native_tensor(env, v)
                env.torque_limits = native_tensor(env, reference.torque_limits)
                env.torques_scale = native_tensor(env, reference.torques_scale)
                result = env._compute_torques(actions)[:, env.policy_dof_indices]
                torch.testing.assert_close(result, expected)

    def test_observations_kinematics_and_rewards_follow_policy_order(self):
        for branch in BRANCHES:
            reference = None
            for order in (POLICY, POLICY[3:] + POLICY[:3], list(reversed(POLICY)), LEGACY):
                env = make_env(branch, order)
                q = torch.tensor([[.1, -.2, 9., -.3, .4, -8.], [.5, .6, 7., .7, -.8, -6.]])
                env.dof_pos = native_tensor(env, q)
                env.dof_vel = native_tensor(env, q * 2)
                env.dof_acc = native_tensor(env, q * 3)
                env.torques = native_tensor(env, q * 4)
                env.default_dof_pos += native_tensor(env, q * .01)
                env.dof_pos_limits = native_tensor(env, torch.tensor(
                    [[-.2, -.3, -1., -.1, -.4, -1.], [.3, .2, 1., .5, .1, 1.]])).T
                env.actions = q * .1
                env.base_ang_vel = torch.ones(2, 3)
                env.projected_gravity = -torch.ones(2, 3)
                env.commands = torch.ones(2, 3)
                env.commands_scale = torch.ones(3)
                env.base_lin_vel = torch.ones(2, 3) * 2
                env.obs_scales = SimpleNamespace(ang_vel=.25, dof_pos=1., dof_vel=.05,
                                                lin_vel=2., dof_acc=.0025, torque=.05,
                                                height_measurements=5.)
                env.root_states = torch.zeros(2, 13)
                env.measured_heights = torch.zeros(2, 77)
                env.last_actions = torch.stack((q * .2, q * .3), dim=-1)
                env.base_mass = torch.tensor([13., 14.])
                env.base_com = torch.zeros(2, 3)
                env.friction_coef = torch.ones(2)
                env.restitution_coef = torch.ones(2) * .5
                env.add_noise = False
                env.envs_steps_buf = torch.zeros(2)
                env.obs_history = torch.zeros(2, 125)
                env.compute_observations()
                theta1, theta2 = env._get_leg_joint_angles()
                torch.testing.assert_close(theta1, torch.stack((q[:, 0], -q[:, 3]), dim=1))
                torch.testing.assert_close(theta2, torch.stack((q[:, 1] + torch.pi/2,
                                                               -q[:, 4] + torch.pi/2), dim=1))
                self.assertEqual(tuple(env.obs_buf.shape), (2, 25))
                self.assertEqual(tuple(env.privileged_obs_buf.shape), (2, 141))
                torch.testing.assert_close(env.obs_buf[:, 13:19], q * 2 * .05)
                torch.testing.assert_close(env.obs_buf[:, 19:25], env.actions)
                torch.testing.assert_close(env._reward_dof_vel(),
                                           (q[:, [0, 1, 3, 4]] * 2).square().sum(dim=1))
                result = (env.obs_buf, env.privileged_obs_buf, env.obs_history,
                          env._reward_dof_pos_limits())
                if reference is None:
                    reference = result
                else:
                    for actual, expected in zip(result, reference):
                        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
