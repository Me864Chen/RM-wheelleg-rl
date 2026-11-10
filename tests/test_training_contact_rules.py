"""Exercise actual termination/reward methods with realistic contact histories."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

from test_joint_mapping import BRANCHES, CODE, HEROCFG, ROOT

FOLDER = ROOT / 'heroleg26_isaac_check'
sys.path.insert(0, str(FOLDER))
try:
    spec = importlib.util.spec_from_file_location('training_env_check', FOLDER/'check_training_env.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.remove(str(FOLDER))


def termination_env(branch, cfg=None):
    path = ROOT / branch / 'wheel_legged_gym/envs/base/legged_robot.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LeggedRobot')
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ('check_termination', '_reward_collision')]
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), 'exec'), namespace)
    env = namespace['LeggedRobot']()
    env.cfg = cfg or HEROCFG[branch]
    env.num_envs, env.num_bodies = 2, 7
    env.dt = .01
    env.contact_forces = torch.zeros(2, 7, 3)
    indices = [0] if env.cfg.asset.terminate_after_contacts_on else []
    env.termination_contact_indices = torch.tensor(indices, dtype=torch.long)
    indices = [0] if env.cfg.asset.penalize_contacts_on else []
    env.penalised_contact_indices = torch.tensor(indices, dtype=torch.long)
    env.projected_gravity = torch.tensor([[0., 0., -1.], [0., 0., -1.]])
    env.fail_buf = torch.zeros(2, dtype=torch.long)
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.max_episode_length = 200
    env.edge_reset_buf = torch.zeros(2, dtype=torch.bool)
    return env


class TrainingContactRulesTests(unittest.TestCase):
    def test_only_own_task_enables_base_contact_rules(self):
        for branch in BRANCHES:
            cfg = HEROCFG[branch]
            original = CODE[branch][1]
            self.assertEqual(cfg.asset.penalize_contacts_on, ['base_link'])
            self.assertEqual(cfg.asset.terminate_after_contacts_on, ['base_link'])
            self.assertEqual(cfg.env.fail_to_terminal_time_s, .05)
            self.assertEqual(original.asset.penalize_contacts_on,
                             [] if branch == 'plane' else ['lf', 'rf', 'base'])
            self.assertEqual(original.asset.terminate_after_contacts_on, [])
            self.assertEqual(original.env.fail_to_terminal_time_s, 1.)
            self.assertEqual(entry.training_rules(ROOT, branch)['env']['fail_to_terminal_time_s'], .05)

    def test_wheel_load_is_allowed_and_base_load_causes_debounced_reset(self):
        for branch in BRANCHES:
            env = termination_env(branch)
            env.contact_forces[:, [3, 6], 2] = 65.
            for _ in range(20):
                env.check_termination()
                self.assertFalse(env.reset_buf.any())
                self.assertEqual(env._reward_collision().tolist(), [0., 0.])
            env.contact_forces[0, 0, 2] = 29.18  # Measured parked-base mean.
            for _ in range(5):
                env.check_termination()
                self.assertFalse(env.reset_buf.any())
            env.check_termination()
            self.assertEqual(env.reset_buf.tolist(), [True, False])
            self.assertEqual(env._reward_collision().tolist(), [1., 0.])
            # The same history on the original task does not activate this rule.
            legacy = termination_env(branch, CODE[branch][1])
            legacy.contact_forces.copy_(env.contact_forces)
            for _ in range(20):
                legacy.check_termination()
            self.assertFalse(legacy.reset_buf.any())

    def test_contact_impulses_clear_and_horizontal_contact_is_detected(self):
        for branch in BRANCHES:
            env = termination_env(branch)
            env.contact_forces[0, 0, 0] = 1000.
            env.check_termination()
            self.assertFalse(env.reset_buf.any())
            env.contact_forces.zero_()
            env.check_termination()
            self.assertEqual(env.fail_buf.tolist(), [0, 0])
            env.contact_forces[0, 0, 0] = 11.
            for _ in range(6):
                env.check_termination()
            self.assertEqual(env.reset_buf.tolist(), [True, False])

    def test_snapshots_preserve_terminal_state_after_task_reset(self):
        env = termination_env('plane')
        env.root_states = torch.zeros(2, 13)
        env.root_states[:, 2] = .217
        for name in ('dof_pos', 'dof_vel', 'torques'):
            setattr(env, name, torch.zeros(2, 6))
        captured = entry.observe_termination(env)
        env.contact_forces[0, 0, 2] = 29.
        for _ in range(6):
            env.check_termination()
        self.assertEqual(captured['reset_buf'].tolist(), [True, False])
        env.root_states[:, 2] = .265
        env.fail_buf.zero_()
        env.contact_forces.zero_()
        self.assertAlmostEqual(captured['root_states'][0, 2].item(), .217, places=6)
        self.assertEqual(captured['fail_buf'].tolist(), [6, 0])
        self.assertEqual(captured['contact_forces'][0, 0, 2].item(), 29.)

    def test_runtime_gain_override_preserves_limits_stance_and_shared_dicts(self):
        cfg = SimpleNamespace(
            env=SimpleNamespace(), terrain=SimpleNamespace(), noise=SimpleNamespace(),
            commands=SimpleNamespace(ranges=SimpleNamespace()),
            domain_rand=SimpleNamespace(randomize_Kp=True, randomize_base_mass=False),
            control=SimpleNamespace(stiffness=dict(HEROCFG['plane'].control.stiffness),
                                    damping=dict(HEROCFG['plane'].control.damping)),
            init_state=copy.deepcopy(HEROCFG['plane'].init_state.default_joint_angles),
            limits=copy.deepcopy(HEROCFG['plane'].asset.actuator_limits))
        original_kp, original_kd = cfg.control.stiffness, cfg.control.damping
        pose, limits = copy.deepcopy(cfg.init_state), copy.deepcopy(cfg.limits)
        entry.configure_check(cfg, 4, 2., 1.5)
        self.assertEqual(cfg.control.stiffness['RJ2'], 40.)
        self.assertEqual(cfg.control.damping['RJ2'], 1.5)
        self.assertEqual(cfg.control.damping['RJ3'], .2)
        self.assertEqual(original_kp['RJ2'], 20.)
        self.assertEqual(original_kd['RJ2'], 1.)
        self.assertEqual(cfg.init_state, pose)
        self.assertEqual(cfg.limits, limits)
        self.assertFalse(cfg.domain_rand.randomize_Kp)

    def test_body_order_uses_actual_handles_and_rejects_missing_links(self):
        mapping = {'base_link': 1, 'wheel': 0}
        env = SimpleNamespace(num_bodies=2, envs=['env'], actor_handles=[0],
                              gym=SimpleNamespace(find_actor_rigid_body_handle=lambda e, a, n: mapping[n]))
        self.assertEqual(entry.body_names_by_handle(env, ['base_link', 'wheel']), ['wheel', 'base_link'])
        with self.assertRaises(RuntimeError):
            entry.body_names_by_handle(env, ['wheel'])
        with self.assertRaises(RuntimeError):
            entry.body_names_by_handle(env, ['wheel', 'wheel'])
        ast.parse((FOLDER/'check_training_env.py').read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
