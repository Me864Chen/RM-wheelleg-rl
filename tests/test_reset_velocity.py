"""SDK reset payload must clear native joint velocities in both feedback modes."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def methods(branch):
    path = ROOT/branch/'wheel_legged_gym/envs/base/legged_robot.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LeggedRobot')
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('compute_dof_vel', '_reset_dofs')]
    namespace = {'torch': torch, 'gymtorch': SimpleNamespace(unwrap_tensor=lambda t: t)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


class ResetVelocityTests(unittest.TestCase):
    def test_actual_reset_clears_sdk_velocity_and_only_selected_actors(self):
        for branch in ('plane', 'jump'):
            for difference_feedback in (False, True):
                with self.subTest(branch=branch, difference_feedback=difference_feedback):
                    code = methods(branch)
                    env = SimpleNamespace(num_envs=3, num_dof=6, sim='sim',
                                          sim_params=SimpleNamespace(dt=.001), pi=torch.tensor(torch.pi),
                                          cfg=SimpleNamespace(env=SimpleNamespace(
                                              dof_vel_use_pos_diff=difference_feedback)))
                    env.dof_state = torch.zeros(3, 6, 2)
                    env.dof_state[:, :, 0] = .2
                    env.dof_state[:, :, 1] = 3.
                    env.dof_pos = env.dof_state[:, :, 0]
                    env.dof_vel = env.dof_state[:, :, 1]
                    env.last_dof_pos = torch.full((3, 6), .19)
                    env.default_dof_pos = torch.full((3, 6), -.1)
                    code['compute_dof_vel'](env)
                    self.assertEqual(env.dof_vel.data_ptr() == env.dof_state[:, :, 1].data_ptr(),
                                     not difference_feedback)
                    payloads = []
                    env.gym = SimpleNamespace(set_dof_state_tensor_indexed=lambda sim, state, ids, n:
                                              payloads.append((state.clone(), ids.clone(), n)))
                    code['_reset_dofs'](env, torch.tensor([0, 2]))
                    state, ids, count = payloads[-1]
                    self.assertEqual(ids.tolist(), [0, 2])
                    self.assertEqual(count, 2)
                    torch.testing.assert_close(state[[0, 2], :, 1], torch.zeros(2, 6))
                    torch.testing.assert_close(state[[0, 2], :, 0], torch.full((2, 6), -.1))
                    torch.testing.assert_close(state[1, :, 1], torch.full((6,), 3.))
                    torch.testing.assert_close(state[1, :, 0], torch.full((6,), .2))
                    torch.testing.assert_close(env.dof_vel[[0, 2]], torch.zeros(2, 6))


if __name__ == '__main__':
    unittest.main()
