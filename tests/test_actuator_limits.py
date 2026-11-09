"""Exercise SDK-property overrides and the actual environment torque controller."""
import copy
import math
import unittest

import numpy as np
import torch

from test_joint_mapping import BRANCHES, CODE, HEROCFG, POLICY, ROOT, make_env

SOURCE = ROOT / 'heroleg26_sim/urdf/heroleg26.urdf'


class ActuatorLimitsTests(unittest.TestCase):
    def test_named_properties_and_no_partial_invalid_update(self):
        for branch in BRANCHES:
            cls = CODE[branch][2]
            apply = cls._process_dof_props.__globals__['apply_actuator_limits']
            cfg = HEROCFG[branch]
            for order in (POLICY, POLICY[3:] + POLICY[:3], list(reversed(POLICY))):
                env = make_env(branch, order, cfg, SOURCE)
                expected_torque = torch.tensor([20., 20., 3 * (268/17) / (3591/187),
                                                20., 20., 3 * (268/17) / (3591/187)])
                expected_speed = torch.tensor([100, 100, 469 * (3591/187) / (268/17),
                                               100, 100, 469 * (3591/187) / (268/17)]) * math.pi / 30
                torch.testing.assert_close(env.torque_limits[env.policy_dof_indices], expected_torque)
                torch.testing.assert_close(env.dof_vel_limits[env.policy_dof_indices], expected_speed)
                knees = env.position_limited_dof_indices
                expected = torch.tensor([[-.41538836197465, .872664625997165]] * 2)
                torch.testing.assert_close(env.dof_position_target_limits[knees], expected)
            props = np.ones(6, dtype=[('effort', 'f4'), ('velocity', 'f4')])
            original = props.copy()
            apply(props, POLICY, None)
            np.testing.assert_array_equal(props, original)
            for invalid in ('missing_joint', 'negative', 'nan'):
                profile = copy.deepcopy(cfg.asset.actuator_limits)
                if invalid == 'missing_joint':
                    profile.pop('LJ3')
                else:
                    profile['LJ3']['effort'] = -1 if invalid == 'negative' else float('nan')
                with self.assertRaises(ValueError):
                    apply(props, POLICY, profile)
                np.testing.assert_array_equal(props, original)

    def test_torque_caps_and_knee_target_clamping(self):
        for branch in BRANCHES:
            for order in (POLICY, list(reversed(POLICY))):
                env = make_env(branch, order, HEROCFG[branch], SOURCE)
                knees = env.position_limited_dof_indices
                # Huge positive/negative actions must not request knee targets
                # beyond raw physical bounds, even before torque saturation.
                env.dof_pos[:, knees] = env.dof_position_target_limits[knees, 1]
                output = env._compute_torques(torch.full((2, 6), 1000.))
                torch.testing.assert_close(output[:, knees], torch.zeros(2, 2))
                self.assertTrue(torch.all(output.abs() <= env.torque_limits + 1e-6))
                env.dof_pos[:, knees] = env.dof_position_target_limits[knees, 0]
                output = env._compute_torques(torch.full((2, 6), -1000.))
                torch.testing.assert_close(output[:, knees], torch.zeros(2, 2))
                self.assertTrue(torch.all(output.abs() <= env.torque_limits + 1e-6))

    def test_overspeed_blocks_drive_but_preserves_braking_both_signs(self):
        for branch in BRANCHES:
            for order in (POLICY, POLICY[3:] + POLICY[:3]):
                env = make_env(branch, order, HEROCFG[branch], SOURCE)
                for sign in (-1., 1.):
                    env.dof_vel[:] = sign * env.dof_vel_limits * 1.1
                    for channel in (0, 1, 3, 4):
                        native = env.policy_dof_indices[channel]
                        env.dof_pos[:] = env.default_dof_pos
                        # Force the PD request to accelerate this leg joint.
                        env.dof_pos[:, native] -= sign * 100.
                        torque = env._compute_torques(torch.zeros(2, 6))
                        torch.testing.assert_close(torque[:, native], torch.zeros(2))
                    env.dof_pos[:] = env.default_dof_pos
                    torque = env._compute_torques(torch.zeros(2, 6))
                    self.assertTrue(torch.all(torque * env.dof_vel < 0.))
                    self.assertTrue(torch.all(torque.abs() <= env.torque_limits + 1e-6))
                    # Huge wheel actions must first be clamped to the configured
                    # velocity target, so they also brake above that target.
                    torque = env._compute_torques(torch.full((2, 6), sign * 1e6))
                    wheels = env.wheel_dof_indices
                    self.assertTrue(torch.all(torque[:, wheels] * env.dof_vel[:, wheels] < 0.))


if __name__ == '__main__':
    unittest.main()
