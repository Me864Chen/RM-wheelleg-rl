"""Check the current CAD export, confirmed standing pose and URDF kinematics."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch

from test_joint_mapping import ROOT, BRANCHES, POLICY, LEGACY, HEROCFG, make_env

SOURCE = ROOT / 'heroleg26_sim/urdf/heroleg26.urdf'


class HerolegCalibrationTests(unittest.TestCase):
    def test_confirmed_standing_pose_and_asset_binding(self):
        expected = torch.tensor([math.radians(-25), math.radians(20), 0.,
                                 math.radians(-25), math.radians(20), 0.])
        for branch in BRANCHES:
            cfg = HEROCFG[branch]
            self.assertEqual(cfg.asset.body_forward_axis, [1., 0., 0.])
            self.assertEqual(cfg.asset.leg_kinematics, 'urdf_serial')
            self.assertTrue(cfg.asset.file.endswith('generated/heroleg26_obj.urdf'))
            self.assertEqual(cfg.asset.foot_name, '_Link3')
            self.assertFalse(cfg.asset.recompute_inertia)
            self.assertFalse(cfg.domain_rand.randomize_default_dof_pos)
            self.assertFalse(cfg.domain_rand.randomize_base_mass)
            self.assertFalse(cfg.domain_rand.randomize_base_com)
            self.assertFalse(cfg.domain_rand.randomize_inertia)
            self.assertEqual(cfg.init_state.pos, [0., 0., .334])
            for order in (POLICY, POLICY[3:]+POLICY[:3], list(reversed(POLICY))):
                env = make_env(branch, order, cfg, SOURCE)
                torch.testing.assert_close(env.raw_default_dof_pos[env.policy_dof_indices], expected)
                # Zero policy action holds the exact confirmed native joint pose.
                torch.testing.assert_close(env._compute_torques(torch.zeros(2, 6)), torch.zeros(2, 6))
                env._update_leg_kinematics()
                self.assertTrue(torch.all((env.L0 > .265) & (env.L0 < .267)))
                torch.testing.assert_close(env.L0[:, 0], env.L0[:, 1], atol=1e-6, rtol=0.)
                # CAD yaw -1.5708 (rather than -pi/2) projects the opposite
                # lateral offsets slightly differently: about 1.7e-6 rad here.
                torch.testing.assert_close(env.theta0[:, 0], env.theta0[:, 1], atol=2e-6, rtol=0.)

    def test_forward_left_up_and_inverse_conversion(self):
        for branch in BRANCHES:
            env = make_env(branch, POLICY, HEROCFG[branch], SOURCE)
            # New native frame is already forward/left/up.
            native = torch.eye(3)
            torch.testing.assert_close(env._body_to_policy(native), torch.eye(3))
            sample = torch.arange(48, dtype=torch.float).reshape(2, 8, 3)
            torch.testing.assert_close(env._policy_to_body(env._body_to_policy(sample)), sample)
            torch.testing.assert_close(torch.linalg.det(env.body_to_policy_rotation), torch.tensor(1.))
            # New initial orientation preserves the native +X nose.
            x, y, z, w = HEROCFG[branch].init_state.rot
            rotated_nose = torch.tensor([1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)])
            torch.testing.assert_close(rotated_nose, torch.tensor([1., 0., 0.]), atol=1e-6, rtol=0.)
            # Terrain sampling is forward/left in policy, converted to native body.
            points = env._init_height_points()
            self.assertEqual(points.shape[-1], 3)
            first = torch.tensor([env.cfg.terrain.measured_points_x[0],
                                  env.cfg.terrain.measured_points_y[0], 0.])
            torch.testing.assert_close(env._body_to_policy(points[0, 0]), first)

    def test_positive_speed_command_rewards_native_positive_x(self):
        for branch in BRANCHES:
            env = make_env(branch, POLICY, HEROCFG[branch], SOURCE)
            env.commands = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
            native_velocity = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
            env.base_lin_vel = env._body_to_policy(native_velocity)
            reward = env._reward_tracking_lin_vel()
            self.assertGreater(float(reward[0]), .9)  # jump has an inherited 2x scale.
            self.assertLess(float(reward[1]), .01)
            # Exercise the same nested-class instantiation used by task registration.
            path = ROOT / branch / 'wheel_legged_gym/envs/base/base_config.py'
            spec = importlib.util.spec_from_file_location('base_config_' + branch, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cfg = HEROCFG[branch]()
            module.BaseConfig.init_member_classes(cfg)
            self.assertEqual(cfg.init_state.default_joint_angles['RJ2'], math.radians(20))
            self.assertEqual(cfg.asset.body_forward_axis, [1., 0., 0.])
            self.assertFalse(cfg.domain_rand.randomize_default_dof_pos)

    def test_correct_geometry_and_joint_direction(self):
        for branch in BRANCHES:
            env = make_env(branch, POLICY, HEROCFG[branch], SOURCE)
            k = env.leg_kinematics
            torch.testing.assert_close(k.lengths, torch.tensor([[.210, .250], [.210, .250]]))
            torch.testing.assert_close(k.joint_signs, torch.tensor([[-1., 1.], [-1., 1.]]))
            zero = torch.zeros(1, 2)
            v0 = k.hip_to_wheel_body(zero, zero)
            torch.testing.assert_close(v0[0], k.vectors.sum(dim=1), atol=1e-7, rtol=1e-6)
            hip = torch.tensor([[math.pi/36, math.pi/36]])
            v_hip = k.hip_to_wheel_body(hip, zero)
            self.assertTrue(torch.all(v_hip[0, :, 0] < v0[0, :, 0]))
            v_knee = k.hip_to_wheel_body(zero, hip)
            old_length = torch.linalg.vector_norm(v0[..., [0, 2]], dim=-1)
            new_length = torch.linalg.vector_norm(v_knee[..., [0, 2]], dim=-1)
            self.assertTrue(torch.all(new_length < old_length))

    def test_only_knees_have_position_limit_penalty(self):
        for branch in BRANCHES:
            env = make_env(branch, POLICY[3:]+POLICY[:3], HEROCFG[branch], SOURCE)
            limited = {env.dof_names[i] for i in env.position_limited_dof_indices.tolist()}
            self.assertEqual(limited, {'RJ2', 'LJ2'})
            env.dof_pos_limits = torch.zeros(6, 2)
            env.dof_pos_limits[env.position_limited_dof_indices, 0] = -.41538836197465
            env.dof_pos_limits[env.position_limited_dof_indices, 1] = .872664625997165
            env.dof_pos[:, env.policy_dof_indices[[0, 2, 3, 5]]] = 100.
            torch.testing.assert_close(env._reward_dof_pos_limits(), torch.zeros(2))
            env.dof_pos[:, env.position_limited_dof_indices[0]] = 1.
            self.assertTrue(torch.all(env._reward_dof_pos_limits() > 0.))

    def test_calibrated_kinematics_cannot_use_legacy_asset(self):
        for branch in BRANCHES:
            with self.assertRaisesRegex(ValueError, 'legacy joint aliases'):
                make_env(branch, LEGACY, HEROCFG[branch], SOURCE)

    @unittest.skipUnless(importlib.util.find_spec('mujoco'), 'Optional local MuJoCo cross-check')
    def test_urdf_forward_kinematics_against_mujoco(self):
        import numpy as np
        import mujoco
        spec = importlib.util.spec_from_file_location(
            'inspect_calibration', ROOT / 'heroleg26_sim/tools/inspect_model.py')
        inspect = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inspect)
        model, resources = inspect.compile_model(ROOT / 'heroleg26_sim/urdf/heroleg26_visual.urdf')
        data = mujoco.MjData(model)
        before = inspect.fingerprints(resources)
        ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in POLICY}
        samples = np.random.default_rng(26).uniform(-1., 1., (100, 4))
        # Sample revolute knees only inside the new source limits.
        samples[:, [1, 3]] = (samples[:, [1, 3]] + 1.) / 2. * (0.872664625997165 + 0.41538836197465) - 0.41538836197465
        samples[0] = [math.radians(-25), math.radians(20), math.radians(-25), math.radians(20)]
        for branch in BRANCHES:
            env = make_env(branch, POLICY[3:]+POLICY[:3], HEROCFG[branch], SOURCE)
            max_error = 0.
            for sample in samples:
                data.qpos[:] = model.qpos0
                for name, value in zip(('RJ1', 'RJ2', 'LJ1', 'LJ2'), sample):
                    data.qpos[model.jnt_qposadr[ids[name]]] = value
                mujoco.mj_forward(model, data)
                hip = torch.tensor(sample[[0, 2]], dtype=torch.float).unsqueeze(0)
                knee = torch.tensor(sample[[1, 3]], dtype=torch.float).unsqueeze(0)
                calculated = env.leg_kinematics.hip_to_wheel_body(hip, knee)[0].numpy()
                actual = np.stack([data.xanchor[ids[p+'3']] - data.xanchor[ids[p+'1']]
                                   for p in ('RJ', 'LJ')])
                max_error = max(max_error, float(np.max(np.abs(calculated-actual))))
                np.testing.assert_allclose(calculated, actual, atol=2e-7, rtol=1e-5)
                # Exercise the actual env update with a shuffled SDK DOF enumeration.
                env.dof_pos[:, env.policy_dof_indices] = torch.tensor(
                    [sample[0], sample[1], 0., sample[2], sample[3], 0.], dtype=torch.float)
                env._update_leg_kinematics()
                np.testing.assert_allclose(env.L0[0].numpy(), np.linalg.norm(actual[:, [0, 2]], axis=1),
                                           atol=2e-7, rtol=1e-5)
                np.testing.assert_allclose(env.theta0[0].numpy(), np.arctan2(-actual[:, 0], -actual[:, 2]),
                                           atol=2e-6, rtol=1e-5)
            print('\n' + branch + ' calibrated FK maximum error (m): ' + str(max_error))
        # Place the confirmed standing pose in its configured world pose; inspect collision mesh heights.
        cfg = HEROCFG['plane']
        data.qpos[:] = model.qpos0
        free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, '__inspection_free__')
        adr = model.jnt_qposadr[free]
        data.qpos[adr:adr+3] = cfg.init_state.pos
        x, y, z, w = cfg.init_state.rot
        data.qpos[adr+3:adr+7] = [w, x, y, z]
        for name in POLICY:
            data.qpos[model.jnt_qposadr[ids[name]]] = cfg.init_state.default_joint_angles[name]
        mujoco.mj_forward(model, data)
        minimum = float('inf')
        for geom in range(model.ngeom):
            if model.geom_group[geom] != 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start+count]
            height = vertices @ data.geom_xmat[geom].reshape(3, 3)[2] + data.geom_xpos[geom, 2]
            minimum = min(minimum, float(height.min()))
        print('Confirmed standing-pose minimum collision-mesh clearance (m): ' + str(minimum))
        self.assertGreater(minimum, 0.)
        self.assertLess(minimum, .03)
        self.assertEqual(inspect.fingerprints(resources), before)


if __name__ == '__main__':
    unittest.main()
