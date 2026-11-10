"""Validate the paired deterministic evaluation protocol without the native SDK."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'heroleg26_isaac_check'
sys.path.insert(0, str(FOLDER))
try:
    spec = importlib.util.spec_from_file_location('balance_eval', FOLDER/'evaluate_balance.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.remove(str(FOLDER))


class BalanceEvaluationTests(unittest.TestCase):
    def test_config_restoration_preserves_nested_objects_and_methods(self):
        method = lambda obj: None
        cfg = SimpleNamespace(asset=SimpleNamespace(limits={'wheel': {'effort': 2.}}),
                              env=SimpleNamespace(num_envs=64), init_member_classes=method)
        values = {'asset': {'limits': {'wheel': {'effort': 3.}}},
                  'env': {'num_envs': 1}, 'init_member_classes': {}, '_private': 4}
        entry.restore_config(cfg, values)
        self.assertEqual(cfg.env.num_envs, 1)
        self.assertEqual(cfg.asset.limits['wheel']['effort'], 3.)
        self.assertIs(cfg.init_member_classes, method)
        values['asset']['limits']['wheel']['effort'] = 10.
        self.assertEqual(cfg.asset.limits['wheel']['effort'], 3.)

    def test_initial_velocities_are_matched_and_seeded(self):
        cases = entry.initial_cases(8, 2026)
        self.assertEqual(cases, entry.initial_cases(8, 2026))
        self.assertNotEqual(cases, entry.initial_cases(8, 2027))
        self.assertEqual(len(cases), 9)
        self.assertEqual(cases[0][2], [0.]*6)
        self.assertTrue(all(-.5 <= v <= .5 for _, _, velocity in cases for v in velocity))

    def test_success_requires_wheels_only_and_full_duration(self):
        episode = {'end_reason': 'time_limit', 'non_wheel_loaded_samples': 0,
                   'both_wheels_loaded_fraction': .99,
                   'maximum_tilt_deg': 5., 'maximum_xy_displacement_m': .1}
        self.assertTrue(entry.clean_balance(episode))
        for name, value in [('end_reason', 'base_contact'), ('non_wheel_loaded_samples', 1),
                            ('both_wheels_loaded_fraction', .9), ('maximum_tilt_deg', 15.),
                            ('maximum_xy_displacement_m', .4)]:
            invalid = dict(episode, **{name: value})
            self.assertFalse(entry.clean_balance(invalid))

    def test_compatibility_rejects_unrelated_code_changes(self):
        original = 'class LeggedRobot:\n    def _reset_dofs(self, env_ids):\n        self.dof_vel[env_ids] = 0.0\n'
        fixed = original.replace('        self.dof_vel', '        '+entry.RESET_FIX+'\n        self.dof_vel')
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory)/'old.py', Path(directory)/'new.py'
            old.write_text(original)
            new.write_text(fixed)
            self.assertIn('Only', entry.check_reset_fix(new, old, entry.sha256(old)))
            new.write_text(fixed+'\nchanged = True\n')
            with self.assertRaises(RuntimeError):
                entry.check_reset_fix(new, old, entry.sha256(old))
            with self.assertRaises(RuntimeError):
                entry.check_reset_fix(old, old, entry.sha256(old))
        ast.parse((FOLDER/'evaluate_balance.py').read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
