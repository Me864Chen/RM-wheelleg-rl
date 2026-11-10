"""Regression checks for free-base/contact diagnostics without a local SDK."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from test_joint_mapping import ROOT

folder = ROOT/'heroleg26_isaac_check'
sys.path.insert(0, str(folder))
try:
    spec = importlib.util.spec_from_file_location('contact_entry_check', folder/'check_contact.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.remove(str(folder))


def valid_metrics():
    return {'completed': True, 'ground_contact_seen': True, 'loaded_ground_contact_seen': True,
            'wheel_ground_contact_seen': ['L_Link3', 'R_Link3'],
            'wheel_loaded_contact_seen': ['L_Link3', 'R_Link3'],
            'minimum_base_height_m': .08, 'maximum_base_speed_m_s': 2.,
            'maximum_joint_speed_ratio': .8, 'maximum_knee_bound_violation_rad': .01,
            'self_contact_count': 0, 'maximum_initial_overlap_m': .005,
            'fall_observed': True}


class ContactEntryTests(unittest.TestCase):
    def test_ordinary_fall_is_not_contact_test_failure(self):
        metrics = valid_metrics()
        self.assertEqual(entry.classify_result(metrics), ('PASS', []))

    def test_missing_contact_and_numerical_issues_require_review(self):
        bad = {'completed': False, 'ground_contact_seen': False,
               'loaded_ground_contact_seen': False, 'wheel_ground_contact_seen': [],
               'wheel_loaded_contact_seen': [], 'minimum_base_height_m': -.2,
               'maximum_base_speed_m_s': 30., 'maximum_joint_speed_ratio': 5.,
               'maximum_knee_bound_violation_rad': .1, 'self_contact_count': 1,
               'maximum_initial_overlap_m': .1}
        for key, value in bad.items():
            metrics = valid_metrics()
            metrics[key] = value
            status, reasons = entry.classify_result(metrics)
            self.assertEqual(status, 'NEEDS_REVIEW', key)
            self.assertTrue(reasons, key)
        metrics = valid_metrics()
        metrics['minimum_base_height_m'] = None
        self.assertEqual(entry.classify_result(metrics)[0], 'NEEDS_REVIEW')

    def test_tilt_uses_world_up_and_ignores_yaw(self):
        self.assertAlmostEqual(entry.tilt_degrees([0, 0, 0, 1]), 0)
        self.assertAlmostEqual(entry.tilt_degrees([0, 0, math.sin(.8), math.cos(.8)]), 0)
        self.assertAlmostEqual(entry.tilt_degrees([math.sin(math.pi/4), 0, 0, math.cos(math.pi/4)]), 90)
        with self.assertRaises(ValueError):
            entry.tilt_degrees([0, 0, 0, 0])

    def test_classic_vec3_formats_and_invalid_force_data(self):
        structured = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        structured['z'] = [5, 10]
        expected = np.array([[0, 0, 5], [0, 0, 10]])
        np.testing.assert_equal(entry.vectors(structured), expected)
        np.testing.assert_equal(entry.vectors(expected), expected)
        objs = [SimpleNamespace(x=0, y=0, z=5), SimpleNamespace(x=0, y=0, z=10)]
        np.testing.assert_equal(entry.vectors(objs), expected)
        with self.assertRaises(ValueError):
            entry.vectors(np.full((2, 3), np.nan))

    def test_contacts_use_named_fields_and_handle_ground_on_either_side(self):
        contacts = np.zeros(3, dtype=[('env0', 'i4'), ('env1', 'i4'),
                                     ('body0', 'i4'), ('body1', 'i4'), ('initial_overlap', 'f4')])
        contacts['body0'] = [-1, 1, 0]
        contacts['body1'] = [0, -1, 1]
        contacts['initial_overlap'] = [.001, .002, .003]
        result = entry.contact_summary(contacts, ['L_Link3', 'R_Link3'])
        self.assertEqual(result['ground_links'], ['L_Link3', 'R_Link3'])
        self.assertEqual(result['self_pairs'], [[0, 1]])
        self.assertAlmostEqual(result['maximum_initial_overlap_m'], .003)
        self.assertFalse(entry.contact_summary(np.array([[0, 0, 1, -1]]), ['base'])['parsed'])

    def test_real_setup_releases_base_and_keeps_self_collision_filtered(self):
        tree = ast.parse((folder/'check_contact.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_check')
        assignment = next(n for n in ast.walk(run) if isinstance(n, ast.Assign)
                          and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                          and n.value.func.attr == 'create_actor')
        option_assignment = next(n for n in ast.walk(run) if isinstance(n, ast.Assign)
                                 and isinstance(n.targets[0], ast.Attribute)
                                 and n.targets[0].attr == 'fix_base_link')
        called = []
        options = SimpleNamespace(fix_base_link=True)
        gym = SimpleNamespace(create_actor=lambda *args: called.append(args) or 0)
        code = compile(ast.fix_missing_locations(ast.Module(body=[option_assignment, assignment],
                                                           type_ignores=[])), '<actual-contact-setup>', 'exec')
        exec(code, {'gym': gym, 'options': options, 'env': 'env', 'asset': 'asset', 'pose': 'pose'})
        self.assertFalse(options.fix_base_link)
        self.assertEqual(called[0][-1], 1)

    def test_setup_requires_validated_time_and_keeps_source_settings(self):
        cfg = entry.load_config(ROOT, 'plane')
        before = copy.deepcopy(cfg)
        entry.validate_setup(cfg, 5., 0.)
        self.assertEqual(cfg, before)
        cfg['sim']['dt'] = .005
        with self.assertRaises(ValueError):
            entry.validate_setup(cfg, 5., 0.)
        ast.parse((folder/'check_contact.py').read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
