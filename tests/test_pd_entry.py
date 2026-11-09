"""The standalone checker must read the same literal settings as the RL task."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

from test_joint_mapping import BRANCHES, CODE, HEROCFG, ROOT

folder = ROOT / 'heroleg26_isaac_check'
sys.path.insert(0, str(folder))
try:
    spec = importlib.util.spec_from_file_location('pd_entry_check', folder/'check_pd.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.remove(str(folder))


class PDEntryTests(unittest.TestCase):
    def test_collision_masks_are_verified_and_bad_masks_rejected(self):
        gym = SimpleNamespace(get_actor_rigid_shape_properties=lambda env, actor:
                              [SimpleNamespace(filter=x) for x in (1, 1, 1)])
        self.assertEqual(entry.verify_no_self_collision(gym, 'env', 0), [1, 1, 1])
        for masks in ([], [0], [1, 2], [1, 0]):
            gym.get_actor_rigid_shape_properties = lambda env, actor: [SimpleNamespace(filter=x) for x in masks]
            with self.assertRaises(RuntimeError):
                entry.verify_no_self_collision(gym, 'env', 0)
        # Exercise the actual actor creation statement to prevent regression
        # back to -1, which preserves imported collision settings.
        tree = ast.parse((folder/'check_pd.py').read_text())
        assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                      and n.value.func.attr == 'create_actor')
        called = []
        gym = SimpleNamespace(create_actor=lambda *args: called.append(args) or 0)
        code = compile(ast.fix_missing_locations(ast.Module(body=[assign], type_ignores=[])), '<actual-create-actor>', 'exec')
        exec(code, {'gym': gym, 'env': 'env', 'asset': 'asset', 'pose': 'pose'})
        self.assertEqual(called[0][-1], 1)

    def test_diagnostic_period_does_not_mutate_training_settings(self):
        original = entry.load_config(ROOT, 'plane')
        result = entry.diagnostic_config(original, .002)
        self.assertEqual(original['sim']['dt'], .001)
        self.assertEqual(result['sim']['dt'], .002)
        self.assertEqual(result['asset'], original['asset'])
        self.assertEqual(result['control'], original['control'])
        self.assertEqual(result['init_state'], original['init_state'])
        for invalid in (0., -.001, float('nan'), .1):
            with self.assertRaises(ValueError):
                entry.diagnostic_config(original, invalid)

    def test_pd_rate_preserves_policy_interval_and_legacy_task(self):
        for branch in BRANCHES:
            own = HEROCFG[branch]
            legacy = CODE[branch][1]
            self.assertEqual(own.sim.dt, .001)
            self.assertEqual(own.control.decimation, 10)
            self.assertEqual(legacy.sim.dt, .005)
            self.assertEqual(legacy.control.decimation, 2)
            self.assertAlmostEqual(own.sim.dt * own.control.decimation,
                                   legacy.sim.dt * legacy.control.decimation)

    def test_blocking_fetch_does_not_treat_falsey_return_as_failure(self):
        # Execute the actual synchronization statements from the runtime loop.
        # Some installed SDK/backend combinations return a falsey value here;
        # finite states and the response checks must decide physics validity.
        tree = ast.parse((folder/'check_pd.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_check')
        loop = next(n for n in ast.walk(run) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == 'step')
        start = next(i for i, n in enumerate(loop.body)
                     if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'simulate')
        end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.AugAssign)
                   and isinstance(n.target, ast.Subscript))
        code = compile(ast.fix_missing_locations(ast.Module(
            body=loop.body[start:end+1], type_ignores=[])), '<actual-physics-sync>', 'exec')
        for returned in (None, False, True):
            calls = []
            def fetch(sim, blocking):
                calls.append(('fetch', sim, blocking))
                return returned
            report = {'physics_steps': 0}
            gym = SimpleNamespace(simulate=lambda sim: calls.append(('simulate', sim)), fetch_results=fetch)
            exec(code, {'gym': gym, 'sim': 'test_sim', 'report': report})
            self.assertEqual(calls, [('simulate', 'test_sim'), ('fetch', 'test_sim', True)])
            self.assertEqual(report['physics_steps'], 1)
            self.assertEqual(report['first_fetch_results_return'], repr(returned))
        def fail(sim, blocking):
            raise RuntimeError('native synchronization failure')
        with self.assertRaisesRegex(RuntimeError, 'native synchronization failure'):
            exec(code, {'gym': SimpleNamespace(simulate=lambda sim: None, fetch_results=fail),
                        'sim': 'test_sim', 'report': {'physics_steps': 0}})

    def test_reader_matches_actual_inherited_classes(self):
        for branch in BRANCHES:
            literal = entry.load_config(ROOT, branch)
            cls = HEROCFG[branch]
            for section, fields in literal.items():
                if not isinstance(fields, dict):
                    continue
                actual = cls
                for key in section.split('.'):
                    actual = getattr(actual, key)
                for key, value in fields.items():
                    self.assertEqual(value, getattr(actual, key), (branch, section, key))

    def test_schedule_checks_both_signs_for_every_joint(self):
        names = HEROCFG['plane'].asset.policy_joint_names
        phases = entry.phase_plan(names)
        active = [(p['joint'], p['sign']) for p in phases if p['joint']]
        self.assertCountEqual(active, [(name, sign) for name in names for sign in (1, -1)])
        self.assertEqual(phases[-1]['label'], 'final_stop')
        self.assertAlmostEqual(sum(p['seconds'] for p in phases), 40.)

    def test_python38_grammar(self):
        for path in [folder/'check_pd.py'] + [
                ROOT/branch/'wheel_legged_gym/utils/motor_limits.py' for branch in BRANCHES]:
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
