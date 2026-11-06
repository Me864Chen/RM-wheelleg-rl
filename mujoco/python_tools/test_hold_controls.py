"""Regression checks for hold-to-accelerate and release-to-zero commands."""
import runpy
from pathlib import Path
from types import SimpleNamespace
import unittest


class HoldControlTests(unittest.TestCase):
    def setUp(self):
        self.ns=runpy.run_path(str(Path(__file__).with_name('onnx_mj_chuanlian.py')))
        self.g=self.ns['on_glfw_key'].__globals__
        self.event=self.ns['on_glfw_key'];self.tick=self.ns['_refresh_cmd_from_keys']

    def hold(self,key,seconds):
        self.event(ord(key),1)
        for _ in range(round(seconds/.02)):self.tick(.02)

    def test_both_axes_accumulate_and_release(self):
        self.hold('2',1)
        self.assertAlmostEqual(self.g['cmd_lin_vel_target'],.5)
        self.hold('3',1)
        self.assertAlmostEqual(self.g['cmd_lin_vel_target'],1)
        self.assertAlmostEqual(self.g['cmd_yaw_vel_target'],1)
        self.event(ord('2'),0)
        self.assertEqual(self.g['cmd_lin_vel_target'],0)
        self.assertAlmostEqual(self.g['cmd_yaw_vel_target'],1)
        self.tick(.2)
        self.assertAlmostEqual(self.g['cmd_yaw_vel_target'],1.2)
        self.event(ord('3'),0)
        self.assertEqual(self.g['cmd_yaw_vel_target'],0)

    def test_negative_directions(self):
        self.hold('1',1);self.assertAlmostEqual(self.g['cmd_lin_vel_target'],-.5)
        self.hold('4',1);self.assertAlmostEqual(self.g['cmd_yaw_vel_target'],-1)

    def test_opposing_keys_cancel(self):
        self.event(ord('1'),1);self.event(ord('2'),1)
        self.event(ord('3'),1);self.event(ord('4'),1)
        self.tick(1)
        self.assertEqual(self.g['cmd_lin_vel_target'],0)
        self.assertEqual(self.g['cmd_yaw_vel_target'],0)

    def test_limits_and_repeat_independence(self):
        self.event(ord('2'),1);self.event(ord('3'),1)
        for _ in range(100):self.event(ord('2'),2)
        self.assertEqual(self.g['cmd_lin_vel_target'],0)
        self.tick(100)
        self.assertEqual(self.g['cmd_lin_vel_target'],self.g['LIN_VEL_MAX'])
        self.assertEqual(self.g['cmd_yaw_vel_target'],self.g['YAW_VEL_MAX'])
        self.event(ord('2'),0);self.event(ord('3'),0)
        self.hold('1',100);self.hold('4',100)
        self.assertEqual(self.g['cmd_lin_vel_target'],self.g['LIN_VEL_MIN'])
        self.assertEqual(self.g['cmd_yaw_vel_target'],self.g['YAW_VEL_MIN'])

    def test_focus_loss_and_emergency_stop(self):
        for event in [(-1,0),(ord('7'),1)]:
            self.hold('2',1);self.hold('3',1)
            self.g['cmd_lin_vel']=.5;self.g['cmd_yaw_vel']=1
            self.event(*event)
            self.assertFalse(self.g['pressed_keys'])
            for name in ['cmd_lin_vel','cmd_lin_vel_target','cmd_yaw_vel','cmd_yaw_vel_target']:
                self.assertEqual(self.g[name],0)

    def test_toggle_only_on_press(self):
        self.event(ord('0'),1);self.event(ord('0'),2);self.event(ord('0'),0)
        self.assertEqual(self.g['view_switch_requests'],1)

    def test_pynput_release_also_resets_targets(self):
        self.g['keyboard']=SimpleNamespace(Key=SimpleNamespace(esc=object(),space=object()))
        for key,target in [('2','cmd_lin_vel_target'),('3','cmd_yaw_vel_target')]:
            value=SimpleNamespace(char=key)
            self.ns['on_press'](value);self.tick(.5)
            self.assertGreater(self.g[target],0)
            self.ns['on_release'](value);self.assertEqual(self.g[target],0)

if __name__=='__main__':unittest.main()
