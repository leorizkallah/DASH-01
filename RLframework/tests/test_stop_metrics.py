import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stop_metrics import brake_quality, hold_quality, protocol, summarize
from config import get_config, config_from_dict, config_to_dict


class StopTests(unittest.TestCase):
    def test_braking_income_is_dense_and_posture_gated(self):
        speeds = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
        reward = brake_quality(speeds, 1, True)
        self.assertTrue(np.all(np.diff(reward) < 0))
        self.assertTrue(np.all(reward > 0))
        self.assertEqual(brake_quality(0, 0.4, True), 0)
        self.assertEqual(brake_quality(0, 1, False), 0)

    def test_hold_needs_support_posture_and_stillness(self):
        self.assertAlmostEqual(float(hold_quality(0, 0, 0, 1, True, True)), 1)
        for args in [(0, 0, 0, 1, False, True), (0, 0, 0, 0.5, True, True),
                     (0, 0, 0, 1, True, False)]:
            self.assertEqual(float(hold_quality(*args)), 0)
        self.assertLess(hold_quality(0.3, 0, 0, 1, True, True), 0.02)
        self.assertLess(hold_quality(0, 0, 3, 1, True, True), 0.001)

    def fixture(self):
        dt = 0.1
        speeds, a, b, ticks = protocol(4, dt)
        v = np.tile(speeds, (ticks, 1))
        for i in range(4):
            v[a[i]:b[i], i] = 0
        t = dict(speed=v.copy(), vx=v.copy(), gyro=np.zeros_like(v), joint_rms=np.zeros_like(v),
                 upright=np.ones_like(v), supported=np.ones_like(v, bool),
                 xy=np.stack([np.cumsum(v, axis=0) * dt, np.zeros_like(v)], axis=-1),
                 done=np.zeros_like(v, bool))
        return t, speeds, a, b, dt

    def test_walk_stop_restart_success(self):
        result = summarize(*self.fixture())
        self.assertEqual(result['success_fraction'], 1)
        self.assertEqual(result['restart_fraction'], 1)

    def test_fall_and_reset_cannot_be_a_success(self):
        t, s, a, b, dt = self.fixture()
        t['done'][a[0] + 1, 0] = True
        result = summarize(t, s, a, b, dt)
        self.assertFalse(result['episodes'][0]['stop_success'])
        self.assertIsNone(result['episodes'][0]['final_stop_speed_mps'])

    def test_never_walked_is_not_a_stop(self):
        t, s, a, b, dt = self.fixture()
        t['vx'][:a[0], 0] = 0
        self.assertFalse(summarize(t, s, a, b, dt)['episodes'][0]['stop_success'])

    def test_shuffle_and_late_drift_fail(self):
        t, s, a, b, dt = self.fixture()
        t['joint_rms'][a[0]:b[0], 0] = 2
        t['speed'][b[1] - 1, 1] = 0.2
        result = summarize(t, s, a, b, dt)
        self.assertFalse(result['episodes'][0]['stop_success'])
        self.assertFalse(result['episodes'][1]['stop_success'])

    def test_stop_preset_roundtrip_and_old_presets_unchanged(self):
        c = get_config('dash_stop')
        self.assertEqual(config_from_dict(config_to_dict(c)), c)
        self.assertEqual(c.n_steps, 64)
        self.assertTrue(c.freeze_obs_stats)
        self.assertFalse(c.lr_kl_adaptive)  # use actual minibatch LR warmup
        self.assertEqual(get_config('dash_joy3_lr').w_stand_hold, 0)
        self.assertFalse(get_config('dash_joy3_lr').freeze_obs_stats)


if __name__ == '__main__':
    unittest.main()
