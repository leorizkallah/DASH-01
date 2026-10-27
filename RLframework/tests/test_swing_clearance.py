"""Focused tests for a physical swing-height target, cap and retained fine-tuning settings."""

from dataclasses import replace
import unittest
import numpy as np
from RLframework.config import get_config
from RLframework import swing_clearance as sc


class SwingClearanceTests(unittest.TestCase):
    def setUp(self):
        self.c = get_config("cad_v2_clearance01")
        self.sr = self.c.stance_ratio_final

    def phase(self, u):
        return (self.sr + np.asarray(u) * (1 - self.sr)) * 2 * np.pi

    def reward(
        self,
        height,
        u=(0.5, 0.5),
        grounded=(False, False),
        walking=True,
        air=(0.1, 0.1),
    ):
        return sc.shaping(
            np.asarray(height),
            self.phase(u),
            self.sr,
            np.asarray(grounded),
            walking,
            np.asarray(air),
            self.c,
        )

    def test_cost_monotone_and_zero_at_20mm(self):
        penalties = [self.reward([h, h])[0] for h in [0, 0.005, 0.01, 0.015, 0.02]]
        np.testing.assert_allclose(penalties, [-16, -12, -8, -4, 0])
        for h in [0.02, 0.025, 0.03, 0.05]:
            self.assertEqual(self.reward([h, h])[0], 0)

    def test_bonus_between_20_and_30mm_and_capped(self):
        np.testing.assert_allclose(
            [self.reward([h, h])[1] for h in [0.01, 0.02, 0.025, 0.03, 0.05]],
            [0, 0, 1, 2, 2],
            atol=1e-8,
        )

    def test_stance_and_endpoints_are_free(self):
        for u in [0.0, 1.0, -0.25]:
            low, bonus, _ = self.reward([0, 0.03], u=(u, u))
            self.assertAlmostEqual(low, 0)
            self.assertAlmostEqual(bonus, 0)
        for u in [0.05, 0.10, 0.15]:
            low, _, m = self.reward([0, 0], u=(u, u))
            np.testing.assert_allclose(m["clearance_target"], 0.020 * u / 0.2)
            self.assertGreater(low, -16)

    def test_grounded_foot_cannot_avoid_swing_cost_or_earn_bonus(self):
        self.assertEqual(self.reward([0, 0], grounded=(True, True))[0], -16)
        self.assertEqual(self.reward([0.03, 0.03], grounded=(True, True))[1], 0)
        self.assertEqual(self.reward([0.03, 0.03], air=(1.0, 1.0))[1], 0)

    def test_no_cross_foot_compensation(self):
        low, bonus, _ = self.reward([0, 0.03])
        self.assertEqual(low, -8)
        self.assertAlmostEqual(bonus, 1.0)
        a = self.reward([0.002, 0.028])
        b = self.reward([0.028, 0.002])
        self.assertAlmostEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1])

    def test_no_stop_reward_or_negative_height_exploit(self):
        self.assertEqual(self.reward([0, 0.03], walking=False)[:2], (0, 0))
        self.assertEqual(self.reward([-0.001, -0.1])[:2], self.reward([0, 0])[:2])

    def test_preset_and_validation(self):
        sc.validate(self.c)
        self.assertEqual(self.c.total_steps, 50_000_000)
        self.assertEqual(self.c.cmd_range, (0.6, 0.6))
        self.assertEqual(self.c.v_max * 0.6, 1.5)
        self.assertEqual(self.c.w_clearance, 0)
        self.assertFalse(self.c.warmstart_reset_log_std)
        self.assertTrue(self.c.symmetry_enable and self.c.flat_support_enable)
        self.assertEqual(self.c.dr_scale_start, self.c.dr_scale_final)
        self.assertFalse(
            get_config("cad_v2_flatfeet_symmetry_v1").swing_clearance_enable
        )
        for field in [
            dict(swing_clearance_min_m=0.0),
            dict(swing_clearance_bonus_cap_m=0.01),
            dict(swing_clearance_ramp_fraction=0.5),
            dict(two_point_feet=False),
        ]:
            with self.assertRaises(ValueError):
                sc.validate(replace(self.c, **field))

    def test_clearance02_band_and_excess_cost(self):
        self.c = get_config("cad_v2_clearance02")
        sc.validate(self.c)
        self.assertEqual(self.c.total_steps, 200_000_000)
        self.assertEqual(self.c.swing_clearance_min_m, 0.010)
        self.assertEqual(self.c.swing_clearance_bonus_cap_m, 0.025)
        self.assertEqual(self.c.walltime_hours, 24.0)
        self.assertEqual(self.c.cmd_range, (0.6, 0.6))
        self.assertTrue(self.c.symmetry_enable and self.c.flat_support_enable)
        np.testing.assert_allclose(
            [self.reward([h, h])[0] for h in [0, 0.005, 0.01, 0.025]], [-16, -8, 0, 0]
        )
        np.testing.assert_allclose(
            [self.reward([h, h])[1] for h in [0.01, 0.0175, 0.025, 0.04]], [0, 1, 2, 2]
        )
        costs = [
            float(self.reward([h, h])[2]["clearance_excess_cost"])
            for h in [0.01, 0.025, 0.0325, 0.04]
        ]
        np.testing.assert_allclose(costs, [0, 0, -2, -4])
        self.assertEqual(
            self.reward([0.04, 0.04], walking=False)[2]["clearance_excess_cost"], 0
        )
        self.assertAlmostEqual(
            self.reward([0.04, 0.04], u=(-0.25, -0.25))[2]["clearance_excess_cost"], 0
        )
        # Same-foot upper/lower violations cannot be cancelled by a compliant other foot.
        self.assertAlmostEqual(
            self.reward([0.04, 0.02])[2]["clearance_excess_cost"], -2
        )
        self.assertAlmostEqual(
            self.reward([0.02, 0.04])[2]["clearance_excess_cost"], -2
        )
        self.assertEqual(get_config("cad_v2_clearance01").w_excess_swing_clearance, 0)


if __name__ == "__main__":
    unittest.main()
