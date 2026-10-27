"""Synthetic evaluation tests: height bands, correct threshold labels and selection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config


@unittest.skipUnless(importlib.util.find_spec("jax"), "SCITAS JAX environment required")
class ClearanceEvaluationTests(unittest.TestCase):
    def result(self, height=0.015, peak=None, preset="cad_v2_clearance02"):
        from clearance_eval import ClearanceEvaluator

        cfg = get_config(preset)
        ticks, n = 3000, 12
        heights = np.full((ticks, n, 2), height)
        if peak is not None:
            heights[1000, :, 0] = peak
        row = dict(
            alive=np.ones((ticks, n), bool),
            fallen=np.zeros((ticks, n), bool),
            symmetry=np.zeros((ticks, n), bool),
            height=heights,
            grounded=np.zeros((ticks, n, 2), bool),
            middle=np.ones((ticks, n, 2), bool),
            error=np.zeros((ticks, n)),
            reward=np.ones((ticks, n)),
            floor=np.zeros((ticks, n), bool),
            valid_step=np.broadcast_to(
                (np.arange(ticks) % 100 == 0)[:, None, None], (ticks, n, 2)
            ),
        )
        ev = ClearanceEvaluator.__new__(ClearanceEvaluator)
        ev.cfg = cfg
        ev.env = SimpleNamespace(control_dt=0.01, n_envs=n)
        ev.fn = lambda *args: row
        agent = SimpleNamespace(
            params=None, stats=SimpleNamespace(mean=None, var=None, count=None), step=0
        )
        return ev.evaluate(agent)

    def test_band_qualification_and_labels(self):
        r = self.result()
        self.assertTrue(r["keeper_eligible"])
        self.assertEqual(r["midswing_ge10mm_fraction"], [1.0, 1.0])
        self.assertNotIn("midswing_ge20mm_fraction", r)
        self.assertEqual(r["minimum_clearance_mm"], 10)
        self.assertEqual(r["bonus_cap_mm"], 25)

    def test_peak_cost_and_qualification(self):
        self.assertFalse(self.result(peak=0.04)["keeper_eligible"])
        self.assertTrue(self.result(peak=0.0255)["keeper_eligible"])
        self.assertFalse(self.result(height=0.002)["keeper_eligible"])

    def test_candidate_prefers_band_over_grounded_or_excessive_height(self):
        good, low, high = self.result(), self.result(0.002), self.result(0.04)
        self.assertGreater(good["candidate_rank"], low["candidate_rank"])
        self.assertGreater(good["candidate_rank"], high["candidate_rank"])
        # Legacy CLEARANCE-01 retains 20 mm labels and no upper-height qualification.
        legacy = self.result(0.04, preset="cad_v2_clearance01")
        self.assertIn("midswing_ge20mm_fraction", legacy)
        self.assertTrue(legacy["keeper_eligible"])


if __name__ == "__main__":
    unittest.main()
