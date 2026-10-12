import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imitation_data import dataset_masks, initialization_gate


class ImitationDataTests(unittest.TestCase):
    def test_no_autoreset_leak_and_whole_episode_holdout(self):
        done = np.zeros((5, 16), bool)
        done[1, 0] = True
        upright = np.ones_like(done, float)
        upright[2, 1] = 0.
        finite = np.ones_like(done)
        finite[3, 9] = False
        hold = np.arange(16) < 8
        m = dataset_masks(done, upright, hold, finite)
        self.assertTrue(m['hold_valid'][0, 0])
        self.assertFalse(m['hold_valid'][1:, 0].any())
        self.assertFalse(m['hold_train'][2, 1])
        self.assertFalse(m['walk_train'][3, 9])
        for task in ('hold', 'walk'):
            self.assertFalse((m[task+'_train'].any(0) & m[task+'_valid'].any(0)).any())
            self.assertTrue(m[task+'_valid'].any())

    def test_fit_cannot_substitute_for_behavior(self):
        before = dict(success_fraction=0., longest_quiet_hold_mean_s=.02)
        after = dict(before, walking_retained=True)
        losses = dict(hold_before=.1, hold_after=.001, walk_after=.001)
        self.assertFalse(all(initialization_gate(before, after, losses).values()))
        after['longest_quiet_hold_mean_s'] = .10
        self.assertTrue(all(initialization_gate(before, after, losses).values()))
        after['walking_retained'] = False
        self.assertFalse(all(initialization_gate(before, after, losses).values()))


if __name__ == '__main__':
    unittest.main()
