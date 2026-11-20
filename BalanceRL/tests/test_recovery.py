"""Behavioral recovery tests: stepping, deadlines and reward-exploitation boundaries."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config
from recovery_logic import initial, advance
from recovery_curriculum import update


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = get_config('recover_v1')
        self.xy = np.array([[0., .2], [0., -.2]])
        self.base = np.zeros(2)
        self.force = np.array([70., 70.])
        self.s = initial(self.xy.copy(), self.base.copy(), np.ones(2, bool))
        self.t = 0.

    def tick(self, **kw):
        self.t += .01
        args = dict(xy=self.xy.copy(), base=self.base.copy(), normal_force=self.force.copy(),
                    clearance=np.zeros(2), kinematic_quiet=True, started=False,
                    duration=.1, sector=0, now=self.t, dt=.01, cfg=self.cfg,
                    lateral_axis=np.array([0., 1.]))
        args.update(kw)
        self.s, m = advance(self.s, **args)
        return m

    def test_small_in_place_steps_are_counted_and_repetition_costs_more(self):
        self.tick(started=True)
        landings = 0
        excess = 0
        for _ in range(4):
            for _ in range(3):
                self.tick(normal_force=np.array([0., 70.]))
            for _ in range(3):
                m = self.tick()
                landings += m['landing'].sum()
                excess += m['excess_cost']
        self.assertEqual(landings, 4)
        self.assertEqual(self.s['total_steps'], 4)
        self.assertGreater(excess, 0)

    def test_chatter_and_subthreshold_travel_are_not_free(self):
        self.tick(started=True)
        path = switches = 0
        for i in range(8):
            xy = self.xy.copy()
            xy[0, 0] = .0001*(i % 2)
            m = self.tick(xy=xy, normal_force=np.array([0. if i % 2 else 70., 70.]))
            path += m['path']; switches += m['switches']
        self.assertGreater(path, 0)
        self.assertGreater(switches, 0)
        self.assertEqual(self.s['total_steps'], 0)  # debounce; continuous costs still see it

    def test_one_pad_rocking_does_not_make_a_whole_foot_step(self):
        # Input is total force on each foot; one pad may carry all of that load.
        for _ in range(10):
            m = self.tick(normal_force=np.array([8., 70.]))
            self.assertFalse(m['liftoff'].any())
            self.assertFalse(m['task_bad'])

    def test_relocated_footprint_commits_once_after_real_settling(self):
        self.tick(started=True, duration=.15)
        old_anchor = self.s['anchor'].copy()
        moved = self.xy.copy(); moved[0, 0] = .08
        self.tick(xy=moved, kinematic_quiet=False)
        np.testing.assert_array_equal(self.s['anchor'], old_anchor)
        oks = 0
        for _ in range(530):
            m = self.tick(xy=moved)
            oks += int(m['success'])
        self.assertEqual(oks, 1)
        np.testing.assert_array_equal(self.s['anchor'], moved)
        self.assertFalse(self.s['recovering'])
        self.assertFalse(m['task_bad'])
        # A new unsolicited whole-foot step after settling fails the task.
        for _ in range(3):
            m = self.tick(xy=moved, normal_force=np.array([0., 70.]))
            if m['task_bad']:
                break
        self.assertTrue(m['task_bad'])

    def test_shuffling_cannot_extend_deadline(self):
        self.tick(started=True)
        end = self.s['pulse_end']
        failed = False
        for _ in range(840):
            m = self.tick(kinematic_quiet=False)
            self.assertEqual(self.s['pulse_end'], end)
            if m['expired']:
                failed = True
                break
        self.assertTrue(failed)
        self.assertLess(self.t, 8.2)

    def test_actual_pulse_duration_controls_quiet_start(self):
        self.tick(started=True, duration=.15)
        for _ in range(13):
            self.tick()
            self.assertEqual(self.s['quiet_s'], 0.)
        self.tick(); self.tick()
        self.assertGreater(self.s['quiet_s'], 0.)

    def test_same_quiet_state_cannot_succeed_twice(self):
        self.tick(started=True)
        oks = sum(bool(self.tick()['success']) for _ in range(700))
        self.assertEqual(oks, 1)

    def test_crossing_or_walking_away_is_invalid(self):
        self.tick(started=True)
        crossed = self.xy[::-1].copy()
        self.assertTrue(self.tick(xy=crossed)['workspace_bad'])
        self.assertTrue(self.tick(base=np.array([.6, 0.]))['workspace_bad'])

    def test_promotion_needs_two_passes_and_all_directions(self):
        agent = SimpleNamespace(cfg=self.cfg, cur=dict(push_level=.1, push_stage=3, n_promote=0))
        good = dict(hard_trials=200, hard_rate=.95, quiet_rate=1., sector_rates=[.96]*8)
        update(agent, good)
        self.assertEqual(agent.cur['push_level'], .1)
        weak_side = dict(good, sector_rates=[.96]*7+[.6])
        update(agent, weak_side)
        self.assertEqual(agent.cur['probe_streak'], 0)
        update(agent, good); update(agent, good)
        self.assertAlmostEqual(agent.cur['push_level'], .11)
        self.assertEqual(agent.cur['probe_streak'], 0)

    def test_variety_changes_separately_and_quiet_regression_retreats(self):
        agent = SimpleNamespace(cfg=self.cfg, cur=dict(push_level=.1, push_stage=0, n_promote=0))
        good = dict(hard_trials=200, hard_rate=1., quiet_rate=1., sector_rates=[1.]*8)
        update(agent, good); update(agent, good)
        self.assertEqual(agent.cur['push_stage'], 1)
        self.assertEqual(agent.cur['push_level'], .1)
        quiet_bad = dict(good, quiet_rate=.75)
        update(agent, quiet_bad); update(agent, quiet_bad)
        self.assertAlmostEqual(agent.cur['push_level'], .08)


if __name__ == '__main__':
    unittest.main()
