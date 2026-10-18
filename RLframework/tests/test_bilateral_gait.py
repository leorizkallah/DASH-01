import unittest
from types import SimpleNamespace
import numpy as np
from RLframework import bilateral_gait as bg

C = SimpleNamespace(bilateral_min_swing_s=.08, bilateral_clearance_m=.015,
                    bilateral_step_min_m=.04, bilateral_swing_frac=.15, bilateral_tolerance=.2)


def trajectory(kind='walk', swap=False, cfg=C):
    s = bg.initial(np.array([-.08, .03]), progress=getattr(cfg, 'bilateral_progress_enable', False),
                   sequence=getattr(cfg, 'bilateral_sequence_enable', False),
                   attempts=getattr(cfg, 'bilateral_attempt_enable', False))
    rewards, steps, alternates = [], np.zeros(2), np.zeros(2)
    for k in range(1200):
        t = k*.01
        phase = (t+np.array([0., 0. if kind == 'hop' else 1.])) % 2.
        contact = phase >= .6
        height = np.where(contact, 0., .03*np.sin(np.pi*phase/.6))
        x = np.where(contact, .08-.16*(phase-.6)/1.4, -.08+.16*phase/.6)
        slip = np.zeros(2)
        if kind == 'drag':
            contact[1], height[1], x[1], slip[1] = True, 0., -.08, .15
        if kind == 'stand':
            contact[:], height[:], x[:] = True, 0., 0.
        if kind == 'chatter':
            contact[:], height[:] = bool(k % 2), 0.
        swing = (phase < .6).astype(float)
        if swap:
            contact, height, x, slip, swing = [a[::-1] for a in (contact, height, x, slip, swing)]
        s, terms, m = bg.update(s, contact, height, x, slip, swing, .01, cfg)
        if k > 300:
            rewards.append(sum(terms.values()))
            steps += m['valid_step']
            alternates += m['alternating_step']
    return np.mean(rewards), steps, alternates


class BilateralTests(unittest.TestCase):
    def test_progress_rewards_partial_lift_and_caps_excessive_height(self):
        cfg = SimpleNamespace(**vars(C), bilateral_progress_enable=True)
        def sample(height):
            s = bg.initial(np.zeros(2), progress=True)
            s['contact'][:] = False
            return bg.update(s, np.zeros(2, bool), np.array([height, 0.]),
                             np.array([.01, 0.]), np.zeros(2), np.array([1., 0.]), .01, cfg)[1]
        scores = [sum(sample(h).values()) for h in (0., .003, .009, .015, .10)]
        self.assertTrue(all(a < b for a, b in zip(scores[:3], scores[1:4])), scores)
        self.assertAlmostEqual(scores[3], scores[4])

    def test_progress_missing_history_has_cost_and_no_pair_credit(self):
        cfg = SimpleNamespace(**vars(C), bilateral_progress_enable=True)
        s = bg.initial(np.zeros(2), progress=True)
        s['age'] = np.full(2, 10.)
        _, terms, _ = bg.update(s, np.ones(2, bool), np.zeros(2), np.zeros(2),
                                np.zeros(2), np.array([1., 0.]), .01, cfg)
        self.assertLess(terms['bilateral_symmetry'], 0.)
        self.assertEqual(terms['bilateral_pair_progress'], 0.)
        self.assertAlmostEqual(terms['bilateral_phase'], -1.)

    def test_forward_progress_cannot_repeat_same_airborne_distance(self):
        cfg = SimpleNamespace(**vars(C), bilateral_progress_enable=True)
        s = bg.initial(np.zeros(2), progress=True)
        s['contact'][:] = False
        credits = []
        for x in (.02, .01, .02, .03):
            s, terms, _ = bg.update(s, np.zeros(2, bool), np.ones(2)*.015,
                np.ones(2)*x, np.zeros(2), np.array([1., 0.]), .01, cfg)
            credits.append(terms['bilateral_forward_progress'])
        self.assertGreater(credits[0], 0.)
        self.assertEqual(credits[1:3], [0., 0.])
        self.assertGreater(credits[3], 0.)

    def test_progress_variant_prefers_alternating_gait_and_is_symmetric(self):
        cfg = SimpleNamespace(**vars(C), bilateral_progress_enable=True)
        good = trajectory(cfg=cfg)[0]
        for kind in ('walk', 'drag', 'stand', 'hop', 'chatter'):
            a, steps, _ = trajectory(kind, cfg=cfg)
            b, swapped, _ = trajectory(kind, swap=True, cfg=cfg)
            self.assertAlmostEqual(a, b, places=6)
            np.testing.assert_array_equal(steps, swapped[::-1])
            if kind != 'walk':
                self.assertGreater(good, a, kind)

    def test_alternating_steps_outscore_dragging_standing_and_hopping(self):
        good, steps, alt = trajectory()
        self.assertTrue(np.all(steps >= 4), steps)
        np.testing.assert_array_equal(steps, alt)
        for kind in ('drag', 'stand', 'hop', 'chatter'):
            score, _, _ = trajectory(kind)
            self.assertGreater(good, score, kind)

    def test_drag_cannot_earn_opposite_foot_credit(self):
        _, steps, alt = trajectory('drag')
        self.assertEqual(steps[1], 0)
        self.assertLess(alt.sum(), steps.sum())

    def test_contact_chatter_and_simultaneous_hops_do_not_count(self):
        for kind in ('chatter', 'hop', 'stand'):
            _, steps, _ = trajectory(kind)
            self.assertEqual(steps.sum(), 0)

    def test_side_exchange_preserves_reward(self):
        for kind in ('walk', 'drag'):
            a, steps, _ = trajectory(kind)
            b, swapped, _ = trajectory(kind, swap=True)
            self.assertAlmostEqual(a, b, places=6)
            np.testing.assert_array_equal(steps, swapped[::-1])

    def test_rolling_contact_point_can_be_stationary(self):
        v = bg.point_velocity(np.array([0., 1., 0., 1., 0., 0.]), np.array([0., 0., -1.]), np.zeros(3))
        np.testing.assert_allclose(v, 0.)

    def test_quality_rejects_drag_and_flight_despite_good_tracking(self):
        m = dict(survived=np.array([True]), track_error=np.array([.03]), heading_error=np.array([.01]),
                 steps=np.array([[6, 6]]), clear_fraction=np.array([[.15, .15]]), alternation=np.array([1.]),
                 max_symmetry=np.array([.1]), flight_fraction=np.array([0.]), contact_slip=np.array([.01]))
        self.assertTrue(bg.assess(m)[0])
        for change in (dict(steps=np.array([[6, 0]])), dict(flight_fraction=np.array([.04])),
                       dict(max_symmetry=np.array([.5])), dict(contact_slip=np.array([.2])),
                       dict(joint_limit_violation=np.array([.025]))):
            self.assertFalse(bg.assess(dict(m, **change))[0])


if __name__ == '__main__':
    unittest.main()
