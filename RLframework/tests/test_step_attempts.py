"""GAIT-09 behavioral regressions, including loopholes found in GAIT-08."""
import importlib.util
import unittest
import numpy as np
from RLframework import bilateral_gait as bg
from RLframework.config import get_config
from RLframework.tests.test_bilateral_gait import trajectory

C = get_config('cad_v2_gait09')


def initial():
    s = bg.initial(np.zeros(2), progress=True, sequence=True, attempts=True)
    s.update(last_foot=np.asarray(0), turn_grace=np.asarray(1.2))
    return s


def bout(s, foot=1, distance=.02, height=.006, ticks=12, cfg=C):
    totals, events = {}, {}
    for k in range(ticks+4):
        contact, h, x, swing = np.ones(2, bool), np.zeros(2), np.zeros(2), np.zeros(2)
        x[foot] = distance*min(k/(ticks-1), 1.)
        if k < ticks:
            contact[foot], h[foot], swing[foot] = False, height, 1.
        s, terms, m = bg.update(s, contact, h, x, np.zeros(2), swing, .01, cfg, frequency_hz=.5)
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.) + float(value)
        for key in ('attempt', 'valid_step', 'incomplete_step', 'repeated_attempt'):
            events[key] = events.get(key, np.zeros(2)) + m[key]
    return s, totals, events


class AttemptTests(unittest.TestCase):
    def test_invalid_same_foot_repeat_is_penalized_without_resetting_deadline(self):
        s, t, m = bout(initial(), foot=0)
        self.assertEqual(m['attempt'].sum(), 1)
        self.assertEqual(m['valid_step'].sum(), 0)
        self.assertEqual(t['bilateral_repeat'], -2.)
        self.assertLess(t['bilateral_incomplete'], 0.)
        self.assertGreater(s['turn_wait'], 0.)
        self.assertEqual(s['last_foot'], 0)

    def test_expected_foot_retries_cannot_recollect_same_reach(self):
        s, first, _ = bout(initial())
        s, second, _ = bout(s)
        self.assertGreater(first['bilateral_forward_progress'], 0.)
        self.assertAlmostEqual(second['bilateral_forward_progress'], 0.)
        self.assertEqual(second['bilateral_repeat'], 0.)
        self.assertLess(second['bilateral_incomplete'], 0.)

    def test_lift_credit_exhausts_until_opposite_valid_step(self):
        s = initial()
        for _ in range(20):
            s, _, _ = bout(s)
        s, t, _ = bout(s)
        self.assertAlmostEqual(t['bilateral_lift_progress'], 0.)
        self.assertLessEqual(s['turn_lift_used'].max(), C.bilateral_lift_budget_s+1e-9)
        s, t, m = bout(s, distance=.06, height=.02)
        self.assertEqual(m['valid_step'].sum(), 1)
        self.assertEqual(t['bilateral_step'], 3.)
        np.testing.assert_allclose(s['turn_lift_used'], 0.)
        np.testing.assert_allclose(s['turn_reach'], 0.)
        s, t, _ = bout(s, foot=0)
        self.assertGreater(t['bilateral_forward_progress'], 0.)

    def test_noise_and_rolling_are_not_attempts(self):
        for kwargs in (dict(height=.001, distance=.001), dict(ticks=3)):
            _, t, m = bout(initial(), foot=0, **kwargs)
            self.assertEqual(m['attempt'].sum(), 0)
            self.assertEqual(t['bilateral_repeat'], 0.)
        s = initial()
        for _ in range(10):
            s, _, m = bg.update(s, np.ones(2, bool), np.zeros(2), np.ones(2)*.02,
                                np.zeros(2), np.ones(2), .01, C)
            self.assertFalse(m['attempt'].any())

    def test_incomplete_cost_is_smooth_and_clean_repeat_not_double_charged(self):
        _, low, _ = bout(initial(), height=.004, distance=.01)
        _, better, _ = bout(initial(), height=.012, distance=.035)
        _, clean, _ = bout(initial(), foot=0, height=.02, distance=.06)
        self.assertLess(low['bilateral_incomplete'], better['bilateral_incomplete'])
        self.assertLess(better['bilateral_incomplete'], 0.)
        self.assertEqual(clean['bilateral_incomplete'], 0.)
        self.assertEqual(clean['bilateral_repeat'], -2.)

    def test_invalid_lift_does_not_change_completed_stride_history(self):
        s = initial()
        s.update(last_stance=np.ones(2), last_air=np.full(2, .2), last_length=np.full(2, .1))
        s, _, _ = bout(s)
        np.testing.assert_allclose(s['last_stance'], 1.)
        np.testing.assert_allclose(s['last_air'], .2)
        s['stance'] = np.full(2, .7)
        s, _, m = bout(s, height=.02, distance=.06)
        self.assertEqual(m['valid_step'][1], 1)
        self.assertAlmostEqual(s['last_stance'][1], .71)
        self.assertGreater(s['last_air'][1], .08)
        self.assertGreater(s['last_length'][1], .04)

    def test_cadence_allowance_and_legacy_opt_out(self):
        self.assertAlmostEqual(bg.stance_allowance(.5, .65, C), 1.5)
        self.assertAlmostEqual(bg.stance_allowance(1., .65, C), .85)
        for n in range(1, 9):
            self.assertFalse(get_config(f'cad_v2_gait{n:02}').bilateral_attempt_enable)

    def test_clean_walking_preferred_and_mirror_invariant(self):
        good = trajectory(cfg=C)[0]
        for kind in ('walk', 'stand', 'drag', 'chatter', 'hop'):
            score, steps, _ = trajectory(kind, cfg=C)
            mirrored, other, _ = trajectory(kind, swap=True, cfg=C)
            self.assertAlmostEqual(score, mirrored)
            np.testing.assert_array_equal(steps, other[::-1])
            if kind != 'walk':
                self.assertGreater(good, score, kind)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX check runs on SCITAS')
    def test_jit_tree_and_finite_terms(self):
        import jax
        import jax.numpy as jnp
        s = bg.initial(jnp.zeros(2), jnp, progress=True, sequence=True, attempts=True)
        fn = jax.jit(lambda s: bg.update(s, jnp.ones(2, bool), jnp.zeros(2),
            jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), .01, C, jnp))
        nxt, terms, _ = fn(s)
        self.assertEqual(jax.tree_util.tree_structure(s), jax.tree_util.tree_structure(nxt))
        self.assertTrue(all(np.isfinite(v) for v in terms.values()))


if __name__ == '__main__':
    unittest.main()
