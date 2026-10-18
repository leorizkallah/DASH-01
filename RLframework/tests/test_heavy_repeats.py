"""GAIT-10: retained heavy costs, smooth duration cost and measurable progress."""
import importlib.util
import unittest
from dataclasses import replace
import numpy as np
from RLframework import bilateral_gait as bg
from RLframework.config import get_config
from RLframework.gait_diagnostics import attempt_summary, continuation_decision
from RLframework.tests.test_step_attempts import initial, bout

C = get_config('cad_v2_gait10')


def evaluation(step, repeats=20, incomplete=.5, survival=1., steps=10):
    return dict(step=step, keeper_eligible=False, finite=True,
                survival_fraction=survival, track_error=.01,
                per_trial=dict(repeated_attempt=[[repeats/2]*2]*12,
                               incomplete_fraction=[incomplete]*12,
                               steps=[[steps]*2]*12, contact_slip=[.01]*12))


class HeavyRepeatTests(unittest.TestCase):
    def test_event_is_ten_times_heavier_and_wrong_airtime_is_costly(self):
        _, old, _ = bout(initial(), foot=0)
        _, new, _ = bout(initial(), foot=0, cfg=C)
        self.assertAlmostEqual(new['bilateral_repeat'], 10*old['bilateral_repeat'])
        self.assertLess(new['bilateral_repeat_air'], -20.)
        self.assertEqual(bout(initial(), foot=1, cfg=C)[1]['bilateral_repeat_air'], 0.)

    def test_longer_wrong_swing_costs_more_and_noise_is_free(self):
        _, short, _ = bout(initial(), foot=0, ticks=12, cfg=C)
        _, long, _ = bout(initial(), foot=0, ticks=24, cfg=C)
        self.assertLess(long['bilateral_repeat_air'], short['bilateral_repeat_air'])
        self.assertAlmostEqual(bout(initial(), foot=0, ticks=2, cfg=C)[1]['bilateral_repeat_air'], 0.)
        cost, intensity = bg.extra_airtime(0, np.ones(2, bool), np.ones(2), np.ones(2), np.ones(2), C)
        self.assertEqual(cost, 0.)  # rolling with a grounded pad is not a lift

    def test_heavy_cost_survives_floor_exactly_once(self):
        terms = dict(bilateral_repeat=-20., bilateral_repeat_air=-4.)
        reward, hit = bg.reward_with_floor(3.-24., terms, C, .5)
        self.assertAlmostEqual(reward, 1.5-12.)
        self.assertFalse(hit)
        reward, hit = bg.reward_with_floor(-3.-24., terms, C, .5)
        self.assertAlmostEqual(reward, -.5-12.)
        self.assertTrue(hit)
        self.assertAlmostEqual(bg.reward_with_floor(3.-24., terms, get_config('cad_v2_gait09'), .5)[0], -.5)

    def test_required_foot_gets_progress_even_outside_old_clock(self):
        s = initial()
        s['contact'][:] = False
        inputs = (np.zeros(2, bool), np.full(2, .015), np.full(2, .02), np.zeros(2), np.zeros(2), .01)
        _, old, _ = bg.update(s, *inputs, get_config('cad_v2_gait09'))
        _, new, m = bg.update(s, *inputs, C)
        self.assertEqual(old['bilateral_lift_progress'], 0.)
        self.assertGreater(new['bilateral_lift_progress'], 0.)
        self.assertEqual(m['lift_progress'][0], 0.)

    def test_duration_is_side_symmetric(self):
        for last in (0, 1):
            args = (np.array([False, True]), np.array([.13, 0.]), np.array([.008, 0.]), np.zeros(2))
            a = bg.extra_airtime(last, *args, C)[0]
            b = bg.extra_airtime(1-last, *[x[::-1] for x in args], C)[0]
            self.assertEqual(a, b)

    def test_progress_requires_sustained_cleaner_walking(self):
        baseline = evaluation(0)
        good = [baseline, evaluation(1, 14, .35), evaluation(2, 14, .35)]
        self.assertEqual(continuation_decision(good)['decision'], 'progress_review_video')
        for records in ([baseline, evaluation(1)], [baseline, evaluation(1, 14, .35)],
                        [baseline, evaluation(1, 14, .35, survival=.5), evaluation(2, 14, .35, survival=.5)],
                        [baseline, evaluation(1, 0, 0., steps=0), evaluation(2, 0, 0., steps=0)]):
            self.assertEqual(continuation_decision(records)['decision'], 'fresh_policy_with_explicit_step_objective')

    def test_metrics_mask_resets_and_empty_quantiles(self):
        q = {k: np.zeros((3, 1, 2), bool) for k in (
            'attempt','incomplete_step','repeated_attempt','rejected_air','rejected_clearance','rejected_travel')}
        for k in ('attempt_length', 'attempt_height', 'attempt_air'):
            q[k] = np.ones((3, 1, 2))
        q['attempt'][1:, 0, 0] = True
        m = attempt_summary(q, np.array([[True], [True], [False]]), np.zeros((1, 3)), .01)
        np.testing.assert_array_equal(m['attempt'], [[1, 0]])
        self.assertEqual(m['attempt_length_quantiles'].tolist(), [[[1., 1., 1.], [None]*3]])

    def test_preset_budget_and_legacy_opt_out(self):
        self.assertEqual(C.total_steps, 15_000_000)
        self.assertTrue(C.bilateral_controller_phase)
        self.assertFalse(C.warmstart_reset_log_std)
        for i in range(1, 10):
            c = get_config(f'cad_v2_gait{i:02}')
            self.assertFalse(c.bilateral_protect_repeat_cost)
            self.assertFalse(c.bilateral_turn_progress)
            self.assertEqual(c.bilateral_repeat_air_weight, 0.)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX validation runs on SCITAS')
    def test_jit_tree_and_protected_reward(self):
        import jax
        import jax.numpy as jnp
        s = bg.initial(jnp.zeros(2), jnp, progress=True, sequence=True, attempts=True)
        def step(s):
            nxt, terms, m = bg.update(s, jnp.ones(2, bool), jnp.zeros(2), jnp.zeros(2),
                                     jnp.zeros(2), jnp.zeros(2), .01, C, jnp)
            reward, _ = bg.reward_with_floor(sum(terms.values()), terms, C, .5, jnp)
            return nxt, reward, m
        nxt, reward, _ = jax.jit(step)(s)
        self.assertEqual(jax.tree_util.tree_structure(s), jax.tree_util.tree_structure(nxt))
        self.assertTrue(np.isfinite(reward))


if __name__ == '__main__':
    unittest.main()
