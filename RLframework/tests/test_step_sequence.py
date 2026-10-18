from dataclasses import replace
import importlib.util
import unittest
import numpy as np
from RLframework import bilateral_gait as bg
from RLframework.config import get_config
from RLframework.tests.test_bilateral_gait import trajectory

C = replace(get_config('cad_v2_gait07'), bilateral_sequence_enable=True)


class SequenceTests(unittest.TestCase):
    def test_long_preset_preserves_policy_setup_and_full_drag(self):
        c = get_config('cad_v2_gait08')
        self.assertEqual(c.total_steps, 30_000_000)
        self.assertTrue(c.bilateral_sequence_enable)
        self.assertEqual(c.drag_scale_start, 1.)
        self.assertEqual(c.learning_rate, 3e-5)
        self.assertEqual(c.v_max*c.cmd_range[0], .25)
        self.assertFalse(c.warmstart_reset_log_std)
        self.assertEqual(c.warmstart_obs_count_cap, 0.)
        self.assertFalse(c.dr_enable)
        self.assertEqual(c.push_interval_s, 0.)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX validation requires SCITAS')
    def test_full_drag_cannot_reset_on_fresh_curriculum_or_failed_gate(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ppo import PPO, initial_params
        c = get_config('cad_v2_gait08')
        agent = PPO.__new__(PPO)
        agent.cfg, agent.cur, agent.step = c, {}, 0
        agent.env_params = initial_params(c)
        self.assertEqual(agent.env_params.drag_scale, 1.)
        agent.update_curricula(100., 1_000_000)
        self.assertEqual(agent.env_params.drag_scale, 1.)
        self.assertEqual(agent.env_params.shape_scale, .15)
        agent.cur['drag_success'] = dict(advancing=True, progress=.5)
        agent.update_curricula(800., 1_000_000)
        self.assertEqual(agent.env_params.drag_scale, 1.)

    def turn(self, last, wait, grace, valid=(False, False), frequency=.5):
        return bg.next_turn(last, wait, grace, np.array(valid), .01, frequency, C)

    def test_first_step_either_side_starts_fair_deadline(self):
        for event in ((True, False), (False, True)):
            wait, grace, cost = self.turn(-1, 0., 0., event)
            self.assertEqual(wait, 0.)
            self.assertAlmostEqual(grace, 1.2)
            self.assertEqual(cost, 0.)
        self.assertEqual(self.turn(-1, 0., 0.)[2], 0.)

    def test_grace_then_gradual_bounded_cost(self):
        self.assertEqual(self.turn(0, .5, 1.2)[2], 0.)
        self.assertAlmostEqual(self.turn(0, 1.69, 1.2)[2], .5)
        self.assertEqual(self.turn(0, 5., 1.2)[2], 1.)

    def test_repeat_and_cadence_change_cannot_reset_or_extend_deadline(self):
        wait, grace, cost = self.turn(0, 1., .7, (True, False), frequency=.5)
        self.assertAlmostEqual(wait, 1.01)
        self.assertEqual(grace, .7)
        self.assertGreater(cost, 0.)

    def test_opposite_step_clears_but_simultaneous_hop_does_not(self):
        for last, valid in ((0, (False, True)), (1, (True, False))):
            wait, grace, cost = self.turn(last, 4., 1.2, valid)
            self.assertEqual(wait, 0.)
            self.assertEqual(cost, 0.)
        self.assertGreater(self.turn(0, 2., 1.2, (True, True))[2], 0.)

    def test_reward_prefers_alternating_walk_and_is_side_symmetric(self):
        good = trajectory(cfg=C)[0]
        for kind in ('walk', 'drag', 'stand', 'hop', 'chatter'):
            score, steps, _ = trajectory(kind, cfg=C)
            mirrored, other, _ = trajectory(kind, swap=True, cfg=C)
            self.assertAlmostEqual(score, mirrored)
            np.testing.assert_array_equal(steps, other[::-1])
            if kind != 'walk':
                self.assertGreater(good, score, kind)

    def test_wrong_foot_lift_does_not_earn_dense_credit_or_clear_wait(self):
        s = bg.initial(np.zeros(2), progress=True, sequence=True)
        s.update(last_foot=np.asarray(0), turn_wait=np.asarray(2.), turn_grace=np.asarray(1.2))
        s['contact'][:] = False
        nxt, terms, metrics = bg.update(s, np.zeros(2, bool), np.array([.015, 0.]),
            np.array([.03, 0.]), np.zeros(2), np.array([1., 0.]), .01, C)
        self.assertEqual(terms['bilateral_lift_progress'], 0.)
        self.assertEqual(terms['bilateral_forward_progress'], 0.)
        self.assertGreater(nxt['turn_wait'], s['turn_wait'])
        self.assertLess(terms['bilateral_next_wait'], 0.)
        np.testing.assert_array_equal(metrics['expected_next'], [0., 1.])

    def test_legacy_presets_remain_opt_out(self):
        for name in ('cad_v2_gait03', 'cad_v2_gait04', 'cad_v2_gait05', 'cad_v2_gait06', 'cad_v2_gait07'):
            self.assertFalse(get_config(name).bilateral_sequence_enable)
        s = bg.initial(np.zeros(2), progress=True)
        self.assertNotIn('turn_wait', s)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX validation requires SCITAS')
    def test_jitted_update_keeps_state_tree_and_sequence(self):
        import jax
        import jax.numpy as jnp
        s = bg.initial(jnp.zeros(2), jnp, progress=True, sequence=True)
        fn = jax.jit(lambda state: bg.update(state, jnp.ones(2, bool), jnp.zeros(2),
            jnp.zeros(2), jnp.zeros(2), jnp.array([1., 0.]), .01, C, jnp))
        nxt, terms, _ = fn(s)
        self.assertEqual(jax.tree_util.tree_structure(s), jax.tree_util.tree_structure(nxt))
        self.assertEqual(float(terms['bilateral_next_wait']), 0.)


if __name__ == '__main__':
    unittest.main()
