import json
import unittest
import importlib.util
import sys
from pathlib import Path
import numpy as np
from RLframework.config import get_config
from RLframework.walking_curriculum import advance, observe, early_metrics
from RLframework.gait_selection import candidate_rank


def evaluation(step=0, seconds=8., speed=.22, error=.05, steps=(2, 0)):
    return dict(step=step, finite=True, per_trial=dict(
        first_episode_seconds=[seconds]*12, early_forward_speed=[speed]*12,
        early_track_error=[error]*12, early_steps=[steps]*12))


class WalkingCurriculumTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX integration runs on SCITAS')
    def test_ppo_consumes_gate_and_keeps_exploration_until_success(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ppo import PPO, initial_params
        c = get_config('cad_v2_gait06')
        agent = PPO.__new__(PPO)
        agent.cfg, agent.cur, agent.step = c, {}, 10_000_000
        agent.env_params = initial_params(c)
        agent.update_curricula(200., 1_000_000)
        agent.update_entropy(.9, 200.)
        self.assertAlmostEqual(agent.env_params.shape_scale, .15)
        self.assertAlmostEqual(agent.ent_coef, .01)
        state = observe(agent.cur['walking_success'], evaluation(0))
        agent.cur['walking_success'] = observe(state, evaluation(1))
        agent.update_curricula(800., 1_000_000)
        agent.update_entropy(.9, 800.)
        self.assertAlmostEqual(agent.env_params.shape_scale, .32)
        self.assertAlmostEqual(agent.ent_coef, .0084)
        agent.cur['walking_success'] = observe(agent.cur['walking_success'], evaluation(2, seconds=2.))
        agent.update_curricula(200., 1_000_000)
        self.assertAlmostEqual(agent.env_params.shape_scale, .32)

    def test_standing_sliding_falling_and_wrong_speed_do_not_open(self):
        for kwargs in (dict(speed=0.), dict(steps=(0, 0)), dict(seconds=2.), dict(error=.3)):
            state = {}
            for step in range(5):
                state = observe(state, evaluation(step, **kwargs))
                state = advance(state, 1_000_000, 5_000_000)
            self.assertFalse(state['advancing'])
            self.assertEqual(state['progress'], 0.)

    def test_repeated_success_gradual_ramp_and_pause(self):
        state = observe({}, evaluation())
        self.assertFalse(state['advancing'])
        self.assertEqual(observe(state, evaluation()), state)  # duplicate is not new evidence
        state = observe(state, evaluation(1))
        self.assertTrue(state['advancing'])  # imperfect bilateral stepping can earn shaping
        state = advance(state, 1_000_000, 5_000_000)
        self.assertAlmostEqual(state['progress'], .2)
        state = json.loads(json.dumps(state))  # checkpoint metadata round trip
        state = observe(state, evaluation(2, seconds=2.))
        self.assertFalse(state['advancing'])
        self.assertAlmostEqual(advance(state, 10_000_000, 5_000_000)['progress'], .2)
        state = observe(state, evaluation(3))
        self.assertFalse(state['advancing'])
        state = observe(state, evaluation(4))
        self.assertEqual(advance(state, 10_000_000, 5_000_000)['progress'], 1.)

    def test_early_metrics_ignore_auto_reset_and_report_censoring_duration(self):
        rows = dict(active=np.array([[1, 1], [1, 1], [0, 1], [0, 1]], bool),
                    speed=np.array([[.2, .2], [.2, .2], [99., .2], [99., .2]]),
                    error=np.ones((4, 2))*.05,
                    quality={'valid_step': np.ones((4, 2, 2))})
        for name in ('term_low', 'term_tip', 'term_floor', 'term_ws', 'term_nan'):
            rows[name] = np.zeros((4, 2), bool)
        rows['term_low'][1, 0] = True
        rows['term_nan'][3, 0] = True  # another episode must not contaminate reasons
        m = early_metrics(rows, .01, window_s=.03)
        np.testing.assert_allclose(m['first_episode_seconds'], [.02, .04])
        np.testing.assert_allclose(m['early_forward_speed'], [.2, .2])
        np.testing.assert_array_equal(m['steps_from_reset'], [[2, 2], [4, 4]])
        np.testing.assert_array_equal(m['early_steps'], [[2, 2], [3, 3]])
        np.testing.assert_array_equal(m['term_low'], [True, False])
        self.assertFalse(m['term_nan'].any())

    def test_early_ranking_preserves_longer_attempt_without_certifying_it(self):
        from RLframework.tests.test_gait_selection import result
        a, b = result(survived=False), result(survived=False)
        a['per_trial']['first_episode_seconds'] = [2.]
        b['per_trial']['first_episode_seconds'] = [4.]
        self.assertGreater(candidate_rank(b, early=True), candidate_rank(a, early=True))
        self.assertEqual(b['qualified_fraction'], 0.)

    def test_fresh_settings_keep_bilateral_terms_and_nonzero_start(self):
        c = get_config('cad_v2_gait06')
        self.assertTrue(c.bilateral_enable and c.bilateral_progress_enable)
        self.assertTrue(c.walking_success_curriculum)
        self.assertEqual(c.shape_scale_start, .15)
        self.assertEqual(c.learning_rate, 3e-4)
        self.assertEqual(c.lr_warmup_updates, 0)
        self.assertEqual(c.ent_coef, .01)
        self.assertEqual(c.ent_anneal_deadline_steps, 0)
        self.assertEqual(c.total_steps, 15_000_000)
        self.assertEqual(c.v_max*c.cmd_range[0], .25)
        self.assertFalse(c.hold_enable or c.bringup_enable)


if __name__ == '__main__':
    unittest.main()
