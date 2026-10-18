import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from RLframework.bilateral_gait import drag_cost, point_velocity
from RLframework.config import get_config
from RLframework.walking_curriculum import advance, observe, slip_metrics
from RLframework.tests.test_walking_curriculum import evaluation


class DragTests(unittest.TestCase):
    def test_motion_without_valid_steps_can_open_only_drag_gate(self):
        walking, drag = {}, {}
        for step in (0, 1):
            result = evaluation(step, speed=.15, error=.11, steps=(0, 0))
            walking = observe(walking, result)
            drag = observe(drag, result, require_steps=False)
        self.assertFalse(walking['advancing'])
        self.assertTrue(drag['advancing'])
        self.assertEqual(advance(drag, 1_000_000, 5_000_000)['progress'], .2)

    def test_gate_standing_reverse_falls_tracking_and_nonfinite(self):
        for change in (dict(speed=0.), dict(speed=-.2), dict(seconds=5.), dict(error=.3)):
            state = {}
            for step in range(3):
                state = observe(state, evaluation(step, **change), require_steps=False)
            self.assertFalse(state['advancing'])
        state = observe({}, dict(evaluation(), finite=False), require_steps=False)
        self.assertEqual(state['streak'], 0)

    def test_pause_resume_serialization_and_no_duplicate_evidence(self):
        state = observe({}, evaluation(0, steps=(0, 0)), require_steps=False)
        self.assertEqual(observe(state, evaluation(0), require_steps=False), state)
        state = observe(state, evaluation(1, steps=(0, 0)), require_steps=False)
        state = advance(state, 500_000, 5_000_000)
        state = json.loads(json.dumps(state))
        state = observe(state, evaluation(2, speed=0.), require_steps=False)
        self.assertFalse(state['advancing'])
        self.assertAlmostEqual(advance(state, 9_000_000, 5_000_000)['progress'], .1)
        state = observe(state, evaluation(3), require_steps=False)
        self.assertFalse(state['advancing'])
        state = observe(state, evaluation(4), require_steps=False)
        self.assertEqual(advance(state, 9_000_000, 5_000_000)['progress'], 1.)

    def test_contact_speed_cost_increases_past_old_cap_in_all_directions(self):
        velocities = np.array([[.2, 0, 0], [-.2, 0, 0], [0, .2, 0], [0, -.2, 0]])
        speeds = np.linalg.norm(velocities[:, :2], axis=-1)
        np.testing.assert_allclose(drag_cost(speeds), [3.]*4)
        self.assertEqual(drag_cost(np.array(0.)), 0.)
        self.assertEqual(drag_cost(np.array(.1)), 1.)
        self.assertTrue(np.all(np.diff(drag_cost(np.array([0., .02, .1, .2, .4]))) > 0.))

    def test_rotating_foot_with_stationary_contact_is_not_slip(self):
        # Foot COM moves forward while rolling about a fixed point below it.
        vel = point_velocity(np.array([0., 1., 0., 1., 0., 0.]),
                             np.array([0., 0., -1.]), np.zeros(3))
        np.testing.assert_allclose(vel, 0.)
        self.assertEqual(drag_cost(np.linalg.norm(vel[:2])), 0.)

    def test_metrics_include_terminal_transition_but_exclude_reset(self):
        rows = dict(active=np.array([[1], [1], [0]], bool),
                    quality={'slip': np.array([[[.1, 0.]], [[.2, .04]], [[99., 99.]]])})
        m = slip_metrics(rows, .01)
        np.testing.assert_allclose(m['integrated_slip_m'], [[.003, .0004]])
        np.testing.assert_allclose(m['slip_speed_mean'], [[.15, .02]])
        np.testing.assert_allclose(m['sliding_fraction'], [[1., .5]])

    def test_preset_preserves_physics_thresholds_and_no_pushes(self):
        old, new = get_config('cad_v2_gait06'), get_config('cad_v2_gait07')
        for name in ('bilateral_enable', 'bilateral_progress_enable', 'bilateral_clearance_m',
                     'bilateral_step_min_m', 'bilateral_min_swing_s', 'model_path',
                     'cmd_range', 'gait_eval_targets', 'total_steps', 'shape_scale_start'):
            self.assertEqual(getattr(old, name), getattr(new, name), name)
        self.assertTrue(new.drag_success_curriculum)
        self.assertFalse(new.warmstart_reset_log_std)
        self.assertEqual(new.warmstart_obs_count_cap, 0.)
        self.assertEqual(new.push_interval_s, 0.)
        self.assertFalse(new.dr_enable)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX integration runs on SCITAS')
    def test_independent_ppo_ramps_and_jitted_cost(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ppo import PPO, initial_params
        import jax
        import jax.numpy as jnp
        c = get_config('cad_v2_gait07')
        agent = PPO.__new__(PPO)
        agent.cfg, agent.cur, agent.step = c, {}, 0
        agent.env_params = initial_params(c)
        self.assertEqual(agent.env_params.drag_scale, 0.)
        state = observe({}, evaluation(0, steps=(0, 0)), require_steps=False)
        agent.cur['drag_success'] = observe(state, evaluation(1, steps=(0, 0)), require_steps=False)
        agent.update_curricula(800., 1_000_000)
        self.assertAlmostEqual(agent.env_params.drag_scale, .2)
        self.assertAlmostEqual(agent.env_params.shape_scale, .15)
        result = jax.jit(lambda v: drag_cost(v, xp=jnp))(jnp.array([0., .1, .2]))
        np.testing.assert_allclose(result, [0., 1., 3.])


if __name__ == '__main__':
    unittest.main()
