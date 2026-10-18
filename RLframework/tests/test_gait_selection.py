import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict
from RLframework.config import get_config
from RLframework.gait_selection import candidate_rank, retain_evaluation


def result(right=0, clear=0., alt=0., survived=True, qualified=0., error=.03):
    return dict(step=100, qualified_fraction=qualified,
                survival_fraction=float(survived), track_error=error,
                per_trial=dict(steps=[[10, right]], clear_fraction=[[.12, clear]],
                               alternation=[alt], survived=[survived]))


class GaitSelectionTests(unittest.TestCase):
    def test_attempt_quality_breaks_tie_before_tracking(self):
        clean, messy = result(10, .1, 1., error=.04), result(10, .1, 1., error=.01)
        for r, fraction in ((clean, .05), (messy, .4)):
            r['per_trial'].update(incomplete_fraction=[fraction], repeated_attempt=[[0, 0]],
                                  max_symmetry=[.3])
        self.assertGreater(candidate_rank(clean), candidate_rank(messy))

    def test_bilateral_progress_beats_fast_dragging(self):
        self.assertGreater(candidate_rank(result(5, .05, .75, error=.04)),
                           candidate_rank(result(error=.01)))

    def test_fallen_progress_does_not_beat_surviving_policy(self):
        self.assertGreater(candidate_rank(result()),
                           candidate_rank(result(10, .15, 1., survived=False)))

    def test_qualification_takes_precedence(self):
        self.assertGreater(candidate_rank(result(3, .08, .9, qualified=1., error=.09)),
                           candidate_rank(result(10, .15, 1., qualified=0.)))

    def test_each_evaluated_policy_is_retained_with_its_metrics(self):
        class Agent:
            def save(self, path):
                path.write_bytes(b'policy')
        with tempfile.TemporaryDirectory() as d:
            for step in (0, 1048576, 15007744):
                r = dict(result(), step=step)
                retain_evaluation(Agent(), r, d)
                self.assertEqual((Path(d)/r['checkpoint']).read_bytes(), b'policy')
                m = json.loads((Path(d)/f'eval_{step}_metrics.json').read_text())
                self.assertEqual(m['step'], step)
                self.assertEqual(m['checkpoint'], r['checkpoint'])

    def test_preset_preserves_reward_model_and_budget(self):
        old, new = get_config('cad_v2_gait04'), get_config('cad_v2_gait05')
        changed = {k for k, v in asdict(old).items() if v != asdict(new)[k]}
        self.assertEqual(changed, {'cmd_range', 'cmd_range_start', 'eval_ladder',
                                  'gait_eval_targets', 'gait_rank_bilateral', 'gait_save_evaluated'})
        self.assertEqual(new.total_steps, 15_000_000)
        self.assertEqual(new.v_max*new.cmd_range[0], .25)
        self.assertEqual(new.gait_eval_targets, (.2, .25, .35))
        self.assertFalse(new.freeze_obs_stats)


if __name__ == '__main__':
    unittest.main()
