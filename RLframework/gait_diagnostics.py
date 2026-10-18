"""NumPy-only attempt summaries and the last-continuation decision rule."""
import numpy as np


def attempt_summary(q, live, symmetry, dt):
    """First-episode quality-window measurements; absent bouts have null quantiles."""
    result = {name: (q[name]*live[..., None]).sum(axis=0) for name in (
        'attempt', 'incomplete_step', 'repeated_attempt',
        'rejected_air', 'rejected_clearance', 'rejected_travel')}
    result['symmetry_components'] = symmetry
    event = q['attempt'] & live[..., None]
    for name in ('attempt_length', 'attempt_height', 'attempt_air'):
        result[name+'_quantiles'] = np.asarray([
            [np.quantile(q[name][:, i, f][event[:, i, f]], [.1, .5, .9]).tolist()
             if event[:, i, f].any() else [None]*3 for f in range(2)]
            for i in range(live.shape[1])], dtype=object)
    if 'repeat_air' in q:
        result['repeat_air_seconds'] = (q['repeat_air']*live[..., None]).sum(axis=0)*dt
        expected_air = (~q['contact']) & (q['expected_foot'] > 0) & live[..., None]
        count = np.maximum(expected_air.sum(axis=0), 1)
        result['expected_air_clock_overlap'] = (
            np.where(expected_air, q['reward_swing'], 0.).sum(axis=0)/count)
        result['controller_reward_swing_error'] = (
            np.where(live[..., None], np.abs(q['reward_swing']-q['controller_swing']), 0.).max(axis=(0, 2)))
    return result


def continuation_decision(records):
    """Require sustained measured progress, not extra return or survival alone.

    Inspect the final five evaluations; at least two consecutive evaluations
    must improve repetition AND incompleteness while preserving walking.
    A qualified candidate anywhere is retained as a success. This reports a
    decision; it never launches a fresh run whose objective is not yet implemented.
    """
    if len(records) < 2:
        return dict(decision='insufficient_evaluations')
    baseline = records[0]
    base_repeats = float(np.asarray(baseline['per_trial']['repeated_attempt']).sum(axis=1).mean())
    base_incomplete = float(np.mean(baseline['per_trial']['incomplete_fraction']))
    successful = [int(r['step']) for r in records if r['keeper_eligible']]
    checks = []
    for r in records[1:][-5:]:
        m = r['per_trial']
        repeats = float(np.asarray(m['repeated_attempt']).sum(axis=1).mean())
        incomplete = float(np.mean(m['incomplete_fraction']))
        walking = (r['finite'] and r['survival_fraction'] >= .9 and r['track_error'] <= .05
                   and np.min(np.asarray(m['steps']).mean(axis=0)) >= 8
                   and np.mean(m['contact_slip']) <= .02)
        improved = (base_repeats > 0 and repeats <= .75*base_repeats
                    and incomplete <= base_incomplete-.10)
        checks.append(dict(step=int(r['step']), walking=bool(walking), improved=bool(improved),
                           repeated_attempts=repeats, incomplete_fraction=incomplete,
                           progress=bool(walking and improved)))
    sustained = any(a['progress'] and b['progress'] for a, b in zip(checks, checks[1:]))
    decision = ('qualified_candidate' if successful else 'progress_review_video' if sustained
                else 'fresh_policy_with_explicit_step_objective')
    return dict(decision=decision, baseline_repeated_attempts=base_repeats,
                baseline_incomplete_fraction=base_incomplete, qualified_steps=successful,
                final_evaluation_checks=checks,
                note='Decision from nominal fixed-seed evaluation; visually review candidate and final. No new job launched.')
