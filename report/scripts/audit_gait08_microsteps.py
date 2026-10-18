"""Reproduce GAIT-08 reward loopholes without physics or training changes.

Run from the repository root: .venv/Scripts/python report/scripts/audit_gait08_microsteps.py
Synthetic inputs isolate reward bookkeeping; they are not simulated robot rollouts.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from RLframework import bilateral_gait as bg
from RLframework.config import get_config


def bouts(foot, distance=.02, height=.006, count=3):
    c = get_config('cad_v2_gait08')
    s = bg.initial(np.zeros(2), progress=True, sequence=True)
    # Left previously completed a valid step, so right is required next.
    s.update(last_foot=np.asarray(0), turn_grace=np.asarray(1.2))
    totals, steps, landings = {}, np.zeros(2, int), 0
    x = np.zeros(2)
    for _ in range(count):
        origin = x.copy()
        # 120 ms airborne, then 40 ms on the ground. Below distance/height
        # thresholds in the microstep case, but above the debounce duration.
        for tick in range(16):
            contact = np.ones(2, bool)
            h, swing = np.zeros(2), np.zeros(2)
            if tick < 12:
                contact[foot] = False
                h[foot], swing[foot] = height, 1.
                x[foot] = origin[foot] + distance * tick / 11.
            old_contact = s['contact'].copy()
            s, terms, m = bg.update(s, contact, h, x, np.zeros(2), swing,
                                    .01, c, frequency_hz=.5)
            landings += int(np.sum(~old_contact & s['contact']))
            steps += m['valid_step'].astype(int)
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.) + float(value)
    return dict(debounced_landings=landings, valid_steps=steps.tolist(),
                last_valid_foot=int(s['last_foot']),
                raw_term_sums=totals,
                note='Isolated bilateral terms before shape/dt scaling; not total environment return.')


def main():
    c = get_config('cad_v2_gait08')
    manifest = json.loads((ROOT/'docs/experiments/gait08_20261017_v1/source-manifest.json').read_text())
    matches = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == manifest['files'][name]
               for name in ('RLframework/config.py', 'RLframework/env.py',
                            'RLframework/bilateral_gait.py', 'RLframework/gait_selection.py')}
    assert all(matches.values()), matches
    wrong = bouts(0)
    expected = bouts(1)
    valid_repeat = bouts(0, distance=.06, height=.02, count=1)
    assert wrong['debounced_landings'] == expected['debounced_landings'] == 3
    assert wrong['valid_steps'] == expected['valid_steps'] == [0, 0]
    assert wrong['raw_term_sums']['bilateral_repeat'] == 0
    assert expected['raw_term_sums']['bilateral_forward_progress'] > 0
    assert expected['raw_term_sums']['bilateral_next_wait'] == 0
    assert valid_repeat['raw_term_sums']['bilateral_repeat'] == -c.bilateral_repeat_weight
    # A small liftoff can change the stance sample even without a valid step.
    s = bg.initial(np.zeros(2), progress=True, sequence=True)
    s.update(stance=np.array([.03, 1.]), last_stance=np.ones(2),
             last_air=np.full(2, .2), last_length=np.full(2, .1),
             seen=np.ones(2, bool), debounce=np.array([1., 0.]))
    s, _, m = bg.update(s, np.array([False, True]), np.array([.006, 0.]),
                        np.zeros(2), np.zeros(2), np.zeros(2), .01, c)
    assert not m['valid_step'].any()
    assert s['last_stance'][0] == .03 and s['last_air'][0] == .2
    result = dict(date='2026-10-18', source_matches_training_snapshot=matches,
                  wrong_foot_microsteps=wrong, expected_foot_microsteps=expected,
                  valid_repeat_control=valid_repeat,
                  mixed_history_symmetry=m['symmetry'].tolist(),
                  nominal_stance_at_half_hz_s=c.stance_ratio_final/.5,
                  slow_stance_penalty_starts_s=c.stance_cap_slow_s,
                  limitations='Synthetic reward-unit checks, not proof of the video motion cause or net reward advantage.')
    out = ROOT/'report/tables/gait08_microstep_audit.json'
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
