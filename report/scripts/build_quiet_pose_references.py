"""Select a feasible recorded quiet posture from each STAND-04 best policy."""
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'BalanceRL/tests'))
from test_standing import load
from standing_math import sole_centers
from pose_reference import features, error
from config import get_config


def main():
    audit = {}
    for model in ('v1','v2','xshape'):
        folder = ROOT/'report/data/standing04/results'/model
        trace_path = folder/'trace.npz'
        with np.load(trace_path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta'].item()))
            assert meta['selected_trials'][0] == 0 and meta['per_trial']['passed'][0]
            times = np.arange(z['qpos'].shape[0])*meta['control_dt']
            ids = np.flatnonzero((times >= 10.) & (times < 20.) & z['alive'][:,0]
                                 & (z['quiet_s'][:,0] >= 1.))[::10]
            assert len(ids) >= 20, 'insufficient validated quiet samples'
            qpos = z['qpos'][ids,0].copy()
            streak = z['quiet_s'][ids,0].copy()
        _, m, d, plant = load(model)
        bid = m.body('torso').id
        qadr = np.array([m.joint(n).qposadr[0] for n in
                        ('hip_roll_L','cam_L','thigh_L','hip_roll_R','cam_R','thigh_R')])
        samples = []
        for q in qpos:
            d.qpos[:] = q
            mujoco.mj_forward(m,d)
            samples.append(features(d.qpos[qadr].copy(), d.xmat[bid].reshape(3,3),
                                     d.xpos[bid], sole_centers(d,plant)))
        matrix = np.array([np.r_[s['motor_q']/.05,s['gravity_body']/.05,
            s['soles_body'].ravel()/.02,s['height_over_soles']/.01] for s in samples])
        # Choose an actual sample nearest the mean; do not average closed-loop qpos.
        chosen = int(np.argmin(np.mean((matrix-matrix.mean(0))**2,axis=1)))
        ref = samples[chosen]
        ref_json = {k:np.asarray(v).tolist() for k,v in ref.items()}
        cfg = get_config('return_'+model)
        matched = [bool(error(s,ref,cfg)['ok']) for s in samples]
        assert np.mean(matched) >= .95, 'standing reference is not representative'
        mp = ROOT/f'BalanceRL/model/stand_{model}.xml'
        record = dict(model=model,source='STAND-04 best, independently replayed successful quiet trial 0',
            checkpoint_step=meta['checkpoint_step'],checkpoint_sha256=meta['checkpoint_sha256'],
            trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
            candidate_window_s=[10.,20.],quiet_streak_min_s=1.,candidate_samples=len(samples),
            selected_trace_tick=int(ids[chosen]),selected_time_s=float(times[ids[chosen]]),
            selected_quiet_streak_s=float(streak[chosen]),candidate_match_fraction=float(np.mean(matched)),
            target_semantics='Posture at any recovered floor location; no global XY or heading target',
            features=ref_json,reference_qpos_for_visualization_only=qpos[chosen].tolist())
        (ROOT/f'BalanceRL/model/quiet_{model}.json').write_text(json.dumps(record,indent=2)+'\n')
        audit[model] = {k:record[k] for k in ('checkpoint_step','selected_time_s','selected_quiet_streak_s','candidate_samples','candidate_match_fraction')}
    out = ROOT/'report/data/standing05'
    out.mkdir(parents=True,exist_ok=True)
    (out/'quiet-reference-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
