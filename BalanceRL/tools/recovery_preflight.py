"""Load the correct parent, test recovery evaluation and a short finite PPO pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['v1', 'v2', 'xshape'], required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import jax
    import numpy as np
    from config import get_config
    from ppo import Trainer
    import recovery_eval
    start = time.monotonic()
    assert all(d.platform == 'gpu' for d in jax.devices())
    cfg = get_config('recover_'+args.model, n_envs=32, n_steps=64,
                     batch_size=1024, est_batch=1024, n_epochs=1, est_epochs=1)
    trainer = Trainer(cfg, args.out.parent/('pilot_'+args.model), seed=1, n_devices=1, n_eval=80)
    parent = trainer.initialize_from(args.parent)
    assert trainer.step == trainer.rollout_n == trainer.n_upd == 0
    assert trainer.cur['push_level'] == .1 and trainer.cur['push_stage'] == 0
    print(f'[{args.model}] parent loaded, fresh optimizer/counters; baseline full-variety evaluation', flush=True)
    baseline = trainer.evaluate(seed=1234)
    assert baseline['quiet_rate'] >= .5, f'parent quiet-standing regression {baseline["quiet_rate"]}'
    print(f'[{args.model}] baseline quiet {baseline["quiet_rate"]:.3f}; direction-balanced probe', flush=True)
    probe = recovery_eval.evaluate(trainer, seed=910000, probe=True)
    assert probe['hard_trials'] == 200 and len(probe['sector_rates']) == 8
    assert all(np.isfinite(probe[k]) for k in ('score', 'hard_rate', 'quiet_rate'))
    logs = []
    print(f'[{args.model}] 32-update PPO pilot; weights will be discarded', flush=True)
    for i in range(32):
        row, _ = trainer.iterate()
        assert all(np.isfinite(row[k]) for k in ('rollout/reward_mean', 'train/kl', 'train/vf'))
        logs.append({k:row[k] for k in ('time/env_steps','rollout/reward_mean','train/kl',
                                      'rollout/recovery_steps','rollout/recovery_timeouts')})
        if i % 8 == 7:
            print(f'[{args.model}] pilot {i+1}/32, {trainer.step} transitions', flush=True)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(trainer.params))
    record = dict(passed=True, model=args.model, parent=parent,
        baseline=baseline, probe=probe, pilot_steps=trainer.step, pilot_logs=logs,
        elapsed_s=time.monotonic()-start, device=[str(d) for d in jax.devices()],
        note='Pilot optimizer/weights discarded; full run starts from frozen STAND-03 best again.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('passed','model','pilot_steps','elapsed_s')}), flush=True)


if __name__ == '__main__':
    main()
