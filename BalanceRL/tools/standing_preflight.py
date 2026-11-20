"""Finite MJX standing and two PPO updates before the three-CAD campaign."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['v1', 'v2', 'xshape'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from config import get_config
    from env import BalanceEnv, EnvParams
    from ppo import Trainer
    start = time.monotonic()
    cfg = get_config('stand_'+args.model, n_envs=16, n_steps=8, batch_size=64, est_batch=64,
                     n_epochs=1, est_epochs=1, total_steps=256)
    assert all(d.platform == 'gpu' for d in jax.devices())
    env = BalanceEnv(cfg, 16)
    state, obs = env.reset(jax.random.PRNGKey(123), EnvParams(0., .02, 0.))
    @jax.jit
    def hold(state, obs):
        def tick(carry, _):
            state, obs, alive = carry
            state, obs, reward, done, info = env.step(state, jnp.zeros((16, 18)), EnvParams(0., .02, 0.))
            finite = jnp.isfinite(obs).all() & jnp.isfinite(reward).all() & jnp.isfinite(state.data.qpos).all()
            return (state, obs, alive & ~done), finite
        return jax.lax.scan(tick, (state, obs, jnp.ones(16, bool)), None, length=200)
    result, finite = jax.device_get(hold(state, obs))
    assert np.asarray(finite).all(), 'nonfinite zero-action physics'
    survival = float(result[2].mean())
    assert survival >= .8, f'2 s nominal fixed-PD survival only {survival:.3f}'
    # Fresh trainer; no preflight weights or optimizer are reused by the real run.
    trainer = Trainer(cfg, args.out.parent/('smoke_'+args.model), seed=0, n_devices=1, n_eval=0)
    logs = [trainer.iterate()[0] for _ in range(2)]
    assert all(np.isfinite(r['rollout/reward_mean']) and np.isfinite(r['train/kl']) for r in logs)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(trainer.params))
    record = dict(passed=True, model=args.model, fixed_PD_2s_survival=survival,
                  preflight_steps=trainer.step, elapsed_s=time.monotonic()-start,
                  updates=[{k:r[k] for k in ('rollout/reward_mean','train/kl','rollout/task_violations')} for r in logs],
                  device=[str(d) for d in jax.devices()],
                  note='Temporary 16x8 smoke; real run reinitializes all weights and optimizer.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
