"""GPU replay of frozen best policies; never train or modify training records."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--model', choices=['v1', 'v2', 'xshape'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'BalanceRL'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from config import config_from_dict
    from env import EnvParams
    from plant import Override
    from ppo import Trainer

    start = time.monotonic()
    assert all(d.platform == 'gpu' for d in jax.devices())
    manifest = json.loads((root/'source-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    checkpoint = root/'BalanceRL/runs'/f'stand03_{args.model}_s0/best'
    side = json.loads(checkpoint.with_suffix('.json').read_text())
    cfg = dataclasses.replace(config_from_dict(side['config']), n_envs=16, n_steps=8,
                              batch_size=64, est_batch=64)
    args.out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg, args.out/'runtime', seed=0, n_devices=1, n_eval=80)
    trainer.load(checkpoint, with_opt=False)
    env, n, ticks = trainer.eval_env, 80, 2000
    levels = np.repeat([0., .02, .05, .10, .15], 16)
    angles = np.tile(np.arange(16)*2*np.pi/16, 5)
    # Trial indices fixed before simulation. Retain all 80 outcomes, not just clips.
    selected = jnp.array([0, 48])  # quiet and +X, 0.10 m/s at 5 s
    prm = EnvParams(plant_scale=1., push_level=.15, push_on=1.)
    ov = Override(push_dv=jnp.asarray(levels), push_az=jnp.asarray(angles), push_el=0.)
    state, obs = env.reset(jax.random.PRNGKey(1234), prm, ov)
    print(f'[{args.model}] loaded best {trainer.step:,}; compiling 80 first-episode replays', flush=True)

    @jax.jit
    def replay(state, obs):
        def tick(carry, _):
            state, obs, alive, ok, quiet, ended, fell, stepped = carry
            # Record pre-step states, so terminal auto-resets cannot appear in a clip.
            frame = dict(qpos=state.data.qpos[selected], qvel=state.data.qvel[selected],
                         force=state.data.xfrc_applied[selected, env.plant.base_bid, :3],
                         alive=alive[selected], quiet_s=state.standing['quiet_s'][selected])
            action = trainer._greedy(trainer.params, trainer.stats.normalize(obs))
            state, obs, _, done, info = env.step(state, jnp.clip(action, -1., 1.), prm)
            ok = ok | (alive & info['push_ok'])
            quiet = jnp.where(alive, info['quiet_ok'], quiet)
            ended = jnp.where(alive & done, info['ep_len'], ended)
            fell = fell | (alive & info['fallen'])
            stepped = stepped | (alive & info['task_violation'])
            return (state, obs, alive & ~done, ok, quiet, ended, fell, stepped), frame
        initial = (state, obs, jnp.ones(n, bool), jnp.zeros(n, bool), jnp.zeros(n, bool),
                   jnp.full(n, ticks), jnp.zeros(n, bool), jnp.zeros(n, bool))
        return jax.lax.scan(tick, initial, None, length=ticks)

    final, trace = jax.device_get(replay(state, obs))
    _, _, alive, ok, quiet, ended, fell, stepped = final
    passed = alive & np.where(levels == 0, quiet, ok)
    rates = {f'{level:.2f}': float(passed[levels == level].mean()) for level in (0., .02, .05, .10, .15)}
    model_path = root/'BalanceRL'/cfg.model_path
    meta = dict(model=args.model, training_job=f'3246822_{("v1","v2","xshape").index(args.model)}',
        checkpoint='best.msgpack', checkpoint_step=trainer.step,
        checkpoint_sha256=hashlib.sha256(checkpoint.with_suffix('.msgpack').read_bytes()).hexdigest(),
        model_path=cfg.model_path, model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        seed=1234, batch_size=80, control_dt=.01, ticks=ticks,
        selected_trials=[0, 48], selected_levels=[0., .10],
        protocol='Same 80-trial evaluation: nominal plant, full sensor noise, 20 s, first episodes only, one push at 5 s. Fixed trial 0/48 selected before outcomes.',
        archived_score=side['eval']['score'], replay_score=float(passed.mean()), rates=rates,
        archived_rates=side['eval']['survival'],
        per_trial=dict(level=levels.tolist(), duration_s=(ended*.01).tolist(), passed=passed.tolist(),
                       physical_fall=fell.tolist(), step_violation=stepped.tolist()),
        elapsed_s=time.monotonic()-start, device=[str(d) for d in jax.devices()])
    for key in ('qpos', 'qvel'):
        assert all(np.isfinite(trace[key][:, i][trace['alive'][:, i]]).all() for i in range(2))
    np.savez_compressed(args.out/'trace.npz', **trace, meta=json.dumps(meta))
    (args.out/'replay.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ('model','checkpoint_step','archived_score','replay_score','rates','elapsed_s')}), flush=True)


if __name__ == '__main__':
    main()
