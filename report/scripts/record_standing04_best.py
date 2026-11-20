"""Reproduce frozen recovery evaluation and retain preselected best-policy videos."""
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
    ap.add_argument('--model', choices=['v1','v2','xshape'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve()
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
    for name, digest in json.loads((root/'source-manifest.json').read_text())['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    ckpt = root/'BalanceRL/runs'/f'stand04_{a.model}_s1/best'
    side = json.loads(ckpt.with_suffix('.json').read_text())
    cfg = dataclasses.replace(config_from_dict(side['config']), n_envs=16, n_steps=8,
                              batch_size=64, est_batch=64)
    a.out.mkdir(parents=True, exist_ok=True)
    agent = Trainer(cfg, a.out/'runtime', seed=1, n_devices=1, n_eval=80)
    agent.load(ckpt, with_opt=False)
    env, n, ticks = agent.eval_env, 80, 2000
    levels = np.repeat([0., .10, .15, .20, .30], 16)
    angles = np.tile(np.arange(16)*2*np.pi/16, 5)
    selected = jnp.array([0, 48])  # quiet, +X horizontal component at 0.20 m/s
    prm = EnvParams(1., float(agent.cur['push_level']), 1., 3.)
    kr, _, _ = jax.random.split(jax.random.PRNGKey(1234), 3)
    state, obs = env.reset(kr, prm, Override(push_dv=jnp.asarray(levels), push_az=jnp.asarray(angles)))
    print(f'[{a.model}] loaded best +{agent.step:,}; full-variety 80-trial replay', flush=True)

    @jax.jit
    def replay(state, obs):
        def tick(carry, _):
            state, obs, alive, ok, quiet, ended, fell, bad, steps, path, settle, torque = carry
            frame = dict(qpos=state.data.qpos[selected], qvel=state.data.qvel[selected],
                force=state.data.xfrc_applied[selected, env.plant.base_bid, :3], alive=alive[selected],
                quiet_s=state.standing['quiet_s'][selected], steps=steps[selected])
            action = agent._greedy(agent.params, agent.stats.normalize(obs))
            state, obs, _, done, info = env.step(state, jnp.clip(action,-1.,1.), prm)
            frame['push_start'] = (alive & info['push_start'])[selected]
            ok = ok | (alive & info['push_ok'])
            quiet = jnp.where(alive, info['quiet_ok'], quiet)
            ended = jnp.where(alive & done, info['ep_len'], ended)
            fell = fell | (alive & info['fallen'])
            bad = bad | (alive & info['task_violation'])
            steps += jnp.where(alive, info['step_event'], 0)
            path += jnp.where(alive, info['path_increment'], 0.)
            settle = jnp.where(alive & info['push_ok'], info['settling_time'], settle)
            torque = jnp.maximum(torque, jnp.where(alive, info['torque_util'], 0.))
            return (state,obs,alive & ~done,ok,quiet,ended,fell,bad,steps,path,settle,torque), frame
        initial = (state,obs,jnp.ones(n,bool),jnp.zeros(n,bool),jnp.zeros(n,bool),
            jnp.full(n,ticks),jnp.zeros(n,bool),jnp.zeros(n,bool),jnp.zeros(n,jnp.int32),
            jnp.zeros(n),jnp.full(n,-1.),jnp.zeros(n))
        return jax.lax.scan(tick, initial, None, length=ticks)

    final, trace = jax.device_get(replay(state, obs))
    _,_,alive,ok,quiet,ended,fell,bad,steps,path,settle,torque = final
    passed = alive & quiet & ((levels == 0) | ok)
    rates = {f'{v:.2f}':float(passed[levels == v].mean()) for v in (0.,.10,.15,.20,.30)}
    pulses = []
    for i in range(2):
        hits = np.flatnonzero(trace['push_start'][:,i])
        active = np.flatnonzero(np.linalg.norm(trace['force'][:,i],axis=1)>1e-5)
        pulses.append(dict(onset_s=float(hits[0]*.01) if len(hits) else None,
            sampled_duration_s=float(len(active)*.01),
            force_n=trace['force'][active[0],i].tolist() if len(active) else [0.,0.,0.]))
    mp = root/'BalanceRL'/cfg.model_path
    meta = dict(model=a.model, training_job=f'3247599_{("v1","v2","xshape").index(a.model)}',
        checkpoint='best.msgpack', checkpoint_step=agent.step, parent_step=side['cur']['parent_step'],
        checkpoint_sha256=hashlib.sha256(ckpt.with_suffix('.msgpack').read_bytes()).hexdigest(),
        model_path=cfg.model_path, model_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
        seed=1234,batch_size=80,control_dt=.01,ticks=ticks,selected_trials=[0,48],
        selected_levels=[0.,.20],selected_pulses=pulses,stage=3,
        protocol='Frozen STAND-04 evaluator: stage3, nominal plant, full sensor noise, first20s, final5s quiet; trials0/48 fixed before outcomes.',
        archived_score=side['eval']['score'],replay_score=float(passed.mean()),rates=rates,
        archived_rates=side['eval']['survival'],
        per_trial=dict(level=levels.tolist(),duration_s=(ended*.01).tolist(),passed=passed.tolist(),
            physical_fall=fell.tolist(),task_violation=bad.tolist(),steps=steps.tolist(),
            sole_path_m=path.tolist(),settling_s=settle.tolist(),max_mean_torque_util=torque.tolist()),
        elapsed_s=time.monotonic()-start,device=[str(d) for d in jax.devices()])
    for k in ('qpos','qvel'):
        assert all(np.isfinite(trace[k][:,i][trace['alive'][:,i]]).all() for i in range(2))
    np.savez_compressed(a.out/'trace.npz',**trace,meta=json.dumps(meta))
    (a.out/'replay.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ('model','checkpoint_step','archived_score','replay_score','rates','elapsed_s')}),flush=True)


if __name__ == '__main__':
    main()
