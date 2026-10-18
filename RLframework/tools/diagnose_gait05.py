"""Short paired, first-episode diagnosis against an immutable GAIT-05 snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.run.resolve().parents[1]))
    import jax
    import jax.numpy as jnp
    from evaluate import load_run
    from env import EnvParams
    from ppo import ObsStats

    cfg, env, agent = load_run(args.run, args.run/'eval_0.msgpack', n_envs=36)
    params = EnvParams.final(cfg)._replace(dr_scale=0., ctrl_jitter_ms=0., ctrl_drop_prob=0.)
    targets = np.repeat([.20, .25, .35], 12)
    reason_names = ('term_low', 'term_tip', 'term_floor', 'term_ws', 'term_nan')
    p = env.plant

    @jax.jit
    def simulate(weights, mean, var, count, zero_action):
        stats = ObsStats(mean=mean, var=var, count=count)
        state, obs = env.reset(jax.random.PRNGKey(3103), params)
        state = state.replace(v_cmd=jnp.asarray(targets, jnp.float32),
                              cmd_left=jnp.full_like(state.cmd_left, 1e9))
        once = jax.vmap(env._once, in_axes=(0, None, 0, 0))(
            state, params, state.phase, state.commit.astype(jnp.float32))
        start = state.hist[:, env.hist_idx].reshape((env.n_envs, -1)).shape[1]
        obs = obs.at[:, start:start+once.shape[1]].set(once)

        def tick(carry, _):
            s, ob = carry
            raw = agent._act_greedy(weights, stats.normalize(ob))
            action = jnp.where(zero_action, jnp.zeros_like(raw), jnp.clip(raw, -1., 1.))
            nxt, ob2, reward, done, info = env.step(s, action, params)
            row = dict(qpos=s.data.qpos, qvel=s.data.qvel, action=action,
                       raw_action=raw, done=done, fallen=info['fallen'], reward=reward,
                       terms=info['reward_terms'], flags=jnp.stack([info[k] for k in reason_names], axis=-1),
                       upright=info['stop_upright'], speed=info['v_body_x'],
                       flight=info['flight_fraction'], freq=info['freq_hz'],
                       torque_util=info['torque_util'], floor_hit=info['reward_floor_hit'],
                       quality=info['bilateral'],
                       base=s.data.xpos[:, p.base_bid],
                       com=s.data.subtree_com[:, p.base_bid],
                       ws=jax.vmap(env._workspace_out)(s.data))
            return (nxt, ob2), row
        return jax.lax.scan(tick, (state, obs), None, length=600)[1]

    cases = [('fresh', args.run/'eval_0.msgpack', False),
             ('step5M', args.run/'eval_5242880.msgpack', False),
             ('final', args.run/'final.msgpack', False),
             ('gait04_14M', args.comparison, False),
             ('zero_action', args.run/'eval_0.msgpack', True)]
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, ck, zero in cases:
        agent.load(ck)
        r = jax.device_get(simulate(agent.params, agent.stats.mean, agent.stats.var, agent.stats.count,
                                   jnp.asarray(zero)))
        counts = np.where(r['done'].any(axis=0), r['done'].argmax(axis=0)+1, len(r['done']))
        live = np.arange(600)[:, None] < counts[None, :]
        last = counts-1
        final_flags = r['flags'][last, np.arange(36)]
        def avg(x, mask):
            return float(np.asarray(x)[mask].mean())
        groups = {}
        for target in (.20, .25, .35):
            take = np.isclose(targets, target)
            mask = live & take[None, :]
            i = int(np.flatnonzero(take)[0])
            sample_ticks = sorted(set([0, min(50, int(last[i])), min(100, int(last[i])), int(last[i])]))
            groups[f'{target:.2f}'] = dict(
                survived_6s=int((~r['done'][:, take].any(axis=0)).sum()),
                seconds=(counts[take]*env.control_dt).tolist(),
                reasons={reason:int(final_flags[take, k].sum()) for k,reason in enumerate(reason_names)},
                terminal_upright=r['upright'][last, np.arange(36)][take].tolist(),
                last_pretermination_base_height=r['base'][last, np.arange(36), 2][take].tolist(),
                steps_from_reset=(r['quality']['valid_step']*mask[..., None]).sum(axis=(0,1)).tolist(),
                reward_mean=avg(r['reward'],mask), floor_fraction=avg(r['floor_hit'],mask),
                freq_mean=avg(r['freq'],mask), torque_util=avg(r['torque_util'],mask),
                residual_saturation=float((np.abs(r['action'][..., -6:][mask])>=.95).mean()),
                terms={k:avg(v,mask) for k,v in r['terms'].items()},
                first_trial_samples=[dict(seconds=t*env.control_dt, base=r['base'][t,i].tolist(),
                    roll_pitch_yaw=r['qpos'][t,i,[p.base_q[k] for k in ('roll','pitch','yaw')]].tolist(),
                    upright=float(r['upright'][t,i]), speed=float(r['speed'][t,i]),
                    contact=r['quality']['contact'][t,i].tolist(),
                    foot_clearance=r['quality']['clearance'][t,i].tolist(),
                    workspace_out=r['ws'][t,i].tolist()) for t in sample_ticks])
        summary = dict(checkpoint=str(ck), step=agent.step, zero_action=zero,
                       sha256=hashlib.sha256(ck.read_bytes()).hexdigest(), groups=groups,
                       seed=3103, seconds=6, reward_scale='final shaping; does not alter policy or physics',
                       note='36 paired nominal trials. First episode only; pre-state poses precede reported transition metrics by one tick. Zero action means default gait/PD, not zero torque.')
        summaries[name] = summary
        (args.out/(name+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
        # Three first trials, one per speed, in the existing video-renderer format.
        ids = [0,12,24]
        meta = dict(step=agent.step, seed=3103, dt=env.control_dt, model=cfg.model_path,
                    label='GAIT-05 diagnosis '+name, checkpoint_sha256=summary['sha256'],
                    trials=[dict(target=float(targets[i]),ticks=int(counts[i]),
                         seconds=float(counts[i]*env.control_dt),
                         fell=bool(r['fallen'][:counts[i],i].any())) for i in ids])
        np.savez_compressed(args.out/(name+'_trace.npz'),meta=json.dumps(meta),
                            qpos=r['qpos'][:,ids],vx=r['speed'][:,ids],flight=r['flight'][:,ids])
        print(name, json.dumps({k: {a:v[a] for a in ('survived_6s','seconds','reasons')} for k,v in groups.items()}), flush=True)
    (args.out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__ == '__main__':
    main()
