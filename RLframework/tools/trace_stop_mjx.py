"""Record an exact MJX walk/STOP/restart evaluation for offline replay and diagnosis.

Runs on a compute node, without changing any training code or checkpoint.
qpos is sampled BEFORE each control tick; reward/termination describe that tick.
Ignore every sample after the first done for each environment (auto-reset).
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-envs', type=int, default=32)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo / 'RLframework'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from evaluate import load_run
    from env import EnvParams
    from stop_metrics import protocol, summarize

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.n_envs, dr=False)
    speeds, starts, ends, ticks = protocol(env.n_envs, env.control_dt)
    if cfg.stop_adapter:
        from stop02_metrics import protocol as protocol02
        from stop_adapter import prepare
        speeds, starts, ends, ticks = protocol02(env.n_envs, env.control_dt, cfg)
    p = EnvParams.final(cfg)._replace(dr_scale=0.0, bringup_scale=0.0,
                                     bringup_p_drop=0.0, bringup_p_held=0.0,
                                     ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0)
    reasons = ('term_low', 'term_tip', 'term_floor', 'term_ws', 'term_nan')

    def run(weights, stats):
        state, obs = env.reset(jax.random.PRNGKey(4102), p)
        if cfg.stop_adapter:
            state = state.replace(stop_previous=jnp.full_like(state.stop_previous, cfg.stop_standing_stage),
                                  effective_speed=jnp.asarray(speeds,jnp.float32),
                                  effective_hold=jnp.full_like(state.effective_hold, cfg.stop_standing_stage))
            obs = obs.at[:,env.actor_dim-3].set(speeds/cfg.v_max)
            obs = obs.at[:,env.actor_dim-2].set(float(not cfg.stop_standing_stage))
        def step(carry, k):
            state, obs = carry
            stop = ((k >= jnp.asarray(starts)) & (k < jnp.asarray(ends))).astype(jnp.float32)
            state = state.replace(v_cmd=jnp.asarray(speeds, jnp.float32), stop_cmd=stop,
                                  cmd_left=jnp.full_like(state.cmd_left, 1e6))
            if cfg.stop_adapter:
                state, obs = prepare(agent.net, weights, stats, env, state, obs)
            else:
                obs = obs.at[:, env.actor_dim - 3].set(jnp.asarray(speeds) / cfg.v_max)
                obs = obs.at[:, env.actor_dim - 2].set(1 - stop)
            action = jnp.clip(agent._act_greedy(weights, stats.normalize(obs)), -1, 1)
            state2, obs2, reward, done, info = env.step(state, action, p)
            record = dict(qpos=state.data.qpos, qvel=state.data.qvel, spec=state.spec,
                          action=action, stop=stop, phase=state.phase,
                          speed=info['stop_speed_xy'], vx=info['v_body_x'],
                          gyro=info['stop_gyro'], joint_rms=info['stop_joint_rms'],
                          upright=info['stop_upright'], supported=info['stop_supported'],
                          xy=info['stop_xy'], done=done, reward=reward,
                          floor=info['reward_floor_hit'], freq=info['freq_hz'],
                          reasons=jnp.stack([info[r] for r in reasons], axis=-1),
                          terms=info['reward_terms'])
            if cfg.stop_adapter:
                record.update(phase=info['stop_phase'], ramp_duration=info['stop_duration'],
                              reference=info['stop_reference'])
            return (state2, obs2), record
        return jax.lax.scan(step, (state, obs), jnp.arange(ticks))[1]

    print(f'[trace] recording {ticks} ticks x {env.n_envs} environments', flush=True)
    data = jax.tree_util.tree_map(np.asarray, jax.jit(run)(agent.params, agent.stats))
    if cfg.stop_adapter:
        from stop02_metrics import summarize as summarize02
        result = summarize02(data, speeds, starts, ends, env.control_dt, cfg.stop_standing_stage)
    else:
        result = summarize(data, speeds, starts, ends, env.control_dt)
    term_names = list(data['terms'])
    data['reward_terms'] = np.stack([data['terms'][k] for k in term_names], axis=-1)
    del data['terms']
    diagnoses = []
    for i in range(env.n_envs):
        done_indices = np.flatnonzero(data['done'][:, i])
        end = int(done_indices[0]) if done_indices.size else ticks - 1
        after = end - int(starts[i])
        reason = [r for r, yes in zip(reasons, data['reasons'][end, i]) if yes] if done_indices.size else []
        mask = (np.arange(ticks) >= starts[i]) & (np.arange(ticks) < end)
        means = {k: float(data['reward_terms'][mask, i, j].mean())
                 for j, k in enumerate(term_names)} if mask.any() else {}
        diagnoses.append(dict(env=i, command_mps=float(speeds[i]),
                              stop_at_s=float(starts[i] * env.control_dt),
                              termination_s=float((end + 1) * env.control_dt) if done_indices.size else None,
                              seconds_after_stop=float((after + 1) * env.control_dt) if done_indices.size else None,
                              reasons=reason, speed_at_end=float(data['speed'][end, i]),
                              height_before_end=float(data['qpos'][end, i, env.plant.base_q['z']]),
                              tilt_before_end_deg=float(np.degrees(np.arccos(np.clip(
                                  data['upright'][max(0, end - 1), i], -1, 1)))),
                              mean_stop_reward_terms_unscaled=means))
    metadata = dict(run=str(args.run), checkpoint=str(args.checkpoint), step=int(agent.step),
                    ramp=cfg.stop_adapter, standing=cfg.stop_standing_stage,
                    control_dt=env.control_dt, model_path=cfg.model_path,
                    seed=4102, n_envs=env.n_envs, speeds=speeds.tolist(),
                    stop_ticks=starts.tolist(), restart_ticks=ends.tolist(),
                    term_names=term_names, reason_names=list(reasons),
                    sampling='qpos/qvel before tick; other diagnostic signals after tick',
                    thresholds={k: getattr(cfg, k) for k in ('term_height', 'term_gravity_z',
                                  'floor_viol_m', 'workspace_dx_max', 'workspace_dz_min',
                                  'workspace_dz_max', 'workspace_grace_s')})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, meta=np.array(json.dumps(metadata)), **data)
    args.out.with_suffix('.json').write_text(json.dumps(dict(metadata=metadata, summary=result,
                                                            diagnoses=diagnoses), indent=2) + '\n')
    print('[trace] ' + json.dumps(dict(success=result['success_fraction'],
          survivors=result['survive_fraction'], reasons={r: sum(r in d['reasons'] for d in diagnoses)
          for r in reasons})), flush=True)
    print(f'[trace] saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
