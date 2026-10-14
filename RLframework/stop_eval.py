"""Deterministic transition benchmark for training. No sensor-only stand can pass it."""
import json
from pathlib import Path

import numpy as np

from stop_metrics import protocol, summarize
from stop_adapter import prepare


class StopEvaluator:
    def __init__(self, agent, env, capture=False, standing=None):
        import jax
        import jax.numpy as jnp
        from env import EnvParams
        from dataclasses import replace

        self.env = env
        self.standing = env.cfg.stop_standing_stage if standing is None else bool(standing)
        self.speeds, self.starts, self.ends, ticks = protocol(env.n_envs, env.control_dt)
        if env.cfg.stop_adapter:
            from stop02_metrics import protocol as protocol02
            self.speeds, self.starts, self.ends, ticks = protocol02(env.n_envs, env.control_dt,
                replace(env.cfg, stop_standing_stage=self.standing))
        p = EnvParams.final(env.cfg)._replace(dr_scale=0.0, bringup_scale=0.0,
                bringup_p_drop=0.0, bringup_p_held=0.0, ctrl_jitter_ms=0.0,
                ctrl_drop_prob=0.0)
        speeds = jnp.asarray(self.speeds, jnp.float32)
        starts, ends = jnp.asarray(self.starts), jnp.asarray(self.ends)

        def run(weights, stats, seed=4102):
            state, obs = env.reset(jax.random.PRNGKey(seed), p)
            if env.cfg.stop_adapter:
                state = state.replace(stop_previous=jnp.full_like(state.stop_previous, self.standing),
                                      effective_speed=speeds, effective_hold=jnp.full_like(state.effective_hold, self.standing))
                obs = obs.at[:, env.actor_dim-3].set(speeds/env.cfg.v_max)
                obs = obs.at[:, env.actor_dim-2].set(float(not self.standing))

            def tick(carry, k):
                state, obs = carry
                stop = ((k >= starts) & (k < ends)).astype(jnp.float32)
                state = state.replace(v_cmd=speeds, stop_cmd=stop,
                                      cmd_left=jnp.full_like(state.cmd_left, 1e6))
                # Pin BOTH state and the task fields the actor actually reads this tick.
                if env.cfg.stop_adapter:
                    state, obs = prepare(agent.net, weights, stats, env, state, obs)
                else:
                    obs = obs.at[:, env.actor_dim - 3].set(speeds / env.cfg.v_max)
                    obs = obs.at[:, env.actor_dim - 2].set(1.0 - stop)
                action = jnp.clip(agent._act_greedy(weights, stats.normalize(obs)), -1, 1)
                replay = dict(qpos=state.data.qpos, qvel=state.data.qvel,
                              spec=state.spec, action=action) if capture else {}
                state, obs, reward, done, info = env.step(state, action, p)
                trace = dict(speed=info['stop_speed_xy'], vx=info['v_body_x'],
                             gyro=info['stop_gyro'], joint_rms=info['stop_joint_rms'],
                             upright=info['stop_upright'], supported=info['stop_supported'],
                             xy=info['stop_xy'], done=done, reward=reward,
                             floor=info['reward_floor_hit'],
                             terms=info['reward_terms'])
                trace['ramp_duration'] = info['stop_duration']
                trace['reference'] = info['stop_reference']
                trace['hold_active'] = info['stop_phase'] == 2
                trace['gate_timeout'] = info['stop_gate_timeout']
                trace['estimated_speed_xy'] = info['stop_gate_speed']
                trace['reasons'] = jnp.stack([info[r] for r in ('term_low','term_tip','term_floor','term_ws','term_nan')], axis=-1)
                if capture:
                    trace.update(replay, freq=info['freq_hz'], phase=info['stop_phase'])
                return (state, obs), trace

            return jax.lax.scan(tick, (state, obs), jnp.arange(ticks))[1]

        self._run = jax.jit(run)

    def evaluate(self, agent, seed=4102):
        import jax
        trace = jax.tree_util.tree_map(np.asarray, self._run(agent.params, agent.stats, seed))
        if self.env.cfg.stop_adapter:
            from stop02_metrics import summarize as summarize02
            result = summarize02(trace, self.speeds, self.starts, self.ends, self.env.control_dt,
                                 self.standing, self.env.cfg.stop_hold_window_s)
        else:
            result = summarize(trace, self.speeds, self.starts, self.ends, self.env.control_dt)
        k = np.arange(trace['speed'].shape[0])[:, None]
        live = ~np.maximum.accumulate(trace['done'], axis=0)
        mask = (k >= self.starts) & (k < self.ends) & live
        result['stop_reward_floor_fraction'] = float(trace['floor'][mask].mean()) if mask.any() else None
        result['stop_reward_mean'] = float(trace['reward'][mask].mean()) if mask.any() else None
        result['stop_reward_terms_mean_unscaled'] = {
            name: float(value[mask].mean()) if mask.any() else None
            for name, value in trace['terms'].items()}
        result['seed'] = int(seed)
        return result


def record_stop_eval(agent, evaluator, run, walking=None, stage=None, standing=None):
    """Append raw results; save a separate keeper only within baseline walking tolerance."""
    run = Path(run)
    result = evaluator.evaluate(agent)
    result['step'] = agent.step
    result['stage'] = stage or ('source' if agent.step == 0 else 'ppo')
    if standing is not None:
        result['standing'] = standing
    result['walking'] = ({k: walking[k] for k in ('track_err_mean', 'alive_frac')}
                         if walking is not None else None)
    positive = [row for row in walking.get('per_cmd', []) if row[0] > 0] if walking else []
    result['walking_run_only'] = (dict(track_err_mean=float(np.mean([r[1] for r in positive])),
                                      alive_frac=float(np.mean([r[2] for r in positive]))) if positive else None)
    baseline_path = run / 'stop_baseline.json'
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    baseline = json.loads(baseline_path.read_text())
    ref, current = baseline.get('walking_run_only'), result['walking_run_only']
    eligible = bool(ref and current
                    and current['alive_frac'] >= ref['alive_frac'] - 0.05
                    and current['track_err_mean'] <= ref['track_err_mean'] + 0.10)
    result['walking_retained'] = eligible
    if standing is not None:
        result['standing_retained'] = bool(standing['success_fraction'] >=
            baseline['standing']['success_fraction'] - .10)
        eligible = eligible and result['standing_retained']
    path = run / 'stop_evaluations.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(result, allow_nan=False) + '\n')
    score = [result['success_fraction'], result['restart_fraction'],
             result['survive_fraction'], -(result['residual_speed_mean_mps']
                                           if result['residual_speed_mean_mps'] is not None else 1e6)]
    best_path = run / 'best_stop_eval.json'
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if eligible and (best is None or score > best['score']):
        agent.save(run / 'best_stop.msgpack')
        best_path.write_text(json.dumps(dict(result, score=score), indent=2, allow_nan=False) + '\n')
    print(f"[stop @ {agent.step:,}] success {result['success_fraction']:.0%}, "
          f"survive {result['survive_fraction']:.0%}, restart {result['restart_fraction']:.0%}, "
          f"walking retained {result['walking_retained']}, keeper eligible {eligible}", flush=True)
    from tools.stop_report import generate
    generate(run)
    return result
