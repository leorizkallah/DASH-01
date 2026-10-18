"""Greedy bilateral gait evaluation; never certify survival alone as walking."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from bilateral_gait import assess
from env import EnvParams
from train import make_eval_env
from gait_selection import candidate_rank, retain_evaluation
from walking_curriculum import early_metrics, observe, slip_metrics
from gait_diagnostics import attempt_summary


class GaitQualityEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.targets = np.repeat(cfg.gait_eval_targets, 12) if cfg.gait_eval_targets else None
        self.env = make_eval_env(cfg, n=12 if self.targets is None else len(self.targets))
        self.fn = None

    def evaluate(self, agent):
        env = self.env
        params = EnvParams.final(self.cfg)._replace(dr_scale=0., ctrl_jitter_ms=0., ctrl_drop_prob=0.)
        if self.fn is None:
            def run(weights, mean, var, count):
                from ppo import ObsStats
                stats = ObsStats(mean=mean, var=var, count=count)
                state, obs = env.reset(jax.random.PRNGKey(3103), params)
                if self.targets is not None:
                    state = state.replace(v_cmd=jnp.asarray(self.targets, dtype=jnp.float32),
                                          cmd_left=jnp.full_like(state.cmd_left, 1e9))
                    once = jax.vmap(env._once, in_axes=(0, None, 0, 0))(
                        state, params, state.phase, state.commit.astype(jnp.float32))
                    start = state.hist[:, env.hist_idx].reshape((env.n_envs, -1)).shape[1]
                    obs = obs.at[:, start:start+once.shape[1]].set(once)
                def tick(carry, _):
                    s, ob, alive = carry
                    a = jnp.clip(agent._act_greedy(weights, stats.normalize(ob)), -1., 1.)
                    nxt, ob2, reward, done, info = env.step(s, a, params)
                    row = dict(active=alive, fallen=info['fallen'], done=done,
                               speed=info['v_body_x'],
                               **{k: info[k] for k in ('term_low', 'term_tip', 'term_floor', 'term_ws', 'term_nan')},
                               error=info['track_err'], heading=jnp.abs(info['yaw_true']),
                               flight=info['flight_fraction'], quality=info['bilateral'],
                               floor=info['reward_floor_hit'], terms=info['reward_terms'],
                               actual_reward=reward, finite=jnp.isfinite(reward) & ~info['term_nan'])
                    return (nxt, ob2, alive & ~done), row
                return jax.lax.scan(tick, (state, obs, jnp.ones(env.n_envs, bool)), None,
                                    length=round(self.cfg.eval_seconds/env.control_dt))[1]
            self.fn = jax.jit(run)
        r = jax.device_get(self.fn(agent.params, agent.stats.mean, agent.stats.var, agent.stats.count))
        q = r['quality']
        live = r['active'] & (np.arange(len(r['active']))[:, None] >= 500)
        den = np.maximum(live.sum(axis=0), 1)
        def average(x):
            return (np.where(live, x, 0.).sum(axis=0)/den)
        steps = (q['valid_step']*live[..., None]).sum(axis=0)
        alts = (q['alternating_step']*live[..., None]).sum(axis=(0, 2))
        pairs = live & q['pair_seen']
        sym = (np.where(pairs[..., None], q['symmetry'], 0.).sum(axis=0)/np.maximum(pairs.sum(axis=0)[:, None], 1))
        sym = np.where(pairs.any(axis=0)[:, None], sym, 1.)
        contacts = q['contact'] & live[..., None]
        slip = (np.where(contacts, q['slip'], 0.).sum(axis=(0, 2))/np.maximum(contacts.sum(axis=(0, 2)), 1))
        fallback_target = self.targets if self.targets is not None else self.cfg.v_max*self.cfg.cmd_range[0]
        m = dict(survived=~(r['fallen'] & r['active']).any(axis=0),
                 track_error=np.where(live.any(axis=0), average(r['error']), fallback_target),
                 heading_error=average(r['heading']), steps=steps,
                 clear_fraction=(q['clear']*live[..., None]).sum(axis=0)/den[:, None],
                 alternation=alts/np.maximum(steps.sum(axis=1), 1), max_symmetry=sym.max(axis=1),
                 flight_fraction=average(r['flight']), contact_slip=slip,
                 joint_limit_violation=np.where(r['active'], q['joint_limit_violation'], 0.).max(axis=0))
        m.update(early_metrics(r, env.control_dt))
        if self.cfg.bilateral_sequence_enable:
            m['next_foot_overdue_fraction'] = (
                ((q['overdue_next'] > 0.) & live[..., None]).sum(axis=0)/den[:, None])
        if self.cfg.drag_success_curriculum:
            m.update(slip_metrics(r, env.control_dt))
        if self.cfg.bilateral_attempt_enable:
            m.update(attempt_summary(q, live, sym, env.control_dt))
        if self.cfg.sequential_steps_enable:
            m['unauthorized_air_fraction'] = (q['unauthorized_air']*live[..., None]).sum(axis=0)/den[:, None]
            m['sequence_rejected'] = (q['sequence_rejected']*live[..., None]).sum(axis=0)
        m['actual_reward_mean'] = average(r['actual_reward'])
        ok = assess(m)
        if self.cfg.bilateral_attempt_enable:
            m['incomplete_fraction'] = m['incomplete_step'].sum(axis=1)/np.maximum(m['attempt'].sum(axis=1), 1)
            # Valid alternation alone cannot certify a micro-stepping gait.
            ok &= (m['incomplete_fraction'] <= .1) & (m['repeated_attempt'].sum(axis=1) == 0)
        result = dict(step=agent.step, trials=env.n_envs, seed=3103, seconds=self.cfg.eval_seconds,
                      finite=bool(np.all(r['finite'])), qualified_fraction=float(ok.mean()),
                      keeper_eligible=bool(ok.mean() >= .9), survival_fraction=float(m['survived'].mean()),
                      track_error=float(m['track_error'].mean()), reward_floor_fraction=float(average(r['floor']).mean()),
                      actual_reward_mean=float(m['actual_reward_mean'].mean()),
                      per_trial={k: np.asarray(v).tolist() for k, v in m.items()},
                      passed=ok.tolist(), reward_means={k: float(average(v).mean()) for k, v in r['terms'].items()},
                      note='12 nominal greedy trials; not independent hardware robustness trials. '
                           'Quality measured after 5 s; survival includes the whole 25 s.')
        if self.targets is not None:
            # The training speed controls qualification. Other speeds test transfer;
            # a good off-target result must not mask failure at the training command.
            def summarize(mask):
                return dict(trials=int(mask.sum()), qualified_fraction=float(ok[mask].mean()),
                            keeper_eligible=bool(ok[mask].mean() >= .9),
                            survival_fraction=float(m['survived'][mask].mean()),
                            track_error=float(m['track_error'][mask].mean()),
                            reward_floor_fraction=float(average(r['floor'])[mask].mean()),
                            actual_reward_mean=float(m['actual_reward_mean'][mask].mean()),
                            per_trial={k: np.asarray(v)[mask].tolist() for k, v in m.items()},
                            passed=ok[mask].tolist(),
                            reward_means={k: float(average(v)[mask].mean()) for k, v in r['terms'].items()})
            result['per_target'] = {f'{v:.2f}': summarize(np.isclose(self.targets, v))
                                    for v in self.cfg.gait_eval_targets}
            primary = float(self.cfg.v_max*self.cfg.cmd_range[0])
            result.update(result['per_target'][f'{primary:.2f}'])
            result.update(target_mps=primary, total_trials=env.n_envs,
                          note='12 nominal greedy trials per target, not hardware robustness trials. '
                               'Top-level metrics and keeper eligibility refer to the training target only. '
                               'Quality after 5 s; survival over all 25 s. All targets retained in per_target.')
        if not result['finite']:
            raise ValueError('Nonfinite gait quality rollout')
        return result


def record(agent, evaluator, run):
    result = evaluator.evaluate(agent)
    run = Path(run)
    if evaluator.cfg.drag_success_curriculum:
        state = observe(agent.cur.get('drag_success', {}), result, require_steps=False,
                        min_speed=evaluator.cfg.walking_gate_min_speed)
        agent.cur['drag_success'] = state
        result['drag_curriculum'] = dict(state, drag_scale=float(agent.env_params.drag_scale))
        print(f"[drag curriculum] success={state['success_fraction']:.0%}, "
              f"advancing={state['advancing']}, scale={agent.env_params.drag_scale:.3f}", flush=True)
    if evaluator.cfg.walking_success_curriculum:
        state = observe(agent.cur.get('walking_success', {}), result,
                        require_steps=evaluator.cfg.walking_gate_requires_steps,
                        min_speed=evaluator.cfg.walking_gate_min_speed)
        agent.cur['walking_success'] = state
        result['walking_curriculum'] = dict(state, shape_scale=float(agent.env_params.shape_scale))
        print(f"[walking curriculum] success={state['success_fraction']:.0%}, "
              f"advancing={state['advancing']}, shape={agent.env_params.shape_scale:.3f}", flush=True)
    if evaluator.cfg.gait_save_evaluated:
        retain_evaluation(agent, result, run)
    with (run/'gait_evaluations.jsonl').open('a') as f:
        f.write(json.dumps(result)+'\n')
    (run/'latest_gait_eval.json').write_text(json.dumps(result, indent=2)+'\n')
    # Candidate is explicitly not a certified gait. Preserve useful failed attempts.
    ranking = evaluator.cfg.gait_rank_bilateral
    early_rank = evaluator.cfg.walking_success_curriculum
    score = candidate_rank(result, early=early_rank) if ranking else result['qualified_fraction']*10 + result['survival_fraction'] - result['track_error']
    old = run/'best_candidate_eval.json'
    previous = json.loads(old.read_text()) if old.exists() else None
    old_score = (candidate_rank(previous, early=early_rank) if ranking else previous['candidate_score']) if previous else None
    if old_score is None or score > old_score:
        agent.save(run/'best_candidate.msgpack')
        old.write_text(json.dumps(dict(result, candidate_score=score,
                                      candidate_rank_version=4 if evaluator.cfg.bilateral_attempt_enable
                                      else (3 if early_rank else (2 if ranking else 1))), indent=2)+'\n')
    print(f"[gait quality @ {agent.step:,}] qualified {sum(result['passed'])}/{result['trials']}, "
          f"survived {result['survival_fraction']:.0%}, error {result['track_error']:.3f} m/s, "
          f"floor-hit {result['reward_floor_fraction']:.1%}", flush=True)
    for target, summary in result.get('per_target', {}).items():
        steps = np.asarray(summary['per_trial']['steps']).mean(axis=0)
        early = np.asarray(summary['per_trial']['steps_from_reset']).mean(axis=0)
        life = np.mean(summary['per_trial']['first_episode_seconds'])
        print(f"  target {target} m/s: qualified {sum(summary['passed'])}/{summary['trials']}, "
              f"survived {summary['survival_fraction']:.0%}, first episode {life:.2f}s, "
              f"L/R steps from reset {early[0]:.1f}/{early[1]:.1f}, "
              f"after 5s {steps[0]:.1f}/{steps[1]:.1f}", flush=True)
    return result
