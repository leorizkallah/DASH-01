"""Bounded behavior cloning: nominal PD stance plus frozen source-policy walking.

Only actor mean weights change. This is an initialization, not a runtime action
override; the unmodified actor must pass physical evaluations before PPO starts.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze, unfreeze

from env import EnvParams
from imitation_data import dataset_masks, initialization_gate
from stop_adapter import prepare


def initialize(agent, env, evaluator, source_result, evaluate_walking):
    from stop_eval import record_stop_eval
    run = Path(agent.run)
    path = run / 'imitation.json'
    report = dict(stage='collecting', seed=5303, ticks=800,
                  teacher='zero action = nominal PD stance; frozen STOP-01B actor for RUN',
                  updates=2000, batch_per_task=256, lr=1e-4, walking_loss_weight=2.0,
                  source_baseline='stop_baseline.json', ppo_steps=0)

    def persist():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('[imitation] ' + report['stage'], flush=True)

    persist()
    p = EnvParams.final(env.cfg)._replace(dr_scale=0., bringup_scale=0.,
        bringup_p_drop=0., bringup_p_held=0., ctrl_jitter_ms=0., ctrl_drop_prob=0.)
    hold = jnp.arange(env.n_envs) < env.n_envs // 2
    speeds = jnp.where(hold, 0., jnp.resize(jnp.array([.125, .5, 1., 1.5, 2., 2.125]), env.n_envs))
    source = agent.params

    def collect(weights, stats):
        state, obs = env.reset(jax.random.PRNGKey(report['seed']), p)
        state = state.replace(stop_previous=hold, effective_hold=hold,
            effective_speed=speeds, stop_duration=jnp.zeros_like(state.stop_duration))
        obs = obs.at[:, env.actor_dim-3].set(speeds/env.cfg.v_max)
        obs = obs.at[:, env.actor_dim-2].set((~hold).astype(jnp.float32))

        def tick(carry, _):
            state, obs = carry
            state = state.replace(v_cmd=speeds, stop_cmd=hold.astype(jnp.float32),
                                  cmd_left=jnp.full_like(state.cmd_left, 1e6))
            state, obs = prepare(agent.net, weights, stats, env, state, obs)
            normalized = stats.normalize(obs)
            teacher = agent.net.apply(weights, normalized, method=agent.net.actor_mean)
            target = jnp.where(hold[:, None], 0., teacher)
            ns, no, _, done, info = env.step(state, jnp.clip(target, -1., 1.), p)
            return (ns, no), (normalized, target, done, info['stop_upright'])

        return jax.lax.scan(tick, (state, obs), None, length=report['ticks'])[1]

    obs, target, done, upright = jax.tree_util.tree_map(np.asarray, jax.jit(collect)(source, agent.stats))
    finite = np.isfinite(obs).all(-1) & np.isfinite(target).all(-1)
    masks = dataset_masks(done, upright, np.asarray(hold), finite)
    report['samples'] = {name: int(mask.sum()) for name, mask in masks.items()}
    report['teacher_hold_survival_8s'] = float((~np.any(done, axis=0))[np.asarray(hold)].mean())
    report['demonstration_env_steps'] = int(done.size)
    report['reset_joint_noise'] = env.cfg.reset_joint_noise
    if any(n < 256 for n in report['samples'].values()):
        report['stage'] = 'rejected_insufficient_demonstrations'
        persist()
        raise SystemExit('Imitation dataset too small; PPO not started')
    # Save demonstrations for audit/reproduction, including rejected rows' masks.
    np.savez_compressed(run / 'imitation_demonstrations.npz', obs=obs, target=target,
                        done=done, upright=upright, hold=np.asarray(hold), **masks)
    data = {name: (jnp.asarray(obs[mask]), jnp.asarray(target[mask])) for name, mask in masks.items()}
    full = unfreeze(source)
    actor = {name: full['params'][name] for name in ('policy_net', 'action_net')}

    def weights(a):
        value = dict(full, params=dict(full['params'], **a))
        return freeze(value) if isinstance(source, FrozenDict) else value

    def mse(a, x, y):
        pred = agent.net.apply(weights(a), x, method=agent.net.actor_mean)
        return jnp.mean(jnp.square(pred-y))

    validation = jax.jit(lambda a: jnp.array([mse(a, *data['hold_valid']), mse(a, *data['walk_valid'])]))
    before = np.asarray(validation(actor))
    optimizer = optax.adam(report['lr'])
    opt_state = optimizer.init(actor)

    @jax.jit
    def block(a, opt, key):
        def update(carry, _):
            a, opt, key = carry
            key, kh, kw = jax.random.split(key, 3)
            xh, yh = data['hold_train']; xw, yw = data['walk_train']
            ih = jax.random.randint(kh, (256,), 0, len(xh))
            iw = jax.random.randint(kw, (256,), 0, len(xw))
            loss, grad = jax.value_and_grad(lambda a: mse(a, xh[ih], yh[ih]) +
                                           2.*mse(a, xw[iw], yw[iw]))(a)
            updates, opt = optimizer.update(grad, opt, a)
            return (optax.apply_updates(a, updates), opt, key), loss
        return jax.lax.scan(update, (a, opt, key), None, length=200)

    key = jax.random.PRNGKey(5304)
    report['stage'] = 'fitting'
    persist()
    history = []
    for step in range(200, report['updates']+1, 200):
        (actor, opt_state, key), losses = block(actor, opt_state, key)
        scores = np.asarray(validation(actor))
        if not np.isfinite(scores).all():
            report['stage'] = 'rejected_nonfinite_fit'
            persist()
            raise SystemExit('Nonfinite imitation fit; PPO not started')
        history.append(dict(update=step, loss=float(np.asarray(losses).mean()),
                            hold_mse=float(scores[0]), walk_mse=float(scores[1])))
        print(f'[imitation {step}] hold MSE {scores[0]:.6f}, RUN MSE {scores[1]:.6f}', flush=True)
    agent.params = weights(actor)
    for name in ('estimator', 'value_net', 'value_head', 'log_std'):
        for a, b in zip(jax.tree_util.tree_leaves(source['params'][name]),
                        jax.tree_util.tree_leaves(agent.params['params'][name])):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    report['validation'] = dict(hold_before=float(before[0]), hold_after=float(scores[0]),
                                walk_before=float(before[1]), walk_after=float(scores[1]))
    report['history'] = history
    report['stage'] = 'evaluating'
    persist()
    # Save even rejected fits, but not under ckpt_*: resume must not bypass the gate.
    agent.save(run / 'imitation_candidate.msgpack')
    walking = evaluate_walking()
    result = record_stop_eval(agent, evaluator, run, walking, stage='post_imitation')
    report['after'] = result
    report['gates'] = initialization_gate(source_result, result, report['validation'])
    passed = all(report['gates'].values())
    report['stage'] = 'accepted_for_ppo' if passed else 'rejected_behavior_gate'
    persist()
    from tools.stop_report import generate
    generate(run)
    if not passed:
        raise SystemExit('Imitation behavioral gate failed; results saved, PPO not started')
