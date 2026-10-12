"""Recover deployed actor/estimator exactly; an NPZ bundle has no critic or optimizer."""
import hashlib
import json
from pathlib import Path

import numpy as np


def restore_bundle(agent, path):
    import jax.numpy as jnp
    from flax.core import unfreeze

    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta'].item()))
        if (meta['actor_dim'] != agent.env.actor_dim or meta['action_dim'] != agent.env.action_dim
                or not meta.get('stop_flag') or meta['objective'] != 'joystick'):
            raise ValueError('Bundle does not match the joystick RUN/STOP training contract')
        if meta['control_dt'] != agent.env.control_dt or meta['v_max'] != agent.cfg.v_max:
            raise ValueError('Bundle command/timing contract differs from training')
        for name in ('nominal_ctrl', 'default_motor_pos'):
            if not np.allclose(z[name], getattr(agent.env.plant, name), atol=1e-5, rtol=1e-5):
                raise ValueError(f'Bundle {name} differs from training plant')
        params = unfreeze(agent.params)
        p = params['params']
        for block, prefix, depth in [('estimator', 'est', 3), ('policy_net', 'pi', 2)]:
            for i in range(depth):
                layer = p[block][f'Dense_{i}']
                kernel, bias = z[f'{prefix}_w{i}'].T, z[f'{prefix}_b{i}']
                if kernel.shape != layer['kernel'].shape or bias.shape != layer['bias'].shape:
                    raise ValueError(f'Layer shape mismatch: {block}/{i}')
                layer['kernel'], layer['bias'] = jnp.asarray(kernel), jnp.asarray(bias)
        p['action_net']['kernel'] = jnp.asarray(z['act_w'].T)
        p['action_net']['bias'] = jnp.asarray(z['act_b'])
        # Per-dimension training std is absent. Record this deliberate exploration initialization.
        std = min(float(meta.get('policy_std_mean', 0.17)), np.exp(agent.cfg.max_log_std))
        p['log_std'] = jnp.full_like(p['log_std'], np.log(std))
        agent.params = params
        n = agent.env.actor_dim
        agent.stats = agent.stats.replace(mean=agent.stats.mean.at[:n].set(z['obs_mean']),
                                          var=agent.stats.var.at[:n].set(z['obs_var']),
                                          count=jnp.asarray(1e6))
        # Independent numpy forward pass over raw observations, including normalization.
        raw = np.random.default_rng(4102).normal(size=(64, n)).astype(np.float32)
        full = np.zeros((64, agent.env.obs_dim), np.float32)
        full[:, :n] = raw
        norm = np.clip((raw - z['obs_mean']) / np.sqrt(z['obs_var'] + meta['obs_eps']),
                       -meta['clip_obs'], meta['clip_obs'])
        est = norm
        for i in range(3):
            est = est @ z[f'est_w{i}'].T + z[f'est_b{i}']
            if i < 2:
                est = np.tanh(est)
        hidden = np.concatenate([norm, est], axis=-1)
        for i in range(2):
            hidden = np.tanh(hidden @ z[f'pi_w{i}'].T + z[f'pi_b{i}'])
        expected = hidden @ z['act_w'].T + z['act_b']
        actual = np.asarray(agent.net.apply(agent.params, agent.stats.normalize(jnp.asarray(full)),
                                            method=agent.net.actor_mean))
        actual_est = np.asarray(agent.net.apply(agent.params, agent.stats.normalize(jnp.asarray(full)),
                                                method=agent.net.estimate))
        error = float(np.max(np.abs(actual - expected)))
        est_error = float(np.max(np.abs(actual_est - est)))
        if not np.allclose(actual, expected, atol=2e-4, rtol=2e-4) or est_error > 2e-4:
            raise ValueError(f'Bundle parity failed: actor={error}, estimator={est_error}')
    record = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_run=meta['run'], source_step=meta['step'],
                  actor_max_abs_error=error, estimator_max_abs_error=est_error,
                  critic='fresh initialization; fitted before PPO',
                  exploration_std=std, normalization='frozen; privileged mean=0 variance=1')
    (agent.run / 'warmstart.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'[bundle] restored actor + estimator; maximum action error {error:.2g}', flush=True)


def prefit_critic(agent, rollouts=16):
    """Collect with the frozen actor and fit only value-network weights before PPO.

    These samples are recorded separately and do not advance PPO schedules. The
    critic uses bootstrapped lambda returns; this is initialization, not convergence.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from ppo import gae_fn

    if agent.n_dev != 1:
        raise ValueError('Bundle critic prefit currently requires one GPU')
    tx = optax.adam(3e-4)
    value = {k: v for k, v in agent.params['params'].items() if k.startswith('value_')}
    opt = tx.init(value)
    fixed = {k: v for k, v in agent.params['params'].items() if not k.startswith('value_')}

    @jax.jit
    def update(value, opt, obs, target):
        def loss(v):
            weights = {'params': dict(fixed, **v)}
            prediction = agent.net.apply(weights, obs, method=agent.net.value)
            return jnp.mean((prediction - target) ** 2)
        loss_value, grad = jax.value_and_grad(loss)(value)
        updates, opt = tx.update(grad, opt, value)
        return optax.apply_updates(value, updates), opt, loss_value

    losses = []
    for i in range(rollouts):
        agent.key, key = jax.random.split(agent.key)
        agent.env_state, agent.obs, _, tr, last, _, _ = agent._rollout(
            agent.params, agent.stats, agent.env_state, agent.obs, key, agent.env_params)
        _, target = gae_fn(tr, last, agent.cfg.gamma, 1.0)
        obs, target = tr.obs.reshape(-1, agent.env.obs_dim), target.reshape(-1)
        for lo in range(0, agent.batch, agent.cfg.batch_size):
            value, opt, loss = update(value, opt, obs[lo:lo + agent.cfg.batch_size],
                                      target[lo:lo + agent.cfg.batch_size])
        agent.params = {'params': dict(fixed, **value)}
        losses.append(float(loss))
        print(f'[critic prefit {i + 1}/{rollouts}] final minibatch MSE {losses[-1]:.3f}', flush=True)
        if not np.isfinite(losses[-1]):
            raise ValueError('Critic prefit became non-finite')
    (agent.run / 'critic_prefit.json').write_text(json.dumps(
        dict(rollouts=rollouts, samples=rollouts * agent.batch, minibatch_mse=losses), indent=2) + '\n')
