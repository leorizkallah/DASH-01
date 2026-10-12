"""Synchronize the STOP-02 actor task and environment reward before acting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'controller' / 'deploy'))
import jax.numpy as jnp
from stop_schedule import advance


def prepare(net, weights, stats, env, state, obs):
    if not env.cfg.stop_adapter:
        return state, obs
    cfg = env.cfg
    estimate = net.apply(weights, stats.normalize(obs), method=net.estimate)[..., 0]
    j = env.actor_dim
    estimate = (estimate * jnp.sqrt(stats.var[j] + 1e-8) + stats.mean[j]) / cfg.obs_scales['base_vel']
    estimate = jnp.nan_to_num(estimate, nan=0.0, posinf=cfg.v_max, neginf=0.0)
    values = advance(state.stop_cmd > 0.5, state.stop_previous, state.v_cmd,
                     estimate, state.stop_estimate, state.stop_v0, state.stop_started,
                     state.stop_duration, state.t, env.control_dt,
                     cfg.stop_ramp_decel, cfg.stop_ramp_min_s, cfg.v_max, jnp)
    names = ('stop_estimate', 'stop_v0', 'stop_started', 'stop_duration',
             'effective_speed', 'effective_hold', 'stop_phase')
    # Idempotent at one simulation tick (rollout bootstrap can prepare it early).
    fresh = state.stop_prepared_t != state.t
    changes = {n: jnp.where(fresh, v, getattr(state, n)) for n, v in zip(names, values)}
    state = state.replace(**changes, stop_previous=jnp.where(fresh, state.stop_cmd > 0.5,
                                                           state.stop_previous),
                          stop_prepared_t=state.t)
    obs = obs.at[..., j - 3].set(state.effective_speed / cfg.v_max)
    obs = obs.at[..., j - 2].set(1.0 - state.effective_hold.astype(jnp.float32))
    return state, obs
