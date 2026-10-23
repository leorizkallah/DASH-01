"""Validate homing CPU/MJX agreement and finite walking-only environment rewards."""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config
from env import DashEnvV2
from ppo import initial_params
from cad_v2_preflight import physics_comparison


def main():
    cfg = get_config("cad_v2_walk01")
    env = DashEnvV2(cfg, n_envs=16)
    params = initial_params(cfg)
    comparison, _ = physics_comparison(env, params)
    print("Homing CPU/MJX:", json.dumps(comparison), flush=True)
    assert comparison["passed"], comparison

    @jax.jit
    def probe():
        state, obs = env.reset(jax.random.PRNGKey(41), params)

        def tick(carry, _):
            s, key = carry
            key, sub = jax.random.split(key)
            action = jnp.clip(
                0.7 * jax.random.normal(sub, (env.n_envs, env.action_dim)), -1, 1
            )
            nxt, ob, r, done, info = env.step(s, action, params)
            return (nxt, key), dict(
                reward=r,
                finite=jnp.isfinite(ob).all(axis=1),
                nan=info["term_nan"],
                stop=nxt.stop_cmd,
                command=nxt.v_cmd,
            )

        return jax.lax.scan(tick, (state, jax.random.PRNGKey(42)), None, length=200)[1]

    r = jax.device_get(probe())
    result = dict(
        preset="cad_v2_walk01",
        physics_comparison=comparison,
        actor_dim=env.actor_dim,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        finite=bool(np.isfinite(r["reward"]).all() and r["finite"].all()),
        nan_terminations=int(r["nan"].sum()),
        stop_ticks=int((r["stop"] != 0).sum()),
        command_min=float(r["command"].min()),
        command_max=float(r["command"].max()),
        reward_mean=float(r["reward"].mean()),
        note="Random-action finite/STOP-disabled probe, not a learned walking result.",
    )
    Path("validation/walk01-preflight.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    assert (
        result["finite"]
        and result["nan_terminations"] == 0
        and result["stop_ticks"] == 0
    )
    assert result["command_min"] > 0


if __name__ == "__main__":
    main()
