"""Fail-fast GPU smoke checks, run inside the allocation BEFORE long training."""

from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from config import get_config
from env import DashEnvV2, EnvParams
from ppo import PPO


def main():
    c = replace(
        get_config("cad_v2_flatfeet_symmetry_v1"),
        n_envs=16,
        n_steps=8,
        batch_size=128,
        n_epochs=1,
    )
    e = DashEnvV2(c, n_envs=16)
    p = EnvParams.final(c)._replace(
        dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0
    )
    s, ob = e.reset(jax.random.PRNGKey(41), p)
    s = s.replace(
        v_cmd=jnp.full(16, 0.25), stop_cmd=jnp.zeros(16), cmd_left=jnp.full(16, 1e9)
    )
    z = jnp.zeros((16, e.action_dim))
    # A persistent symmetry event is terminal, charged once and auto-resets.
    fs = dict(s.flatfeet_state)
    fs.update(
        count=jnp.full_like(fs["count"], 3),
        checked=jnp.full_like(fs["checked"], 3),
        bad_pairs=jnp.full_like(fs["bad_pairs"], c.symmetry_persistent_pairs),
        walking_s=jnp.full_like(fs["walking_s"], 10.0),
    )
    s = s.replace(flatfeet_state=fs)
    nxt, obs, r, done, info = e.step(s, z, p)
    assert np.asarray(info["term_symmetry"]).all()
    assert np.asarray(done).all() and (np.asarray(nxt.step_n) == 0).all()
    assert not np.asarray(info["fallen"]).any()
    np.testing.assert_allclose(
        np.asarray(info["reward_terms"]["symmetry_terminal"]), -10.0
    )
    _, _, _, _, info2 = e.step(nxt, z, p)
    assert not np.asarray(info2["term_symmetry"]).any()
    assert np.isfinite(np.asarray(obs)).all() and np.isfinite(np.asarray(r)).all()
    # Fresh weights: two actual PPO updates, no checkpoint reused for training.
    agent = PPO(c, e, Path("validation/smoke"), c.total_steps, seed=41, eval_env=None)
    agent.reset_envs(jax.random.PRNGKey(42))
    records = []
    for _ in range(2):
        row = agent.iterate()
        keys = (
            "train/loss_pg",
            "train/loss_vf",
            "train/approx_kl",
            "rollout/reward_mean",
            "reward_terms/flat_support",
            "reward_terms/symmetry",
            "flatfeet/termination_fraction",
        )
        assert all(np.isfinite(row[k]) for k in keys), row
        records.append({k: float(row[k]) for k in keys})
    out = dict(
        passed=True,
        device=str(jax.devices()[0]),
        preset="cad_v2_flatfeet_symmetry_v1",
        smoke_steps=256,
        terminal_once_and_autoreset=True,
        updates=records,
        note="16x8, one epoch for smoke only; training retains original 512x36 and four epochs.",
    )
    Path("validation").mkdir(exist_ok=True)
    Path("validation/flatfeet-preflight.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
