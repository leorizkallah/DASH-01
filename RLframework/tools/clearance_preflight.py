"""Short in-allocation warm-start PPO smoke gate before a clearance run."""

import argparse
from dataclasses import replace
import hashlib
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--preset", default="cad_v2_clearance01")
    args = ap.parse_args()
    c = replace(
        get_config(args.preset),
        n_envs=16,
        n_steps=8,
        batch_size=128,
        n_epochs=1,
    )
    e = DashEnvV2(c, n_envs=16)
    a = PPO(c, e, Path("validation/smoke"), c.total_steps, seed=41, eval_env=None)
    a.load(args.parent, warm_start=True)
    assert a.step == 0 and a.env_params.dr_scale == c.dr_scale_start
    assert abs(a.env_params.stance_ratio - c.stance_ratio_final) < 1e-8
    assert a.env_params.shape_scale == 1.0 and a.env_params.cmd_lo == 0.6
    # Include the new reward path in a compiled actual MJX step.
    s, ob = e.reset(jax.random.PRNGKey(41), EnvParams.final(c))
    _, ob, reward, _, info = e.step(
        s, jnp.zeros((16, e.action_dim)), EnvParams.final(c)
    )
    assert np.isfinite(np.asarray(reward)).all() and np.isfinite(np.asarray(ob)).all()
    assert (
        "low_swing_clearance" in info["reward_terms"]
        and "extra_swing_clearance" in info["reward_terms"]
    )
    a.reset_envs(jax.random.PRNGKey(42))
    rows = []
    for _ in range(2):
        row = a.iterate()
        keys = (
            "train/loss_pg",
            "train/loss_vf",
            "train/approx_kl",
            "rollout/reward_mean",
            "reward_terms/low_swing_clearance",
            "reward_terms/extra_swing_clearance",
            "reward_terms/excess_swing_clearance",
            "clearance/left_mean_air_mm",
            "clearance/right_mean_air_mm",
        )
        assert all(np.isfinite(row[k]) for k in keys), row
        rows.append({k: float(row[k]) for k in keys})
    result = dict(
        passed=True,
        preset=args.preset,
        device=str(jax.devices()[0]),
        parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        warm_start_step_reset=True,
        updates=rows,
        smoke_steps=256,
        note="Temporary 16x8, one epoch; real training initializes again from the unchanged parent.",
    )
    Path("validation/clearance-preflight.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
