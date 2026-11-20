"""Short in-allocation warm-start PPO smoke gate before a clearance run."""

import argparse
from dataclasses import replace
import faulthandler
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import get_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--preset", default="cad_v2_clearance01")
    args = ap.parse_args()
    started = time.monotonic()
    stage_path = Path("validation/clearance-preflight-stage.json")
    stage_path.parent.mkdir(parents=True, exist_ok=True)

    def stage(name):
        row = dict(stage=name, elapsed_s=time.monotonic() - started, preset=args.preset)
        stage_path.write_text(json.dumps(row, indent=2) + "\n")
        print(f"[preflight +{row['elapsed_s']:.1f}s] {name}", flush=True)

    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    stage("import_jax")
    import jax
    import jax.numpy as jnp

    stage("import_environment")
    from env import DashEnvV2, EnvParams
    from ppo import PPO

    stage("initialize_gpu")
    gpu = jax.devices("gpu")
    if not gpu:
        raise RuntimeError("Preflight requires a GPU")
    print(f"[preflight] GPU devices: {gpu}", flush=True)
    c = replace(
        get_config(args.preset),
        n_envs=16,
        n_steps=8,
        batch_size=128,
        n_epochs=1,
    )
    stage("construct_environment")
    e = DashEnvV2(c, n_envs=16)
    stage("construct_ppo")
    smoke_dir = Path("validation/smoke")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    a = PPO(c, e, smoke_dir, c.total_steps, seed=41, eval_env=None)
    stage("load_parent")
    a.load(args.parent, warm_start=True)
    assert a.step == 0 and a.env_params.dr_scale == c.dr_scale_start
    assert abs(a.env_params.stance_ratio - c.stance_ratio_final) < 1e-8
    assert a.env_params.shape_scale == 1.0 and a.env_params.cmd_lo == 0.6
    # Include the new reward path in a compiled actual MJX step.
    stage("mjx_reset")
    s, ob = e.reset(jax.random.PRNGKey(41), EnvParams.final(c))
    jax.block_until_ready(ob)
    stage("mjx_step")
    _, ob, reward, _, info = e.step(
        s, jnp.zeros((16, e.action_dim)), EnvParams.final(c)
    )
    assert np.isfinite(np.asarray(reward)).all() and np.isfinite(np.asarray(ob)).all()
    assert (
        "low_swing_clearance" in info["reward_terms"]
        and "extra_swing_clearance" in info["reward_terms"]
    )
    stage("reset_smoke_rollout")
    a.reset_envs(jax.random.PRNGKey(42))
    rows = []
    for update in range(2):
        stage(f"ppo_update_{update + 1}")
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
        elapsed_s=time.monotonic() - started,
        note="Temporary 16x8, one epoch; real training initializes again from the unchanged parent.",
    )
    Path("validation/clearance-preflight.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    stage("completed")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
