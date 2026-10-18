"""Evaluate the pilot's best/final policies before a separate long continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import get_config
from env import EnvParams
from ppo import PPO
from train import make_eval_env, command_ladder, keeper_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = get_config("cad_v2_gait01")
    env = make_eval_env(cfg, n=12)
    agent = PPO(cfg, env, args.out, total_steps=cfg.total_steps, seed=0, eval_env=env)
    params = EnvParams.final(cfg)._replace(dr_scale=0., ctrl_jitter_ms=0., ctrl_drop_prob=0.)
    ladder = command_ladder(cfg, env.n_envs)
    rows = []
    for name in ("best", "final"):
        path = args.source / f"{name}.msgpack"
        agent.load(path)
        ev = agent.evaluate(params=params, ladder=ladder,
                            n_max_steps=round(cfg.eval_seconds/env.control_dt), seed=123)
        row = dict(name=name, path=str(path.resolve()),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   source_step=agent.step, keeper_score=keeper_score(cfg, ev, None),
                   mean_episode_s=float(np.mean(ev["ep_len_s"])),
                   episode_s=np.asarray(ev["ep_len_s"]).tolist(),
                   alive_fraction=ev["alive_frac"], track_error=ev["track_err_mean"],
                   speed_mean=ev["speed_mean"], falls=ev["falls"], trials=ev["n"])
        if not np.isfinite([row["keeper_score"], row["mean_episode_s"]]).all():
            raise ValueError(f"Nonfinite selection metrics: {name}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Existing keeper score first; use survival duration when all early falls tie.
    selected = max(rows, key=lambda r: (r["keeper_score"], r["mean_episode_s"], r["source_step"]))
    destination = args.out / "selected.msgpack"
    shutil.copyfile(selected["path"], destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == selected["sha256"]
    report = dict(candidates=rows, selected=selected,
                  protocol="12 nominal replicas, 25 s, commands 0.2/0.35/0.5 m/s; same seed; "
                           "keeper score then episode duration then latest step. Replicas are "
                           "deterministic, not independent robustness trials.")
    (args.out / "selection.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Selected {selected['name']} at {selected['source_step']:,} steps", flush=True)


if __name__ == "__main__":
    main()
