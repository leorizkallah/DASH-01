"""First-episode speed evaluation and survival-first checkpoint selection."""

import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from env import EnvParams
from train import make_eval_env
from speed_metrics import summarize


class SpeedEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = make_eval_env(cfg, n=16)
        self.fn = None

    def evaluate(self, agent):
        env, cfg = self.env, self.cfg
        if self.fn is None:

            def rollout(weights, mean, var, count):
                from ppo import ObsStats

                stats = ObsStats(mean=mean, var=var, count=count)
                params = EnvParams.final(cfg)._replace(
                    dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0
                )
                state, obs = env.reset(jax.random.PRNGKey(3103), params)

                def tick(carry, _):
                    s, ob, active = carry
                    a = jnp.clip(
                        agent._act_greedy(weights, stats.normalize(ob)), -1.0, 1.0
                    )
                    nxt, ob2, reward, done, info = env.step(s, a, params)
                    d = info["speed_metrics"]
                    row = dict(
                        active=active,
                        fallen=info["fallen"],
                        reward=reward,
                        vx=d["vx"],
                        contact=d["contact"],
                        slip=d["slip"],
                        clear=d["clear"],
                        joint_violation=d["joint_violation"],
                        heading=info["yaw_true"],
                        thermal=info["thermal_max"],
                        flight=info["flight_fraction"],
                        ceiling=jnp.abs(d["vx"]) >= 0.95 * cfg.v_ceiling,
                        floor=info["reward_floor_hit"],
                        terms=info["reward_terms"],
                    )
                    return (nxt, ob2, active & ~done), row

                return jax.lax.scan(
                    tick,
                    (state, obs, jnp.ones(env.n_envs, bool)),
                    None,
                    length=round(cfg.eval_seconds / env.control_dt),
                )[1]

            self.fn = jax.jit(rollout)
        r = jax.device_get(
            self.fn(agent.params, agent.stats.mean, agent.stats.var, agent.stats.count)
        )
        result = summarize(r, env.control_dt, cfg.eval_seconds)
        mask = r["active"]
        result.update(
            step=agent.step,
            seconds=cfg.eval_seconds,
            seed=3103,
            reward_means={
                k: float(np.where(mask, v, 0.0).sum() / max(mask.sum(), 1))
                for k, v in r["terms"].items()
            },
        )
        if not result["finite"]:
            raise ValueError("Nonfinite speed evaluation")
        return result


def training_summary(r):
    m = r["per_trial"]
    return dict(
        finishes=0,
        falls=r["trials"] - sum(m["survived"]),
        n=r["trials"],
        t_line_mean=float("nan"),
        dist_mean=float(np.mean(m["distance_m"])),
        speed_mean=r["selection_score"],
        track_err_mean=float("nan"),
        heading_err_mean=float(np.mean(m["heading_error"])),
        alive_frac=r["survival_fraction"],
    )


def record(agent, evaluator, run):
    r = evaluator.evaluate(agent)
    run = Path(run)
    for label, rank in [
        ("best_candidate", r["candidate_rank"]),
        ("best_speed", [r["selection_score"]]),
    ]:
        if label == "best_speed" and not r["keeper_eligible"]:
            continue
        path = run / (label + "_eval.json")
        old = json.loads(path.read_text()) if path.exists() else None
        old_rank = (
            old["candidate_rank"]
            if old and label == "best_candidate"
            else ([old["selection_score"]] if old else None)
        )
        if old_rank is None or rank > old_rank:
            agent.save(run / (label + ".msgpack"))
            path.write_text(json.dumps(r, indent=2) + "\n")
    (run / "latest_speed_eval.json").write_text(json.dumps(r, indent=2) + "\n")
    with (run / "speed_evaluations.jsonl").open("a") as f:
        f.write(json.dumps(r) + "\n")
    print(
        f"[speed @ {agent.step:,}] survived {r['survival_fraction']:.0%}; "
        f"feasible {r['qualified_fraction']:.0%}; full-horizon world-X speed "
        f"{r['selection_score']:.3f} m/s",
        flush=True,
    )
    return r
