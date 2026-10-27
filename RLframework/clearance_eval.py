"""Nominal first-episode height evaluation; keep clearance candidates separately from speed keepers."""

import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from env import EnvParams
from train import make_eval_env


class ClearanceEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = make_eval_env(cfg, n=12)
        self.fn = None

    def evaluate(self, agent):
        env, c = self.env, self.cfg
        ticks = round(c.eval_seconds / env.control_dt)
        if self.fn is None:

            def rollout(weights, mean, var, count):
                from ppo import ObsStats

                stats = ObsStats(mean=mean, var=var, count=count)
                params = EnvParams.final(c)._replace(
                    dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0
                )
                state, obs = env.reset(jax.random.PRNGKey(1000), params)

                def tick(carry, _):
                    s, ob, alive = carry
                    a = jnp.clip(
                        agent._act_greedy(weights, stats.normalize(ob)), -1.0, 1.0
                    )
                    nxt, ob2, reward, done, info = env.step(s, a, params)
                    f = info["flatfeet"]
                    row = dict(
                        alive=alive,
                        done=done,
                        fallen=info["fallen"],
                        symmetry=info["term_symmetry"],
                        height=f["clearance"],
                        grounded=f["grounded"],
                        middle=f["clearance_middle"],
                        slip=f["slip"],
                        error=info["track_err"],
                        reward=reward,
                        floor=info["reward_floor_hit"],
                        valid_step=f["valid_step"],
                    )
                    return (nxt, ob2, alive & ~done), row

                return jax.lax.scan(
                    tick, (state, obs, jnp.ones(env.n_envs, bool)), None, length=ticks
                )[1]

            self.fn = jax.jit(rollout)
        r = jax.device_get(
            self.fn(agent.params, agent.stats.mean, agent.stats.var, agent.stats.count)
        )
        live = r["alive"] & (np.arange(ticks)[:, None] >= c.eval_warm_ticks)
        air = live[..., None] & ~r["grounded"]
        middle = live[..., None] & r["middle"]
        h = np.maximum(r["height"], 0.0)
        air_mean = np.where(air, h, 0.0).sum(axis=0) / np.maximum(air.sum(axis=0), 1)
        mid_covered = np.where(middle, h >= c.swing_clearance_min_m, False).sum(
            axis=0
        ) / np.maximum(middle.sum(axis=0), 1)
        mid_deficit = np.where(
            middle,
            np.clip((c.swing_clearance_min_m - h) / c.swing_clearance_min_m, 0, 1),
            0.0,
        ).sum(axis=0) / np.maximum(middle.sum(axis=0), 1)
        air_peak = np.where(air, h, 0.0).max(axis=0)
        air_over_cap = np.where(air, h > c.swing_clearance_bonus_cap_m, False).sum(
            axis=0
        ) / np.maximum(air.sum(axis=0), 1)
        excess_mean = np.where(
            air, np.maximum(h - c.swing_clearance_bonus_cap_m, 0.0), 0.0
        ).sum(axis=0) / np.maximum(air.sum(axis=0), 1)
        upper_enabled = c.w_excess_swing_clearance > 0
        error = np.where(live, r["error"], 0.0).sum(axis=0) / np.maximum(
            live.sum(axis=0), 1
        )
        error = np.where(live.any(axis=0), error, 1.5)
        counts = r["alive"].sum(axis=0)
        steps = np.where(live[..., None], r["valid_step"], False).sum(axis=0)
        falls = (r["fallen"] & r["alive"]).any(axis=0)
        symmetry = (r["symmetry"] & r["alive"]).any(axis=0)
        survived = (counts >= ticks) & ~falls & ~symmetry
        qualified = (
            survived
            & (error <= 0.15)
            & (air_mean.min(axis=1) >= c.swing_clearance_min_m)
            & (mid_covered.min(axis=1) >= 0.8)
            & (steps.min(axis=1) >= 8)
            & (
                (air_peak.max(axis=1) <= c.swing_clearance_bonus_cap_m + 0.001)
                | (not upper_enabled)
            )
        )
        result = dict(
            step=int(agent.step),
            target_mps=1.5,
            seed=1000,
            trials=env.n_envs,
            survival_fraction=float(survived.mean()),
            qualified_fraction=float(qualified.mean()),
            keeper_eligible=bool(qualified.mean() >= 0.9),
            track_error_mps=float(error.mean()),
            mean_air_clearance_mm=(air_mean.mean(axis=0) * 1000).tolist(),
            minimum_clearance_mm=c.swing_clearance_min_m * 1000,
            bonus_cap_mm=c.swing_clearance_bonus_cap_m * 1000,
            upper_height_penalty_enabled=upper_enabled,
            upper_qualification_tolerance_mm=1.0 if upper_enabled else None,
            midswing_ge_min_fraction=mid_covered.mean(axis=0).tolist(),
            peak_air_clearance_mm=(air_peak.max(axis=0) * 1000).tolist(),
            air_over_cap_fraction=air_over_cap.mean(axis=0).tolist(),
            mean_excess_air_mm=(excess_mean.mean(axis=0) * 1000).tolist(),
            midswing_mean_deficit_fraction=mid_deficit.mean(axis=0).tolist(),
            actual_reward_mean=float(
                np.where(r["alive"], r["reward"], 0.0).sum() / max(r["alive"].sum(), 1)
            ),
            reward_floor_fraction=float(
                np.where(r["alive"], r["floor"], False).sum() / max(r["alive"].sum(), 1)
            ),
            per_trial=dict(
                seconds=(counts * env.control_dt).tolist(),
                survived=survived.tolist(),
                qualified=qualified.tolist(),
                mean_air_clearance_mm=(air_mean * 1000).tolist(),
                midswing_ge_min_fraction=mid_covered.tolist(),
                peak_air_clearance_mm=(air_peak * 1000).tolist(),
                air_over_cap_fraction=air_over_cap.tolist(),
                track_error_mps=error.tolist(),
                symmetry_terminated=symmetry.tolist(),
                valid_steps_after_warmup=steps.tolist(),
            ),
            note=f"12 nominal first episodes; warmup {c.eval_warm_ticks * env.control_dt:g} s. "
            f"Qualified: full survival, error <=0.15 m/s, both mean air heights >={c.swing_clearance_min_m * 1000:g} mm, "
            "both scheduled mid-swing threshold fractions >=80%, at least 8 valid completed cycles per foot. "
            + (
                f"Both observed airborne peaks <= {c.swing_clearance_bonus_cap_m * 1000 + 1:g} mm (cap + 1 mm tolerance)."
                if upper_enabled
                else "No upper-height criterion for this legacy preset."
            ),
        )
        # Threshold-specific aliases remain accurate for readers of historical files.
        threshold_key = f"midswing_ge{c.swing_clearance_min_m * 1000:g}mm_fraction"
        result[threshold_key] = result["midswing_ge_min_fraction"]
        result["per_trial"][threshold_key] = result["per_trial"][
            "midswing_ge_min_fraction"
        ]
        result["candidate_rank"] = [
            result["survival_fraction"],
            -float(np.maximum(error - 0.15, 0.0).mean()),
            float(np.clip(steps.min(axis=1) / 8.0, 0.0, 1.0).mean()),
            float(air_mean.mean(axis=0).min()),
            float(mid_covered.mean(axis=0).min()),
        ]
        if upper_enabled:
            # Do not prefer ever higher mean lift, as the old one-sided selector did.
            result["candidate_rank"] = result["candidate_rank"][:3] + [
                -float(
                    c.w_low_swing_clearance * mid_deficit.mean()
                    + c.w_excess_swing_clearance
                    * excess_mean.mean()
                    / max(c.swing_clearance_bonus_cap_m - c.swing_clearance_min_m, 1e-6)
                ),
                float(mid_covered.mean(axis=0).min()),
                float(np.minimum(air_mean, c.swing_clearance_min_m).mean(axis=0).min()),
            ]
        assert np.isfinite(result["actual_reward_mean"]) and np.isfinite(air_mean).all()
        return result


def record(agent, evaluator, run):
    r = evaluator.evaluate(agent)
    run = Path(run)
    for name, eligible in [
        ("best_clearance_candidate", True),
        ("best_clearance", r["keeper_eligible"]),
    ]:
        if not eligible:
            continue
        path = run / (name + "_eval.json")
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous is None or r["candidate_rank"] > previous["candidate_rank"]:
            agent.save(run / (name + ".msgpack"))
            path.write_text(json.dumps(r, indent=2) + "\n")
    path = run / "clearance_baseline.json"
    if not path.exists():
        path.write_text(json.dumps(r, indent=2) + "\n")
    (run / "latest_clearance_eval.json").write_text(json.dumps(r, indent=2) + "\n")
    with (run / "clearance_evaluations.jsonl").open("a") as out:
        out.write(json.dumps(r) + "\n")
    print(
        f"[clearance @ {agent.step:,}] survived {r['survival_fraction']:.0%}; "
        f"mean air L/R {r['mean_air_clearance_mm']} mm; qualified {r['qualified_fraction']:.0%}",
        flush=True,
    )
    return r
