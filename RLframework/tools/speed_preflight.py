"""Reward-budget probes; never supplies weights to the subsequent training process."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jnp
from config import get_config
from env import DashEnvV2, EnvParams
from ppo import PPO, ObsStats, initial_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    a = ap.parse_args()
    cfg = get_config("cad_v2_speed01")
    env = DashEnvV2(cfg, n_envs=16)
    agent = PPO(
        cfg, env, a.out.parent / "_budget", cfg.total_steps, seed=0, eval_env=None
    )
    bound = cfg.step_reward_floor * env.reward_dt_scale / (1 - cfg.gamma)
    assert cfg.fall_penalty > bound
    assert not env.plant.planar
    assert all(
        env.plant.base_q[k] >= 0 for k in ("x", "y", "z", "roll", "pitch", "yaw")
    )

    @jax.jit
    def probe(weights, mean, var, count, params, mode):
        stats = ObsStats(mean=mean, var=var, count=count)
        state, obs = env.reset(jax.random.PRNGKey(431), params)

        def tick(carry, _):
            s, ob, active, key = carry
            key, sub = jax.random.split(key)
            # Reference GAIT-09 expects its original 0.25 m/s task feature (.5).
            # This overrides only reference policy inputs; physics/rewards are SPEED-01.
            ob = jnp.where(mode == 2, ob.at[:, env.actor_dim - 3].set(0.5), ob)
            mu = agent._act_greedy(weights, stats.normalize(ob))
            action = jnp.where(
                mode == 0,
                jnp.zeros_like(mu),
                jnp.where(mode == 1, mu + 0.7 * jax.random.normal(sub, mu.shape), mu),
            )
            nxt, ob2, reward, done, info = env.step(
                s, jnp.clip(action, -1.0, 1.0), params
            )
            row = dict(
                active=active,
                reward=reward,
                fallen=info["fallen"],
                vx=info["speed_metrics"]["vx"],
                floor=info["reward_floor_hit"],
                terms=info["reward_terms"],
            )
            return (nxt, ob2, active & ~done, key), row

        return jax.lax.scan(
            tick,
            (state, obs, jnp.ones(env.n_envs, bool), jax.random.PRNGKey(53)),
            None,
            length=600,
        )[1]

    def measure(label, params, mode):
        r = jax.device_get(
            probe(
                agent.params,
                agent.stats.mean,
                agent.stats.var,
                agent.stats.count,
                params,
                mode,
            )
        )
        live = r["active"]
        nonterminal = live & ~r["fallen"]
        average = lambda v, mask: float(
            np.where(mask, v, 0.0).sum() / max(mask.sum(), 1)
        )
        item = dict(
            label=label,
            finite=bool(np.isfinite(r["reward"]).all()),
            mean_first_episode_s=float(live.sum(axis=0).mean() * env.control_dt),
            mean_speed=average(r["vx"], live),
            nonterminal_reward=average(r["reward"], nonterminal),
            floor_fraction=average(r["floor"], live),
            discounted_return=float(
                (
                    np.where(live, r["reward"], 0.0)
                    * cfg.gamma ** np.arange(600)[:, None]
                )
                .sum(axis=0)
                .mean()
            ),
            reward_terms={k: average(v, live) for k, v in r["terms"].items()},
        )
        print(json.dumps(item), flush=True)
        assert item["finite"], label
        return item

    results = [
        measure(
            "zero-action nominal controller; not guaranteed standing",
            initial_params(cfg),
            0,
        ),
        measure("fresh policy sampled with std 0.7", initial_params(cfg), 1),
        measure("zero action at full regularization", EnvParams.final(cfg), 0),
    ]
    reference = None
    if a.reference:
        agent.load(a.reference, warm_start=False)
        reference = dict(
            path=str(a.reference),
            sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
            note="Diagnostic only; task feature pinned to old value. Never training initialization.",
        )
        results.append(
            measure(
                "archived GAIT-09 motion under full SPEED-01 reward",
                EnvParams.final(cfg),
                2,
            )
        )
    record = dict(
        preset="cad_v2_speed01",
        actor_dim=env.actor_dim,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        discounted_worst_living_cost=bound,
        fall_penalty=cfg.fall_penalty,
        reference=reference,
        probes=results,
        note="Six-second first-episode probes; cold failures are expected. Same-state speed monotonicity "
        "and delayed-fall ranking are regression-tested, not guarantees of learning.",
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
