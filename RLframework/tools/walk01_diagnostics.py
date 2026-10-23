"""Read-only WALK-01 diagnostics: instrument a copy of frozen env.py, never training.

All motor and ground reaction samples are captured at 1 kHz. The contact decoder
is restricted to this model's elliptic cone. Forces are floor-on-foot, world XYZ.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def ground_forces(data, plant, xp):
    impl = data._impl if hasattr(data, "_impl") else data
    con = impl.contact
    addr = con.efc_address
    f = impl.efc_force[xp.maximum(addr, 0)[:, None] + xp.arange(3)]
    world = xp.einsum("nji,nj->ni", con.frame.reshape((-1, 3, 3)), f)
    pair = con.geom
    out = []
    for side in range(2):
        foot = (pair == plant.foot_gids[side]) | (pair == plant.heel_gids[side])
        floor0, floor1 = pair[:, 0] == plant.floor_gid, pair[:, 1] == plant.floor_gid
        valid = (addr >= 0) & ((floor0 & foot[:, 1]) | (floor1 & foot[:, 0]))
        sign = xp.where(floor0, 1.0, -1.0)
        out.append(xp.sum(xp.where(valid[:, None], world * sign[:, None], 0.0), axis=0))
    return xp.stack(out)


def instrument(source, destination):
    text = source.read_text()
    edits = [
        (
            "import numpy as np",
            "import numpy as np\nfrom walk01_diagnostics import ground_forces",
        ),
        (
            "return (d, tau_sq + tau ** 2, con_acc | toe_c | heel_c, tau, flight_ticks), None",
            "return (d, tau_sq + tau ** 2, con_acc | toe_c | heel_c, tau, flight_ticks), "
            "dict(tau=tau, qd=qd, q=q, limit=lim, target=live[:6], grf=ground_forces(d, p, jnp))",
        ),
        (
            "(data, tau_sq, contact_acc, tau_last, flight_ticks), _ = lax.scan(",
            "(data, tau_sq, contact_acc, tau_last, flight_ticks), diagnostics = lax.scan(",
        ),
        (
            "commit=commit, reward_terms=terms, foot_air=",
            "diagnostics=diagnostics, commit=commit, reward_terms=terms, foot_air=",
        ),
    ]
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    destination.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.run.resolve().parents[1]
    sys.path.insert(0, str(src))
    import jax
    import jax.numpy as jnp
    import mujoco

    patched = args.out / "env_instrumented.py"
    instrument(src / "env.py", patched)
    spec = importlib.util.spec_from_file_location("env", patched)
    module = importlib.util.module_from_spec(spec)
    # Preserve frozen model/library paths used relative to env.__file__.
    module.__file__ = str(src / "env.py")
    sys.modules["env"] = module
    spec.loader.exec_module(module)
    from evaluate import load_run
    from env import EnvParams
    from ppo import ObsStats
    import gait

    targets = jnp.asarray([0.25, 0.75, 1.5])
    cfg, env, agent = load_run(args.run, args.run / "best.msgpack", n_envs=3)
    p = env.plant
    assert p.m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    params = EnvParams.final(cfg)._replace(
        dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0
    )

    @jax.jit
    def simulate(weights, mean, var, count):
        stats = ObsStats(mean=mean, var=var, count=count)
        state, obs = env.reset(jax.random.PRNGKey(1000), params)
        state = state.replace(
            v_cmd=targets,
            cmd_left=jnp.full_like(state.cmd_left, 1e9),
            stop_cmd=jnp.zeros_like(state.stop_cmd),
        )
        once = jax.vmap(env._once, in_axes=(0, None, 0, 0))(
            state, params, state.phase, state.commit.astype(jnp.float32)
        )
        start = state.hist[:, env.hist_idx].reshape((3, -1)).shape[1]
        obs = obs.at[:, start : start + once.shape[1]].set(once)

        def tick(carry, _):
            s, ob = carry
            a = jnp.clip(agent._act_greedy(weights, stats.normalize(ob)), -1.0, 1.0)
            nxt, ob2, reward, done, info = env.step(s, a, params)
            R = s.data.xmat[:, p.base_bid].reshape((-1, 3, 3))
            toe = s.data.geom_xpos[:, p.foot_gids]
            relative = jnp.einsum(
                "nji,nfj->nfi", R, toe - s.data.xpos[:, p.base_bid, None]
            )
            row = dict(
                qpos=s.data.qpos,
                feet=relative,
                foot_world=toe,
                clearance=jax.vmap(env._foot_heights)(s.data),
                grounded=s.grounded_prev,
                thermal=s.thermal_x,
                asymmetry=s.spec[:, gait.I_S],
                action=a,
                done=done,
                fallen=info["fallen"],
                speed=info["v_body_x"],
                terms=info["reward_terms"],
                reward=reward,
                freq=info["freq_hz"],
                **info["diagnostics"],
            )
            return (nxt, ob2), row

        return jax.lax.scan(tick, (state, obs), None, length=2500)[1]

    for checkpoint in ["best", "final"]:
        agent.load(args.run / (checkpoint + ".msgpack"))
        rows = jax.device_get(
            simulate(agent.params, agent.stats.mean, agent.stats.var, agent.stats.count)
        )
        terms = rows.pop("terms")
        rows.update({"reward_" + k: v for k, v in terms.items()})
        meta = dict(
            checkpoint=checkpoint,
            step=agent.step,
            targets=np.asarray(targets).tolist(),
            dt=env.control_dt,
            physics_dt=p.sim_dt,
            seed=1000,
            motor_names=p.actuator_names(),
            kt=cfg.motor_kt_joint,
            resistance=cfg.motor_r_ohm,
            bus_volts=cfg.motor_bus_volts,
            torque_peak=p.tau_peak.tolist(),
            torque_cont=cfg.thermal_tau_cont,
            q_lo=p.q_lo.tolist(),
            q_hi=p.q_hi.tolist(),
            mass=p.total_mass,
            ws_ref=p.ws_ref.tolist(),
            model=cfg.model_path,
            workspace=[
                cfg.workspace_dx_max,
                cfg.workspace_dz_min,
                cfg.workspace_dz_max,
            ],
            source_env_sha256=hashlib.sha256((src / "env.py").read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256(
                (args.run / (checkpoint + ".msgpack")).read_bytes()
            ).hexdigest(),
            note="1 kHz actual applied torque and elliptic-contact world ground reaction; "
            "current is simulated joint-equivalent tau/Kt, not measured battery current. "
            "Geometry at tick start; rewards describe following transition. "
            "Discard termination tick if auto-reset occurs.",
        )
        np.savez_compressed(
            args.out / (checkpoint + ".npz"), meta=json.dumps(meta), **rows
        )
        (args.out / (checkpoint + ".json")).write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        print(
            checkpoint,
            "saved",
            rows["tau"].shape,
            "falls",
            rows["fallen"].sum(),
            flush=True,
        )


if __name__ == "__main__":
    main()
