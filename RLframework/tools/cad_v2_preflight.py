"""Gate the first articulated-foot training run on actual GPU environment behavior."""
import argparse
import json
from pathlib import Path
import sys
import time

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from config import get_config
from env import DashEnvV2, EnvParams


def physics_comparison(env, params):
    """Compare unassisted CPU/MJX motion before unstable balance modes diverge.

    Fixed-target stability is a controller result, distinct from valid physics.
    Three one-second traces include +/-0.001 rad pitch disturbances, no resets.
    Compare every coordinate and velocity with identical nominal model fields.
    """
    from plant import model_with
    from drive import pd_torque, torque_limit
    p, c = env.plant, env.cfg
    initial, _ = env.reset(jax.random.PRNGKey(123), params)
    draw = jax.tree_util.tree_map(lambda x: x[0], initial.draw)
    mx = model_with(p, draw.fields)
    q0 = np.repeat(p.key_qpos[None, :], 3, axis=0)
    q0[:, p.base_q["pitch"]] += [0, -.001, .001]
    cmd = jnp.asarray(p.nominal_ctrl)

    def one(q):
        d = p.data0.replace(qpos=q, qvel=jnp.zeros(p.nv), ctrl=jnp.zeros(6),
                            qacc_warmstart=jnp.zeros(p.nv))
        from mujoco import mjx
        d = mjx.forward(mx, d)
        def tick(d, _):
            speed = d.qvel[p.act_dadr]
            limit = torque_limit(speed, jnp.asarray(p.tau_peak), draw.torque_scale,
                                 jnp.asarray(c.motor_kt_joint), jnp.asarray(c.motor_r_ohm), c.motor_bus_volts)
            tau = pd_torque(d.qpos[p.act_qadr], speed, cmd, jnp.asarray(c.drive_kp),
                            jnp.asarray(c.drive_kd), limit)
            d = mjx.step(mx, d.replace(ctrl=tau))
            return d, jnp.concatenate([d.qpos, d.qvel])
        return jax.lax.scan(tick, d, None, length=1000)[1]

    gpu = np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(q0)))
    cpu = []
    # Apply precisely the same fields as the environment, including its friction convention.
    fields = jax.device_get(draw.fields)
    for name in fields._fields:
        owner = p.m.opt if name == "gravity" else p.m
        getattr(owner, name)[:] = np.asarray(getattr(fields, name))
    for q in q0:
        d = mujoco.MjData(p.m)
        d.qpos[:] = q.astype(np.float32)
        mujoco.mj_forward(p.m, d)
        rows = []
        for _ in range(1000):
            speed = d.qvel[p.act_dadr]
            kt, resistance = np.asarray(c.motor_kt_joint), np.asarray(c.motor_r_ohm)
            limit = np.minimum(p.tau_peak, kt*np.maximum(c.motor_bus_volts-kt*np.abs(speed), 0)/resistance)
            d.ctrl[:] = np.clip(np.asarray(c.drive_kp)*(np.asarray(cmd)-d.qpos[p.act_qadr])-
                                np.asarray(c.drive_kd)*speed, -limit, limit)
            mujoco.mj_step(p.m, d)
            rows.append(np.r_[d.qpos, d.qvel])
        cpu.append(rows)
    cpu = np.asarray(cpu)
    finite = bool(np.isfinite(gpu).all() and np.isfinite(cpu).all())
    q_error = float(np.max(np.abs(gpu[:, :, :p.nq]-cpu[:, :, :p.nq])))
    v_error = float(np.max(np.abs(gpu[:, :, p.nq:]-cpu[:, :, p.nq:])))
    report = dict(seconds=1., pitch_perturbations_rad=[0, -.001, .001], finite=finite,
                  max_coordinate_error=q_error, max_velocity_error=v_error,
                  coordinate_tolerance=.005, velocity_tolerance=.05,
                  passed=bool(finite and q_error < .005 and v_error < .05))
    return report, dict(cpu=cpu, gpu=gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate", choices=("standing", "physics"), default="standing",
                        help="physics permits learning active balance; standing requires stable zero action")
    args = parser.parse_args()
    cfg = get_config("cad_v2_gait01")
    env = DashEnvV2(cfg, n_envs=1)
    p = env.plant
    print(f"[preflight] {jax.devices()} nq={p.nq} nu={p.nu} loops={p.m.neq} loop_sites={len(p.loop_sites)}", flush=True)
    if p.nq != 20 or p.nu != 6 or p.m.neq != 4 or len(p.loop_sites) != 8:
        raise ValueError("Wrong model loaded")
    params = EnvParams.final(cfg)
    state, obs = env.reset(jax.random.PRNGKey(123), params)
    obs.block_until_ready()
    actions = jnp.zeros((1, env.action_dim))

    @jax.jit
    def hold(state):
        def tick(state, _):
            state, obs, reward, done, info = env.step(state, actions, params)
            row = dict(q=state.data.qpos[0], done=done[0], reward=reward[0],
                       finite=jnp.isfinite(obs).all() & jnp.isfinite(reward).all(),
                       term_nan=info["term_nan"][0],
                       upright=info["stop_upright"][0], flight=info["flight_fraction"][0])
            return state, row
        return jax.lax.scan(tick, state, None, length=500)

    start = time.monotonic()
    state, rows = hold(state)
    rows = jax.device_get(rows)
    elapsed = time.monotonic()-start
    q = rows["q"]
    height = q[:, p.base_q["z"]]
    tilt = np.degrees(np.arccos(np.clip(rows["upright"], -1, 1)))
    xy = q[:, [p.base_q["x"], p.base_q["y"]]]
    initial_xy = p.key_qpos[[p.base_q["x"], p.base_q["y"]]]
    sites = np.asarray(state.data.site_xpos[0])
    gap = max(np.linalg.norm(sites[p.loop_sites[i]]-sites[p.loop_sites[i+1]]) for i in range(0, 8, 2))
    report = dict(preset="cad_v2_gait01", devices=[str(d) for d in jax.devices()],
                  seconds=500*env.control_dt, elapsed_including_compile_s=elapsed,
                  finite=bool(np.all(rows["finite"])), terminated=bool(np.any(rows["done"])),
                  max_tilt_deg=float(np.max(tilt)), min_height_m=float(np.min(height)),
                  max_drift_xy_m=float(np.max(np.linalg.norm(xy-initial_xy, axis=1))),
                  final_loop_gap_m=float(gap), flight_fraction=float(np.mean(rows["flight"])),
                  observation_dim=int(obs.shape[-1]), action_dim=env.action_dim)
    report["standing_passed"] = bool(report["finite"] and not report["terminated"] and
                            report["max_tilt_deg"] < 5 and report["max_drift_xy_m"] < .03 and
                            report["min_height_m"] > p.height_stand-.02 and gap < .001)
    report["gate"] = args.gate
    comparison_rows = None
    if args.gate == "physics":
        report["physics_comparison"], comparison_rows = physics_comparison(env, params)
        report["passed"] = bool(report["physics_comparison"]["passed"] and
                                report["finite"] and not np.any(rows["term_nan"]))
    else:
        report["passed"] = report["standing_passed"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(out.with_suffix(".npz"), **rows)
    if comparison_rows is not None:
        np.savez_compressed(out.with_name("cpu-gpu-comparison.npz"), **comparison_rows)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(f"GPU {args.gate} gate failed; training not started")


if __name__ == "__main__":
    main()
