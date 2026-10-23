"""Paired CPU MuJoCo replay of an untouched deployment bundle; no learning."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from play_joystick import Sim

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "controller/deploy/bundles/dash_joy3_lr_s2_180M.npz"
MODELS = {
    "original": ROOT / "RLframework/model/dash01_free.xml",
    "cad_v2": ROOT / "RLframework/model/dash01_cad_v2_free.xml",
}
OUT = ROOT / "report/data/policy-transfer01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_probes():
    """Small ablations, not a calibrated kinematic retargeting or a policy update."""
    results = []
    for mode in ["clamp_only", "reset_relative_and_clamp"]:
        for target in [0.25, 0.5, 0.75]:
            sim = Sim(BUNDLE, MODELS["cad_v2"])
            c = sim.ctrl
            limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
            c.q_lo, c.q_hi = limits[:, 0].copy(), limits[:, 1].copy()
            delta = np.zeros(6)
            if mode == "reset_relative_and_clamp":
                reset = sim.data.qpos[sim.act_q].copy()
                delta = reset - c.default_motor_pos
                c.nominal = c.nominal + delta
                c.default_motor_pos = reset
                c.gait_eval = type(c.gait_eval)(c.gp, c.nominal)
            q, qd, tau = sim._motor_state()
            c.start(q, qd, tau, sim._grav_body(), sim._gyro())
            sim.set_run(True)
            sim.set_stick(target / sim.v_max)
            c.set_speed(target, immediate=True)
            peak = np.zeros(6)
            poses = [sim.data.qpos.copy()]
            for _ in range(2000):
                sim.control_tick()
                poses.append(sim.data.qpos.copy())
                cmd = np.asarray(sim._last_cmd.target)
                peak = np.maximum(
                    peak, np.maximum(0.0, np.maximum(c.q_lo - cmd, cmd - c.q_hi))
                )
                if sim.fallen() or not np.isfinite(sim.data.qpos).all():
                    break
            stem = f"{mode}_{target:.2f}"
            np.savez_compressed(OUT / f"{stem}.npz", qpos=np.asarray(poses))
            item = dict(
                mode=mode,
                target_mps=target,
                elapsed_s=sim.t,
                survived=sim.t >= 20.0 - 1e-6 and not sim.fallen(),
                trace=stem + ".npz",
                reset_offset_rad=delta.tolist(),
                command_excess_max_rad=peak.tolist(),
            )
            results.append(item)
            print(json.dumps(item), flush=True)
    (OUT / "adapter-probes.json").write_text(
        json.dumps(
            dict(
                trials=results,
                note="Untouched network weights. Clamp-only replaces controller command bounds with CAD-v2 "
                "bounds. Reset-relative additionally shifts observation zero and nominal target by the "
                "reset-joint difference and restarts history. This heuristic is not a verified physical "
                "frame transform or linkage retargeting. Original paired replay is unchanged.",
            ),
            indent=2,
        )
        + "\n"
    )


def record(seconds, targets):
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    audits = {}
    for label, path in MODELS.items():
        sim = Sim(BUNDLE, path)
        m = sim.model
        joints = m.actuator_trnid[:, 0]
        names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)
        ]
        assert names == sim.bundle.meta["actuator_names"]
        assert sim.lock == "none"
        lo, hi = m.jnt_range[joints].T
        audits[label] = dict(
            model=str(path.relative_to(ROOT)),
            xml_sha256=sha(path),
            mass_kg=float(m.body_mass.sum()),
            actuator_names=names,
            motor_axes=m.jnt_axis[joints].tolist(),
            q_lo=lo.tolist(),
            q_hi=hi.tolist(),
            reset_motor_q=sim.data.qpos[sim.act_q].tolist(),
            bundle_observation_zero=sim.ctrl.default_motor_pos.tolist(),
            bundle_nominal_target=sim.ctrl.nominal.tolist(),
        )
        for target in targets:
            sim.set_stick(target / sim.v_max)
            sim.reset()
            sim.set_run(True)
            sim.ctrl.set_speed(target, immediate=True)
            poses = [sim.data.qpos.copy()]
            rows = []
            reason = None
            for _ in range(round(seconds / sim.control_dt)):
                sim.control_tick()
                q = sim.data.qpos[sim.act_q]
                cmd = np.asarray(sim._last_cmd.target)
                command_excess = np.maximum(0.0, np.maximum(lo - cmd, cmd - hi))
                actual_excess = np.maximum(0.0, np.maximum(lo - q, q - hi))
                rows.append(
                    [
                        sim.t,
                        sim.speed_world(),
                        sim.speed(),
                        sim.heading_deg(),
                        -sim._grav_body()[2],
                        *command_excess,
                        *actual_excess,
                    ]
                )
                poses.append(sim.data.qpos.copy())
                if (
                    not np.isfinite(sim.data.qpos).all()
                    or not np.isfinite(sim.data.qvel).all()
                ):
                    reason = "nonfinite physics"
                    break
                if sim.fallen():
                    reason = "tilt cosine < 0.3 or base height < 0.45 m"
                    break
            r = np.asarray(rows)
            late = r[:, 0] >= 5.0
            stem = f"{label}_{target:.2f}"
            np.savez_compressed(OUT / f"{stem}.npz", qpos=np.asarray(poses), metrics=r)
            item = dict(
                model=label,
                target_mps=target,
                trace=f"{stem}.npz",
                elapsed_s=sim.t,
                survived=reason is None,
                termination=reason,
                distance_world_x_m=float(sim.data.qpos[0] - poses[0][0]),
                mean_world_x_mps=float(r[:, 1].mean()),
                steady_world_x_mps=float(r[late, 1].mean()) if late.any() else None,
                steady_body_forward_mps=float(r[late, 2].mean())
                if late.any()
                else None,
                final_heading_deg=float(r[-1, 3]),
                command_outside_range_fraction=float(
                    (r[:, 5:11] > 1e-6).any(axis=1).mean()
                ),
                command_excess_max_rad=r[:, 5:11].max(axis=0).tolist(),
                actual_excess_max_rad=r[:, 11:17].max(axis=0).tolist(),
            )
            results.append(item)
            print(json.dumps(item), flush=True)
    record = dict(
        bundle=str(BUNDLE.relative_to(ROOT)),
        bundle_sha256=sha(BUNDLE),
        mujoco_version=mujoco.__version__,
        seconds=seconds,
        control_dt=0.01,
        targets=targets,
        model_audits=audits,
        trials=results,
        protocol="One deterministic nominal reset per model/target; no training or random perturbations. "
        "Original weights, observation normalization, gait decoding, command limits and 12 ms "
        "actuation delay retained. Each model uses its own reset keyframe and physical joint "
        "limits. No artificial base locks. Command excursions are measured, not silently remapped. "
        "CPU deployment replay, not MJX multi-trial evaluation. Termination uses the existing "
        "play_joystick fall check; not the SPEED-01 qualification criteria.",
    )
    (OUT / "results.json").write_text(json.dumps(record, indent=2) + "\n")


def render():
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    meta = json.loads((OUT / "results.json").read_text())
    video_dir = ROOT / "report/videos"
    video_dir.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    models, data, renderers = {}, {}, {}
    try:
        for label, path in MODELS.items():
            m = mujoco.MjModel.from_xml_path(str(path))
            m.vis.global_.offwidth, m.vis.global_.offheight = 640, 400
            m.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
            for side, rgba in [
                ("L", [0.15, 0.55, 1.0, 1.0]),
                ("R", [1.0, 0.4, 0.08, 1.0]),
            ]:
                body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_" + side)
                if body >= 0:
                    ids = m.geom_bodyid == body
                    m.geom_rgba[ids] = rgba
                    m.geom_matid[ids] = -1
            models[label], data[label] = m, mujoco.MjData(m)
            renderers[label] = mujoco.Renderer(m, height=400, width=640)
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation = 2.6, -12
        for target in meta["targets"]:
            trials = {
                x["model"]: x for x in meta["trials"] if x["target_mps"] == target
            }
            traces = {k: np.load(OUT / v["trace"]) for k, v in trials.items()}
            duration = max(t["elapsed_s"] for t in trials.values())
            path = video_dir / f"policy_transfer01_{target:.2f}.mp4"
            with imageio.get_writer(
                path, fps=25, codec="libx264", quality=7, macro_block_size=16
            ) as writer:
                for frame in range(round((duration + 2) * 25)):
                    t = frame / 25
                    canvas = Image.new("RGB", (1280, 800))
                    for col, label in enumerate(MODELS):
                        tr, m, d, ren = (
                            traces[label],
                            models[label],
                            data[label],
                            renderers[label],
                        )
                        k = min(round(t / 0.01), len(tr["qpos"]) - 1)
                        d.qpos[:] = tr["qpos"][k]
                        mujoco.mj_forward(m, d)
                        camera.lookat[:] = [d.qpos[0], d.qpos[1], 0.55]
                        for row, az in enumerate([90, 155]):
                            camera.azimuth = az
                            ren.update_scene(d, camera=camera)
                            canvas.paste(
                                Image.fromarray(ren.render()), (640 * col, 400 * row)
                            )
                        draw = ImageDraw.Draw(canvas)
                        x = 640 * col
                        draw.rectangle((x, 0, x + 640, 84), fill=(15, 20, 28))
                        draw.text(
                            (x + 12, 8),
                            f"{label} | target {target:.2f} m/s | same original policy",
                            font=font,
                        )
                        ended = t >= trials[label]["elapsed_s"]
                        status = (
                            (
                                "FELL - last pose held"
                                if not trials[label]["survived"]
                                else "Completed - last pose held"
                            )
                            if ended
                            else "Replay"
                        )
                        draw.text(
                            (x + 12, 37),
                            f"t={min(t, trials[label]['elapsed_s']):.2f}s | {status}",
                            font=small,
                            fill=(255, 180, 90) if ended else (255, 255, 255),
                        )
                        draw.text(
                            (x + 12, 60),
                            "Blue: left foot | Orange: right foot | real time",
                            font=small,
                        )
                    writer.append_data(np.asarray(canvas))
            for z in traces.values():
                z.close()
            print(path, flush=True)
    finally:
        for r in renderers.values():
            r.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--adapters", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    args = ap.parse_args()
    if args.adapters:
        adapter_probes()
    elif args.render:
        render()
    else:
        record(args.seconds, args.targets)
