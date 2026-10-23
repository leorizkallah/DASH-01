"""Plot WALK-01 bilateral kinematics and optional 1 kHz MJX motor/force records."""

import json
from pathlib import Path
import numpy as np
import mujoco
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/experiments/walk01_20261019_v1"
OUT = ROOT / "report/data/walk01-diagnostics"
COLORS = ["#168bd2", "#ef8018"]
NAMES = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + ".png"), dpi=140)
    plt.close(fig)


def kinematics(qpos, model):
    d = mujoco.MjData(model)
    feet = [model.geom("foot_" + s + "_col").id for s in "LR"]
    heels = [model.geom("heel_" + s + "_col").id for s in "LR"]
    torso = model.body("torso").id
    xy, clear = [], []
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        R = d.xmat[torso].reshape(3, 3)
        xy.append((d.geom_xpos[feet] - d.xpos[torso]) @ R)
        h = []
        for f, b in zip(feet, heels):
            h.append(
                min(
                    d.geom_xpos[g, 2]
                    - np.abs(d.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
                    for g in [f, b]
                )
            )
        clear.append(h)
    return np.asarray(xy), np.asarray(clear)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((DATA / "resolved_config.json").read_text())["config"]
    model = mujoco.MjModel.from_xml_path(str(ROOT / "RLframework" / cfg["model_path"]))
    qad = [model.jnt_qposadr[model.joint(n).id] for n in NAMES]
    sections, summary = [], {}
    for label in ["best", "final"]:
        tr = np.load(DATA / f"walk01_trace_{label}.npz")
        dgpath = OUT / (label + ".npz")
        dg = np.load(dgpath) if dgpath.exists() else None
        dm = json.loads(str(dg["meta"].item())) if dg is not None else None
        for i in [1, 0, 2]:
            target = [0.25, 0.75, 1.5][i]
            name = f"{label}_{target:.2f}"
            q = tr["qpos"][:, i]
            feet, clear = kinematics(q, model)
            joints = q[:, qad]
            n = len(q)
            if dg is not None:
                ends = np.flatnonzero(dg["done"][:, i])
                n = int(ends[0]) if ends.size else len(dg["done"])
                if n <= 500:
                    raise ValueError(
                        "Diagnostic falls before analysis window; inspect raw record"
                    )
                # Diagnostic and visual rollouts may diverge numerically after instrumentation.
                q = dg["qpos"][:n, i]
                joints = q[:, qad]
                feet = dg["feet"][:n, i]
                clear = dg["clearance"][:n, i]
            t = np.arange(n) * 0.01
            ss = slice(500, n)
            row = dict(
                checkpoint=label,
                target=target,
                seconds=n * 0.01,
                motor_range_deg=np.rad2deg(np.ptp(joints[ss], axis=0)).tolist(),
                foot_x_range_m=np.ptp(feet[ss, :, 0], axis=0).tolist(),
                foot_z_range_m=np.ptp(feet[ss, :, 2], axis=0).tolist(),
                clearance_p95_m=np.percentile(clear[ss], 95, axis=0).tolist(),
                clear_over_5mm_fraction=(clear[ss] > 0.005).mean(axis=0).tolist(),
            )
            fig, ax = plt.subplots(1, 3, figsize=(13, 4))
            for f, color in enumerate(COLORS):
                ax[0].plot(
                    feet[ss, f, 0],
                    feet[ss, f, 2],
                    color=color,
                    alpha=0.7,
                    label="LR"[f],
                )
                win = (t >= 10) & (t < 12)
                ax[1].plot(t[win], clear[win, f] * 1000, color=color, label="LR"[f])
                if dg is not None:
                    s = np.clip(dg["asymmetry"][:n, i], -1, 1)
                    ax[2].plot(
                        t, 1 + s if f == 0 else 1 - s, color=color, label="LR"[f]
                    )
                else:
                    ax[2].plot(
                        t[win],
                        joints[win, 1 + 3 * f],
                        color=color,
                        label="cam " + "LR"[f],
                    )
            ax[0].set(
                xlabel="Forward x in torso frame (m)",
                ylabel="Vertical z in torso frame (m)",
                title="Visited toe workspace, 5–25 s",
            )
            ax[0].axis("equal")
            ax[1].set(
                xlabel="Time (s)",
                ylabel="Lowest sole point (mm)",
                title="Foot clearance, 10–12 s",
            )
            ax[2].set(
                xlabel="Time (s)",
                ylabel="Multiplier" if dg is not None else "Joint angle (rad)",
                title="Generator amplitude: 1+s / 1−s"
                if dg is not None
                else "Cam motion",
            )
            for a in ax:
                a.legend()
                a.grid(alpha=0.2)
            fig.suptitle(f"{label} checkpoint: target {target:.2f} m/s")
            save(fig, name + "_workspace")
            images = [name + "_workspace"]
            if dg is not None:
                # Saved arrays are [control tick, environment, substep, motor/foot, ...].
                tau = dg["tau"][:n, i].reshape(-1, 6)
                qd = dg["qd"][:n, i].reshape(-1, 6)
                qm = dg["q"][:n, i].reshape(-1, 6)
                lim = dg["limit"][:n, i].reshape(-1, 6)
                grf = dg["grf"][:n, i].reshape(-1, 2, 3)
                assert np.isfinite(tau).all() and np.isfinite(grf).all()
                tt = np.arange(len(tau)) * 0.001
                steady = tt >= 5
                win = (tt >= 10) & (tt < 12)
                kt = np.asarray(dm["kt"])
                current = tau / kt
                row.update(
                    current_rms_A=np.sqrt(
                        np.mean(current[steady] ** 2, axis=0)
                    ).tolist(),
                    current_peak_A=np.max(np.abs(current[steady]), axis=0).tolist(),
                    torque_rms_Nm=np.sqrt(np.mean(tau[steady] ** 2, axis=0)).tolist(),
                    torque_peak_Nm=np.max(np.abs(tau[steady]), axis=0).tolist(),
                    saturation_fraction=(
                        np.abs(tau[steady]) >= 0.98 * np.maximum(lim[steady], 1e-6)
                    )
                    .mean(axis=0)
                    .tolist(),
                    speed_rms_rad_s=np.sqrt(np.mean(qd[steady] ** 2, axis=0)).tolist(),
                    positive_mech_power_W=np.maximum(tau[steady] * qd[steady], 0)
                    .mean(axis=0)
                    .tolist(),
                    mean_grf_N=grf[steady].mean(axis=0).tolist(),
                    peak_vertical_grf_N=grf[steady, :, 2].max(axis=0).tolist(),
                    contact_force_duty=(grf[steady, :, 2] > 5).mean(axis=0).tolist(),
                    generator_asymmetry_mean=float(dg["asymmetry"][ss, i].mean()),
                    generator_asymmetry_min=float(dg["asymmetry"][ss, i].min()),
                    generator_asymmetry_max=float(dg["asymmetry"][ss, i].max()),
                    thermal_max=dg["thermal"][ss, i].max(axis=0).tolist(),
                    reward_means={
                        k.removeprefix("reward_"): float(dg[k][ss, i].mean())
                        for k in dg.files
                        if k.startswith("reward_")
                    },
                    replay_qpos_max_difference=float(
                        np.max(np.abs(q - tr["qpos"][:n, i]))
                    ),
                )
                fig, ax = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
                for family in range(3):
                    for side, color in enumerate(COLORS):
                        j = family + side * 3
                        for col, values in enumerate([np.rad2deg(qm), tau, current]):
                            ax[family, col].plot(
                                tt[win],
                                values[win, j],
                                color=color,
                                lw=0.8,
                                label="LR"[side],
                            )
                    for col, unit in enumerate(
                        ["Angle (deg)", "Torque (Nm)", "Estimated current (A)"]
                    ):
                        ax[family, col].set(title=NAMES[family][:-2], ylabel=unit)
                        ax[family, col].grid(alpha=0.2)
                for a in ax[-1]:
                    a.set_xlabel("Time (s)")
                ax[0, 0].legend()
                fig.suptitle(f"{name}: all six motors, actual 1 kHz samples")
                save(fig, name + "_motors")
                images.append(name + "_motors")
                fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                for dim, title in enumerate(
                    ["Fore-aft Fx", "Lateral Fy", "Vertical Fz"]
                ):
                    for f, color in enumerate(COLORS):
                        ax[dim].plot(
                            tt[win],
                            grf[win, f, dim],
                            color=color,
                            lw=0.8,
                            label="LR"[f],
                        )
                    ax[dim].set(ylabel=title + " (N)")
                    ax[dim].grid(alpha=0.2)
                ax[2].axhline(
                    dm["mass"] * 9.81, color="gray", ls="--", label="Whole robot weight"
                )
                ax[2].legend()
                ax[0].legend()
                ax[2].set_xlabel("Time (s)")
                fig.suptitle(
                    f"{name}: floor-on-foot reaction, world axes; all toe + heel contacts"
                )
                save(fig, name + "_forces")
                images.append(name + "_forces")
                fig, ax = plt.subplots(3, 2, figsize=(11, 10))
                for family in range(3):
                    for side in range(2):
                        j = family + side * 3
                        a = ax[family, side]
                        a.scatter(
                            np.abs(qd[steady, j])[::10],
                            np.abs(tau[steady, j])[::10],
                            s=2,
                            alpha=0.2,
                            color=COLORS[side],
                        )
                        xx = np.linspace(
                            0,
                            max(dm["bus_volts"] / kt[j], np.max(np.abs(qd[steady, j]))),
                            200,
                        )
                        yy = np.minimum(
                            dm["torque_peak"][j],
                            kt[j]
                            * np.maximum(dm["bus_volts"] - kt[j] * xx, 0)
                            / dm["resistance"][j],
                        )
                        a.plot(xx, yy, "k--", label="Drive limit")
                        a.axhline(
                            dm["torque_cont"][j],
                            color="red",
                            ls=":",
                            label="Continuous rating",
                        )
                        a.set(
                            title=NAMES[j],
                            xlabel="|joint speed| (rad/s)",
                            ylabel="|torque| (Nm)",
                        )
                        a.grid(alpha=0.2)
                ax[0, 0].legend()
                fig.suptitle(
                    f"{name}: torque-speed use, 5–25 s (display subsampled; stats use all samples)"
                )
                save(fig, name + "_envelope")
                images.append(name + "_envelope")
            summary[name] = row
            sections.append(
                "<section><h2>"
                + name.replace("_", " — target ")
                + " m/s</h2>"
                + "".join(
                    f'<img loading="lazy" src="data/walk01-diagnostics/{im}.png">'
                    for im in images
                )
                + "</section>"
            )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    html = """<!doctype html><meta charset="utf-8"><title>WALK-01 leg and motor diagnostics</title>
<style>body{font:17px/1.5 system-ui;max-width:1350px;margin:30px auto;padding:20px}img{width:100%}section{margin:50px 0}a{color:#1674ac}</style>
<h1>WALK-01: both legs and all six motors</h1>
<p>Blue = left; orange = right. Plots use the supplied CAD-v2 homing model.
Workspace means positions actually visited by this policy, not the complete reachable envelope.
Joint angles use the model's native signs; opposite signs can be physically symmetric.
Forces are ground-on-foot, in world coordinates (positive z upward); forces the foot exerts
on the ground have the opposite sign. Current is estimated phase/joint-equivalent current tau/Kt,
not measured battery current. Summaries use 5–25 s; time-series panels zoom to 10–12 s.</p>
<p><a href="../docs/flatfeet-symmetry-v1.md">Reward changes and baseline findings</a> · <a href="walk01-videos.html">Videos</a> ·
<a href="data/walk01-diagnostics/summary.json">All numerical results</a></p>""" + "".join(
        sections
    )
    (ROOT / "report/walk01-diagnostics.html").write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                k: {
                    x: v[x]
                    for x in [
                        "motor_range_deg",
                        "foot_x_range_m",
                        "clear_over_5mm_fraction",
                    ]
                }
                for k, v in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
