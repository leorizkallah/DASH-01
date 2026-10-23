"""Completed flat-feet run: comparison plots, bilateral diagnostics and replay gallery."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
DATA = ROOT / "docs/experiments/dash01_flatfeet_symmetry_v1"
OLD = ROOT / "docs/experiments/walk01_20261019_v1"
COLORS = ["#168bd2", "#ef8018"]


def read(path):
    return list(csv.DictReader(path.open()))


def col(rows, k):
    return np.array([float(r[k]) for r in rows])


def save(fig, name):
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(REPORT / f"figures/{name}.{ext}", dpi=150)
    plt.close(fig)


def main():
    ev, old, pr = (
        read(DATA / "eval.csv"),
        read(OLD / "eval.csv"),
        read(DATA / "progress.csv"),
    )
    best = json.loads((DATA / "best_eval.json").read_text())
    oldbest = json.loads((OLD / "best_eval.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for rows, label in [(old, "WALK-01"), (ev, "Flat-feet v1")]:
        axs[0].plot(col(rows, "step") / 1e6, 100 * col(rows, "alive_frac"), label=label)
        axs[1].plot(col(rows, "step") / 1e6, col(rows, "track_err"), label=label)
    axs[0].set(
        ylabel="Full-duration trials (%)",
        ylim=(-3, 103),
        title="Fixed seven-speed evaluation, 28 trials",
    )
    axs[1].set(ylabel="Mean speed error (m/s)", title="Same nominal command ladder")
    for ax in axs:
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_xlabel("Training steps (millions)")
    save(fig, "flatfeet_learning")
    fig, ax = plt.subplots(figsize=(9, 4))
    for data, label in [(oldbest, "WALK-01 best"), (best, "Flat-feet v1 best")]:
        vals = np.asarray(data["per_cmd"])
        ax.plot(vals[:, 0], vals[:, 1], "-o", label=label)
    ax.set(
        xlabel="Target speed (m/s)",
        ylabel="Mean absolute speed error (m/s)",
        title="Saved best checkpoints, not an isolated reward ablation",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "flatfeet_speed_comparison")
    # Bin adjacent logged updates for legible training trends; no interpolation across missing data.
    blocks = [pr[i : i + 50] for i in range(0, len(pr), 50)]
    avg = lambda k: np.array([col(b, k).mean() for b in blocks])
    x = avg("time/env_steps") / 1e6
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for i, side in enumerate(["left", "right"]):
        for ax, key, title, unit in [
            (axs[0, 0], "grounded", "Any-foot support", "Fraction of control ticks"),
            (
                axs[0, 1],
                "flat",
                "Simultaneous heel + toe support",
                "Fraction of control ticks",
            ),
            (
                axs[0, 2],
                "slip",
                "Contact-point slip (includes zero when airborne)",
                "m/s",
            ),
            (
                axs[1, 0],
                "valid_step",
                "Completed valid cycles",
                "Events / foot / second",
            ),
        ]:
            ax.plot(
                x,
                avg("flatfeet/" + side + "_" + key)
                * (100 if key == "valid_step" else 1),
                color=COLORS[i],
                label=side,
            )
            ax.set(title=title, ylabel=unit)
            ax.legend()
    axs[1, 1].plot(x, avg("flatfeet/error"), label="Mean timing error E")
    axs[1, 1].axhline(
        0.4, color="gray", ls="--", label="Persistent termination threshold"
    )
    axs[1, 1].set(
        title="Timing asymmetry (ineligible ticks contribute zero)",
        ylabel="Normalized error",
    )
    axs[1, 1].legend()
    axs[1, 2].plot(x, 100 * avg("flatfeet/gate"), label="Termination gate active")
    axs[1, 2].plot(
        x, 100 * avg("flatfeet/reward_floor_fraction"), label="Reward floor hit"
    )
    axs[1, 2].set(title="Curriculum / clipping", ylabel="% of control ticks")
    axs[1, 2].legend()
    for ax in axs.flat:
        ax.grid(alpha=0.2)
        ax.set_xlabel("Training steps (millions)")
    save(fig, "flatfeet_bilateral_training")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for key in ["track", "air_time", "foot_slip", "symmetry", "flat_support"]:
        axs[0].plot(x, avg("reward_terms/" + key), label=key)
    axs[0].set(
        title="Logged dense reward contributions",
        ylabel="Before global dt scale / floor",
    )
    axs[0].legend()
    axs[1].plot(x, 1000 * avg("flatfeet/termination_fraction"))
    axs[1].set(
        title="Symmetry terminations", ylabel="Events per 1,000 control transitions"
    )
    for ax in axs:
        ax.grid(alpha=0.2)
        ax.set_xlabel("Training steps (millions)")
    save(fig, "flatfeet_reward_terms")
    recent = pr[-100:]
    summary = dict(
        training_job=3226505,
        steps=200005632,
        elapsed="12:54:12",
        best_step=best["step"],
        best_error=best["track_err_mean"],
        old_best_error=oldbest["track_err_mean"],
        last_eval=ev[-1],
        recent_start=int(float(recent[0]["time/env_steps"])),
        recent_end=int(float(recent[-1]["time/env_steps"])),
        recent_means={
            k: float(col(recent, k).mean())
            for k in recent[0]
            if k.startswith("flatfeet/")
        },
        replays={},
    )
    sections = []
    for label in ["best", "final"]:
        trace = DATA / f"flatfeet_trace_{label}.npz"
        if not trace.exists():
            continue
        with np.load(trace) as z:
            meta = json.loads(str(z["meta"].item()))
            summary["replays"][label] = meta
            for i, trial in enumerate(meta["trials"]):
                n = trial["ticks"]
                t = np.arange(n) * meta["dt"]
                target = trial["target"]
                prefix = f"flatfeet_{label}_{target:.2f}"
                fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                axs[0].plot(t, z["vx"][:n, i], label="Forward speed")
                axs[0].axhline(target, color="gray", ls="--", label="Command")
                axs[0].set(
                    ylabel="m/s",
                    title=f"{label}, target {target:.2f} m/s — {trial['outcome']}",
                )
                for j, side in enumerate(["left", "right"]):
                    # Offset binary contact tracks to distinguish both legs and both pads.
                    for pad, offset in [("toe", 0), ("heel", 1.2)]:
                        axs[1].step(
                            t,
                            z["flatfeet_" + pad][:n, i, j] * 0.8
                            + (1 - j) * 2.4
                            + offset,
                            where="post",
                            color=COLORS[j],
                            lw=0.7,
                            label=side + " " + pad,
                        )
                    axs[2].plot(
                        t,
                        z["flatfeet_slip"][:n, i, j],
                        color=COLORS[j],
                        lw=0.7,
                        label=side,
                    )
                axs[1].set(ylabel="Contact tracks", yticks=[])
                axs[2].set(ylabel="Contact-point slip (m/s)", xlabel="Time (s)")
                axs[2].set_xlim((10, 12) if n >= 1200 else (0, n * meta["dt"]))
                for ax in axs:
                    ax.legend(loc="upper right", ncol=2)
                    ax.grid(alpha=0.2)
                save(fig, prefix + "_contacts")
                video = f"videos/{prefix}mps.mp4"
                content = (
                    f'<video controls playsinline preload="metadata" src="{video}"></video>'
                    if (REPORT / video).exists()
                    else "<p>Video rendering pending.</p>"
                )
                sections.append(
                    f"<section><h2>{label.title()} — {target:.2f} m/s</h2>{content}"
                    f"<p>Checkpoint {meta['step']:,}; {trial['seconds']:.2f} s; {trial['outcome']}. "
                    f"Mean speed after 5 s: {trial['mean_speed_after_5s']:.3f} m/s. "
                    f"Completed cycles L/R after 5 s: {trial['completed_steps_after_5s']}.</p>"
                    f'<details><summary>Speed, toe/heel contacts and slip (10–12 s detail)</summary><img loading="lazy" src="figures/{prefix}_contacts.png"></details></section>'
                )
    charts = "".join(
        f'<h2>{title}</h2><img loading="lazy" src="figures/{name}.png">'
        for name, title in [
            ("flatfeet_learning", "Learning compared with WALK-01"),
            ("flatfeet_speed_comparison", "Tracking by target speed"),
            ("flatfeet_bilateral_training", "Both legs during training"),
            ("flatfeet_reward_terms", "Reward contributions and termination"),
        ]
    )
    html = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flat-feet symmetry v1: results and videos</title><style>body{font:17px/1.5 system-ui;background:#111925;color:#edf4ff;max-width:1280px;margin:30px auto;padding:20px}a{color:#8dccff}video,img{width:100%;border-radius:8px}section{margin:40px 0}button{padding:8px 16px;margin:5px}</style>
<h1>Flat-feet symmetry v1: completed 200M steps</h1>
<p>Job 3226505 completed in 12h54m. Best checkpoint: 184.32M, 28/28 full-duration trials,
0.049 m/s tracking error (WALK-01 best: 0.181 m/s). Final periodic evaluation: 27/28,
0.044 m/s. Better tracking is not proof of clean walking: inspect slip and foot motion.</p>
<p>Videos use the frozen training simulator, nominal plant, fixed commands, seed 1000.
Both cameras show the same instant. Blue = left foot; orange = right. Trials stop at their
first termination; resets are not spliced into the video. These are single trials per condition.</p>
<p><strong>Important: the best checkpoint is not consistently the cleanest gait.</strong>
At 0.75 m/s it registers only 1 valid left cycle versus 80 right cycles after 5 s,
without terminating. Brief contact interruptions can prevent valid-cycle history from updating
while also resetting the planted-leg timer. The final checkpoint is more balanced at this speed
(77/68), but its 1.5 m/s trial ends through the symmetry rule. Inspect both checkpoints.</p>
<p><button onclick="rate(1)">Real time</button><button onclick="rate(.5)">Half speed</button><button onclick="rate(.25)">Quarter speed</button></p>
"""
        + (
            "\n".join(sections)
            if sections
            else "<p>GPU video replay 3226907 is in progress.</p>"
        )
        + charts
        + """
<p>Training plots average 50 adjacent logged updates and include randomized plants and commands.
They are not the same protocol as the fixed-command nominal clips. Aggregate L/R agreement can
hide asymmetry within individual episodes. Flat contact fractions are over all ticks, not just mid-stance.</p>
<p>Early slip-curve gaps mark nonfinite diagnostic entries (31 logged updates per side,
9.49–56.99M steps). Logged mean rewards remain finite. These missing diagnostic values are not interpolated.</p>
<p><a href="../docs/flatfeet-symmetry-v1.md">Experiment report</a> · <a href="../docs/experiments/dash01_flatfeet_symmetry_v1/results-summary.json">Numerical results</a> · <a href="walk01-videos.html">Previous WALK-01 videos</a></p>
<script>function rate(r){document.querySelectorAll('video').forEach(v=>v.playbackRate=r)}</script></html>"""
    )
    (REPORT / "flatfeet-videos.html").write_text(html, encoding="utf-8")
    (DATA / "results-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = [
        *DATA.glob("flatfeet_trace*"),
        *REPORT.glob("videos/flatfeet_*.mp4"),
        Path(__file__),
        ROOT / "RLframework/tools/gait_video.py",
    ]
    (DATA / "replay-provenance.json").write_text(
        json.dumps(
            dict(
                replay_job=3226907,
                files={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print("Saved flat-feet gallery, figures, summary and provenance")


if __name__ == "__main__":
    main()
