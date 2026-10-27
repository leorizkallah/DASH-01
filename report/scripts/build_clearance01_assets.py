"""Rebuild CLEARANCE-01 evaluation plots, pose-derived clearance and video gallery."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RLframework/tools"))
from measure_swing_clearance import air_bouts, pad_clearances

DATA = ROOT / "docs/experiments/clearance01_20261025_v1"
OUT = ROOT / "report/data/clearance01"
FIG = ROOT / "report/figures"
COLORS = ["#168bd2", "#ef8018"]


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=160)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(s)
        for s in (DATA / "clearance_evaluations.jsonl").read_text().splitlines()
    ]
    x = np.array([r["step"] for r in records]) / 1e6
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, side in enumerate(("Left", "Right")):
        y = np.array([r["mean_air_clearance_mm"][i] for r in records])
        lo = [
            np.array(r["per_trial"]["mean_air_clearance_mm"])[:, i].min()
            for r in records
        ]
        hi = [
            np.array(r["per_trial"]["mean_air_clearance_mm"])[:, i].max()
            for r in records
        ]
        axs[0, 0].plot(x, y, "-o", color=COLORS[i], label=side)
        axs[0, 0].fill_between(x, lo, hi, color=COLORS[i], alpha=0.15)
        axs[0, 1].plot(
            x,
            [100 * r["midswing_ge20mm_fraction"][i] for r in records],
            "-o",
            color=COLORS[i],
            label=side,
        )
    axs[0, 0].axhline(20, ls="--", color="black", label="20 mm criterion")
    axs[0, 0].set(
        title="Mean height while airborne (12 trials; band = range)",
        ylabel="Whole-foot clearance (mm)",
    )
    axs[0, 1].axhline(80, ls="--", color="black", label="80% criterion")
    axs[0, 1].set(
        title="Scheduled mid-swing samples reaching 20 mm",
        ylabel="Samples (%)",
        ylim=(-3, 103),
    )
    axs[1, 0].plot(
        x,
        [100 * r["survival_fraction"] for r in records],
        "-o",
        label="Full 30 s survival",
    )
    axs[1, 0].plot(
        x,
        [100 * r["qualified_fraction"] for r in records],
        "-o",
        label="All clearance criteria",
    )
    axs[1, 0].set(
        ylabel="Trials (%)", ylim=(-3, 103), title="Survival and qualification"
    )
    axs[1, 1].plot(
        x,
        [r["track_error_mps"] for r in records],
        "-o",
        label="Mean absolute speed error",
    )
    axs[1, 1].set(ylabel="m/s", title="Tracking target 1.50 m/s")
    for ax in axs.flat:
        ax.set_xlabel("Additional training steps (millions)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    save(fig, "clearance01_learning")
    b, f = records[0], records[-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, side in enumerate(("Left", "Right")):
        vals = [b["mean_air_clearance_mm"][i], f["mean_air_clearance_mm"][i]]
        bars = ax.bar(
            np.arange(2) + (i - 0.5) * 0.32, vals, 0.32, color=COLORS[i], label=side
        )
        ax.bar_label(bars, fmt="%.2f mm", padding=3)
    ax.axhline(20, color="black", ls="--", label="20 mm")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Inherited parent", "After 50.01M steps"],
        ylabel="Mean airborne clearance (mm)",
        ylim=(0, 29),
        title="Matched nominal evaluation: 12 trials at 1.50 m/s",
    )
    ax.legend()
    save(fig, "clearance01_comparison")
    write_csv(
        OUT / "evaluations.csv",
        [
            dict(
                step=r["step"],
                left_mean_mm=r["mean_air_clearance_mm"][0],
                right_mean_mm=r["mean_air_clearance_mm"][1],
                left_midswing_ge20_fraction=r["midswing_ge20mm_fraction"][0],
                right_midswing_ge20_fraction=r["midswing_ge20mm_fraction"][1],
                survival=r["survival_fraction"],
                qualified=r["qualified_fraction"],
                track_error_mps=r["track_error_mps"],
            )
            for r in records
        ],
    )
    trace = DATA / "clearance01_trace_final.npz"
    if not trace.exists():
        return
    cfg = json.loads((DATA / "resolved_config.json").read_text())["config"]
    model_path = ROOT / "RLframework" / cfg["model_path"]
    manifest = json.loads((DATA / "source-manifest.json").read_text())
    assert (
        hashlib.sha256(model_path.read_bytes()).hexdigest()
        == manifest["files"][model_path.relative_to(ROOT).as_posix()]
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(trace) as z:
        meta = json.loads(str(z["meta"].item()))
        trial = meta["trials"][0]
        n, dt = trial["ticks"], meta["dt"]
        # qpos is pre-step; contact is post-step. Align without running dynamics.
        pads = pad_clearances(model, z["qpos"][:n, 0])[1:]
        ground = z["flatfeet_grounded"][: n - 1, 0]
        np.testing.assert_array_equal(
            ground, z["flatfeet_toe"][: n - 1, 0] | z["flatfeet_heel"][: n - 1, 0]
        )
    t = np.arange(1, n) * dt
    h = np.maximum(pads.min(axis=-1), 0) * 1000
    rows, bouts = [], []
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, side in enumerate(("Left", "Right")):
        air = ~ground[:, i]
        mask = (t >= 5) & air
        peaks = []
        for a, e, full in air_bouts(air):
            if t[a] < 5:
                continue
            peak = float(h[a:e, i].max())
            bouts.append(
                dict(
                    foot=side,
                    start_s=t[a],
                    duration_s=(e - a) * dt,
                    complete=full,
                    peak_mm=peak,
                )
            )
            if full and (e - a) * dt >= 0.04:
                peaks.append(peak)
        rows.append(
            dict(
                foot=side,
                mean_air_mm=float(h[mask, i].mean()) if mask.any() else None,
                median_complete_bout_peak_mm=float(np.median(peaks)) if peaks else None,
                max_air_mm=float(h[mask, i].max()) if mask.any() else None,
                airborne_seconds=float(mask.sum() * dt),
            )
        )
        axs[i].plot(
            t,
            h[:, i],
            color=COLORS[i],
            label=f"{side}: minimum of toe and heel pad heights",
        )
        axs[i].fill_between(
            t,
            0,
            65,
            where=ground[:, i],
            color="grey",
            alpha=0.12,
            label="Force-qualified ground contact",
        )
        axs[i].axhline(20, color="black", ls="--", label="20 mm")
        axs[i].axhline(30, color="green", ls=":", label="30 mm bonus cap")
        window = (10, 12) if t[-1] >= 12 else (0, float(t[-1]))
        detail = (t >= window[0]) & (t <= window[1])
        axs[i].set(
            ylabel="Clearance (mm)",
            ylim=(0, max(65, h[detail, i].max() * 1.05)),
            xlim=window,
        )
        axs[i].legend(fontsize=8, ncol=2)
        axs[i].grid(alpha=0.2)
    axs[-1].set_xlabel("Replay time (s); first 5 s excluded from summary")
    save(fig, "clearance01_trace")
    write_csv(OUT / "replay-summary.csv", rows)
    if bouts:
        write_csv(OUT / "replay-bouts.csv", bouts)
    np.savez_compressed(
        OUT / "replay-clearance.npz",
        time=t,
        pad_signed_m=pads,
        height_mm=h,
        grounded=ground,
    )
    (OUT / "replay-summary.json").write_text(
        json.dumps(
            dict(
                trial=trial,
                feet=rows,
                note="Single nominal 25 s trial; seed 1000; warmup 5 s; force threshold 2 N. Means use all airborne samples; peak summaries use complete bouts >=40 ms. Heights from frozen-model rigid kinematics at 100 Hz, no re-simulation.",
            ),
            indent=2,
        )
        + "\n"
    )
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLEARANCE-01 results</title>
<style>body{font:17px/1.5 system-ui;background:#111925;color:#edf4ff;max-width:1280px;margin:30px auto;padding:20px}a{color:#8dccff}video,img{width:100%;border-radius:8px}section{margin:35px 0}button{padding:8px 15px;margin:5px}</style>
<h1>CLEARANCE-01: higher foot lift at 1.50 m/s</h1>
<p>Job 3230928 completed 50,006,016 additional steps in 3h30m21s. Final checkpoint is also the best clearance candidate. In 12 nominal 30 s trials, all survive; mean airborne clearance is 23.62 mm left / 20.91 mm right, versus 2.46 / 2.25 mm for the inherited parent under the same evaluation protocol.</p>
<p><strong>Progress, but not fully qualified:</strong> only 57.2% / 71.4% of scheduled mid-swing samples reach 20 mm; the required minimum is 80% per foot. Mean height alone hides this timing/consistency shortfall. Speed error is 0.029 m/s; 99-100 valid cycles per foot after warmup.</p>
<p><strong>Replay discrepancy:</strong> the first standalone pinned-command replay (batch size 1) falls at 3.67 s, despite the original evaluation's 12/12 survival. That failure is retained below. A second recording preserves the saved reset command and batch size 12. Batch size and command-reset handling both differ between these protocols, so the cause is not isolated; successful evaluation does not establish reliable standalone startup.</p>
<p><button onclick="rate(1)">Real time</button><button onclick="rate(.5)">Half speed</button><button onclick="rate(.25)">Quarter speed</button></p>
<section><h2>New policy: evaluation-protocol replay, trial 0</h2><video controls playsinline preload="metadata" src="videos/clearance01_final_1.50mps.mp4"></video><p>Final / best clearance candidate, nominal greedy MJX replay, seed 1000, first trial of a batch of 12; saved fixed command retained. Both cameras show the same instant. Left foot blue; right foot orange. Playback ends at first termination, with no reset splicing. See numerical replay metadata for actual duration and outcome.</p></section>
<section><h2>Standalone replay: startup failure retained</h2><video controls playsinline preload="metadata" src="videos/clearance01_standalone_1.50mps.mp4"></video><p>Same checkpoint and nominal plant, one environment with pinned command: fall at 3.67 s. No steady-state mean is computed because the episode ends before the 5 s warmup.</p></section>
<section><h2>Parent before fine-tuning</h2><video controls playsinline preload="metadata" src="videos/flatfeet_best_1.50mps.mp4"></video><p>Flat-feet v1 best at 184.32M steps; archived nominal seed-1000 replay. Both clips have the same target, but the earlier recording used a batch of three speeds, so initial random samples are not paired.</p></section>
<h2>Mean clearance before and after</h2><img src="figures/clearance01_comparison.png">
<h2>Clearance and contact detail</h2><img src="figures/clearance01_trace.png"><p>Whole-foot height is the lower of the oriented heel/toe pad surfaces. Grey bands show force-qualified contact; proximity alone is not a contact. Trace uses frozen-model forward kinematics aligned to saved MJX contact flags.</p>
<h2>Training progress and clearance consistency</h2><img src="figures/clearance01_learning.png">
<p>Nominal evaluations do not establish robustness to pushes, different speeds or hardware. Qualification retains the full existing criteria; it was not relaxed after training.</p>
<p><a href="../docs/clearance01-experiment-report.md">Experiment report</a> | <a href="data/clearance01/evaluations.csv">Evaluation data</a> | <a href="data/clearance01/replay-summary.json">Replay measurements</a> | <a href="flatfeet-clearance.md">Earlier clearance baseline</a></p>
<script>function rate(r){document.querySelectorAll('video').forEach(v=>v.playbackRate=r)}</script></html>"""
    (ROOT / "report/clearance01-videos.html").write_text(html, encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
