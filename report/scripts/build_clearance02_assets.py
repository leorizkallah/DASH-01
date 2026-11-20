"""Build CLEARANCE-02 convergence curves and a gallery from archived evidence."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/experiments/clearance02_20261027_v2"
OUT = ROOT / "report/data/clearance02"
FIG = ROOT / "report/figures"
COLORS = ["#168BD2", "#EF8018"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    records = [json.loads(s) for s in (DATA / "clearance_evaluations.jsonl").read_text().splitlines()]
    best = json.loads((DATA / "best_clearance_candidate_eval.json").read_text())
    with (DATA / "progress.csv").open() as f:
        progress = list(csv.DictReader(f))
    x = np.array([r["step"] for r in records]) / 1e6
    selected = best["step"] / 1e6
    assert np.all(np.diff(x) > 0)
    assert x[0] == 0 and records[-1]["step"] == 200005632
    assert any(r["step"] == best["step"] for r in records)
    assert all(r["minimum_clearance_mm"] == 10 and r["bonus_cap_mm"] == 25 for r in records)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold"})
    fig, axs = plt.subplots(3, 2, figsize=(14.4, 11.0))
    fig.suptitle("CLEARANCE-02: 200M-step fine-tuning at 1.50 m/s", fontsize=20, fontweight="bold")
    tx = np.array([float(r["time/env_steps"]) for r in progress]) / 1e6
    reward = np.array([float(r["rollout/reward_mean"]) for r in progress])
    assert np.isfinite(reward).all()
    axs[0, 0].plot(tx, reward, lw=.45, alpha=.18, color="#176A55", rasterized=True)
    bx, by = [], []
    for start in range(0, len(tx), 100):
        bx.append(tx[start:start + 100].mean())
        by.append(reward[start:start + 100].mean())
    axs[0, 0].plot(bx, by, lw=2, color="#176A55", label="100-update block mean")
    axs[0, 0].set(title="Training reward (faint = raw rollout mean)", ylabel="Reward per control transition")
    axs[0, 1].plot(x, [r["track_error_mps"] for r in records], "o-", ms=4, color="#26323F", label="12-trial mean error")
    axs[0, 1].axhline(.15, color="#5A626D", ls="--", label="0.15 m/s criterion")
    axs[0, 1].set(title="Nominal evaluation: speed tracking", ylabel="Mean absolute error (m/s)", ylim=(0, None))
    for side, name in enumerate(("Left", "Right")):
        color = COLORS[side]
        mean = [r["mean_air_clearance_mm"][side] for r in records]
        lo = [np.asarray(r["per_trial"]["mean_air_clearance_mm"])[:, side].min() for r in records]
        hi = [np.asarray(r["per_trial"]["mean_air_clearance_mm"])[:, side].max() for r in records]
        axs[1, 0].plot(x, mean, "o-", ms=4, color=color, label=name)
        axs[1, 0].fill_between(x, lo, hi, color=color, alpha=.12)
        axs[1, 0].scatter(selected, best["mean_air_clearance_mm"][side], marker="*", s=140, color=color, zorder=5)
        axs[1, 1].plot(x, [r["peak_air_clearance_mm"][side] for r in records], "o-", ms=4, color=color, label=name)
        axs[2, 0].plot(x, [100 * r["midswing_ge_min_fraction"][side] for r in records], "o-", ms=4, color=color, label=name)
    axs[1, 0].axhline(10, color="#5A626D", ls="--", label="10 mm mean criterion")
    axs[1, 0].set(title="Mean airborne foot height (band = trial range)", ylabel="Whole-foot clearance (mm)", ylim=(0, None))
    axs[1, 1].axhline(25, color="#5A626D", ls="--", label="25 mm upper target")
    axs[1, 1].axhline(26, color="#E2001A", ls=":", label="26 mm acceptance tolerance")
    axs[1, 1].set(title="Largest sampled airborne peak across 12 trials", ylabel="Whole-foot clearance (mm)", ylim=(0, None))
    axs[2, 0].axhline(80, color="#5A626D", ls="--", label="80% criterion")
    axs[2, 0].set(title="Scheduled mid-swing samples reaching 10 mm", ylabel="Samples (%)", ylim=(-3, 105))
    axs[2, 1].plot(x, [100 * r["survival_fraction"] for r in records], "o-", ms=4, color="#176A55", label="Full 30 s duration")
    axs[2, 1].plot(x, [100 * r["qualified_fraction"] for r in records], "s--", ms=4, color="#E2001A", label="All criteria met")
    axs[2, 1].set(title="Survival and complete qualification", ylabel="Trials (%)", ylim=(-3, 105))
    for ax in axs.flat:
        ax.axvline(selected, color="#747C84", ls=":", alpha=.7)
        ax.set(xlabel="Additional training transitions (millions)", xlim=(0, 200.01))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="best")
    fig.text(.015, .015,
             f"Dotted vertical line: selected candidate at {selected:.2f}M; it is not fully qualified. One training seed; 29 nominal evaluations.\n"
             "Evaluation: 12 trials, 30 s, 5 s warmup. Height bands show trial range, not confidence intervals. Peaks sampled at 100 Hz.",
             fontsize=10, color="#5A626D")
    fig.tight_layout(rect=(0, .06, 1, .96))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"clearance02_convergence.{ext}", dpi=170)
    plt.close(fig)
    rows = [dict(step=r["step"], left_mean_mm=r["mean_air_clearance_mm"][0], right_mean_mm=r["mean_air_clearance_mm"][1],
                 left_peak_mm=r["peak_air_clearance_mm"][0], right_peak_mm=r["peak_air_clearance_mm"][1],
                 left_midswing_ge10_fraction=r["midswing_ge_min_fraction"][0],
                 right_midswing_ge10_fraction=r["midswing_ge_min_fraction"][1],
                 survival_fraction=r["survival_fraction"], qualified_fraction=r["qualified_fraction"],
                 speed_error_mps=r["track_error_mps"]) for r in records]
    with (OUT / "evaluations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inputs = [DATA / n for n in ("progress.csv", "clearance_evaluations.jsonl", "best_clearance_candidate_eval.json")]
    (OUT / "plot-provenance.json").write_text(json.dumps(dict(
        selected_step=best["step"], smoothing="Non-overlapping means of 100 PPO updates (1.8432M transitions); final partial block retained",
        evaluation_points="Raw; lines connect sparse checkpoints without interpolation or fitted convergence model",
        inputs_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    ), indent=2) + "\n", encoding="utf-8")
    video = ROOT / "report/videos/clearance02_best_1.50mps.mp4"
    trace = DATA / "clearance02_trace_best_eval12.json"
    replay_note = "Replay job 3238862 submitted; video is being prepared."
    replay_warning = ""
    if trace.exists():
        replay = json.loads(trace.read_text())
        t = replay["trials"][0]
        replay_note = (f"Trial 0 of the retained 12-trial replay: {t['seconds']:.2f} s; "
                       f"outcome: {t.get('outcome', 'see metadata')}; speed error after 5 s: "
                       + (f"{t['speed_error_after_5s']:.4f} m/s." if t["speed_error_after_5s"] is not None else "not available (early termination)."))
        completed = sum(t["seconds"] >= 25 and not t["fell"] and not t.get("symmetry_terminated", False) for t in replay["trials"])
        failures = [(i, t["seconds"]) for i, t in enumerate(replay["trials"]) if t["fell"]]
        replay_warning = (f"<p><strong>Fresh replay: {completed}/12 trials complete 25 seconds.</strong> "
                          + ("Early falls: " + "; ".join(f"trial {i} at {seconds:.2f} s" for i, seconds in failures) + ". " if failures else "")
                          + "This differs from the archived candidate evaluation's 12/12 full 30 s trials. "
                          "The checkpoint, batch size, nominal settings and saved reset-command convention are retained, "
                          "but the recorder is a separate compiled computation; the cause of the discrepancy has not been isolated. "
                          "The stored evaluation curves are not a guarantee of reliable startup.</p>")
    movie = ('<video controls playsinline preload="metadata" src="videos/clearance02_best_1.50mps.mp4"></video>'
             if video.exists() else "<p>The replay is not ready yet.</p>")
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLEARANCE-02 best candidate and convergence</title>
<style>body{{font:17px/1.5 system-ui;background:#111925;color:#edf4ff;max-width:1280px;margin:30px auto;padding:20px}}a{{color:#8dccff}}video,img{{width:100%;border-radius:8px}}section{{margin:30px 0}}button{{padding:8px 15px;margin:5px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #3d4d60;text-align:left}}</style>
<h1>CLEARANCE-02: best candidate at 132.71M additional steps</h1>
<p>Training job 3232851 completed 200.01M additional transitions from the flat-feet baseline. The selected clearance candidate is from 132.71M, not the final checkpoint. This is a learned simulation policy, not an animation or a hardware test.</p>
<p><strong>Promising, but not fully qualified:</strong> the selected candidate completes 12/12 nominal 30 s evaluations at 1.50 m/s. Mean airborne lift is 16.94 / 20.06 mm (left/right), and all scheduled mid-swing samples reach 10 mm. Sampled peaks reach 30.22 / 30.04 mm, above the 26 mm acceptance limit. No evaluated checkpoint met all criteria.</p>
<section><h2>Best candidate: first retained trial</h2>{replay_warning}{movie}
<p><button onclick="rate(1)">Real time</button><button onclick="rate(.5)">Half speed</button><button onclick="rate(.25)">Quarter speed</button></p>
<p>{replay_note}</p><p>Nominal greedy MJX replay, seed 1000, batch of 12 at 1.50 m/s with the saved reset command. Trial 0 is selected before inspecting appearance; all trial metadata are retained. Two cameras show the same instant. Left foot blue, right foot orange. Video ends at the first termination; no reset trajectory is spliced in.</p></section>
<section><h2>Learning curves</h2><img src="figures/clearance02_convergence.png" alt="Reward, speed tracking, mean and peak lift, mid-swing coverage and survival across 200M additional transitions">
<p>The dotted vertical line marks the saved candidate. Training reward and mid-swing coverage improve, but the peak-height objective remains unmet. Continued reward improvement does not imply that every gait criterion improves. This is one training seed, not proof of full convergence.</p>
<p><a href="figures/clearance02_convergence.pdf">Vector PDF</a> | <a href="figures/clearance02_convergence.svg">SVG</a> | <a href="data/clearance02/evaluations.csv">Evaluation CSV</a></p></section>
<p><a href="../docs/experiments/clearance02_20261027_v2/STATUS.md">Run status</a> | <a href="../docs/clearance02-experiment-report.md">Experiment report</a> | <a href="../docs/experiments/clearance02_20261027_v2/clearance02_trace_best_eval12.json">Retained replay trials</a></p>
<script>function rate(r){{document.querySelectorAll('video').forEach(v=>v.playbackRate=r)}}</script></html>'''
    (ROOT / "report/clearance02-videos.html").write_text(html, encoding="utf-8")
    print(json.dumps(dict(evaluations=len(records), training_updates=len(progress), selected_step=best["step"], video_ready=video.exists()), indent=2))


if __name__ == "__main__":
    main()
