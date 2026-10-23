"""WALK-01 training charts, MJX replay gallery and provenance."""

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
DATA = ROOT / "docs/experiments/walk01_20261019_v1"


def main():
    rows = list(csv.DictReader((DATA / "eval.csv").open()))
    steps = np.array([float(r["step"]) / 1e6 for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axs[0].plot(steps, [100 * float(r["alive_frac"]) for r in rows])
    axs[0].set(
        ylabel="Survived 30 s (%)",
        ylim=(-3, 103),
        title="Nominal seven-speed ladder, 28 trials",
    )
    axs[1].plot(
        steps, [float(r["track_err"]) for r in rows], label="Full command ladder"
    )
    axs[1].plot(
        steps,
        [float(r["cur_track_err"]) for r in rows],
        label="Current curriculum band",
    )
    axs[1].set(
        ylabel="Mean absolute speed error (m/s)",
        title="Tracking peaked before the final checkpoint",
    )
    axs[1].legend()
    for ax in axs:
        ax.set_xlabel("Training steps (millions)")
        ax.grid(alpha=0.2)
        ax.axvline(169.5744, color="gray", ls="--", alpha=0.6)
    for ext in ["png", "svg"]:
        fig.savefig(REPORT / f"figures/walk01_progress.{ext}", dpi=160)
    plt.close(fig)
    metas = {
        k: json.loads((DATA / f"walk01_trace_{k}.json").read_text())
        for k in ["best", "final"]
    }
    sections = []
    for speed in [0.75, 0.25, 1.5]:
        for label in ["best", "final"]:
            m = metas[label]
            t = next(t for t in m["trials"] if abs(t["target"] - speed) < 0.001)
            video = f"videos/walk01_{label}_{speed:.2f}mps.mp4"
            assert (REPORT / video).exists(), video
            sections.append(f'''<section><h2>{label.title()} policy — requested {speed:.2f} m/s</h2>
<video controls playsinline preload="metadata" src="{video}"></video>
<p>{m["step"] / 1e6:.2f}M steps; completed {t["seconds"]:.0f} s without falling.
Mean body-forward speed after 5 s: <strong>{t["mean_speed_after_5s"]:.3f} m/s</strong>.
<a href="{video}">Open / download</a></p></section>''')
    html = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WALK-01 policy videos</title><style>
body{font:17px/1.5 system-ui;background:#111925;color:#edf4ff;max-width:1280px;margin:32px auto;padding:0 20px}
a{color:#8dccff}video,img{width:100%;border-radius:8px}section{margin:36px 0}button{padding:8px 16px;margin-right:8px;cursor:pointer}
</style><h1>WALK-01: inspect the learned gait</h1>
<p>Training completed: <strong>200.01M steps in 13h29m</strong>. Best saved checkpoint:
169.57M steps. Start with the 0.75 m/s clips below. Left foot is blue, right foot orange;
both camera views show the same instant.</p>
<p><strong>These are GPU MJX replay recordings.</strong> Job 3226290 completed in 4m27s.
The saved policies drive the frozen training simulator from the supplied homing pose;
local rendering displays the recorded motion. Each condition is one nominal trial,
seed 1000, with a fixed speed command and no STOP command.</p>
<p>Look at foot sliding, clearance, alternating support and small repeated steps.
All six trials survive, but survival and speed tracking do not establish clean walking.</p>
<p><button onclick="rate(1)">Real time</button><button onclick="rate(.5)">Half speed</button>
<button onclick="rate(.25)">Quarter speed</button><span id="rate">1x</span></p>
"""
        + "\n".join(sections)
        + """
<h2>Training evaluation</h2><img src="figures/walk01_progress.png" alt="Training survival and speed tracking history">
<p>The saved best policy survived 28/28 fixed-ladder trials with 0.181 m/s mean
tracking error. The last periodic evaluation near 199.1M steps also survived 28/28,
but error rose to 0.328 m/s. The final checkpoint's separate command-redraw evaluation
had no falls in 28 thirty-second trials. These are different protocols.</p>
<p><a href="../docs/walk01-experiment-report.md">Experiment report</a> ·
<a href="../docs/experiments/walk01_20261019_v1/STATUS.md">Run status</a> ·
<a href="../docs/experiments/walk01_20261019_v1/replay-provenance.json">Replay provenance</a> ·
<a href="walk01-cpu-previews.html">Earlier CPU deployment previews</a></p>
<script>function rate(r){document.querySelectorAll('video').forEach(v=>v.playbackRate=r);document.getElementById('rate').textContent=r+'x';}</script></html>"""
    )
    (REPORT / "walk01-videos.html").write_text(html, encoding="utf-8")
    files = (
        list((REPORT / "videos").glob("walk01_best_*.mp4"))
        + list((REPORT / "videos").glob("walk01_final_*.mp4"))
        + list(DATA.glob("walk01_trace_*"))
    )
    files += [
        ROOT / "RLframework/tools/gait_video.py",
        ROOT / "RLframework/model/dash01_cad_v2_homing.xml",
        Path(__file__),
    ]
    record = dict(
        training_job=3224989,
        mjx_replay_job=3226290,
        replay_state="COMPLETED",
        backend="MJX GPU rollout, local MuJoCo rendering",
        replays=metas,
        files={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        caveats="Single nominal greedy rollout per condition; all trials retained. Not a robustness "
        "evaluation or proof of clean walking. Videos rendered at 20 fps; shadows disabled for rendering speed.",
    )
    (DATA / "replay-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Saved WALK-01 gallery, progress figure and provenance")


if __name__ == "__main__":
    main()
