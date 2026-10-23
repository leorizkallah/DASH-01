"""Build the small policy-transfer comparison figure and local video gallery."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
DATA = REPORT / "data/policy-transfer01"
m = json.loads((DATA / "results.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for label, color in [("original", "#2679bd"), ("cad_v2", "#de772b")]:
    rows = [x for x in m["trials"] if x["model"] == label]
    axes[0].plot(
        [r["target_mps"] for r in rows],
        [r["elapsed_s"] for r in rows],
        "o-",
        color=color,
        label=label,
    )
    axes[1].plot(
        [r["target_mps"] for r in rows],
        [100 * r["command_outside_range_fraction"] for r in rows],
        "o-",
        color=color,
        label=label,
    )
axes[0].set(
    ylabel="Observed duration (s)",
    ylim=(0, 22),
    title="Original survives; direct CAD-v2 swap falls",
)
axes[1].set(
    ylabel="Control ticks outside model joint range (%)",
    ylim=(0, 35),
    title="Requested positions, not measured joint overshoot",
)
for ax in axes:
    ax.set_xlabel("Requested speed (m/s)")
    ax.set_xticks(m["targets"])
    ax.grid(alpha=0.2)
    ax.legend()
fig.suptitle("Untouched Nathann policy: one nominal CPU replay per target")
for ext in ["png", "svg"]:
    fig.savefig(REPORT / f"figures/policy_transfer01.{ext}", dpi=160)
plt.close(fig)

sections = []
for target in m["targets"]:
    a, b = [
        next(
            t for t in m["trials"] if t["model"] == label and t["target_mps"] == target
        )
        for label in ["original", "cad_v2"]
    ]
    name = f"videos/policy_transfer01_{target:.2f}.mp4"
    sections.append(f"""<section><h2>Requested speed {target:.2f} m/s</h2>
<p>Original: survives 20 s; mean world-X speed {a["mean_world_x_mps"]:.2f} m/s.
CAD-v2: falls at {b["elapsed_s"]:.2f} s. Left column: original. Right: CAD-v2.</p>
<video controls playsinline preload="metadata" src="{name}"></video>
<p><button onclick="rate(this,1)">Real time</button>
<button onclick="rate(this,0.25)">Quarter speed</button>
<button onclick="restart(this)">Restart</button> <a href="{name}">Open video</a></p></section>""")
(REPORT / "policy-transfer01-videos.html").write_text(
    """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nathann policy: original vs CAD-v2</title>
<style>body{font:17px/1.5 system-ui;background:#111925;color:#eef3fa;max-width:1280px;margin:32px auto;padding:0 20px}
a{color:#8dccff}video,img{width:100%}section{margin:36px 0}button{padding:8px;margin-right:8px;cursor:pointer}</style>
<h1>Same policy, two robot models</h1>
<p>Nathann's original 180M-step joystick policy, without training. Original weights,
normalization, gait decoder and motor delay are retained. Both models are free in 3D.</p>
<p>Each column has side and oblique views. On CAD-v2, blue identifies the left foot and
orange the right; the original model's foot meshes retain their dark appearance.
The CAD-v2 pose freezes at the first fall and is labelled accordingly; it is not balancing in that pose.
Use quarter speed to inspect the short failure.</p>
<p>These are one deterministic nominal trial per condition, not robustness statistics.
Commands of 0.25 / 0.50 / 0.75 m/s do not imply those speeds were achieved.
Clamping and reset-offset diagnostic variants also fell within 0.71–0.75 s.</p>
<p><a href="policy-transfer01.md">Protocol, results and interpretation</a> ·
<a href="data/policy-transfer01/results.json">Raw metrics</a> · <a href="index.html">Report home</a></p>
<img src="figures/policy_transfer01.png" alt="Twenty-second survival on the original model; subsecond falls and out-of-range commands on CAD-v2">
"""
    + "\n".join(sections)
    + """
<script>
function rate(b,r){b.closest('section').querySelector('video').playbackRate=r;}
function restart(b){const v=b.closest('section').querySelector('video');v.currentTime=0;v.play();}
</script></html>""",
    encoding="utf-8",
)
files = [
    ROOT / "RLframework/tools/policy_transfer_replay.py",
    ROOT / "RLframework/tools/play_joystick.py",
    ROOT / "controller/deploy/controller_v2.py",
    ROOT / "controller/deploy/gait_v2.py",
    ROOT / "controller/deploy/bundle.py",
    ROOT / "controller/deploy/bundles/dash_joy3_lr_s2_180M.npz",
    ROOT / "RLframework/model/dash01_free.xml",
    ROOT / "RLframework/model/dash01_cad_v2_free.xml",
    Path(__file__),
]
files += list(DATA.glob("*.npz"))
files += list(DATA.glob("*results.json")) + list(DATA.glob("adapter-probes.json"))
files += list((REPORT / "videos").glob("policy_transfer01_*.mp4"))
(DATA / "manifest.json").write_text(
    json.dumps(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        indent=2,
    )
    + "\n"
)
print(REPORT / "policy-transfer01-videos.html")
