# QUIET-01 v2 best-policy videos

[Open the video gallery](quiet01v2-videos.html).

Requested23November2026. Each20s clip shows four fixed trials side by side:
quiet0, moderate48, larger64, hard80. Trials are fixed before replay outcomes.
Nominal requested initial noise0/3/5/8deg is projected onto physically valid poses;
actual initial motor RMS is shown. Full sensor noise, no pushes, seed1234,
original96-member per-trial PRNG keys. Only the four displayed independent trials
are replayed. Each model uses its selected best RETURN
checkpoint and the stricter20s retention evaluation.

| Model | Checkpoint additional M | Replay backend | Replayed retention | Archived GPU retention | Video |
|---|---:|---|---:|---:|---|
| X-shape | 58.98 | cpu | 3/4 | 78/96 | Ready |
| V2 | 100.01 | cpu | 3/4 | 78/96 | Ready |
| V1 | 66.36 | cpu | 2/4 | 62/96 | Ready |

GPU replay3263991 was cancelled while still pending after long queue estimates.
The full-batch CPU attempt3263994 was stopped to reduce replay work.
CPU-only selected-trial MJX replay3264005 uses the same frozen source/weights/model/solver and
float32 arithmetic. Backend numerical differences may change trajectories and
verdicts; replay outcomes are reported separately from archived GPU scores.
There are no learning updates and no changes to reward, targets or physics.
Display-only lighting and foot colors improve visibility.

Videos run at1x,20fps,1280x960,H264. The gallery offers0.5x/1x/2x controls.
Failure freezes the last valid pose and labels the termination; there is no
reset splice or physics resimulation during rendering. Blue=leftfoot,orange=rightfoot.
Foot-position error is relative to the torso; it is not clearance.

Raw traces, replay metadata and video provenance are in
`report/data/quiet01v2/videos/{model}/`; pipeline state in
`report/data/quiet01v2/videos/pipeline.json`.

## Clarification of the visible outcomes

X-shape trial48 completes the initial five-second hold (start0.59s) and
has a longest continuous quiet streak12.05s, but fails the final retention
window. Terminal sole-relative error20.56mm exceeds20mm, by0.56mm;
peak during the final five seconds21.35mm. No fall or task termination.
Thus the clip shows successful initial recovery with marginal later drift.

V1 trial48 ends at1.54s on the planted-foot rule: native FK of recorded
states gives left sole-center drift54.62mm versus50mm allowed, right8.45mm.
No whole-foot air episode above5mm and no physical fall. Trial64 succeeds:
peak sole-center drifts2.74/10.73mm; final quiet streak10.25s.

The two V1 starts are different joint patterns: initial motor RMS2.19/2.58deg;
left-thigh offsets+2.94/-3.36deg respectively. Their noise-bank labels
do not order the difficulty of these particular samples.
