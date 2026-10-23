# DASH-01 flat-feet symmetry v1 - completed

Job **3226505 COMPLETED**, exit 0:0, **200,005,632 steps in 12h54m12s**.
GPU smoke checks passed. Best saved checkpoint: **184,320,000 steps**;
28/28 fixed-ladder trials last 30 seconds; mean speed error **0.0489 m/s**.
Last periodic evaluation at 199,065,600: 27/28, one physical fall, error 0.0442 m/s.
Final command-redraw evaluation: 26/28 reach 30 s, zero physical falls, two early
non-fall endings consistent with the symmetry rule (reason not exported there).

[Videos and useful graphs](../../../report/flatfeet-videos.html).
MJX replay job **3226907 completed** in 4m18s. Best/final videos at 0.25, 0.75, 1.5 m/s.
Important: best 0.75 m/s has only 1/80 valid L/R cycles after 5 s and evades
symmetry termination through short contact bouts. Final 0.75 m/s is 77/68;
final 1.5 m/s triggers symmetry termination. Tracking is not proof of clean gait.
[Full report](../../flatfeet-symmetry-v1.md) and `results-summary.json` contain
comparison results, bilateral metrics and limitations. No new training launched.

Source archive SHA256:
`a1fcf3b67ae24dd99c8272a1cf9574f80aab4c5274d4f0ec27a40e22a22c94da`.
The training snapshot is unchanged; only report/replay tooling was updated.
