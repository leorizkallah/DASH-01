# GAIT-02 status — checked 15 October 2026

Job **3219597: COMPLETED**, exit 0, runtime **2 h 23 min 48 s**.
Finished **30,015,488 additional steps** (rounded to a complete rollout).
The pilot's final checkpoint at 5,013,504 steps was selected: mean evaluation
survival 2.06 s versus 1.21 s for its earlier best checkpoint. Both candidates
fell in all selection trials. Total selected lineage exposure: 35,028,992 steps.

## Best recorded evaluation

Checkpoint **28,311,552**: **72/96 survived all 25 seconds (75%)**;
mean absolute speed error **0.123 m/s**, mean absolute heading error **9.35°**.

| Target speed | Full-episode survivors | Mean absolute speed error |
|---|---:|---:|
| 0.20 m/s | 25/32 (78.1%) | 0.124 m/s |
| 0.35 m/s | 32/32 (100%) | 0.058 m/s |
| 0.50 m/s | 15/32 (46.9%) | 0.186 m/s |

Tracking errors exclude the initial five seconds. These are nominal simulation
trials, not a hardware or randomized robustness result. Walking-only contact
patterns have not been certified by this evaluation.

The last periodic evaluation at 29,360,128 steps regressed to about 60% survival;
the best checkpoint is preserved separately. The saved final checkpoint has not
had a separate final-step evaluation. Last sampled training metrics: mean episode
duration 16.10 s, forward speed 0.293 m/s and flight fraction 3.44%. These sampled
training metrics are not measurements of the best policy's deterministic gait.

Downloaded evidence: `selection.json`, `progress.csv`, `eval.csv`, `best_eval.json`,
`best.json`, `final.json`. No new job was submitted during this status check.

Subsequent video request: replay job **3219901** completed; three recorded trials
are linked in the [report](../../gait02-experiment-report.md). At targets 0.20 and
0.35 m/s the trials survived 25 s; at 0.50 m/s the robot fell at 20.31 s. Flight
fractions of 2.27–4.61% show that strict walking has not yet been achieved in these
examples. Replay metadata is `best_gait_trace.json`; no new training was launched.

Preset `cad_v2_gait02`, run `cad_v2_gait02_s0`, seed 0, one GPU, six-hour limit.
Remote root: `/home/rizkalla/running_robot/experiments/gait02_20261014_v1`.

Reward formulas/coefficients and physical plant match GAIT-01. The longer stage
restarts optimizer and schedules while retaining weights and observation statistics.
See [report](../../gait02-experiment-report.md) and [source manifest](source-manifest.json).
