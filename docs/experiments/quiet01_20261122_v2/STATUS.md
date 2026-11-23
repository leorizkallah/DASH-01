# QUIET-01 v2 — completed successfully

Best-policy[video gallery](../../../report/quiet01v2-videos.html) prepared23November:
20s quiet/disturbed-start comparisons, four fixed cases/model. SeparateCPU-MJX
replay3264005 completesexit0, selected outcomes matcharchivedGPUverdicts.
No newtraining. [Media provenance](../../../report/quiet01v2-media.md).

Checked23November2026. All preflight3263184 and training3263185 tasks completed,
exit0. Each model reaches100,012,032 additional transitions, reason`step_limit`.
No jobs remain running. Scheduler start estimates changed; training ran overnight.

| Model | Runtime | Baseline /96 | Saved best /96 | Final /96 | Best quiet /16 | Best quiet retention /16 |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 4h23m | 59 | 71 | 68 | 16 | 16 |
| V2 | 5h58m | 37 | 85 | 85 | 16 | 15 |
| X-shape | 5h02m | 45 | 89 | 85 | 16 | 16 |

Quiet retention is the separate20s evaluation. Best return checkpoints occur at
66.36M(V1),100.01M(V2),58.98M(X). Final curriculum level2(V1) and3(V2/X);
hardest8deg-requested bank remains imperfect. No non-finite logged learning loss,
KL or reward diagnostics. No new jobs submitted in this check.

[Results and learning curves](../../../report/quiet01-v2-results.md).
Raw metrics, receipts, best/final checkpoints and scheduler accounting archived
under `report/data/quiet01v2/results/`. Historical diagnostic3263170 also completed;
its traces and plots are archived separately under`report/data/quiet01/hold-diagnostics/`.

## Original submission

22 November 2026,17:08 CET (Europe/Zurich).

- Preflight array **3263184**, three GPU tasks,1h limit.
- Training array **3263185**, three models, depends on all preflight tasks passing.
- Task0=V1, task1=V2, task2=X-shape.
- 100M additional transitions/model from each model's own STAND-04 best and
  observation normalizer; fresh optimizers/counters/curriculum, seed2.
- Training fixed12s: timely five-second reference hold remains completed;
  subsequent physical/task failure still rejects. Separate20s retention evaluation
  requires timely initial hold and independent final five-second window.
- Same reward weights, posture tolerances, models, noise and randomized reset banks.
- Preflight checks numerical and physical validity instead of requiring75%
  mastery of the new pose task. Baseline/pilot use matching seed1234 and include
  both tracking and retention. Pilot weights are discarded.
- Local33 tests, compilation/whitespace checks passed. All six SCITAS abstract
  reset/step checks passed. At submission, GPU results were not yet checked.
- Allocation12h, training save cutoff10.5h.

Remote root: `/home/rizkalla/running_robot/experiments/quiet01_20261122_v2`.
Run names: `quiet01v2_v1_s2`, `quiet01v2_v2_s2`, `quiet01v2_xshape_s2`.
Original cancelled attempt and queued historical diagnostic3263170 are unchanged.

[Report](../../../report/quiet01-v2-pose-tracking.md), [receipt](submission.json),
[source manifest](source-manifest.json), [local validation](local-validation.json).
