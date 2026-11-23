# QUIET-01 — preflight failed; training cancelled

Subsequent [hold inspection](../../../report/quiet01-hold-diagnosis.md) establishes
V2 completed a valid five-second hold in15/16 reference trials, X-shape11/16;
final acceptance remained10/16 due to additional retention/timing checks.
Diagnostic-only array3263170 is queued for detailed reset/noise controls. Local
bounded fetch helper PID37496 is waiting; no training or task changes made.

Checked 22 November 2026, 16:18 CET. Array **3263104 never started**, zero main
training transitions; Slurm cancelled it after two tasks in preflight array
3263103 failed. No resubmission made during this status check.

| Model | Preflight | Baseline reference success | Discarded PPO pilot |
|---|---|---:|---:|
| V1 | Completed, 7m35s, exit 0 | 16/16 | 49,152 steps |
| V2 | Failed, 2m52s, exit 1 | 10/16 | Not started |
| X-shape | Failed, 2m39s, exit 1 | 10/16 | Not started |

Both failures are the deliberate assertion `baseline['quiet_rate'] >= .75`:
the parents achieve 0.625. All reset/no-push checks and 29 native/logic tests
passed on the GPU nodes. Among the 16 exact-reference trials, there were zero
physical falls for every model. V2 has one task violation, X-shape five; other
failed trials miss the strict final posture/continuous-hold criteria.

V1's post-pilot reference success is 12/16 with another sensor-noise seed versus
16/16 baseline. This is not a controlled same-seed improvement comparison;
the pilot weights were discarded and no learned checkpoint is being promoted.
Preflight completion alone does not establish training quality.

Raw preflight logs: `report/data/quiet01/preflight-results/`.

## Original submission

22 November 2026, 13:33 CET (Europe/Zurich).

- Preflight array: **3263103**, one-hour GPU limit, three model tasks.
- Training array: **3263104**, depends on successful completion of all preflight tasks.
- Models: task 0 V1, task 1 V2, task 2 X-shape; up to three concurrent GPUs.
- Each model initializes from its own STAND-04 best actor/critic/estimator weights
  and observation normalizer, with fresh optimizers and task counters.
- Budget: 100M additional transitions/model; 12-hour GPU allocation, 10.5-hour
  training save cutoff. No pushes; zero-velocity feasible pose disturbances.
- Goal: regain learned quiet posture at the recovered location and hold five seconds.
- Local validation: 29 native/logic tests, compile checks, audited 1,440 projected
  initial poses. SCITAS CPU abstract reset/step tracing passes for all three models.
- At submission, GPU numerical preflight and training outcomes were not yet checked.

Remote root: `/home/rizkalla/running_robot/experiments/quiet01_20261122_v1`.

[Experiment report](../../../report/quiet01-pose-tracking.md),
[submission receipt](submission.json), [source manifest](source-manifest.json).

Run names: `quiet01_v1_s2`, `quiet01_v2_s2`, `quiet01_xshape_s2` under the remote
root's `BalanceRL/runs/`. Save baseline/test/probe evaluations, posture errors,
hold successes, settling times and individual reward contributions for the next review.
