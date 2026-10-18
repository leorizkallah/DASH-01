# GAIT-05 status

Checked 16 October 2026: job **3220364 COMPLETED**, elapsed **1h24m56s**,
exit **0:0**, at **15,007,744 steps**. Training ran to completion, but the policy
failed to learn stable walking.

All 16 recorded evaluations (baseline plus training/final evaluations) had
**0/12 survivors and 0/12 qualified trials at each of 0.20, 0.25 and 0.35 m/s**.
No valid steps were recorded in the post-5-second measurement windows. This
must not be interpreted as proof of no stepping before the early falls.
Final mean training episode length was approximately **2.18 s** (217.86 ticks).
The logged 0.25 m/s primary evaluation error is the no-window fallback, not a
measured tracking error. Candidate ranking tied throughout and retained step 0;
there is no useful learned gait candidate or qualified keeper.

Downloaded evaluation history, final/candidate metrics, progress.csv and eval.csv.
No new run or training change was made during this check.

Fresh policy, **15M steps at 0.25 m/s**, GAIT-04 reward unchanged. No checkpoint
or observation statistics inherited. Gait evaluation: 12 trials each at 0.20,
0.25 and 0.35 m/s; primary acceptance at 0.25 m/s. Every evaluated checkpoint
is retained. Diagnostic candidate ranking now includes bilateral progress.

Local tests: 57 passed, three JAX tests skipped. Transferred archive SHA-256
verified against local source:
`6c65a649a7fb7a7f7e29c5dc2d25b7b332b0475dbbc839d384277b5a2b20dbd1`.
The prerequisite GPU benchmark completed and the training run reached its budget.

Remote root: `/home/rizkalla/running_robot/experiments/gait05_20261015_v1`.
Log: `RLframework/logs/cad-v2-gait05-3220364.out`.
Run: `RLframework/runs/cad_v2_gait05_s0`.
Allocation: one GPU, 10 CPUs, 90 GB RAM, four-hour cap.

See the [experiment report](../../gait05-experiment-report.md) and
[source manifest](source-manifest.json). This file is a dated observation, not a
live dashboard. The completed result is an unsuccessful walking-learning experiment.
