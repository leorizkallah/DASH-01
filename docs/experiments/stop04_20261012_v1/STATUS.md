# STOP-04 paired diagnosis and standing-preservation pilot

Submitted 12 October 2026: Slurm job **3215784**, SCITAS Izar.
Confirmed at **11:00:20 Europe/Zurich: PENDING (Priority)**. The job has not
started, so a batch log and new physical results do not exist yet.
Snapshot: `/home/rizkalla/running_robot_experiments/stop04_20261012_v1`.
Training run: `dash_stop04_rehearsal_s0`.

Pipeline: paired STOP-03 best/final traces and criterion diagnosis -> replay videos
-> physical eligibility gate -> conditional 1M-step PPO with standing/walking
rehearsal -> three-seed assessment. No automatic low-speed transition training.

Source keeper: STOP-03 `best_stop.msgpack`, 524,288 steps. Copied checkpoints,
configuration and demonstrations are under snapshot `inputs/stop03/`; hashes are
recorded by the batch script. Old snapshots and policies are preserved.

Preflight: 21 tests passed locally, one JAX gradient test skipped for unavailable
local JAX. All **22 tests passed on SCITAS CPU**, including the actor-only rehearsal
gradient check. Python compilation, batch script syntax and upload hash passed.
GPU physics, regression diagnosis and learning results are pending execution.

Archive SHA-256:
`0c80e5072e23f6bb0a9a66abf0c73495c3ba888e8fff2889fa958069a3e261bd`.

Resources: 1 GPU, 10 CPUs, 90 GB, two-hour wall-clock limit. PPO targets 1M new
steps (1,048,576 after rollout rounding), with checks every 262,144 steps.

Reports on SCITAS:

- `RLframework/results/stop04_diagnosis/DIAGNOSIS.md`: condition failures and paired traces.
- `RLframework/results/stop04_diagnosis/comparison.json`: diagnosis gate and video selection/status.
- `RLframework/runs/dash_stop04_rehearsal_s0/STOP_PROGRESS.md`: training outcomes and any regression guard.
- `RLframework/runs/dash_stop04_rehearsal_s0/MULTISEED.md`: source/keeper tests at seeds 4102, 5102, 6102.
- `RLframework/logs/stop04-rehearsal-3215784.out`: batch log.

See [protocol and change log](../../stop04-experiment-report.md). No success or
improvement is claimed at submission. Status retrieval is on demand to minimize
Codex usage; the batch generates its own artifacts without an assistant session.
