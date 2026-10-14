# STOP-04 paired diagnosis and standing-preservation pilot

Submitted 12 October 2026: Slurm job **3215784**, SCITAS Izar.
Checked 12 October at **12:56:58 Europe/Zurich: COMPLETED**, exit **0:0**.
Ran **11:09:52-11:38:43**, elapsed **28 min 51 s**. No time remains.
PPO reached **1,048,576** new environment steps. Diagnosis and both replay renders
completed; the saved hold keeper is step **262,144**, not the final checkpoint.

Three-seed reassessment of that keeper: **29/32**, **32/32**, **30/32** holds
(seeds 4102, 5102, 6102). The >=90%-on-each-seed standing review gate passed,
with walking retained in the keeper's standard evaluation. Moving STOP is still
unvalidated; no transition training was started automatically.

Paired STOP-03 traces identified angular-motion quietness as the limiting criterion
in the surviving final-policy failures: removing only that measurement condition
changes final successes from 13/32 to 31/32. This is a metric ablation, not proof
that changing an angular reward alone would fix control.

The final PPO step has two differing evaluation records, including a walking
retention failure in the final one. Keep both records; use the earlier validated
keeper and its separate multi-seed assessment. See [assessment](MULTISEED.md),
[diagnosis](DIAGNOSIS.md) and [raw evaluations](stop_evaluations.jsonl).
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
GPU physics, diagnosis, rehearsal PPO and three-seed assessment have now completed.

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

See [protocol and change log](../../stop04-experiment-report.md). Results above are
downloaded measurements, not submission-time expectations. Status retrieval is on demand to minimize
Codex usage; the batch generates its own artifacts without an assistant session.
