# STOP-03 imitation-assisted standing

Submitted 2026-10-07: Slurm job **3205920** on SCITAS Izar.
Latest retrieved state: **COMPLETED**, exit code **0:0**, elapsed **26 min 28 s**.
Reached **2,031,616 PPO steps** after imitation passed every initialization gate.
The earlier DNS failure was resolved on retry.

Slurm records start `2026-10-08T00:30:14` and end `2026-10-08T00:56:42`;
the remote clock reported `2026-10-08T10:57:01+02:00` at retrieval. These are
remote timestamps, preserved verbatim; the client date context for this check is
2026-10-07. The clock discrepancy has not been reconciled.
Run: `dash_stop03_imitation_s0`.
Snapshot: `/home/rizkalla/running_robot_experiments/stop03_20261007_v1`.

Pipeline: source evaluation -> standing/walking demonstrations -> 2,000 actor-only
imitation updates -> physical behavior gate -> critic prefit -> 2M-step standing PPO.
Imitation passed its gate and PPO ran to completion. Standing improvement does
not establish successful stopping from motion.

| Stage | Five-second holds | Survived 7 s | Mean longest quiet hold | Walking retained |
|---|---:|---:|---:|:---:|
| Source | 0/32 | 0/32 | 0.0184 s | yes |
| After imitation | 21/32 | 22/32 | 4.360 s | yes |
| PPO 524,288 (best keeper) | 21/32 | 22/32 | 4.434 s | yes |
| PPO 1,048,576 | 14/32 | 30/32 | 4.158 s | yes |
| PPO 1,572,864 | 14/32 | 30/32 | 3.646 s | yes |
| PPO 2,031,616 (final) | 14/32 | 31/32 | 3.929 s | yes |

Imitation enabled sustained standing. Later PPO improved survival but reduced
strict quiet-hold success. The best hold keeper is `best_stop.msgpack` at 524,288
steps. These are near-stance trials; moving STOP and restart remain unvalidated.
No follow-on training was submitted during this status check.

Resources: one GPU, 10 CPUs, 90 GB RAM, two-hour wall-clock limit.
All 18 unit tests passed both locally and on SCITAS before submission. Python
compilation and Slurm script syntax checks passed. GPU collection, imitation,
physical gates, critic prefit and PPO completed. Downloaded metrics are alongside
this file: [automatic progress](STOP_PROGRESS.md), [imitation](imitation.json),
and [best hold evaluation](best_stop_eval.json).

Archive SHA-256, verified after upload:
`5e6d65fd5fec10f64c909e32029e541899bd21e7be8b9d96d291555e7b73640a`.
Source file hashes: [source-manifest.json](source-manifest.json).

Details: [experiment report](../../stop03-experiment-report.md).
Cluster log: `RLframework/logs/stop03-imitation-3205920.out`.
Automatic report: `RLframework/runs/dash_stop03_imitation_s0/STOP_PROGRESS.md`.
Imitation stage, fitting metrics and gates: same run directory, `imitation.json`.

Refresh on demand (no continuous Codex polling):

```powershell
.\RLframework\slurm\stop_progress.ps1 -Snapshot stop03_20261007_v1 -Run dash_stop03_imitation_s0 -Job 3205920 -LogPrefix stop03-imitation
```
