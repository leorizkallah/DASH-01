# STOP-06 status

## Completed outcome — checked 2026-10-13 13:14:59 Europe/Zurich

Both tasks completed normally (exit 0:0) at **5,046,272 new PPO steps each**.

| Arm | Completed on 12 October (Zurich) | Elapsed | Final moving stops | Final standing holds | Final walking retained |
|---|---|---|---:|---:|:---:|
| Timed, `3216469_0` | 21:52:00 | 2h01m21s | 0/32 | 32/32 | Yes |
| Feedback, `3216469_1` | 23:54:36 | 2h02m36s | 0/32 | 30/32 | No |

Neither arm recorded a successful moving stop in any of its 21 evaluation
records, and final survival through the complete moving STOP interval was 0/32.
Both selected keepers therefore remain the source checkpoint (zero new steps).
Three-seed post-run assessment evaluates these source keepers, not the final
trained checkpoints; it recorded 0/32 moving stops on each seed for each arm.

Timed final walking survival was 62.5% versus its 66.7% baseline, within the
five-percentage-point retention tolerance. Feedback final walking survival was
66.7% versus its 75.0% baseline, and tracking error rose from 0.315 to 0.420 m/s;
both exceeded its retention tolerance. Its failures were not three consecutive
evaluations after the grace period, so the persistent guard permitted completion.
The arms' baseline evaluations differ despite identical source weights; retain
both records and avoid attributing every difference to learned behavior.

Both replay recordings and renders succeeded. They show unsuccessful approaches
from the retained source policies, not successful stops or final trained policies.
Videos and full traces remain on SCITAS; small metrics, reports and plots were
downloaded into the two run subdirectories here.

Evidence: [comparison](STOP06_COMPARISON.md),
[timed settling report](dash_stop06_timed_s0/SETTLING_REPORT.md),
[feedback settling report](dash_stop06_feedback_s0/SETTLING_REPORT.md),
[timed learning plot](dash_stop06_timed_s0/learning.png),
[feedback learning plot](dash_stop06_feedback_s0/learning.png).

Conclusion: these longer runs did not solve moving stops. Standing was retained;
feedback gating did not produce a successful transition under this setup. Inspect
the settling/termination traces before choosing another change. No new job was
submitted during this status check.

## Initial submission observation

Observed **2026-10-12 19:50:47 Europe/Zurich**. Submitted array **3216469**:

| Task | Arm | State | Observation |
|---|---|---|---|
| `3216469_0` | Timed HOLD | RUNNING | `i03`, elapsed 8 seconds |
| `3216469_1` | Feedback HOLD | PENDING | JobArrayTaskLimit; runs after the other task |

Scheduler confirmation only: first baseline/learning results are not yet available.
Each arm has a five-million-step maximum (5,046,272 with full rollouts), subject
to baseline checks and the documented persistent-regression guard. Each task has
a three-hour wall-time limit including assessment; this is not an ETA.

Snapshot: `/home/rizkalla/running_robot_experiments/stop06_20261012_v1`.
Runs: `RLframework/runs/dash_stop06_timed_s0` and
`RLframework/runs/dash_stop06_feedback_s0` within that snapshot.
Logs: `RLframework/logs/stop06-compare-3216469_0.out` and `_1.out`.
Protocol: [STOP-06 report](../../stop06-experiment-report.md).

## Validation and provenance

- Local: 33 tests, 30 passed, three JAX-dependent tests skipped.
- SCITAS CPU: all 33 tests passed, including JIT adapter/idempotency and gate parity.
- Python compilation and Slurm script syntax passed.
- Offline report generation passed using existing evaluation records.
- Archive SHA-256 matched after upload.

Source archive SHA-256:
`66c2ce908631871f5f5d5f01c3679cb19e7e7d267e14e62f82b995aaab6f21f6`

Both arms use the STOP-04 selected keeper at 262,144 steps, SHA-256:
`9c074f03f74c1c104bfa719acae97501ad7653b6a0fa44feb6e1c92c2e5b4266`

Frozen demonstration dataset SHA-256:
`994e7daa1e2153053f4aa746572d36aa4be1e4315507c8da3be84913a8f75caa`

Per-file source hashes: [source-manifest.json](source-manifest.json).
Submission metadata was added after packaging; the package retains the prelaunch
protocol. Previous training snapshots and keepers are preserved.

## Automatic outputs and follow-up

Each arm records STOP/standing/walking evaluations, `guard_monitor.json`, selected
keeper metrics, settling event times and a learning graph. After PPO, the keeper
is assessed on selection seed 4102 and held-out seeds 5102/6102. Replay recording,
video and a speed/rotation/tilt/phase graph are attempted, with their outcomes
recorded separately. The comparison table is written after each arm's training
and again by the exit report. An early guard stop retains exit code 1 and assesses
the previous keeper; it is not a successful completion of the full budget.

Next review: compare moving-stop success, standing/walking retention, readiness
times, actual HOLD entry and timeout rates. Failure before readiness remains a
failure, not a measured settling time. Do not infer a winner from reward alone.
No continual polling or further training stages are scheduled by this submission.

Status command on SCITAS: `squeue -r -j 3216469`.
After completion: `sacct -j 3216469 --format=JobID,State,Elapsed,ExitCode`.
