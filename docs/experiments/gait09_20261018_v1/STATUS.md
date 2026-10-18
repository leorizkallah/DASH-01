# GAIT-09 status

**Job 3222597 COMPLETED**, exit code 0:0, elapsed **3h26m45s**.
Finished **32,014,336 additional steps**; final checkpoint saved remotely.

Final evaluation at 0.25 m/s: 12/12 survive 25 seconds, speed error 0.00881 m/s,
10 valid steps per foot after five seconds. However, 20 attempts per foot include
10 incomplete and 10 repeated same-foot attempts per foot; incomplete fraction
50%. No qualified trial in any of the 32 recorded evaluations at the training
target. Contact slip improved from 0.01098 to 0.00578 m/s, but the intended
micro-step correction did not succeed. Final clear fractions L/R: 7.67%/8.62%.
At 0.20 m/s, 12/12 survive; at 0.35 m/s, 0/12 survive (mean episode 3.61 s).
Best diagnostic candidate remains step 18,874,368; it also fails qualification.

Final metrics, candidate metrics, evaluation history, progress CSV, job log and
benchmark JSON are now saved locally in this directory. Source archive/manifest
remain unchanged. Compare candidate/final video before deciding the next change;
no further training was submitted.

Requested video replay job **3223177** also completed, in 3m15s. Both candidate
and final clips are available in the [comparison page](../../../report/gait09-videos.html).
Each is 25 s at 0.25 m/s, seed 1000, with two views and colored feet. Replay
traces, per-attempt diagnostics and provenance are stored in this directory.

## Submission and validation record

- Snapshot: `gait09_20261018_v1`; run: `cad_v2_gait09_s0`.
- Budget: 32,000,000 additional steps (32,014,336 at full-rollout rounding).
- Parent: GAIT-08 `eval_25165824.msgpack`, best diagnostic candidate at 0.25 m/s.
- One GPU, 10 CPUs, 90 GB, six-hour time limit. Rough estimate 3.5–4 hours
  after starting, based on GAIT-08; queue time is additional.
- Local focused suite: 49 tests across six modules, 44 passed, 5 JAX-dependent
  tests skipped locally and scheduled in the job. Python compilation passed.
- Uploaded source archive SHA-256 verified remotely. The job passed source and
  parent checkpoint verification, focused tests, benchmark and baseline checks.
- Startup order: hash verification, all focused tests including JAX, finite
  MJX/PPO benchmark, inherited-policy full-penalty reward/survival preflight,
  then training. An unsuccessful check stops the job with diagnostics retained.
- Remote root: `/home/rizkalla/running_robot/experiments/gait09_20261018_v1`.
- Log: `RLframework/logs/cad-v2-gait09-3222597.out` under that root; a local copy
  is retained in this directory.

Changes and evaluation protocol: [experiment report](../../gait09-experiment-report.md).
Frozen source manifest and archive metadata are in this directory.

```powershell
ssh -i C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519 -o IdentitiesOnly=yes rizkalla@izar.hpc.epfl.ch "sacct -j 3222597 --format=JobID,State,Elapsed,ExitCode"
```
