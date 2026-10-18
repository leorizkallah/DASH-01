# GAIT-11 status

**Job 3223293 COMPLETED**, exit 0:0, elapsed **7h51m56s**, with
**65,650,688 steps**. The wall-clock stop worked: updates ended with 598 seconds
remaining, final checkpoint and evaluation completed before the allocation limit.
All 67 cluster tests passed. No evaluated checkpoint qualified as a clean gait.

At 0.15 m/s the final policy survived all 12 nominal 25-second trials with mean
speed error 0.0164 m/s. Mean accepted steps L/R: 9.92 / 9.92, from 11.25 / 15.17
attempts after five seconds. Incomplete attempts: **24.44%**; repeated attempts:
0.83 / 4.75. Whole-sole clearance above 15 mm occupied only 4.72% / 1.43% of
the quality window (required 8% per foot). Mean maximum symmetry mismatch was
0.310 (criterion 0.20); contact slip 0.0214 m/s. Zero trials passed all criteria
in any of 64 evaluations. Equal accepted steps alone cannot establish clean
alternation because the new supervisor accepts alternating steps by construction.

The saved diagnostic candidate is step 49,283,072: 12/12 survival, 32.11%
incomplete attempts, mean slip 0.0444 m/s. Final is better on these two measures;
the candidate comes from the existing composite ranking, not visual selection.
Final transfer survival: 0/12 at 0.10 m/s and 6/12 at 0.20 m/s.

Replay job **3223517 COMPLETED**, exit 0:0, elapsed 4m27s. Candidate and final
were recorded at 0.10/0.15/0.20 m/s, seed 1000, in a separate directory without
altering frozen training sources. Local videos use two camera angles and colored
feet; see the [video gallery](../../../report/gait11-videos.html).

Both 0.15 m/s replays survive 25 seconds. Candidate has 9/9 accepted steps from
13/12 attempts; final has 10/10 accepted steps from 10/14 attempts, including
four right-foot repeats after five seconds. Candidate at 0.10 m/s survives with
10/10 attempts, all accepted, but low clearance. Final at 0.10 m/s falls at
9.73 s. At 0.20 m/s candidate/final fall at 6.97/2.89 s. These are individual
fresh replays, distinct from the aggregate evaluation trials above.

## Run contract

- Fresh random policy, optimizer and observation statistics; no parent checkpoint.
- Target 0.15 m/s; alternating supported swings with no mandatory inter-step stop.
- Retain symmetry, weaker-foot activity, clearance, bounded progress and gradual
  all-direction contact-sliding costs. All meaningful attempts count for repeats.
- Eight-hour allocation, no step cap. Ten-minute final-save/evaluation reserve;
  setup and compilation are included. Updates should stop around 08:36 Zurich.
- 512 environments, 32 ticks, one GPU, 10 CPUs, 90 GB.
- Learning-rate schedule horizon 80M steps is not a stopping condition.
- Evaluation targets 0.10/0.15/0.20 m/s; qualification uses 0.15 m/s.

## Validation and provenance

Local regression suite: 67 tests, 60 passed and seven JAX tests deferred to SCITAS;
all 67 subsequently passed in this job.
Python compilation passed. Both launch scripts passed `bash -n` remotely.
All 53 source hashes verified in the cluster job. Source archive SHA-256:
`70595ee507b71b6389162b62efbe3823962befbd9207002eba42ec7f8322c686`.

Remote root: `/home/rizkalla/running_robot/experiments/gait11_20261019_v1`.
Run directory: `RLframework/runs/cad_v2_gait11_s0` under that root.
Log: `RLframework/logs/cad-v2-gait11-3223293.out` under that root.
Frozen source and manifest are archived alongside this file.

[Design and retained lessons](../../gait11-experiment-report.md).
The four new actor commands depend on simulated contacts; hardware requires a
validated estimator and runtime changes. Results are simulation-only.

Status command after connecting to SCITAS: `squeue -j 3223293`.
