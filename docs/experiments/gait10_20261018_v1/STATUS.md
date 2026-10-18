# GAIT-10 status

**Job 3223224 COMPLETED**, exit 0:0, elapsed **1h43m23s**.
Finished **15,007,744 additional steps**; final checkpoint saved remotely.
All 58 tests, source/parent checks, benchmark and reward/survival preflight passed.

Final 0.25 m/s evaluation: 12/12 survive, speed error 0.00988 m/s, 10 valid steps
per foot after five seconds, but 50.41% of attempts remain incomplete. Repeated
attempts L/R: 10 / 0.17, versus 10 / 10 at baseline. Total attempts remain
20 / 20.33: the lower repetition counter does not mean extra bouts disappeared.
Mean maximum symmetry mismatch worsened 0.386 -> 1.046; contact slip rose
0.00593 -> 0.01641 m/s. No trial qualified at the training target in any of the
16 recorded evaluations. Best diagnostic candidate: step 11,534,336, also fails.

`continuation_decision.json` selects **fresh_policy_with_explicit_step_objective**:
the agreed sustained progress criteria were not met. Next branch is a fresh policy
and explicit sequential stepping objective; no further continuation is planned.
No new training job was submitted during this status check.

Final/candidate metrics, decision file, all evaluation records, progress CSV,
job log and benchmark are archived locally in this directory.

## Submission record

- Run: `cad_v2_gait10_s0`, frozen snapshot `gait10_20261018_v1`.
- Parent: GAIT-09 `final.msgpack` at 32,014,336 steps.
- Budget: 15,000,000 additional steps, rounded to 15,007,744.
- Target 0.25 m/s; 512 environments; one GPU, 10 CPUs, 90 GB; four-hour limit.
- Estimated duration: approximately 1h40–2h after starting; not a scheduler guarantee.
- Local tests: 58 total, 52 passed, 6 JAX tests subsequently passed in this job; compilation
  passed. Legacy GAIT-06/08/09 state/terms/metrics matched frozen code over 600
  transitions each. Launch scripts passed `bash -n` on SCITAS.
- Archive SHA-256: `b6b7c1a60e867ff102fee68e78dde61c2ccf9c5e271f0b980f1b65a41a4874e0`.
- Remote root: `/home/rizkalla/running_robot/experiments/gait10_20261018_v1`.
- Log under that root: `RLframework/logs/cad-v2-gait10-3223224.out`.
- Completed training wrote `continuation_decision.json` under the run directory,
  using the progress criteria fixed in the [report](../../gait10-experiment-report.md).
  No fresh policy has been submitted: it is the agreed next branch if this last
  continuation fails to show sustained progress, with its new objective prepared then.

```powershell
ssh -i C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519 -o IdentitiesOnly=yes rizkalla@izar.hpc.epfl.ch "sacct -j 3223224 --format=JobID,State,Elapsed,ExitCode"
```
