# STOP-05 status

Latest observation **2026-10-12 19:21:54 Europe/Zurich**: job **3216384 ended
at 19:14:28**, after **28 minutes 3 seconds**, with Slurm state FAILED (exit 1:0).
PPO was deliberately stopped by the preservation guard at **262,144 new steps**:
walking retention and standing retention both failed; moving-stop success remained
0/32. This was an early guard stop, not a time-limit expiration.

The keeper remains the source policy (zero new steps). Automatic assessment
completed: 0/32 moving stops on each of three seeds, with standing holds of
29/32, 32/32 and 29/32. Replay recording and rendering both succeeded; the chosen
clip is an unsuccessful qualifying approach, not a successful stop. No new job
was launched. See [assessment](MOTION_ASSESSMENT.md), [guard record](guard_stop.json)
and [evaluation records](stop_evaluations.jsonl). Artifacts remain in the remote run.

## Submission observation

Observed **2026-10-12 18:46:42 Europe/Zurich**: Slurm job **3216384** is
**RUNNING** on `i35`, elapsed 17 seconds. This is scheduler confirmation;
the first policy evaluation and learning results are not yet available.

- Run: `dash_stop05_motion_s0`; preset: `dash_stop05_motion`; training seed: 0.
- Snapshot: `/home/rizkalla/running_robot_experiments/stop05_20261012_v1`.
- Log: `RLframework/logs/stop05-motion-3216384.out` inside that snapshot.
- Source: STOP-04 selected keeper at step 262,144, with frozen STOP-03 rehearsal.
- Budget: 1M new steps, rounded to 1,048,576; one GPU; two-hour job limit.
  Scheduler time remaining is a wall-time limit, not a training completion estimate.
- Protocol: [STOP-05 experiment report](../../stop05-experiment-report.md).

Local preflight: 25 tests, 24 passed and one JAX-dependent test skipped.
SCITAS CPU preflight: all 25 tests passed. Slurm script syntax check passed.
Uploaded archive SHA-256 matched the local package. The GPU job repeats tests
before initialization, then applies baseline and periodic preservation checks.

## Frozen provenance

Source archive SHA-256:
`6cfe2c33c7059171f2afa2371a3b6691b5adf754b51e54b07eee100d5dfe65f0`

Warm keeper SHA-256:
`9c074f03f74c1c104bfa719acae97501ad7653b6a0fa44feb6e1c92c2e5b4266`

Demonstration dataset SHA-256:
`994e7daa1e2153053f4aa746572d36aa4be1e4315507c8da3be84913a8f75caa`

The package's per-file hashes are in [source-manifest.json](source-manifest.json).
Submission metadata is recorded after freezing; the packaged experiment report
records the protocol before submission.

## Automatic outputs

The run directory will contain `STOP_PROGRESS.md`, raw evaluation JSONL,
`best_stop_eval.json`, checkpoint files and resolved configuration. After training,
the selected keeper is assessed on three evaluation seeds, producing
`MOTION_ASSESSMENT.md` and `motion_assessment.json`. A separate trace and video
are attempted under `replay/`, with selection and rendering status recorded.
If a preservation guard stops PPO, assessment uses the retained keeper and the
job retains the unsuccessful training exit code. There is no automatic next job.

Check status with `squeue -j 3216384`; after completion, use
`sacct -j 3216384 --format=JobID,State,Elapsed,ExitCode` on SCITAS.
