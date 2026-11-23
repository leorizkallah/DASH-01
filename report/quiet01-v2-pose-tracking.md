# QUIET-01 v2: corrected hold acceptance and resubmission

**Completed successfully, checked23November:** all three models finish100.01M
additional steps, exit0. Saved best test passes71/96(V1),85/96(V2),89/96(X-shape);
all pass16/16 reference-return trials. [Results and learning curves](quiet01-v2-results.md).
The remainder records the original correction and submission history.

22 November 2026. User authorized fixing the hold/preflight issues and rerunning
all three quiet-pose tasks. [Original diagnosis](quiet01-hold-diagnosis.md).
The frozen v1 root, cancelled job3263104 and diagnostic-only job3263170 are retained.

## What changes

**Training success is latched:** the robot must begin a valid quiet-reference
streak within six seconds and hold for five continuous seconds with sufficient
loaded contact. Once completed, this remains completed. Episodes still last12s,
and a subsequent fall, forbidden step/drift or invalid physics still fails.
There is no second, deadline-constrained final hold. Keeping fixed-duration
episodes also avoids encouraging the policy to delay success to collect extra
per-tick reward before an early successful termination.

Dense reference/quiet rewards stay active throughout the full episode, including
after hold completion. All posture tolerances, planted-foot/contact checks,
reward weights, motors, physical model, reference poses, reset banks and sensor
noise remain as in v1. No extra pushes, initial velocities or physical DR.

**Retention is separate:** fixed20s evaluations require a timely initial hold
AND a valid final five-second quiet window. The final window may start after
the six-second first-return deadline. Record retention separately from the12s
return success. This evaluation does not change PPO episodes or promotion gates.

**Preflight tests physical/numerical validity:** the old policy no longer needs
75% mastery of the new posture task. Require finite evaluation rates and arrays,
valid trial durations, no very early reference terminations, and at least75%
reference trials without a physical fall. Task timeouts/posture misses remain
reported honestly. A finite24-update discarded PPO pilot checks both weights and
optimizers; the same physical-validity checks apply after the pilot.

Baseline/postpilot use identical seed1234, independent initial-pose test split,
96-member layout and noise settings. Both12s tracking and20s retention results
are saved, avoiding the different-seed comparison in the original preflight.
A GPU integration regression checks that a previously completed hold is accepted
at the horizon even when the latest streak is short, provided safety passes.

## What remains the same

- Three models V1/V2/X-shape, each initialized from its own STAND-04 best weights
  and observation normalizer, with fresh optimizer/counters/curriculum, seed2.
- Reference tolerances: motor RMS1.5deg/max3deg, gravity1.5deg, torso-relative
  sole error20mm, height10mm. The20mm sole error is not clearance or floor slip.
- Physically consistent zero-velocity reset banks, 320 train +80 probe +80 test
  poses per model, requested noise levels0.5/1.5/3/5/8deg before projection.
- Training20% reference,30% easier/current,50% frontier; independent probes every
  200updates. Two passes needed to advance: reference>=95%, frontier>=90%,
  previous levels>=80%. These task criteria have not been relaxed.
- Fixed test evaluation every400updates; best qualification still requires
  reference-task success>=95%. Retention curves are logged separately.
- 100M additional transitions/model, 12h GPU allocation,10.5h training save cutoff.
- All three GPU preflights must pass before the dependent three-model training
  array starts. Pilot weights are discarded; training reloads the original parent.

## Validation and provenance

**33 local tests pass**, including four new regressions: completed hold followed
by a late interruption; insufficient/late first hold; independent retention
window; and physical preflight accepting an unlearned task but rejecting falls
or non-finite results. Python compilation and whitespace checks run as well.
All1,440 initial poses remain audited by the existing native tests.

Frozen inputs and submission receipt:
`docs/experiments/quiet01_20261122_v2/`. Remote isolated root:
`/home/rizkalla/running_robot/experiments/quiet01_20261122_v2`.
Runs will be named `quiet01v2_v1_s2`, `quiet01v2_v2_s2`, `quiet01v2_xshape_s2`.

Local changes to the shared modules do not change the original frozen diagnostic
job3263170: it imports the v1 snapshot explicitly. Its helper may append historical
v1 replay results to the old diagnosis; they must not be presented as v2 outcomes.

Scheduler receipts and remote abstract-validation results will be recorded below.
No training outcome is claimed at preparation time.

## Submitted: 22 November 2026, 17:08 CET

- GPU preflight array **3263184**, three model tasks, one-hour limit each.
- Training array **3263185**, dependency `afterok:3263184`, three parallel model
  tasks with100M additional steps each. Failed preflight cancels dependent training.
- All six SCITAS CPU abstract checks pass: reset/step for tracking AND retention
  on V1, V2 and X-shape, observation shape `(2, 463)`. These are tracing checks;
  GPU numerical rollouts and pilots remain gated by the preflight array.
- Archive SHA256: `9fd599d9bccbf0470160b3c40951c0583533506abe0bff90e4c23989e47da5fe`.

[Submission receipt](../docs/experiments/quiet01_20261122_v2/submission.json),
[frozen source manifest](../docs/experiments/quiet01_20261122_v2/source-manifest.json),
[local validation](../docs/experiments/quiet01_20261122_v2/local-validation.json),
[abstract traces](../docs/experiments/quiet01_20261122_v2/abstract-trace.txt).
No GPU preflight/training outcomes monitored after submission. The source archive
contains the preparation-time report snapshot; scheduler receipts were added later.
