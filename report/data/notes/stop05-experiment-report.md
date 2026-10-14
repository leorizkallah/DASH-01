# STOP-05: stopping from low-speed motion

Prepared 2026-10-12 following approval to progress from standing to moving stops.
Snapshot: `stop05_20261012_v1`; run: `dash_stop05_motion_s0`.

## Question and source

Can the standing policy learn to brake from 0.25–0.5 m/s, hold still for five
continuous seconds, and restart, while retaining standing and walking?

Warm start: STOP-04 selected `best_stop.msgpack`, step 262,144, from
`/home/rizkalla/running_robot_experiments/stop04_20261012_v1/RLframework/runs/dash_stop04_rehearsal_s0/`.
That keeper achieved 29/32, 32/32 and 30/32 standing holds across evaluation seeds
4102, 5102 and 6102. Those results establish standing performance, not moving-stop
success. The final STOP-04 checkpoint is not the source because its final walking
check failed. Weights and observation statistics are restored; PPO optimization
and the step counter restart for this pilot.

## Changes

- New `dash_stop05_motion` preset. RUN commands sample 0.25–0.5 m/s and last
  3–4 seconds; commands then alternate with 10–12 seconds of STOP. Episodes last
  up to 30 seconds. Initial resets are 75% RUN and 25% HOLD.
- The existing adapter still latches estimated speed on STOP and supplies a cubic
  reference ramp with duration `max(2 s, 1.5 * speed / 0.5 m/s²)`. HOLD follows
  braking. This pilot does not lengthen the ramp or change network architecture.
- Dense HOLD rewards, actuator settings and reset perturbations are retained.
  Sensor noise and domain randomization remain disabled for this learning stage.
- Continue STOP-04 demonstration rehearsal inside the PPO loss: 128 HOLD and
  128 RUN samples per minibatch, actor loss coefficient 1 and RUN weight 2.
  The frozen STOP-03 demonstration dataset and normalization must match the source.
- Add a separate standing evaluation alongside moving-stop and walking checks.
  Periodic and final evaluations now use the same preservation guard. Avoid a
  duplicate final evaluation when that exact step was already evaluated.
  This closes the STOP-04 guard gap; it does not explain its differing repeated
  evaluation results.

## Budget, selection and guards

One training seed (0), one GPU, nominally 1,000,000 new PPO steps (1,048,576 with
complete rollouts). Slurm time limit: two hours including automated assessment.
This is a bounded pilot, with no automatic speed increase or follow-on job.

Before PPO, require at least 80% standing success on the new baseline. During
training, stop if standing or primary moving-stop success drops more than ten
percentage points from baseline, or if walking retention fails. Walking compares
the nonzero-command ladder aggregate: survival may drop at most five percentage
points and tracking error may rise at most 0.10 m/s. A rejected checkpoint and
guard reason are retained. Evaluate every four rollouts (262,144 steps).

Only checkpoints passing walking and standing retention may become keepers;
rank by moving-stop success, restart success, survival and residual speed.
Baseline itself may remain the keeper if learning does not improve performance.

## Evaluation and evidence

32 deterministic trials at 0.25, 0.375 and 0.5 m/s, with staggered stop times.
A successful stop requires a valid moving approach (incoming forward speed at
least `max(0.15 m/s, half the command)`), survival, and five continuous seconds
meeting the existing speed, angular-speed, joint-speed, tilt, support and drift
criteria. Being stationary before STOP cannot pass the moving-stop benchmark.

The quiet-hold search deadline is seven seconds after the measured ramp ends.
For compatibility, the command schedule reserves the maximum ramp duration at
2.5 m/s (7.5 s), then seven seconds of STOP, followed by a three-second restart
check. Thus moving trials also require survival through an intentionally longer
14.5-second STOP command, even when the actual low-speed ramp is two seconds.
Standing evaluation uses its separate near-stance protocol.

After training, automatically reassess the keeper on seeds 4102, 5102 and 6102.
Seed 4102 is used for selection; the other two are held out from selection.
Walking retention is the standard ladder check, not a three-seed walking study.
Write `motion_assessment.json`, `MOTION_ASSESSMENT.md`, ongoing
`STOP_PROGRESS.md`, raw evaluation JSONL, source hashes and resolved configuration.
Record a separate pose trace and half-speed video. Select the first successful
moving-stop trial if present; otherwise label the replay as an unsuccessful
qualifying approach (or trial zero if none qualify). Save recording results and
selection metadata because a separately compiled replay can differ from selection.

Submission and measured outcomes will be recorded in
[the dated status file](experiments/stop05_20261012_v1/STATUS.md).
No moving-stop improvement is claimed before those results are available.
