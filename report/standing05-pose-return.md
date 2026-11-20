# STAND-05 preparation: recover, regain the quiet posture, then hold

Prepared **11 November 2026**. User clarified: **regain the same posture at the
recovered floor location**. Returning to the original world XY coordinates is
not part of this task. Implementation and validation are prepared; **no new
training job has been submitted** and no new learned behavior is claimed.

## Reference posture

Each model gets a fixed target from its own STAND-04 best policy's successful
quiet-standing replay, trial 0. Use the 10–20 s segment, select samples with a
quiet streak >=1 s, and choose an actual recorded pose closest to the normalized
feature mean. This preserves a feasible closed-loop pose rather than averaging
joint coordinates into an unvalidated configuration. The reference is frozen
per model; it does not drift toward whatever stance a new policy settles into.

| Model | Policy checkpoint (additional steps) | Selected replay time | Quiet streak | Candidate frames matching target tolerances |
|---|---:|---:|---:|---:|
| V1 | 169.57M | 15.10 s | 15.10 s | 100/100 |
| V2 | 58.98M | 12.80 s | 12.80 s | 100/100 |
| X-shape | 132.71M | 14.50 s | 14.50 s | 100/100 |

[Reference-selection audit](data/standing05/quiet-reference-audit.json).
Targets and source model/checkpoint/trajectory hashes are saved in
`BalanceRL/model/quiet_v1.json`, `quiet_v2.json`, `quiet_xshape.json`.
Full source qpos is retained for visualization/provenance only; global XY is
absent from the posture-error calculation.

Target features are **actual six motor-joint positions**, torso gravity direction,
sole positions relative to the torso, and torso height relative to the soles.
These are measured positions, not the PD command offsets needed to generate
holding torque. All features are invariant to translating the robot on the floor
or changing its global heading. Body shape/lean and leg arrangement still matter.
The fixed goal needs no new actor inputs or changes to network dimensions;
variable user-commanded posture goals would require revisiting the observation design.

## Implemented sequence

New opt-in presets: `return_v1`, `return_v2`, `return_xshape`. The existing
`recover_*` task and frozen STAND-04 runs remain available.

1. **Recover balance.** After the actual pulse ends, regain supported, low-motion
   stance within three seconds. Require both feet loaded and the projected CoM
   inside/near the actual support hull, with a 250 ms stability dwell.
2. **Return to the reference posture.** Once balance is recovered, allow four
   seconds to start the final posture hold. Recovery steps remain legal here,
   including a corrective placement of the trailing foot. Do not commit the
   footprint and prohibit steps merely because the robot has become still.
3. **Hold the target posture quietly for five continuous seconds.** Only then
   commit the new recovered footprint, award recovery success once, and return
   to quiet-mode stepping restrictions. Another step or a failed pose match
   resets the hold timer without extending the fixed return deadline.

Small disturbances may require no step and may already leave the target posture
nearly restored. Such responses are valid. There is no reward for producing a
step or a forced alternating-foot sequence. Existing contact event, chatter,
sole travel, grounded slip and excessive-step penalties remain active.

Initial configurable pose acceptance tolerances:

- Six motor angles: RMS error <=3 degrees, maximum individual error <=6 degrees.
- Torso gravity direction: <=3 degrees from the reference lean.
- Each sole's position in the torso frame: <=30 mm from reference.
- Torso height relative to soles: <=20 mm from reference.
- Existing low-velocity, footprint-stability and >=95% loaded-support requirements
  still apply throughout the five-second hold.

The old broad `0.2 * exp(-joint_MSE / 0.50²)` tie-breaker is replaced in the new
mode by a posture score of maximum weight 1, combining normalized joint, gravity,
foot-placement and height errors. It is disabled during the balance-recovery
phase, then ramps up over 0.5 s after balance recovery. Quiet standing also
receives the fixed-posture objective. This avoids forcing the original stance
while the robot is still trying to avoid a fall.

## Evaluation and evidence

New-mode first-episode evaluations last **25 s**, allowing the three-second balance
deadline, four-second return allowance and five-second final hold after randomized
push onset. Quiet and pushed trials both need the final five seconds at the
reference posture. Push success additionally requires the timely intermediate
recovery/return sequence. These scores have a new protocol and must not be
equated directly to STAND-04's scores.

Logs distinguish balance recovery from full pose-return success, returning-phase
occupancy, pose-match fraction, joint RMS/max error, gravity error, foot-relative
error and height error. Evaluations retain per-trial balance/return timings and
final pose errors, alongside falls, task violations, steps and sole travel.

The existing X-shape +X 0.20 m/s clip illustrates why the new goal matters: it
passes the old task for 20 s without a detected step, but its last sample has
about **7.78 degrees motor-joint RMS error** and **130 mm maximum sole-position
error relative to the torso** compared with its quiet reference. It therefore
does not match the desired posture. The quiet trial matches the target. This
is offline measurement of an old trajectory, not a newly trained result.
V1/V2's corresponding pushed clips fail at 6.90/6.80 s, so their last pose errors
are pre-termination measurements, not recovered-standing outcomes.

## Validation and remaining work

All **23 native/pure-logic tests pass**, including reference reconstruction on
all models, invariance to floor translation/heading, rejection of the wrong
posture, a separate balance milestone, steps during return, fixed deadline expiry
and one final footprint commit. Python compilation passes. Abstract JAX tracing
on SCITAS passes for all models' reference loading, reset, step and 25 s evaluator.
Validation is isolated under `reviews/pose_return_20261111`; previous experiment
source snapshots and training checkpoints are untouched.

Numerical GPU baseline/pilot validation and a controlled training launch remain
to be done. This change alone does not demonstrate that the policy will discover
useful recovery steps. The curriculum coupling and PPO diagnostic/update concerns
in [the STAND-04 assessment](standing04-assessment.md) remain separate issues.

Reproduction: `report/scripts/build_quiet_pose_references.py`,
`BalanceRL/pose_reference.py`, `recovery_logic.py`, `recovery.py`,
`recovery_eval.py`, `tests/test_pose_return.py`, and
`report/scripts/check_pose_return_trace.py`.

## Follow-up recommendation: remove the remaining planted-recovery bias

User asked whether planted feet should be retained only as the final reference.
Recommendation: **yes for the final standing objective; keep feet free during
both balance recovery and posture return**. STAND-04 already permits recovery
steps, and the prepared return mode permits corrective steps until final hold.
Removing termination alone has not removed the reward's preference for still feet.

The current quiet term still requires both feet loaded and suppresses foot and
motor motion; it ramps back within one second after the pulse, independently of
whether recovery is complete. Once fully active, lifting a foot for 0.3 s can
forgo up to 60 unscaled reward units at 100 Hz, versus an explicit 0.2 cost for
one liftoff plus landing. This is an upper-bound comparison of reward terms,
not a measured causal attribution or a discounted-return calculation.

Proposed next controlled change (not implemented by this note):

- During recovery, prioritize staying upright, reducing excessive body motion
  and regaining support; permit single-foot support and purposeful foot motion.
  Reduce/disable the both-feet quiet bonus until final settling. Also review
  the static CoM-over-support shaping term during dynamic stepping.
- During posture return, reward progress toward the fixed relative posture
  while allowing the corrective placements needed to achieve it.
- Restore full quiet/flat support requirements for the final five-second hold.
- Keep ground-contact sliding costs and deadline/step-chatter protection.
  Initially keep the first useful recovery placements inexpensive, with rising
  cost for excessive stepping. Never add a positive reward simply for lifting
  a foot or completing a step.
- Preserve challenging push exposure separately from quiet-retention repair.
  Compare recovery-step incidence and final-pose success in a short pilot before
  a further long run. If useful steps remain undiscovered, consider a separate
  step-and-return learning curriculum or feasible demonstrations.

Keep the best existing policy as initialization for that first pilot. This
recommendation does not imply that its standing-only response must be used
throughout recovery or that a fresh restart is already necessary.
