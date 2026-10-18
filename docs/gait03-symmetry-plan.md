# Proposed GAIT-03 — alternating, bilateral walking

15 October 2026. Approved and implemented; SCITAS job **3219917** submitted.
See the [GAIT-03 experiment report](gait03-experiment-report.md) for the final
weights, acceptance thresholds, validation and dated status. The proposal below
records the reasoning; the experiment report describes the implementation.

## Failure observed

The user identified right-foot dragging in GAIT-02's video. A kinematic check
of the 0.35 m/s replay over seconds 5–25 confirms severe clearance asymmetry:
the entire left sole clears 1 cm for 13.15% of samples, versus 0.35% on the right.
The 95th-percentile whole-sole clearances are 13.09 cm and 0.139 cm respectively.
This measures geometric clearance from recorded poses, not support forces.
Source: `experiments/gait02_20261014_v1/foot_asymmetry_diagnostic.json`.

GAIT-02 is useful forward locomotion progress, but **not an accepted walking gait**.
Its survival/tracking keeper missed bilateral step quality. More of the same
reward can reinforce this strategy rather than resolve it.

## Existing-code findings

- `w_swing_floor=0`: the existing weaker-foot swing-deficit penalty is disabled.
- `w_duty_sym=8`, `duty_floor=0.3`: despite the name, this term only penalizes
  a foot spending too little time grounded. It does not compare left and right
  duty fractions. An always-grounded dragging foot meets this minimum.
- Air-time and clearance credits are summed over feet, allowing one foot to
  earn credit while the other contributes little.
- Phase/contact weight is 0.5. Its current mismatch only bills ground contact
  during expected swing; it does not explicitly require support during stance.
- Slip penalizes horizontal toe-geom-center displacement while grounded.
  For articulated feet, rolling about the heel can move that point without true
  contact slip. Use actual contacting-pad/point velocity for a better penalty.
- `w_sym=w_sym_res=0` was a deliberate initial precaution because CAD inertials
  and joint limits are not exactly mirrored. These are policy regularizers,
  not direct footstep rewards. Re-enabling them alone would not establish a
  symmetric periodic gait or correct the contact reward gaps.

## Recommended reward revision

1. **Both legs must swing.** Enable a normalized weaker-foot swing deficit
   penalty. The current raw squared 0.15 deficit is only 0.0225, so assigning a
   small coefficient blindly would have little effect. Normalize by the desired
   fraction before choosing the weight. Use filtered contact plus measured
   whole-foot clearance; do not count contact flicker as a valid swing.
2. **Reward alternating useful steps.** Give bounded credit for touchdown only
   after that foot clears a modest threshold and advances relative to the torso.
   Penalize repeated same-foot steps and long periods without an opposite-foot
   step. Define left/right expected swing phases half a cycle apart, permitting
   double support. Let RL learn the trajectories rather than prescribing joint
   angles. Avoid using an unconstrained learned phase offset to move the reward
   schedule away from the intended alternating pattern.
3. **Match outcomes over completed cycles.** Penalize normalized left/right
   differences in stance duration, swing duration and forward step length, with
   tolerances. Compare opposite phases, not simultaneous joint angles. Pair this
   with minimum bilateral stepping: both feet doing nothing would also be equal.
4. **Make dragging costly.** Measure tangential velocity at actual support
   contacts/pads, penalize sliding while loaded, and reward modest swing clearance
   during each foot's swing. Start by testing 1–2 cm clearance as a proposal,
   checking geometric feasibility before tuning. Cap clearance credit to avoid
   rewarding the existing excessive left lift.
5. **Preserve the task.** Keep speed tracking, uprightness, motor constraints and
   flight penalty. Avoid solving the symmetry objective by hopping on both feet
   or standing still. Introduce new penalties gradually and inspect their measured
   scale relative to tracking and the total-reward floor/caps.

The target is symmetric *step function*, allowing small joint/torque differences
needed by the asymmetric physical model. Exact mirrored actions are optional
regularization only after verifying the v2 mirror transform and feasible joint
ranges. No reward alone guarantees the desired gait.

## Validation and next experiment

Before training, replay the existing checkpoint with per-foot contact, clearance,
stance/swing duration, foot progress and slip traces. Verify left/right labels,
linkage freedom and joint-limit headroom; a mechanical asymmetry must not be
mistaken for a reward problem. Compare reward values on valid alternating steps,
right-foot drag, left-foot drag, contact chatter, standing and hopping.

Use a bounded fine-tuning pilot from GAIT-02 best at 0.35 m/s first. Keep the
original checkpoint as baseline and log every reward contribution. Promote a
checkpoint only if both legs make repeated alternating forward steps, clearance
and slip are acceptable, left/right timing and step-length differences are within
predeclared tolerances, and survival/tracking are retained. Illustrative initial
tolerance: within 20% between sides over several complete cycles, to be checked
against model feasibility rather than presented as an established standard.
Then test 0.20 and 0.50 m/s, inspect video, and only afterward extend speed.

Periodic contact/velocity rewards have precedent in
[Siekmann et al., ICRA 2021](https://arxiv.org/abs/2011.01387).
The implementation choices above are proposals tailored to this code and
observed failure; the paper is not evidence that these weights will work here.
