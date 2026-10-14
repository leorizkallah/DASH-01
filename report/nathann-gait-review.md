# Nathann's gait and braking methods — code review, 2026-10-13

Historical reference only. The proposed follow-up was set aside at the user's
request on 2026-10-14; it is not an active experiment plan.

This is a source-code review, not a new simulation result. No training was launched.

## What generated our walking policy

The bundled `controller/deploy/bundles/dash_joy3_lr_s2_180M.npz` identifies
`spec_source=policy`, `action_dim=47`, run `dash_joy3_lr_s2`, checkpoint
`ckpt_180486144.msgpack`. This is the PPO route, not the offline gait-library route.

`RLframework/gait.py` defines 41 gait parameters: three families of third-order
Fourier joint trajectories, phase-varying proportional/derivative gains, cadence,
left/right phase and amplitude relationships, and offsets. Six further outputs
provide corrections each control tick. `RLframework/env.py` commits gait parameters
at cycle boundaries and applies residuals at 100 Hz. The actor thus adapts a
structured gait; it does not merely replay one globally optimized trajectory.

PPO optimizes the weighted reward in `RLframework/env.py`. The walking checkpoint
score in `RLframework/train.py:keeper_score` combines tracking error, fall rate
and heading error; optional bring-up survival contributes to the fall score.
"Best" therefore means best under the chosen objective, model and evaluations,
not a proof of global gait optimality or five-second stopping ability.

## Separate older gait-library optimizer

`sprint_runner/walk_mit/library/library_solve.py` implements CMA-ES over 17 reduced
gait parameters, expanding them into the older 44-dimensional specification.
`fixed_point.py` seeks a repeated-cycle state and measures local disturbance growth
using Floquet multipliers. The executed cost includes squared target-speed error,
periodicity defect, falls, and actuator/contact/workspace envelope violations.
Candidates at successive target speeds warm-start from the preceding solution.

These are weighted penalties, not guaranteed feasibility constraints. In particular,
`--lam-bar` defaults to zero: stability multipliers are reported without being
penalized. The standing entry is inserted from a stance state, rather than solved
and validated by the gait optimizer. No generated gait-library JSON was found in
the restored `sprint_runner/walk_mit` tree during this review.

`library/search.py` additionally searches a gait while freezing its learned
stabilizer, scoring dash time, residual effort, thermal load, lateral drift,
asymmetry and falls. This older stack/model and specification are not a drop-in
replacement for the current flat-foot controller. The restored 3.06 m/s runner's
saved configuration uses the older Fourier PPO action mode; its performance should
not be attributed to the library optimizer merely because both are in that tree.

## Existing braking search: the more relevant next experiment

Git revision `dd2bdad13e0cfa61587370b379c7209754309594` contains
`legacy_trainer_to_port/tools/brake_search.py` (parent of Nathann's `c5edf08`).
It uses the cross-entropy method (CEM), distinct from CMA-ES, to search four
channels with three time knots each: cadence, cam/thigh amplitude, cam offset and
thigh bias. It changes the gait specification while keeping the policy's residual
feedback. Candidate schedules are evaluated in parallel, with repeated episodes
and optional staggered braking times. Elite candidates update the search distribution.

The current `controller/deploy/controller_v2.py:BrakeSchedule` and
`RLframework/export.py:read_brake` still contain support for that 12-number format.
Comments report older braking successes and dependence on source checkpoint,
starting state and schedule duration. Those comments are historical evidence,
not a validation on our present robot/configuration.

Critical measurement difference: the archived search accumulates the MINIMUM
absolute forward speed reached while alive. Its positive verdict requires this
minimum below 0.3 m/s and survival, not five continuous quiet seconds. Momentary
slowing or reversing can therefore satisfy its speed criterion. Its negative
verdict also overstates what a finite unsuccessful search can establish: failure
to find a stop does not prove the action space cannot express one.

## Recommendation

Adapt the small braking-schedule search to the current model/checkpoint, with our
real approach, five-second quiet hold, support, drift and survival criteria.
Include dense guidance for partial progress, but select successful schedules using
physical success and validate across unseen gait phases, seeds and initial speeds.
Keep the existing gait generator and feedback initially. If a valid transition is
found, use those trajectories as demonstrations for RL and robustness training.
If none is found, inspect action constraints and the search parameterization;
do not treat that result as an impossibility proof or just add more PPO steps.

Also evaluate the already available `BalanceRL` policy/bundle before rebuilding
push recovery. Its README reports limited fore-aft recovery, uses a different
18-output controller, and scores two-second post-push survival rather than our
five-second hold. A direct switch from walking is not validated. The existing
`RLframework/config.py` records 135 unsuccessful fixed-stance handovers on an
older flat-foot checkpoint, reinforcing the need to validate the transition itself.

This revises the previous suggestion to start a fresh push-recovery curriculum:
first reuse and assess the code and policies Nathann already supplied.

Method background: [Hansen's CMA-ES tutorial](https://arxiv.org/abs/1604.00772).
Hierarchical gait selection with a separate low-level controller also has research
precedent, e.g. [Yang et al.](https://proceedings.mlr.press/v164/yang22d.html), but
that quadruped result does not establish that the same method solves DASH-01 stopping.
