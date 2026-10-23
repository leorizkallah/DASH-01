# SPEED-01: fresh maximum-speed discovery on CAD-v2

19 October 2026, branch `dash-01-v2-speed-discovery`.
[Source review](../report/nathann-speed-reward-audit.md) and
[run status](experiments/speed01_20261019_v1/STATUS.md).

## Completed run

SCITAS job **3223896** completed normally on 19 October at 23:39:38 Swiss time,
after 4h52m14s and **59,916,288 steps**. The walltime reserve triggered the stop;
final evaluation and saving succeeded. Thirty nominal evaluations are retained.

The best eligible checkpoint at 37.75M steps survived 16/16 thirty-second trials
at **0.950 m/s**, passing joint/thermal criteria. The final checkpoint survived
16/16 at **1.215 m/s**, but all trials exceeded the 0.01 rad joint-limit tolerance
(peak overshoot 0.0185–0.0203 rad). Both show substantial contact sliding and
almost no whole-foot clearance above 15 mm. Speed and survival improved, but the
metrics do not establish clean walking. Inspect selected and final checkpoint
videos before choosing another reward change or training continuation.
See the linked status for per-foot metrics and downloaded source records.

## Objective and initialization

Fresh policy, optimizer and observation statistics. The preset `cad_v2_speed01`
derives from Nathann's modern `_DASH_SPRINT`, independently of GAIT-11 presets.
The corrected CAD-v2 model, six calibrated motors, rigid Achilles links and torso
mass allowance are retained. The base is fully free; there is no holding device,
assist force, homing/standing task, STOP command or target-speed tracking.

Training maximizes signed world-X speed, so moving in circles does not keep
earning positive speed income. It may discover running or hopping; symmetry or
human-like walking are not promised. The structured Fourier/PD action space is
still a prior: 41 gait parameters plus six residual corrections, at 100 Hz over
1 kHz physics. Actor observations return to the ordinary layout, without GAIT-11's
four contact-derived supervisor commands.

## Reward contract

Only the explicit list in `speed_metrics.REWARD_KEYS` and active-contact slip
contribute. Terms computed by the legacy environment outside this list are
excluded. No alive income, step-count income, alternation/repetition/incomplete
cost, phase-contact schedule, duty/swing quota, clearance bonus or flight cost.

```text
speed_income = 2 * clip(world_vx, -4, 4)
upright_gate = clip((upright_cosine - 0.5) / 0.5, 0, 1)
positive speed_income *= upright_gate; negative income remains negative

dense = speed_income + selected regularization costs
reward = max(0.5 * dense, -0.5) - 150 * fallen
```

There is no finish bonus in the speed task. The 4 m/s reward cap is a numerical
search bound, not a measured robot limit; saturation is logged and must be reviewed
if reached. Cadence is bounded at 0.5–4 Hz, without the preceding 1.5 Hz floor.
No physical contact timing or double-support duration is imposed.

| Selected cost | Raw coefficient / schedule |
|---|---|
| Uprightness / below-height-floor | 5 / 2.5; active from initialization |
| Heading error | 1; active from initialization |
| Lateral speed | 0.5; shaping ramp |
| Action change / residual / residual change | 0.05 / 0.05 / 0.01; shaping ramp |
| Actual foot-ground contact slip | `-0.2 * shape * mean(min((slip/0.1)^2, 4))` |
| Torque / motor speed / positive mechanical power | Inherited 0.0001 / 0.0001 / 0.0002; efficiency ramp 0 to 0.3 |
| Thermal load above configured margin | Inherited bounded thermal cost, coefficient 20 |

Shaping ramps from 0.05 to 1 over 15M steps; efficiency over the same interval.
These are clock-based ramps for this baseline. No GAIT-11 success gates remain.
Signed horizontal slip uses RMS active contact-point speed, so forward, backward
and sideways slip all contribute. Torque-speed constraints and joint command
limits remain active independently of the reward ramps.

## Why no alive bonus is needed for the worst-case fall comparison

At 100 Hz, gamma=0.995 and floor=-0.5, the magnitude of an infinite discounted
sequence at the reward floor is `0.5/(1-0.995)=100`. The terminal fall cost is
150. Therefore postponing a fall is preferable to falling immediately even if
every intervening reward is at the floor. This holds for the reward contract;
it is not a guarantee of exploration, convergence or critic accuracy. It avoids
the exact floor/fall indifference of the previous 100-point fall penalty.

At equal posture/control costs, forward speed earns monotonically more than
standing below the speed cap. Actual gait costs still matter. GPU reward probes
measure zero-action nominal control, sampled fresh-policy motion, full-penalty
zero-action control, and archived GAIT-09 motion under the new reward. The last
is diagnostic only, with its original task feature retained; its weights never
initialize this training process. Finite probes and finite PPO updates are required;
the randomly initialized policy is not required to walk before training.

Falls include nonfinite physics, low torso, excessive tilt, foot penetration,
workspace violation, and active ground contact outside the sole pads. This last
check only covers collision-enabled geoms; the torso-height termination remains
important where CAD visual meshes do not generate contacts.

## Selection and budget

One fresh seed (0), 512 environments, rollout length 32. Five-hour SCITAS allocation
includes validation, compilation and evaluations; stop updates with a ten-minute
reserve for saving/final evaluation. No environment-step cap. The 60M-step
learning-rate horizon is scheduling only, not termination. Checkpoint every 1M
steps or 15 minutes, at an update boundary. Evaluate about every 2.10M steps.

Sixteen nominal, fixed-seed (3103), first-episode 30-second trials. A feasible
speed checkpoint needs at least 90% trials to survive the full duration, remain
within joint/thermal criteria and have post-5-second speed above 0.02 m/s.
Among those, rank world-X distance divided by the full 30-second horizon,
including startup and failed trials. This prevents short fast falls from gaining
an inflated ranking. Store `best_speed.msgpack` only when feasible; always retain
a clearly labeled `best_candidate.msgpack`, final and periodic checkpoints.
This is speed feasibility, not a clean-walking certificate. Sixty-second and
held-out-seed checks remain necessary after this initial experiment.

Per-trial metrics retain actual elapsed time, distance, steady speed, contact/slip,
clearance, flight, heading, motor thermal peak, joint violations and reward-floor
usage. Review videos and more than one training seed before claiming improvement.
