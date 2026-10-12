# STOP-02 experiment log

Started 2026-10-06 (Europe/Zurich), following user approval of the
[staged plan](stop02-training-plan.md). This log separates checks, training, and
measured competence. A successfully submitted/completed job is not a successful stop.

## Implementation

- Shared command adapter v1: operator STOP stays asserted; actor receives a
  smooth cubic speed reference in BRAKE, then zero speed plus its STOP bit in HOLD.
  Duration is max(2 s, 1.5*v0/0.5), with a 100 ms causal speed-estimate filter.
  Estimate units are reconstructed from privileged-observation normalization and
  base-velocity scaling, not taken from simulator ground truth. Checkpoints exported
  with this adapter carry those constants and its version in bundle metadata.
- Adapter state belongs to each environment; STOP edges snapshot the initial
  speed, repeated STOP requests do not restart the ramp, and random command redraws
  cannot interrupt the required braking-plus-holding horizon. Legacy presets
  retain their original behavior; actor input/action dimensions are unchanged.
- Actor speed input and reward target use one prepared value before every action.
  PPO collection, greedy evaluation and the NumPy controller use this path.
  Walking shaping fades below 0.25 m/s during BRAKE. Standing penalties wait for
  HOLD so necessary braking steps are not penalized as failed standing.
- New standing preset: 75% of initial states request HOLD immediately from the
  existing supported keyframe with the original joint perturbations; other resets
  retain RUN. Subsequent random command draws retain the old 25% STOP probability.
  STOP command durations are 16–18 s; episode cap is 30 s. Noise and domain
  randomization are disabled for this nominal feasibility pilot.
- HOLD adds individually logged dense speed (maximum 2), angular-motion (1),
  joint-motion (1), and support (0.5) incomes, gated by upright posture and height.
  Existing multiplicative hold income remains (maximum 3). A duration income rises
  from 0 to 0.5 across five continuously quiet seconds, paid while supported;
  motion or >0.10 m displacement resets its timer. The pose cost weight decreases
  from 2 to 0.5 and oscillation cost from 10 to 1. These are explicit experimental
  changes, not individually proven improvements. Immediate braking income is off
  in this preset; speed tracking follows the reference instead.
- Five-second success is evaluated independently of reward: speed <0.10 m/s,
  angular speed <0.30 rad/s, joint RMS speed <0.50 rad/s, tilt <10 degrees,
  support >=90% and maximum displacement <=0.10 m within the window. No reset
  can rescue a failed first episode. Standing evaluation permits 2 s settling
  then 5 s quiet; transition evaluation permits the measured ramp plus 2+5 s.
  Transition rollout length also accommodates the maximum configured ramp.
- Pilot budget: 2M steps, 1024 environments x 64 ticks, evaluations every eight
  rollouts (~0.524M steps), plus explicit final-checkpoint evaluation. LR warmup
  is 100 minibatch updates rather than the prior 1000 for the much shorter pilot.
  Discount gamma=0.995 and GAE lambda=0.95 remain unchanged: learning uses dense
  per-tick hold income and bootstrapped values, not a lone delayed finish reward.
  Passing this pilot is not automatic authorization by metrics to advance: the
  subsequent speed stage is gated by measured standing and walking retention.

## Validation and diagnostic jobs

- Local tests: 15 pass, covering ramp endpoints/maximum reference deceleration,
  STOP edges/repeated requests, actual NumPy controller command timing, uninterrupted
  five-second windows, support/drift, rejection of falls followed by resets,
  original STOP metrics and explicit torso sensing. Python source compiles.
- SCITAS check job `3204730`, snapshot `stop02_20261006_v1`: tests the unchanged
  policies with abrupt and ramped STOP, nominal zero-action stance feasibility,
  and final-policy standing with dense rewards. The nominal zero-action probe
  removes reset joint noise only to test whether the target stance is physically
  attainable; it is not learned-policy success or a robustness claim.
- The v1 diagnostic snapshot precedes the additional duration income, final
  checkpoint evaluation hook, and two extra integration tests. Its exact source
  hashes are retained in `docs/experiments/stop02_20261006_v1/source-manifest.json`.
  Training will use a separate frozen snapshot with its own validation.

## Results and decisions

Job `3204730` completed its first physical probe but failed while serializing a
NumPy integer in the summary. The saved recording was recovered and scored
locally: 32/32 nominal fixed-action stance trials passed, mean live speed
0.00556 m/s, maximum tilt 1.55 degrees, mean scaled reward 5.9163, no reward-floor
hits. This is a feasibility probe, not a learned-policy result.

Job `3204978` runs the corrected frozen snapshot `stop02_20261006_v2` (archive
SHA-256 `c11ae1acc40060f118eed023bb506b270ffe51c4dcc1bf5a8e1fa39852a6d265`).
Fifteen unit tests pass on SCITAS. The repeated nominal fixed-action probe with
duration income passes 31/32 trials, mean live reward 6.0620 and reward-floor
fraction 0.054%. The changed compiled computation can produce different physical
trajectories; these are two nominal probes, not proof of perfect robustness.
Both establish that the model/action interface can represent a quiet stance.

[Nominal stance video](../RLframework/results/stop02_checks/stance_feasibility.mp4)
uses the first recording and labels it as a fixed-action feasibility probe.
The corrected validation job completed normally in 16 min 38 s (exit 0:0).
All unchanged-policy comparisons still scored 0/32 successful holds. The original
bundle's ramp trial has 1.124 m/s estimator error; its original privileged velocity
normalization is absent from the bundle, so this comparison cannot establish the
effect of a correctly calibrated physical speed ramp. Do not use that bundle to
initialize the new command adapter without recalibrating its estimator.

The STOP-01B final checkpoint has 0.0383 m/s approach estimator MAE, making it the
selected warm start. Under the ramp its live-STOP reward-floor fraction is 9.63%
(24.71% under abrupt STOP). Its new dense-reward standing baseline still scores
0/32, with mean longest quiet period 0.0194 s, mean live reward 2.1112 and 11.49%
reward flooring. The reward audit therefore passes the existing <50% floor guard;
the policy's standing competence remains unestablished.

Raw summaries and the ramp recording are archived in
`RLframework/results/stop02_checks/`. The first pilot uses a new v3 source snapshot
with reporting additions (braking tracking error, distance, drift and phase of
termination), robust JSON handling, and a bounded extension of the training
episode cap when a late STOP needs its full ramp+2+5 s window. Sixteen local tests
pass. This timeout change does not alter the earlier diagnostics, whose complete
protocol fits inside the ordinary 30-second cap.

The pilot is limited to 2M steps. It must pass standing and walking-retention
checks before any faster transition stage. Reports are generated on the cluster
after evaluations and on job exit, avoiding repeated manual polling.

## Standing pilot submission

- Slurm job **3204982**, submitted 2026-10-06; run `dash_stop02_stand_s0`.
- Scheduler check on 2026-10-07: **COMPLETED**, exit 0:0; outcome below.
- Frozen snapshot `/home/rizkalla/running_robot_experiments/stop02_20261006_v3`.
- Archive SHA-256 `ba1fd4e3915248588fa2d45ed853facf004fb6ec9b6330f210cd9bce3c577f18`.
- One GPU, 10 CPU cores, 90 GB RAM, two-hour wall-clock limit; 2M PPO-step budget.
- Warm start: exported STOP-01B final actor/estimator at 20,054,016 steps;
  fresh critic fitted before PPO. Training seed 0, evaluation seed 4102.
- Evaluations every 524,288 PPO steps and at final checkpoint. No faster stage
  is scheduled automatically. No learned STOP-02 result is available at submission.

The final-policy command-only comparison increased median time from STOP to
termination from 0.98 s (abrupt) to 3.83 s (ramp), counting trials that reached
STOP. Both still failed all 32 holds. This is improved survival during the
transition, not evidence that the robot has learned to stand. The representative
[ramp video](../RLframework/results/stop02_checks/final_ramp_1mps.mp4) shows the
reference decreasing during BRAKE, followed by tipping during HOLD.

Refresh the pilot's measured report and CSVs on demand:

```powershell
.\RLframework\slurm\stop_progress.ps1 -Snapshot stop02_20261006_v3 -Run dash_stop02_stand_s0 -Job 3204982 -LogPrefix stop02-stand
```

The cluster job writes its reports without an active assistant session. Local
copies update when this command runs; no repeated assistant polling is required.

## Completed pilot outcome — reviewed 2026-10-07

Job 3204982 ran on i04 from 2026-10-06 18:20:15 to 18:44:47 Europe/Zurich,
elapsed 24 min 32 s. It completed normally with 2,031,616 PPO environment steps
(31 full rollouts). `final.msgpack` and its metadata exist; the final checkpoint
was evaluated explicitly. Metrics and the generated report were downloaded into
`docs/experiments/stop02_20261006_v3/`.

| PPO steps | Five-second holds | Mean longest quiet period | Walking retained |
|---:|---:|---:|:---:|
| 0 | 0/32 | 0.0188 s | Yes |
| 524,288 | 0/32 | 0.0200 s | Yes |
| 1,048,576 | 0/32 | 0.0328 s | Yes |
| 1,572,864 | 0/32 | 0.0288 s | Yes |
| 2,031,616 | 0/32 | 0.0328 s | No |

All final trials terminated for excessive tilt; mean first termination time was
2.054 s after the near-standing start. No trial survived the seven-second standing
benchmark, and the longest quiet period in any final trial was only 0.06 s. Thus
the small change in quiet-period average is not meaningful standing competence.
These are repeated fixed trials across checkpoints, not independent samples to pool.

Final RUN-only mean survival was 59.52%, versus 67.26% at baseline: a loss of
7.74 percentage points, exceeding the permitted five-point loss. Tracking error
improved from 0.4143 to 0.3498 m/s, but that does not override the survival gate.
Live-STOP reward-floor fraction changed from 11.30% to 10.68%. Reduced reward
flooring did not produce a learned hold. `best_stop.msgpack` remains step zero.

Decision: do not advance to faster stopping stages or automatically extend the
same pilot. Next diagnosis should compare the learned HOLD actions with the known
feasible fixed-stance actions, then assess whether a supervised standing
initialization or a more targeted standing curriculum is needed. This is a proposed
next investigation, not an implemented change or an additional training launch.

## Replay and next-method discussion — 2026-10-07

The longest-surviving recorded moving approach in the STOP-01B final-policy ramp
diagnostic is trial 23: requested 2 m/s, STOP at 3.65 s, ramp duration 5.151 s,
termination at 9.99 s (6.34 s after STOP). It never achieves a quiet hold.
The [full half-speed video](../RLframework/results/stop02_checks/longest_upright_stop_trial23.mp4)
is selected by survival duration, not stopping success; longer braking duration
contributes to that survival time. All stopping trials still fail.

A replay-only job, 3205811, was submitted for STOP-02 checkpoint 1,048,576.
This checkpoint ties the best mean longest quiet period and retains walking,
unlike the final checkpoint. The job was queued at the last check; it does not
perform training or modify the frozen training source/checkpoints.

Method recommendation: retain a moderate ramp, but prioritize learning an actual
standing controller. Standing-only trials fail without walking momentum, so ramp
duration alone cannot address the measured failure. Explore initialization from
the feasible stance (imitation or a stance reference plus learned corrections),
retain walking examples, then learn recovery from small perturbations. The fixed
stance is a nominal starting point, not a demonstrated robust controller. Advance
to low-speed braking only after five-second holding succeeds. Also inspect actual
speed tracking and make the BRAKE/HOLD handoff depend on low estimated speed and
stable posture, with a bounded timeout, rather than elapsed ramp time alone.

Relevant primary references:

- [legged_gym implementation](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
  provides velocity tracking and a separate stand-still reward term; term
  availability does not mean every configuration enables it.
- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html) combines motion
  imitation with task objectives; it motivates example-guided initialization,
  not a claim that its character results directly validate this robot.
- [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201)
  combines conventional control and learned corrections. Its reported task is
  manipulation; using that pattern for our standing controller is a proposed
  engineering adaptation, not a result from that paper on this robot.

No additional training or reward/architecture change was launched from this discussion.
