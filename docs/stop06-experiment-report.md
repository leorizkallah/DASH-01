# STOP-06: longer training with timed versus feedback HOLD

Outcome checked 2026-10-13: both arms completed all 5,046,272 steps normally.
Neither achieved a successful moving stop in any recorded evaluation. Final
standing holds were 32/32 (timed) and 30/32 (feedback); final walking retention
passed for timed and failed for feedback. Both selected keepers remain the source
checkpoint, so automatic three-seed assessments and replay videos describe that
source under each adapter, not the final trained policies. See the
[completed outcome and evidence](experiments/stop06_20261012_v1/STATUS.md).

Approved 2026-10-12. Snapshot `stop06_20261012_v1`; training seed 0 for both arms.
The user authorized a longer monitored comparison and measurement of settling time.

## Motivation and hypotheses

STOP-05 ended after 262,144 new PPO steps because walking and standing retention
failed. This short run cannot establish that longer learning would fail. All trials
with valid moving approaches terminated in HOLD. That association motivates a
transition experiment; it does not prove the timer caused the falls.

Hypothesis A: additional learning can improve the original timed transition.
Hypothesis B: checking physical readiness before changing the actor's task to HOLD
improves entry into standing. Both may fail; neither outcome is assumed.

## Matched arms

| Setting | Timed arm | Feedback arm |
|---|---|---|
| Preset | `dash_stop06_timed` | `dash_stop06_feedback` |
| Source | STOP-04 keeper at 262,144 steps | Same checkpoint bytes |
| Maximum new PPO steps | 5M (5,046,272 with full rollouts) | Same |
| RUN commands | 0.25–0.5 m/s, 3–4 seconds | Same |
| STOP commands | 14–16 seconds, extended if required by hold window | Same |
| Initial HOLD resets | 25% | Same |
| Ramp | Existing cubic reference, minimum 2 seconds | Same |
| HOLD entry | At ramp completion | After readiness dwell, bounded by timeout |
| Learning rate | 2.5e-5 to 5e-6 | Same |
| Rehearsal | Frozen STOP-03 HOLD/RUN demonstrations | Same |
| Evaluation | Every 262,144 steps, 32 trials | Same |

Rewards, network dimensions, normalization, actuator settings and perturbations
are inherited from STOP-05. The learning rate is halved relative to STOP-05 in
both arms. Weights/statistics are restored; optimizer, schedules and step counter
start fresh. This comparison isolates the gate within STOP-06, not every change
relative to STOP-05. Run sequential array tasks, one GPU at a time; each task has
a three-hour wall-time limit including assessment. There is no later training stage.

## Feedback transition and measurement

After the speed-reference ramp ends, keep the actor in BRAKE with zero speed
reference while waiting for estimated horizontal speed <0.15 m/s, measured
angular speed <0.5 rad/s and measured tilt <15 degrees for 0.3 continuous seconds.
The readiness thresholds are looser than the five-second quiet-hold criteria.
Use the policy velocity estimator and latest IMU observation, not privileged
simulator velocity for control. No change to actor input dimensions is needed.

Latch HOLD after entry to avoid switching back and forth. Reset the latch and
dwell on RUN/new STOP. Initial standing resets enter HOLD directly. If readiness
has not occurred four seconds after ramp completion, enter HOLD as a bounded
fallback and record a timeout. Timeout is not success; all physical success
conditions still apply. This fallback is an experiment, not a validated hardware
emergency-stop mechanism. The shared NumPy deployment gate and export metadata
are updated consistently; no hardware deployment is included.

Both arms have the same nine-second post-ramp evaluation window: up to four
seconds to settle plus five continuous quiet seconds. Count quiet time only in
actual controller HOLD. Preserve moving-approach qualification, physical quietness,
support, drift and survival checks. Moving command schedules conservatively reserve
the maximum 7.5-second ramp plus nine seconds of STOP, followed by a three-second
restart check. This adds survival exposure beyond the typical two-second ramp.
Separate standing tests retain their seven-second protocol.

Record actual controller HOLD entry time, timeout, and ground-truth readiness
time (same readiness thresholds, 0.3-second dwell) independently. Missing events
are unobserved, never zero-time successes. Record five-second quiet-hold onset,
failure time/phase and distance. Ground-truth readiness is a diagnostic, not proof
of a successful stop, and may occur before a later fall.

## Longer learning without selecting degraded policies

Keep STOP-05 baseline standing floor (80%) and all keeper-selection thresholds.
Allow one million new steps before enforcing the regression early-stop rule.
After that, stop after three consecutive evaluations fail walking retention,
standing retention, or baseline moving-stop retention. A passing evaluation resets
the streak; duplicate-step evaluations cannot increase it. Read the streak from
saved evaluations so resume cannot erase failures. Record `guard_monitor.json`.

The earliest persistent-regression stop is 1,572,864 steps. A grace period and
patience permit temporary degradation; they never make a degraded checkpoint
eligible for keeper selection. Realized budgets can differ if guards stop an arm.
Treat such a result as a monitored comparison, not an equal-step causal estimate.

## Automated evidence

Each arm writes ordinary PPO/STOP metrics plus `SETTLING_REPORT.md`,
`settling_trials.csv`, a learning graph and three-seed keeper assessment. Seed
4102 selects keepers; 5102/6102 are held out. A separate pose recording and
half-speed replay are attempted, with a speed/rotation/tilt/phase plot. The replay
records its own outcome and selection rule; a failure clip is explicitly labeled.
`RLframework/runs/STOP06_COMPARISON.md` summarizes both arms as outputs arrive.

Source archive and input hashes, test results, scheduler IDs and observations are
recorded in [STATUS.md](experiments/stop06_20261012_v1/STATUS.md). No success is
claimed before evaluation. One training seed limits repeatability conclusions.
