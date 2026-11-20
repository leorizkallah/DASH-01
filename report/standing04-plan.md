# STAND-04 proposal: stronger disturbances with bounded recovery stepping

10 November 2026. Reviewed proposal retained below. **Implementation and launch
are now recorded in [STAND-04 progress](standing04-progress.md)**: three model-specific
continuations, rather than only the strongest V1 candidate. The frozen snapshot
retains the original proposal version; the progress record is the implemented contract.

## Parent and scope

Use the strongest overall checkpoint: **original V1, 81,100,800 transitions**,
`report/data/standing03/results/v1/best.{json,msgpack}`, archived/replayed task
score 78/80. Preserve actor weights and observation normalization; initialize a
new optimizer and new curriculum statistics because the task changes. Preserve
the existing parent checkpoint and planted-task evaluator as regression references.
Checkpoint metadata contains curriculum/step counters, so this needs an explicit
fine-tuning initialization path rather than blindly resuming all old counters.

This parent belongs to **V1 geometry**, not V2/X-shape. If the intended hardware
target is one of the other designs, fine-tune that design's own best checkpoint
and record the different parent; do not directly swap the CAD under the V1 policy.

## Findings in the current implementation

- `BalanceRL/env.py::_push` already samples uniform horizontal azimuth across
  360 degrees. STAND-03 restricts elevation to zero and applies at the torso COM.
- `BalanceRL/ppo.py::update_standing_curriculum` already promotes push strength
  by 10% after >=200 completed hard outcomes, >=90% recovery success and >=95%
  quiet success; retreat is 20% below 70%. Its standing-mode cap is hard-coded
  at 0.15 m/s. Hard samples span [L/2,L], so their average success does not alone
  demonstrate competence at the top of level L or in every direction.
- `standing.py` currently terminates for whole-foot lift >3 mm for >50 ms or
  footprint drift >20 mm. Removing termination alone is insufficient: the drift
  cost and quiet success criterion still reference the original footprint.
- STAND-03 best/final discrepancy motivates preserving quiet-standing regression
  tests and selecting checkpoints on task outcomes, not only dense reward.
- Quiet/push evaluations currently use different quiet-window timing. Use the
  same sustained final-standing requirement for the next task, and report the
  old protocol separately for comparisons.

## Proposed curriculum (initial values, validate before freezing)

1. Validate the loaded parent at 0.10 m/s-equivalent impulses before updates;
   start the new task there, below the prior 0.15 cap. Keep full existing sensor
   noise, unchanged physical parameters, 20% no-push episodes and easy replay
   samples. Avoid adding large model randomization at the same time.
2. Allow bounded recovery steps. Grow horizontal impulse strength by 10% only
   after near-level probes at [0.9L,L], spread over eight direction sectors:
   >=200 completed probes, >=90% overall success, >=80% in each sector, and
   >=95% independent no-push success. Require two independent passing batches
   before promotion; reset level-specific counters. Probe seeds differ from the
   fixed evaluation ladder. Do not count censored/pending outcomes as successes.
3. Hold or retreat on failed gates; retreat by 20% if overall recovery success
   drops below 70%, and reduce difficulty if quiet-standing retention degrades.
   Proposed initial ceiling 0.30 m/s; it is a training ceiling, not a demonstrated
   capability. Raise it only after reviewing direction-resolved results.
4. Once the step-enabled horizontal task is stable, introduce duration 80–150 ms,
   randomized onset, small elevation (start +/-10 degrees), then small torso
   COM-relative application offsets (start +/-2 cm). These are separate
   difficulty axes; do not increase all simultaneously. Normalize impulse by
   actual episode mass and rounded pulse duration, and bound peak force/moment.
   Replace the existing hard-coded 100 ms verdict timing with the actual pulse
   end time before duration randomization is enabled.

The only task is recovery to standing; commanded walking and gait clocks remain
absent. A no-step response to an easy push is a valid success.

## Recovery steps and protection against continuous shuffling

Use task states **quiet stance -> external disturbance/recovery -> settling ->
quiet stance**. Only an external disturbance opens the recovery window; the
policy cannot reopen it by shaking itself. Start with at most about three seconds
after the pulse to begin settling, followed by five continuous seconds quiet.
No new push occurs until the full verdict completes and quiet support is restored.
An episode cannot extend that deadline by taking more steps.

- Keep upright/height/support, torque, smoothness and true contact-point slip
  terms. Support one foot during recovery; flat support and both-feet quiet
  requirements return during settling. Retain soft joint/torque bounds.
- Give **no positive reward for step count, foot movement, alternating feet or
  swing time**. The benefit of a step is avoiding a fall and regaining quiet.
- Charge a modest cost for every real whole-foot liftoff/landing cycle, including
  tiny steps and steps returning to the same place. Add accumulated sole-travel
  and excessive contact-switching costs so sub-threshold motions are not free.
  Keep grounded forward/backward/lateral slip penalized throughout recovery.
- Aim for one or two useful recovery steps; progressively increase the cost
  after roughly three landings. This is a soft efficiency budget, not an initial
  hard rule that prevents discovering a viable response. Actual tolerances and
  weights need contact-noise/trajectory validation and reward-scale checks.
- Do not define a tiny displacement as "not a step" and then leave it uncharged.
  Use force-qualified whole-foot contact, hysteresis and debouncing for event
  detection, with continuous travel/slip costs covering threshold exploitation.
  Heel/toe rocking on one loaded sole is not automatically a whole-foot step.
- Do not force alternating feet or a minimum human-like stride during recovery:
  a small adjustment can be useful, and a second adjustment by the same foot
  can be necessary. Penalize excess activity while optimizing for real settling.
- Once both feet are loaded and sole velocities are low, freeze a **candidate
  landing footprint** at the start of the quiet streak. Require both feet to
  remain near it for five seconds. A disturbance of the streak resets the timer
  but does not move the recovery deadline. Commit the new reference footprint
  once after successful settling, not continuously while shuffling. Keep total
  displacement/stance geometry within a separately validated reachable envelope.
- After settling, further spontaneous steps or sliding incur the quiet-mode
  penalty/failure. No-push trials must retain stillness for the full test horizon.
  Pushed trials must also remain quiet after recovery through the test end;
  earning an early five-second window cannot license later walking.

Do not reward returning to the exact original foot locations: a useful recovery
step often must leave a new stable stance. Penalize unnecessary total travel
without making all relocation impossible.

## Validation and evidence before interpreting improvement

Test step/rocking/sliding detection, sub-threshold shuffling, fixed deadlines,
single footprint commit, pulse-duration timing, no duplicate success events and
quiet-standing retention. Run a short GPU smoke and fine-tuning pilot before a
long continuation. Select on held-out recovery + final-standing outcomes; retain
the parent whenever the continuation loses quiet stability.

Log per direction/level: full duration, physical falls, successful recovery,
time to settle, whole-foot steps, repeated small contacts, total sole path,
grounded slip, residual drift, torque/current estimates and each reward cost.
Report distributions as well as mean score; show random trials and failures.

Reactive stepping for standing recovery has been demonstrated with RL on the
Atalante exoskeleton, including attention to policy smoothness and termination:
[Duburcq et al., IROS 2022](https://arxiv.org/abs/2203.01148).
That supports the general task direction; the thresholds and curriculum above
are our proposed engineering choices, not values taken from that paper.
