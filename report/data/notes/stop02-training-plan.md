# STOP-02 proposed training plan

Updated 2026-10-07. Status: implemented and validated; standing pilot job 3204982
completed at 2,031,616 steps but achieved 0/32 holds and failed final walking
retention. No faster stage has been launched. Results and source provenance are in the
[STOP-02 experiment log](stop02-experiment-report.md). Later speed stages remain
gated on standing success and walking retention. Previous measurements remain in
[the experiment report](stop-training-report.md).

## 1. What changes, and why

The operator's STOP request should initiate controlled braking, followed by
active standing balance. Success requires at least five consecutive seconds of
measured quiet standing after braking, not merely reaching zero target speed.

STOP-01 already evaluated eight seconds under STOP, including a quiet final five
seconds. The final policy terminated before that window could be completed:
two trials failed before STOP and the remaining 30 failed 0.25–1.88 seconds
after STOP. Extending the window alone will not fix the transition. The video
shows a gait change followed by loss of balance; it does not prove that inertia
or the instantaneous command change is the sole cause.

In the current joystick branch, STOP immediately sets the reward target to zero
and disables walking-specific shaping. The actor sees a binary mode switch while
its speed input remains the previous walking request. Older `stop_decel_s` and
`stop_cmd_continuous` logic belongs to other objective branches; changing those
configuration fields alone will not implement this proposal.

## 2. One shared command schedule

Introduce explicit WALK, BRAKE and HOLD phases. Keep the operator STOP request
latched throughout BRAKE and HOLD; it must not be overwritten by a random command
redraw. Repeated STOP requests must not restart the braking clock. RUN/restart is
issued only after the evaluation has completed its holding window.

At STOP, snapshot a short, causally filtered forward-speed estimate as v0. Use the
same estimate available to the controller at deployment, not privileged simulator
velocity. Log ground-truth velocity separately for evaluation. Do not continually
reinitialize v0 as the robot moves.

Use a smooth decreasing reference:

`u = clip(time_since_STOP / T, 0, 1)`

`v_ref = v0 * (1 - 3*u^2 + 2*u^3)`

This has zero reference acceleration at the two endpoints and peak reference
deceleration `1.5*v0/T`. Initial candidate: `T = max(2 s, 1.5*v0/a_ref)`, with
`a_ref = 0.5 m/s^2`. This gives T = 3 s at 1 m/s and T = 6 s at 2 m/s. These are
starting values to test, not verified physical limits. The reference is smooth;
the robot's actual motion still depends on the learned controller. Longer braking
also needs more distance (ideal ramp travel is v0*T/2).

Feed v_ref to both the actor's speed input and the tracking reward. During BRAKE,
keep the actor's existing RUN task bit active so the inherited walker sees a
decreasing walking request instead of an immediate standing request. At the end
of the ramp, switch to HOLD with zero speed and the actor's STOP task bit. The
external STOP indicator remains asserted throughout. This preserves the network's
input dimensions but changes the command adapter semantics, which must be versioned
and shared by training, evaluation, and the local/deployment player. Test the final
BRAKE-to-HOLD switch explicitly; moving that switch later is not proof it is solved.

## 3. Time and success criteria

Each test includes a qualifying moving approach, the full braking ramp, up to
2 s to settle, and at least 5 s of uninterrupted quiet standing. Allocate a
post-STOP horizon of at least T + 2 + 5 s; shorten neither the episode nor the
command duration below that horizon. Training STOP durations must also cover it
instead of retaining the previous random 6–10 s interval.

The five-second timer starts only after the ramp ends and measured quiet-standing
conditions are met. A violation resets the timer; the fixed overall deadline
prevents unlimited retries. No auto-reset episode may count as continuation of
the original stop. Keep genuine fall/invalid-state terminations; do not relax
them to make a trial survive. Ordinary time limits must allow the full protocol.

Retain the current physical quiet thresholds initially: horizontal speed below
0.10 m/s, body angular speed below 0.30 rad/s, joint RMS speed below 0.50 rad/s,
and tilt below 10 degrees throughout the five seconds. Require existing support
criteria on at least 90% of that interval. Report horizontal drift as well, with
an initial proposed maximum displacement of 0.10 m from the hold-start position,
so slow sliding cannot masquerade as standing. Verify threshold feasibility before
training rather than silently loosening metrics after observing failures.

Record time/distance to stop, longest quiet hold, speed tracking error during
BRAKE, tilt, support, drift, termination phase/reason and reward components.
Evaluate restart in a separate three-second window after HOLD; restart performance
must not obscure whether the five-second hold itself succeeded.

## 4. Rewards follow the phases

- WALK: preserve the established walking objective and retention benchmark.
- BRAKE: reward tracking v_ref while staying upright and supported. Retain useful
  gait shaping while steps are needed. Fade it as the reference approaches zero.
  Do not apply immediate zero-speed incentives or strong stillness/amplitude
  penalties that punish necessary braking steps.
- HOLD: reward low speed, upright posture, low angular motion, low joint motion
  and support with separately visible dense components. Add a sustained-hold
  bonus, but do not rely on the current near-zero multiplicative holding bonus as
  the only useful signal. Fade posture/amplitude costs in near the ramp's end;
  preserve authority for small balancing corrections.

Choose numerical weights after a reward audit on real traces, checking flooring,
terminal penalties and discounted returns. Verify that a sustained balanced hold
beats drifting, shuffling and early falling. Per-tick positive income alone is
insufficient evidence. Check the value-function/rollout treatment of the longer
holding interval; do not arbitrarily enlarge every PPO setting at once.

## 5. Staged experiments and decision gates

1. **Command-only diagnostic, no learning:** apply the new schedule to the source
   walker and STOP-01B final checkpoint. Record abrupt versus ramped STOP at
   matched initial states/timing, starting at 0.5 and 1 m/s. This isolates whether
   changing the command schedule alone improves braking or merely delays a fall.
2. **Standing feasibility and training:** verify a supported near-standing state
   is attainable through the current action interface. Train short nominal
   trials from near-stance states with small perturbations, mixed with RUN trials
   to retain walking. Do not force a zero motor output or require a frozen gait
   clock; active balance may need corrections. Initial advancement gate: >=80%
   five-second hold success, confirmed on a separate validation set.
3. **Slow transitions:** train 0.5 m/s WALK-to-BRAKE-to-HOLD transitions, retaining
   standing and RUN examples. Progress only after >=80% stop success on the stage
   benchmark, confirmation on held-out timing/seeds, and walking retention.
4. **Increase difficulty:** add 1.0, then 1.5, then 2.0 m/s, with ramp durations
   scaled to speed. Vary STOP gait phase and timing. Test shorter ramps only after
   gentle braking and five-second holding work. Add plant variation gradually
   after nominal competence.

Start each learning stage with a small diagnostic allocation (candidate 2M steps,
evaluation every 0.5M); extend only when stage-specific progress is visible. Audit
early termination and longest hold even while binary success remains zero.
Stop and diagnose a plateau instead of automatically spending another 20M steps.
Stage budgets and gates are proposed experiment settings, not convergence promises.

Use small separate comparisons: abrupt/current rewards; ramp/current rewards;
ramp/phase-aware rewards; and the standing curriculum. Label any simultaneous
changes explicitly. Select the warm-start checkpoint based on these measured
diagnostics and walking retention, rather than automatically selecting the newest
failed STOP checkpoint. Preserve original checkpoints and independent experiment
snapshots throughout.

## 6. Validation, SCITAS launch and reporting

Before a training submission, check ramp continuity/monotonicity and endpoints,
STOP latching, repeated commands, actor/reward schedule agreement, BRAKE-to-HOLD
behavior, adequate episode duration, quiet-timer resets, drift detection and
first-episode-only scoring. Run an MJX stepping/reward audit and inspect recorded
videos. Share those concrete checks and finalized weights before the next training
launch; this document itself does not launch work.

Final evaluation covers all four walking speeds, varied STOP timing, held-out
seeds, and the existing RUN retention limits (survival no more than 5 percentage
points below reference; tracking error no more than 0.10 m/s worse). Save complete
videos through the full five-second hold for successful trials, and through first
termination for failures. Longer 10-second holds are a follow-up robustness check.

The report records each code/configuration change, command adapter version,
checkpoint and source hashes, Slurm job/resources, seed, reward audit, per-phase
metrics, ramp duration and stopping distance, videos, and the decision to advance
or revise the next stage. Label STOP-02 metrics separately because its longer,
ramp-dependent timing and added drift criterion differ from STOP-01.
