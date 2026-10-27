# CLEARANCE-01: 20-30 mm swing clearance at 1.50 m/s

## Completed results — 26 October 2026

SCITAS job **3230928 completed**, exit 0:0, in **3h30m21s**.
The step budget stopped training at **50,006,016 additional steps**;
the eight-hour ceiling was not reached. Final is also the best clearance
candidate (same 50,006,016-step selection). No checkpoint passed every
qualification condition, so no `best_clearance.msgpack` was promoted.

| Matched nominal evaluation, 12 trials | Parent | Final |
|---|---:|---:|
| Mean airborne whole-foot clearance, left | 2.46 mm | **23.62 mm** |
| Mean airborne whole-foot clearance, right | 2.25 mm | **20.91 mm** |
| Scheduled mid-swing samples >=20 mm, left | 0% | **57.18%** |
| Scheduled mid-swing samples >=20 mm, right | 0% | **71.43%** |
| Full 30 s survival | 12/12 | **12/12** |
| Mean absolute speed error, target 1.50 m/s | 0.0257 m/s | **0.0292 m/s** |
| Trials meeting all clearance criteria | 0/12 | **0/12** |

All final trials individually exceed 20 mm mean airborne height for both feet.
They register 99–100 valid completed cycles per foot in the measured 25 s;
none falls or triggers the symmetry termination. However, both feet fail the
**80% scheduled mid-swing coverage** requirement. Higher average lift is substantial
progress, but does not establish consistent phase-aligned clearance. The next
decision should consider the video and height/contact trace, not mean height alone.
Final dense reward-floor activation is 1.44% of live control ticks.

Data-quality note: at additional step 14,321,664, one of the 2,713 logged updates
contains nonfinite clearance-component/height diagnostics. Its logged rollout
mean reward is finite (2.433); this alone does not establish the source of the
diagnostic failure. Raw entries are retained. Nominal clearance evaluations
used for the plots have finite height and tracking metrics.

These are nominal simulated trials with one evaluation seed, not evidence of
hardware performance, push robustness or retention across different commands.
The matched parent evaluation differs slightly from the earlier single replay
(2.44/2.17 mm) because the evaluation batch and duration differ.

[Video gallery and clearance graphs](../report/clearance01-videos.html),
[evaluation CSV](../report/data/clearance01/evaluations.csv),
[raw final evaluation](experiments/clearance01_20261025_v1/latest_clearance_eval.json).
Short replay job **3231710** records the final checkpoint using the frozen
experiment simulator. It does not train or modify the policy.

The preparation contract below is retained as the pre-training specification.

### Replay discrepancy

Replay job **3231710**, one nominal environment with a pinned 1.50 m/s command,
falls at **3.67 s**. This conflicts with the original 12/12 full-duration
evaluation and is retained, not discarded. Replay job **3231713** repeats
recording with **12 environments and the saved fixed-command reset path**.
It completed successfully in **2m21s**: all 12 first episodes last the full
25 s, each with 80 valid completed cycles per foot after warmup. The first
trial is shown in the main video, with no selection among trials. Its aligned
pose/contact measurements give **23.55/20.91 mm** mean airborne height and
**46.55/40.80 mm** median complete-bout peaks (left/right). Complete bouts must
last >=40 ms for the peak statistic; the means include every airborne sample
after the 5 s warmup. These differ from the training evaluation's 30 s window.
The two protocols change batch size and command-reset handling together, so
the cause is not isolated. Nominal evaluation survival must not be presented
as proof of reliable standalone startup. The recorder change adds an opt-in
`--keep-reset-command`; it does not alter training, the model or frozen source.

Prepared 25 October 2026, Europe/Zurich, at the user's request. Submit one
50-million-step fine-tuning run on SCITAS and stop after submission; no continuing
Codex monitoring. The independent batch process logs and checkpoints automatically.

## Parent and objective

- Experiment: `clearance01_20261025_v1`; preset `cad_v2_clearance01`;
  run `cad_v2_clearance01_s0`, seed 0.
- Warm start from flat-feet v1 **best at 184,320,000 steps**, not its final policy.
  At 1.50 m/s this parent completed the nominal 25 s replay. Mean whole-foot swing
  height L/R was **2.44/2.17 mm**, pooled 2.31 mm; typical peaks **6.26/5.59 mm**.
  See [measured baseline](../report/flatfeet-clearance.md).
- Parent: `/home/rizkalla/running_robot/experiments/dash01_flatfeet_symmetry_v1/RLframework/runs/dash01_flatfeet_symmetry_v1_s0/best.msgpack`.
  Verified SHA256: `b1d82e549c8402b185fb90eb5665ebe4d334838f65b81b55e3268443c693f3cc`.
- Restore actor, critic, estimator, normalization and learned action noise. Start
  a fresh optimizer and step counter. Request **50,000,000 additional steps**
  (complete rollouts may overshoot slightly), within an **8-hour allocation**
  with the established 10-minute save reserve. No automatic continuation.
- Fixed command **1.50 m/s**: normalized command `.6` with `v_max=2.5`.
  Command and difficulty ramps do not restart from a cold-policy curriculum.
- Preserve CAD-v2 homing model, six motors, 384/47 actor interface, PPO, gait
  generator, slip costs, mid-stance flat support and completed-cycle symmetry.
  The known contact-chatter/history blind spot is not separately rewritten here.

## Reward change

Enable only through `swing_clearance_enable`, default off for historical presets.
Height `h` is the minimum signed corner distance of **both oriented sole pads**
to the floor, clipped at zero for reward calculation. Toe-only lift cannot meet
the target while the heel stays down.

For each foot, the existing gait clock defines normalized swing progress `u`.
The first and last **20% of swing** use linear height ramps between ground level
and the target; the middle **60%** uses the full **20 mm minimum**. The envelope
`e` runs from zero to one and back. It is zero throughout stance. This permits
normal liftoff and landing instead of demanding an instantaneous 20 mm jump.
It does not require a stop or pause between steps.

```text
target(u) = 0.020 * e(u)
deficit = clip((target(u) - max(h, 0)) / 0.020, 0, 1)
low_clearance_cost = -8.0 * sum_over_feet(deficit)

extra = clip((max(h, 0) - 0.020) / (0.030 - 0.020), 0, 1)
extra_clearance_bonus = +1.0 * sum_over_feet(e(u) * extra * bonus_gate)
```

- At mid-swing, a foot at zero clearance costs **-8**; 10 mm costs **-4**;
  20 mm or higher costs zero. There is useful improvement below 20 mm, unlike
  the old positive-only clearance term's dead zone.
- At mid-swing, bonus is zero through 20 mm, **+0.5 at 25 mm**, **+1 at 30 mm**,
  with no additional height income above 30 mm.
- The low-height cost applies during scheduled swing **even if that foot never
  leaves the floor**. Gating this cost on airborne contact would let dragging
  avoid the penalty.
- Bonus requires actual force-qualified airborne state and an air duration no
  longer than the existing `swing_fresh_s=0.45`. Keeping a leg aloft indefinitely
  cannot keep collecting the bonus. Both terms are zero during STOP/zero commands
  or scheduled stance. Each foot is charged separately, once, independent of
  how many pad contacts it has.
- The coefficients above are before the existing control-rate scale (0.5),
  shaping scale and dense reward floor. Shaping is fixed at 1 for this warm run.
  The maximum deficiency cost per active foot therefore becomes -4 per control
  tick before the global dense floor. Floor activation is logged and the parent
  reward-budget check below guards against destroying the learning signal.
- Set the old `w_clearance=0` **only in this new preset** to avoid a second bonus
  that continues growing toward 50 mm. New components have independent log keys:
  `reward_terms/low_swing_clearance`, `reward_terms/extra_swing_clearance`.

This is a phase-dependent 20 mm requirement, not a physically impossible constant
20 mm throughout takeoff and touchdown. Whole-swing mean height remains a separate
evaluation metric; a 20 mm mid-swing target does not guarantee a 20 mm whole-swing mean.

## Warm-start settings and evaluation

Retain the parent's training values: DR scale **0.28418496**, stance ratio
**0.53129792**, efficiency scale **0.516096**, tracking sigma **0.3**, shaping **1**.
The constant DR interval uses equal start/target endpoints and a one-step span
so the existing initializer uses the intended scale rather than its no-curriculum
fallback of 1.0. No new domain-randomization difficulty is introduced.

The inherited six-motor PD and action ranges remain. Use conservative PPO learning
rate **5e-5 -> 1e-5**, existing adaptive KL rule with target **0.015**. Retain the
parent-compatible optimizer structure for checkpoint decoding. Preserve learned
action noise (no reset to the cold-policy std 0.7), cap std at 0.25 with target
0.18; entropy coefficient 0.001 -> 0.0005. Preserve/freeze normalization statistics
and do not apply a warm-start variance floor or count cap.

Training batch remains **512 x 36**, four PPO epochs. Save periodic checkpoints
every **2M** new steps and the existing wall-clock checkpoint interval. Evaluate
every **200 rollouts** (~3.69M new steps), plus baseline and final quality checks.

`ClearanceEvaluator` uses **12 nominal first episodes**, seed 1000, 30 s cap,
discarding the first 5 s for height/tracking measurements. No reset samples enter
the first episode's statistics. Log per-trial and aggregate:

- Full-duration survival and symmetry terminations.
- Mean airborne whole-foot height L/R and mid-swing fraction reaching 20 mm.
- Tracking error, completed cycles, actual reward and dense floor activation.

Keep `best_clearance_candidate.msgpack` separately from the speed-based `best`.
Rank by survival, tracking-error excess above 0.15 m/s, minimum-foot cycle activity,
minimum-foot mean clearance, then mid-swing threshold fraction. Qualification
requires at least 90% of trials individually meeting all of: full survival,
tracking error <=0.15 m/s, >=8 valid completed cycles per foot after warmup,
**both mean airborne heights >=20 mm**, and both mid-swing threshold fractions
>=80%. Only a qualified candidate becomes `best_clearance.msgpack`.
An unqualified candidate must not be described as successful clearance training.

## Validation and submission

Local: **23 tests pass** (clearance, flat-feet, walking-recipe and walltime tests),
with syntax/static checks. Tests cover smooth edge ramps, monotonic deficiency
cost, the 20 mm crossing, the 30 mm bonus cap, stopped/stance exemptions, negative
height, per-foot symmetry, grounded swing evasion and indefinite-air bonus prevention.

The batch checks frozen source and parent hashes, repeats lightweight tests, then
runs a short **GPU warm-start smoke** with two 16x8 PPO updates. It subsequently
reinitializes from the unchanged parent for the real run. These GPU checks are
pending at preparation time, not claimed as completed local validation.

Before long training, automatically evaluate the inherited policy with the full
new rewards. Abort if survival is below 80%, mean actual reward is nonpositive,
or more than 10% of live steps hit the dense floor. Retain `clearance_baseline.json`
and validation outputs if that safety gate fails. Do not silently weaken the penalty.

Launcher: `RLframework/slurm/izar_cad_v2_clearance01.sbatch`.
Remote experiment: `/home/rizkalla/running_robot/experiments/clearance01_20261025_v1`.
Logs: `RLframework/logs/cad-v2-clearance01-JOBID.out`.
Checkpoints/metrics: `RLframework/runs/cad_v2_clearance01_s0/`.
The submission command writes the job ID locally and remotely to `submission.txt`.

No claim of achieved 20 mm clearance is made before evaluating the trained policy.
