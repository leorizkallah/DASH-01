# CLEARANCE-02: 10–25 mm swing-height band from the flat-feet baseline

Prepared **26 October 2026**, Europe/Zurich. User authorized one **200-million-step**
fine-tuning job, then no repeated monitoring. Experiment `clearance02_20261026_v1`,
preset `cad_v2_clearance02`, run `cad_v2_clearance02_s0`, seed 0.

**Submitted successfully as SCITAS job 3231747.** All 27 focused tests passed on
SCITAS, the batch shell syntax passed, and all 92 frozen source hashes plus the
exact 184.32M parent checkpoint were verified before submission. Archive SHA256:
`5e21b61f32c72127378d0632b44e88fb0f718d8a2bc2a5fd7ddca567e7ce945a`.
No post-submission training monitoring. GPU smoke/baseline gates remain automatic
in-allocation checks, not pre-submission results.

## Starting policy and retained settings

Start from the original **flat-feet v1 best checkpoint at 184,320,000 steps**,
not CLEARANCE-01 and not the flat-feet final checkpoint:
`/home/rizkalla/running_robot/experiments/dash01_flatfeet_symmetry_v1/RLframework/runs/dash01_flatfeet_symmetry_v1_s0/best.msgpack`.
Expected SHA256: `b1d82e549c8402b185fb90eb5665ebe4d334838f65b81b55e3268443c693f3cc`.

Restore actor, critic, estimator, observation normalization and learned action
noise; reset optimizer and step count. The **200,000,000 steps are additional**;
the final complete rollout can overshoot slightly. Keep the homing model and
motor/observation/action interfaces, symmetry termination, mid-stance flat-foot
support, slip costs and 1.50 m/s command. No STOP or new perturbation curriculum.
Inherited plant randomization remains fixed at 0.28418496; stance ratio 0.53129792,
efficiency scale 0.516096, shaping 1 and tracking sigma 0.3. These match CLEARANCE-01's
warm-start settings and the selected parent's training difficulty.

The previous pilot proved larger lift is learnable, but its median swing peaks
were approximately 47/41 mm despite a 30 mm bonus cap. It also exposed a replay
protocol discrepancy: all 12 evaluation-protocol trials survived, whereas a
single pinned-command replay fell at 3.67 s. This run does not claim that issue
is resolved. Preserve both protocols when reviewing its resulting policy.

## Reward contract

Height is the minimum signed floor distance of the oriented toe and heel pad
surfaces, clipped at zero for reward. Each foot is evaluated independently.
Use the same scheduled swing clock, with linear takeoff/landing ramps over the
first/last 20% of swing and full target in the middle 60%. Normal liftoff and
touchdown may pass below 10 mm. No pause between steps is introduced.

For each foot, let `h` be height in metres and `e` the existing swing envelope:

```text
target = 0.010 * e
low_cost = -8 * clip((target - max(h,0)) / 0.010, 0, 1)
extra_bonus = +1 * e * clip((max(h,0)-0.010) / 0.015, 0, 1) * bonus_gate
excess_cost = -2 * max(max(h,0)-0.025, 0) / 0.015 * scheduled_swing
```

`bonus_gate` requires walking, scheduled swing, neither pad force-qualified in
contact, and continuous air time <=0.45 s. Low cost applies even if a scheduled
swing foot stays grounded. All three terms are zero for stance or stopped
commands. Sum left and right separately, then apply the existing control-rate
scale 0.5 and dense reward floor. Keep independent log terms
`low_swing_clearance`, `extra_swing_clearance`, `excess_swing_clearance`.

At mid-swing, per foot: 0 mm costs -8; 5 mm costs -4; >=10 mm incurs no low-height
cost. Bonus is zero at 10 mm, +0.5 at 17.5 mm and +1 at 25 mm or higher.
Excess height costs zero through 25 mm, -1 at 32.5 mm and -2 at 40 mm.
The excess cost keeps growing; the bonus does not. This is an explicitly
penalized upper target, **not a hard geometric clamp or guarantee of <=25 mm**.
No episode terminates solely because a height bound is crossed.

The old positive-only clearance reward remains disabled in this preset. The
new excess-height coefficient defaults to zero in historical presets, preserving
their reward behavior. CLEARANCE-01 retains its 20/30 mm configuration.

## Evaluation and selection

Evaluate 12 nominal first episodes, seed 1000, 30 s, first 5 s excluded from
height/tracking statistics. Record mean airborne height, per-foot maximum
airborne height, scheduled mid-swing fraction >=10 mm, airborne fraction >25 mm,
mean excess height, speed error, completed cycles, survival and floor activation.
Threshold-specific keys use **10 mm**, not a misleading inherited 20 mm label.

Keep normal speed-based checkpoints separate from `best_clearance_candidate`.
Rank candidates by survival, tracking error above 0.15 m/s, weaker-foot cycle
activity, then reduced combined low/excess height deficit, mid-swing coverage,
and mean height capped at the minimum. The selector no longer prefers unlimited
mean lift. Deficit ranking prevents an unchanged low-lift parent from winning
merely because it never exceeds the upper limit.

Qualification requires >=90% of trials individually meeting all: full 30 s
survival, speed error <=0.15 m/s, >=8 valid completed cycles per foot after warmup,
both mean airborne heights >=10 mm, both scheduled mid-swing coverages >=80%,
and both observed airborne peaks <=26 mm (**25 mm + 1 mm evaluation tolerance**).
No qualification is claimed before the results are available. Airborne peaks
are sampled at the 100 Hz control rate and may miss sub-tick extrema.

## Budget, validation and evidence

One GPU, 10 CPUs, 90 GB; **24 h allocation ceiling**, 10-minute final-save reserve.
Stops normally at 200M additional steps. At the previous throughput this is
roughly a 12–14 h run; queue time and hardware variation are not included.
512 environments x36 ticks, four PPO epochs; LR 5e-5 to 1e-5 with existing adaptive
KL rule, frozen observation statistics, retained learned action noise capped at
0.25. Save every 2M steps; evaluate every 400 rollouts (~7.37M), plus baseline/final.
No automatic follow-on job or continuous Codex monitoring.

Local: 24 tests pass; 3 synthetic evaluator tests require SCITAS JAX. Tests cover
the new band, excess penalty, STOP/stance exemptions, bilateral independence,
accurate threshold labels, upper-limit qualification and candidate selection.
SCITAS repeats these checks and verifies frozen source and parent hashes.
Inside the allocation, run two small warm-start PPO updates, then reinitialize
the real run from the unchanged parent. Baseline abort gates: survival <80%,
nonpositive actual mean reward, or reward-floor activation >10%.

Launcher: `RLframework/slurm/izar_cad_v2_clearance02.sbatch`.
Remote root: `/home/rizkalla/running_robot/experiments/clearance02_20261026_v1`.
Logs: `RLframework/logs/cad-v2-clearance02-JOBID.out`.
Metrics/checkpoints: `RLframework/runs/cad_v2_clearance02_s0/`.
[Frozen sources and submission status](experiments/clearance02_20261026_v1/STATUS.md).
