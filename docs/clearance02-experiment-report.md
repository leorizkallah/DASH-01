# CLEARANCE-02: 10–25 mm swing-height band from the flat-feet baseline

## Selected-policy video and convergence — 2 November 2026

[Video and graph gallery](../report/clearance02-videos.html) shows the **132.71M
best candidate**, with a dotted checkpoint marker on the archived learning curves.
PNG/SVG/PDF exports include mean reward, speed error, mean/peak airborne height,
scheduled mid-swing >=10 mm coverage, duration survival and full qualification.
Raw evaluation points and 100-update block-smoothed training reward are separate.
No convergence guarantee or repeated-seed statistical claim is made.

Short replay job **3238862** completed in **3m01s**, without training. The video
shows trial 0, selected in advance, from a retained nominal batch of twelve:
25 s, error 0.0360 m/s after 5 s, 80/80 completed cycles. Both cameras show the
same instant; no reset trajectory is spliced into the clip.

The fresh recorder exposes **three early falls** (trials 1/4/5 at
0.70/0.95/0.95 s), giving **9/12 25 s completions**, unlike archived evaluation
12/12 full 30 s. Batch size and saved reset-command convention match, but the
compiled recording computation differs and the cause remains unisolated.
This is further evidence that stable startup is not established. The full
metadata and failure outcomes remain available beside the displayed clip.
No hardware validation or new training has been performed.

## Retry completed — checked 2 November 2026

Job **3232851 completed**, exit 0:0, on **28 October at 00:36:40 Europe/Zurich**,
after **11h07m53s**. Full **200,005,632 additional steps**, stopped at the step
limit. Startup GPU probe, 27 tests and warm-start smoke passed on `ixl01`.

Final 1.50 m/s nominal evaluation: **12/12 full 30 s trials**, mean airborne
height **15.82/18.20 mm** left/right, scheduled mid-swing >=10 mm coverage
**100%/100%**, speed error **0.0271 m/s**. Largest sampled peaks are
**31.75/34.74 mm**; every trial fails the <=26 mm peak acceptance limit.
All other individual duration/tracking/height/coverage/cycle criteria pass.
Thus **0/12 qualify**, and none of the 29 recorded checkpoint evaluations qualifies.

Saved best clearance candidate: **132.71M additional steps**, mean lift
**16.94/20.06 mm**, peaks **30.22/30.04 mm**, error **0.0357 m/s**, 12/12 full
duration, also unqualified. It reduces height excess relative to final. The
minimum lift objective is achieved in these nominal tests; the soft upper cost
does not enforce the full height band. More training was not monotonically better.

[Detailed outcome and archived evidence](experiments/clearance02_20261027_v2/STATUS.md).
No new replay or video assessment, no hardware validation, no new job submission.
The earlier standalone-replay startup discrepancy is not resolved by this check.

## Authorized retry — 27 October 2026

**Resubmitted successfully as job 3232851.** All 27 tests passed on SCITAS;
93 source hashes and the original baseline checkpoint were verified. A real
sleeping-process test confirmed timeout exit 124. Frozen archive SHA256:
`4554e966eaa9836350443081b59eba1b40e44d166811d544899e4ef90daf741c`.
No post-submission training monitoring was performed.

The user authorized a new submission after the failed attempt below.
New isolated experiment: `clearance02_20261027_v2`, same preset and exact
200M-step training configuration, same original flat-feet best parent.
Only startup diagnostics and guards change: 60 s NVIDIA query, 180 s GPU
backend/kernel probe, 300 s CPU unit tests, 1,200 s GPU smoke with stage logs and
stack dumps. Each bounded command receives a 20 s TERM-to-KILL grace. Temporary
per-job CUDA/JAX caches isolate this attempt; existing caches are not deleted.
Node i39 is excluded as a precaution, not a confirmed diagnosis.

[Retry snapshot, verification and receipt](experiments/clearance02_20261027_v2/STATUS.md).
GPU initialization and physics smoke checks remain in-allocation requirements.
No result is claimed from a submission alone. No continuous monitoring requested.

## Job outcome — checked 27 October 2026

Job **3231747 timed out after 24 hours before main training started**.
Hash verification and all 27 tests passed, but the GPU smoke/preflight did not
finish. Neither the preflight result nor the main run/checkpoint directory exists.
The log does not reveal the exact blocking operation, so a driver, hardware or
reward-code cause is not established. No 200M-step result or gait improvement
can be claimed for this attempt. The original flat-feet parent is unchanged.

[Archived status and log](experiments/clearance02_20261026_v1/STATUS.md).
Diagnose and time-bound the preflight before a retry; no new job was launched
during this status check. The original training contract below is retained.

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
