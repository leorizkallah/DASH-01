# GAIT-07: gradual contact-drag refinement

17 October 2026. User-approved continuation from GAIT-06 final, submitted to
SCITAS as job **3222141**, snapshot `gait07_20261017_v1`.
[Status](experiments/gait07_20261017_v1/STATUS.md) |
[source manifest](experiments/gait07_20261017_v1/source-manifest.json).

## Question and baseline

Can the improved survival in GAIT-06 be retained while learning to lift and
advance both feet rather than sliding them along the floor? The final GAIT-06
evaluation survived 25 s in 8/12 trials at 0.25 m/s, but had only one valid step
per foot pooled across all trials. Its representative replay survived 25 s at
0.122 m/s mean speed, with no valid steps after five seconds.

The old curriculum never opened because it required valid steps. The existing
slip penalty stayed at 15% shaping strength and saturated at 0.10 m/s contact
slip. This experiment introduces an independent drag ramp, leaving the original
bilateral rewards and gait acceptance criteria intact.

## Initialization and budget

- Run `cad_v2_gait07_s0`, seed 0, **15M additional environment steps**.
- Warm-start exact GAIT-06 `final.msgpack` at 15,007,744 parent steps. Preserve
  weights, observation statistics and learned action standard deviations.
- New optimizer and curriculum state; learning rate 1e-4 to 3e-5, no warmup.
  Starting rate matches GAIT-06's configured final rate. Entropy starts at 0.01;
  the existing walking-success exploration schedule remains in use.
- Fixed **0.25 m/s** training target, same nominal CAD plant, 512 environments,
  32-tick rollouts, 100 Hz control. No speed randomization or pushes in this stage.
- Parent SHA-256:
  `c70063fd9d2159d877de96285d51ceb2965c1f55eb4344076db9f6fccf0507dd`.
  Launcher verifies this hash and all 41 source hashes before training.
- One GPU, 10 CPUs, 90 GB RAM, four-hour limit. No automatic extension.

## Reward change

Continue measuring RMS horizontal velocity of the actual active foot-floor
contact points, separately for each foot. It is zero without active contact.
The horizontal norm charges forward, backward and lateral sliding equally.
World contact-point velocity distinguishes sliding from a stationary support
foot moving backward relative to the torso, and from rolling about a stationary
contact point.

Keep the old `bilateral_slip` reward. Add a separately logged term:

```text
x = contact_slip_speed / 0.10 m/s
cost(x) = x*x                  if x <= 1
          2*x - 1              otherwise
contact_drag = -0.24 * drag_scale * mean(cost(left), cost(right)) * gait_on
```

The extra term starts at **zero**. Its scale increases to one, independently of
`shape_scale`; there is no double multiplication by the existing 0.15 factor.
The cost and slope are continuous at 0.10 m/s and remain increasing above it.
At 0.10 m/s slip on both feet, the new term reaches -0.24 before normal environment
reward scaling, versus -0.06 from the old term at shape 0.15. The combined cost at
that reference speed is therefore five times the baseline once fully ramped.
This coefficient is an experimental choice, not a validated optimum.

Symmetry, weaker-foot activity, missing steps, swing-contact, clearance,
alternation, lift and forward-progress rewards remain present. Their existing
walking gate and shaping/exploration ramp remain separate: clean steps can still
unlock those later. Step definitions stay at 1.5 cm clearance, 4 cm forward
body-relative travel and 80 ms airborne; neither counts nor acceptance thresholds
are relaxed in this experiment.

## Independent progression gate

Only the 12 nominal greedy trials at the **training target** control this gate.
A trial must survive its first six seconds, average at least 0.125 m/s forward
speed and at most 0.15 m/s absolute tracking error over that window. **No valid
step prerequisite.** All 12 saved final GAIT-06 trials satisfy these motion
conditions, although they have zero valid steps in that window.

At least 6/12 trials must pass in two successive evaluations. The extra penalty
then ramps linearly over **five million eligible training steps**. A failed
evaluation pauses further increases; two passing evaluations reopen the gate.
Already-earned strength is retained. Duplicate records do not count as new
evidence. Standing, reverse motion and early falling cannot open the gate.
Evaluations occur every 1,048,576 training steps, so regression detection can lag
by one interval. Both ramp states persist in checkpoint JSON sidecars.

## Evaluation and interpretation

Keep the 25 s evaluations at 0.20/0.25/0.35 m/s, qualification checks, early-fall
metrics and retention of every evaluated checkpoint. Step zero evaluates and
stores the inherited policy before any updates, providing the direct baseline.

New first-episode metrics, separately for each foot:

- Mean contact-slip speed over all active episode ticks, including zero-slip swing.
- Integrated RMS contact-slip speed in metres. This is a slip-distance proxy,
  not the trajectory length of a particular material point.
- Fraction of episode ticks with contact-slip speed greater than 0.02 m/s.

Auto-reset episodes are excluded. Compare slip together with survival, achieved
speed, valid steps, clearance and alternation: less slip by standing or falling
early is not improvement. Existing candidate ranking is unchanged; every
evaluated checkpoint remains available for inspection. Training CSV records the
additional reward, drag scale, gate success and reward-floor incidence. Full-strength
evaluation reward-floor rates must not be confused with current training rates.

This changes reward shaping and the optimizer schedule together; it is not an
isolated coefficient ablation. Penalties encourage reduced sliding and do not
impose a hard no-slip constraint. Sustained clipping or lost survival would call
for revisiting the ramp/weight before increasing difficulty further.

## Subsequent difficulty stages

1. **Current stage:** cleaner stepping at the fixed target. Retain the baseline
   and examine videos as well as numerical metrics.
2. **Variable commands:** after repeated strong qualification at 0.25 m/s
   (proposed promotion gate: at least 90% qualified in two successive evaluations),
   gradually widen a narrow command band and evaluate smooth speed transitions.
3. **Small pushes:** after reliable clean stepping across that band, introduce
   perturbations gradually and measure recovery without sustained foot drag.

These later stages are recorded as the intended sequence, not automatically
launched jobs or guarantees that this run will reach them. This run introduces
no new standing/homing task.

## Validation and provenance

Local suite: **74 tests; 69 passed, five skipped because JAX is unavailable**.
Focused tests cover the independent movement gate, nonwalking rejection,
consecutive evaluations, pauses, checkpoint-state serialization, all-direction
slip cost, rolling contact, reset masking, preserved gait settings and independent
PPO ramps. SCITAS runs the focused suite, including JAX integration, then a GPU
rollout/PPO benchmark before training. Source archive SHA-256:
`85d22a52b1f97b34961c3229c862f5b7dc7c7d21814577cda86d13692c10a925`.

Remote root: `/home/rizkalla/running_robot/experiments/gait07_20261017_v1`.
Job **COMPLETED** on `i15` in **1h47m16s**, exit 0, at **15,007,744 additional
steps**. Source/parent hashes verified and all 29 focused SCITAS tests passed.
GPU benchmark and training completed.

## Completed results

Downloaded all 16 gait evaluations, final and best-candidate summaries and training
CSV. The final policy is also the best saved diagnostic candidate.

| Target | Survived 25 s | Qualified gait | Speed error (m/s) | Valid L/R steps after 5 s |
|---|---|---|---|---|
| 0.20 m/s | 12/12 | 0/12 | 0.039 | 0.00 / 8.17 |
| 0.25 m/s | 12/12 | 0/12 | 0.049 | 0.25 / 8.50 |
| 0.35 m/s | 11/12 | 0/12 | 0.098 | 0.42 / 6.17 |

At the training target, per-foot mean contact-slip speed changed from
**0.095 / 0.073 m/s (left/right)** at step zero to **0.112 / 0.036 m/s** at the
end. The fraction of first-episode ticks with slip above 0.02 m/s changed from
21.5% / 20.3% to 23.1% / 15.0%. These are all-first-episode means, including
zero-slip airborne ticks; different lifetimes and swing fractions affect them.
Right-foot stepping and slip improved together, but left-foot sliding remains.

The independent drag gate worked: it reached full scale by the 6,291,456-step
evaluation. The original walking gate never earned shaping progress, leaving
other gait rewards at 0.15 and entropy at 0.01. That does not mean symmetry rewards
were absent; it means their strength never increased. At 14.68M, primary survival
temporarily fell to 6/12 before recovering to 12/12 at the final checkpoint.
The final training row reports 14.27 s mean episode length, 0.227 m/s mean forward
speed and 2.37% reward-floor incidence. Greedy evaluation floor incidence is 10.94%
under full evaluation shaping and is not the training rate.

Use the in-run step-zero baseline for this comparison: it had 11/12 survivors at
0.25 m/s, versus 8/12 in the historical GAIT-06 evaluation despite the verified
same parent checkpoint. Those evaluations are not numerically identical; the
cause of that difference has not been isolated. Neither establishes independent
hardware robustness.

The result supports further work on bilateral stepping before broader speeds or
pushes. It does not support treating stronger global drag penalties as sufficient
to produce symmetry. Review the clips before choosing the next reward change.
No additional training submitted during this completion check.

The [training figure](../report/figures/gait07_progress.png) records survival,
per-foot steps and slip, and both shaping scales across the run. Regenerate with
`python report/scripts/build_gait07_assets.py`.

## Video replay

Replay job **3222197** completed in **3m13s**, exit 0. Open the
[video comparison page](../report/gait07-videos.html), which also includes the
previous GAIT-06 clip at 0.25 m/s. Final and best candidate are the same
15,007,744-step GAIT-07 checkpoint, so they are not duplicated.

| Target | Video | Duration / outcome | Mean speed after 5 s | Valid L/R steps after 5 s |
|---|---|---|---|---|
| 0.20 m/s | [Watch](../report/videos/gait07_final_0.20mps.mp4) | 19.25 s / fell | 0.184 m/s | 0 / 7 |
| 0.25 m/s | [Watch](../report/videos/gait07_final_0.25mps.mp4) | 25 s / survived | 0.223 m/s | 0 / 11 |
| 0.35 m/s | [Watch](../report/videos/gait07_final_0.35mps.mp4) | 25 s / survived | 0.270 m/s | 0 / 8 |

All three illustrative trials are retained. Replay uses seed 1000, distinct from
the 36-environment evaluation at seed 3103. The 0.20 m/s replay fall despite the
12/12 nominal evaluation survivors shows that robustness is not established.
At 0.25 m/s, the left foot never meets the 1.5 cm whole-sole clearance threshold
after five seconds; the right foot does for 2.6% of that window.

Two simultaneous camera angles, 30 fps, normal speed, left foot blue/right orange.
Clips end at first termination or 25 seconds. Colors affect rendering only. Local
XML and all 15 meshes match the frozen training manifest. See
[trace metadata](experiments/gait07_20261017_v1/gait07_trace.json) and
[video provenance](experiments/gait07_20261017_v1/replay-provenance.json).
