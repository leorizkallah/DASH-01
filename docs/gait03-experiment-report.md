# GAIT-03: bilateral stepping pilot

15 October 2026. SCITAS job **3219917**, snapshot `gait03_20261015_v1`.
Completed in **40m29s**, exit **0:0**, at **5,013,504 additional steps**.
The final policy survived all 12 nominal trials, but **0/12 met gait criteria**.
Right-foot dragging remains unresolved according to the per-foot metrics.

## Question and baseline

Can direct bilateral step rewards correct GAIT-02's right-foot dragging while
retaining forward speed tracking? In the 0.35 m/s baseline replay, seconds 5–25,
the whole left sole clears 1 cm in 13.15% of samples, versus 0.35% on the right.
Survival and speed tracking alone therefore do not establish a walking gait.
See the [approved plan](gait03-symmetry-plan.md) and
[baseline diagnostic](experiments/gait02_20261014_v1/foot_asymmetry_diagnostic.json).

The right actuated joints were not pinned against configured limits in that
replay: minimum margins were approximately 0.156 rad (hip), 0.446 rad (cam),
and 0.382 rad (thigh). Left hip/thigh soft-limit overshoots reached approximately
0.0024/0.0254 rad. This does not prove mechanical correctness; limit violations
are now included in gait acceptance. The corrected CAD model is unchanged.

## Implementation

`RLframework/bilateral_gait.py` provides shared NumPy/JAX step bookkeeping.
A touchdown counts only after at least 80 ms airborne, 1.5 cm whole-sole clearance,
and 4 cm forward foot travel relative to the torso. Contacts require two consecutive
control ticks to change state. Simultaneous valid landings receive no step credit.
The first valid step earns credit; later credited steps must alternate sides.

| Added reward | Coefficient and purpose |
|---|---|
| Weaker-foot swing | −0.4 times squared normalized deficit below 15% clear-time EMA (1 s time constant) |
| Missing steps | −0.2 times mean per-foot deficit, starting at 3 s without a valid step and saturating at 6 s |
| Alternating touchdown | +3 per valid alternating step |
| Repeated same-foot touchdown | −0.5 per valid nonalternating step |
| Cycle asymmetry | −0.3 times mean capped squared excess above 20% left/right difference in stance duration, swing duration and step length |
| Phase/contact mismatch | −0.5 times mismatch, with left/right expected swing phases half a cycle apart |
| Swing clearance deficit | −0.3 times mean squared normalized clearance deficit during expected swing |
| Contact slip | −0.4 times mean capped squared tangential contact-point speed, normalized by 0.1 m/s |

The new terms follow the walking gate and shaping multiplier, ramped from 0.25
to 1 over 2M steps. Old summed air-time, clearance, phase-contact and toe-center
slip rewards are disabled in this preset because their replacements are above.
Speed tracking, uprightness, support-duty guard, flight penalty and motor
constraints remain. Slip uses active penetrating foot/floor contact-point
velocities, not force-weighted contact detection. This accounts for foot rotation
around a stationary contact. Mirrored-action regularization remains disabled.

Actor observations and the control architecture are unchanged. This experiment
adds reward/evaluation bookkeeping, not a prescribed joint trajectory or a new
balance controller. Homing, startup and stopping remain deferred.

## Training and provenance

- Preset `cad_v2_gait03`, run `cad_v2_gait03_s0`, seed 0.
- **5M additional environment steps at a fixed 0.35 m/s target.** This is a
  bounded correction pilot before expanding speed or spending a larger budget.
- Warm-start GAIT-02 best at 28,311,552 new-stage steps, retaining policy weights
  and exact observation statistics; optimizer and schedules restart.
- Learning rate 5e-5 to 1e-5, 100-update warmup; entropy coefficient 0.003 to
  0.0005. Efficiency shaping retains its final 0.3 target.
- Allocation: one GPU, 10 CPUs, 90 GB RAM, four-hour wall-time cap.
- Frozen source: [SHA-256 manifest](experiments/gait03_20261015_v1/source-manifest.json).
  Remote root: `/home/rizkalla/running_robot/experiments/gait03_20261015_v1`.
- Source checkpoint:
  `/home/rizkalla/running_robot/experiments/gait02_20261014_v1/RLframework/runs/cad_v2_gait02_s0/best.msgpack`.
  SHA-256: `bd6aad4a148c86d26d3963f417c4a20778b8dd461d035d6e18feddde096a6ab7`.
  The job verifies this hash before loading.

## Evaluation and checkpoint acceptance

A greedy gait evaluation runs before training, periodically and at the final step:
12 nominal trials, seed 3103, 25 s each. Quality metrics exclude the first 5 s;
survival and maximum joint-limit overshoot cover the full trial. Each trial needs:

- Survival, mean speed error ≤0.10 m/s and mean heading error ≤15°.
- At least three valid steps per foot and at least 8% clear time per foot.
- At least 90% alternating steps.
- Mean normalized left/right stance, swing and step-length differences ≤20%
  for each quantity, using the latest completed measurements when both feet
  have a valid step history.
- Flight fraction ≤1%, mean contact slip ≤0.08 m/s, and maximum actuated
  joint-limit overshoot ≤0.01 rad.

At least **11/12 trials** must pass to permit a regular best-checkpoint update.
Passing final weights are also saved as `final_qualified.msgpack`.
`best_candidate.msgpack` preserves useful attempts even when none qualify; it
must not be described as an accepted gait. Per-trial results and reward means
are written to `gait_evaluations.jsonl` and `latest_gait_eval.json`.
These are declared simulation thresholds, not hardware validation. The 1%
flight allowance is not proof of strict continuous-ground-contact walking.

## Validation and results so far

Local suite: 51 tests, 48 passed and three JAX-dependent tests skipped because
local JAX is unavailable. Six new bilateral tests cover dragging, standing,
alternating steps, contact chatter, hopping, left/right exchange, contact-point
velocity and acceptance failures. All six also passed on SCITAS. The point
velocity formula matched MuJoCo's CPU geom-velocity calculation for both feet
in a sampled state.

The job passed its prerequisite GPU/PPO benchmark and completed training.
Downloaded [evaluation history](experiments/gait03_20261015_v1/gait_evaluations.jsonl),
[final metrics](experiments/gait03_20261015_v1/latest_gait_eval.json) and
[candidate record](experiments/gait03_20261015_v1/best_candidate_eval.json).

| Metric (mean across 12 nominal trials) | Inherited baseline | Final |
|---|---:|---:|
| Survived full 25 s | 12/12 | 12/12 |
| Qualified gait | 0/12 | 0/12 |
| Speed error | 0.05747 m/s | 0.03574 m/s |
| Valid steps, left / right, seconds 5–25 | 10 / 0 | 10 / 0 |
| Clear time, left / right (1.5 cm threshold) | 12.56% / 0.0167% | 13.30% / 0.217% |
| Alternating steps in measured window | 0% | 0% |
| Flight fraction | 3.11% | 3.42% |
| Mean contact slip | 0.08371 m/s | 0.07733 m/s |
| Maximum actuated joint-limit overshoot | 0.02437 rad | 0.02837 rad |

All six evaluations, including the baseline, had zero qualifying trials.
Survival briefly regressed to 8/12 at 3,145,728 steps before recovering.
The diagnostic candidate score selected the final checkpoint, but this does
not certify walking. Every final trial failed right-step count, right clear
time, alternation, flight and joint-limit criteria. The reported symmetry value
also fails; without a valid bilateral history it can be a fallback value rather
than a measured cycle comparison. Speed tracking and average slip improved,
but this pilot did not establish alternating walking. No new replay was generated
during this status check, and no continuation job was submitted.

After completion, compare baseline and candidate per-foot metrics and inspect
a 0.35 m/s replay. Only a convincing bilateral gait should advance to tests at
0.20/0.50 m/s and a longer or broader training run.
