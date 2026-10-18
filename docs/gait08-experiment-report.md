# GAIT-08: longer opposite-foot sequencing continuation

17 October 2026. User approved training with the new next-foot rule and a longer
budget. SCITAS job **3222272**, snapshot `gait08_20261017_v1`.
[Status](experiments/gait08_20261017_v1/STATUS.md) |
[source manifest](experiments/gait08_20261017_v1/source-manifest.json).

## Question and setup

Can a continuous penalty for an overdue opposite-foot step break GAIT-07's
unilateral stepping pattern while preserving its improved survival and tracking?
GAIT-07 final is the starting policy: 12/12 final evaluation survivors at 0.25 m/s,
but mean valid steps after five seconds were only 0.25 left / 8.5 right.
The representative replay made 0 left / 11 right steps.

- **30M additional environment steps**, twice GAIT-07's budget; seed 0.
- Preset `cad_v2_gait08`, run `cad_v2_gait08_s0`, fixed **0.25 m/s** target.
- Warm-start GAIT-07 `final.msgpack`: preserve weights, observation statistics
  and learned action standard deviations. Fresh optimizer and curriculum state.
- Learning rate 3e-5 to 1e-5, no warmup. Starting rate matches GAIT-07's configured
  final rate. Existing exploration schedule retained, starting entropy 0.01.
- Same corrected CAD model, 512 environments, 32-tick rollouts and 100 Hz control.
- One GPU, 10 CPUs, 90 GB RAM, **six-hour wall-time limit**. Based on GAIT-07,
  roughly 3-4 hours of execution is expected, excluding queue time; this is an
  estimate, not the six-hour limit or a guaranteed completion time.

Parent checkpoint:
`/home/rizkalla/running_robot/experiments/gait07_20261017_v1/RLframework/runs/cad_v2_gait07_s0/final.msgpack`

Parent SHA-256:
`64c881d3591ea717a87ef52244f9bf75b5060386f0302079e8e9d49ee5dd8e1d`.

## Reward and curriculum changes

Enable the [next-foot sequencing rule](next-foot-reward.md): after one foot
completes a valid step, the other must follow. Allow half a gait cycle plus
0.2 s for weight transfer and stepping (1.2 s at 0.5 Hz). A waiting cost grows
linearly over the following second to -0.8 before shaping/control-rate scaling.
Repeated same-foot steps cost -2 instead of -0.5 and cannot reset the deadline.
Dense lifting/forward-progress rewards apply to the expected foot, so continued
motion of the already-active leg does not keep earning those bonuses.

Preserve **full-strength contact drag from the first rollout** through
`drag_scale_start=1`. A newly initialized curriculum must not silently weaken
the drag penalty earned by GAIT-07. This start parameter defaults to zero for
previous presets; a regression test checks that the GAIT-08 drag scale stays at
one regardless of gate state.

Other gait shaping starts at the parent's 0.15 scale. The existing walking gate
can raise it and reduce exploration if sustained stepping emerges. The new
sequencing term uses this same scale: its initial maximum waiting cost is 0.12
before control-rate scaling. This is not an immediate jump to full gait shaping.

Keep valid-step criteria unchanged: 1.5 cm whole-sole clearance, 4 cm forward
body-relative travel and 80 ms airborne. Symmetry, weaker-foot activity, clearance,
tracking and sliding rewards remain. No command-band widening or pushes yet;
those remain subsequent stages after reliable bilateral walking.

## Evaluation and interpretation

Evaluate every 64 rollouts (1,048,576 steps) at 0.20/0.25/0.35 m/s, 12 nominal
greedy trials per speed. Qualification remains unchanged; use the training target
for checkpoint ranking. Retain every evaluated checkpoint, including step zero
and final. Check survival, speed error, per-foot steps/clearance/slip, alternation,
overdue-foot fraction and reward-floor incidence. Inspect videos before accepting
the gait or increasing difficulty.

New reward bookkeeping is internal to the simulator; it does not add privileged
foot-contact information to the actor or command particular joint trajectories.
Either foot may start. Contact flicker, scraping, simultaneous hops and repeated
same-foot steps cannot discharge the expected opposite-foot step.

This changes the sequencing reward, learning-rate schedule and training budget
together. It is not a controlled single-factor ablation, and a longer run does
not guarantee escape from the asymmetric strategy. No automatic extension or
next-stage training is scheduled.

## Validation and provenance

Local full suite: **84 tests; 77 passed, seven skipped because JAX is unavailable**.
The launcher verifies all 42 source hashes and the parent checkpoint hash, runs
39 focused tests (including JAX sequence/state-tree and full-drag persistence
checks), then the GPU rollout/PPO benchmark before training. It rejects an
existing run directory to avoid accidental duplicate training.

Uploaded archive SHA-256:
`439e5ee53e90b1c09cf52d58d5e1dd8d76def7e16a04c84f06cea385977bf740`.

Remote root: `/home/rizkalla/running_robot/experiments/gait08_20261017_v1`.
Completed results are recorded below.

First status check: job RUNNING on `i12` (Tesla V100). Source/parent hashes
verified and **all 39 focused SCITAS tests passed**. GPU benchmark underway;
training updates not yet confirmed.

## Completed results - checked 18 October 2026

Job **3222272 COMPLETED** in **3h21m54s**, exit 0, at **30,015,488 additional
steps**. Downloaded 30 evaluation records, final/best summaries and training CSV.
No evaluation fully qualified a gait.

| Final target | Survived 25 s | Qualified | Tracking error (m/s) | Valid L/R steps after 5 s |
|---|---|---|---|---|
| 0.20 m/s | 6/12 | 0/12 | 0.067 | 5.58 / 5.33 |
| 0.25 m/s | 12/12 | 0/12 | 0.017 | 10.17 / 10.00 |
| 0.35 m/s | 10/12 | 0/12 | 0.098 | 9.83 / 9.25 |

The primary-target comparison uses GAIT-08's inherited step-zero policy and its
final policy, under the same evaluator:

| Metric at 0.25 m/s | Step zero | Final |
|---|---|---|
| Valid steps, left/right, after 5 s | 0.17 / 8.75 | 10.17 / 10.00 |
| Alternating fraction | 7.5% | 99.2% |
| Contact-slip speed, contact-conditioned after 5 s | 0.0579 m/s | 0.0075 m/s |
| Per-foot slip speed over first episode, L/R | 0.110 / 0.035 m/s | 0.0092 / 0.0108 m/s |
| Overdue-next-foot fraction, L/R | 92.1% / 1.55% | 0% / 0% |
| Tracking error | 0.0506 m/s | 0.0172 m/s |

This is substantial progress in bilateral stepping and reduced sliding. It is
evidence for the combined sequencing/training setup, not an isolated causal proof
for one reward term or the longer budget.

### Why final qualification is still zero

All 12 primary trials pass survival, tracking, heading, minimum step count,
alternation, flight, slip and joint-limit checks. All fail these two checks:

- **Clear time:** whole-sole clearance exceeds 1.5 cm for 10.88% of measured time
  on the left and 7.08% on the right, versus the required 8% for each foot.
- **Cycle symmetry:** the mean per-trial maximum normalized mismatch across stance
  time, swing time and forward step length is 0.645, versus a limit of 0.20.
  Similar step counts do not establish equal timing or equal step length. The
  saved summary does not identify which component dominates; a trace is needed.

Best saved diagnostic candidate is **25,165,824 steps**:
`eval_25165824.msgpack` (also saved as `best_candidate.msgpack`). At 0.25 m/s it
has 12/12 survival, exactly 10/10 valid steps per trial, 100% alternation, mean
clear-time fractions 11.32% / 8.71%, contact slip 0.0112 m/s and tracking error
0.0182 m/s. It still fails full qualification. Its cycle-symmetry statistic
averages 0.672. Preserve both this policy and final for video comparison.

Other gait shaping reached full strength by the 6,291,456-step evaluation; drag
stayed at full strength throughout. Final entropy coefficient is 0.002. Final
training episodes averaged 22.41 s; reward-floor incidence was 0.56%. These
stochastic training metrics differ from nominal greedy evaluation results.

Recommended next evidence: replay the 25.17M candidate and final at the training
target, then inspect each foot's swing/stance duration and travel. Poor transfer
at 0.20 m/s and residual cycle asymmetry argue for assessing gait quality before
adding pushes. No replay or new training was launched during this status check.

## Video comparison - 18 October 2026

Subsequently requested paired replay job **3222519** completed in **3m18s**,
exit 0. Open the [video comparison page](../report/gait08-videos.html), including
the earlier GAIT-07 clip for reference.

| Checkpoint | Video | Outcome | Mean speed after 5 s | Valid L/R steps after 5 s | Clear-time L/R |
|---|---|---|---|---|---|
| 25,165,824 candidate | [Watch](../report/videos/gait08_best_0.25mps.mp4) | 25 s, no fall | 0.245 m/s | 10 / 10 | 11.45% / 8.80% |
| 30,015,488 final | [Watch](../report/videos/gait08_final_0.25mps.mp4) | 25 s, no fall | 0.239 m/s | 10 / 10 | 10.75% / 7.05% |

Both target 0.25 m/s with seed 1000 and the same nominal model, each recorded as
a fresh one-environment greedy trial. Neither replay has a both-feet-airborne
interval according to its flight metric. These are illustrative trials, not the
12-trial evaluations at seed 3103 or evidence of hardware robustness.

Two simultaneous views, normal speed, 30 fps; left foot blue, right orange.
Colors affect rendering only. The local XML and all 15 meshes match the frozen
training manifest. The candidate's right-foot clear-time fraction exceeds the
final policy's, consistent with the evaluation difference. Equal step counts
still do not establish equal stride timing or length.

Trace metadata:
[candidate](experiments/gait08_20261017_v1/gait08_trace_eval_25165824.json),
[final](experiments/gait08_20261017_v1/gait08_trace_final.json).
Checkpoint, source and video hashes:
[replay provenance](experiments/gait08_20261017_v1/replay-provenance.json).
