# GAIT-09 — discourage incomplete and repeated foot swings

Date: 18 October 2026. Snapshot: `gait09_20261018_v1`.
Run: `cad_v2_gait09_s0`. Submission/status: see
[STATUS](experiments/gait09_20261018_v1/STATUS.md).

## Completed results

Job **3222597 completed**, exit 0:0, in **3h26m45s**, reaching **32,014,336**
additional steps. Preflight checks passed and `final.msgpack` was saved. Metrics,
evaluation history, progress and the job log are archived in the status directory.

At 0.25 m/s, comparing the inherited policy reevaluated under GAIT-09 with the
final checkpoint (12 trials, quality after five seconds):

| Metric | Baseline | Final |
| --- | ---: | ---: |
| Surviving trials | 12/12 | 12/12 |
| Fully qualified trials | 0/12 | 0/12 |
| Mean absolute speed error, m/s | 0.01819 | 0.00881 |
| Contact-conditioned slip, m/s | 0.01098 | 0.00578 |
| Valid steps, left/right | 10.17 / 10.00 | 10.00 / 10.00 |
| Detected attempts, left/right | 20 / 20 | 20 / 20 |
| Incomplete-attempt fraction | 49.58% | 50.00% |
| Repeated same-foot attempts, left/right | 10 / 10 | 10 / 10 |
| Valid-step alternation | 99.21% | 100.00% |
| Mean maximum cycle mismatch | 0.2983 | 0.3808 |
| Clear-time fraction, left/right | 11.38% / 8.70% | 7.67% / 8.62% |

No training-target trial qualified in any of the 32 recorded evaluations. The
best diagnostic candidate is step **18,874,368**, with 12/12 survival, 0.01279 m/s
tracking error, 48.13% incomplete attempts and unchanged 10/10 repeated attempts.
It is not an accepted clean gait. All evaluated checkpoints remain on SCITAS.

Final transfer evaluations: 12/12 survive at 0.20 m/s; 0/12 survive at 0.35 m/s,
with mean first episode 3.61 s. The policy was trained only at 0.25 m/s.

Interpretation: tracking and contact slip improved, but this continuation did
not remove micro-stepping. Perfect valid-step alternation still coexists with
extra attempts. The new counters expose that failure; a larger reward/longer
training claim is not supported by these results. Next inspect candidate and
final videos and attempt timing before selecting another correction. No new
run was submitted during this status check.

## Video replay

[Compare candidate and final with slow-motion controls](../report/gait09-videos.html).
Replay job **3223177** completed in **3m15s**, exit 0:0. Each checkpoint was
replayed at 0.25 m/s, seed 1000, using the frozen training environment. Saved MJX
poses were rendered locally using geometry verified against the training manifest.
Two simultaneous camera angles; left foot blue, right foot orange; original
recording speed 1x. Each clip ends at first termination or 25 s, without resets.

- [Candidate, step 18,874,368](../report/videos/gait09_best_0.25mps.mp4): 25 s,
  no fall, mean speed 0.250 m/s. After 5 s: 11/10 valid left/right steps,
  20/20 attempts, 9/10 incomplete attempts, 10/10 repeated attempts.
- [Final, step 32,014,336](../report/videos/gait09_final_0.25mps.mp4): 25 s,
  no fall, mean speed 0.248 m/s. After 5 s: 10/10 valid steps, 20/20 attempts,
  10/10 incomplete attempts and 10/10 repeated attempts.

These are single illustrative replays, separate from the 12-trial evaluation.
Attempt/contact/clearance traces and checkpoint hashes are retained alongside
[replay provenance](experiments/gait09_20261018_v1/replay-provenance.json).

## Post-video diagnosis and proposed next test

The final replay shows a regular second airborne bout on each foot, rather than
only brief contact chatter. After five seconds, per-foot medians are:

| Bout | Forward body-relative travel | Peak sole clearance | Air time |
| --- | ---: | ---: | ---: |
| Left valid step | 25.67 cm | 2.68 cm | 0.37 s |
| Left incomplete repeat | 3.14 cm | 1.41 cm | 0.12 s |
| Right valid step | 19.88 cm | 4.65 cm | 0.25 s |
| Right incomplete repeat | 0 cm positive travel | 0.92 cm | 0.13 s |

Travel is the detector's nonnegative body-relative endpoint difference, not world
foot displacement. In particular, zero positive travel does not establish a
stationary foot. These are ten bouts of each type per foot in one replay, not
population estimates.

Final 12-trial reward means (before common timestep scaling): survival plus
tracking = 4.80855 per tick; repeat cost = 0.02000; incomplete cost = 0.00581.
Combined new event costs are only about 0.54% of those two income terms. This
comparison explains their small scale, but does not establish a causal optimum:
removing an extra lift could change other costs, stability and tracking too.
Costs are applied at landing, so their event weights must be compared with
income integrated over the stride, not just with per-tick coefficients.

Lift, forward-progress and pair-progress bonuses are all exactly zero in the
final evaluation. The expected-foot mask, phase gate and credit budgets can each
suppress them. Current traces lack controller/reward phase and budget state,
so timing misalignment is a hypothesis to test, not yet a proven cause.

Recommended next test: instrument those phase/budget signals; align controller
and rewarded swing windows; add a smooth duration-based cost while the previous
foot makes an extra airborne bout before its partner completes a step. Retain
an event cost with a meaningful per-stride budget, and allow contact debounce,
normal foot rolling and bounded landing grace. Apply a soft reward constraint,
not a mechanical foot lock. Preserve survival, tracking and all-direction slip.
Compare a short continuation with fresh learning under the same corrected setup
before committing another long run. If a second motion persists, separately test
regularizing higher gait harmonics; do not assume those harmonics are its cause.

Phase-based gait rewards have precedent in
[Siekmann et al., Periodic Reward Composition](https://arxiv.org/abs/2011.01387),
which demonstrates periodic force/velocity costs for bipedal gaits. That result
supports testing consistent timing, not a guarantee for this robot. These are
recommendations only; this diagnosis did not modify rewards or launch training.

## Question and source

Can the robot replace extra small swings with clean alternating walking?
The [GAIT-08 audit](gait08-reward-audit.md) found that valid-step alternation
ignored micro-steps and that incomplete attempts could replenish partial rewards.

Continue the GAIT-08 **25,165,824-step candidate**, which had better bilateral
clearance than its final checkpoint. Preserve weights, observation statistics
and learned action standard deviation; start a fresh optimizer and curricula.
Parent SHA-256: `e558d8b810c7a260d1d50011e26513a115c26d3faf366a2dbf1aa2ff99ad4ad8`.
This is a continuation, not fresh-policy training.

## Changes

- New opt-in `bilateral_attempt_enable`; GAIT-01 through GAIT-08 retain their
  previous reward logic. No action/observation/model changes.
- An attempt is a debounced whole-foot landing after at least 40 ms airborne,
  with peak sole clearance >=3 mm **or** absolute body-relative travel >=1 cm.
  Toe/heel rolling with one pad still contacting does not count. Valid steps
  still require 80 ms, 1.5 cm clearance and 4 cm forward travel.
- An incomplete landing receives weight 2 times the mean normalized shortfall
  in airtime, clearance and forward travel. It is bounded and becomes gentler
  as a step improves. A meaningful attempt by the last valid foot receives the
  existing repeat-event weight 2, even when invalid. Qualifying repeats are not
  charged twice. Both costs can apply to an incomplete wrong-foot attempt.
- Lift credit has a 0.4 quality-second budget per expected-foot turn; forward
  credit uses the greatest reach across attempts in that turn. Neither budget
  resets on landing; only an alternating valid step replenishes them. The pair
  quality signal uses the same budgeted lift credit, preventing unlimited partial
  lift bonuses. The existing next-foot deadline remains unchanged.
- Stance-duration allowance is `stance_ratio / frequency + 0.2 s`, bounded to
  0.2–1.5 s. At 0.5 Hz and 65% stance this permits 1.5 s rather than penalizing
  contact after 1.0 s during a nominal 1.3 s stance.
- Store the stance preceding each swing and commit stance/airtime/travel together
  at its valid touchdown. Extra attempts are separate metrics, not mixed histories.
- Existing bilateral shaping, smoothness and contact drag stay at full strength.
  Only repetition/incomplete event costs ramp from 25% to 100% over five million
  eligible steps, using the existing two-evaluation walking gate. Gate failure
  pauses progression. This reuses `shape_scale`; its GAIT-09 meaning is narrower.
- Evaluations record per-foot attempts, incomplete/repeated attempts, rejection
  reasons, symmetry components, and 10th/50th/90th percentiles of attempted travel,
  peak clearance and airtime. Qualification additionally requires <=10% incomplete
  attempts and zero repeated attempts after five seconds. Candidate selection uses
  incomplete fraction, repetition and symmetry before tracking as tie-breakers
  after the existing bilateral-progress/survival criteria.

## Training and validation

- **32,000,000 additional environment steps**, rounded up to a full rollout.
- Fixed target 0.25 m/s; 512 environments, 32 control ticks per rollout, seed 0.
- Learning rate 3e-5 to 1e-5. Same six motors, rigid links, CAD/mass model and physics.
- No speed curriculum or pushes yet. Cleaner stepping remains the priority.
- Evaluation every 1,048,576 steps: 12 nominal greedy trials at each of
  0.20/0.25/0.35 m/s, 25 s, seed 3103. Quality after 5 s, survival over the entire
  episode. Every evaluated checkpoint is retained; transfer speeds do not mask
  failure at the training speed.
- One GPU, 10 CPUs, 90 GB, six-hour allocation. GAIT-08 took 3h22m for 30M;
  expect roughly 3.5–4 h plus queue, subject to measured throughput.
- Focused local regression suite and SCITAS JAX tests precede training. A finite
  PPO/physics benchmark runs before the long job. Initial full-penalty evaluation
  must have positive mean raw reward, <=10% reward-floor hits and >=80% survival
  at 0.25 m/s; otherwise training exits with diagnostics retained.

## Interpretation and limits

Synthetic checks demonstrate that invalid same-foot repeats are penalized, retries
cannot recollect reach, lift credit exhausts, clean opposite steps release budgets,
small contact noise is ignored, and incomplete costs decrease with better attempts.
They also check mirror symmetry, coherent history and legacy opt-out.

Several related corrections change together; this is a practical improvement run,
not an isolated weight ablation. Symmetry numbers are not directly interchangeable
with GAIT-08 because bookkeeping changed. New acceptance and candidate ranking
are stricter. Phase-offset and actuator architecture are unchanged. Physical
improvement requires evaluation and video; penalties alone do not guarantee
a clean gait. The completed results above show the remaining limitations.
