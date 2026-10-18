# GAIT-11: fresh, slow, continuous alternating walking

Prepared 19 October 2026. [Live status](experiments/gait11_20261019_v1/STATUS.md).

[Watch candidate and final videos](../report/gait11-videos.html), including
0.10/0.15/0.20 m/s and slow-motion controls. [Learning curves](../report/figures/gait11_progress.png).
Replay job 3223517 completed in 4m27s. Both training-speed clips survive 25 s;
final has four repeated right-foot attempts after five seconds. The candidate
0.10 m/s clip survives with no incomplete attempts but low clearance; final
falls at 9.73 s. Both 0.20 m/s clips fall (candidate 6.97 s, final 2.89 s).
All six requested replays are retained; these single trials use seed 1000,
whereas the multi-trial evaluation below uses seed 3103.

## Completed outcome

Job 3223293 finished normally after 7h51m56s and 65,650,688 steps. The intended
wall-time reserve triggered and the final checkpoint/evaluation were saved.
All 67 tests passed; all 64 evaluations had zero qualified trials at 0.15 m/s.

| Metric at 0.15 m/s, mean over 12 trials | Saved candidate (49.28M) | Final (65.65M) |
|---|---:|---:|
| Survived 25 seconds | 12/12 | 12/12 |
| Speed error, m/s | 0.0190 | 0.0164 |
| Valid steps L/R after 5 s | 8.83 / 8.83 | 9.92 / 9.92 |
| All meaningful attempts L/R | 13.00 / 13.08 | 11.25 / 15.17 |
| Incomplete attempts | 32.11% | 24.44% |
| Repeated attempts L/R | 2.75 / 2.83 | 0.83 / 4.75 |
| Clear-time fraction L/R | 4.35% / 3.27% | 4.72% / 1.43% |
| Maximum symmetry mismatch | 0.702 | 0.310 |
| Contact slip, m/s | 0.0444 | 0.0214 |

The final policy moves and survives at its training command, with fewer incomplete
attempts than GAIT-10's roughly 50%. This comparison changes speed, initialization
and rewards together and does not isolate which change helped. Persistent right
foot retries and low whole-sole clearance still prevent clean-gait qualification.
Accepted-step counts are filtered by the new turn rule, so equal counts must not
be mistaken for proof of naturally alternating movement. Final transfer survival
is 0/12 at 0.10 m/s and 6/12 at 0.20 m/s: slower is not automatically easier for
this policy. Inspect the replays before choosing the next training change.

## Why start fresh

GAIT-10 survived at 0.25 m/s but still produced roughly 50% incomplete attempts.
Its lower repeat counter did not establish clean walking: attempts could move
around the last-valid-foot definition. This run initializes new network weights,
optimizer and observation statistics; it does not inherit GAIT-10's policy.

## Walking objective and fluidity

The target is **0.15 m/s**, with alternate left/right supported swings. A valid
step requires at least 80 ms airborne, 15 mm sole clearance, 40 mm forward
body-relative foot travel and a supported touchdown. These are minimum quality
criteria, not a prescribed exact trajectory or fixed step duration.

The expected foot is selected randomly at reset. After its valid, debounced
touchdown, the opposite foot is immediately eligible to swing on the next control
tick. **There is no required zero speed, settling dwell or fixed pause between
steps.** Brief double support is allowed. The 20 ms support filter applies to
initialization/retries; successful steps do not incur an extra dwell. Losing the
supporting foot during a swing invalidates that attempt. No mechanical constraint
forces a foot down: the policy must learn the rule through observations/rewards.

An incomplete or wrong-foot attempt cannot advance the turn. Repetition is counted
across all meaningful attempts, including incomplete ones. Unauthorized airborne
time has a continuous cost. Tiny contact chatter remains filtered. Accepted-step
alternation alone is now true by construction, so videos, total/incomplete attempts,
unauthorized airtime and slip are necessary to establish actual gait quality.

## Lessons retained

| Prior finding | GAIT-11 treatment |
|---|---|
| Excessive full penalties caused early falls | Fresh negative shaping starts at 5%; uprightness, speed tracking and positive progress remain active |
| Survival can hide standing or sliding | Track forward speed and retain weaker-foot activity, clearance and bilateral symmetry |
| Drag could never ramp if clean steps were a prerequisite | Independent motion-based drag gate; no valid-step prerequisite |
| Forward-only drag leaves other sliding exploits | Actual contact-point motion penalized in all horizontal directions |
| Small lifts could repeatedly earn progress | Per-turn lift/reach budgets reset only after accepted completion |
| Reward clipping masked heavy repetition penalties | Repeat-event and unauthorized-air costs remain outside the dense reward floor |
| Last-valid-foot counting missed repeated failed attempts | Remember the last meaningful attempt, independently of valid-step history |
| A clock can conflict with event-based steps | Explicit expected-foot supervisor replaces the bilateral phase penalty |

Shaping and drag ramps require two successive evaluations where at least half
the training-target trials last six seconds, advance at least 0.075 m/s, and
have early speed error at most 0.15 m/s. A failed evaluation pauses progression.
Shaping rises from 0.05 to 1 over five million competence-earned steps; drag
uses its separate inherited ramp. No pushes or speed curriculum in this run.

## Architecture change

The 100 Hz policy still outputs 47 gait-generator/residual controls; physics runs
at 1 kHz. Four supervisor features are inserted before the actor's commit flag:
two expected-foot bits, swing-active and support-present bits. This changes the
observation width and requires a fresh policy. Raw contacts remain in the critic,
but the new actor commands are **derived from simulated contacts**. Hardware
deployment needs a validated contact/step estimator and matching supervisor;
the existing deployment runtime is not yet compatible with these observations.

## Budget, evaluation and reproducibility

One SCITAS GPU, 512 environments, 32 ticks per rollout. **Eight-hour allocation;
no environment-step cap.** Setup, compilation and evaluation count toward the
allocation. Training stops with a ten-minute reserve for final checkpoint and
evaluation. Checkpoints are also saved every million steps or 15 minutes, at
update boundaries. An 80-million-step learning-rate schedule is only a schedule
horizon, not a training termination limit. Queue time does not consume allocation.

Evaluate 12 nominal greedy trials each at 0.10, 0.15 and 0.20 m/s; the 0.15 m/s
trials control qualification and curricula. The random baseline is recorded but
is not required to walk before training. Nonfinite updates still fail preflight.
Save diagnostic candidates as well as genuinely qualified policies. Inspect
candidate/final videos before deciding whether the gait is satisfactory.

Source hashes and archive are frozen with the experiment. Regression tests cover
legacy rewards, immediate opposite-foot eligibility, wrong-foot and support-loss
rejection, incomplete retries, clean alternation, JAX state compatibility and
wall-clock stopping. Results and submission details belong in the linked status.
