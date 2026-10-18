# Next-foot sequencing reward

Prepared after GAIT-07 video review. The user requests a penalty when one foot
steps forward and the opposite foot does not follow. The opt-in feature is now
enabled in [GAIT-08](gait08-experiment-report.md), submitted as job **3222272**
for a 30M-step continuation. GPU validation precedes training. Historical presets
remain opted out; the new preset sets `bilateral_sequence_enable=True`.

## Why the existing alternation term was insufficient

The existing `bilateral_repeat` term already penalizes successive valid landings
by the same foot, and `bilateral_step` rewards alternation. However, repetition
is only charged at completed steps, while unilateral lifting/forward movement
can still receive dense progress rewards. In GAIT-07, the left foot rarely
completed valid steps despite the right doing so repeatedly.

## Added behavior

Either foot may make the first valid step. After it lands, the opposite foot is
expected next. Give it half a gait cycle plus 0.2 seconds to complete its step.
This allowance is latched at the accepted landing and bounded to 0.2-1.5 seconds;
changing cadence while waiting cannot extend it. At the observed 0.5 Hz gait
frequency, the allowance is **1.2 seconds**.

Once overdue, a continuous cost grows linearly to its maximum over one additional
second. Before the existing shaping and control-rate factors:

```text
bilateral_next_wait = -0.8 * clip((time_waiting - allowance) / 1 second, 0, 1)
bilateral_repeat    = -2.0 for a repeated valid step by the same foot
```

Only a valid opposite-foot touchdown clears the waiting timer. A repeated step,
scraping, contact flicker or an in-place lift cannot reset it. Repeated valid-step
cost increases from the legacy 0.5 to 2.0 when this feature is enabled. Lift and
forward-progress bonuses then apply only to the expected foot; either foot can
earn them before the first valid step. Weaker-foot progress, clearance, symmetry,
speed tracking and sliding rewards remain in place.

The term uses the existing `shape_scale`: at scale 0.15, maximum waiting cost is
0.12 before normal control-rate reward scaling. This change alone does not repair
the stalled broader shaping curriculum. Weight and grace values are initial
experimental choices and must be assessed with survival and reward-floor metrics.

No extra waiting penalty applies before the first valid step; existing missing
step and weaker-foot costs still cover that case. The active policy's reward
history, not an actor-visible command or forced joint trajectory, tracks whose
turn it is. Normal weight transfer is allowed; there is no instantaneous demand
to lift as soon as the other foot touches down.

## Measurement and validation

Valid-step criteria remain 1.5 cm whole-sole clearance, 4 cm forward body-relative
travel and at least 80 ms airborne. Simultaneous landings do not count. A separate
future experiment could relax these thresholds, but no relaxation is included here.

Training logs record expected-next-foot and overdue-foot activity, along with
the new reward term. Gait evaluations record per-foot overdue fractions over the
existing post-five-second window. Neither new diagnostic changes gait acceptance.

Focused local suite: **37 tests, 34 passed and three skipped for missing JAX**.
Tests cover either starting foot, normal grace, gradual penalty, repeated-step
and cadence loopholes, opposite-step completion, simultaneous hops, left/right
symmetry, wrong-foot progress bonuses, and legacy opt-out behavior. The skipped
JAX test for this feature checks the compiled update and unchanged state-tree
structure. GPU evaluation and real-policy replay with the new reward are still
pending at the time of preparation; no learning improvement is claimed.

GAIT-08 launch update: all 39 focused tests, including the compiled JAX sequencing
and full-drag persistence checks, subsequently passed on SCITAS. The full GPU
benchmark was in progress at the first status check.
