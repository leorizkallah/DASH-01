# GAIT-10 — last continuation before a fresh walking objective

18 October 2026. Snapshot `gait10_20261018_v1`, preset `cad_v2_gait10`, run
`cad_v2_gait10_s0`. Live submission/results: [STATUS](experiments/gait10_20261018_v1/STATUS.md).

## Completed results

Job **3223224 completed**, exit 0:0, in **1h43m23s**, reaching **15,007,744**
additional steps. All preflight checks passed. Final checkpoint and each evaluated
policy are retained on SCITAS; metrics and logs are archived locally.

Comparison at 0.25 m/s, 12 trials, quality after the first five seconds:

| Metric | Parent reevaluated under GAIT-10 | Final |
| --- | ---: | ---: |
| Survival | 12/12 | 12/12 |
| Fully qualified trials | 0/12 | 0/12 |
| Speed error, m/s | 0.00884 | 0.00988 |
| Valid steps L/R | 10 / 10 | 10 / 10 |
| Attempts L/R | 20 / 20 | 20 / 20.33 |
| Repeated attempts L/R | 10 / 10 | 10 / 0.17 |
| Incomplete-attempt fraction | 50.00% | 50.41% |
| Weighted extra-airtime L/R, s | 0.867 / 0.946 | 0.254 / 0.082 |
| Mean maximum symmetry mismatch | 0.386 | 1.046 |
| Contact slip, m/s | 0.00593 | 0.01641 |
| Clear-time fraction L/R | 7.73% / 8.62% | 15.47% / 9.32% |

Weighted extra-airtime integrates the smooth penalty gate, not literal total
airborne time. The right-foot repeat counter falls sharply, but total attempts
remain about twice valid steps. This does not establish that its extra movements
vanished: the repeat label depends on which foot last completed a valid step.
Changed timing or bout classification needs video/trace inspection.

No training-target trial qualified in any of the 16 recorded evaluations. Best
diagnostic candidate **11,534,336** has 12/12 survival, 0.01035 m/s tracking error,
46.57% incomplete attempts and repeated attempts 10 / 0.58; it also fails the
agreed progress criteria. Final transfer: 12/12 survive at 0.20 m/s; 0/12 survive
at 0.35 m/s (mean first episode 3.52 s).

The [saved decision](experiments/gait10_20261018_v1/continuation_decision.json) is
`fresh_policy_with_explicit_step_objective`. None of the final five evaluations
met the combined improvement criterion: incompleteness never fell ten percentage
points below baseline. Thus there was a response to the stronger penalties, but
not the sustained clean-walking improvement required to continue this policy.
Per the user's decision, the next experiment should start fresh with an explicit
one-step-at-a-time objective. No new run was launched during this status check.

## User decision and parent

One last **15-million-step continuation** of the current policy, with heavy
penalties on repeated steps. If this does not show meaningful improvement, switch
to a fresh policy with an explicit one-step-at-a-time objective. This run uses
GAIT-09 **final**, step 32,014,336, not the earlier diagnostic candidate.
Parent SHA-256: `a4a1cdea6a7d115f031cc0d1bdcf3433c12366fff6533fb25d89bf8fe8dd7bf2`.
Preserve weights, observation statistics and learned action std; reset optimizer,
step counter and curricula. Keep 0.25 m/s, CAD, motors, physics, actor observations
and action space unchanged. No pushes or varying training speeds.

GAIT-09 final had 12/12 survival and 0.00881 m/s speed error, but 50% incomplete
attempts and 10 repeated attempts per foot after five seconds. Repetition plus
incompleteness cost only 0.02581 per tick versus 4.80855 survival/tracking income.
Replays showed regular extra airborne bouts of approximately 0.12–0.13 s.

## Reward changes

1. **Repeated-attempt event weight 2 → 20.** The existing meaningful-bout detector
   still covers invalid repeats; one valid repeat is not charged twice. Whole-foot
   contacts are debounced; toe/heel rolling is not a new airborne bout.
2. **Extra-airtime weight 4 per tick at full strength.** Penalize the previous
   valid foot while airborne before its partner completes a valid step. A 20 ms
   grace and 40 ms smooth ramp tolerate brief contact noise. A bounded motion
   gate uses peak clearance relative to 3 mm or absolute travel relative to 1 cm.
   Cost stops on contact or when the other foot completes its turn. No foot lock.
3. **Protect those two costs from the dense reward floor.** Otherwise making a
   landing penalty larger would often have no effect. Floor the other reward,
   then add the scaled repetition event/duration costs once. Fall penalties remain
   unchanged. `reward_floor_hit` now measures clipping of that floored base, not
   intentional negative repeat-event rewards. Record actual delivered reward too.
4. **Reward/controller phase consistency.** Clock-based phase/clearance penalties
   use the controller's right-foot phase-offset knob rather than assuming exactly
   half a cycle. This does not assert that all physical motion follows that clock.
5. **Required-foot progress independent of clock overlap.** Keep expected-foot
   masking and GAIT-09 turn budgets; remove the additional clock gate from partial
   lift/reach quality. A physically useful step can earn credit even if the learned
   waveform's main swing occurs outside the nominal window. Landing cannot renew
   either budget. Record clock overlap, controller/reward disagreement and budgets.

At 100 Hz, the usual reward scaling is 0.5: full-strength event cost is 10 reward
units per repeat, plus up to 2 per tick of the extra lift. These are bounded costs,
but much larger than GAIT-09. New costs start at 50% and ramp to 100% over two
million eligible steps after the existing two-evaluation walking gate opens.
Existing walking/drag shaping stays at full strength; incomplete-event weight
remains 2. A full-strength parent-policy evaluation must still have positive raw
and delivered mean reward, <=10% base reward-floor hits and >=80% survival before
training can proceed. These checks do not guarantee learning or hardware safety.

## Training and decision criteria

- 15,000,000 additional environment steps, rounded to **15,007,744** full-rollout
  steps. 512 environments × 32 steps, seed 0, learning rate 3e-5 → 1e-5.
- One GPU, 10 CPUs, 90 GB, four-hour allocation. Approximate expected runtime
  1h40–2h plus queue, based on GAIT-09; actual throughput may differ.
- Evaluate the inherited policy and then every 1,048,576 steps; also evaluate the
  final checkpoint. 12 nominal 25 s trials per target 0.20/0.25/0.35 m/s, seed
  3103. The 0.25 m/s trials determine progress; all evaluated checkpoints retained.
- Clean-gait qualification retains GAIT-09 thresholds, including <=10% incomplete
  attempts and no repeated attempts in the post-five-second measurement window.
- **Meaningful progress**, relative to the reevaluated parent: at least 25% fewer
  total repeated attempts AND at least a 10-percentage-point decrease in incomplete
  fraction. Require >=90% survival, tracking error <=0.05 m/s, mean >=8 valid steps
  per foot and mean contact slip <=0.02 m/s. Standing or falling cannot pass.
- Require these conditions at two consecutive evaluations among the final five.
  A keeper-eligible clean candidate anywhere is retained as a success regardless.
- Training writes `continuation_decision.json`: qualified candidate, sustained
  progress requiring video review, or **fresh policy with explicit step objective**.
  This reports the agreed branch; it does not silently submit an undefined fresh
  experiment. No further continuation of this policy is planned if progress fails.
  Inspect candidate/final video before concluding that a metric improvement is
  meaningful. One training seed and repeated nominal trials are not robustness proof.

## Cleanup and validation

- Extracted meaningful-attempt detection and reward-floor handling into named,
  testable functions in `bilateral_gait.py`.
- Moved attempt statistics out of the evaluator into NumPy-only `gait_diagnostics.py`;
  included the explicit continuation decision rule there.
- Added reusable source packaging and a common Slurm validation/training body.
  Older launchers, snapshots, videos and reports remain reproducible and untouched.
- Fixed replay recorder compatibility with old frozen config classes; added
  phase/expected-foot/budget fields for future GAIT-10 replay diagnosis.
- Local focused suite: **58 tests, 52 passed, 6 JAX-dependent checks deferred to
  SCITAS**. Changed Python modules compile. A 600-transition comparison per preset
  confirmed identical state, terms and metrics for GAIT-06/08/09 against the frozen
  GAIT-09 implementation after the bookkeeping refactor.
- SCITAS runs all focused tests, a finite physics/PPO benchmark and the full-strength
  inherited-policy reward/survival preflight before the long continuation.

The frozen source manifest records the exact code used. Completed results are
reported above; the original experiment design remains below that results section.
