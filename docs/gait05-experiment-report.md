# GAIT-05: fresh policy at a slower walking target

15 October 2026. User approved fresh training at 0.25 m/s after GAIT-04 retained
substantial dragging. Snapshot: `gait05_20261015_v1`, job **3220364**. See
[status](experiments/gait05_20261015_v1/STATUS.md) for the submitted job.

## Question and comparison

Can a fresh policy learn bilateral stepping under the current reward, at a
moderately slower speed, without inheriting the previous asymmetric strategy?
GAIT-04 made late progress (5–8 right steps per trial at 14.68M) but never met
the gait criteria and regressed at its final checkpoint. This run changes both
initialization and training speed; it cannot isolate their individual effects.
A slower target is a hypothesis, not an established improvement.

## Training setup

- Preset `cad_v2_gait05`, run `cad_v2_gait05_s0`, seed 0, **15M environment steps**.
- Fixed training target **0.25 m/s**, encoded as command 0.5 with `v_max=0.5`.
- Fresh network weights, optimizer, observation statistics and action exploration.
  **No warm-start, resume, source checkpoint or imitation bundle.** The launcher
  rejects an existing run directory to prevent accidental reuse.
- **GAIT-04 rewards and physical model unchanged.** The tested preset differences
  are only command ranges, evaluation targets, candidate ranking and retention.
- Same 512 environments, 32-step rollouts, 100 Hz control and 1 kHz physics.
  Same shaping ramp: 0.5 to 1 over 3M steps. Same learning rate 5e-5 to 1e-5,
  100-update warmup and entropy settings. Fresh initialization provides the
  configured initial exploration instead of GAIT-04's learned action variance.
- One GPU, 10 CPUs, 90 GB RAM, four-hour cap. GAIT-04 took 1h43m; roughly two
  hours is a planning estimate, with extra evaluation targets and fresh-policy
  behavior potentially changing runtime. Queue time is additional.

## Step-size sanity check

The valid-step requirement remains 4 cm forward foot travel relative to the
torso, at least 80 ms airborne and 1.5 cm whole-sole clearance. At a nominal
0.25 m/s, 65% stance fraction and the fastest configured 2.5 Hz cycle, ideal
no-slip stance displacement is `0.25 * 0.65 / 2.5 = 0.065 m`. A periodic foot
trajectory would recover that body-relative distance during swing. This makes
4 cm a reasonable initial threshold; it is not a kinematic feasibility proof.
At 0.20 m/s the same estimate is 5.2 cm. Thresholds are not weakened in this run.

## Evaluation, selection and retention

- Baseline at step zero, then every 64 rollouts (**1,048,576 steps**), and final.
- Gait evaluation has **12 nominal greedy trials per target** at **0.20, 0.25,
  and 0.35 m/s** (36 total), 25 s each, seed 3103. Commands are fixed for the
  first episode and reset observation features are updated to match each target.
- Top-level metrics and gait keeper eligibility refer to **0.25 m/s** only;
  `per_target` retains all three results. Off-target trials measure transfer and
  cannot compensate for failing at the training speed. The separate regular
  tracking evaluation also covers the three commands with 36 environments.
- The full GAIT-04 acceptance criteria remain unchanged: at least 11/12 primary
  trials must pass. Metrics after 5 s are measured only until first termination;
  an early fall with no measurement window gets its target speed as fallback
  tracking error, rather than a hard-coded 0.35 m/s.
- **Every gait-evaluated policy is saved** as `eval_<step>.msgpack` with
  `eval_<step>_metrics.json`, including baseline and final. These are diagnostic
  snapshots regardless of qualification; periodic checkpoints also remain.
- Diagnostic candidate rank is lexicographic: qualified fraction; surviving
  bilateral progress; survival fraction; negative speed error. Bilateral progress
  averages `survived * min(weaker-foot steps / 3, 1) *
  min(weaker-foot clear fraction / 0.08, 1) * min(alternation / 0.9, 1)` across
  primary trials. A fallen trial contributes zero progress. This prevents a
  no-right-step policy winning merely on tracking, but does not certify gait
  quality; flight, symmetry, slip and joint limits still control acceptance.

Rank changes are enabled only for GAIT-05. Previous frozen runs and their results
are preserved. `best_candidate.msgpack` remains distinct from a qualified keeper.

## Validation and provenance

Local suite: **60 tests, 57 passed, three JAX-dependent tests skipped**. Added
checks verify ranking of bilateral progress versus dragging, exclusion of fallen
progress, precedence of qualification, exact evaluated-checkpoint retention,
and preservation of the GAIT-04 rewards/model/budget in the new preset.
Python compilation checks passed. The SCITAS job reruns 15 focused tests and a
512-environment GPU/PPO benchmark before training and the baseline evaluation.
At the initial check, job 3220364 was RUNNING on i55 (46 s elapsed): all 38 source
hashes verified and all 15 focused tests passed remotely. GPU initialization was
underway; completed GPU validation and training steps were not yet confirmed.

Changed code: `config.py`, `gait_quality_eval.py`; added `gait_selection.py`,
`tests/test_gait_selection.py`, `slurm/izar_cad_v2_gait05.sbatch`.
The [source manifest](experiments/gait05_20261015_v1/source-manifest.json) records
the frozen executable inputs. The reward implementation `bilateral_gait.py` is
byte-identical to the GAIT-04 snapshot.

After completion, inspect right-foot stepping and clearance over training,
compare all three speed targets, and replay a retained policy with bilateral
progress. Longer training is not assumed to guarantee improvement.

## Completed result ? checked 16 October 2026

Job **3220364** completed in **1h24m56s**, exit **0:0**, reaching **15,007,744**
steps. This is successful execution of an unsuccessful walking-learning run.

All **16 evaluations**, including the random-policy baseline, had **zero survivors
and zero qualified trials out of 12 at every speed**: 0.20, 0.25 and 0.35 m/s.
The final evaluation therefore had 0/36 survivors. No valid steps were recorded
in the post-5-second measurement windows. Early falls leave these windows empty;
these zeros do not establish that no foot moved or stepped before termination.
The final logged mean training episode length was 217.86 ticks, approximately
**2.18 s** at 100 Hz. This is a training-rollout mean, not a measured evaluation
fall time. Final evaluation numerical-finiteness checks passed.

The reported 0.20/0.25/0.35 m/s evaluation errors are target-speed fallback values
for missing post-warmup data, not measured tracking errors. The diagnostic rank
was tied at `[0, 0, 0, -0.25]` for the primary target and kept the step-zero
candidate. It is not a learned gait. The final evaluation record points to
`eval_15007744.msgpack`; evaluated-policy retention remains available for diagnosis.

The experiment does not support extending this exact setup without diagnosis.
It also does not isolate whether initialization, speed or reward/schedule
suitability caused the failure. No new reward changes or continuation were
submitted during the status check.

Evidence: [evaluation history](experiments/gait05_20261015_v1/gait_evaluations.jsonl),
[final evaluation](experiments/gait05_20261015_v1/latest_gait_eval.json),
[candidate record](experiments/gait05_20261015_v1/best_candidate_eval.json),
[training progress](experiments/gait05_20261015_v1/progress.csv),
[tracking evaluations](experiments/gait05_20261015_v1/eval.csv).

Follow-up: [early-fall diagnosis](gait05-early-fall-diagnosis.md), with first-fall
reasons, paired policy controls, learning-setup findings and replay videos.
