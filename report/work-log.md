# Work log and next questions

2026-10-14: set aside the gait-search proposal at the user's request. Cleaned up
report tooling and Git artifact rules before publishing completed STOP-04–06 work.
Downloaded archives, TensorBoard logs and intermediate checkpoint sidecars stay
local. Existing results and frozen SCITAS snapshots are preserved.

2026-10-13: reviewed Nathann's PPO gait generator, older CMA-ES gait library,
archived 12-parameter CEM braking search, and existing BalanceRL stander. The
braking search uses a weaker speed criterion than our sustained-stop benchmark.
Historical proposal, set aside on 14 October. [Code review and evidence](nathann-gait-review.md).
No training or controller changes were made for this review.

2026-10-13: both STOP-06 arms completed 5,046,272 steps overnight with 0/32
moving stops at every evaluation. Final standing holds were 32/32 timed and 30/32
feedback; final walking retention passed only for timed. Source keepers were
retained. Automatic replay clips therefore show the source policies, not the
final trained policies. See [completed outcome](data/stop06_20261012_v1/STATUS.md).

2026-10-12: STOP-05 stopped at 262,144 steps after retention regressions, with no
successful moving stops. User approved [STOP-06](../docs/stop06-experiment-report.md):
matched longer learning with timed versus feedback HOLD, bounded settling and
automatic timing diagnostics. See [status](../docs/experiments/stop06_20261012_v1/STATUS.md).
Array **3216469**: timed arm running and feedback arm queued at 19:50 Zurich;
33 SCITAS tests passed. Results pending.
These new experiments are not represented by the older gallery figures/videos.

| Date / record | Change or decision | Reason / outcome |
|---|---|---|
| 2 October, STOP-01B | STOP fine-tuning from walking bundle | 20M steps without successful benchmark stops |
| 6 October, STOP-02 | Shared smooth speed adapter; five-second evaluation; dense HOLD rewards | Ramp helped survival; standing-only PPO still failed |
| 7 October, STOP-03 submission | Actor imitation followed by gated standing PPO | Supply a feasible stance while retaining walking |
| STOP-03 completion record | Keep 524k hold keeper, not only final weights | 21/32 best holds; final 14/32 despite 31/32 survival |
| 12 October | Create this report folder, copy evidence, generate figures/gallery | Make final report and presentations reproducible |
| 12 October, STOP-04 outcome | Standing keeper at step 262,144 | 29/32, 32/32, 30/32 holds across three evaluation seeds; standing gate passed |
| 12 October, STOP-05, job 3216384 | Low-speed moving-stop pilot with rehearsal and preservation guards | Stopped by retention guard at 262,144 steps; 0/32 moving stops |
| 12 October, STOP-06, array 3216469 | Longer timed versus feedback HOLD comparison | Both completed 5.05M steps; no successful moving stops |

Latest experiment details: [STOP-06 protocol](../docs/stop06-experiment-report.md)
and [completed outcome](data/stop06_20261012_v1/STATUS.md).
The existing figures and videos describe earlier experiments; they are not
STOP-05 results. The report's copied code remains a historical snapshot.

No policy, reward or training changes were made for the report-folder task. Local
metrics are a downloaded snapshot, not a live SCITAS dashboard. The existing source
manifests identify the exact training snapshots; this folder's manifest identifies
the evidence and current code copied into the report.

## Next investigations, not new results

12 October, approved continuation: [STOP-04](../docs/stop04-experiment-report.md)
automates paired checkpoint diagnosis and video recording, then a conditional
one-million-step standing PPO pilot with continued HOLD/RUN imitation. A physical
regression guard and three-seed final assessment are included. This supersedes the
earlier "next investigation" status of items 1-3 below; outcomes remain pending.
Slurm job **3215784** was confirmed queued (Priority) at 11:00:20 Europe/Zurich.

1. Record the STOP-03 best and post-imitation policies, with a declared trial
   selection rule and overlays for the five-second conditions.
2. Trace surviving final-policy failures to identify which conditions interrupt
   quietness, before changing reward weights.
3. Decide how to preserve the learned stance during PPO (for example, continued
   imitation/rehearsal); test this as a separate intervention.
4. Validate small perturbations and additional seeds, then low-speed moving STOP
   transitions. Consider actual-speed/posture conditions for the phase handoff.
5. Evaluate restarting and, later, broader plant/sensing variation.

## How to add an experiment

Record the hypothesis and intended comparison before running. Store snapshot/job
IDs, configuration, source policy/checkpoint hashes, changed files and training
budget. Keep evaluation protocol/seed, scalar reward and physical success separate.
Copy small result files from SCITAS on demand, then add the new experiment to
`scripts/build_report.py` and rebuild this folder.

For a video, record checkpoint, seed, trial, command timing, playback speed and
selection criterion. Prefer a representative or predeclared trial; if selecting
the best one, say so and give the aggregate success rate alongside it. Keep a
failed trial available for comparison. A replay does not replace quantitative
evaluation.

Update `experiments.md`, `evaluation.md` and this log with observations first,
then interpretations and decisions. Add figure captions and source paths; never
overwrite an earlier run's raw evidence to make the narrative cleaner.

## New entry template

Repository housekeeping: added an experiment index and explicit artifact rules.
Raw traces, transfer archives and scratch previews stay local; useful metrics,
curated media and source manifests stay in Git. Existing experiment paths and
training snapshots were preserved.

```text
Date / experiment / Slurm job:
Question and hypothesis:
Source policy and exact checkpoint:
Changes (and what the comparison isolates):
Training budget, seed and environment:
Evaluation protocol and seed:
Measured results / artifact paths:
Interpretation, limitations and rejected explanations:
Decision and next test:
```
