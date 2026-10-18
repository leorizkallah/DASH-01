# GAIT-04 status

Completed: SCITAS job **3220118**, **1h43m02s**, exit **0:0**.
Finished **15,007,744 additional steps**. All 16 gait evaluations had **0/12**
qualifying trials, but repeated right-foot stepping emerged late in training.

At 14,680,064 steps: 12/12 survived, 0.03165 m/s mean speed error, ten valid
left steps and 5?8 right steps per trial (mean 6.58) during seconds 5?25.
Right clear time averaged 5.08%, alternation 75.8%, flight 2.85%.
Final weights regressed: 8/12 survived; surviving trials had 2?6 right steps
(mean 3.125). Final aggregate speed error was 0.0710 m/s.

The diagnostic candidate selector chose 8,388,608 steps, which still had zero
right steps: when all qualification fractions are zero, its score only separates
survival and tracking. It must not be interpreted as the best bilateral gait.
The 14.68M evaluation is promising evidence, not an accepted gait or confirmation
that an exact matching checkpoint was retained.

Downloaded `gait_evaluations.jsonl`, `latest_gait_eval.json` and
`best_candidate_eval.json`. No new training or replay launched during this check.

Local validation: 55 tests, 52 passed, three JAX tests skipped. All 36 submitted
source hashes verified on SCITAS before submission. The job checks the source
checkpoint hash, reruns ten bilateral tests, and runs a GPU/PPO benchmark.

Remote root: `/home/rizkalla/running_robot/experiments/gait04_20261015_v1`.
Log: `RLframework/logs/cad-v2-gait04-3220118.out`.
Run: `RLframework/runs/cad_v2_gait04_s0`.
This is a dated record, not a live scheduler dashboard.

See the [experiment report](../../gait04-experiment-report.md) and
[source manifest](source-manifest.json). Best candidates are diagnostic unless
they pass the unchanged gait-quality criteria.

Replay job 3220346 completed in 3m37s. [Videos](../../../report/gait04-videos.html)
show saved 14.22M and final policies at 0.35 m/s. Both fresh seed-1000 trials
survived 25 s; valid L/R steps after warmup: 10/5 and 10/3. Exact 14.68M weights
were not retained. These replays do not override the multi-trial evaluations.
