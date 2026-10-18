# GAIT-03 status

Observed 15 October 2026: job **3219917 COMPLETED**, elapsed **40m29s**, exit **0:0**.
Finished **5,013,504 additional steps**. Final gait evaluation: 12/12 survived,
mean speed error 0.03574 m/s, but **0/12 qualified**. Every final trial had ten
valid left steps and zero valid right steps during seconds 5–25. Right-foot
clear time averaged 0.217%, flight 3.42%, and maximum joint-limit overshoot
averaged 0.0284 rad. The bilateral stepping problem remains unresolved.

No evaluation qualified for the gait keeper. The diagnostic best candidate is
the final checkpoint; it is not an accepted walking gait. Downloaded
`gait_evaluations.jsonl`, `latest_gait_eval.json` and `best_candidate_eval.json`.

Budget: 5M additional steps at 0.35 m/s; four-hour allocation cap. Warm-start:
GAIT-02 best. The new keeper requires bilateral gait quality, not survival alone.
This file is a dated observation, not a live scheduler dashboard.

Remote root: `/home/rizkalla/running_robot/experiments/gait03_20261015_v1`.
Log: `RLframework/logs/cad-v2-gait03-3219917.out` under that root.

See the [experiment report](../../gait03-experiment-report.md) for reward changes,
tests, checkpoint provenance and acceptance thresholds, and the
[source manifest](source-manifest.json) for submitted file hashes.
