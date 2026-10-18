# GAIT-01 v2 status — 14 October 2026

SCITAS job **3219542: COMPLETED**, exit 0, in **27 min 19 s**.
Final progress: **5,013,504 steps**, rounded to a complete rollout.
Final checkpoint and metadata saved successfully. No sustained walking demonstrated.
The [GAIT-02 continuation](../gait02_20261014_v1/STATUS.md) has been submitted.

The CPU/MJX comparison and finite-PPO benchmark passed. Maximum coordinate
difference was 5.13e-6 and velocity difference 0.00412. Fixed-target standing
still fails, as already documented; it is not a walking prerequisite.

Learning has not yet produced a successful gait. At the latest evaluation
(4,194,304 steps), **all 96 nominal trials fell**, with 0% full-episode survival.
Final training episodes average roughly **2.20 s**, versus 0.23 s initially.
Final sampled training flight fraction is about 6.3%; walking-only behavior is
not yet demonstrated. The final checkpoint will also be evaluated during
GAIT-02 checkpoint selection; it was saved after the last periodic evaluation.

Fresh walking policy, 0.2–0.5 m/s, 5M steps, 512 environments, seed 0.
One GPU, 10 CPUs, 90 GB RAM; four-hour allocation cap.
No separate standing/balance phase. The user will provide a homing pose later.

The job checked CPU/MJX numerical agreement under small pitch disturbances,
then a finite PPO benchmark, before starting training.
The original fixed-target standing failure remains recorded separately.

Remote root: `/home/rizkalla/running_robot/experiments/gait01_20261014_v2`.
Source bundle: 35 files, 2.85 MB; all 34 payload hashes verified before submission.
Downloaded evidence: `gpu-standing.json`, `benchmark.json`, `progress.csv`, `eval.csv`.

See [experiment report](../../gait01-experiment-report.md) and
[source hashes](source-manifest.json).
