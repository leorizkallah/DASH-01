# SPEED-01 status

Job **3223896** started on SCITAS node `i28` at **19 October 2026,
18:47:24 Europe/Zurich**. Allocation deadline: **23:47:24** the same day.
**COMPLETED**, exit code `0:0`, at **19 October 2026, 23:39:38 Europe/Zurich**.
Elapsed allocation: **4h52m14s**. Training reached **59,916,288 steps** and stopped
normally at the walltime reserve; final saving/evaluation completed before the
allocation deadline. Selected and final checkpoints are present on SCITAS.

| Nominal 30-second evaluation | Best eligible checkpoint | Final checkpoint |
|---|---:|---:|
| Training steps | 37,748,736 | 59,916,288 |
| Survived | 16/16 | 16/16 |
| Passed speed/joint/thermal criteria | 16/16 | 0/16 |
| Mean world-X speed, including startup | 0.950 m/s | 1.215 m/s |
| Mean speed after 5 seconds | 0.961 m/s | 1.235 m/s |
| Mean contact-slip metric, left/right | 0.318 / 0.124 m/s | 0.351 / 0.170 m/s |
| Whole-foot clearance >=15 mm, left/right | 0% / 0.115% | 0% / 0.375% |

The final policy exceeds the configured motor-joint position bounds by peaks of
0.0185–0.0203 rad across trials, above the 0.01 rad selection tolerance.
Thermal limits and survival pass. Thirty evaluations were recorded; the selected
`best_speed.msgpack` passes the feasibility checks but is **not a clean-walking
certificate**. The slip and clearance measurements suggest predominantly
sliding/shuffling motion; videos are needed to inspect the actual gait.
No new training was launched during this status check.

Downloaded evidence: [selected evaluation](best_speed_eval.json),
[final evaluation](latest_speed_eval.json), [evaluation history](speed_evaluations.jsonl),
[training progress](progress.csv), and [walltime stop](walltime_stop.json).

- Preset: `cad_v2_speed01`; run: `cad_v2_speed01_s0`; seed 0.
- Fresh policy, optimizer and observation statistics; no parent checkpoint.
- Five-hour allocation, no environment-step cap; ten-minute saving/evaluation reserve.
- 512 environments; one GPU, 10 CPUs, 90 GB RAM.
- Branch: `dash-01-v2-speed-discovery`; previous campaign preserved at `c035774`.
- Remote directory: `/home/rizkalla/running_robot/experiments/speed01_20261019_v1`.
- Frozen archive: 57 files, SHA256
  `2ddf0d754ed8956dd6454f02dd522cd44b416ac5b23aa9bca4a60beadef8e753`.

The launcher verifies source hashes, runs regression tests, checks finite GPU PPO
updates and measures the reward budget before starting training. The archived
GAIT-09 checkpoint is used only by a separate diagnostic process; it does not
initialize the new policy. Local tests: 66 passed, 7 skipped because local JAX is
unavailable. All 67 scheduled GPU regression tests passed, with finite PPO
updates. Benchmark: 4,541 simulator steps/s and 994 PPO steps/s (512 environments).
See [benchmark](benchmark.json) and [reward probes](reward-budget.json).

All four reward probes were finite. The sampled untrained policy lasted 0.709 s
on average; the zero-action controller lasted 2.083 s. The archived GAIT-09
comparison survived all six seconds at 0.246 m/s mean world-X speed, with positive
nonterminal reward (0.117/control tick) and no reward-floor hits under full
regularization. This validates a moving reference's reward budget, not the new
policy's performance or its eventual convergence.

Check with `ssh rizkalla@izar.hpc.epfl.ch 'squeue -j 3223896'` using your configured
SCITAS authentication. [Experiment and reward contract](../../speed01-experiment-report.md).
