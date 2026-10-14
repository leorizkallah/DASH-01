# STOP measured progress

Updated: 2026-10-12T11:38:43+02:00

Run: `dash_stop04_rehearsal_s0`

Continued actor rehearsal is enabled within the PPO objective.
Dataset SHA-256: `994e7daa1e2153053f4aa746572d36aa4be1e4315507c8da3be84913a8f75caa`.
Warm checkpoint SHA-256: `297516343e16659c78b8b586e1bd4037a4f020e94e8ddfee5b8a0175f87df5dd`.
Each minibatch adds 128 HOLD and 128 RUN training examples; validation examples are excluded.

Three-seed low-speed review gate: **True**.
See MULTISEED.md. No next training stage is launched automatically.

Latest PPO row (training reward is not stopping success):

| Metric | Value |
|---|---|
| time/env_steps | 1048576 |
| rollout/ep_len_mean | 326.61744966442956 |
| rollout/reward_mean | 2.0985395908355713 |
| train/approx_kl | 0.002700063570203391 |
| train/std_mean | 0.11982391029596329 |
| time/sps | 7235.3157202802695 |

Nominal MJX benchmark: STOP-02-standing. Near-stance starts; 7 s HOLD, including at least 5 s quiet with drift <=0.10 m.

| Stage | PPO steps | Stop success | Survived STOP | Restart success | Walking retained | Reward-floor fraction |
|---|---:|---:|---:|---:|:---:|---:|
| source | 0 | 68.8% | 68.8% | n/a | True | 1.0% |
| ppo | 262,144 | 93.8% | 96.9% | n/a | True | 0.1% |
| ppo | 524,288 | 93.8% | 93.8% | n/a | True | 0.3% |
| ppo | 786,432 | 93.8% | 93.8% | n/a | True | 0.2% |
| ppo | 1,048,576 | 81.2% | 81.2% | n/a | True | 0.5% |
| ppo | 1,048,576 | 93.8% | 96.9% | n/a | False | 0.1% |

Per-speed results from the latest evaluation:

| Command (m/s) | Successful stops | Survived STOP |
|---:|---:|---:|
| 0.0 | 30/32 | 31/32 |

Mean longest quiet hold: 6.114 s.

These are simulation results. Hardware stopping has not been validated.
