# STOP-01 measured progress

Updated: 2026-10-05T15:00:10+02:00

Run: `dash_stop01b_s0`

Status checked 2026-10-05: Slurm job **3200768 completed normally** on
2026-10-02 at 18:16:19 Europe/Zurich, after 1 h 22 min 28 s (exit code 0:0).
It reached **20,054,016 PPO steps**. All nine recorded post-update stopping
evaluations scored **0/32 successful stops** and **0/32 surviving STOP**.
The last evaluation during training was at 18,874,368 steps. A separate recorded
evaluation of the final 20,054,016-step checkpoint on 2026-10-05 (job 3203602)
also scored 0/32 successful stops and 0/32 surviving STOP. Two trials terminated
before STOP; the other 30 terminated within 0.25–1.88 s after STOP (median 1.04 s).
Excessive tilt was the termination reason in 24/32 trials. The stopping objective
remains unmet. See the [motion replay and diagnosis](../../stop-training-report.md#final-checkpoint-motion-replay-and-diagnosis--2026-10-05).

Source: `dash_joy3_lr_s2` at 180,486,144 steps.
Bundle SHA-256: `b095ad572cb80db7704018e3fd3f41cb05c073f2dda103ca54f4927a753beaeb`.
Actor restoration maximum absolute error: 3.49e-07.

Latest PPO row (training reward is not stopping success):

| Metric | Value |
|---|---|
| rollout/ep_len_mean | 526.938596491228 |
| rollout/reward_mean | 1.9177075624465942 |
| train/approx_kl | 0.0025200940205767214 |
| train/std_mean | 0.11989908665418625 |
| time/sps | 7343.879311529583 |

Nominal MJX transition benchmark: 0.5/1.0/1.5/2.0 m/s, varied stop times,
8 s STOP, then 3 s restart. Every episode counts, including failed approaches.

| PPO steps | Stop success | Survived STOP | Restart success | Walking retained | Reward-floor fraction |
|---:|---:|---:|---:|:---:|---:|
| 0 | 0.0% | 0.0% | 0.0% | True | 25.3% |
| 2,097,152 | 0.0% | 0.0% | 0.0% | True | 23.1% |
| 4,194,304 | 0.0% | 0.0% | 0.0% | True | 27.3% |
| 6,291,456 | 0.0% | 0.0% | 0.0% | False | 29.9% |
| 8,388,608 | 0.0% | 0.0% | 0.0% | True | 35.7% |
| 10,485,760 | 0.0% | 0.0% | 0.0% | True | 27.9% |
| 12,582,912 | 0.0% | 0.0% | 0.0% | False | 24.9% |
| 14,680,064 | 0.0% | 0.0% | 0.0% | True | 19.8% |
| 16,777,216 | 0.0% | 0.0% | 0.0% | True | 24.8% |
| 18,874,368 | 0.0% | 0.0% | 0.0% | True | 25.0% |

Per-speed results from the latest evaluation:

| Command (m/s) | Successful stops | Survived STOP |
|---:|---:|---:|
| 0.5 | 0/8 | 0/8 |
| 1.0 | 0/8 | 0/8 |
| 1.5 | 0/8 | 0/8 |
| 2.0 | 0/8 | 0/8 |

These are simulation results. Hardware stopping has not been validated.

Latest recorded PPO environment steps: 20054016.
