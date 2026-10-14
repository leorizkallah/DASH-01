# STOP measured progress

Updated: 2026-10-12T21:51:59+02:00

Run: `dash_stop06_timed_s0`

Continued actor rehearsal is enabled within the PPO objective.
Dataset SHA-256: `994e7daa1e2153053f4aa746572d36aa4be1e4315507c8da3be84913a8f75caa`.
Warm checkpoint SHA-256: `9c074f03f74c1c104bfa719acae97501ad7653b6a0fa44feb6e1c92c2e5b4266`.
Each minibatch adds 128 HOLD and 128 RUN training examples; validation examples are excluded.

Latest PPO row (training reward is not stopping success):

| Metric | Value |
|---|---|
| time/env_steps | 5046272 |
| rollout/ep_len_mean | 533.6222222222223 |
| rollout/reward_mean | 1.1755812168121338 |
| train/approx_kl | 0.001642371887066414 |
| train/std_mean | 0.11978496611118317 |
| time/sps | 7257.664988618095 |

Nominal MJX benchmark: STOP-02-ramp. Ramped STOP, 4 s settling allowance, 5 s quiet hold; approach speeds listed below.

| Stage | PPO steps | Stop success | Survived STOP | Restart success | Walking retained | Reward-floor fraction |
|---|---:|---:|---:|---:|:---:|---:|
| source | 0 | 0.0% | 0.0% | 0.0% | True | 13.5% |
| ppo | 262,144 | 0.0% | 0.0% | 0.0% | True | 10.1% |
| ppo | 524,288 | 0.0% | 0.0% | 0.0% | True | 15.3% |
| ppo | 786,432 | 0.0% | 0.0% | 0.0% | False | 11.9% |
| ppo | 1,048,576 | 0.0% | 0.0% | 0.0% | True | 16.2% |
| ppo | 1,310,720 | 0.0% | 0.0% | 0.0% | True | 13.7% |
| ppo | 1,572,864 | 0.0% | 0.0% | 0.0% | True | 16.8% |
| ppo | 1,835,008 | 0.0% | 0.0% | 0.0% | True | 15.3% |
| ppo | 2,097,152 | 0.0% | 0.0% | 0.0% | True | 16.0% |
| ppo | 2,359,296 | 0.0% | 0.0% | 0.0% | True | 17.3% |
| ppo | 2,621,440 | 0.0% | 0.0% | 0.0% | False | 11.6% |
| ppo | 2,883,584 | 0.0% | 0.0% | 0.0% | True | 14.2% |
| ppo | 3,145,728 | 0.0% | 0.0% | 0.0% | True | 12.0% |
| ppo | 3,407,872 | 0.0% | 0.0% | 0.0% | True | 14.2% |
| ppo | 3,670,016 | 0.0% | 0.0% | 0.0% | False | 15.8% |
| ppo | 3,932,160 | 0.0% | 0.0% | 0.0% | True | 17.2% |
| ppo | 4,194,304 | 0.0% | 0.0% | 0.0% | True | 16.5% |
| ppo | 4,456,448 | 0.0% | 0.0% | 0.0% | True | 12.6% |
| ppo | 4,718,592 | 0.0% | 0.0% | 0.0% | True | 17.7% |
| ppo | 4,980,736 | 0.0% | 0.0% | 0.0% | True | 17.4% |
| ppo | 5,046,272 | 0.0% | 0.0% | 0.0% | True | 16.9% |

Per-speed results from the latest evaluation:

| Command (m/s) | Successful stops | Survived STOP |
|---:|---:|---:|
| 0.25 | 0/11 | 0/11 |
| 0.375 | 0/11 | 0/11 |
| 0.5 | 0/10 | 0/10 |

Mean longest quiet hold: 0.000 s.

Separate near-stance hold success: 100.0%.
Standing retained: True.
Moving approach qualification: 65.6%.

These are simulation results. Hardware stopping has not been validated.
