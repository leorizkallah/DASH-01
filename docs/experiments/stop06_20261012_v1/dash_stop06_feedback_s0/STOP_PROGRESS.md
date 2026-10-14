# STOP measured progress

Updated: 2026-10-12T23:54:35+02:00

Run: `dash_stop06_feedback_s0`

Continued actor rehearsal is enabled within the PPO objective.
Dataset SHA-256: `994e7daa1e2153053f4aa746572d36aa4be1e4315507c8da3be84913a8f75caa`.
Warm checkpoint SHA-256: `9c074f03f74c1c104bfa719acae97501ad7653b6a0fa44feb6e1c92c2e5b4266`.
Each minibatch adds 128 HOLD and 128 RUN training examples; validation examples are excluded.

Latest PPO row (training reward is not stopping success):

| Metric | Value |
|---|---|
| time/env_steps | 5046272 |
| rollout/ep_len_mean | 811.3258426966293 |
| rollout/reward_mean | 0.8454756140708923 |
| train/approx_kl | 0.0019256237356334888 |
| train/std_mean | 0.11973122507333755 |
| time/sps | 7271.784181706678 |

Nominal MJX benchmark: STOP-02-ramp. Ramped STOP, 4 s settling allowance, 5 s quiet hold; approach speeds listed below.

| Stage | PPO steps | Stop success | Survived STOP | Restart success | Walking retained | Reward-floor fraction |
|---|---:|---:|---:|---:|:---:|---:|
| source | 0 | 0.0% | 0.0% | 0.0% | True | 19.6% |
| ppo | 262,144 | 0.0% | 0.0% | 0.0% | True | 15.8% |
| ppo | 524,288 | 0.0% | 0.0% | 0.0% | True | 24.7% |
| ppo | 786,432 | 0.0% | 0.0% | 0.0% | False | 15.2% |
| ppo | 1,048,576 | 0.0% | 0.0% | 0.0% | True | 18.2% |
| ppo | 1,310,720 | 0.0% | 0.0% | 0.0% | True | 15.5% |
| ppo | 1,572,864 | 0.0% | 0.0% | 0.0% | True | 16.5% |
| ppo | 1,835,008 | 0.0% | 0.0% | 0.0% | True | 21.2% |
| ppo | 2,097,152 | 0.0% | 0.0% | 0.0% | False | 18.3% |
| ppo | 2,359,296 | 0.0% | 0.0% | 0.0% | True | 23.3% |
| ppo | 2,621,440 | 0.0% | 0.0% | 0.0% | True | 26.1% |
| ppo | 2,883,584 | 0.0% | 0.0% | 0.0% | True | 22.7% |
| ppo | 3,145,728 | 0.0% | 0.0% | 0.0% | False | 29.8% |
| ppo | 3,407,872 | 0.0% | 0.0% | 0.0% | False | 33.0% |
| ppo | 3,670,016 | 0.0% | 0.0% | 0.0% | True | 31.2% |
| ppo | 3,932,160 | 0.0% | 0.0% | 0.0% | True | 28.8% |
| ppo | 4,194,304 | 0.0% | 0.0% | 0.0% | False | 30.4% |
| ppo | 4,456,448 | 0.0% | 0.0% | 0.0% | False | 23.5% |
| ppo | 4,718,592 | 0.0% | 0.0% | 0.0% | True | 23.1% |
| ppo | 4,980,736 | 0.0% | 0.0% | 0.0% | False | 25.8% |
| ppo | 5,046,272 | 0.0% | 0.0% | 0.0% | False | 23.8% |

Per-speed results from the latest evaluation:

| Command (m/s) | Successful stops | Survived STOP |
|---:|---:|---:|
| 0.25 | 0/11 | 0/11 |
| 0.375 | 0/11 | 0/11 |
| 0.5 | 0/10 | 0/10 |

Mean longest quiet hold: 0.002 s.

Separate near-stance hold success: 93.8%.
Standing retained: True.
Moving approach qualification: 50.0%.

These are simulation results. Hardware stopping has not been validated.
