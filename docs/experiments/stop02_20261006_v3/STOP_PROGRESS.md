# STOP measured progress

Updated: 2026-10-06T18:44:47+02:00

Run: `dash_stop02_stand_s0`

Source: `dash_stop01b_s0` at 20,054,016 steps.
Bundle SHA-256: `ec047be7388993f1c0ce6164a514a0272808688e1c26602f4bd8cc30ef848a88`.
Actor restoration maximum absolute error: 4.2e-07.

Latest PPO row (training reward is not stopping success):

| Metric | Value |
|---|---|
| time/env_steps | 2031616 |
| rollout/ep_len_mean | 532.5090909090909 |
| rollout/reward_mean | 1.7472059726715088 |
| train/approx_kl | 0.0034114337799323927 |
| train/std_mean | 0.11983954906463623 |
| time/sps | 7306.843315556438 |

Nominal MJX benchmark: STOP-02-standing. Near-stance starts; 7 s HOLD, including at least 5 s quiet with drift <=0.10 m.

| PPO steps | Stop success | Survived STOP | Restart success | Walking retained | Reward-floor fraction |
|---:|---:|---:|---:|:---:|---:|
| 0 | 0.0% | 0.0% | n/a | True | 11.3% |
| 524,288 | 0.0% | 0.0% | n/a | True | 10.3% |
| 1,048,576 | 0.0% | 0.0% | n/a | True | 10.2% |
| 1,572,864 | 0.0% | 0.0% | n/a | True | 10.5% |
| 2,031,616 | 0.0% | 0.0% | n/a | False | 10.7% |

Per-speed results from the latest evaluation:

| Command (m/s) | Successful stops | Survived STOP |
|---:|---:|---:|
| 0.0 | 0/32 | 0/32 |

Mean longest quiet hold: 0.033 s.

These are simulation results. Hardware stopping has not been validated.
