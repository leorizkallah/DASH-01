# STOP measured progress

Updated: 2026-10-08T00:56:42+02:00

Run: `dash_stop03_imitation_s0`

Source: `dash_stop01b_s0` at 20,054,016 steps.
Bundle SHA-256: `ec047be7388993f1c0ce6164a514a0272808688e1c26602f4bd8cc30ef848a88`.
Actor restoration maximum absolute error: 4.2e-07.

Standing imitation stage: **accepted_for_ppo**.
Imitation updates and demonstration steps are separate from PPO steps.

Held-out HOLD action MSE: 0.192867 -> 0.000055.
Held-out RUN action MSE: 0.000031.

- walking_retained: True
- holding_improved: True
- hold_fit_improved: True
- walking_fit_bounded: True

Latest PPO row (training reward is not stopping success):

| Metric | Value |
|---|---|
| time/env_steps | 2031616 |
| rollout/ep_len_mean | 606.2745098039215 |
| rollout/reward_mean | 2.0131757259368896 |
| train/approx_kl | 0.00309974464799589 |
| train/std_mean | 0.1198263168334961 |
| time/sps | 7892.892760159701 |

Nominal MJX benchmark: STOP-02-standing. Near-stance starts; 7 s HOLD, including at least 5 s quiet with drift <=0.10 m.

| Stage | PPO steps | Stop success | Survived STOP | Restart success | Walking retained | Reward-floor fraction |
|---|---:|---:|---:|---:|:---:|---:|
| source | 0 | 0.0% | 0.0% | n/a | True | 11.7% |
| post_imitation | 0 | 65.6% | 68.8% | n/a | True | 1.0% |
| ppo | 524,288 | 65.6% | 68.8% | n/a | True | 1.0% |
| ppo | 1,048,576 | 43.8% | 93.8% | n/a | True | 0.2% |
| ppo | 1,572,864 | 43.8% | 93.8% | n/a | True | 0.2% |
| ppo | 2,031,616 | 43.8% | 96.9% | n/a | True | 0.1% |

Per-speed results from the latest evaluation:

| Command (m/s) | Successful stops | Survived STOP |
|---:|---:|---:|
| 0.0 | 14/32 | 31/32 |

Mean longest quiet hold: 3.929 s.

These are simulation results. Hardware stopping has not been validated.
