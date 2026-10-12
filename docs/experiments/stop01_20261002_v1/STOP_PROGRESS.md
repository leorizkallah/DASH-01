# STOP-01 measured progress

Updated: 2026-10-02T16:48:26+02:00

Run: `dash_stop01_s0`

Source: `dash_joy3_lr_s2` at 180,486,144 steps.
Bundle SHA-256: `b095ad572cb80db7704018e3fd3f41cb05c073f2dda103ca54f4927a753beaeb`.
Actor restoration maximum absolute error: 3.49e-07.

Nominal MJX transition benchmark: 0.5/1.0/1.5/2.0 m/s, varied stop times,
8 s STOP, then 3 s restart. Every episode counts, including failed approaches.

| PPO steps | Stop success | Survived STOP | Restart success | Walking retained | Reward-floor fraction |
|---:|---:|---:|---:|:---:|---:|
| 0 | 0.0% | 0.0% | 0.0% | True | 79.7% |

Per-speed results from the latest evaluation:

| Command (m/s) | Successful stops | Survived STOP |
|---:|---:|---:|
| 0.5 | 0/8 | 0/8 |
| 1.0 | 0/8 | 0/8 |
| 1.5 | 0/8 | 0/8 |
| 2.0 | 0/8 | 0/8 |

These are simulation results. Hardware stopping has not been validated.
