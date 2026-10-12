# STOP-02 standing pilot

Submitted 2026-10-06: Slurm job **3204982**, run `dash_stop02_stand_s0`.
Budget: 2M PPO steps, one GPU, two-hour wall-clock limit.
Checked 2026-10-07: **COMPLETED**, exit code 0:0, node i04.
Ran 2026-10-06 18:20:15–18:44:47 Europe/Zurich (24 min 32 s).
Reached **2,031,616 PPO steps**; final checkpoint and final evaluation exist.

Outcome: **0/32 successful five-second holds** at all four post-update evaluations.
At the final checkpoint all 32 trials terminated for excessive tilt, on average
2.054 s after the near-standing start. Mean longest quiet period was 0.0328 s;
the longest in any final trial was 0.06 s. The standing objective remains unmet.

Final RUN-only survival fell from 67.26% to 59.52%, failing the walking-retention
gate despite tracking error improving from 0.4143 to 0.3498 m/s.
No subsequent speed stage was launched. `best_stop.msgpack` remains step zero.

Validation job 3204978 completed successfully. Sixteen local tests pass. The
selected warm start has 0.0383 m/s approach velocity-estimator error. Its untrained
standing baseline is 0/32; a nominal fixed-stance feasibility probe passes 31/32.

Automatic cluster reports: `RLframework/runs/dash_stop02_stand_s0/STOP_PROGRESS.md`.
See [experiment log](../../stop02-experiment-report.md) for findings and the command
to refresh local metrics. Later speed stages are not scheduled yet.
