# WALK-01 status

**COMPLETED: job 3224989, exit 0:0, 200,005,632 steps in 13h29m16s.**
Stopped at the requested step budget, not the time ceiling. SCITAS timestamps:
start `2026-10-20T16:40:05+02:00`, end `2026-10-21T06:09:21+02:00`.
The snapshot name retains the preparation date `20261019`.

The best checkpoint is **169,574,400 steps**: 28/28 survived the 30-second
seven-speed ladder, mean absolute speed error **0.181 m/s**. Errors at targets
0.25 / 0.5 / 0.75 / 1.0 m/s are 0.071 / 0.061 / 0.068 / 0.092 m/s.
The last periodic evaluation at 199,065,600 steps also survived 28/28, but mean
error rose to **0.328 m/s**. Final separate command-redraw evaluation: no falls
and 30-second elapsed time in all 28 trials. Its `alive` field becomes false at
episode timeout; do not interpret that field alone as failure.

**[Six MJX replay videos](../../../report/walk01-videos.html)** compare best/final
at 0.25, 0.75 and 1.5 m/s. All six complete 25 seconds without falling.
Replay job **3226290 completed**, exit 0:0, in 4m27s. The gallery includes
opposite camera views and slow motion. Each clip is one nominal fixed-command
trial, seed 1000, using the frozen training simulator. No training was launched.

| MJX replay | 0.25 m/s requested | 0.75 m/s requested | 1.50 m/s requested |
|---|---:|---:|---:|
| Best: achieved after 5 s | 0.314 | 0.714 | 1.241 |
| Final: achieved after 5 s | 0.388 | 0.724 | 1.359 |

Earlier labelled CPU deployment previews are retained separately in the gallery.

Training cadence reached the 4 Hz upper bound. Survival and tracking do not
certify clearance, symmetry or absence of sliding; inspect the videos.

- Preset `cad_v2_walk01`, run `cad_v2_walk01_s0`, fresh seed 0.
- Target: **200,000,000 steps**, with a **24-hour allocation ceiling** and
  ten-minute final-save reserve. The clock starts when the allocation begins.
- Nathann walking reward/curriculum, STOP and standing tasks disabled.
- Calibrated CAD-v2 with supplied homing reset; no imported actor weights.
- One GPU, 10 CPUs, 90 GB RAM; 512 environments × 36 ticks.
- Source archive: 84 files, SHA256
  `0deeab9de48592ba2f3006df9bfb5b15d9023d758038553493e37a6e29df8618`.
- Remote directory: `/home/rizkalla/running_robot/experiments/walk01_20261019_v1`.

Local validation: six configuration/model/budget tests passed; Python compilation
passed. The supplied homing pose fits the existing calibrated geometry and
passes a five-second unassisted nominal PD hold. **GPU checks passed**: finite
PPO benchmark, CPU/MJX coordinate error 7.82e-6 and velocity error 0.00479,
finite random-action probe, no numerical terminations and zero STOP ticks.

The keeper uses the original joystick score and does not certify clean walking.
Retained checkpoints permit video reviews near 20M, 60M, 120M and 180M steps.
No second job or automatic training continuation has been scheduled.

```powershell
ssh -i C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519 -o IdentitiesOnly=yes rizkalla@izar.hpc.epfl.ch 'squeue -j 3224989'
```

[Experiment contract](../../walk01-experiment-report.md) ·
[Homing audit](../../../report/data/homing01/audit.json) ·
[Pose preview](../../../report/data/homing01/pose.png)
