# GAIT-07 status

SCITAS job **3222141 COMPLETED**, exit 0, in **1h47m16s**.
Finished **15,007,744 additional steps**. Downloaded 16 evaluations and training CSV.
Final policy is also the best saved diagnostic candidate.

| Final target | Survived 25 s | Qualified gait | Mean valid L/R steps after 5 s |
|---|---|---|---|
| 0.20 m/s | 12/12 | 0/12 | 0.00 / 8.17 |
| 0.25 m/s | 12/12 | 0/12 | 0.25 / 8.50 |
| 0.35 m/s | 11/12 | 0/12 | 0.42 / 6.17 |

Drag reached full strength by the 6.29M evaluation; other shaping stayed at 0.15.
At 0.25 m/s, right-foot slip decreased, while left-foot slip increased slightly.
Stepping remains asymmetric. Video replay job **3222197 COMPLETED** in 3m13s.
The 0.25 m/s replay survives 25 s with 0 left / 11 right valid steps after 5 s.
See the [video comparison](../../../report/gait07-videos.html), including all three
targets and GAIT-06. The 0.20 m/s replay falls at 19.25 s.

- Warm-start GAIT-06 final, 15M additional steps at 0.25 m/s.
- Independent movement-gated contact-drag ramp; bilateral rewards retained.
- Local suite: 69 passed, five JAX-dependent skips. Remote focused suite: all
  29 passed, including JAX tests. The GPU benchmark precedes training.
- [Experiment details](../../gait07-experiment-report.md),
  [source manifest](source-manifest.json), [archive hash](bundle.json).

Remote root: `/home/rizkalla/running_robot/experiments/gait07_20261017_v1`

Log: `RLframework/logs/cad-v2-gait07-3222141.out`

Results: `RLframework/runs/cad_v2_gait07_s0/`

Check with: `ssh -i C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519 -o IdentitiesOnly=yes rizkalla@izar.hpc.epfl.ch 'squeue -j 3222141'`
