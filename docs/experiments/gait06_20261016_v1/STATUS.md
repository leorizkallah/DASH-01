# GAIT-06 status

SCITAS job **3221307 COMPLETED**, exit 0, in **1h28m23s**.
Finished **15,007,744 steps**. Metrics downloaded locally.

| Final target | Survived 25 s | Qualified gait | Mean first-episode duration |
|---|---|---|---|
| 0.20 m/s | 0/12 | 0/12 | 4.29 s |
| 0.25 m/s | 8/12 | 0/12 | 22.54 s |
| 0.35 m/s | 1/12 | 0/12 | 12.36 s |

At the training target, only one valid left step and one valid right step were
detected across all 12 final trials. Survival improved, but acceptable walking
was not learned. The success gate never opened across 16 evaluations; shaping
remained at 0.15 and entropy at 0.01. Best diagnostic candidate is the final policy.
No further training submitted.

- Snapshot: `gait06_20261016_v1`; run: `cad_v2_gait06_s0`.
- Fresh policy, 15M steps at 0.25 m/s; success-gated bilateral shaping.
- Local validation: 62 passed, four JAX-dependent tests skipped.
- Remote launch checks completed sufficiently to enter and finish training.
- [Experiment and exact gate](../../gait06-experiment-report.md).
- [Source manifest](source-manifest.json); [archive checksum](bundle.json).

Remote log:
`/home/rizkalla/running_robot/experiments/gait06_20261016_v1/RLframework/logs/cad-v2-gait06-3221307.out`

Remote results:
`/home/rizkalla/running_robot/experiments/gait06_20261016_v1/RLframework/runs/cad_v2_gait06_s0/`

Status: `ssh -i C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519 -o IdentitiesOnly=yes rizkalla@izar.hpc.epfl.ch 'squeue -j 3221307'`
