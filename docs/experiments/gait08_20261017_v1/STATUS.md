# GAIT-08 status

18 October 2026 check: job **3222272 COMPLETED**, exit 0, in **3h21m54s**.
Finished **30,015,488 additional steps**. Downloaded all 30 gait evaluations,
training CSV and final/best-candidate summaries.

| Final target | Survived 25 s | Qualified gait | Valid L/R steps after 5 s |
|---|---|---|---|
| 0.20 m/s | 6/12 | 0/12 | 5.58 / 5.33 |
| 0.25 m/s | 12/12 | 0/12 | 10.17 / 10.00 |
| 0.35 m/s | 10/12 | 0/12 | 9.83 / 9.25 |

At 0.25 m/s, final mean speed error is 0.017 m/s and contact slip 0.0075 m/s.
All trials fail the cycle-symmetry and minimum-clear-time criteria; other gait
checks pass. Both feet now step, but full qualification remains unmet.

Best diagnostic candidate: **25,165,824 steps**, 12/12 survival and 10/10 valid
L/R steps with 100% alternation. Exact evaluated checkpoint retained.

Requested replay job **3222519 COMPLETED** in 3m18s. Candidate and final both
survive their 25-second 0.25 m/s replays, with 10/10 valid steps after five seconds.
Open the [video comparison](../../../report/gait08-videos.html).
No further training launched.

- **30M additional steps**, warm-start GAIT-07 final, fixed 0.25 m/s.
- Opposite-foot sequence rule enabled; full drag penalty retained from step zero.
- Local tests: 77 passed, seven JAX-dependent skips. Remote focused suite: 39
  passed. GPU benchmark precedes training.
- [Experiment report](../../gait08-experiment-report.md),
  [source manifest](source-manifest.json), [archive hash](bundle.json).

Remote root: `/home/rizkalla/running_robot/experiments/gait08_20261017_v1`

Log: `RLframework/logs/cad-v2-gait08-3222272.out`

Results: `RLframework/runs/cad_v2_gait08_s0/`

Status command:
`ssh -i C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519 -o IdentitiesOnly=yes rizkalla@izar.hpc.epfl.ch 'squeue -j 3222272'`
