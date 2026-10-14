# STOP-05 low-speed motion assessment

Selected checkpoint: 0 new PPO steps.

| Seed | Qualified moving approaches | Successful stops | Restart successes | Standing holds |
|---:|---:|---:|---:|---:|
| 4102 | 17/32 | 0/32 | 0/32 | 29/32 |
| 5102 | 21/32 | 0/32 | 0/32 | 32/32 |
| 6102 | 17/32 | 0/32 | 0/32 | 29/32 |

Per-speed and per-trial results are in motion_assessment.json.
4102 is the keeper-selection seed; 5102 and 6102 are held out from selection.
Walking retention uses the standard ladder, not a three-seed walking test.
No automatic increase in speed or additional training is scheduled.
