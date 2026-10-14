# STOP-06 settling study: dash_stop06_timed_s0

Ground-truth readiness: horizontal speed <0.15 m/s, angular speed <0.5 rad/s,
tilt <15 degrees for 0.3 s after the ramp. This diagnostic is not five-second hold success.
Times are relative to STOP. Missing readiness is a failure to observe the event, not a zero time.

| Steps | Valid approaches | Stops | Standing | Walking retained | HOLD entry median (s) | Readiness events |
|---:|---:|---:|---:|:---:|---:|---:|
| 0 | 62.5% | 0.0% | 90.6% | True | 2.00 | 0/32 |
| 262144 | 62.5% | 0.0% | 96.9% | True | 2.00 | 0/32 |
| 524288 | 40.6% | 0.0% | 87.5% | True | 2.00 | 0/32 |
| 786432 | 34.4% | 0.0% | 93.8% | False | 2.00 | 0/32 |
| 1048576 | 31.2% | 0.0% | 90.6% | True | 2.00 | 0/32 |
| 1310720 | 18.8% | 0.0% | 93.8% | True | 2.00 | 0/32 |
| 1572864 | 37.5% | 0.0% | 100.0% | True | 2.00 | 0/32 |
| 1835008 | 25.0% | 0.0% | 96.9% | True | 2.00 | 0/32 |
| 2097152 | 40.6% | 0.0% | 96.9% | True | 2.00 | 0/32 |
| 2359296 | 40.6% | 0.0% | 100.0% | True | 2.00 | 0/32 |
| 2621440 | 40.6% | 0.0% | 96.9% | False | 2.00 | 0/32 |
| 2883584 | 46.9% | 0.0% | 100.0% | True | 2.00 | 0/32 |
| 3145728 | 43.8% | 0.0% | 96.9% | True | 2.00 | 0/32 |
| 3407872 | 37.5% | 0.0% | 96.9% | True | 2.00 | 0/32 |
| 3670016 | 31.2% | 0.0% | 96.9% | False | 2.00 | 0/32 |
| 3932160 | 43.8% | 0.0% | 96.9% | True | 2.00 | 0/32 |
| 4194304 | 59.4% | 0.0% | 96.9% | True | 2.00 | 0/32 |
| 4456448 | 43.8% | 0.0% | 93.8% | True | 2.00 | 0/32 |
| 4718592 | 53.1% | 0.0% | 100.0% | True | 2.00 | 0/32 |
| 4980736 | 50.0% | 0.0% | 96.9% | True | 2.00 | 0/32 |
| 5046272 | 65.6% | 0.0% | 100.0% | True | 2.00 | 0/32 |

Keeper selection still requires standing/walking retention. Guard patience only
changes when training stops; it does not relax success or selection thresholds.
Equal maximum budgets do not guarantee equal realized budgets if a guard stops an arm.
Single training seed: this comparison is exploratory, not a repeatability claim.
