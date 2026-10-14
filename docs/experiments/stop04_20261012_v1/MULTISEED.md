# STOP-04 multi-seed assessment

Low-speed stage review gate: **True**.
No transition training is launched automatically.

| Policy | Seed | Successful holds | Survived |
|---|---:|---:|---:|
| source_best | 4102 | 21/32 | 21/32 |
| source_best | 5102 | 19/32 | 27/32 |
| source_best | 6102 | 25/32 | 29/32 |
| pilot_best | 4102 | 29/32 | 29/32 |
| pilot_best | 5102 | 32/32 | 32/32 |
| pilot_best | 6102 | 30/32 | 31/32 |

4102 is the checkpoint selection seed; 5102 and 6102 are held out from selection.
Walking retention uses the standard ladder evaluation, not a multi-seed walking test.
