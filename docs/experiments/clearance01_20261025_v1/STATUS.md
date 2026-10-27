# CLEARANCE-01: completed — 26 October 2026

Training job **3230928 COMPLETED**, exit 0:0, elapsed **3h30m21s**.
Stopped normally at **50,006,016 additional steps**, not the wall-clock ceiling.
Final is also best clearance candidate. Twelve nominal 30 s trials survive;
mean airborne clearance L/R **23.62/20.91 mm**, speed error **0.0292 m/s**.
No qualified checkpoint: scheduled mid-swing >=20 mm fraction **57.18%/71.43%**
fails the 80% per-foot threshold. Each trial records 99–100 valid cycles per foot.

Replay **3231710** completed: standalone pinned-command trial falls at 3.67 s.
Replay **3231713 COMPLETED**, exit 0:0, 2m21s, preserving the saved reset command
and 12-environment batch: all 12 trials complete 25 s, 80/80 valid cycles after
warmup. Trial 0 is shown; no selection across trials. Its mean airborne lift
L/R is 23.55/20.91 mm. Batch/reset protocol differences remain a limitation,
not a proven cause of the standalone fall.
[Videos and graphs](../../../report/clearance01-videos.html).
Historical preparation/submission notes follow.

50M additional PPO steps from flat-feet v1 best (184.32M), fixed target 1.50 m/s.
20 mm mid-swing minimum with strong low-height cost, bonus increasing to 30 mm.
23 local tests pass. GPU warm-start smoke and baseline budget/survival gates run
inside the allocation before training. No post-submission monitoring requested.

The submission command writes its receipt to `submission.txt` here.
[Exact reward and training contract](../../clearance01-experiment-report.md).

Pre-submission: 23 tests also pass on SCITAS. Parent warm-start decoding and JAX
reset/step shape tracing passed; fixed inherited difficulty/1.5 m/s command verified.
Archive SHA256: `b266783b85181987e6342910a9b3cc2ba0abee68e4c63c6c8359c5800a590b32`.

Submitted successfully as SCITAS job 3230928. No post-submission monitoring performed.
