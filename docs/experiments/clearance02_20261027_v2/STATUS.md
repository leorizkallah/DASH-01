# CLEARANCE-02 retry: completed — job 3232851

## Video and convergence curves — 2 November 2026

[Watch the selected candidate and inspect its learning curves](../../../report/clearance02-videos.html).
Best candidate **132.71M**: 25 s two-view video at 1.50 m/s, real-time with
slow-motion controls. Convergence exports: PNG/SVG/PDF, raw/smoothed reward,
tracking, mean lift, peak lift, mid-swing coverage and survival/qualification.
The selected candidate is marked; source evaluation values remain unchanged.

Replay job **3238862 COMPLETED**, exit 0, **3m01s**, node `i48`. Same selected
checkpoint SHA256 `3465b06b08398c43835e8a851c2e083c801a3e68fa67ecdf0757231052c105b8`.
The video shows preselected **trial 0 of 12**, saved fixed reset command, seed
1000, nominal settings, no auto-reset splicing. Trial 0 lasts 25 s, speed error
0.0360 m/s after warmup, 80/80 completed cycles. All trial metadata are retained.

**Fresh replay discrepancy:** only **9/12** complete 25 s; trials 1/4/5 fall at
0.70/0.95/0.95 s. The archived best evaluation reports 12/12 full 30 s.
Checkpoint, batch size and saved reset-command convention match, but the
recorder is a separate compiled computation; the cause is not isolated.
Do not claim robust startup from the displayed successful clip or archived
evaluation alone. No standalone batch-1 replay or hardware validation performed.
[Replay evidence and video checks](replay-provenance.json).
No training job was launched for this request.

## Training outcome

Checked **2 November 2026** on explicit request. Job **COMPLETED**, exit **0:0**,
after **11h07m53s**, node `ixl01` (Tesla V100-SXM2-32GB). Started 27 October
13:28:47 and ended **28 October 00:36:40**, Europe/Zurich. Finished the full
**200,005,632 additional steps**; stop reason `step_limit`, not wall time.
GPU initialization, all 27 tests and warm-start smoke passed. The user queue is empty.

At **1.50 m/s**, the final checkpoint's 12/12 nominal trials last the full 30 s:

| Metric | Parent baseline | Best clearance candidate | Final |
|---|---:|---:|---:|
| Additional steps | 0 | 132.71M | 200.01M |
| Mean airborne clearance L/R | 2.47 / 2.27 mm | 16.94 / 20.06 mm | 15.82 / 18.20 mm |
| Scheduled mid-swing samples >=10 mm, L/R | 0 / 0% | 100 / 100% | 100 / 100% |
| Largest sampled airborne peak L/R | 15.70 / 7.85 mm | 30.22 / 30.04 mm | 31.75 / 34.74 mm |
| Mean speed error | 0.0258 m/s | 0.0357 m/s | 0.0271 m/s |
| Fully qualified trials | 0/12 | 0/12 | 0/12 |

**Clearance improved, but the upper-height objective is not met.** Every best/final
trial passes duration, tracking, mean height, mid-swing coverage and cycle-count
checks, but fails the maximum peak criterion: <=26 mm (25 mm plus 1 mm tolerance).
No checkpoint qualified in any of the 29 recorded evaluations. The saved best
candidate at 132,710,400 is an unqualified candidate, not a validated keeper.
Its excess-height metrics are better than final; extra training did not improve
every objective monotonically. Peaks are sampled at 100 Hz.

Archived raw progress/evaluations, baseline/best/final metrics, training plot,
startup checks and job log. [Results summary and hashes](results-summary.json),
[Slurm accounting](sacct-20261102.txt), [training plot](training_plots.png).
During the earlier status-only check, no new replay/video or hardware test was
performed and no job was submitted. The subsequent requested replay is documented
above; the startup discrepancy remains unresolved.

## Original submission record

Successfully submitted **27 October 2026** as **SCITAS job 3232851**.
Receipt: `submission.txt`. All **27 tests passed on SCITAS**; all 93 source
hashes and the exact flat-feet parent hash/184.32M step were verified. The real
timeout mechanism returned exit 124 in 0.20 s for a deliberately sleeping child.
Archive SHA256: `4554e966eaa9836350443081b59eba1b40e44d166811d544899e4ef90daf741c`.
The complete training configuration matches the failed attempt exactly.
GPU checks remain automatic in-allocation gates, not completed validations.
No training monitoring after successful submission.

27 October 2026. User authorized resubmission after job 3231747 used 24 h
without completing preflight. New isolated experiment; previous failure evidence
and original parent checkpoint remain unchanged.

Same `cad_v2_clearance02` preset and `cad_v2_clearance02_s0` run name inside this
new directory. **200M additional steps from flat-feet best 184.32M**, fixed
1.50 m/s; 10 mm swing minimum, 25 mm bonus cap, explicit excess-height cost.
No reward, network, motor, optimizer or curriculum changes for the retry.

Startup changes only:

- Exclude node i39 as a precaution; node failure is not established.
- Separate per-job CUDA/JAX caches without deleting existing caches.
- NVIDIA driver/device query: 60 s bound.
- JAX GPU initialization and small compiled kernel: 180 s bound; no CPU fallback.
- Unit tests on CPU: 300 s bound.
- Existing GPU warm-start smoke: 1,200 s bound, stage JSON/log messages,
  repeated stack dumps every 300 s. Timeout sends TERM then KILL after 20 s.
- The unchanged real trainer runs only after the checks pass, with its existing
  parent reward/survival gate, 200M-step limit, 24 h ceiling and save reserve.

Local: 24 tests pass, 3 JAX-only tests skipped; CLI/syntax/lint checks pass.
GPU checks run in the allocation, not on the login node. Submission receipt
will be saved here. No repeated Codex monitoring after submission.
