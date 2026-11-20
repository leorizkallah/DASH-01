# CLEARANCE-02: timed out before training — job 3231747

Checked on explicit user request, **27 October 2026**. Slurm reports **TIMEOUT**
after the full **24 h allocation**; the batch step was terminated by SIGTERM.
Source/parent hash verification and all 27 tests passed. The next GPU preflight
did not complete: no `validation/clearance-preflight.json` or `validation/smoke/`
directory exists, and the main `RLframework/runs/cad_v2_clearance02_s0/`
directory does not exist. **No main training or trained checkpoint was produced.**

The 438-byte log contains only validation success, optional Warp import warnings
and Slurm's time-limit termination. It does not identify the exact hang location
or cause. Resource accounting: node i39, one Volta GPU, total CPU 23h56m23s,
maximum batch RSS 722,464 KB. These counters alone do not establish the cause.
Archived log: `cad-v2-clearance02-3231747.out`.

Next step: diagnose/instrument the preflight and give it a separate timeout
before retrying. No replacement job was submitted during this status check.
Historical submission notes follow.

Successfully submitted on 26 October 2026. Receipt: `submission.txt`.
All **27 tests passed on SCITAS**, including synthetic evaluator tests. All 92
frozen source hashes and the original parent checkpoint hash/184.32M step were
verified. Archive SHA256:
`5e21b61f32c72127378d0632b44e88fb0f718d8a2bc2a5fd7ddca567e7ce945a`.
GPU smoke and baseline gates execute inside the allocation; they are not yet
claimed as passed. No training monitoring after successful submission.

26 October 2026. Original flat-feet best at 184.32M is the parent, not CLEARANCE-01.
200M additional steps at 1.50 m/s. Swing minimum 10 mm with edge ramps;
bonus cap 25 mm, plus explicit excess-height cost above 25 mm.
Preserve symmetry, flat support and slip. 24 h ceiling, periodic saves and
automatic baseline/GPU gates. No repeated monitoring after submission.

[Exact training/reward contract](../../clearance02-experiment-report.md).
Submission ID: **3231747**. Saved configuration: `requested_config.json`;
pre-submission verification: `pre-submission.json`.
