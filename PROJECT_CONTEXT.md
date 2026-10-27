# DASH-01 project context

Updated **27 October 2026**, Europe/Zurich. This is a project handoff, not a new
training instruction. Read this first, then the linked experiment records.

## Current state

- Latest authorized training: **CLEARANCE-02**, `clearance02_20261026_v1`,
  `cad_v2_clearance02_s0`, submitted as **job 3231747**. All 27 tests pass on
  SCITAS; source/parent hashes verified. No post-submission monitoring.
  Start again from **flat-feet best 184.32M**, not
  CLEARANCE-01. **200M additional steps**, fixed 1.50 m/s, 10 mm swing minimum,
  25 mm bonus cap and explicit excess-height cost above 25 mm. Upper height is
  a soft reward bound, not a kinematic clamp. Preserve previous symmetry,
  flat support, slip, plant difficulty and warm-start settings. 24 h ceiling,
  10 min save reserve; checkpoint every 2M, evaluation every ~7.37M.
  [Contract](docs/clearance02-experiment-report.md),
  [submission/status](docs/experiments/clearance02_20261026_v1/STATUS.md).
  Stop after successful submission; user explicitly requested no repeated monitoring.

- Latest completed training: **CLEARANCE-01**, job **3230928**, 50,006,016
  additional steps, 3h30m21s, exit 0:0. Final is also the best clearance candidate.
  Twelve nominal 30 s trials at 1.50 m/s all survive; mean airborne height
  **23.62/20.91 mm**, speed error **0.0292 m/s**. No checkpoint qualifies:
  scheduled mid-swing >=20 mm coverage is **57.18%/71.43%**, below 80% per foot.
  [Results and video gallery](report/clearance01-videos.html);
  [exact report](docs/clearance01-experiment-report.md). Replay job **3231710**.
  A separate single-environment pinned-command replay falls at 3.67 s despite
  that evaluation survival; this failure is retained in the gallery. Replay
  **3231713** preserves the saved reset command and 12-environment batch.
  It completed in 2m21s: 12/12 nominal 25 s replays survive, each 80/80 cycles.
  Main video shows trial 0, with mean height 23.55/20.91 mm and typical peaks
  46.55/40.80 mm. Both good and failure clips retained.
  This discrepancy remains unexplained; do not claim reliable standalone startup.
  No additional training authorized or launched during this status/video check.

- Working branch: `dash-01-v2-speed-discovery`.
- Previous published checkpoint: **7fc07a2**, pushed and verified on
  `origin/dash-01-v2-speed-discovery` (`leorizkallah/DASH-01`) on 23 October.
- Previous completed training: **dash01_flatfeet_symmetry_v1**, SCITAS job
  **3226505**, 200,005,632 steps in **12h54m12s**, exit 0.
- Best checkpoint: **184,320,000 steps**, `best.msgpack`; final:
  **200,005,632 steps**, `final.msgpack`.
- No further training was launched in the subsequent review or cleanup.
  SCITAS was not queried for live jobs during this 25 October summary.
- The 27 October cleanup groups clearance code, experiment evidence and the
  current midterm presentation for publication on this branch. Use `git log -1`
  for the resulting commit. Frozen experiment manifests and archives stay intact.
  Office lock files, caches, previews and the superseded local deck are ignored.
  Report dependencies are listed in `report/requirements.txt`; no SCITAS job
  monitoring was performed as part of the cleanup.

## New authorized experiment: CLEARANCE-01 (25 October)

The user requested **50M additional steps at 1.50 m/s** with a large penalty for
low swing clearance, a 20 mm target and a capped 20-30 mm bonus. Implemented as
`cad_v2_clearance01`, warm-starting the flat-feet **best 184.32M** checkpoint.
The full target applies in the middle 60% of swing, with 20% liftoff/landing ramps.
Deficiency is charged even if a scheduled swing foot stays grounded. Bonus requires
actual short-duration air and saturates at 30 mm. Old positive clearance is disabled
only in this new preset. Existing symmetry, support and slip terms remain.

Frozen inherited plant difficulty, lower fine-tuning LR, 50M extra steps, 8 h
allocation ceiling. 23 local/SCITAS tests pass; an in-job GPU smoke and baseline
reward/survival check must pass before training. Separate clearance-aware candidates
avoid selecting only for speed tracking. See [contract](docs/clearance01-experiment-report.md)
and [job receipt/status](docs/experiments/clearance01_20261025_v1/STATUS.md).
The submission command records the actual job ID. Do not monitor repeatedly or
start additional runs; wait for an explicit user status request.

## Objective and physical model

The long-term Robotics Day demonstration is: start from a defined pose and stand,
walk after START with adjustable target speed, then stop and remain upright after
STOP. The immediate task is useful, fluid walking on CAD-v2. Do not require a
pause between successive steps. Startup and STOP remain later integration tasks.

The new CAD and PDF mass properties were integrated and checked. Motors and
battery are already included in the PDF. Its COMs/inertias use the export frames;
do not add inferred frame transforms or duplicate motor masses. An additional
**1 kg torso allowance** covers cables/accessories; total mass **14.594154 kg**.
The six motors are unchanged; the Achilles links have **no spring**.

The supplied homing assembly has since been imported. The current training model
is `RLframework/model/dash01_cad_v2_homing.xml`, keyframe `homing`, with a free
base, articulated flat feet, toe/heel collision pads and four linkage constraints.
A short nominal PD hold passed. This does not establish a trained startup skill
or hardware homing procedure. See [CAD audit](report/cad-v2-import.md) and
[homing audit](report/data/homing01/audit.json).

## What has been tried

| Campaign | Outcome / lesson |
|---|---|
| STOP-01 through STOP-06, previous model | Investigated braking ramps, settling and five-second holds, standing demonstrations and walking retention. Standing improved; robust stopping from motion was not established. These are historical results for another model. |
| CAD-v2 GAIT-01 through GAIT-11 | Explored slower walking, bilateral activity, clearance, progressive contact sliding costs, repeated/incomplete-step penalties and explicit alternating commands. Survival and some gait metrics improved, but dragging/microsteps remained; no clean gait qualified. |
| SPEED-01, job 3223896 | Fresh maximum-speed objective based on Nathann's audited code. 59.92M steps in 4h52m. Best eligible speed 0.950 m/s; final 1.215 m/s survived but violated joint-limit qualification. Sliding and low clearance remained. |
| Original policy transfer | Nathann's original 180M policy survived on its original model. Swapping to CAD-v2, including simple bounds/offset probes, fell after about 0.71-0.75 s. Direct transfer was insufficient. |
| WALK-01, job 3224989 | Fresh Nathann joystick recipe with the new homing reset, no STOP. 200.01M steps in 13h29m. Best at 169.57M: 28/28 nominal 30 s trials, mean speed error 0.181 m/s. Strong leg-motion asymmetry remained. |
| Flat-feet symmetry v1, job 3226505 | Fresh 200M run with corrected two-pad reward handling, completed-cycle symmetry and mid-stance flat support. Better tracking and bilateral training averages, with important episode-level failures below. |

Earlier work is preserved in the Git history, including CAD-v2/GAIT archival
commit `c035774`. Experiment reports and galleries remain under `docs/` and
`report/`. Speed discovery is historical context; the latest run tracks commands.

## Current controller and rewards

PPO trains an actor using **384 observations**, with a learned velocity estimator;
the training critic has **413 inputs**. The actor outputs **47 actions** at
**100 Hz**: 41 gait parameters latched at cycle boundaries and six live joint
residuals. The gait generator creates joint targets and scheduled PD gains.
The motor model applies torque/speed limits at **1 kHz**. Motor order is
`hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R`.

The preset is `cad_v2_flatfeet_symmetry_v1`: 512 environments x 36 ticks, seed 0,
fresh weights/statistics, original PPO/controller/curriculum retained, STOP off.

- Support requires an actual toe/heel floor contact with normal force >=2 N.
  Toe OR heel defines whole-foot contact; toe AND heel defines flat support.
  Height proximity alone does not count. Rollover is not a new step.
- Slip uses tangential velocity at actual contact points, averaged by normal
  force within each foot. Its existing coefficient is unchanged.
- The old all-stance `w_foot_flat` is zero. A separate `w_flat_support=0.25`
  cost applies only during **20-80% of scheduled stance**, with no planted bonus.
- Completed-cycle symmetry uses recent stance/swing durations: three-cycle
  windows, at least two valid cycles per foot, minimum 0.04 s per phase.
  `w_symmetry=5`; persistent error above 0.40 can terminate after grace and
  curriculum gating (`shape_scale >=0.65`). Recovery/STOP exemptions apply.
- This does not enforce instantaneous equal joint angles. The unchanged gait
  generator permits asymmetric amplitudes (`1+s`, `1-s`).

Relevant files: [environment/reward](RLframework/env.py),
[two-pad and symmetry helpers](RLframework/flatfeet.py),
[presets](RLframework/config.py), [gait decoder](RLframework/gait.py),
[plant](RLframework/plant.py), [motor model](RLframework/drive.py),
[PPO](RLframework/ppo.py). Full numerical definitions are in the latest report.

## Latest evidence and limits

Best flat-feet checkpoint: **28/28 full-duration nominal fixed-ladder trials**,
mean speed error **0.0489 m/s**, versus WALK-01's 0.1809 m/s. These are different
single-seed runs with several simultaneous changes, not a controlled ablation.
Last periodic evaluation at 199.07M: 27/28, one fall, error 0.0442 m/s.

Fixed-command replay, seed 1000; cycle counts exclude the first five seconds:

| Checkpoint | Target | Mean achieved speed | Valid cycles L/R | Outcome |
|---|---:|---:|---:|---|
| Best | 0.25 m/s | 0.344 m/s | 78 / 80 | 25 s completed |
| Best | 0.75 m/s | 0.787 m/s | 1 / 80 | 25 s completed |
| Best | 1.50 m/s | 1.483 m/s | 80 / 80 | 25 s completed |
| Final | 0.25 m/s | 0.298 m/s | 36 / 60 | 25 s completed |
| Final | 0.75 m/s | 0.721 m/s | 77 / 68 | 25 s completed |
| Final | 1.50 m/s | 1.501 m/s | 54 / 36 | Symmetry termination at 19.22 s |

**Confirmed blind spot:** short invalid contact phases can leave completed-step
history stale, while brief liftoffs reset the continuous planted-leg timer.
The best 0.75 m/s trial therefore escapes termination despite 1/80 valid cycles.
This is not evidence that the left motor is inactive. Aggregate duty symmetry
or low logged timing error does not certify clean alternating walking.

Recent randomized training slip averages about **0.20 m/s**; the clock reaches
its **4 Hz upper bound**. Early slip diagnostics have 31 nonfinite logged updates
per side, displayed as gaps; logged mean rewards remain finite.

The final separate command-redraw evaluation has zero physical falls but two
early non-fall endings: only 26/28 reach 30 s. Its legacy evaluator omits the
symmetry reason and its zero red-light `track_err` is not walking tracking error.
No hardware validation or robust clean gait/START/STOP demonstration is claimed.

## Evidence and reproducibility

- [Latest full report](docs/flatfeet-symmetry-v1.md),
  [six videos and graphs](report/flatfeet-videos.html),
  [status](docs/experiments/dash01_flatfeet_symmetry_v1/STATUS.md),
  [numerical summary](docs/experiments/dash01_flatfeet_symmetry_v1/results-summary.json).
- SCITAS root: `/home/rizkalla/running_robot/experiments/dash01_flatfeet_symmetry_v1`.
  Checkpoints: `RLframework/runs/dash01_flatfeet_symmetry_v1_s0/` under that root.
  Replay job **3226907** completed; recorded MJX traces are stored locally.
- Frozen source/configuration manifests and archives describe the trained code.
  Later formatting means current source hashes can differ; do not rewrite old
  manifests to make them match the current working tree.
- 23 October cleanup passed **22 local tests**, lint/syntax checks, and verified
  all 21 compact videos. Video sizes fell from 342 MB to 52.7 MB; full-quality
  originals remain ignored in `report/videos/.originals/`. Original/compact hashes
  are mapped in `report/video-encoding.json`; source formatting hashes are in
  `report/cleanup-source-hashes.json`.
- Local Windows Python environment: `.venv/Scripts/python`. Full JAX/MJX GPU
  execution uses SCITAS's established `~/venvs/dash-v2` environment.

## Recommended next work and user preferences

Latest user direction (25 October): **focus on target 1.50 m/s**, including mean
swing height as well as peaks, when investigating extra swing clearance for the
promising flat-feet policy. First measured the existing best/final replay data;
see [swing clearance report](report/flatfeet-clearance.md). Typical whole-foot
peak medians are 0.24-9.00 mm depending on foot/checkpoint/speed. The current
clearance bonus starts at 20 mm and is zero over the analysed steady-state
windows. The extra constraint is now implemented in CLEARANCE-01 above; the parent policy
is preserved. Do not describe the new target as achieved before evaluation.
At 1.50 m/s, mean airborne height L/R is **2.44/2.17 mm** for best and
**0.52/0.52 mm** for final (pooled means 2.31 and 0.52 mm). The best 1.50 m/s
trial completes 25 s; final ends through symmetry termination at 19.22 s.

First diagnose and test contact-event debouncing and stale/missing step-history
detection, using the recorded failure as a regression case. Review gait quality
at individual speeds before selecting a checkpoint or extending training.
This recommendation has not been implemented or submitted as a new experiment.

Keep useful plots, videos, exact configs, provenance and limitations for the final
report/presentation. Conserve **Codex credits** (the user's clarification), batch
checks and avoid repeated training monitoring. After an authorized submission,
stop when requested; later explicit status/video requests authorize a new review.
Do not launch training merely because this handoff recommends a next experiment.

CLEARANCE-01 submission receipt: SCITAS job 3230928. Completed; reviewed on
explicit user requests. Latest result is recorded above.
