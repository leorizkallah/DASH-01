# Work log and next questions

2026-10-27: prepared repository cleanup and publication of the clearance work
and current Tim-template presentation. Removed unused imports, made repeated
presentation builds reset their section state, and added local report
dependencies. Ignored caches, Office locks, derived previews and the superseded
local draft without deleting files. Compact clearance gallery videos retain
full-quality originals under ignored `report/videos/.originals/`; sizes fall
from 24.3 MB to 3.7 MB, with both hashes recorded in `video-encoding.json`.
Frozen experiment manifests/archives, raw metrics, replay traces and delivered
presentation bytes are preserved. No training monitoring or reward changes
were introduced by this cleanup. Focused local tests: 24 passed, 3 JAX-only
evaluator tests skipped (all 27 passed on SCITAS before CLEARANCE-02 submission).

2026-10-26: revised the controller midterm at the user's request to follow
**Tim’s actual presentation template**, rather than only matching colors.
Reused the supplied masters/theme, COVER/LIGHT/DARK layouts, cover, outline,
four section dividers and closing. Kept controller evidence, four embedded clips,
speaker notes and relative roadmap. New separate file:
`DASH01_controller_midterm_Tim_template.pptx`, with PDF preview. 18 main slides
including dividers (~9m35) plus three hidden backups. Original source deck and
the first controller deck are preserved. No training monitoring performed.

2026-10-26: prepared the controller midterm presentation using Tim Kobler’s
supplied PPTX as the visual reference; his original is unchanged. User chose
8–10 minutes and a relative timeline. Created 12 main slides (~9 minutes) and
three hidden backup slides, with editable architecture/demo/timeline diagrams,
four embedded simulation clips, sourced tracking/clearance plots and speaker
notes. Roadmap retains the original startup → START/speed control → STOP/5 s hold
sequence and prioritizes treadmill testing at the next available prepared test
session. Simulation results, known replay failure and unreviewed CLEARANCE-02
submission are explicitly distinguished. No training monitoring performed.
[Deck, PDF, notes and provenance](presentations/midterm-2026-10-29/README.md).

2026-10-26: user authorized CLEARANCE-02, 200M additional steps starting again
from **flat-feet v1 best 184.32M**, not the higher-lift CLEARANCE-01 checkpoint.
Lower scheduled mid-swing target to 10 mm, bonus saturation to 25 mm, add explicit
excess-height cost above 25 mm because bonus saturation alone did not prevent
40–50 mm peaks. Keep all existing symmetry/flat-support/slip shaping, fixed
1.50 m/s, inherited plant difficulty and warm-start normalization/noise settings.
Preserve normal swing-edge ramps and fluid motion. Candidate selection balances
low and excessive lift; evaluate minimum coverage, mean, maximum and time above
cap with truthful 10 mm metric labels. See [contract](../docs/clearance02-experiment-report.md).
Frozen archive SHA256 `5e21b61f32c72127378d0632b44e88fb0f718d8a2bc2a5fd7ddca567e7ce945a`.
Local: 24 tests passed; 3 evaluator tests deferred to SCITAS JAX. GPU smoke and
baseline survival/reward gates run within the allocation before real training.
Submit once; no automatic follow-on run or repeated Codex monitoring.
Submitted successfully as **SCITAS job 3231747**. All 27 tests pass remotely,
92 source hashes and exact parent hash/step verified. Submission receipt saved;
no training monitoring after submission.

2026-10-26: reviewed CLEARANCE-01 on explicit request and archived results.
Job 3230928 completed 50,006,016 additional steps in 3h30m21s, exit 0:0.
Matched nominal parent/final evaluations (12 first episodes, 30 s, seed 1000,
5 s warmup, fixed 1.50 m/s) improve mean airborne height L/R from 2.46/2.25 mm
to 23.62/20.91 mm. Final survival 12/12 and error 0.0292 m/s, but mid-swing
>=20 mm fractions 57.18%/71.43% fail 80%; no qualified checkpoint.
Archived evaluation records and generated learning/mean-comparison graphs.
Replay 3231710 falls at 3.67 s in a single pinned-command environment; retained
the failure rather than selecting around it. Replay 3231713 uses the evaluator's
12-environment batch and saved reset command. Both batch size and command handling
differ, so no causal attribution yet. Recorded this limitation in the report and
handoff. No new training launched. See [results/videos](clearance01-videos.html).
Replay 3231713 completed in 2m21s: 12/12 full 25 s, 80/80 cycles per foot.
Rendered trial 0 without selecting among trials, plus the standalone failure.
Recomputed whole-foot clearance from oriented pad corners with exact frozen-model
hash verification and post-step contact alignment. Trial-0 mean L/R 23.55/20.91 mm;
median complete >=40 ms air-bout peaks 46.55/40.80 mm. Saved raw traces, CSV/NPZ
measurements, source/checkpoint hashes and the known single nonfinite diagnostic
update at 14.32M in the experiment report.

2026-10-25: prepared CLEARANCE-01 at the user's request: 50M additional steps
from flat-feet best 184.32M, fixed command 1.50 m/s. Added a dense deficiency cost
for scheduled-swing whole-foot height below a 20 mm target, with takeoff/landing
ramps, and a separate bonus from 20 to 30 mm. Kept parent plant difficulty and
added clearance-aware candidate checkpointing. 23 tests pass locally and on
SCITAS. GPU smoke and baseline reward-budget gates precede long training in-job.
[Contract](../docs/clearance01-experiment-report.md),
[submission receipt](../docs/experiments/clearance01_20261025_v1/STATUS.md).
No continuous Codex monitoring after submission.

2026-10-25: user narrowed clearance work to **1.50 m/s** and requested mean
height alongside peaks. Promoted the time-averaged airborne measurements in the
[clearance report](flatfeet-clearance.md): best L/R 2.44/2.17 mm, final
0.52/0.52 mm; pooled means 2.31/0.52 mm. Added a focused mean/peak figure and
1.50 m/s CSV, retaining other speeds as supplementary data. No training change.

2026-10-25: measured swing clearance from the existing flat-feet MJX recordings,
using the frozen homing XML and orientation-aware minimum toe/heel corner height.
Aligned pre-step poses with previous-step contact flags; excluded startup/resets.
Typical swing peaks are below 10 mm. Final left-foot median peaks at targets
0.25/0.75/1.50 m/s are 0.24/1.91/1.05 mm. The existing 20 mm clearance deadband
makes its bonus zero in these steady-state clips. Saved summary/bout CSVs, signed
height traces, source hashes and figures in [the clearance report](flatfeet-clearance.md).
The user's next interest is gradually increasing foot lift; no reward change or
new training job has been made for this data request.

2026-10-25: reviewed the local experiment records and wrote
[`PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md) as the current summary and session
handoff. Updated README/architecture descriptions to reflect supplied homing,
completed flat-feet training and the remaining contact-history blind spot.
No new training, remote status query, commit or push was performed for this update.

2026-10-23: repository cleanup for push. Formatted new reward, diagnostic and report
modules; removed an unused import; declared MessagePack; corrected stale README
status and text encoding. Kept the existing Git attributes and normalized shell
launchers to LF. Curated new videos were encoded as smaller H.264 report copies,
with full-quality originals retained locally under ignored `videos/.originals/`.
`video-encoding.json` maps original and compact hashes; `cleanup-source-hashes.json`
records source formatting hashes. Frozen training archives and results are preserved.
22 local tests passed; lint and syntax checks passed on cleaned modules.

Flat-feet v1 completion review: job 3226505 completed 200,005,632 steps in
12h54m12s. Best at 184.32M: 28/28, tracking error 0.049 m/s (WALK-01: 0.181).
Closer bilateral aggregate timing, but contact slip remains ~0.20 m/s.
Downloaded CSVs and evaluation records; built comparison, support/slip, reward
and termination plots. MJX replay job 3226907 completed in 4m18s. Best 0.75 m/s has 1/80 valid
L/R cycles after startup but does not terminate: short contact interruptions
can leave paired history stale while resetting the planted-leg timer. Final
0.75 m/s is 77/68; final 1.5 m/s ends through symmetry termination.
Six clips retain all trial outcomes.
[Gallery](flatfeet-videos.html), [report](../docs/flatfeet-symmetry-v1.md).
No new training launched.

2026-10-20 (client date): implemented the user's focused flat-foot adaptation: actual
toe/heel support and contact-point slip, completed-cycle symmetry, delayed persistent
termination, and mid-stance-only flat-support cost. Fresh 200M run prepared with
16 passing local tests and a mandatory in-allocation GPU smoke gate.
[Exact contract](../docs/flatfeet-symmetry-v1.md);
[submission record](../docs/experiments/dash01_flatfeet_symmetry_v1/STATUS.md).
Stop after submission; do not monitor training.

WALK-01 completion check: 200,005,632 steps in 13h29m16s, exit 0. Best checkpoint
169.57M: 28/28 survival, 0.181 m/s error; last periodic evaluation: 28/28,
0.328 m/s. Final command-redraw evaluation: all 28 last 30 s without falling.
Downloaded metrics and best/final weights. Submitted faithful MJX replay 3226290;
GPU queue estimated roughly two hours. CPU-only export allocation was rejected
by SCITAS QOSMinGRES, so prepared labelled local deployment previews instead,
using a small local MessagePack reader. Six 25-second trials survive. CPU solver
requires XML 100/50 settings; MJX 16/8 caps were numerically unstable on CPU.
The queue cleared early: MJX replay 3226290 completed in 4m27s, exit 0.
All six MJX trials survive 25 seconds; the main gallery now uses these recorded
training-simulator motions. CPU previews remain separately labelled.
No training changes. [Video gallery](walk01-videos.html),
[results and limitations](../docs/walk01-experiment-report.md).

WALK-01 submitted as SCITAS job **3224989**; initial state **PENDING (Priority)**.
Frozen 84-file source snapshot, 200M-step target, 24-hour ceiling, no STOP.
Local model/config/budget checks passed; GPU preflight and training await allocation.
[Status](../docs/experiments/walk01_20261019_v1/STATUS.md).

2026-10-19: prepared WALK-01 at the user's request: fresh 200M-step reproduction
of Nathann's joystick recipe without STOP practice. Imported the supplied homing
assembly as a pose in the calibrated CAD-v2 plant, preserving corrected masses
and rigid links. Meshes match, pose fit is sub-micrometre, and five-second nominal
PD hold passes. Documented action-envelope, height, rigid-loop randomization and
GPU batching adaptations. [Experiment contract](../docs/walk01-experiment-report.md).

2026-10-19: tested the user's original-policy transfer proposal. Verified original
joystick bundle and original XML match fetched upstream Git blobs. Recorded paired
20-second CPU replays at 0.25/0.50/0.75 m/s: original survives all; unchanged
CAD-v2 swap falls at 0.74/0.72/0.71 s with unreachable cam/thigh commands.
Clamping and reset-relative offset probes remove command-bound violations but
still fall within 0.71–0.75 s. Saved traces, metrics, paired videos and comparison
figure; no training or reward change. [Transfer report](policy-transfer01.md).

SPEED-01 completion: job 3223896 finished on 19 October 2026 at 23:39:38 Swiss
time, exit 0, after 4h52m14s and 59,916,288 steps. Downloaded training and all
30 evaluation records. Best eligible checkpoint: 0.950 m/s, 16/16 survival;
final: 1.215 m/s, 16/16 survival but 0/16 joint-limit qualification. Substantial
sliding and almost no 15 mm clearance remain. Checkpoints verified on SCITAS;
video inspection is the next diagnostic, with no additional training submitted.
[Completed results](../docs/experiments/speed01_20261019_v1/STATUS.md).

2026-10-19: submitted SPEED-01 job **3223896**, five hours on SCITAS `i28`,
18:47:24 to 23:47:24 Europe/Zurich. All 67 scheduled GPU tests and finite PPO
checks passed. Four reward probes were finite; archived GAIT-09 motion earned
positive reward at 0.246 m/s under full regularization, without reward-floor
hits. The independent training process starts fresh at step zero.
[Run status and validation evidence](../docs/experiments/speed01_20261019_v1/STATUS.md).

2026-10-19: implemented the authorized five-hour SPEED-01 experiment, based on
Nathann's modern speed objective and corrected free CAD-v2 plant. No alive bonus
or prescribed foot-order rewards. Signed world-X income, explicit regularization
list, actual contact-point slip and a 150-point fall cost versus a 100-point
discounted worst-living bound. Added first-episode survival-first speed selection,
physical diagnostics, budget probes and regression tests. Archived GAIT-09 is
used only as a moving reward-budget probe, never training initialization.
[Experiment details](../docs/speed01-experiment-report.md).

2026-10-19: user rejected the GAIT-11 videos and changed direction to fresh
maximum-speed discovery. Committed the existing CAD-v2/GAIT-01–11 campaign on
`dash-01-v2` (`c035774`) and created `dash-01-v2-speed-discovery`. Fetched and
inspected seven Nathann upstream branches, the M3 saved configuration, modern
`dash_sprint`, and historical free-body M6 records. Preserved nine historical
JSON records with source hashes. The review distinguishes clipped speed income,
alive bonus, gait shaping and physical-model assumptions. No reward code was
changed and no new job launched during this review.
[Source audit](nathann-speed-reward-audit.md) and
[new plan](../docs/speed-discovery-plan.md).

2026-10-19: GAIT-11 job 3223293 completed in 7h51m56s, 65,650,688 steps.
Final 0.15 m/s survival 12/12, incomplete attempts 24.44%, with repeated right
foot attempts and insufficient clearance. No qualified gait in 64 evaluations.
Downloaded training/evaluation records; submitted replay job 3223517 for the
saved candidate (49.28M) and final policy at 0.10/0.15/0.20 m/s. No training
changes or new training run were made during this status/video request.
Replay job completed normally in 4m27s. Six clips and a learning-progress figure
are collected in the [GAIT-11 video gallery](gait11-videos.html); both 0.15 m/s
replays survive, with four right-foot repeats in the final clip. Low/high-speed
failures are retained. Replay metadata, hashes and scripts make assets reproducible.

2026-10-19: implemented GAIT-11 after GAIT-10 failed the agreed progress criterion.
Fresh policy at 0.15 m/s with explicit supported alternating turns, all-attempt
repeat accounting, retained bilateral quality rewards and competence-earned
penalty ramps. User clarified that walking must remain fluid: a successful
touchdown permits the next swing immediately, without a stop or settling dwell.
Eight-hour allocation includes setup/evaluation and a ten-minute final-save
reserve, with no environment-step limit. Four new contact-derived actor commands
require a future hardware estimator/runtime update. See
[design](../docs/gait11-experiment-report.md) and
[status](../docs/experiments/gait11_20261019_v1/STATUS.md).

2026-10-14: set aside the gait-search proposal at the user's request. Cleaned up
report tooling and Git artifact rules before publishing completed STOP-04–06 work.
Downloaded archives, TensorBoard logs and intermediate checkpoint sidecars stay
local. Existing results and frozen SCITAS snapshots are preserved.

2026-10-13: reviewed Nathann's PPO gait generator, older CMA-ES gait library,
archived 12-parameter CEM braking search, and existing BalanceRL stander. The
braking search uses a weaker speed criterion than our sustained-stop benchmark.
Historical proposal, set aside on 14 October. [Code review and evidence](nathann-gait-review.md).
No training or controller changes were made for this review.

2026-10-13: both STOP-06 arms completed 5,046,272 steps overnight with 0/32
moving stops at every evaluation. Final standing holds were 32/32 timed and 30/32
feedback; final walking retention passed only for timed. Source keepers were
retained. Automatic replay clips therefore show the source policies, not the
final trained policies. See [completed outcome](data/stop06_20261012_v1/STATUS.md).

2026-10-12: STOP-05 stopped at 262,144 steps after retention regressions, with no
successful moving stops. User approved [STOP-06](../docs/stop06-experiment-report.md):
matched longer learning with timed versus feedback HOLD, bounded settling and
automatic timing diagnostics. See [status](../docs/experiments/stop06_20261012_v1/STATUS.md).
Array **3216469**: timed arm running and feedback arm queued at 19:50 Zurich;
33 SCITAS tests passed. Results pending.
These new experiments are not represented by the older gallery figures/videos.

| Date / record | Change or decision | Reason / outcome |
|---|---|---|
| 2 October, STOP-01B | STOP fine-tuning from walking bundle | 20M steps without successful benchmark stops |
| 6 October, STOP-02 | Shared smooth speed adapter; five-second evaluation; dense HOLD rewards | Ramp helped survival; standing-only PPO still failed |
| 7 October, STOP-03 submission | Actor imitation followed by gated standing PPO | Supply a feasible stance while retaining walking |
| STOP-03 completion record | Keep 524k hold keeper, not only final weights | 21/32 best holds; final 14/32 despite 31/32 survival |
| 12 October | Create this report folder, copy evidence, generate figures/gallery | Make final report and presentations reproducible |
| 12 October, STOP-04 outcome | Standing keeper at step 262,144 | 29/32, 32/32, 30/32 holds across three evaluation seeds; standing gate passed |
| 12 October, STOP-05, job 3216384 | Low-speed moving-stop pilot with rehearsal and preservation guards | Stopped by retention guard at 262,144 steps; 0/32 moving stops |
| 12 October, STOP-06, array 3216469 | Longer timed versus feedback HOLD comparison | Both completed 5.05M steps; no successful moving stops |

Latest experiment details: [STOP-06 protocol](../docs/stop06-experiment-report.md)
and [completed outcome](data/stop06_20261012_v1/STATUS.md).
The existing figures and videos describe earlier experiments; they are not
STOP-05 results. The report's copied code remains a historical snapshot.

No policy, reward or training changes were made for the report-folder task. Local
metrics are a downloaded snapshot, not a live SCITAS dashboard. The existing source
manifests identify the exact training snapshots; this folder's manifest identifies
the evidence and current code copied into the report.

## Next investigations, not new results

12 October, approved continuation: [STOP-04](../docs/stop04-experiment-report.md)
automates paired checkpoint diagnosis and video recording, then a conditional
one-million-step standing PPO pilot with continued HOLD/RUN imitation. A physical
regression guard and three-seed final assessment are included. This supersedes the
earlier "next investigation" status of items 1-3 below; outcomes remain pending.
Slurm job **3215784** was confirmed queued (Priority) at 11:00:20 Europe/Zurich.

1. Record the STOP-03 best and post-imitation policies, with a declared trial
   selection rule and overlays for the five-second conditions.
2. Trace surviving final-policy failures to identify which conditions interrupt
   quietness, before changing reward weights.
3. Decide how to preserve the learned stance during PPO (for example, continued
   imitation/rehearsal); test this as a separate intervention.
4. Validate small perturbations and additional seeds, then low-speed moving STOP
   transitions. Consider actual-speed/posture conditions for the phase handoff.
5. Evaluate restarting and, later, broader plant/sensing variation.

## How to add an experiment

Record the hypothesis and intended comparison before running. Store snapshot/job
IDs, configuration, source policy/checkpoint hashes, changed files and training
budget. Keep evaluation protocol/seed, scalar reward and physical success separate.
Copy small result files from SCITAS on demand, then add the new experiment to
`scripts/build_report.py` and rebuild this folder.

For a video, record checkpoint, seed, trial, command timing, playback speed and
selection criterion. Prefer a representative or predeclared trial; if selecting
the best one, say so and give the aggregate success rate alongside it. Keep a
failed trial available for comparison. A replay does not replace quantitative
evaluation.

Update `experiments.md`, `evaluation.md` and this log with observations first,
then interpretations and decisions. Add figure captions and source paths; never
overwrite an earlier run's raw evidence to make the narrative cleaner.

## 14 October 2026 — v2 CAD preparation

Imported the supplied closed-loop CAD and PDF physical properties. Replaced all
15 body inertials and added 1 kg to the torso using an explicitly documented
mass-distribution approximation. User confirmed motors/battery are included
and PDF frames already match the export; no inferred transforms were applied.
Corrected mass: 14.594154 kg. Seven regression tests and the numerical MuJoCo
checks pass. Four loop anchors close within 0.000474 mm. Raw CAD preserved.

Recorded geometry/limit discrepancies and missing training-plant setup in the
[v2 audit](cad-v2-import.md), with PNG/SVG/PDF mass charts, CSV and a static CAD
render. No new policy or cluster run. Next: validate geometry/limits and prepare
the floating-base/contact/actuator configuration for the articulated-foot model.

### 14 October 2026 — GAIT-01 articulated-foot walking pilot

Integrated the reviewed 14.594154 kg model, six unchanged motors, four rigid
loops and separate heel/toe contacts. Job 3219514 failed its fixed-target
standing check before training. Small local pitch disturbances reproduce the
instability; nominal equilibrium is not evidence of stable balance. Gain and
stance scans did not resolve it and were not adopted.

The user requested walking first, postponing separate balance/standing work
until a homing position is supplied. Job 3219542 submits fresh 0.2–0.5 m/s PPO
after a CPU/MJX comparison and finite-update benchmark. No balance controller
or standing phase was introduced. At submission it is pending priority.
See [GAIT-01 report](../docs/gait01-experiment-report.md) for source manifests,
evidence, rewards, limits and the explicitly labeled nominal-reset video.

Status check later on 14 October: job 3219542 is running, 3,588,096/5M steps.
Numerical preflight and finite PPO passed. Evaluation at 3,145,728 steps has
96/96 falls; no successful gait yet. Training episode duration has increased
from roughly 0.23 to 1.95 s. Metrics downloaded to the v2 experiment snapshot;
estimated remaining time at the check is 5–7 minutes. No training changes made.

### 14 October 2026 — GAIT-02 longer walking stage

User approved a longer run with the existing walking reward. GAIT-01 completed
as job 3219542 in 27m19s, with no successful periodic evaluation. Submitted
GAIT-02 as job 3219597 for 30M additional steps. It compares best/final pilot
checkpoints before loading the selected weights and exact observation statistics.
Optimizer and schedules restart; shaping/efficiency ramps extend to 30M and
entropy scales to the larger budget. Physical model, reward terms/coefficients
and 0.2–0.5 m/s target range remain unchanged. Homing/standing remains deferred.
See [GAIT-02 report](../docs/gait02-experiment-report.md) for exact provenance.

### 15 October 2026 — visualize the learned gait

User requested video and clarification of the run's purpose. GAIT-02 was the
low-speed gait-discovery stage, not the maximum-speed search. Training had saved
checkpoints/metrics but no video. Replay job 3219901 recorded the best checkpoint
at three fixed commands. All trials retained: 25 s survival at 0.20/0.35 m/s,
fall at 20.31 s at 0.50 m/s. Mean speed after 5 s at the middle target is 0.333 m/s.
Replay flight fractions 2.27–4.61% prevent a strict walking-only success claim.
Videos, metadata and interpretation are linked in the GAIT-02 report. No training
or reward changes were made.

### 15 October 2026 — gait rejected after bilateral inspection

User observed right-foot dragging. Recorded 0.35 m/s trajectory, seconds 5–25:
whole sole above 1 cm in 13.15% of left-foot samples versus 0.35% on the right.
Code review found disabled swing-floor penalty and a misleading `duty_sym` name:
it only enforces minimum support time. Proposed bilateral swing, alternating
step credit, cycle-level timing/length symmetry and contact-point slip penalties,
with explicit gait acceptance checks. GAIT-02 is not accepted as walking.
No reward edits or new training submitted. See
[GAIT-03 proposal](../docs/gait03-symmetry-plan.md).

### 15 October 2026 — GAIT-02 completed

Job 3219597 completed in 2h23m48s, exit 0, at 30,015,488 new steps. Selection
used the pilot's final checkpoint (5,013,504 steps). Best evaluation at 28,311,552
new steps: 72/96 survived 25 s, speed error 0.123 m/s; per-command survival
25/32 at 0.20 m/s, 32/32 at 0.35 m/s, 15/32 at 0.50 m/s. The last periodic
evaluation regressed to about 60%; final weights lack a final-step evaluation.
Downloaded metrics/provenance; retained best separately. Walking-only contacts,
robustness and hardware transfer remain unverified. No follow-up job launched.
See [GAIT-02 results](../docs/gait02-experiment-report.md).

Repository housekeeping: added an experiment index and explicit artifact rules.
Raw traces, transfer archives and scratch previews stay local; useful metrics,
curated media and source manifests stay in Git. Existing experiment paths and
training snapshots were preserved.

### 15 October 2026 — GAIT-03 bilateral correction pilot submitted

Implemented the approved bilateral swing/alternating-step rewards, completed-cycle
symmetry and contact-point slip measurement. Retained the CAD, actor observations
and walking task. Added per-foot diagnostics and an explicit keeper requiring
11/12 nominal trials to meet survival, tracking and bilateral gait criteria.
Warm-start GAIT-02 best; 5M additional steps at 0.35 m/s, gradual shaping ramp.
Job **3219917** observed RUNNING on i25 at 1m31s, in GPU validation. Six new tests
passed locally and remotely; full local suite: 48 passed, three JAX tests skipped.
Training progress and an improved gait are not yet confirmed. No new video exists.
See the [experiment report](../docs/gait03-experiment-report.md) for the exact
changes, source hash, thresholds and next comparison.

### 15 October 2026 — GAIT-04 medium continuation submitted

User requested stronger expected-swing grounded penalties and symmetric progress
rewards, with a medium run rather than another short pilot. Submitted job **3220118**
for 15M additional steps at 0.35 m/s from GAIT-03 final. Swing-contact cost is four
times stronger; weaker-foot and missing-step costs are also increased. Added
capped partial-lift, new forward-reach and weaker-leg EMA progress rewards, with
an explicit missing-history cost. Physical model and gait acceptance stay unchanged.
Reward shaping reaches full strength after 3M steps; periodic evaluations allow
early/late comparison. Local validation: 52 passed, three JAX tests skipped;
36 uploaded source hashes verified. GPU validation/results pending. See the
[GAIT-04 report](../docs/gait04-experiment-report.md) for coefficients and provenance.

### GAIT-04 completion check

Job 3220118 completed in 1h43m02s, exit 0, at 15,007,744 steps. Downloaded all
16 gait evaluations and candidate metadata. Repeated right steps emerged late:
at 14.68M, 12/12 survived with 5?8 right steps each; final survival fell to 8/12.
No trial qualified. Candidate scoring chose 8.39M on tracking/survival despite
zero right steps, so it is not a bilateral-gait ranking. Detailed results and
this selection limitation are recorded in the GAIT-04 report. No new run or
replay launched.

### GAIT-04 video inspection

User requested visual inspection. Replay job 3220346 completed in 3m37s.
Exact 14.68M weights were not saved; recorded the nearest saved late checkpoint
at 14.22M and final 15.01M, target 0.35 m/s, seed 1000. Both fresh trials survived
25 s; valid L/R steps were 10/5 and 10/3 after warmup. Rendered two simultaneous
views with anatomical foot colors, and added an evaluation-history figure.
Videos and their limits are documented in the GAIT-04 report and
[comparison page](gait04-videos.html). No policy training or model changes.

### 15 October 2026 — GAIT-03 completed, bilateral gait still unresolved

Job 3219917 completed in 40m29s, exit 0, at 5,013,504 additional steps.
Downloaded evaluation history and final/candidate metrics. Final 12/12 survived,
speed error 0.03574 m/s versus baseline 0.05747, but 0/12 qualified throughout
the run. Every final trial recorded ten valid left steps and zero right steps
after warmup. Right clear time averaged 0.217%; flight 3.42%; maximum actuated
joint-limit overshoot averaged 0.02837 rad. The final diagnostic candidate is
not an accepted gait. Updated the [experiment report](../docs/gait03-experiment-report.md);
no training changes, new replay or continuation submitted during this status check.

### GAIT-04 completion check

Job 3220118 completed in 1h43m02s, exit 0, at 15,007,744 steps. Downloaded all
16 gait evaluations and candidate metadata. Repeated right steps emerged late:
at 14.68M, 12/12 survived with 5?8 right steps each; final survival fell to 8/12.
No trial qualified. Candidate scoring chose 8.39M on tracking/survival despite
zero right steps, so it is not a bilateral-gait ranking. Detailed results and
this selection limitation are recorded in the GAIT-04 report. No new run or
replay launched.

### 16 October 2026 ? GAIT-05 completed without stable walking

Job 3220364 completed in 1h24m56s, exit 0, at 15,007,744 steps. Downloaded the
16 evaluation records and training/tracking CSVs. Every evaluation had zero
survivors and zero qualified trials at each of 0.20/0.25/0.35 m/s. Post-warmup
step counts were zero; missing-window tracking values are fallbacks, not measured
errors. Final training episodes averaged 2.18 s. Candidate rank tied throughout,
retaining step zero. Updated the GAIT-05 report; no continuation or code changes.

### 16 October 2026 - GAIT-05 early-fall diagnosis

Diagnostic job 3221291 completed in 3m41s. Final-policy falls are torso-height
terminations (36/36), with no numerical/floor/workspace termination. At 0.25 m/s,
mean first-fall time is 2.24 s after an initial stepping attempt. The GAIT-04
control survives all paired six-second trials at the same speed. Fine-tuning
optimizer settings, a strong early shaping ramp, reward clipping and early
exploration decay are candidate learning obstacles, not isolated causal proofs.
Added an [evidence report](../docs/gait05-early-fall-diagnosis.md), learning plots,
three short fall clips and source/checkpoint provenance. No new training launched.

### 16 October 2026 - GAIT-06 fresh walking curriculum

User approved retaining asymmetry/inactive-foot/slip penalties with a gradual
success-gated increase and fresh optimization settings. Submitted job **3221307**
from the verified 40-file snapshot `gait06_20261016_v1`: 15M steps, seed 0, 0.25 m/s.
No learned state inherited. Shaping starts at 0.15; two successive evaluations
with sustained six-second movement open a 5M eligible-step ramp. Failed evaluations
pause increases. Exploration follows earned progress rather than a time deadline.
Added first-episode/fall reasons and early L/R stepping metrics, plus a diagnostic
duration tie-breaker. Local suite: 62 passed, four JAX-dependent skips; remote
launcher performs focused tests and GPU validation. No outcome claimed yet.
See [GAIT-06 report](../docs/gait06-experiment-report.md) for thresholds, limitations,
provenance and the comparison with GAIT-05. Walking-only scope remains unchanged.

### GAIT-06 completion check - job 3221307

Completed 15,007,744 steps in 1h28m23s, exit 0. Downloaded 16 evaluations and
training metrics. Final 0.25 m/s survival improved to 8/12, mean capped lifetime
22.54 s, but no qualified gait and only one valid step per foot across all trials.
Final survival at 0.20/0.35 m/s: 0/12 and 1/12. The movement gate never opened;
shaping stayed at 0.15, so progressive strengthening remains untested. Updated
the experiment report and status. No additional training launched.

### 17 October 2026 - GAIT-06 replay and proposed drag refinement

Replay job 3221442 completed in 3m04s. Saved three videos, motion traces and
checkpoint/source/video hashes. The 0.25 m/s clip survives 25 seconds, averages
0.122 m/s after five seconds, and has no valid steps in that window. Both feet
stay below the 1.5 cm whole-sole clearance threshold. Two synchronized views
identify left/right feet with blue/orange colors. See `gait06-videos.html`.

The user finds the movement promising and requests stronger discouragement of
grounded forward sliding and more difficulty. Code review confirms actual
floor-contact-point horizontal velocity is penalized, but the slip cost saturates
at 0.10 m/s and remains multiplied by the stalled 0.15 shaping scale. Recommended
next experiment: continue from the saved final policy, keep 0.25 m/s, separate
the drag-strength ramp from the valid-step gate, and preserve bilateral/clearance
rewards. First increase difficulty through cleaner stepping; broaden commands
and then perturbations after improvement. This is a proposal, not a reward-code
change or a submitted training run.

### 15 October 2026 — GAIT-05 fresh policy at 0.25 m/s submitted

User approved fresh 15M-step training at a slower target. Submitted job **3220364**
with no warm-start/resume and byte-identical GAIT-04 reward code. Evaluations cover
0.20/0.25/0.35 m/s; primary acceptance remains the unchanged gait criteria at the
0.25 m/s training target. Every evaluated checkpoint is saved, and diagnostic
ranking favors surviving bilateral progress before tracking. Retained the 4 cm
valid-step threshold after a nominal stance-displacement sanity check. Local
tests: 57 passed, three skipped; upload checksum verified. GPU validation and
results pending. Both speed and initialization change, so their effects cannot
be isolated. See the [GAIT-05 report](../docs/gait05-experiment-report.md).

### 16 October 2026 ? GAIT-05 completed without stable walking

Job 3220364 completed in 1h24m56s, exit 0, at 15,007,744 steps. Downloaded the
16 evaluation records and training/tracking CSVs. Every evaluation had zero
survivors and zero qualified trials at each of 0.20/0.25/0.35 m/s. Post-warmup
step counts were zero; missing-window tracking values are fallbacks, not measured
errors. Final training episodes averaged 2.18 s. Candidate rank tied throughout,
retaining step zero. Updated the GAIT-05 report; no continuation or code changes.

### 16 October 2026 - GAIT-05 early-fall diagnosis

Diagnostic job 3221291 completed in 3m41s. Final-policy falls are torso-height
terminations (36/36), with no numerical/floor/workspace termination. At 0.25 m/s,
mean first-fall time is 2.24 s after an initial stepping attempt. The GAIT-04
control survives all paired six-second trials at the same speed. Fine-tuning
optimizer settings, a strong early shaping ramp, reward clipping and early
exploration decay are candidate learning obstacles, not isolated causal proofs.
Added an [evidence report](../docs/gait05-early-fall-diagnosis.md), learning plots,
three short fall clips and source/checkpoint provenance. No new training launched.

### 17 October 2026 - GAIT-07 drag refinement submitted

User authorized gradually stronger all-direction sliding penalties, an independent
movement gate and retention of bilateral rewards. Submitted job **3222141** from
41-file snapshot `gait07_20261017_v1`: warm-start GAIT-06 final, 15M additional
steps at 0.25 m/s, seed 0. New contact-drag term starts at zero, ramps over 5M
eligible steps after two successful movement evaluations, and pauses on regression.
The smooth cost keeps increasing above 0.10 m/s slip. Valid-step thresholds and
gait acceptance remain unchanged; new per-foot slip metrics track improvement.
Local validation: 69 passed, five JAX-dependent skips; GPU validation precedes
training. Speed variation and then small pushes are later, performance-dependent
stages. See [experiment report](../docs/gait07-experiment-report.md) for equations,
gate thresholds, checkpoint hash, validation and limitations. At the first status
check, job RUNNING on i15; hashes verified and all 29 focused SCITAS tests passed.
GPU benchmark underway; training results pending.

### GAIT-07 completion and video review

Training job 3222141 completed 15,007,744 additional steps in 1h47m16s, exit 0.
Downloaded all 16 evaluations. Final/best candidate survives 12/12 at 0.25 m/s,
with 0.25 left / 8.5 right valid steps after five seconds and 0/12 qualified gaits.
Right-foot slip decreased, left-foot slip increased slightly. Drag reached full
strength; other gait shaping stayed at 0.15. Preserved the step-zero baseline
and documented differences from the historical GAIT-06 evaluation.

Replay job 3222197 completed in 3m13s. Three targets retained with two camera
angles; the 0.20 m/s trial falls at 19.25 s and the other two last 25 s. The
0.25 m/s replay averages 0.223 m/s and makes 0 left / 11 right valid steps after
five seconds. Added `gait07-videos.html`, a GAIT-06 comparison, progress figures,
trace metrics and video provenance. No further training launched.



### 16 October 2026 ? GAIT-05 completed without stable walking

Job 3220364 completed in 1h24m56s, exit 0, at 15,007,744 steps. Downloaded the
16 evaluation records and training/tracking CSVs. Every evaluation had zero
survivors and zero qualified trials at each of 0.20/0.25/0.35 m/s. Post-warmup
step counts were zero; missing-window tracking values are fallbacks, not measured
errors. Final training episodes averaged 2.18 s. Candidate rank tied throughout,
retaining step zero. Updated the GAIT-05 report; no continuation or code changes.

### 16 October 2026 - GAIT-05 early-fall diagnosis

Diagnostic job 3221291 completed in 3m41s. Final-policy falls are torso-height
terminations (36/36), with no numerical/floor/workspace termination. At 0.25 m/s,
mean first-fall time is 2.24 s after an initial stepping attempt. The GAIT-04
control survives all paired six-second trials at the same speed. Fine-tuning
optimizer settings, a strong early shaping ramp, reward clipping and early
exploration decay are candidate learning obstacles, not isolated causal proofs.
Added an [evidence report](../docs/gait05-early-fall-diagnosis.md), learning plots,
three short fall clips and source/checkpoint provenance. No new training launched.

### Opposite-foot sequencing reward prepared

User requested a penalty when the opposite foot does not follow a forward step.
Added an opt-in waiting timer, a cadence-based grace period, a bounded growing
cost, stronger repeated-step cost, and dense progress credit only for the expected
foot. Repeats cannot reset the deadline. Added per-foot diagnostics and tests.
Focused local suite: 34 passed, three JAX-dependent skips. Existing presets stay
opted out; no training launched and GPU validation remains pending. Details in
[the reward note](../docs/next-foot-reward.md).

### 17 October 2026 - GAIT-08 longer sequence-rule run submitted

User approved the opposite-foot rule and a longer run. Submitted job **3222272**,
42-file snapshot `gait08_20261017_v1`: 30M additional steps from GAIT-07 final at
0.25 m/s, seed 0. Preserve learned weights/statistics/std and full drag strength;
fresh optimizer at 3e-5 to 1e-5. Enable the waiting/repeat penalties and expected-foot
progress rewards. Valid-step thresholds stay unchanged. Local suite: 77 passed,
seven JAX-dependent skips. Launcher verifies hashes, runs 39 focused tests and
GPU benchmark before training. Six-hour allocation; roughly 3-4 hours execution
estimated from the previous run. See [GAIT-08 report](../docs/gait08-experiment-report.md).
First check: RUNNING on i12; hashes verified, all 39 focused SCITAS tests passed,
GPU benchmark underway. No training outcome claimed yet.

### 18 October 2026 - GAIT-08 completed

Job 3222272 completed 30,015,488 additional steps in 3h21m54s, exit 0. Downloaded
30 evaluations and training metrics. At 0.25 m/s, final survival is 12/12, valid
steps average 10.17 left / 10 right, alternation 99.2%, tracking error 0.017 m/s
and contact slip 0.0075 m/s. No fully qualified gait: all primary trials fail
cycle symmetry and right-foot clear-time fraction. Best saved candidate is
25,165,824 steps, with 10/10 steps and all primary trials passing clearance but
failing symmetry. Full gait shaping was reached by 6.29M. Updated the report and
status; no new training or replay launched during this check.

### 18 October 2026 - GAIT-08 candidate/final videos

Requested replay job 3222519 completed in 3m18s, exit 0. Recorded the exact
25,165,824-step candidate and 30,015,488-step final checkpoint at 0.25 m/s, seed
1000. Both survive 25 s and make 10/10 valid steps after five seconds. Mean
speeds: 0.245 and 0.239 m/s. Right-foot clear time: 8.8% and 7.05%. Added paired
videos with two angles and colored feet, trace metadata, provenance and a
comparison page including GAIT-07. No training changes.

## 18 October 2026 — GAIT-08 micro-step reward audit

User observed small repeated foot movements in the replay. Verified reward/config
source hashes against the training snapshot and reproduced ignored incomplete
repeat events and renewable partial-progress rewards using the actual bilateral
module. Identified the 1.0 s stance cap versus 1.3 s nominal stance at 0.5 Hz,
and mixed-event symmetry bookkeeping. Saved findings and proposed corrections in
[the audit](../docs/gait08-reward-audit.md), with a reproduction script and JSON
in `report/scripts/` and `report/tables/`. Synthetic checks establish logic, not
the physical cause or net reward benefit of the video motion. No training or
reward changes; next action is to correct and validate event accounting before
selecting weights for another run.

## 18 October 2026 — GAIT-09 submitted, job 3222597

User authorized the audited reward corrections and another run exceeding 30M
steps. Implemented an opt-in preset with incomplete/repeated whole-foot attempts,
turn-level partial-credit limits, cadence-aware stance allowance and coherent
stride history. Preserved full existing walking/drag shaping; new event costs
ramp from 25% to 100% through the walking gate. Added attempt/rejection metrics,
distributions and stricter qualification/ranking. Historical presets retain
their original reward paths.

Continue GAIT-08's 25,165,824-step candidate at 0.25 m/s for 32M additional
steps (fresh optimizer/curricula, retained weights/stats/action std). Local
focused checks: 44 passed, 5 JAX checks deferred to SCITAS; compilation passed.
Frozen 44-file archive hash verified on SCITAS; job 3222597 submitted and
confirmed PENDING (Priority). Job includes JAX checks, finite PPO benchmark
and full-strength inherited-policy reward/survival preflight before training.
No new training results yet. See [experiment report](../docs/gait09-experiment-report.md)
and [status](../docs/experiments/gait09_20261018_v1/STATUS.md).

## GAIT-09 completion — job 3222597

Confirmed COMPLETED, exit 0:0, 3h26m45s, 32,014,336 additional steps. Downloaded
final/candidate metrics, all 32 evaluations, progress CSV, job log and benchmark.
Final 0.25 m/s survival 12/12, tracking error 0.00881 m/s and contact slip
0.00578 m/s. Incomplete fraction remains 50%, with 10 repeated attempts per foot
after five seconds; no qualified training-target trial at any evaluation. Best
candidate is step 18,874,368 and also fails. Final 0.35 m/s survival is 0/12.
Updated experiment/status reports: improved tracking/slip does not establish
clean walking. Next inspect attempt timing and candidate/final video; no new run.

## GAIT-09 candidate/final video replay — job 3223177

Recorded the exact 18,874,368-step candidate and 32,014,336-step final checkpoint
at 0.25 m/s, seed 1000. Replay job completed in 3m15s, exit 0. Both survive 25 s.
After five seconds, candidate has 11/10 valid steps and 9/10 incomplete attempts;
final has 10/10 valid steps and 10/10 incomplete attempts. Both have 20 attempts
and 10 repeated attempts per foot. Added attempt/contact/clearance fields to the
replay recorder for later diagnosis; training files and checkpoints unchanged.
Verified local rendering geometry against the frozen source manifest. Prepared
two-view videos with colored feet and a comparison page with slow-motion controls.

## 18 October 2026 — GAIT-10 implementation and code cleanup

User authorized one last 15M continuation with heavy repetition penalties, then
a fresh policy with an explicit sequential objective if no progress. Selected
GAIT-09 final as parent. Event weight 2→20, bounded continuous extra-airtime cost,
protected from dense reward clipping; controller/reward phase agreement and
turn-gated partial progress. New costs ramp 50→100% over 2M eligible steps.
Preserved physical model and policy interface. Extracted event/floor helpers,
NumPy diagnostics/decision logic and reusable packaging/Slurm workflow. Kept
historical artifacts. Fixed replay access to optional fields for frozen old configs.
Local checks: 52 passed, 6 JAX checks deferred; 600 transitions per GAIT-06/08/09
matched frozen legacy state/rewards/metrics exactly. See
[GAIT-10 report](../docs/gait10-experiment-report.md) for acceptance and progress
criteria and [status](../docs/experiments/gait10_20261018_v1/STATUS.md) for submission.

Submitted **job 3223224**, confirmed running startup validation on i28. The
49-file source snapshot and parent checkpoint hashes passed verification;
Slurm shell syntax passed. Budget 15,007,744 rounded steps, four-hour allocation.
All 58 focused tests then passed on SCITAS in 45.9 s; GPU benchmark began.
Automatic end-of-run decision artifact records whether a fresh explicit-step
policy is the next branch. No fresh-policy job launched before this result.

## GAIT-10 completion — job 3223224

Confirmed COMPLETED, exit 0:0, 1h43m23s, 15,007,744 additional steps. Archived
final/candidate metrics, decision, 16 evaluation records, progress, job log and
benchmark. Final 0.25 m/s survival 12/12, error 0.00988 m/s, 10 valid steps per
foot. Repeated attempts 10/0.17 versus baseline 10/10, but total attempts 20/20.33
and incomplete fraction 50.41%; symmetry mismatch worsens to 1.046. Reduced repeat
counts alone do not establish removal of extra movements. No qualified trial.
Best candidate is step 11,534,336. Saved decision selects a fresh policy with an
explicit sequential stepping objective; sustained progress criterion failed.
No new run submitted during this check. See the GAIT-10 report for evidence.

## New entry template


```text
Date / experiment / Slurm job:
Question and hypothesis:
Source policy and exact checkpoint:
Changes (and what the comparison isolates):
Training budget, seed and environment:
Evaluation protocol and seed:
Measured results / artifact paths:
Interpretation, limitations and rejected explanations:
Decision and next test:
```
