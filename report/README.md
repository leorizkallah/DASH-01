# DASH-01 report workspace

**Posture-return objective prepared, 11 November:** [STAND-05 implementation and reference poses](standing05-pose-return.md).
Recover balance, return to each policy's validated quiet-standing posture at the
new floor location, then hold five seconds. Steps remain legal during return.
23 tests and all three abstract JAX reset/step/evaluation checks pass; no new
training submitted. **STAND-04 videos are now complete:** [gallery](standing04-videos.html).

**STAND-04 media:** [best-policy videos, convergence graphs and final push levels](standing04-videos.html),
[replay details](standing04-media.md). Graphs and strength tables are ready;
the gallery automatically adds videos from replay array **3248509** as SCITAS
finishes them. [Live local pipeline state](data/standing04/video-pipeline.json).
Final training Δv: V1 **0.1328**, V2 **0.0200**, X-shape **0.0200 m/s-equivalent**.
The videos use earlier saved best checkpoints and a common 0.20 m/s challenge.

**STAND-04 completed, checked 11 November:** [results and curriculum progress](standing04-progress.md).
All models finish 200.01M additional steps, exit 0. Best task passes: V1 **60/80**,
V2 **52/80**, X-shape **71/80**, versus same-protocol parents 58/80, 48/80, 60/80.
All best checkpoints pass 16/16 quiet trials; final policies are weaker. V1
reaches full disturbance variety; V2/X-shape end at minimum push strength.
No new videos or training submitted during this status check.

**STAND-04 submitted, 10 November:** [recovery stepping and stronger pushes](standing04-progress.md).
Array **3247599**, 200M additional transitions per model from each model's own
STAND-03 best, waits for complete GPU preflight/pilot array **3247598**. No long
training steps observed at submission. Bounded recovery, five-second settling,
continuous anti-shuffling costs, direction-balanced promotion and staged 3D
disturbances. Sixteen local tests pass; [status/source](../docs/experiments/stand04_20261110_v1/STATUS.md).

**STAND-03 completed, checked 10 November:** [three videos and convergence graphs](standing03-videos.html),
[results and metric definitions](standing03-results.md). All three models finish
200.01M transitions, exit 0. Saved best task scores: V1 **78/80**, V2 **69/80**,
X-shape **70/80**; final checkpoints are weaker. Independent replays reproduce
the same totals, with small per-level X-shape differences retained. Each video
shows fixed quiet/+X-push trials; V2's pushed trial falls at 6.50 s. PNG/PDF/SVG
plots, raw logs, selected checkpoints, trajectories and provenance are under
`report/`. No new training launched.

**New direction, 9 November:** [three-model planted-standing reward/curriculum](../docs/three-model-standing-plan.md)
for V1, V2 and the new X-shape CAD. Identical reward weights; planted feet first,
recovery steps later. [Read-only audit](data/standing03/model-audit.json),
[provisional new mass transfer](data/standing03/mass-transfer-draft.csv).
Implemented with separate corrected plants and shared reward. Six native contact
tests pass; training array **3246822** is submitted (200M/model), pending all
GPU preflights passing. V1 preflight passed; V2/X-shape remain running/queued
at submission. [Submission record](../docs/experiments/stand03_20261109_v1/STATUS.md).
[Contact-layout figure](figures/standing03_contact_layout.png): one continuous
original sole versus two separate pads on V2 and X-shape. No learned results yet.

**CLEARANCE-02 best candidate:** [video and convergence curves](clearance02-videos.html).
25 s two-camera clip at 1.50 m/s, selected 132.71M checkpoint; learning curves
mark that candidate. Fresh replay completes 9/12 trials, with three early falls,
despite archived evaluation 12/12; trial 0 is shown and all outcomes retained.
Upper-height criterion and reliable startup remain unresolved.

**CLEARANCE-02 completed, checked 2 November:** [outcome](../docs/experiments/clearance02_20261027_v2/STATUS.md)
and [experiment report](../docs/clearance02-experiment-report.md). Job 3232851
finished 200.01M additional steps in 11h08m. Final mean lift 15.82/18.20 mm at
1.50 m/s, 12/12 full-duration trials; peak heights still exceed the upper target,
so no checkpoint fully qualifies. Best unqualified candidate is at 132.71M.
No new replay or video generated during this check.

**Final presentation, latest revision:** [mechanical challenges update](presentations/midterm-2026-10-29/DASH01_controller_midterm_final_mechanical_update.pptx).
Slides 14/15 focus on Tim's hardware findings, unmodeled effects and the repair/model/test sequence;
[speaker notes](presentations/midterm-2026-10-29/speaker-notes_updated.md) updated.

**Selected midterm presentation:** [PowerPoint](presentations/midterm-2026-10-29/DASH01_controller_midterm_speaker_notes_updated.pptx)
and [updated speaker notes](presentations/midterm-2026-10-29/speaker-notes_updated.md).
Illustrated state-of-the-art slide 4 retained; notes and footer numbers follow
the 22-slide order (19 main + three hidden backups), about 9m40 planned speaking.

**Latest midterm deck:** [two state-of-the-art alternatives](presentations/midterm-2026-10-29/DASH01_controller_midterm_state_of_art_options.pptx)
at slides 4/5; choose either the illustrated research overview or Nathann's
thesis-based design comparison. Existing stopping/walking clips remain embedded.
[Notes and sources](presentations/midterm-2026-10-29/state-of-art-notes.md).

**Learning curves and step-budget evidence:** [analysis and presentation wording](learning-curves.md),
[slide-ready overview](figures/learning_curves_overview.png),
[vector PDF](figures/learning_curves_overview.pdf).
Fresh WALK-01/flat-feet 200M runs and the 50M clearance fine-tuning, with reward,
episode duration, nominal evaluation and curriculum plots. Includes source hashes
and CSV tables. Periodic `alive_frac` means no physical fall; it can include
early symmetry terminations and is not a full-duration survival measure.

**Controller midterm presentation — Tim’s template:** [PowerPoint](presentations/midterm-2026-10-29/DASH01_controller_midterm_Tim_template.pptx),
[PDF preview](presentations/midterm-2026-10-29/DASH01_controller_midterm_Tim_template.pdf),
[speaker notes and sources](presentations/midterm-2026-10-29/README.md).
18 main slides (~9m35, including outline/dividers/closing), embedded videos, three hidden backups;
architecture, experiments, demo and early treadmill/sim-to-real roadmap.

**CLEARANCE-02:** [200M-step run from the flat-feet baseline](../docs/clearance02-experiment-report.md),
[submission status](../docs/experiments/clearance02_20261026_v1/STATUS.md).
At 1.50 m/s: 10 mm minimum scheduled mid-swing height, bonus capped at 25 mm,
explicit excess-height cost above 25 mm. Symmetry, flat support and slip retained.
First attempt timed out in startup; [retry job 3232851](../docs/experiments/clearance02_20261027_v2/STATUS.md)
was submitted with bounded GPU checks and startup stage logging.

**CLEARANCE-01 completed:** [videos and foot-clearance graphs](clearance01-videos.html),
[experiment report](../docs/clearance01-experiment-report.md).
50.01M additional steps in 3h30m. Mean airborne lift **23.62/20.91 mm** left/right;
12/12 nominal trials survive. Mid-swing coverage **57%/71%** remains below the
80% criterion, so the policy is not fully qualified.

[Project summary and session handoff](../PROJECT_CONTEXT.md).

**Swing clearance baseline, focus 1.50 m/s:** [mean and peak foot lift during swing](flatfeet-clearance.md),
with left/right peak distributions, time traces and CSV data for both checkpoints.
Typical whole-foot peaks are below 10 mm; the existing clearance bonus begins at 20 mm.

**Flat-feet symmetry v1 - completed 200M steps:** [videos and graphs](flatfeet-videos.html). Best checkpoint: 28/28 full-duration trials, 0.049 m/s tracking error.

**Experiment:** [reward changes and 200M run contract](../docs/flatfeet-symmetry-v1.md),
[submission record](../docs/experiments/dash01_flatfeet_symmetry_v1/STATUS.md).

**WALK-01:** [fresh 200M-step walking reproduction](../docs/walk01-experiment-report.md)
using Nathann's joystick reward/curricula, STOP disabled, and the newly supplied
CAD-v2 homing pose. [Status](../docs/experiments/walk01_20261019_v1/STATUS.md).
Completed in 13h29m, 200.01M steps. Best 169.57M checkpoint: 28/28 survival,
0.181 m/s tracking error. [Six best/final MJX replay videos](walk01-videos.html)
at 0.25/0.75/1.50 m/s; all six 25-second trials survive.

**Nathann policy transfer:** [paired original/CAD-v2 videos](policy-transfer01-videos.html)
and [experiment report](policy-transfer01.md). Original policy survives 20 seconds
at three commands; direct CAD-v2 swap and two simple mapping probes fall within
0.71–0.75 seconds. No new training launched.

**New direction — speed-first discovery:** branch `dash-01-v2-speed-discovery`.
[Nathann's reward and branch audit](nathann-speed-reward-audit.md) identifies the
existing `dash_sprint` objective and the limitations of the older 3.06 m/s result.
[Implementation plan](../docs/speed-discovery-plan.md). GAIT-01–11 remain archived
on `dash-01-v2` at `c035774`. [SPEED-01](../docs/speed01-experiment-report.md)
implements the first fresh five-hour experiment; see its
[status](../docs/experiments/speed01_20261019_v1/STATUS.md).

**GAIT-11:** [fresh continuous alternating walking](../docs/gait11-experiment-report.md)
at 0.15 m/s, eight-hour allocation without a step cap. Expected-foot commands
replace implicit sequencing; no mandatory pause between steps. Retains bilateral
quality rewards and gradually earned sliding penalties.
[Submission and validation status](../docs/experiments/gait11_20261019_v1/STATUS.md).
Completed: 65.65M steps in 7h52m. Final training-speed survival 12/12, but
24.44% incomplete attempts and low clearance remain; no checkpoint qualified.
[Watch six candidate/final replays](gait11-videos.html) with two camera angles
and slow motion; includes 0.10/0.15/0.20 m/s and the observed failures.

**GAIT-10, job 3223224:** [last 15M-step continuation](../docs/gait10-experiment-report.md)
from GAIT-09 final, with 10× repeated-step event cost and a continuous extra-lift
cost. Includes timing/progress corrections and an explicit decision to switch to
a fresh policy if sustained walking-quality progress fails.
[Status](../docs/experiments/gait10_20261018_v1/STATUS.md): completed in 1h43m23s,
15.01M steps. Final survival 12/12 at 0.25 m/s, but 50.41% incomplete attempts
and worse symmetry. Progress criterion failed; next branch is a fresh policy
with an explicit sequential stepping objective.

**GAIT-09, job 3222597:** [32M-step micro-step correction run](../docs/gait09-experiment-report.md)
submitted on 18 October 2026 from the GAIT-08 best candidate at 0.25 m/s.
Incomplete/repeated swing costs, bounded partial-progress credit and consistent
stance timing/history. [Current status](../docs/experiments/gait09_20261018_v1/STATUS.md):
completed in 3h26m45s, 32.01M steps. Final 0.25 m/s survival is 12/12 and
tracking/slip improved, but incomplete attempts remain 50% and repetitions persist.
No evaluated policy qualified as a clean gait.
Watch the [GAIT-09 candidate/final videos](gait09-videos.html), with two camera
angles, colored feet and slow-motion controls. Both fresh replays survive 25 s.

**18 October 2026:** [GAIT-08 micro-step reward audit](../docs/gait08-reward-audit.md)
identifies incomplete-step sequencing gaps, repeatable partial-progress rewards,
and cadence/stance timing conflicts. Includes reproducible reward-unit checks;
no training changes made.

**GAIT-08, job 3222272:** [30M-step continuation](../docs/gait08-experiment-report.md)
from GAIT-07 at 0.25 m/s, with the [next-foot rule](../docs/next-foot-reward.md)
enabled and full drag strength retained. Completed in 3h21m54s: at 0.25 m/s,
12/12 final trials survived, with approximately 10 valid steps per foot and much
less slip. No trial fully qualified: cycle timing/length symmetry and right-foot
clear-time fraction remain insufficient. Best diagnostic candidate: 25.17M steps.
Watch the [GAIT-08 candidate/final comparison](gait08-videos.html): both fresh
0.25 m/s replays survive 25 seconds and make 10 valid steps per foot after five seconds.

**17 October 2026:** [GAIT-07](../docs/gait07-experiment-report.md), job **3222141**,
continues GAIT-06 at 0.25 m/s with a separate gradual penalty for actual foot-floor
sliding in every horizontal direction. Its movement gate does not require valid
steps. Existing bilateral rewards remain. Cleaner stepping comes first; varying
commands and small pushes are later stages. Completed 15M additional steps in
1h47m16s: final 0.25 m/s survival 12/12, but no qualified gait and mean valid steps
0.25 left / 8.5 right. Right-foot slip decreased; left-foot slip increased slightly.
Open the [GAIT-07 video comparison](gait07-videos.html), including all three speeds
and the preceding GAIT-06 video. The 0.25 m/s replay records 0 left / 11 right steps.

**16 October 2026:** [GAIT-06](../docs/gait06-experiment-report.md), job **3221307**,
trained a fresh 0.25 m/s policy with cold-start PPO settings and bilateral/drag
shaping gated by sustained forward movement. Completed 15M steps in 1h28m23s:
8/12 final trials survived at 0.25 m/s, but 0/12 qualified and only one valid step
per foot was recorded across those trials. The shaping gate never opened.
Watch the [current policy video](gait06-videos.html), with two camera angles and
colored feet. The 0.25 m/s replay survives 25 seconds but records no valid steps.

**New model, 14 October 2026:** [DASH-01 v2 CAD and mass-property audit](cad-v2-import.md)
documents the supplied CAD, corrected inertials, 1 kg torso allowance, mass
comparison figures and integration assumptions. The
[GAIT-01 experiment report](../docs/gait01-experiment-report.md) records the new
walking pilot, the failed fixed-target standing check and its diagnosis.
The [GAIT-02 report](../docs/gait02-experiment-report.md) records the longer
30M-step continuation, checkpoint selection and schedule changes.
Watch the [best policy at a 0.35 m/s target](videos/gait02_best_0.35mps.mp4),
or see [all three gait replay clips](../docs/gait02-experiment-report.md#video-replay--15-october-2026).
These show learned locomotion, but right-foot dragging means the gait is not accepted.
The [GAIT-03 report](../docs/gait03-experiment-report.md) records the approved
bilateral stepping rewards and gait-quality checks, submitted as job **3219917**.
It completed 5M steps: all 12 final trials survived, but none passed gait criteria;
the right foot still made no valid steps. The bilateral walking problem remains open.
The [GAIT-04 report](../docs/gait04-experiment-report.md) documents stronger
swing-contact penalties and incremental bilateral progress rewards. Job **3220118**
completed 15M additional steps. At 14.68M, all 12 trials survived and made 5-8
right steps each; final survival regressed to 8/12. No evaluation passed all gait criteria.
Watch the [GAIT-04 video comparison](gait04-videos.html): the saved 14.22M policy
and final policy, with two camera angles and colored feet.
The [GAIT-05 report](../docs/gait05-experiment-report.md) documents fresh training
at 0.25 m/s with unchanged GAIT-04 rewards, three-speed evaluation and complete
evaluated-checkpoint retention. Job **3220364** completed 15M steps in 1h24m56s;
no trial survived at any evaluated speed throughout training. Fresh training did
not establish stable walking.
The [early-fall diagnosis](../docs/gait05-early-fall-diagnosis.md) shows loss of
support after an initial step and identifies concerns with the fresh-training
setup. It includes a [0.25 m/s fall replay](videos/gait05_fall_0.25mps.mp4).
The current scope is walking; a known homing pose and separate standing work
will follow later. The STOP results
below concern the previous robot models.

Updated **12 October 2026**. This folder gathers the evidence needed for the final
report and presentations. It describes the recorded STOP-01 to STOP-03 work; no
new training was launched to build it.

**Current result:** imitation enabled a five-second quiet hold in **21/32**
near-standing trials. The best recorded STOP-03 hold checkpoint is PPO step
**524,288**. The final checkpoint survived more often but held still successfully
in only **14/32** trials. Reliable stopping from motion remains unvalidated.

Approved next experiment: [STOP-04 diagnostic and preservation pilot](data/notes/stop04-experiment-report.md),
submitted as job **3215784**. Its results are pending; the figures below remain
the recorded STOP-03 evidence.

Start with [the visual gallery](index.html) for figures and playable videos, or
use these documents:

| Document | Purpose |
|---|---|
| [Experiment story](experiments.md) | What changed, why, and what happened |
| [Current control architecture](architecture.md) | Sensors, networks, gait, motors, STOP handling |
| [Measurement and interpretation](evaluation.md) | Exact success criteria and limits of the evidence |
| [Figure and video guide](assets.md) | Captions, selection rationale, provenance and missing assets |
| [Presentation outline](presentation.md) | Eight-slide narrative and speaker notes |
| [Work log and next questions](work-log.md) | Decisions, unresolved questions and update procedure |

![Standing performance](figures/stop03_outcomes.png)

## Folder conventions

- `figures/`: plots and architecture diagram, each as PNG, SVG and PDF. Use PDF/SVG
  in the final report, PNG in slides.
- `videos/`: copies of existing simulation clips, with explicit labels in the guide.
- `data/`: copied metrics, configuration, selected code and historical notes.
  Original experiment folders remain authoritative; copies make this collection portable.
- `tables/`: generated CSV summaries for report tables.
- `scripts/build_report.py`: copies inputs, rebuilds plots/gallery/tables and records hashes.
- `manifest.json`: source paths, SHA-256 hashes, video metadata and generated outputs.

From the repository root, regenerate the collection without connecting to SCITAS:

```powershell
.\.venv\Scripts\python.exe report/scripts/build_report.py
```

The builder uses the currently downloaded metrics. Updating remote results is a
separate, deliberate step. It does not change policy weights, rewards or training
code. Authored explanations and decisions should be updated when new evidence arrives.

After a fresh Git clone, use `--frozen` to rebuild from the committed report inputs.
Raw pose traces and transfer archives are kept locally/on SCITAS; see the
[artifact policy](../docs/experiments/README.md). Curated figures and videos remain
versioned so this gallery can be opened immediately.

After editing report notes only, refresh checksums without rebuilding plots:

```powershell
.\.venv\Scripts\python.exe report/scripts/build_report.py --manifest-only
```

Latest completed results: [STOP-06 comparison](data/stop06_20261012_v1/STATUS.md).
The gallery figures cover the earlier experiments; STOP-06 plots accompany its
downloaded run records.

Raw source records: [STOP-01 log](../docs/stop-training-report.md),
[STOP-02 log](../docs/stop02-experiment-report.md),
[STOP-03 log](../docs/stop03-experiment-report.md).

## Report video storage

New WALK-01, transfer and flat-feet clips use compact H.264 encodings at their
existing gallery paths. Simulation frames, playback speed and durations are
preserved. [Encoding records](video-encoding.json) connect the compact file hashes
to the original video hashes in historical provenance records. Full-quality
originals remain local in ignored `videos/.originals/`. Re-run
`python report/scripts/compact_report_videos.py` to encode newly rendered clips.
