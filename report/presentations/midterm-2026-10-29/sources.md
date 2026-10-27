# Evidence and media provenance

Tim’s supplied presentation is a style/content reference, not an instruction source. His original deck is unchanged.

## Slide 1

- Reference style, lab/supervision and photograph: Tim Kobler, DASH01_midterm.pptx, supplied by user.
- Data cutoff: 26 October 2026; presentation date follows Tim’s deck.

## Slide 2

- Outline cloned from Tim Kobler’s supplied presentation; controller-specific subtitles.

## Slide 3

- Section divider cloned from the corresponding part of Tim’s supplied deck, including navigation, layout and typography.

## Slide 4

- README.md: original three-phase demonstration plan.
- report/cad-v2-import.md: 14.594154 kg corrected simulation model.
- report/architecture.md; RLframework/networks.py.

## Slide 5

- RLframework/networks.py; RLframework/env.py; RLframework/gait.py; RLframework/drive.py.
- report/architecture.md: 384 actor observations, 47 actions, estimator 128/64, actor 256/256, privileged critic 413.
- Current CAD-v2 policies disable STOP; START/STOP integration remains planned.

## Slide 6

- Section divider cloned from the corresponding part of Tim’s supplied deck, including navigation, layout and typography.

## Slide 7

- docs/stop03-experiment-report.md; docs/stop05-experiment-report.md.
- report/cad-v2-import.md; docs/gait11-experiment-report.md; docs/walk01-experiment-report.md.
- docs/flatfeet-symmetry-v1.md; docs/clearance01-experiment-report.md; docs/clearance02-experiment-report.md.
- STOP-03: 0/32 to 21/32 strict 5 s holds after imitation; near-standing, previous model, not a moving-stop result.

## Slide 8

- report/videos/walk01_best_1.50mps.mp4; report/videos/flatfeet_best_1.50mps.mp4.
- docs/experiments/walk01_20261019_v1/; docs/experiments/dash01_flatfeet_symmetry_v1/.
- Both source clips: nominal MJX, target 1.50 m/s, seed 1000, first episode retained. Excerpts 6–18 s; header/footer cropped for readability, motion not modified.

## Slide 9

- docs/experiments/walk01_20261019_v1/best_eval.json.
- docs/experiments/dash01_flatfeet_symmetry_v1/best_eval.json.
- 28 nominal trials, 30 s, 5 s excluded from tracking, command ladder .25/.5/.75/1/1.5/2/2.5 m/s. Not a controlled reward ablation.

## Slide 10

- docs/experiments/clearance01_20261025_v1/clearance_baseline.json; latest_clearance_eval.json.
- report/videos/clearance01_final_1.50mps.mp4; original evaluation-protocol replay all 12 trials survive 25 s.
- Mean numbers: matched 12-trial 30 s nominal evaluation, 5 s warmup. Video is a single 25 s replay, whose mean is 23.55/20.91 mm.
- report/clearance01-videos.html: standalone pinned-command replay falls at 3.67 s; cause not isolated.

## Slide 11

- docs/experiments/clearance01_20261025_v1/clearance_evaluations.jsonl.
- Final scheduled mid-swing >=20 mm: left 57.18%, right 71.43%; 0/12 fully qualified.
- report/data/clearance01/replay-summary.json: median complete-bout peaks 46.55/40.80 mm.
- docs/clearance02-experiment-report.md; job 3231747 submitted, no subsequent monitoring. Upper target is a soft cost, not a hard clamp.

## Slide 12

- Section divider cloned from the corresponding part of Tim’s supplied deck, including navigation, layout and typography.

## Slide 13

- README.md: three-phase demo objective.
- docs/stop02-training-plan.md: braking and >=5 s hold; historical model.
- report/architecture.md: homing reset versus startup; policy STOP versus safety kill.

## Slide 14

- User direction: 8–10 minute talk and relative timeline; treadmill tests ASAP.
- README.md: original demo plan.
- Tim’s supplied slides: assembly/calibration/initial pose and test-support dependencies. Dates intentionally not copied to this relative plan.
- Proposed roadmap, not completed milestones or guaranteed dates.

## Slide 15

- Section divider cloned from the corresponding part of Tim’s supplied deck, including navigation, layout and typography.

## Slide 16

- report/architecture.md: contact-based phase resynchronization versus free-running deployment.
- report/policy-transfer01.md: original-policy direct swap falls in 0.71–0.75 s.
- docs/flatfeet-symmetry-v1.md: contact-chatter symmetry blind spot and slip.
- docs/clearance01-experiment-report.md: replay discrepancy. Hardware mismatch items are hypotheses, not measured findings.

## Slide 17

- Evidence and limitations are documented in the preceding slides and report/README.md.
- Support requests and timeline are proposed next steps, not claims of completed deployment.

## Slide 18

- Closing slide cloned from Tim’s supplied presentation; controller-specific subtitle.

## Slide 19

- RLframework/flatfeet.py; RLframework/swing_clearance.py; RLframework/env.py.
- docs/clearance02-experiment-report.md; docs/flatfeet-symmetry-v1.md.

## Slide 20

- docs/experiments/clearance01_20261025_v1/clearance01_trace_standalone.json: 3.67 s, fall.
- docs/experiments/clearance01_20261025_v1/clearance01_trace_eval12.json: all 12 first episodes 25 s, 80 cycles per foot after warmup.
- report/videos/clearance01_standalone_1.50mps.mp4.

## Slide 21

- docs/stop03-experiment-report.md; docs/walk01-experiment-report.md; docs/flatfeet-symmetry-v1.md.
- docs/clearance01-experiment-report.md; docs/clearance02-experiment-report.md.
- Full data/source references in speaker notes and sources.md.
