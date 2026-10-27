# Figure and video guide

## GAIT-11 fresh walking

[Six-video comparison with slow motion](gait11-videos.html), candidate/final at
0.10/0.15/0.20 m/s, seed 1000. Both 0.15 m/s clips survive; the final still
has extra right-foot attempts. Includes failed transfer replays, without resets.
Left foot blue, right orange; two synchronized camera views. Videos are in
`videos/gait11_*.mp4`; [progress figure](figures/gait11_progress.png) and
[SVG](figures/gait11_progress.svg) show the 64 fixed-seed evaluations.
[Provenance and hashes](../docs/experiments/gait11_20261019_v1/replay-provenance.json).
Rebuild figures/gallery with `python report/scripts/build_gait11_assets.py`.

## GAIT-09 walking comparison

[Video page with slow-motion controls](gait09-videos.html) |
[candidate](videos/gait09_best_0.25mps.mp4) |
[final](videos/gait09_final_0.25mps.mp4).
Both nominal 0.25 m/s replays last 25 seconds without falling, seed 1000.
Two simultaneous views; anatomical left foot blue, right orange. Extra incomplete
attempts persist. See the [experiment report](../docs/gait09-experiment-report.md)
for trial metrics and provenance. These are walking-policy clips, not stopping demos.

All videos are existing simulation recordings. No STOP-03 replay is available
locally in this collection. Do not present earlier clips as STOP-03 evidence.

## Figures

### Current control architecture

Current STOP-03 policy path: 384 actor observations, estimated velocity, 47 actions, gait decoding and motor feedback. Critic and privileged inputs are training-only. The runtime path is implemented; STOP-03 hardware use is not validated.

[PNG](figures/control_architecture.png) | [SVG](figures/control_architecture.svg) | [PDF](figures/control_architecture.pdf)

### Imitation enables holding; later PPO trades stillness for survival

32 repeated near-stance trials per checkpoint, evaluation seed 4102. Source 0/32, post-imitation and 524k PPO 21/32, final 14/32. Survival alone is not five-second hold success. Reward-floor fraction uses live STOP ticks.

[PNG](figures/stop03_outcomes.png) | [SVG](figures/stop03_outcomes.svg) | [PDF](figures/stop03_outcomes.pdf)

### Standing PPO with and without imitation initialization

STOP-02 and STOP-03 use the standing protocol. STOP-03 step zero here is post-imitation; STOP-02 is its source baseline. One training seed per run; baseline differences and repeated evaluations preclude a multi-seed causal claim.

[PNG](figures/standing_comparison.png) | [SVG](figures/standing_comparison.svg) | [PDF](figures/standing_comparison.pdf)

### Held-out action fitting

2,000 actor-only Adam updates. Whole demonstration environments are reserved for validation. MSE concerns action matching, not physical success; the physical gate is reported separately.

[PNG](figures/imitation_fit.png) | [SVG](figures/imitation_fit.svg) | [PDF](figures/imitation_fit.pdf)

### Walking retained relative to the source

Nonzero-command means from the walking ladder. Dashed thresholds are this run's source survival minus 0.05 and source error plus 0.10 m/s. Aggregate retention does not guarantee every speed succeeds.

[PNG](figures/walking_retention.png) | [SVG](figures/walking_retention.svg) | [PDF](figures/walking_retention.pdf)

### Every trial: longest continuous quiet interval

All 32 trials at each selected stage. Green points pass the full hold criterion; red points fail. The five-second line alone does not capture support, drift and full-window survival requirements.

[PNG](figures/quiet_periods.png) | [SVG](figures/quiet_periods.svg) | [PDF](figures/quiet_periods.pdf)

### A ramp changes the command, not the measured motion directly

Recorded STOP-01B final-policy ramp diagnostic, trial 1, 1 m/s request. Trace stops at first termination; no auto-reset trajectory is shown. This is an earlier diagnostic, not STOP-03.

[PNG](figures/ramp_trace.png) | [SVG](figures/ramp_trace.svg) | [PDF](figures/ramp_trace.pdf)

## Videos

Playback rate overlays, when present, refer to simulation time. File durations
below are wall-clock video durations. Trial-selection information is limited to
what existing records establish; historical clips are not new matched evaluations.

### Original walking policy

Historical 180M walking-source evaluation. Background locomotion capability, not a STOP-03 result; a pre-existing command-ladder recording.

[Play / copy video](videos/source_walking.mp4) (31.5 s, 50 fps).
Original: `RLframework/results/dash_joy3_lr_s2_180M_ladder.mp4`.

### Abrupt STOP failure

STOP-01B final policy, requested 1 m/s. Historical trace replay illustrating failure after STOP; not a successful hold.

[Play / copy video](videos/stop01_abrupt_failure.mp4) (10.1 s, 30 fps).
Original: `RLframework/results/stop01_replay/final_stop_1mps.mp4`.

### Smooth reference, continued failure

STOP-01B final policy with the STOP-02 command adapter, trial 1 at 1 m/s. This is a command-only diagnostic, not the trained STOP-02 standing policy.

[Play / copy video](videos/stop02_ramp_failure.mp4) (7.8 s, 30 fps).
Original: `RLframework/results/stop02_checks/final_ramp_1mps.mp4`.

### Longest recorded survival after STOP

Selected for survival, not success: trial 23, requested 2 m/s, STOP at 3.65 s, ramp 5.151 s, tip at 9.99 s. Zero quiet hold. Half-speed replay.

[Play / copy video](videos/stop02_longest_survival.mp4) (22.0 s, 30 fps).
Original: `RLframework/results/stop02_checks/longest_upright_stop_trial23.mp4`.

### Fixed nominal stance feasibility

Zero-action controller: nominal PD targets and gains, not zero torque. This illustrates a feasible teacher stance, not a learned STOP policy. Probe resets disable joint noise; corrected aggregate diagnostic is 31/32 holds.

[Play / copy video](videos/nominal_stance_teacher.mp4) (12.0 s, 30 fps).
Original: `RLframework/results/stop02_checks/stance_feasibility.mp4`.

## Evidence and provenance

### CAD v2 bilateral gait — GAIT-04

[Video comparison page](gait04-videos.html): saved 14.22M and final 15.01M policies,
both at 0.35 m/s. Two simultaneous camera angles, left foot blue/right orange,
normal speed, 30 fps. Both fresh seed-1000 replays completed 25 s; valid L/R
steps after warmup were 10/5 and 10/3. Exact 14.68M weights were not saved.
Neither policy is an accepted gait. Replay job 3220346; hashes and metadata
under `docs/experiments/gait04_20261015_v1/`.

`figures/gait04_progress.{png,svg,pdf}` shows all recorded gait evaluations,
generated by `scripts/build_gait04_assets.py`. The figure's aggregate metrics
and the individual video trials must not be conflated.

### CAD v2 learned gait — GAIT-02

- [0.20 m/s target](videos/gait02_best_0.20mps.mp4): 25 s without falling; mean speed 0.255 m/s after 5 s.
- [0.35 m/s target](videos/gait02_best_0.35mps.mp4): 25 s without falling; mean speed 0.333 m/s after 5 s.
- [0.50 m/s target](videos/gait02_best_0.50mps.mp4): termination by fall at 20.31 s; mean speed 0.377 m/s after 5 s.

All clips are 30 fps, normal speed, recorded MJX motion from checkpoint 28,311,552.
They are fresh illustrative trials with seed 1000, not the 96-trial evaluation.
All three trials are shown. Both-feet-airborne fractions are respectively
2.27%, 3.41%, 4.61%, so these are not certified walking-only gaits.
Replay job 3219901, `RLframework/tools/gait_video.py`; metadata and checkpoint
hash are in `docs/experiments/gait02_20261014_v1/best_gait_trace.json`.

`manifest.json` records hashes of copies and generated files. Metrics under
`data/` are copied from the downloaded experiment folders. Selected current
source files under `data/code/` document the architecture; original training
source manifests remain in each copied experiment directory. Historical notes
under `data/notes/` preserve their original relative links; use this report's
top-level documents and manifest to navigate the portable evidence.

Regenerate from the frozen local report inputs:

```powershell
.\.venv\Scripts\python.exe report/scripts/build_report.py --frozen
```

### GAIT-05 early-fall diagnosis

[0.25 m/s replay](videos/gait05_fall_0.25mps.mp4) shows a representative final-policy
fall at 2.25 s, ending at the torso-height termination before auto-reset.
Companion clips at 0.20/0.35 m/s end at 1.22/1.51 s. Left foot blue, right orange,
two simultaneous views, 1x speed. `figures/gait05_learning_diagnosis.{png,svg,pdf}`
plots training survival, reward-floor frequency and shaping/exploration schedules.
See the [diagnosis](../docs/gait05-early-fall-diagnosis.md) for uncertainty and provenance.

### GAIT-06 final-policy videos

[Video page](gait06-videos.html), or [0.25 m/s clip](videos/gait06_final_0.25mps.mp4).
Final checkpoint 15,007,744; seed 1000; replay job 3221442. Three targets retained:
0.20 m/s (4.34 s, fall), 0.25 m/s (25 s, no fall), 0.35 m/s (11.21 s, fall).
Two simultaneous views, 30 fps, normal speed, blue left/orange right foot. No
valid steps after five seconds at the training target. Trace, checkpoint hash and
rendering provenance are stored in `docs/experiments/gait06_20261016_v1/`.

### GAIT-07 final-policy videos and progress

[Video comparison](gait07-videos.html): final/best candidate at 0.20, 0.25 and
0.35 m/s, plus GAIT-06 at 0.25 m/s. Replay job 3222197, seed 1000, normal speed,
30 fps, two simultaneous angles with blue left/orange right foot. The 0.20 m/s
clip falls at 19.25 s; the other two last 25 s. At the training target, valid
steps after five seconds are 0 left / 11 right. Metadata, trace and hashes are in
`docs/experiments/gait07_20261017_v1/`. Training figures:
`figures/gait07_progress.{png,svg,pdf}`, rebuilt by `scripts/build_gait07_assets.py`.

### GAIT-08 paired policy videos

[Comparison page](gait08-videos.html): the 25.17M candidate and 30.02M final
checkpoint at 0.25 m/s, plus the earlier GAIT-07 clip for reference. Replay job
3222519, seed 1000, both trials 25 seconds without falling and 10/10 valid L/R
steps after five seconds. Two simultaneous angles, 30 fps, normal speed, blue
left/orange right foot. Source/checkpoint/video hashes and traces are under
`docs/experiments/gait08_20261017_v1/`. These are individual illustrative replays,
not the multi-trial gait evaluation.


### CLEARANCE-01 and controller midterm (27 October cleanup)

[Clearance gallery](clearance01-videos.html) retains the successful evaluation
replay and standalone startup failure. The compact encodings preserve duration,
frame order and playback speed. Full-quality originals remain locally in
`videos/.originals/`; original and encoded hashes are in `video-encoding.json`.
Historical replay manifests still describe their original bytes; do not rewrite
those manifests to match a later display encoding.

[Midterm presentation](presentations/midterm-2026-10-29/README.md): the current
Tim-template PPTX/PDF, embedded-clip assets, notes and provenance are versioned.
The superseded first draft, Office lock file and regenerable previews stay local.
Install report tooling using `pip install -r report/requirements.txt`.
