# Figure and video guide

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
