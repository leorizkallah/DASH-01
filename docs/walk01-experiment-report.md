# WALK-01: Nathann walking recipe, CAD-v2 homing, no STOP

User authorized a fresh long walking run and supplied a homing assembly.
Prepared 19 October 2026 on `dash-01-v2-speed-discovery`.

## Completed results and videos

Job **3224989** completed normally: **200,005,632 steps in 13h29m16s**.
The best saved checkpoint at 169.57M steps survived 28/28 nominal 30-second
fixed-command trials with 0.181 m/s mean tracking error. The last periodic
evaluation near 199.07M also survived 28/28, but tracking error increased to
0.328 m/s. A separate final evaluation with command redraws had no falls and
full 30-second durations in all 28 episodes. These protocols are distinct.

[Six MJX replay videos](../report/walk01-videos.html) compare best and final at
0.25, 0.75 and 1.5 m/s. Replay job **3226290 completed in 4m27s**, exit 0:0.
All six nominal 25-second trials survived. Mean achieved speeds after 5 s:
0.314/0.714/1.241 m/s (best), 0.388/0.724/1.359 m/s (final). These are greedy
fixed-command trials, seed 1000, using the frozen training environment. Local
MuJoCo rendering displays recorded MJX coordinates, with no re-simulation.
[Replay provenance](experiments/walk01_20261019_v1/replay-provenance.json)
retains checkpoint, trace and video hashes. No additional training was submitted.

Earlier [CPU deployment previews](../report/walk01-cpu-previews.html) are retained
as supplementary material; the main gallery now uses MJX recordings.
The previews retain trained weights, normalization, homing reset, gait decoder
and nominal motor delay. Local MessagePack decoding uses Flax's numeric-array
format, with original-source shapes checked and weights/statistics copied
directly. They use classic MuJoCo's XML solver settings 100/50: copying MJX's
16/8 iteration caps into classic MuJoCo produced numerical instability and
those invalid probes were discarded. Thermal evolution and the full MJX
termination logic are not reproduced, so these are qualitative previews, not
a replacement for the training evaluations. Local reader dependency: msgpack
1.2.3. Source, trace and video hashes are retained in
[preview provenance](experiments/walk01_20261019_v1/preview-provenance.json).

Cadence reaches the 4 Hz upper bound. Further learning did not consistently
improve tracking after the best checkpoint. Gait quality still needs visual
assessment; no claim of clean walking or hardware readiness follows from survival.

## Objective and retained recipe

Fresh weights, optimizer, critic, observation statistics and seed 0. Based on
`dash_joy3_lr` / `_DASH_JOY3`, independently of GAIT-01–11 and SPEED-01.
Target **200,000,000 environment steps**. This reproduces the available original
recipe with documented robot/task adaptations; it is not a claim of identical
historical training conditions or guaranteed recovery of the previous gait.

Keep Nathann's speed tracking plus forward-speed income, alive bonus, air-time,
phase/contact, slip, posture, foot-flatness, control and energy costs. Keep
adaptive learning rate, tracking-width annealing, command progression, shaping/
efficiency/stance-ratio progression and competence-paced plant randomization.
No new repeat-step, alternation, incomplete-step or bilateral reward.
All non-STOP `w_*` coefficients are regression-tested against `dash_joy3_lr`.
Frequency range remains 1.5–4 Hz. This recipe contains gait priors, not a purely
unconstrained maximum-speed objective.

Disable STOP flag/draws, zero-speed draws, standing-at-zero objective and STOP/
standing reward terms. Initial command band remains **0.375–1.125 m/s**; final
band is **0.125–2.5 m/s**, rather than approaching exact zero. Upper speeds are
training requests, not a measured CAD-v2 capability. No pushes/wind/trips, as in
Nathann's clean-day recipe. No startup/hold controller or artificial balance rail.

## Supplied homing pose

Source: `dash-01CAD-v2/homing/EXPORT_assembly_stable_closed/EXPORT_assembly-v36_closed.xml`.
All 15 mesh files match the previous CAD-v2 export byte-for-byte. Fit body poses
to that export while retaining the calibrated training plant, corrected
inertials, included motors/battery, added torso kilogram, contacts and four rigid
loop constraints. Do not import the raw export's uncorrected inertials.

Output: `RLframework/model/dash01_cad_v2_homing.xml`, keyframe `homing`.
The prior model is preserved. Pose fit: maximum body-position error **0.072 µm**.
Loading the feet onto the floor changes motor angles by at most **0.000047 rad**.
Reset torso height **0.862863 m**; total mass **14.594154 kg**.

Five-second constant-PD hold: maximum tilt **1.00°**, XY drift **12.9 mm**, height
drop **1.0 mm**, no motor saturation; maximum loop gap **2.7 µm**. This validates
a usable nominal reset, not robustness or a trained standing policy.
[Full homing audit](../report/data/homing01/audit.json).

## Explicit differences from the original preset

| Setting | Original | WALK-01 |
|---|---|---|
| Plant / reset | Old robot / stand | Calibrated CAD-v2 / supplied homing |
| STOP and standing | Enabled | Disabled |
| Final positive command band | 0–2.5 m/s, plus STOP draws | 0.125–2.5 m/s |
| Cam/thigh/roll amplitude budgets | 0.45 / 0.45 / 0.20 rad | 0.25 / 0.30 / 0.12 rad |
| Cam/thigh/hip offset budgets | 0.06 / 0.06 / 0.15 rad | 0.03 / 0.03 / 0.06 rad |
| Height floor / termination | 0.67 / 0.37 m | 0.694875 / 0.394875 m |
| Independent hinge reset noise | 0.03 rad | 0, to preserve closed loops and homing |
| Loop-site randomization | 1.5 mm | 0, to preserve rigid closure geometry |
| Parallel environments × rollout | 2048 × 9 | 512 × 36, same 18,432 samples/update |
| Solver iterations / line search | 8 / 8 | 16 / 8 |
| Evaluation | 12 s, five commands | 30 s, seven positive commands |

Height thresholds shift by the reset-height difference, preserving the original
allowed drop. Physical motor calibration and torque-speed limits stay unchanged.
Other plant/sensor randomization remains; the pose is a nominal reset, not a
promise of identical randomized dynamics. Shorter vectorization is a GPU-memory
adaptation and changes temporal rollout length; report it as such.

## Validation, selection and records

Before learning: source hashes, reward/config/pose regression tests, finite PPO
benchmark, CPU/MJX one-second physics comparison with ±0.001 rad pitch offsets,
finite random-action probe and confirmation of zero STOP ticks.

Nominal greedy evaluations every 400 rollouts (~7.37M steps), with requested
speeds 0.25/0.5/0.75/1/1.5/2/2.5 m/s, plus the original current-command-band and
short auxiliary evaluation. The existing joystick keeper scores tracking,
survival and heading; it does not certify clean walking. Review foot clearance,
slip, symmetry and joint limits from replay diagnostics before claiming a good
gait. Preserve checkpoints about every 2M steps or 15 minutes. Review videos
around 20M/60M/120M/180M; checkpoint retention permits this without retraining.

Single SCITAS GPU, 10 CPUs, 90 GB RAM, **24-hour allocation ceiling**, ten-minute
final-save reserve. Stop on reaching 200M steps or the time budget. The allocation
ceiling is not a duration prediction; if it expires first, the saved state can
be resumed in a later allocation. Final checkpoint followed by a nominal
28-episode evaluation with command redraws (separate from the pinned training
evaluation ladder). No automatic second training job is submitted.

[Run status](experiments/walk01_20261019_v1/STATUS.md).
