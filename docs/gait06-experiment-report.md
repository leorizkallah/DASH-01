# GAIT-06: fresh walking with success-gated gait shaping

16 October 2026. SCITAS job **3221307**, snapshot `gait06_20261016_v1`.
Job **COMPLETED** in **1h28m23s**, exit 0, at **15,007,744 steps**.
See [live record](experiments/gait06_20261016_v1/STATUS.md) and
[source manifest](experiments/gait06_20261016_v1/source-manifest.json).

## Motivation and hypothesis

[GAIT-05 diagnosis](gait05-early-fall-diagnosis.md) found falls during the first
support transitions, rather than numerical failure or an inability to move the
right foot. Its fresh weights were trained with continuation-oriented PPO settings,
strong early shaping, and exploration reduction before sustained walking emerged.
Those are candidate learning obstacles, not individually proven causes.

The user approved a fresh attempt with cold-start optimization, retaining penalties
for asymmetry and dragging but increasing them only after demonstrated movement.
The hypothesis is that mild initial shaping permits support and forward movement
to develop, while earned increases subsequently improve bilateral gait quality.

## Setup and changes

Fresh network, optimizer, observation statistics and exploration; no source policy,
warm-start, resume or imitation. The launcher rejects an existing run directory.
One seed (0), 15M environment steps, 512 environments, 32-step rollouts, fixed
0.25 m/s target, same corrected CAD plant and six motors. Walking only; homing and
separate standing training remain deferred.

| Setting | GAIT-05 | GAIT-06 |
|---|---|---|
| Learning-rate schedule | 5e-5 to 1e-5; 100-update warmup | 3e-4 to 1e-4; no warmup |
| Initial shaping scale | 0.50 | 0.15 |
| Shaping progression | Clock to full strength in 3M steps | Five million **eligible** steps; pauses when evaluations fail |
| Entropy coefficient | 0.003 to 0.0005; deadline at 6M | 0.010 to 0.002, tied to earned shaping progress |
| Final action-standard-deviation cap | 0.25 | 0.35, tied to the same progress |
| Evaluation before five seconds | Not reported in gait-quality summaries | First-episode duration, early speed/error, L/R steps and termination flags |

The existing bilateral reward formulas are unchanged. All their terms start at
15% of full strength (including positive progress rewards): weaker-foot swing
deficit, missing steps, unequal cycle timing/step length, nonalternating steps,
ground contact during expected swing, insufficient clearance and contact slip.
Alternating valid steps, lifting, forward swing progress and weaker-foot progress
retain positive rewards. A foot must clear 1.5 cm, spend at least 0.08 s in swing,
and advance at least 4 cm relative to the body for a valid step. This shapes
functional symmetry; it does not force mirrored joint actions.

The existing `shape_scale` also scales other gait/posture shaping terms. This is
a coordinated shaping ramp, not an isolated change to the slip coefficient.
Physical limits, fall termination/penalty, speed objective and reward floor remain
unchanged. Training reward-floor incidence remains an explicit diagnostic.

## When penalties increase

Every 64 rollouts (1,048,576 environment steps), evaluate greedy actions at
0.20, 0.25 and 0.35 m/s. Only the 12 nominal trials at **0.25 m/s** control the ramp.
A trial passes the initial-movement gate if:

- Its first episode lasts at least six seconds.
- Over its first six seconds, mean forward speed is at least 0.125 m/s and mean
  absolute speed error is at most 0.15 m/s.
- It completes at least two valid steps total during that window.

At least half the trials must pass in **two consecutive evaluations**. Subsequent
training steps then advance shaping linearly from 0.15 to 1 over 5M eligible
steps. Any failing evaluation immediately pauses further increases; two successive
passes are required to resume. Already-earned strength is retained. Duplicate
evaluation records do not count twice. Gate state and progress are saved in the
existing checkpoint JSON sidecars.

Standing, sliding without any valid steps, and early falls cannot open this gate.
It deliberately permits unequal stepping initially: requiring a perfect gait
before increasing gait-shaping rewards would make the curriculum circular.
Six seconds is a preliminary competence threshold, not gait acceptance. This is
a fixed-speed nominal experiment; these thresholds are not calibrated for other
targets. Evaluation trials are not independent hardware robustness trials.

Exploration reduction follows the same earned progress, without an elapsed-step
deadline. If walking never emerges, shaping remains at 0.15 and entropy at 0.01;
the run ends at its budget without automatically extending or relaxing the gate.
Periodic checking means regression can take up to one evaluation interval to pause
the ramp. A pause does not automatically undo increased penalty strength.

## Evaluation and checkpoint interpretation

Keep the existing 25-second survival and post-five-second bilateral qualification
criteria unchanged. Report first-episode elapsed time (censored at 25 seconds if
still alive), first-six-second mean speed/error, step counts both from reset and
after five seconds, and height/tilt/floor/workspace/nonfinite termination flags.
Mask auto-reset episodes out of these measurements.

Every evaluated checkpoint is retained. Diagnostic candidate rank v3 prioritizes
qualification, surviving bilateral progress, full survival, then first-episode
duration and tracking. The added duration tie-breaker avoids keeping the random
initial policy simply because all attempts fell before five seconds. A best
candidate is still not a certified gait. Evaluation reward-floor statistics use
full evaluation shaping; use training CSV values to assess the active curriculum.

## Validation and provenance

Local suite: **66 tests, 62 passed, four skipped for missing JAX**. Focused tests
cover nonwalking rejection, consecutive success, duplicate evaluations, gradual
progress, pausing/resuming, checkpoint-state serialization, first-episode masking,
early candidate selection and unchanged bilateral behavior. The SCITAS launcher
runs the focused suite including PPO integration, followed by a finite-state GPU
rollout/PPO benchmark, before starting training.

The uploaded 40-file source bundle was SHA-256 verified before submission.
Archive hash: `7081621e943592ed07093541f6840cef5afa34143e8d96d2e0231888990fc168`.
Resources: one GPU, 10 CPUs, 90 GB RAM, four-hour limit. Run name
`cad_v2_gait06_s0`; remote root
`/home/rizkalla/running_robot/experiments/gait06_20261016_v1`.

This comparison changes optimization, shaping and exploration together. Improvement
would support the combined approach, not establish which component caused it.
Fifteen million steps are an observation budget, not a convergence guarantee.
## Completed results

Downloaded all 16 gait evaluations, the latest and best-candidate summaries, and
training CSV into the snapshot directory. Final nominal greedy evaluation:

| Target | Survived 25 s | Qualified | Mean first-episode duration | Valid L/R steps pooled across 12 trials |
|---|---|---|---|---|
| 0.20 m/s | 0/12 | 0/12 | 4.29 s | 0 / 0 |
| 0.25 m/s | 8/12 | 0/12 | 22.54 s | 1 / 1 |
| 0.35 m/s | 1/12 | 0/12 | 12.36 s | 0 / 7 |

Duration is capped at 25 s for survivors; it is not an uncensored mean fall time.
At the training target, mean post-five-second tracking error was 0.131 m/s.
All four failed 0.25 m/s trials terminated on low torso height. Zero valid steps
does not imply literally motionless feet: steps must satisfy the clearance,
air-time and travel thresholds. A replay is needed to distinguish shuffling,
sliding and other strategies visually.

No evaluation qualified a gait. Initial-movement gate success peaked at 1/12,
below the required 6/12, and never opened. Shaping stayed at 0.15; exploration
entropy stayed at 0.01. Thus this run tested the mild-shaping learning phase;
it provides no evidence yet for the later progressive drag increase.

The final training row reports mean episode length 581.25 ticks (5.81 s), mean
forward speed 0.198 m/s and reward-floor incidence 3.19%. These stochastic training
metrics differ from the nominal greedy evaluations. Evaluation floor incidence
at 0.25 m/s was 54.5% under full evaluation shaping and is not the active training
floor rate.

The final evaluated checkpoint is also the best diagnostic candidate. Compared
with GAIT-05's zero survivors and roughly 2.24 s final greedy lifetime at 0.25 m/s,
survival has improved markedly, but the near-absence of qualifying steps means
this is not yet an acceptable walking solution. No continuation launched during
this status check; inspect a replay before choosing the next change.

## Final-policy video replay

Replay job **3221442** completed in 3m04s (exit 0). Open the
[video page](../report/gait06-videos.html), or the
[0.25 m/s video](../report/videos/gait06_final_0.25mps.mp4).
Two simultaneous camera angles, 30 fps, normal speed; left foot blue, right orange.
Colors affect rendering only. Local XML and all 15 meshes match the training
source manifest. These are fresh individual nominal trials with seed 1000,
not a reconstruction of the 12-trial evaluations.

| Target | Replay duration | Outcome | Mean speed after 5 s | Valid L/R steps after 5 s |
|---|---|---|---|---|
| 0.20 m/s | 4.34 s | Fell | No measurement window | 0 / 0 |
| 0.25 m/s | 25.00 s | Survived | 0.122 m/s | 0 / 0 |
| 0.35 m/s | 11.21 s | Fell | 0.281 m/s | 0 / 0 |

At 0.25 m/s, neither foot met the 1.5 cm whole-sole clearance threshold after
five seconds. This illustrative trial supports the concern that survival and
forward movement have improved without proper stepping. Videos stop at the first
termination; auto-resets are omitted. See [replay metrics](experiments/gait06_20261016_v1/gait06_trace.json)
and [provenance](experiments/gait06_20261016_v1/replay-provenance.json).

## Proposed next test - 17 October 2026

The user finds the 0.25 m/s movement promising but wants forward foot dragging
penalized more strongly and difficulty increased. The current slip input is
horizontal velocity of actual foot-floor contact points in world coordinates,
not foot velocity relative to the moving torso. This distinction preserves
legitimate stance and foot rolling while charging for sliding on the ground.

Current bilateral slip cost is `-0.4 * mean(min((slip/0.1)^2, 1))`, before the
shared shaping scale and environment reward normalization. At scale 0.15 it
remains mild, and larger slips beyond 0.10 m/s receive no additional cost. The
movement gate also requires valid steps before allowing stronger shaping, which
can hold a sliding policy at the starting scale.

Recommended controlled continuation from GAIT-06 final: retain the training
target and bilateral/clearance rewards, strengthen contact-slip shaping gradually
with its own survival-and-forward-movement gate, and use a smooth cost that still
distinguishes moderate from large slip speeds. Do not require already-clean
steps to open that specific ramp. Retain overall reward-floor diagnostics so a
larger cost does not merely erase distinctions through clipping. Increased
penalties encourage behavior; they do not impose a hard no-slip constraint.

Compare against the saved baseline using survival, speed tracking, each foot's
valid steps and clearance, and contact-slip distance/fraction. Cleaner gait at
the present speed is the first difficulty increase. Only then widen target speeds
gradually, followed by small perturbations. No new reward implementation or
training job has been submitted for this proposal.
