# STAND-04: recovery stepping and stronger randomized pushes

[Assessment, 11 November: task convergence is not established](standing04-assessment.md).
Reward has plateaued but fixed task scores fluctuate; recovery landings remain
rare and quiet-retention failures can drive impulse retreat independently of
push success. A numerical KL-diagnostic issue is also recorded for review.

**Media requested 11 November:** [gallery: videos, curves and final push levels](standing04-videos.html),
[media provenance and outcomes](standing04-media.md). All convergence plots and
strength tables are ready (PNG/PDF/SVG and CSV). Replay-only array **3248509**
is queued at the time of preparation; scheduler estimate around **19:38 CET**,
subject to change. A bounded hidden local helper automatically fetches completed
traces, renders each MP4 and updates the gallery. It requires the computer awake
and connected; [current pipeline state](data/standing04/video-pipeline.json).

Final curriculum levels: V1 **0.1327996 m/s-equivalent**, V2 **0.0200**, X-shape
**0.0200**. At the upper level with a 100 ms pulse these correspond to about
**19.06 N / 2.92 N / 2.91 N**. Actual pulse durations are randomized. Levels at
the saved best checkpoints were **0.0997743 / 0.0262144 / 0.0200**, so the
best-policy video is not a recording of the final checkpoint or its curriculum.
The fixed video challenge is 0.20 m/s-equivalent across models, with full variety.

Archived best-policy passing trials with at least one detected whole-foot landing:
V1 **0/60**, V2 **0/52**, X-shape **2/71**. Allowing recovery steps has not established
general stepping recovery. These are contact-event counts, not a visual judgment.

**Completed, checked 11 November 2026.** All three preflights and training jobs
finished successfully, exit 0:0. Every model reached **200,005,632 additional
transitions** and stopped at the step budget. No jobs remain in these arrays.
[Status data](data/standing04/status-20261111.json).

| Model | Scheduler duration | Parent baseline /80 | Saved best /80 | Final /80 | Best additional step |
|---|---:|---:|---:|---:|---:|
| Original V1 | 8h37m35s | 58 | 60 | 49 | 169.57M |
| V2 | 10h27m56s | 48 | 52 | 37 | 58.98M |
| X-shape | 10h29m37s | 60 | 71 | 66 | 132.71M |

These scores use the **new STAND-04 protocol**, including stronger test levels
and final quiet standing; compare against the parent baseline evaluated with
this same protocol, not the old STAND-03 scores. All saved best checkpoints
qualify with **16/16 quiet trials**. Finals have 13/16, 15/16 and 14/16 quiet
passes respectively, so use saved best checkpoints for further assessment.

V1 reached variety stage 3 and a maximum probed impulse level of 0.1611 m/s,
ending at 0.1328 m/s. V2 reached stage 2 and X-shape stage 1; both ended at the
0.02 m/s minimum. Thus the full 0.30 m/s training curriculum was not completed.
The final V2 probe has 97% push success but only 82.5% quiet success, blocking
promotion. X-shape's final probe has 97%/95%, but advancement requires two
independent passing batches. These metrics alone do not establish the quality
of learned recovery steps; new videos have not been recorded in this status check.
No additional training or replay was submitted.

## Original submission record

Started **10 November 2026**, Europe/Zurich. Training array **3247599** is
submitted with `afterok:3247598` and cancellation on an invalid dependency.
At submission, GPU validation array 3247598 is queued; no new training steps
have been observed. Up to three tasks may run concurrently when SCITAS permits.

| Task | Model | Model-specific parent | Additional budget |
|---|---|---:|---:|
| `3247599_0` | Original V1 | STAND-03 best 81.10M | 200M |
| `3247599_1` | V2 | STAND-03 best 73.73M | 200M |
| `3247599_2` | X-shape | STAND-03 best 29.49M | 200M |

Every continuation retains its own weights and observation normalization,
starts fresh optimizers and curriculum counters, and uses training seed 1.
Policies are not transferred across CAD geometries. Sample counters report
**additional** transitions. Same models, six motors, limits, nominal masses,
friction and full existing sensor noise. X-shape mass estimates remain provisional.
One GPU, 512 environments x36 ticks, four PPO epochs, 24 h allocation ceiling,
22.5 h training cutoff with checkpoint/final-evaluation reserve. Learning rate
starts at 3e-5 and is capped at 1e-4. No physical domain randomization is added.

## Recovery task implemented

The original planted task remains available. New `recover_v1/v2/xshape` presets
enable a separate recovery adapter and pure state machine. No actor observations
or network dimensions changed. Force/phase information is used for task scoring,
not supplied as unavailable hardware feedback to the actor.

An external pulse opens a recovery window. A successful response starts quiet
standing within **3 s of the actual pulse end**, then completes **5 continuous
quiet seconds**. The deadline cannot be extended by more steps. Quiet standing
requires low torso/motor motion, sole speed <0.02 m/s, low clearance, <=10 mm
footprint drift and <=30 mm torso drift over the streak, with both feet loaded
for >=95% of samples. Actual force-qualified contacts retain the 2 N threshold.

During recovery, the old footprint penalty becomes a weak displacement cost;
a new candidate footprint is frozen at the beginning of each attempted quiet
streak. It is committed as the reference **once after successful settling**.
After recovery, unsolicited whole-foot steps, >20 mm footprint drift or sustained
whole-foot lift end the trial. Candidate updates never move the recovery deadline.
No-push trials retain the quiet task. A 0.5 m episode displacement envelope,
0.65 m maximum foot separation and 0.12 m minimum anatomical lateral separation
prevent cumulative walking away or crossing the feet. These are simulation task
bounds, not validated hardware workspace specifications.

Foot contact hysteresis uses 2 N on / 1 N off, debounced for 20 ms. Heel/toe
rocking with continuing whole-foot load is not counted as a whole-foot step.
Every liftoff and landing is charged, even for in-place/tiny steps. Continuous
sole travel and raw contact switching remain charged below the debounce boundary.
No step-length, alternating-foot, gait-clock, swing-time or step-count bonus exists.

Shared upright, height, support, effort, smoothness and true contact-point slip
terms are retained. During recovery, flat-support cost is disabled to permit
single-foot support; quiet standing restores it. New weighted costs are:

| Contribution | Value at the 100 Hz control rate |
|---|---|
| Sole travel | -2 per metre, summed over both feet |
| Whole-foot liftoff or landing | -0.10 per event |
| Raw whole-foot contact switch | -0.005 per event |
| Extra landings | -0.30 x max(landings_this_push - 3, 0) on each landing |
| Successful recovery | +2 once per external push |
| Physical fall | -20 |
| Task violation / recovery timeout, without physical fall | -5 |

The small success bonus is for measured settling, not for stepping. Small
adjustments and repeated same-foot adjustments remain legal during recovery.
Persistent shuffling fails the fixed settling deadline; quiet-mode stepping also
fails. This mechanism still requires learned-policy video review to establish
that no useful exploitation remains.

## Curriculum and independent evaluation

Start each model at **0.10 m/s equivalent impulse**; keep 20% no-push episodes
and a 30% easy-push mixture. Hard training pushes occupy [0.9L,L]. Horizontal
azimuth already spans 360 degrees. Difficulty stages are:

0. Horizontal, torso-COM pulses, 100 ms, first pulse at 5 s.
1. Random onset 5–8 s and duration 80–150 ms.
2. Add elevation +/-10 degrees.
3. Add torso-COM-relative offsets +/-2 cm on each axis.

At each stage/level, **two independent greedy probe batches must pass** before
advancement. Each batch has 40 quiet trials and 200 near-level push trials,
25 in each of eight direction sectors. Requirements: >=90% push success,
>=80% in every sector, >=95% quiet success. Probes run every 200 updates
(3.6864M additional training transitions), with changing seeds independent of
checkpoint evaluation. Pending events receive no success credit. Probes use
20 s first episodes and the final-standing requirement below.

Successful batches introduce the next variety stage while holding impulse fixed.
After stage 3, they raise impulse by 10%, up to an initial **0.30 m/s ceiling**.
Two poor batches (<70% push success or <90% quiet success) retreat impulse by
20%, with a 0.02 m/s floor. Pulse force is capped at 60 N; the logged impulse
is the actual delivered value if clipped. At the configured masses, ceiling and
minimum duration, the requested maximum is below this cap. COM offsets and the
force cap also bound the induced moment to at most about 2.1 Nm.

For checkpoint selection, every model is evaluated under the **same full stage-3
disturbance distribution** at levels 0/.10/.15/.20/.30 m/s, 16 directions each,
full sensor noise, 20 s first episodes and fixed seed. Both quiet and pushed
trials must finish with five quiet seconds. Pushed trials additionally need a
timely completed recovery. This fixes STAND-03's differing final-quiet criteria;
new and old scores are different protocols and must not be directly equated.
Qualifying checkpoints (>=95% no-push success) take priority over unqualified
ones. The unchanged parent is also saved as `baseline` with its new-task scores.

## Validation and provenance

All **16 native/pure-logic tests pass locally**, including contact definitions,
hovering, actual slip, in-place micro-step counting, debounce-boundary chatter,
deadline expiry, one footprint commit/success, quiet-mode stepping, pulse timing,
direction coverage and quiet-regression curriculum retreat. Python compilation
passes. GPU preflight must then load each frozen parent, evaluate the new task,
run a balanced 240-trial probe and complete a **32-update / 65,536-transition**
finite PPO pilot. Pilot weights/optimizers are discarded before long training.
All three validation JSON gates must pass before any long task starts.

SCITAS **abstract JAX reset/step tracing also passes for all three models**:
state shapes and dtypes are consistent in the new mode. This performs no physics
rollout or learning and does not substitute for the queued GPU pilot checks.

Remote root: `/home/rizkalla/running_robot/experiments/stand04_20261110_v1`.
[Source/parent manifest](../docs/experiments/stand04_20261110_v1/source-manifest.json),
[submission status](../docs/experiments/stand04_20261110_v1/STATUS.md).
The frozen snapshot includes the reviewed proposal; this record describes the
implemented three-continuation contract. No training progress or learned recovery
capability is claimed at submission.
