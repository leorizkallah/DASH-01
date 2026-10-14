# Learning to stop on command: experiment report

This is a living experiment record for the final project report. Dates and times
use Europe/Zurich. Planned outcomes are distinguished from measured results.

2026-10-13: STOP-06 completed both full 5,046,272-step budgets normally on
12 October (timed 21:52, feedback 23:54 Zurich), with no successful moving stops
in any recorded evaluation. Final standing holds: timed 32/32, feedback 30/32.
Final walking retention passed only for timed. Both selected keepers remain the
source, which is also what the automatic multi-seed assessments/videos evaluate.
Metrics and plots downloaded; see [STOP-06 outcome](experiments/stop06_20261012_v1/STATUS.md).

2026-10-12: user approved [STOP-06](stop06-experiment-report.md), a matched comparison
of timed and feedback HOLD transitions, each with up to five million new steps.
Both arms retain rehearsal and strict keeper selection, use a longer settling
window and permit temporary retention regressions through a grace/patience guard.
Settling-time diagnostics, graphs, three-seed assessment and replay are automated.
Submission observations are recorded in [STOP-06 status](experiments/stop06_20261012_v1/STATUS.md).
Array **3216469** was submitted: timed arm running on `i03`, feedback arm queued
behind it at 19:50:47 Europe/Zurich. All 33 SCITAS preflight tests passed.

2026-10-12, 19:21 status check: STOP-05 job 3216384 ended at 19:14:28 after
28m03s. The preservation guard stopped PPO at 262,144 steps because walking and
standing retention failed; moving-stop success remained 0/32. The selected keeper
therefore remains the source policy. Three-seed assessment and replay rendering
completed; no moving-stop successes were observed and no follow-on job was launched.
See [STOP-05 outcome and evidence](experiments/stop05_20261012_v1/STATUS.md).

2026-10-12: approved [STOP-05 low-speed moving-stop pilot](stop05-experiment-report.md)
from the STOP-04 262,144-step keeper. Train alternating RUN/STOP at 0.25–0.5 m/s
with continued standing/walking rehearsal, separate standing preservation checks,
and the unchanged five-second quiet hold requirement. The final-evaluation guard
gap is fixed. Submission and outcomes: [STOP-05 status](experiments/stop05_20261012_v1/STATUS.md).
Job **3216384** was confirmed running on `i35` at 18:46:42 Europe/Zurich.
All 25 SCITAS preflight tests passed. Training results are pending.

2026-10-12: the [report workspace](../report/README.md) now collects the experiment
story, current control architecture, seven figures, five labeled simulation videos,
evaluation criteria, presentation outline, copied evidence and a rebuild script.
Use the [local visual gallery](../report/index.html) for reviewing figures and clips.
The collection preserves the distinction between near-stance HOLD results and
unvalidated stopping from motion; a STOP-03 replay video is still missing locally.

## Objective and starting point

Teach the existing flat-foot DASH-01 walker to transition from walking to a stable
standing state after an explicit STOP command, then resume walking on RUN.
Preserve its existing walking ability and deployable observation/action interface.

Source policy: `controller/deploy/bundles/dash_joy3_lr_s2_180M.npz`, source run
`dash_joy3_lr_s2`, exported from `ckpt_180486144.msgpack` (180,486,144 training steps).
The actor runs at 100 Hz, with 384 observation values and 47 action values: 41
cycle-latched gait parameters and six joint corrections updated every tick.
The velocity estimator and actor are deployed; the value critic is training-only.

## Experiment STOP-01 — 2026-10-02

Status: STOP-01A was stopped after its baseline revealed reward-floor saturation.
STOP-01B completed normally on SCITAS as job `3200768`, but did not demonstrate
successful stopping in any recorded evaluation. Status checked on 2026-10-05.
No policy updates were performed by STOP-01A; four critic-only fitting rollouts
completed before cancellation.

Completed run (STOP-01B):

- Job `3200768`, started 2026-10-02 16:53:51 Europe/Zurich on `i14`.
- Finished 2026-10-02 18:16:19 Europe/Zurich; elapsed 1 h 22 min 28 s;
  Slurm state `COMPLETED`, exit code `0:0`.
- Total PPO steps: 20,054,016 (306 full rollouts, slightly above the 20M target).
- Final checkpoint exists: `RLframework/runs/dash_stop01b_s0/final.msgpack` (5.2 MB).
- One V100 32 GB, 10 CPU cores, 90 GB RAM; time limit 4 h 30 min (21:23:51).
- Snapshot `/home/rizkalla/running_robot_experiments/stop01_20261002_v2`.
- Run `RLframework/runs/dash_stop01b_s0`; log `RLframework/logs/dash-stop01-3200768.out`.
- Archive SHA-256: `9f6ba2575401f4feeee5d0e54108ba0481e84c8b2bc8b37d0b3377bed9f395b9`.
- [STOP-01B source manifest](stop01b-source-manifest.json).
- Seven tests passed on the cluster; actor/estimator restoration parity passed.
- Full GPU stepping preflight passed. The reward-budget guard passed; critic
  initialization completed (16 rollouts, 1,048,576 samples), and `ckpt_0.msgpack`
  was saved. PPO subsequently completed its full pilot budget.

### Outcome reviewed on 2026-10-05

All nine post-update transition evaluations scored **0/32 successful stops**,
**0/32 surviving the full STOP interval**, and **0/32 successful restarts**.
This is a failure of the pilot's stopping objective despite a normal training-job
completion. The same fixed trials were repeated at each checkpoint; they are not
independent samples that should be pooled into a larger success-rate estimate.

The latest transition evaluation during training was at **18,874,368 steps**.
The final 20,054,016-step checkpoint was saved without an additional evaluation
during training. A separate recorded evaluation on 2026-10-05 also scored 0/32
successful stops and 0/32 surviving STOP; details and video are below.
`best_stop.msgpack` remains the step-zero baseline because no evaluated candidate
improved its stopping ranking.

At the latest evaluation, RUN-only mean tracking error improved from 0.6970 to
0.3247 m/s; mean survival across nonzero command rungs was 67.26% at both baseline
and the latest evaluation. Retention passed at seven of nine post-update evaluations
and failed at 6.29M and 12.58M steps. Improved walking tracking did not transfer to
a successful stopping transition.

The latest measured live-STOP reward-floor fraction was 25.0%, versus 25.27% at
the revised baseline. The final training reward was 1.9177 per tick and KL was
0.002520; these optimization metrics do not establish stopping competence.

Next investigation: inspect representative STOP trajectories and termination
reasons, including falls before STOP and during braking/holding, before choosing
another reward or curriculum change. No additional training job was submitted
during this status check.

### Final-checkpoint motion replay and diagnosis — 2026-10-05

Diagnostic Slurm job `3203602` completed successfully on `i06` in 2 min 38 s
(exit `0:0`). It loaded `final.msgpack` at 20,054,016 steps from the v2 snapshot;
`best_stop.msgpack` was deliberately not used because it remains the step-zero
baseline. No new training was performed. A separate NPZ export of the final actor
was saved for local interactive simulation; the deployed bundle was not replaced.

The diagnostic records the existing nominal MJX transition protocol: 32 trials,
seed 4102, four speeds (0.5/1/1.5/2 m/s), STOP between 3.00 and 3.91 s, eight
seconds of requested STOP, then restart. Command state and actor input are both
pinned to the protocol. Domain randomization, bring-up, jitter and command drops
are disabled. Recorded poses come directly from MJX. Local video rendering does
not re-simulate the physics. Analysis and videos stop at the first termination,
so subsequent automatic resets cannot masquerade as recovery.

Measured final-checkpoint results:

- 26/32 qualifying moving approaches; 0/32 successful stops, full-STOP survivors,
  or successful restarts.
- Two trials terminated before STOP, both for foot-floor penetration. The other
  30 terminated 0.25–1.88 s after STOP (median 1.04 s).
- First termination reasons across all 32 trials: excessive tilt 24, foot-floor
  penetration 4, workspace violation 3, low base height 1, non-finite state 0.
  Thus the result is not solely a stringent quiet-standing metric: every episode
  actually reaches an environment termination condition.
- Trial 1 (requested walking speed 1 m/s): mean approach speed 0.644 m/s; STOP at
  3.00 s; tilt termination at 4.06 s (60-degree threshold). Horizontal speed is
  approximately 0.67 m/s just before STOP, 0.59 m/s at 3.25 s, then 1.88 m/s at
  termination as the robot tips. The last rendered pre-termination pose has
  57.5 degrees of tilt. Video freezes that pose; it does not show a floor impact.
- This trial changes its latched gait coefficients after STOP, but its gait clock
  stays at 4 Hz through most of the transition. Reducing the oscillatory motion
  has not produced a stable balance controller. Nonzero clock frequency alone
  does not make standing impossible: zero oscillatory coefficients can produce
  a stationary reference even with the clock running.
- Trial 1 live-STOP mean unscaled braking income is 1.3555; holding income is
  only 1.2e-9. Pose/joint-velocity cost averages -0.0337 and gait-amplitude cost
  -0.0333. Across all live STOP ticks before first termination, holding income
  averages 0.00533 (configured maximum 3) and 27.66% hit the reward floor.
  These are per-tick, pre-fall measurements, not episode returns.

Interpretation: STOP is an input to the learned controller, not an automatic
motor brake or a built-in standing controller. Stopping requires placing the
feet to arrest momentum and then actively balancing. The holding reward is
multiplicative in speed, angular velocity, joint velocity, posture and support;
it becomes very small when several conditions are poor. The trace demonstrates
that the final policy rarely visits a rewarding quiet stance. This supports a
learning/exploration bottleneck, but does not establish a unique cause of failed
convergence. Neither these trajectories nor higher average training reward show
that more of the same training will solve it.

Recommended next experiment (not yet implemented): first verify an attainable
balanced stance using this plant/action interface, then train STOP balance from
near-stance initial states with dense, separately inspectable posture, support,
speed and joint-velocity signals. Gradually introduce slow walking-to-STOP
transitions and increase speed once holding succeeds, while retaining RUN trials
and the existing walking-retention check. Test reward/curriculum changes as
separate ablations before assigning causality.

Artifacts in `RLframework/results/stop01_replay/`:

- [Half-speed video, trial 1](../RLframework/results/stop01_replay/final_stop_1mps.mp4)
- [Speed, tilt, joint motion and reward plot](../RLframework/results/stop01_replay/final_stop_1mps_diagnostics.png)
- [Per-trial diagnostics and provenance](../RLframework/results/stop01_replay/final_stop_trace.json)
- `final_stop_trace.npz`: recorded poses, actions, gait parameters, physical
  diagnostics, reward components and metadata for all 32 trials.
- `dash_stop01b_final.npz`: exported final actor for the interactive CPU player.

Reproduce the video or inspect a recorded trial interactively:

```powershell
.\.venv\Scripts\python.exe RLframework/tools/replay_stop_trace.py --trace RLframework/results/stop01_replay/final_stop_trace.npz --env 1 --video RLframework/results/stop01_replay/final_stop_1mps.mp4
.\.venv\Scripts\python.exe RLframework/tools/replay_stop_trace.py --trace RLframework/results/stop01_replay/final_stop_trace.npz --env 1 --viewer
```

Recorded viewer: Space pauses, R replays, Q closes; default playback is half speed.
To control the exported final policy in a fresh local CPU simulation:

```powershell
.\.venv\Scripts\python.exe RLframework/tools/play_joystick.py --bundle RLframework/results/stop01_replay/dash_stop01b_final.npz --start-stick 0.4
```

Live player: X toggles RUN/STOP; Space only sets the stick to zero and does not
assert STOP. The CPU player and its reset conditions differ from the MJX training
environment, so use the recorded MJX trajectory for the diagnosis above.

Replay tooling also fixes a stale body-name lookup in `play_joystick.py`: resolve
`torso` explicitly and reject a missing body instead of silently indexing body
-1. In the current XML that last body is a torso-aligned motor housing, so its
orientation happened to match; this lookup was not the cause of the observed
STOP failures. A regression test rotates that housing to verify the sensor
frame remains the torso frame. Eight local unit tests pass; replay and recording
scripts compile, and the real SCITAS recording and local MP4 rendering complete.

Current submission command, from the v2 snapshot:

```bash
sbatch --parsable --export=ALL,REPO=/home/rizkalla/running_robot_experiments/stop01_20261002_v2 RLframework/slurm/izar_stop.sbatch
```

Initial diagnostic run (STOP-01A, cancelled):

- Slurm job: `3200744`, submitted/started 2026-10-02 16:38:02 Europe/Zurich.
- Node: `i14`; GPU: NVIDIA Tesla V100 PCIe 32 GB; 10 CPU cores; 90 GB RAM.
- Time limit: 4 h 30 min (scheduled limit 21:08:02); one seed (`0`).
- Snapshot: `/home/rizkalla/running_robot_experiments/stop01_20261002_v1`.
- Run: `RLframework/runs/dash_stop01_s0` inside that snapshot.
- Slurm log: `RLframework/logs/dash-stop01-3200744.out` inside that snapshot.
- Archive SHA-256: `208298aa9eabb67ebf22471037f8d3728cb467938993ea638ee7bbeef1338bda`.
- Local source manifest: [stop01-source-manifest.json](stop01-source-manifest.json).

Exact submission, from the snapshot directory:

```bash
sbatch --parsable --export=ALL,REPO=/home/rizkalla/running_robot_experiments/stop01_20261002_v1 RLframework/slurm/izar_stop.sbatch
```

### Hypothesis

The existing stopping objective already penalizes speed, gait amplitude and joint
motion. A positive reward for supported stillness, less restrictive posture shaping
during braking, and longer STOP intervals may teach the missing transition and hold.
This is a combined pilot, not an ablation proving which individual change matters.

### Changes relative to `dash_joy3_lr`

| Setting | Previous preset | STOP-01 | Reason |
|---|---:|---:|---|
| Supported hold reward weight | 0 | 3 | Make quiet standing explicitly worthwhile |
| STOP speed penalty weight | 10 | 0 (initial attempt: 4) | Initial attempt still clipped 79.7% of STOP ticks |
| Positive braking income | 0 | 4 × upright_gate / (1 + speed/0.5) | Keep a useful preference for slowing down above zero speed |
| Standing pose penalty weight | 5 | 2 | Allow a range of balanced stopping configurations |
| Minimum standing-penalty fade | 0.25 | 0.05 | Allow recovery steps while still moving |
| Joint-speed / gait-amplitude weights | 0.02 / 10 | unchanged | Discourage persistent shuffling once slow |
| STOP duration per draw | 2.4–5.6 s | 6–10 s | Practise holding after the transient |
| STOP probability per draw | ramp toward 0.25 | fixed 0.25 | Practise stopping from the first rollout |
| RUN commands | curriculum | 0.125–2.125 m/s | Broad walking practice near the source command range |
| Plant randomization | curriculum toward 0.5 | disabled for pilot | Establish the transition first |
| Observation noise / nominal delay | enabled / source model | retained | Preserve sensor/actuation conditions |
| Observation normalization | updated during PPO | frozen from bundle | Preserve the pretrained input mapping |
| Learning rate | 3e-4 to 1e-4, adaptive | 5e-5 to 1e-5, 1,000-update warmup | Conservative fine-tuning |
| PPO rollout | 2,048 × 9 in preset | 1,024 × 64 | Longer continuous trajectories per rollout |
| Budget | 200 million steps | 20 million PPO steps | One bounded pilot before expanding scope |

The existing adaptive-learning-rate branch bypasses the minibatch warmup schedule;
STOP-01 uses the non-adaptive schedule so the requested warmup is actually applied.
The velocity estimator's learning rate remains the inherited value (1e-3).
The initial exploration standard deviation is the source bundle's mean (~0.1693),
uniform across actions because the bundle does not store each action's standard deviation.

Cold-start curricula are disabled. Tracking width stays 0.3 m/s, shaping scale stays
1, and the source checkpoint's efficiency scale (0.5001216) and stance ratio
(0.534972032) are retained. The locomotion network and control law are unchanged.
At mean RUN/STOP durations of 4/8 s and a STOP draw probability of 0.25, about 40%
of command time would be STOP in uninterrupted episodes; falls/resets change the observed share.

### Added reward

For STOP ticks only:

`r_hold = 3 × support × height_ok × clip((upright − 0.9)/0.1, 0, 1)`

`         × exp(−(v_xy/0.15)² − (|gyro|/0.5)² − (joint_speed_RMS/1.0)²)`

`v_xy` is world-horizontal speed in m/s; angular and joint speeds are in rad/s.
`upright` is the cosine of tilt relative to vertical. Support uses the environment's
both-feet-grounded predicate (contact during the control interval or its foot-height
tolerance). Height must exceed the existing fall threshold. This bonus is paid
continuously; there is no termination or repeated threshold-crossing bonus.
Existing reward scaling, clipping and fall penalties remain in force.
The transition evaluation records STOP reward-floor frequency and reward components.

STOP-01B additionally pays `4 × clip((upright−0.5)/0.5, 0, 1) × height_ok /
(1 + v_xy/0.5)` on STOP ticks. It replaces the negative speed bill rather than
increasing it. The hold bonus still distinguishes quiet standing from moving in
place. A pre-training guard rejects a configuration if more than 50% of live STOP
ticks hit the total reward floor. This threshold is an experiment guard, not a
theoretical guarantee of learnability.

### Source recovery and reproducibility

SCITAS access uses `rizkalla@izar.hpc.epfl.ch`. The installed `dash-v2` environment
contains Python 3.12.15, JAX 0.11.1, MuJoCo/MJX 3.13.0, Flax 0.12.10 and Optax 0.2.8.
No original `.msgpack` checkpoint was found in the project or scratch search.
Therefore STOP-01 restores actor weights, estimator weights and actor observation
statistics from the exported NPZ. It does **not** resume the original optimizer,
critic, normalization sample count, or per-action exploration distribution.

Before training, independent NumPy and JAX forward passes must agree (64 test
observations, absolute/relative tolerance 2e-4). Plant nominal/standing joint
positions and command/timing dimensions must also match the bundle.
The fresh critic is fitted for 16 rollouts (1,048,576 samples at the pilot batch
size) with the actor/estimator fixed. These samples are separate from the 20M PPO
budget. This initialization is not a claim that the critic has converged.
The critic-prefit loss and exact bundle checksum are saved with the run.

### Evaluation and checkpoint selection

Before the first policy update and every 32 PPO rollouts (2,097,152 steps):

1. Retain the existing fixed-command walking ladder evaluation.
2. Run 32 fixed transition trials: commands 0.5, 1.0, 1.5 and 2.0 m/s, each at
   eight STOP times from 3.00 to 3.91 s after reset. STOP lasts 8 s; then RUN returns.
   The timing variation samples different gait phases without resetting the clock.
3. Require a genuinely moving approach: average incoming forward speed over the
   previous second at least max(0.15 m/s, half the requested speed).
4. A successful stop requires 2 s continuously below 0.1 m/s horizontal speed,
   0.3 rad/s body angular speed, 0.5 rad/s joint-speed RMS and 10° tilt. These
   conditions must also hold throughout the final 5 s of STOP; both feet must be
   supported for at least 90% of that final window.
5. Restart success additionally requires survival through 3 s after RUN and mean
   forward speed during the last second at least half the requested speed.

Falls and failed approaches count as failures. Auto-reset episodes cannot hide a
failure. Stop time measures the beginning of the first qualifying 2 s interval;
distance is horizontal path length to that point, or to failure/end if unsuccessful.
No maximum stopping-distance target has yet been established.

`best_stop.msgpack` is a separate keeper, ranked by stop success, restart success,
STOP survival, then residual speed. In STOP-01B, retention uses the nonzero-command
rungs only (equal weighting per rung), so improved standing cannot mask worse walking.
It is eligible only if walking-ladder survival
is within 5 percentage points and mean tracking error within +0.10 m/s of baseline.
The original `best.msgpack` walking keeper is retained separately.
The initial unmodified actor is allowed to be the keeper until a better candidate appears.

This initial benchmark is nominal/noiseless MJX simulation. Generalization to other
random seeds, plant variation, sensor noise and CPU deployment replay is a later
validation stage. Hardware stopping is not established by this experiment.

### Validation log

- 2026-10-02: six local unit tests passed: hold reward eligibility, successful
  transition/restart, falls with auto-reset, never-walked rejection, shuffle/late-drift
  rejection, and preset serialization/old-preset defaults.
- Python source compilation passed locally.
- Six unit tests also passed on SCITAS; Bash syntax and Python compilation passed.
- GPU initialized successfully (JAX 0.11.1, CUDA device 0, driver 535.154.05).
- Bundle restoration passed on GPU: maximum absolute actor-output difference
  approximately 3.5e-7 against independent NumPy inference; estimator check also passed.
- GPU environment stepping passed: finite observations/rewards, synchronized STOP
  state/input, no hold bonus during RUN, no air-time reward during STOP.
- STOP-01A baseline: 0/32 successful stops, 0/32 surviving the full STOP interval,
  29/32 qualifying moving approaches. All four speed groups had 0/8 successful stops.
  The original fixed ladder (including its zero/STOP rung) had 50% survival and
  0.5361 m/s mean tracking error. This is a nominal MJX measurement, not hardware data.
- STOP-01A reward audit: 79.66% of live STOP ticks hit the reward floor;
  mean scaled STOP reward was -0.4240. Mean unscaled speed penalty was -1.6677;
  hold income was only 1.86e-5. This motivated STOP-01B's dense positive braking income.
- Job 3200744 cancelled after 12 min 6 s, before any PPO policy update; original
  snapshot and baseline artifacts retained. The four critic-only final-minibatch
  MSE values printed before cancellation were 14.162, 51.492, 59.862, 56.334.
- STOP-01B adds a seventh unit test for monotone, posture-gated braking income,
  scores walking retention without the STOP rung, masks support metrics after a fall,
  and skips redundant bring-up/current-band evaluations when those curricula are disabled.
- STOP-01B baseline: 0/32 successful stops and 0/32 surviving STOP; 32/32 qualifying
  moving approaches. Live STOP reward-floor fraction decreased to 25.27%, and
  mean scaled live STOP reward was +0.3336. These averages exclude the terminating
  fall tick and its fall penalty; positive live-tick income is not proof of positive
  episode return or successful learning.
- STOP-01B RUN-only retention reference: 67.26% mean survival across nonzero command
  rungs and 0.6970 m/s mean tracking error. The full ladder, including STOP, measured
  53.125% survival and 0.5278 m/s error. The full requested ladder extends to 2.5 m/s,
  beyond the source bundle's recorded training command upper bound (~2.075 m/s).
- First PPO update: 65,536 environment steps, mean per-tick training reward +1.676,
  KL 0.0004, exploration standard deviation ~0.17, reported throughput 4,868 steps/s.
  These are optimization/startup measurements, not evidence of successful stopping.
  The first post-update transition evaluation is scheduled at 2,097,152 PPO steps.
- Startup snapshot recorded on 2026-10-02: 524,288 PPO steps, training reward
  1.6938, KL 0.003589. This preceded the first post-update stopping evaluation;
  the completed-run outcome above supersedes that initial status.
- The log prints `dr 1.000` because the inherited initialization sets the scale to
  one when the DR curriculum is disabled. Actual plant randomization is disabled
  by `dr_enable=False`; the printed scale does not override that switch.

### Artifacts and progress tracking

The run saves `resolved_config.json`, `warmstart.json`, `critic_prefit.json`,
`progress.csv`, `eval.csv`, `stop_baseline.json`, `stop_evaluations.jsonl`,
`best_stop_eval.json`, checkpoints, and `STOP_PROGRESS.md`.
The Markdown progress summary is regenerated after each stop evaluation and on
normal batch-script exit. Raw per-trial metrics remain in JSONL for final figures.

Regenerate the measured report at any time:

```bash
python RLframework/tools/stop_report.py --run RLframework/runs/dash_stop01b_s0
```

From the Windows workspace, refresh the remote report and download current metrics:

```powershell
.\RLframework\slurm\stop_progress.ps1
```

Downloads go to `docs/experiments/<snapshot>/`. This is an on-demand refresh;
it does not install an unattended local monitoring service. The job itself updates
its report at every stopping evaluation.

Source files and assets are uploaded as a separate experiment snapshot so running
jobs do not depend on later edits in the shared checkout. The snapshot manifest
records SHA-256 hashes, the base Git commit and the experiment definition.
The original bundle and existing policies are retained.

Initial measured artifacts: [STOP-01A progress](experiments/stop01_20261002_v1/STOP_PROGRESS.md),
[baseline trials](experiments/stop01_20261002_v1/stop_baseline.json).

Current measured artifacts: [STOP-01B progress](experiments/stop01_20261002_v2/STOP_PROGRESS.md),
[baseline trials](experiments/stop01_20261002_v2/stop_baseline.json).
The local progress renderer also exposes `time/env_steps` from the training CSV;
this reporting-only adjustment is outside the immutable v2 training snapshot.

### Decisions after the pilot

2026-10-12 completion check: STOP-04 job 3215784 completed at 11:38:43 Zurich,
after 28 min 51 s and 1,048,576 PPO steps. The 262,144-step keeper achieved
29/32, 32/32 and 30/32 holds on three evaluation seeds, passing the standing review
gate. Paired STOP-03 diagnostics identified angular motion as the blocking
quietness criterion in surviving final-policy failures. Final STOP-04 repeated
evaluations differ and include a walking-retention failure; retain the selected
keeper. See [full outcome and limitations](stop04-experiment-report.md#completed-outcome---12-october-2026).
Moving STOP remains unvalidated; no next-stage job was submitted in this check.

2026-10-12: user approved the [STOP-04 diagnostic and preservation pilot](stop04-experiment-report.md).
Compare STOP-03 best/final quietness conditions, then conditionally continue from
the best keeper with standing/walking rehearsal in PPO. Record videos, regression
guards and a three-seed assessment; do not automatically start transition training.
Submitted job **3215784**, confirmed queued (Priority) at 11:00:20 Europe/Zurich.
All 22 SCITAS preflight tests passed. [STOP-04 status](experiments/stop04_20261012_v1/STATUS.md).

STOP-03 completion, confirmed after connection retry: job 3205920 finished normally
in 26 min 28 s at 2,031,616 PPO steps. Imitation raised five-second standing holds
from 0/32 to 21/32 while retaining walking. The best hold keeper is PPO step
524,288 (21/32). Final PPO improved seven-second survival to 31/32 but reduced
strict five-second holds to 14/32; walking retention passed throughout. This is
standing from near-stance resets, not yet stopping from motion. See the
[measured STOP-03 outcome](stop03-experiment-report.md#completed-pilot-measured-results)
and [current status](experiments/stop03_20261007_v1/STATUS.md). No follow-on run was
submitted during this check.

2026-10-07: approved [STOP-03 imitation-assisted standing pilot](stop03-experiment-report.md).
Initialize the actor from feasible nominal stance demonstrations while retaining
source-policy walking examples; require measured HOLD improvement and retained
walking before the existing standing PPO stage. Rewards and ramp stay fixed for
this comparison. Record source, post-imitation, and PPO evaluations separately.
The user's credit-efficiency request concerns Codex credits; automatic cluster
reporting and batched checks reduce assistant usage.
Submitted as SCITAS job **3205920**; confirmed queued (Priority) at 19:00:36
Europe/Zurich. See [STOP-03 status](experiments/stop03_20261007_v1/STATUS.md).

2026-10-07: STOP-02 standing job 3204982 completed normally at 2,031,616 steps,
but all four post-update evaluations scored 0/32 five-second holds. Final walking
retention failed (59.52% survival versus 67.26% baseline). No faster stage was
launched; see the [completed STOP-02 outcome](stop02-experiment-report.md#completed-pilot-outcome--reviewed-2026-10-07).

2026-10-06, after user approval: STOP-02 implementation and validation are recorded
in the [STOP-02 experiment log](stop02-experiment-report.md). The earlier planning
entry below records the state before approval.

2026-10-06: documented the proposed [STOP-02 staged braking and holding plan](stop02-training-plan.md).
It introduces a smooth speed reference, phase-specific rewards, a full five-second
measured holding window after braking, and standing-to-transition curricula.
This is a planning update only; no STOP-02 implementation or training was launched.

- Continue only if transition success improves without losing walking performance.
- If reward flooring remains frequent, inspect braking reward components before
  increasing penalties.
- If the robot becomes still but cannot restart, rebalance transition examples.
- If the pilot succeeds, restore physical variation gradually and evaluate on
  held-out timing offsets/seeds before considering hardware deployment.
- Run ablations and additional seeds before attributing improvement to an individual
  reward term or claiming repeatability.
