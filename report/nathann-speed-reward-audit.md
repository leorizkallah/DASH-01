# Speed-first gait discovery: Nathann's actual reward and its limits

Source review, 19 October 2026. Active branch: `dash-01-v2-speed-discovery`.
No new training was launched and no reward implementation was changed in this review.
The earlier experiments are preserved on `dash-01-v2`, commit
`c03577431fd1b208a4cbf7ab53402c9f961f6d8e`, including CAD corrections, code,
reports and curated videos. Ignored training outputs/source archives remain in
their existing local directories; remote checkpoints remain on SCITAS. This is
a local Git checkpoint, not a new remote backup or push.

## Main finding

Nathann already implemented **continuous maximum-speed training** in the current
MJX family: `dash_sprint`, with `objective="speed"`. This is the closest starting
point for the proposed experiment. It rewards signed forward speed, not tracking
0.15 or 0.25 m/s. It has no finish line or stop phase. The code is in
[`RLframework/config.py`](../RLframework/config.py), `_DASH_SPRINT`, and
[`RLframework/env.py`](../RLframework/env.py), `DashEnvV2._reward`.

The older **3.06 m/s runner** used a related but different objective: finish a
100 m sprint and then stop. Its saved configuration and measured result are
available, so it is a useful historical reference rather than a guess based on
comments. It does **not** establish that maximizing speed produces attractive,
symmetric walking: this runner used a constrained body and a one-legged hop.

## Branches and history inspected

Fetched `upstream` before inspecting all seven actual remote branches. The
[source audit](data/speed-discovery-reference/source-audit.json) records full
commit IDs, Git blob IDs, SHA-256 hashes and direct source links.

| Upstream branch | Tip | Relevant finding |
|---|---|---|
| `main` | `6af2971` | Current `RLframework`, restored older sprint stack; explicit `dash_sprint` |
| `balance-rl` | `c5edf08` | Same `RLframework/config.py` blob as main; separate standing/push work |
| `loop-rate` | `4e84331` | Same modern configuration; retains `legacy_trainer_to_port` |
| `feat/dynamic-parameter-identification` | `0cdf951` | Older `training` stack with sprint/speed modes |
| `research/ankle-spring-study` | `bd4c88f` | Older `training` stack, ankle/morphology study; not a newer free-body gait solution |
| `moteus_pi_controller_working` | `7bd891f` | Early `RL` stack; different terrain/path robot setup |
| `pendulum_sim2real` | `1c25c2f` | Earlier modeling/control work; no corresponding gait-training configuration |

Relevant history: `f11eed8` is the pre-deletion `walk_mit` tree; `0031885`
removed legacy stacks, and `c5edf08` restored the runner under `sprint_runner`.
Commit `a91e7d8` introduced motor-envelope/speed-frame corrections; `8323c89`
contains the speed/standing-bonus work; `504d35c` raises the cold-start
`dash_sprint` alive bonus. The current preset definition is not necessarily
the definition used by an older saved run. Use **resolved run configuration**
and curriculum together, not just today's preset name.

## What rewarded the 3.06 m/s policy

Evidence: [saved M3 run configuration](data/speed-discovery-reference/sprint_m3_mit_s0_config.json),
[final curriculum](data/speed-discovery-reference/sprint_m3_mit_s0_curriculum.json),
[`walk_mit/env.py`, `_reward`](../sprint_runner/walk_mit/env.py), and
[archived evaluation output](../sprint_runner/walk_mit/monitor/eval_sprint_s0_final.txt).
The requested run budget was 600M steps, despite the nested preset default of 800M.

Before the finish line, its main income is:

```text
I = 2 * clip(forward_speed, -3, +3)
U = clip((upright_cosine - 0.5) / 0.5, 0, 1)
if I > 0: I *= U

raw_reward = I - 0.5 + gait_credits - posture/motion/efficiency_costs
reward = max(raw_reward * dt/0.02, -1 * dt/0.02)
reward -= 100 on a fall
reward += 100 on successful finish-and-stop (if not fallen)
```

This was 200 Hz control (`dt=0.005`, reward scale 0.25) over 1 kHz physics.
Negative forward speed remains negative even when tilted. Positive income fades
as the torso tips, reducing reward for falling forward. Individual penalties
are also capped; the exact source remains authoritative for their definitions.

| Component | Saved raw coefficient / behavior |
|---|---|
| Forward speed | 2.0 per m/s, capped at 3 m/s |
| Alive bonus | **0** in this sprint run |
| Time cost | 0.5 each control step before dt scaling |
| Foot sliding | 8.0, squared horizontal toe motion above a deadband while grounded |
| Clearance / touchdown airtime | 0.4 / 2.0 |
| Long stance / phase-contact mismatch | 0.5 / 1.0 |
| Upright / height / vertical speed | 5.0 / 2.5 / 0.5 |
| Action change / Fourier coefficient change / residual | 0.1 / 0.5 / 0.02 |
| Torque / motor velocity / positive mechanical power | 0.0001 / 0.0001 / 0.0002, curriculum-scaled |
| Duty-sharing / weaker-foot swing-floor penalties | **Both zero** in this saved run |

It also charges lateral motion, angular motion/momentum, leg crossing and hip-roll
deviation. This was not a reward containing only speed and falls.

Why a time cost for a fixed-distance dash? Ignoring discounting and clipping,
integrated forward speed over the dash is its distance regardless of duration.
The time cost and discounted finish bonus help distinguish a faster completion.
After crossing the line the objective changes to stopping; the archived output
reports eight line crossings and **"never stopped"** in all eight. Do not describe
this as eight successful complete sprint-and-stop tasks. Diagnostic `fell` flags
also differ from the headline finish counts; they are not a new robustness test.

## What the later `dash_sprint` changes

The modern preset uses `objective="speed"`, 30-second episodes, a 4 m/s speed
reward ceiling, 100 Hz control, the 47-output gait-generator/residual policy,
and no clock cost, finish reward or stopping objective. It keeps posture/contact,
smoothness, actuator and curriculum terms. Initially `w_alive=2.25`; its scale
decays to 0.25 over steps 60M–120M, leaving a raw bonus of 0.5625.

**An important comment/code distinction:** the code comments say the standing
basin is closed because standing earns zero speed income. Only the *speed term*
is zero: the alive bonus still contributes. Whether standing beats moving or
falling depends on the complete reward budget and horizon. We must measure that
on CAD-v2 instead of accepting the comment as a guarantee.

The preset also retains a phase-contact schedule whose stance ratio can fall
below 0.5, encouraging flight. That is a gait prior, not unconstrained discovery.
Its default speed income is body-forward (`sprint_world_speed=False`), which can
reward circling despite poor progress along the intended direction. Its cadence
range and speed cap are search limits, not measured robot capabilities.

## Limits of copying the historical result

1. **The 3.06 m/s robot was not freely balancing in 3D.** Its saved `base_lock`
   fixes lateral translation, roll and yaw. Its audit reports one-legged hopping,
   approximately 80% flight and strongly asymmetric motor loading. The archived
   torque report is in [`sprint_runner/README.md`](../sprint_runner/README.md).
2. **That saved M3 run omitted velocity/back-EMF constraints.** `motor_vel_limit=0`
   and `motor_r_ohm=[]`. The later source changed these defaults, so merely calling
   the same preset today would not reproduce the original physical assumptions.
3. **The later fully free M6 evidence is mixed.** We recovered
   [M6 seed 0 configuration](data/speed-discovery-reference/sprint_m6_lim2_s0_config.json)
   and [diagnostics](data/speed-discovery-reference/sprint_m6_lim2_s0_diagnostics.json),
   plus [seed 1 diagnostics](data/speed-discovery-reference/sprint_m6_lim2_s1_diagnostics.json).
   Motor limits are present, but these diagnostic episodes record zero line
   crossings for seed 0 and one for seed 1; several surviving trials have negative
   or small world-X displacement. They do not certify reliable straight running.
4. **The learner is reusable; the trained policy is not robot-independent.**
   `gait.py` assumes six named motors, cam/thigh/hip families, mirrored signs,
   specific ranges and PD controls. `plant.py` resolves named joints, contacts and
   actuator properties. Correct CAD, masses, inertias, transmission geometry,
   motor limits and reset pose still need the existing CAD-v2 integration.
5. **"Fastest" is not automatically "best walking."** A speed objective can select
   running, hopping or sliding. It finds a policy within the simulator and action
   space, not a proven global optimum. This experiment changes the question to
   fastest sustained feasible locomotion; it cannot promise a human-like gait.

There is also a separate CMA-ES periodic-gait solver at
[`library/library_solve.py`](../sprint_runner/walk_mit/library/library_solve.py).
Its executed cost uses target-speed error, periodicity defect, falls and envelope
penalties; stability-multiplier penalization defaults off (`lam_bar=0`). It is
not the PPO method that produced the saved 3.06 m/s policy and should not be
confused with the reference experiment.

## Recommended starting point

Use the modern MJX/PPO speed objective with the corrected **free CAD-v2 plant**,
fresh weights, optimizer and observation statistics. Set aside the GAIT-01–11
training objectives rather than inheriting GAIT-11 and switching one coefficient.
Keep their results available as evidence, with diagnostics separate from rewards.

The [new branch plan](../docs/speed-discovery-plan.md) specifies a compact reward,
explicit physical limits, reward-budget checks and survival-first speed selection.
It deliberately separates a proposal from a tested implementation: no claim is
made here that another long run will necessarily discover satisfactory locomotion.
