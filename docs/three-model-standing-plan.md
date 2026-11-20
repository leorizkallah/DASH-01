# Three-model standing comparison: reward and curriculum

Prepared **9 November 2026**, Europe/Zurich. The reward/model review below is
now implemented. [Implementation and submission record](experiments/stand03_20261109_v1/STATUS.md)
tracks validation and job IDs. The frozen training snapshot retains the original
review-stage version of this document; this working copy records the later status.

Contact clarification: **one continuous sole per original V1 foot, two separate
pads per V2/X-shape foot**. These are collision regions, not literal point
contacts. V1 front/rear support is measured within its single sole, without
adding another collision box. All three begin with planted feet; recovery steps
are disabled in this campaign.

User clarification: **start with both feet planted; allow recovery steps only in
a later, harder stage**. The initial task is supported standing from a known
pose, not startup from the floor, walking, or stopping from motion.

## Models and mass corrections

| Version | Reference | Current mass | Preparation needed |
|---|---|---:|---|
| V1, original fixed foot | `RLframework/model/dash01_free.xml`, `stand` key | 14.356 kg | Use the established corrected plant; adapt standing contact/anchor measurements. |
| V2, articulated foot | `RLframework/model/dash01_cad_v2_homing.xml`, `homing` key | 14.594154 kg | Use PDF-corrected inertials, including the existing 1 kg torso allowance. Adapt BalanceRL to its four loops. |
| New X-shape | `xshape_closed/EXPORT_assembly-v8_closed.xml` | Raw 5.315278 kg; proposed corrected **about 14.537 kg** | Raw export is fixed-base, has no floor/contact collision masks, actuators or standing key. Build and validate a separate floating-base plant. |

All three keep six identical motors and their existing torque/speed/delay model.
Do not add motors or battery again to the inclusive V2 PDF assembly masses.
The new model receives the same **1 kg torso accessory allowance**, once.
V1 retains its established assembly properties; do not silently add another kilogram.

The X-shape export has **nine bodies whose raw local mass, COM and inertia match
V2**, and **six changed bodies: both thighs, both feet and both Achilles links**.
Their COMs and tensors differ. The old confirmation about V2 PDF frames does
not automatically confirm the new export's frames.

Proposed transfer, pending frame/geometry checks:

- Unchanged parts: reuse the corresponding V2 PDF mass, COM and COM tensor.
- Changed parts: use the new CAD COM and shape tensor, scaled by the corresponding
  V2 component's mass-correction factor: `k = V2_PDF_mass / V2_raw_CAD_mass`;
  `m_new = k * new_raw_mass`, `I_new = k * new_raw_COM_tensor`.
- This produces about **14.537 kg including the torso allowance**. It is a
  density-based estimate for changed parts, not a new measurement. Do not force
  the total to equal V2, and do not transplant an old COM/tensor into a changed frame.
- Confirm local frames, sole surfaces, closed-loop geometry, joint ranges and
  actuator signs before constructing the training snapshot. Replace estimates
  if new measured part properties become available.

[Read-only model audit](../report/data/standing03/model-audit.json),
[draft mass transfer](../report/data/standing03/mass-transfer-draft.json),
[per-body CSV](../report/data/standing03/mass-transfer-draft.csv).
All draft tensors pass positivity and principal-inertia triangle checks. The raw
X-shape loop sites coincide to below 0.000001 mm in its exported pose. These
checks do not establish equilibrium, structural strength or hardware stability.
V1's legacy distal 0.405 kg segment is itself an unweighed estimate.

## What we can reuse, and what needs correction

Use the standing-specific `BalanceRL` controller/task, not the locomotion reward.
Keep the same policy architecture, action mapping, actuator limits and optimizer
settings across the three new policies. Start fresh rather than transferring
the walking policy or giving V1 a trained balance policy that the others lack.
Separate model files and explicit motor mappings account for different passive
joint counts; normalize all six commanded axes to the same anatomical order.

The existing `BalanceRL/env.py` reward is:

```text
1 alive + 1 upright + 0.5 height + 1.5 CoM-centering
+ 0.3 calm velocity + 0.2 calm pose
- 0.05 angular speed squared - 0.5 normalized torque squared
- 0.02 action change squared - 0.5 foot-body-origin motion squared
- 0.02 normalized gain - 1 joint-limit proximity - 20 on fall.
```

That is an existing baseline, **not yet a coherent three-CAD experiment**:

1. `_foot_xy` uses the attached body's origin. In V1 the contact-pad body origin
   is at z=0.495 m in the standing key, while the pad is near z=0.0003 m. In V2
   the corresponding body origin is at z=0.042 m. Body-origin motion therefore
   measures different things. Use **sole-level reference markers** for anchoring
   and **actual contact-point velocity** for slip on every model.
2. `_contacts` currently tests penetration, without a normal-force threshold.
   Use actual MuJoCo/MJX contact and >=2 N per pad, not height/proximity alone.
3. `_support_xy` invents a two-foot support midpoint when neither foot contacts
   the floor. No-support states must receive **zero support/quiet credit**.
4. The plant adapter hard-codes V1's two-loop site names. V2/X-shape have four
   loops. Missing body/site lookup must fail loudly; an index of -1 is invalid.
5. Independent hinge/reset and loop-site noise can violate closed mechanisms.
   Disable loop-site/axis jitter; reset perturbations must preserve closure.
6. Current push promotion counts being alive 2 s later, not settling. Replace
   that criterion with recovery **and five seconds of quiet standing**.
7. Do not inherit full +/-3 cm CoM randomization or a 0.15–2.5 m/s push range
   into the simplest fixed-foot stage. Those are a different recovery problem.
8. The existing pitch reflex is specific to the old motor conventions. Disable
   it for this comparison unless independently validated on all three mappings.

## One shared reward, three model-specific states

Use **the same coefficients and SI-unit tolerances** for all three models:

```text
R_i = (dt / 0.010) * (
        1.00 alive
      + 1.00 U_i + 0.50 H_i + 1.50 S_i + 2.00 Q_i + 0.20 P_i
      - 0.50 E_i - 0.02 A_i - 0.50 L_i - 0.02 G_i
      - 1.00 J_i - 1.00 D_i - 0.25 F_i
      )
      - 20 * physical_fall
      -  5 * planted_task_violation_without_physical_fall

R_V1     = R(state_V1;     pose_ref_V1,     height_ref_V1,     soles_V1)
R_V2     = R(state_V2;     pose_ref_V2,     height_ref_V2,     soles_V2)
R_Xshape = R(state_Xshape; pose_ref_Xshape, height_ref_Xshape, soles_Xshape)
```

These are **proposed initial coefficients**, not experimentally tuned values.
Do not give one geometry different reward weights to make it pass. At 100 Hz,
`dt/0.010 = 1`. Any common PPO reward scale must apply equally to all rewards
and terminal events. Do not add a dense negative-reward floor.

| Symbol | Definition and purpose |
|---|---|
| `alive` | 1 only for a finite, non-fallen state. No income for an artificial torso hold. |
| `U` | `exp(-(tilt/0.20 rad)^2)`: upright torso; use a true tilt angle, not an upside-down-ambiguous sine. |
| `H` | `exp(-((height-height_ref_i)/0.04 m)^2)`: remain at the model's feasible standing height. |
| `S` | `exp(-(d_out/0.02 m)^2)`, where `d_out` is distance from projected whole-robot CoM to the actual force-qualified support polygon; zero when no valid support exists. A point already inside has zero outside distance. This is shaping, not a proof of dynamic stability. |
| `Q` | `calm_gate * both_feet_loaded * exp(-(||v_xy||/0.05)^2 -(||omega||/0.25)^2 -(rms(qdot_motor)/0.30)^2)`: substantial quiet-standing reward. |
| `P` | Small nominal-pose tie-breaker `exp(-mean((q-q_ref_i)^2)/0.50^2)`, only when calm and CoM is supported. Do not force an unsuitable pose against balance. |
| `E` | Mean over six motors and physics substeps of `(torque/available_peak_torque)^2`; bill actual effort, not just the final substep's torque. |
| `A` | Sum of squared changes of normalized actions at the common 100 Hz control rate. |
| `L` | Force-weighted mean `clip(||v_contact_xy||^2 / 0.10^2, 0, 4)` over real floor contacts; use body translation **plus angular contribution at the contact point**, penalizing every horizontal direction. Zero when there are no contacts; lack of contact is handled elsewhere. |
| `G` | Mean `Kp/Kp_max`; a modest gain cost so a stiff stance is not made artificially expensive. Bounds and hardware torque limits remain in force. |
| `J` | Mean motor-joint soft-limit proximity cost within the last 5% of each valid range; also validate passive joint/loop limits independently. |
| `D` | Mean over feet of `clip(||sole_xy-sole_anchor_xy||^2/0.020^2, 0, 4)`: discourage moving/sliding the stance away from its initial footprint. Use comparable sole markers, not CAD body origins. |
| `F` | Mean missing heel-AND-toe support during calm standing; each pad needs real contact force >=2 N. Modest cost, no large planted-foot bonus. Heel-only/toe-only rocking under a push is allowed. |

`calm_gate` is 1 in ordinary quiet standing, zero during a push, and ramps back
to 1 over the first second after the pulse. Recovery joint motion is therefore
allowed. Actor inputs remain robot measurements/history; simulator truth used
for reward/support scoring is not added as unavailable hardware feedback.

Per foot, distinguish heel, toe, any-foot (OR) and flat support (AND). The Q term
requires force-qualified **any-foot contact on both sides**. No foot is welded
to the world; the base has all six free coordinates. These rewards cannot
validate printed-part strength or automatically model backlash/flexibility.

### Initial planted-foot rule

The reference footprint is set once from the validated starting pose and does
not follow the robot. A single pad losing contact is **not** an episode-ending
event. A whole foot above 3 mm sole clearance for >50 ms, or moving its sole
marker >20 mm from its anchor, is a planted-task violation. End that episode
with -5 unless a physical fall already receives -20; log the reason separately.
These initial numerical tolerances require contact/reset validation before training.

Physical falls: non-finite state, forbidden non-foot floor contact, torso height
<65% of the model's nominal standing height, or torso tilt >45 degrees. Do not
reuse V1's absolute 0.55 m height threshold on other geometries. Episode timeout
is a truncation with value bootstrapping, not a fall or an automatic success.

**Disabled in every model:** walking-speed income, gait clock/cadence rewards,
swing clearance, air-time bonuses, weaker-foot stepping, alternating-step and
walking-symmetry termination. Those would conflict with standing planted.

## Perturbation curriculum

1. **Quiet standing first.** Nominal masses/friction, no pushes, no springs or
   ground fixtures, no artificial torso hold. Start at each model's validated
   standing pose. Begin with minimal reset noise and then increase measured
   sensor noise. Unassisted standing must pass before push escalation.
2. **Small pushes, feet planted.** First level `L=0.02 m/s` equivalent velocity
   impulse. Horizontal 100 ms force pulses at the torso COM, uniform direction;
   start after 5 s, separated by 10–14 s, allowing 2 s recovery plus a full 5 s
   quiet window before another event. Keep 20% no-push episodes. Within push
   episodes, draw 30% easy pushes in [0,L/2], 70% hard in [L/2,L].
3. **Increase planted difficulty based on actual success.** After at least
   200 hard-push outcomes at a level: promote `L *= 1.10` if >=90% recover and
   settle, with >=95% success on separate quiet-standing trials. Retreat
   `L *= 0.80` below 70%; otherwise hold. Start the planted pilot with a
   provisional upper level of 0.15 m/s; reaching it is not a promised capability.
   Keep push and model/noise curricula separately logged; do not make one
   permanently depend on another completing.
4. **Recovery steps later, after reviewing planted results.** Explicitly change
   the task mode: remove step-violation termination, relax footprint cost only
   during a bounded recovery, keep real contact slip penalties, and require
   renewed quiet standing. Reset a recovery footprint once, after measured
   settling, never continuously to reward walking away. Report these results
   separately from the planted-foot comparison.

For a fair push, use **equal impulse per robot mass**, not an equal raw force:

```text
J_i = M_i * delta_v
F_i = J_i / 0.100 seconds
```

At delta_v=0.10 m/s, this means about 14.36 N (V1), 14.59 N (V2), and 14.54 N
(provisional X-shape). No artificial velocity teleport at this stage. Apply at
the torso COM initially to avoid an uncontrolled additional moment. Add offset
pushes/rotations and small model-parameter variations only after the basic test.
If mass randomization is later enabled, use the actual episode's drawn total
mass for impulse normalization, rather than an unchanged nominal constant.
Do not randomize closed-loop anchors/axes or use unconstrained passive-joint noise.

## Success and comparison

Train three independent fresh policies, one per CAD, on SCITAS with the same
architecture, PPO settings, control/physics rate, motor limits and sample budget.
One GPU per job, up to three jobs concurrently subject to allocation. No existing
walking or old trained balance policy gives any model a head start. Freeze the
budget and source manifests after this reward/model review, before submission.

A push is successful only if the first episode survives, does not take a step
in planted mode, recovers within **2 s after the pulse**, then remains quiet for
**5 continuous seconds**. During that window require: base speed <0.05 m/s,
angular speed <0.25 rad/s, motor RMS speed <0.30 rad/s, tilt <10 degrees,
height within 5 cm of nominal, torso XY drift <=3 cm, each foot <=10 mm from
its anchor, and force-qualified support on both feet for >=95% of samples.
Apply the planted-foot violation rule throughout, including recovery.

Evaluate all models on **the same fixed levels**, including no push and
0.02/0.05/0.10/0.15 m/s, with paired direction/duration schedules. Each model
can progress at its own rate, but do not rank models only by their last training
level or raw return. Log quiet success, push survival, recovery+settle success,
step violations, slip, drift, settling time, torque/current estimates and every
reward term separately. Validate recovery phases with actual completed windows;
pending pushes at timeout are censored, not successes.

One policy/seed per model is an exploratory comparison, not statistical proof
that one design is superior. The mass estimates and mechanical geometry both
differ; this is not a pure geometry-only ablation. Include zero-action fixed-PD
baselines under identical conditions. Before interpreting success, check that
all models have a free base, real contact load, valid motors/loops, no hidden
support and no NaNs. Test an independently reloaded recorder as well as the
training evaluator, given the previous 12/12 versus 9/12 replay discrepancy.

## Status

The reviewed task is implemented as an opt-in `BalanceRL` mode, with isolated
`stand_v1.xml`, `stand_v2.xml` and `stand_xshape.xml` plants. Native MuJoCo
contact/equilibrium tests pass for all three. Six shared tests check contact
counts, support forces, hovering, contact-point slip, sole markers and support
hulls. GPU physics/PPO preflight tasks are submitted as array **3246816**;
training array **3246822** is submitted with a whole-array success dependency
on those checks. V1 preflight passes; V2/X-shape are running/queued at the
13:48 CET submission check. Training has not started at this check.

The common budget is **200M fresh-policy steps per model**, one seed each,
512 environments, 100 Hz policy / 1 kHz physics, with a 24 h allocation ceiling
and 22.5 h training reserve cutoff. Saved completion metadata distinguishes a
step-budget finish from a time-budget finish. Sensor noise starts at zero and
ramps over 20M steps after quiet-standing unlock; mass/friction/loop geometry
randomization remains disabled in this first comparison.

[Actual collision layout](../report/figures/standing03_contact_layout.png)
shows the native loaded contact points in each reference pose. A static
equilibrium and passing preflight do not establish learned robustness.
