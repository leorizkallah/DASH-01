# CAD-v2 speed discovery: fresh baseline after source review

Branch: `dash-01-v2-speed-discovery`. Previous experiments: `dash-01-v2` at
`c035774`. [Nathann reward/source audit](../report/nathann-speed-reward-audit.md).
The first implementation is now [SPEED-01](speed01-experiment-report.md), authorized
for a five-hour run. It implements the reward-budget comparison without an alive
bonus and preserves the earlier campaign independently.

## Objective

Find the highest **sustained forward speed without falling**, on the corrected
CAD-v2 model with physical actuator limits and a freely moving base. Start from
new weights and normalization. Do not copy a GAIT-11 trajectory or checkpoint.
Allow the locomotion pattern to emerge; this speed-first objective can produce
running or hopping, rather than guaranteeing walking. Human-like appearance is
a separate requirement and must not be claimed as a consequence of speed alone.

## Proposed reward contract

Use Nathann's modern `objective="speed"` path as the reference, not the old railed
M3 simulator and not a derivative of `cad_v2_gait11`.

| Part | Proposed treatment and reason |
|---|---|
| Main income | Signed **world-X** speed, positive income gated by uprightness; backward motion remains a cost |
| Falling | Physical fall/unsafe-state termination and a terminal cost; no credit for post-fall motion |
| Survival during cold start | Small provisional bootstrap bonus only if measured reward budget requires it; expose and audit its decay |
| Physical feasibility | Preserve calibrated torque-speed, delay, joint/workspace and thermal model; no rails or artificial balancing force |
| Regularization | A short list of bounded posture, heading/lateral drift, contact slip and control-effort/smoothness costs, with explicit units and scales |
| GAIT-11 additions | Disable expected-foot supervisor, four supervisor observations, repeated/incomplete-attempt costs and per-turn reward budgets |
| Prescribed foot timing | Disable phase-contact reward during initial discovery; do not enforce a one-foot-at-a-time schedule or reward count thresholds |
| Speed commands / STOP | No target-speed tracking, command curriculum, finish-and-stop task or standing objective in this stage |
| Diagnostics | Record contacts, slip, clearance, alternation, foot activity, flight, motor loads and falls independently of reward |

Keep valid non-foot body contacts explicitly diagnosed/limited so forward sliding
on the torso cannot qualify. A successful movement must respect actuator and
contact assumptions, even if its gait is unlike human walking.

The reward's numerical speed ceiling must not silently become the claimed maximum
speed. Pick it from the physically feasible search envelope, log saturation, and
raise/review it if sustained velocities reach it. Cadence and action ranges need
the same audit. Do not copy M3's 0.5–50 Hz parameter range unexamined.

## Implementation and verification sequence

1. **Construct an independent speed preset.** Reuse the validated CAD-v2 model,
   same six motors, no spring and corrected inertias including torso allowance.
   Start with ordinary actor observations and the existing gait-generator/control
   interface; document that this structured action space is still a search prior.
2. **Audit the complete reward budget before long training.** Measure standing,
   random initial motion, representative moving traces, backward motion and falls.
   Compare per-second and discounted episode returns. Verify that early death is
   not rewarded, standing cannot dominate useful forward locomotion, and reward
   clipping does not erase the intended ranking. Test positive speed monotonicity
   below its ceiling, directionality, tilt gating, dt scaling and terminal costs.
   Do not choose final weights from old robot coefficients alone.
3. **Run a finite/numerical GPU preflight.** Validate model geometry/contacts,
   action bounds, motor-envelope enforcement and PPO updates on the new preset.
   This checks implementation; it does not require the random policy to walk.
4. **Train fresh policies on SCITAS.** Use an initial learning check before the
   longer allocation. Compare at least two independent seeds when claiming a
   method improvement; use the same reward across those seeds. Preserve source
   manifests, resolved settings, periodic checkpoints and reward-term histories.
   Agree the actual run budget with the implementation results, rather than
   silently launching another eight-hour continuation now.
5. **Select survival first, then speed.** Proposed selection: at least 90% full
   30-second survival on the selection trials, within physical/contact limits,
   followed by sustained world-X speed. Separately report speed over the whole
   episode including startup, and steady speed after a fixed warmup. Failed trials
   count against survival and retain their distance/fall time; do not rank their
   few fast pre-fall seconds above sustained locomotion. If none passes, label
   saved candidates diagnostic. Validate longer 60-second rollouts and held-out
   seeds before advertising the best speed.
6. **Review videos before changing the task.** Show the best qualified or clearly
   labeled diagnostic candidate and final policy, including failures. Classify
   the actual pattern (walk/run/hop/slide), asymmetric loading and slip. Only then
   decide whether to constrain gait appearance or train speed-command tracking.

This branch keeps the old experiment code and results for reproducibility, but
they are historical references rather than the active reward-design baseline.
The Robotics Day startup/joystick/stop demonstration remains a later stage.
