# Presentation outline

An eight-slide narrative, adaptable to a short progress meeting or the final report.
Use the SVG/PDF figures for crisp export; all clips are simulation recordings.

| Slide | Message | Evidence / visual | Speaker note |
|---|---|---|---|
| 1. Objective | Stop on command and stay still for five seconds | Robot walking clip; success thresholds | Define upright, supported and quiet, not just zero speed |
| 2. Controller | Learned gait parameters plus fast joint corrections | `control_architecture.png` | 100 Hz actor; 41 latched parameters and 6 live residuals |
| 3. Initial failure | STOP fine-tuning did not produce stable standing | Abrupt STOP clip | Label STOP-01B checkpoint and older protocol |
| 4. Smooth braking | A smooth reference delays failure but cannot supply balance | `ramp_trace.png`, ramp clip | Actual speed can differ from the reference |
| 5. Diagnose the missing skill | Failure also occurs from near-stance | `standing_comparison.png`, fixed-stance clip | Fixed stance is a teacher/feasibility result, not learned STOP |
| 6. Imitation initialization | Show the actor feasible HOLD actions and preserve walking | `imitation_fit.png` | Frozen estimator; held-out trajectories; physical gate |
| 7. Outcome and tradeoff | 0/32 -> 21/32 holds; later PPO loses some stillness | `stop03_outcomes.png`, `walking_retention.png` | Final 31/32 survival is distinct from 14/32 holds |
| 8. What remains | Validate transitions and understand surviving failures | Quiet-period distribution and open questions | Avoid claiming robust stopping or sim-to-real transfer |

Suggested closing statement: "Imitation gave the policy a feasible standing
behavior. The next challenge is to preserve stillness during RL and connect that
behavior to controlled braking from motion."

For the written report, expand slides 2 and 6 into Methods, slides 3-7 into Results,
and slide 8 into Discussion. The experiment CSV table supplies exact numbers.
There is no STOP-03 replay in this local collection yet; do not place an older clip
beside STOP-03 statistics without explicitly identifying the mismatch.
