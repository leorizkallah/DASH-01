# DASH-01 controller midterm - speaker notes

Updated 28 October 2026. **State-of-the-art slide 4 retained; thesis alternative removed.**
**19 main slides + three hidden backups; planned speaking time 9 min 40 s**, including the short video sections. Allow for playback and discussion pauses.
[Matching PowerPoint](DASH01_controller_midterm_final_mechanical_update.pptx). Slide numbers follow the current deck order: stopping failures at slide 9 and walking evolution at slide 10.
Results cutoff: 26 October 2026. Relative roadmap; no new training results or hardware validation are claimed.

## Slide 1 (20 s)

**DASH-01**

My project continues Nathann’s controller work on DASH-01, alongside Tim’s mechanical redesign. The first part of the semester has been about adapting the simulator and recovering useful locomotion on the new robot. I will show the architecture we inherited, what the experiments taught us, and how we move from promising simulation videos to treadmill tests and a short demonstration.

## Slide 2 (10 s)

**Outline**

I will follow the same four parts as Tim: context, what was done, challenges, and objectives and timeline. My focus is the controller and its validation.

## Slide 3 (5 s)

**Context**

Part 1: Context.

## Slide 4 (30 s)

**State of the art: learned locomotion**

This slide gives three representative examples rather than a robot ranking. On Cassie, periodic costs on contact forces and foot velocities specify several gaits without prescribing every joint trajectory. On ANYmal, GPU-parallel simulation and a curriculum accelerate training. On Digit, a history-based transformer trained across randomized simulation environments transfers to outdoor walking. These are different robots and tasks: their performance numbers cannot be compared directly with DASH-01. The message for our project is to combine useful gait structure, enough training and early physical validation.

## Slide 5 (35 s)

**Semester goal**

The starting point was not an empty software project: Nathann already had a working learning pipeline and a structured walking controller. But new geometry, feet and mass distribution change the dynamics. My part is to adapt the model, train useful behavior, diagnose what the policy actually does and validate it on the real robot. Tim and I therefore share the sim-to-real objective. The final goal remains startup, commanded walking and a controlled stop, rather than just a fast simulation clip.

## Slide 6 (30 s)

**Starting point: Nathann’s controller**

Nathann developed a hybrid reinforcement-learning controller. The network receives the robot’s joint and IMU measurements, along with the target speed. Instead of commanding each motor directly, it chooses parameters for a periodic gait generator, plus small corrections for balance. The gait generator produces joint targets, which the low-level motor controllers track. This structure combines a regular walking pattern with learned, real-time adaptation. My work builds on this architecture for the redesigned robot.

## Slide 7 (5 s)

**What was done**

Part 2: What was done.

## Slide 8 (55 s)

**What we explored and what we learned**

There were several phases, and the early failures were useful. On the previous robot, simply asking a walking policy to stop did not teach balance. Standing demonstrations improved near-standing holding, but that was not a robust stop from motion. With the new CAD we then saw dragging, unilateral activity and micro-steps, so reward and diagnostic definitions became central. After reproducing sustained walking, we moved to actual heel and toe contact measurements. The latest experiments focus on foot lift. This progression is why I now separate staying upright, following a speed and having an acceptable gait.

## Slide 9 (40 s)

**Stopping trials: early failures**

These failed stopping trials used the previous robot model. Both begin with a requested walking speed of one metre per second. On the left, the STOP-01B final policy receives an abrupt STOP. On the right, the same policy uses the STOP-02 smooth speed-reference adapter; this is a command-only diagnostic, not the trained STOP-02 standing policy. Neither clip demonstrates a successful five-second quiet stand. Slowing the command alone did not provide the missing standing and transition skills. The left recording is half speed; the right is normal speed. These are illustrative single trials, not a controlled estimate of success rates.

Evidence: report/assets.md; report/videos/stop01_abrupt_failure.mp4; report/videos/stop02_ramp_failure.mp4; docs/stop02-experiment-report.md.

## Slide 10 (55 s)

**Walking policies: visual evolution**

These clips have the same target speed and show two stages of the work. The first is the reproduced walking recipe; the second is the flat-feet baseline. Please look at the feet, rather than only the body moving forward. Even when the policy survives, support can be rolled onto one pad and a foot can slide instead of taking a clean swing. The clips are normal-speed simulation recordings, not hardware experiments. They illustrate the evolution, but they are not a controlled single-reward ablation, and one good clip cannot establish robustness.

## Slide 11 (45 s)

**Foot clearance: first results**

The flat-feet baseline had only two to three millimetres of average airborne clearance. A dense penalty below the target gave a learning signal even when feet were still dragging, and fifty million additional steps increased mean lift to roughly twenty-four and twenty-one millimetres. This clip is the first trial from the retained batch, not a selected best-looking trial. There is an important caveat: a separate standalone replay fell after three-point-seven seconds. We keep that failure in the backup slides. The result demonstrates learnable foot lift, not a validated standalone startup or hardware-ready controller.

## Slide 12 (50 s)

**Next training objective**

This plot shows why the latest policy is progress but not the end point. Mean height rises, yet the fraction of scheduled mid-swing samples above twenty millimetres remains below our eighty-percent criterion. Peaks also reach roughly forty to fifty millimetres. A bonus that merely saturates does not cap the actual motion. The next run therefore returns to the flat-feet baseline and targets a lower band: ten millimetres minimum, bonus saturation at twenty-five, plus an explicit penalty above that upper target. We submitted two hundred million additional steps. There are no reviewed results for that run in this deck.

## Slide 13 (5 s)

**Challenges**

Part 3: Challenges.

## Slide 14 (55 s)



**Where we are stuck**

Tim's first hardware tests revealed a problem that the simulation results could not establish. The Achilles rod is in tension at the heel. In Tim's report, the printed attachments cannot carry this load, and play in the leg makes that arrangement unreliable. The robot cannot yet establish stable standing with this foot design. For the controller, this is a concrete example of the sim-to-real gap: rigid links and ideal loop constraints do not validate attachment strength or reproduce real structural flex and joint play. Actuator and contact models exist, but their match to the assembled hardware still needs validation. We have identified a mechanical blocker; the remaining mismatch must be measured after repair. More policy training alone cannot repair the mechanism.

## Slide 15 (30 s)



**What we need**

With Tim, the proposed repair moves the Achilles to the toe and anchors it above the knee. Two rods are intended to resist foot torsion. This needs CAD changes, machining and assembly. On the control side, I then need to update the model, recheck the closed loops and starting pose, and log supported standing and motion. The hardware kill switch is a prerequisite. We can retrain and attempt walking only once the repaired robot can support itself reliably.

## Slide 16 (5 s)

**Objectives and timeline**

Part 4: Objectives and timeline.

## Slide 17 (45 s)

**Objectives for Robotics Day**

The demonstration plan has not changed. First, the robot starts from a defined position, reaches an upright stance and tolerates small disturbances. Then a START command enables walking, with joystick-style speed changes within a walking regime. Finally, STOP should decelerate the robot and lead to stable standing for at least five seconds. A simulation homing reset is not an automatic hardware startup controller, and current locomotion training has STOP disabled. The policy’s stop behavior is also separate from the emergency kill. These remain integration tasks that must be checked one transition at a time.

## Slide 18 (55 s)

**Timeline**

I do not want to postpone hardware testing until the simulator appears perfect. The immediate priority is the first short treadmill session, as soon as assembly, calibration and test support are ready with Tim. In parallel, simulation work continues on the clearance comparison and transition skills. Each hardware trial should produce a useful log: what command was sent, what motion and current were measured, and how the failure developed. We then reproduce the mismatch in simulation and retest. Before the event, we freeze and rehearse a tested sequence, keeping a fallback demonstration if a full autonomous stop is not reliable. The timeline is deliberately relative.

## Slide 19 (5 s)

**Questions?**

Thank you. I am happy to discuss the controller, the experimental results and the first treadmill tests.

## Slide 20 (0 s)

**Backup: reward definitions** - hidden backup

Use if asked about reward design. The actor does not receive direct contact flags. Mid-stance flat support has a deliberately modest missing-support cost rather than a large bonus for remaining planted. Clearance penalties apply to scheduled swing even if the foot never leaves the ground, so dragging cannot avoid them. An upper-height penalty is a soft training objective, not a guaranteed kinematic bound. Known brief-contact history blind spots remain documented.

## Slide 21 (0 s)

**Backup: the failed replay** - hidden backup

This video is retained explicitly to prevent overclaiming. The same checkpoint survives all twelve trials in an evaluation-style recording but fails in a separate one-environment pinned-command recording. Two protocol factors changed together, and small numerical differences can also matter; we have not isolated the cause. Do not state that the batch size caused the fall or that the successful evaluation proves robust startup. No auto-reset trajectory is spliced into this failure clip.

## Slide 22 (0 s)

**Backup: evidence and comparison limits** - hidden backup

Numbers are from locally archived results, not live training status. Speed statistics use saved best policies and the same nominal ladder. Clearance means use matched parent/final twelve-trial evaluation at 1.50 m/s; do not confuse airborne time means with mean peaks. Earlier standing results concern the previous robot. All training/evaluation results here remain simulation results.

## References for slide 4

- [Siekmann et al., ICRA 2021 - periodic reward composition on Cassie](https://arxiv.org/abs/2011.01387).
- [Rudin et al., CoRL 2021 / PMLR 2022 - massively parallel locomotion learning](https://proceedings.mlr.press/v164/rudin22a.html).
- [Radosavovic et al., Science Robotics 2024 - real-world humanoid locomotion](https://doi.org/10.1126/scirobotics.adi9579).
