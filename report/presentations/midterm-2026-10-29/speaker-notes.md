# DASH-01 controller midterm — speaker notes

18 main slides (~9-10 minutes), including the supplied outline, dividers and closing, then three hidden backups. Videos are click-to-play and embedded.
Results cutoff: 26 October 2026. Relative roadmap; dates from Tim’s timeline are not adopted.

## Slide 1 (20 s)

My project continues Nathann’s controller work on DASH-01, alongside Tim’s mechanical redesign. The first part of the semester has been about adapting the simulator and recovering useful locomotion on the new robot. I will show the architecture we inherited, what the experiments taught us, and how we move from promising simulation videos to treadmill tests and a short demonstration.

## Slide 2 (10 s)

I will follow the same four parts as Tim: context, what was done, objectives and timeline, and challenges. My focus is the controller and its validation.

## Slide 3 (5 s)

Part 1: Context.

## Slide 4 (35 s)

The starting point was not an empty software project: Nathann already had a working learning pipeline and a structured walking controller. But new geometry, feet and mass distribution change the dynamics. My part is to adapt the model, train useful behavior, diagnose what the policy actually does and validate it on the real robot. Tim and I therefore share the sim-to-real objective. The final goal remains startup, commanded walking and a controlled stop, rather than just a fast simulation clip.

## Slide 5 (55 s)

This is a hybrid controller. A neural network uses a short history of joint measurements and the IMU, plus a learned velocity estimate. At one hundred hertz it outputs forty-seven values: forty-one parameters describing periodic joint motion and gains, and six immediate corrective joint offsets. Gait parameters are committed on the gait clock; corrections remain active every tick. PD control and motor limits then generate torque. During PPO training, a separate critic can see simulator ground truth, including contact. The actor cannot. That separation matters for transfer: simulation contact rewards are not additional hardware sensors.

## Slide 6 (5 s)

Part 2: What was done.

## Slide 7 (55 s)

There were several phases, and the early failures were useful. On the previous robot, simply asking a walking policy to stop did not teach balance. Standing demonstrations improved near-standing holding, but that was not a robust stop from motion. With the new CAD we then saw dragging, unilateral activity and micro-steps, so reward and diagnostic definitions became central. After reproducing sustained walking, we moved to actual heel and toe contact measurements. The latest experiments focus on foot lift. This progression is why I now separate staying upright, following a speed and having an acceptable gait.

## Slide 8 (55 s)

These clips have the same target speed and show two stages of the work. The first is the reproduced walking recipe; the second is the flat-feet baseline. Please look at the feet, rather than only the body moving forward. Even when the policy survives, support can be rolled onto one pad and a foot can slide instead of taking a clean swing. The clips are normal-speed simulation recordings, not hardware experiments. They illustrate the evolution, but they are not a controlled single-reward ablation, and one good clip cannot establish robustness.

## Slide 9 (40 s)

The numerical comparison is encouraging. On the same nominal seven-speed evaluation ladder, the saved best flat-feet policy reduces average absolute speed error from about zero-point-eighteen to zero-point-zero-five metres per second. Both saved best policies complete all twenty-eight trials, so survival alone would not show this improvement. We must still inspect support, slip and individual leg behavior. Several things changed together in the flat-feet experiment, and these are single-seed training results, so the graph supports progress rather than a causal claim for one isolated reward term.

## Slide 10 (45 s)

The flat-feet baseline had only two to three millimetres of average airborne clearance. A dense penalty below the target gave a learning signal even when feet were still dragging, and fifty million additional steps increased mean lift to roughly twenty-four and twenty-one millimetres. This clip is the first trial from the retained batch, not a selected best-looking trial. There is an important caveat: a separate standalone replay fell after three-point-seven seconds. We keep that failure in the backup slides. The result demonstrates learnable foot lift, not a validated standalone startup or hardware-ready controller.

## Slide 11 (50 s)

This plot shows why the latest policy is progress but not the end point. Mean height rises, yet the fraction of scheduled mid-swing samples above twenty millimetres remains below our eighty-percent criterion. Peaks also reach roughly forty to fifty millimetres. A bonus that merely saturates does not cap the actual motion. The next run therefore returns to the flat-feet baseline and targets a lower band: ten millimetres minimum, bonus saturation at twenty-five, plus an explicit penalty above that upper target. We submitted two hundred million additional steps. There are no reviewed results for that run in this deck.

## Slide 12 (5 s)

Part 3: Objectives and timeline.

## Slide 13 (45 s)

The demonstration plan has not changed. First, the robot starts from a defined position, reaches an upright stance and tolerates small disturbances. Then a START command enables walking, with joystick-style speed changes within a walking regime. Finally, STOP should decelerate the robot and lead to stable standing for at least five seconds. A simulation homing reset is not an automatic hardware startup controller, and current locomotion training has STOP disabled. The policy’s stop behavior is also separate from the emergency kill. These remain integration tasks that must be checked one transition at a time.

## Slide 14 (55 s)

I do not want to postpone hardware testing until the simulator appears perfect. The immediate priority is the first short treadmill session, as soon as assembly, calibration and test support are ready with Tim. In parallel, simulation work continues on the clearance comparison and transition skills. Each hardware trial should produce a useful log: what command was sent, what motion and current were measured, and how the failure developed. We then reproduce the mismatch in simulation and retest. Before the event, we freeze and rehearse a tested sequence, keeping a fallback demonstration if a full autonomous stop is not reliable. The timeline is deliberately relative.

## Slide 15 (5 s)

Part 4: Challenges.

## Slide 16 (55 s)

The main difficulty is not just more training time. We know some likely sources of mismatch: contact friction and compliance, inertia, motor response, delay and state estimation. There is also an observability difference: the simulator can use contact for the gait clock, while the robot does not have that same contact signal. And we have already seen evaluation protocols produce different startup outcomes. Beyond these known suspects are the unknown unknowns that only the first real tests reveal. That is why early, short and instrumented treadmill trials are more useful now than trying to perfect every detail in simulation before touching hardware.

## Slide 17 (30 s)

We now have a substantially better simulation baseline, and the foot-clearance experiments show that the robot can learn the missing swing motion. The next useful step is to test the real robot early, measure the gap and connect the individual skills into the original demonstration sequence. The most useful support is coordinated treadmill access with Tim, a repeatable setup and agreement on which demo behaviors to prioritize if time is tight. I have backup slides with the reward definitions, the standalone failure and exact evidence sources.

## Slide 18 (5 s)

Thank you. I am happy to discuss the controller, the experimental results and the first treadmill tests.

## Slide 19 (0 s)

Use if asked about reward design. The actor does not receive direct contact flags. Mid-stance flat support has a deliberately modest missing-support cost rather than a large bonus for remaining planted. Clearance penalties apply to scheduled swing even if the foot never leaves the ground, so dragging cannot avoid them. An upper-height penalty is a soft training objective, not a guaranteed kinematic bound. Known brief-contact history blind spots remain documented.

## Slide 20 (0 s)

This video is retained explicitly to prevent overclaiming. The same checkpoint survives all twelve trials in an evaluation-style recording but fails in a separate one-environment pinned-command recording. Two protocol factors changed together, and small numerical differences can also matter; we have not isolated the cause. Do not state that the batch size caused the fall or that the successful evaluation proves robust startup. No auto-reset trajectory is spliced into this failure clip.

## Slide 21 (0 s)

Numbers are from locally archived results, not live training status. Speed statistics use saved best policies and the same nominal ladder. Clearance means use matched parent/final twelve-trial evaluation at 1.50 m/s; do not confuse airborne time means with mean peaks. Earlier standing results concern the previous robot. All training/evaluation results here remain simulation results.
