# State-of-the-art alternatives

Two alternative slides are inserted after the Context divider. Keep either slide 4 or slide 5; they are **alternatives, not two required talks**. Each adds about 30 seconds. The latest stopping and walking videos are preserved and are now at slides 10 and 11 while both alternatives remain.

## Slide 4 — Selected research advances

This slide gives three representative examples rather than a robot ranking. On Cassie, periodic costs on contact forces and foot velocities specify several gaits without prescribing every joint trajectory. On ANYmal, GPU-parallel simulation and a curriculum accelerate training. On Digit, a history-based transformer trained across randomized simulation environments transfers to outdoor walking. These are different robots and tasks: their performance numbers cannot be compared directly with DASH-01. The message for our project is to combine useful gait structure, enough training and early physical validation. Choose this slide OR the following thesis-based alternative.

- [Siekmann et al., ICRA 2021: Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/abs/2011.01387). Image adapted from Fig. 1: cropped Cassie photograph.
- [Rudin et al., CoRL 2021 / PMLR 2022: Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://proceedings.mlr.press/v164/rudin22a.html). Image adapted from Fig. 1: parallel simulation.
- [Radosavovic et al., Science Robotics 2024: Real-world humanoid locomotion with reinforcement learning](https://doi.org/10.1126/scirobotics.adi9579). Image adapted from Fig. 1: cropped outdoor deployment examples.

## Slide 5 — Nathann's literature review

This alternative follows the way Nathann organizes his literature review. The first decision is what the policy commands: torque, position targets or impedance parameters. The second is how much gait structure we provide: a reference motion, a parameterized generator, or a policy that learns the motion directly. The third is the curriculum: how commands, terrain or model variation become harder. DASH-01 uses periodic gait parameters plus rapid learned corrections, joint targets tracked by local PD control, and gated command and randomization curricula. These choices narrow the search and connect to the controller architecture shown next. The green column describes our current code; it is not a hardware capability claim. This is an adaptation of the thesis review, not a copy of unavailable presentation slides. Choose this slide OR the preceding literature overview.

Source: Nathann Morand, *DASH-01: RL Framework to Develop a Bipedal Humanoid Robot Aimed at Breaking the Speed Record* (2026), sections 2.3–2.4, PDF pp. 9–12. The DASH-01 column is our interpretation of the current code. No slides from his unavailable presentation are reproduced.

Supporting primary reference: [Iscen et al., PMTG, CoRL 2018](https://proceedings.mlr.press/v87/iscen18a.html). These are selected relevant papers, not an exhaustive 2026 survey or cross-platform benchmark. No speed record or hardware capability is claimed for DASH-01.
