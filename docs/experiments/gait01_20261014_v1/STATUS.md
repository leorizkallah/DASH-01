# GAIT-01 status — 14 October 2026

SCITAS job **3219514: FAILED** after 1 min 48 s, exit 1.

The fixed-target GPU standing check failed. **Training never started.**
The robot fell at approximately 4.97 s; maximum tilt 55.67°, drift 0.701 m.
Values remained finite. Local perturbations reproduce the unstable equilibrium.
The exact nominal CPU hold was insufficient evidence of stable balance.

The original [source manifest](source-manifest.json),
[GPU result](gpu-standing.json) and remote source snapshot are preserved.
Successor: [v2 walking pilot](../gait01_20261014_v2/STATUS.md), with numerical
validation and direct walking training, without a standing prerequisite.

Remote root: `/home/rizkalla/running_robot/experiments/gait01_20261014_v1`.
One GPU, four-hour allocation cap. Queue wait is not a training ETA.

See [experiment report](../../gait01-experiment-report.md) and
[source hashes](source-manifest.json).
