# STAND-04 submission

**11 November 2026 check: COMPLETED.** All preflights 3247598 and training tasks
3247599 finished, exit 0:0. All models reach **200,005,632 additional steps**.
Scheduler durations: V1 8h37m35s, V2 10h27m56s, X-shape 10h29m37s.
Saved best task passes **60/80, 52/80, 71/80**, respectively, compared with
same-protocol starting baselines **58/80, 48/80, 60/80**. Finals are weaker:
49/80, 37/80, 66/80. All best checkpoints have 16/16 quiet passes.
[Results and curriculum limits](../../../report/standing04-progress.md),
[structured status](../../../report/data/standing04/status-20261111.json).
No new training or replay submitted. The following is the historical submission.

10 November 2026, Europe/Zurich. Training array **3247599** submitted, tasks
`0=V1`, `1=V2`, `2=X-shape`, maximum three concurrently. **PENDING (Dependency)**
on the complete GPU preflight array **3247598**. Failed dependencies cancel
training. The training script also checks all three successful validation JSONs.

Each model fine-tunes its own frozen STAND-03 best checkpoint, retaining weights
and observation normalization with fresh optimizers/curriculum/counters. Parent
steps: **81,100,800 / 73,728,000 / 29,491,200** respectively. Training budget:
**200M additional transitions per model**, seed 1, 24 h scheduler limit and 22.5 h
save-reserve cutoff. Models/masses and physical contact geometry are unchanged.

[Implemented reward, curriculum and anti-shuffling rules](../../../report/standing04-progress.md).
[Source and parents](source-manifest.json), [archive receipt](packaging.json).
Frozen archive SHA-256:
`bd3b6004ab5077ebbf4f014fb29d2cc5a4220f5d31b909f1bc0bf68c22f56fe3`.

Sixteen local tests and Python compilation pass. GPU preflight includes fresh
parent loading, full-variety evaluation, 240-trial curriculum probe and 32 PPO
updates; its weights are discarded. At submission the preflights are queued;
no long-run steps or trained recovery result has been observed.
Abstract JAX reset/step tracing on SCITAS also passes for all three models;
GPU numerical/parent/PPO checks remain queued. [Scheduler receipt](submission-receipt.txt).

Remote root:
`/home/rizkalla/running_robot/experiments/stand04_20261110_v1`.
Use `squeue -j 3247598,3247599` to check both arrays.
