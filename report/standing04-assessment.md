# STAND-04 assessment: convergence is not established

11 November 2026. Based on archived training, fixed evaluation and independent
curriculum-probe logs. No training code or running jobs changed for this review.
Videos are handled by the existing replay/render pipeline.

## Measured evidence

| Model | Parent under new test | Saved best | Final | Task-score range after 150M |
|---|---:|---:|---:|---:|
| V1 | 58/80 | 60/80 | 49/80 | 61.25–75% |
| V2 | 48/80 | 52/80 | 37/80 | 0–62.5% |
| X-shape | 60/80 | 71/80 | 66/80 | 48.75–82.5% |

All scores use STAND-04's fixed full-variety protocol, including final five-second
quiet standing. Selection also prioritizes checkpoints with >=95% quiet success;
the selected best need not be the maximum combined score of an unqualified policy.
The parent and continuation use the same new evaluation protocol. They should
not be compared directly to STAND-03's earlier score definitions.

V1 shows a small best-checkpoint gain; V2 a modest gain; X-shape the largest gain
on this test set. All finals are weaker than the selected best. These are one-seed,
repeated fixed-set results; best-checkpoint selection and sampling variability
limit claims of general improvement.

Over the last 1,000 updates, mean rewards are approximately 5.987 / 6.008 / 6.025.
Quiet reward contributions are 1.846 / 1.846 / 1.877 out of a maximum 2, and
support contributions are 1.492 / 1.500 / 1.500 out of 1.5. Those smooth rewards
look strong even while complete quiet/recovery success remains variable.

Archived passing trials with at least one detected whole-foot landing: **0/60,
0/52, 2/71**, respectively. A landing event is not proof of a useful displaced
recovery step; traces/videos are needed for that. The data do not establish a
general recovery-stepping behavior. Final push levels are .1328 / .0200 / .0200
m/s-equivalent; V2 and X-shape did not advance to full disturbance variety.

## Working diagnosis, with uncertainty

1. **Reward plateau and task convergence differ.** A short wobble can reset the
   required five-second quiet streak while barely affecting mean per-tick reward.
   That is a plausible reward/objective mismatch, not proof that any single
   coefficient is wrong. Per-trial failure reasons should separate physical
   falls, workspace/quiet-step violations and incomplete settling.
2. **The quiet gate can reduce useful push exposure.** In the current curriculum,
   two batches with <90% no-push quiet success reduce impulse, even if pushed
   trials succeed. V2's last probe is 97% push success and 82.5% quiet success.
   Reducing push strength cannot directly remove a no-push failure. Preserving
   quiet competence is necessary, but coupling this to impulse retreat can leave
   a policy training on disturbances too weak to require a recovery step.
3. **The inherited planted strategy dominates.** The rarity of landings and easy
   late-training pushes suggest that recovery-step discovery is limited. Step
   costs or exploration may also contribute; their causal role has not been
   isolated. Stronger penalties alone are not justified by these observations.
4. **Policy-update stability needs checking.** V2 has a zero-score evaluation at
   +184.32M and later recovers. That establishes instability on the fixed test,
   not its cause. Separately, V1 logs three infinite approximate-KL diagnostics
   at +3.078144M, +6.248448M and +14.505984M. The code calculates
   `log(exp(new_log_prob-old_log_prob))`, which can underflow before taking the
   logarithm. These diagnostic infinities do not prove nonfinite policy weights
   or explain V2's later collapse. Stabilize the calculation and explicitly
   reject nonfinite or excessive policy changes before another long run.

## Proposed next experiment, not launched

- Inspect the pending videos and retain the existing best checkpoints.
- Audit actual settling failure reasons and step/foot-motion measurements before
  modifying reward weights. Make sustained settling success more influential
  relative to receiving high dense reward while only nearly still.
- Separate quiet-retention failures from push-strength retreat: pause promotion
  and repair/preserve quiet competence without automatically making already
  successful pushes easier. Keep no-push replay and the anti-shuffling rules.
- Use a short, controlled continuation at a fixed challenging push distribution,
  with stable policy-change diagnostics, tighter update limits and validation
  rollback. Verify that useful recovery steps appear and that independent quiet
  retention improves before spending another 200M steps on every model.

PPO clipping alone is not sufficient to prevent large updates; explicit KL
limits are used in [Stable-Baselines3's official PPO implementation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters).
This supports considering update guards, but does not establish that excessive
updates caused the behavior of these particular runs.

Sources: `report/data/standing04/results/*/progress.jsonl`,
`recovery_evaluations.jsonl`, `recovery_curriculum_probes.jsonl`, `best.json`,
`BalanceRL/recovery.py`, `recovery_logic.py`, `recovery_curriculum.py`, and `ppo.py`.
