# STOP-04: diagnose lost stillness and preserve standing during PPO

Prepared 12 October 2026 after user approval. The user asked to minimize **Codex
credits**, so diagnosis, video capture, training guards and assessment run in one
automated SCITAS job instead of repeated assistant polling.

## Hypothesis and sequence

STOP-03's best hold keeper (524,288 PPO steps) achieved 21/32 five-second holds.
The final policy achieved 14/32 while surviving 31/32. We hypothesize that continued
standing/walking imitation during PPO can reduce forgetting while balance improves.
This is a hypothesis, not an established cause of the earlier regression.

1. Re-evaluate both STOP-03 checkpoints with the same compiled trace evaluator,
   seed 4102, 32 environments. Record physical signals, joint poses, actions and
   first-episode termination. Save traces for replay.
2. For each trial, record speed, angular-motion, joint-motion and tilt violation
   fractions after two seconds. Recompute five-second success while removing one
   condition at a time (also support or drift). These are diagnostic metric
   ablations, not interventions proving causation. Never count auto-resets.
3. Render the lowest-index successful best-policy trial and lowest-index surviving
   final-policy failure, at half speed. Save selection metadata. A render failure
   is recorded separately and does not invalidate the physical trace.
4. Start PPO only if the best still achieves >=50% holds and the final is worse in
   success or mean longest quiet interval by >0.1 s. Otherwise save diagnosis and
   end the job for review; do not silently proceed from an unsuitable checkpoint.
5. Warm-start a short standing pilot from STOP-03 `best_stop.msgpack`. Continue
   supervised rehearsal inside the PPO objective using the original saved
   demonstration **training** partitions. Held-out demonstration trajectories are
   never used for updates.
6. Assess the selected pilot keeper and original warm keeper on seeds 4102, 5102,
   6102. Require >=90% holds on **each** seed and ordinary walking retention before
   considering low-speed transition training. 4102 is the selection seed; 5102
   and 6102 are held out from checkpoint selection. No automatic transition run.

## Exact intervention

Preset `dash_stop04_rehearsal`, run `dash_stop04_rehearsal_s0`, training seed 0.
Source: STOP-03 snapshot `stop03_20261007_v1`, step 524,288 hold keeper. Use full
weights (including the trained critic) and frozen observation statistics; reset
optimizer state, step count and schedules for this new experiment. No critic
prefit or fresh imitation initialization is needed.

One million additional PPO environment steps, rounded to complete 65,536-step
rollouts (expected 1,048,576). 1,024 environments x 64 ticks; evaluate every four
rollouts (262,144 steps); checkpoint every ~250k steps. One GPU, 10 CPU cores,
90 GB, two-hour total job limit including diagnostics and assessment.

Each PPO minibatch draws 128 HOLD and 128 RUN examples from the frozen STOP-03
demonstrations. Add this actor objective with coefficient 1:

```text
L_rehearsal = (MSE_HOLD + 2 * MSE_RUN) / (2 * 0.12^2)
L_total = L_existing_PPO + L_rehearsal
```

The scale is fixed; this is a scaled action MSE, not a Gaussian KL. There is one
joint optimizer update, rather than an extra actor update after PPO. Rehearsal
gradients affect actor mean layers only; estimator outputs are detached as in the
existing actor path. Ordinary estimator training and critic PPO updates continue.
The dataset normalization must exactly match the source checkpoint. Dataset and
checkpoint hashes are checked on resumption and saved for provenance.

Rewards, smooth reference, time-based phase handoff, 0.03 rad joint reset noise,
75% initial HOLD resets, and absence of sensing/plant randomization remain unchanged.
Learning-rate endpoints and warmup settings remain as in STOP-03; schedules span
the shorter new budget. This is a preservation pilot, not a perfectly matched
ablation against STOP-03's original optimizer history or horizon.

## Regression protection

The new run's source baseline is the warm keeper under its current evaluation
graph. Abort after an evaluation if hold success drops by more than 10 percentage
points or walking retention fails. Walking survival may fall by at most five
points and tracking error rise by at most 0.10 m/s relative to that warm baseline.
Save `guard_rejected.msgpack` and `guard_stop.json`; retain the best keeper.
Still perform multi-seed assessment of the retained keeper when available.

## Outputs and validation

`RLframework/results/stop04_diagnosis/`: paired NPZ/JSON traces, `comparison.json`,
`DIAGNOSIS.md`, render logs and videos when rendering succeeds.

`RLframework/runs/dash_stop04_rehearsal_s0/`: `rehearsal.json`, checkpoints,
PPO CSV, physical evaluation JSONL, automatic `STOP_PROGRESS.md`, `multiseed.json`
and `MULTISEED.md`. Input copies and SHA-256 list are in `inputs/` inside the
isolated snapshot. Existing STOP-03 artifacts are not modified.

Local tests cover condition ablations, fall masking, support/drift requirements,
rehearsal scaling and regression gates. A JAX gradient test runs on SCITAS and
checks that rehearsal affects the actor, not estimator, critic or log standard
deviation. The job repeats all tests before physics or training.

Prepared implementation is not a result. Record actual job status and diagnosis
in [the snapshot status](experiments/stop04_20261012_v1/STATUS.md) after submission.
No outcome or improvement is claimed until measured.

## Submission

Submitted as **3215784** in snapshot `stop04_20261012_v1`. All 22 SCITAS CPU tests
passed, including the JAX actor-gradient test (21 passed locally, one skipped).
Source archive hash matched after upload. This entry is appended after source
freezing; the packaged report retains the pre-submission protocol.

## Completed outcome - 12 October 2026

Job 3215784 completed normally (exit 0:0), 11:09:52-11:38:43 Europe/Zurich,
elapsed 28 min 51 s. It reached 1,048,576 new PPO steps. The diagnosis gate passed;
both selected STOP-03 replay videos rendered successfully on SCITAS.

The paired trace graph measured 22/32 holds for STOP-03 best and 13/32 for final.
Removing only the angular-speed quietness condition raised these to 23/32 and
31/32 respectively. Other single-condition removals did not increase final
success. Angular motion is therefore the limiting measured condition for the
surviving final-policy failures in these traces. This does not establish a causal
reward change. Different evaluation graphs give slightly different counts; retain
the original STOP-03 evaluations rather than replacing them with these replays.

STOP-04's selected keeper is step 262,144, with 30/32 holds, 31/32 survivors and
walking retention at selection. Independent post-run reassessment recorded:

| Seed | Source keeper holds | STOP-04 keeper holds | STOP-04 survivors |
|---|---:|---:|---:|
| 4102 | 21/32 | 29/32 | 29/32 |
| 5102 | 19/32 | 32/32 | 32/32 |
| 6102 | 25/32 | 30/32 | 31/32 |

The predefined >=90% hold gate passed on each seed. Seed 4102 was used for keeper
selection; 5102 and 6102 were held out from selection. Walking retention here is
the standard ladder gate, not a multi-seed walking evaluation. This is still one
training seed and a near-stance benchmark, not moving STOP validation.

The two records at final step 1,048,576 differ: periodic evaluation reported
26/32 holds with walking retained; final evaluation reported 30/32 holds with
walking retention false. Both are preserved in the raw JSONL. The periodic
preservation guard did not stop the job; it does not cover the separate final
evaluation branch. Review this reporting/guard gap before reusing the final
checkpoint or extending training. No cause of the repeated-evaluation variation
has been established by this status check.

Downloaded evidence: [three-seed report](experiments/stop04_20261012_v1/MULTISEED.md),
[diagnosis](experiments/stop04_20261012_v1/DIAGNOSIS.md),
[raw training evaluations](experiments/stop04_20261012_v1/stop_evaluations.jsonl).
Trace NPZs and videos remain on SCITAS under `RLframework/results/stop04_diagnosis/`.
No additional training was submitted during this check.
