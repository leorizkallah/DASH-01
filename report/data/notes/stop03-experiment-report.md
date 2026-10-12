# STOP-03: imitation-assisted standing pilot

Date: 2026-10-07. User approved starting the next SCITAS experiment. The request
to optimize credits refers to **Codex usage**, not the SCITAS training allocation.
Use batched checks and automatic reports instead of continuous assistant polling.

## Motivation and hypothesis

STOP-02 finished 2,031,616 PPO steps with 0/32 successful five-second holds at every
post-update evaluation. Its final policy tipped after about 2.05 s even from
near-standing resets. This failure cannot be explained solely by walking momentum.
A fixed zero-action feasibility probe held nominal stance in 31/32 trials, with
reset joint noise disabled. Zero action here commands nominal PD stance; it does
not remove actuator torque. That probe is not a learned or robust stopping policy.

Hypothesis: supervised exposure to this feasible stance can give PPO a better
starting point than discovering standing from a walking policy through reward alone.
The teacher is a limited nominal controller, not a general recovery expert.

## Controlled change

New preset: `dash_stop03_imitation`; run: `dash_stop03_imitation_s0`.
Warm source: `RLframework/results/stop01_replay/dash_stop01b_final.npz`, preserving
its actor, calibrated velocity estimator, and frozen observation normalization.
Do not warm-start from the degraded final STOP-02 policy.

1. Evaluate the source under the existing standing benchmark and walking ladder.
   Keep this walking reference throughout imitation and PPO.
2. Collect 800 ticks at 100 Hz in 32 nominal MJX environments, seed 5303:
   16 HOLD with zero action, 16 RUN with the frozen source actor. RUN commands
   cycle through 0.125, 0.5, 1.0, 1.5, 2.0, and 2.125 m/s.
   Keep reset joint noise at 0.03 rad, matching the prior standing pilot.
   Report teacher survival under that noise rather than assuming it matches the
   earlier noise-free feasibility probe.
3. Exclude termination ticks and everything after the first termination, nonfinite
   data, and HOLD samples with tilt exceeding 10 degrees. Reserve whole environments
   (every fourth index) for action-fitting validation; never split adjacent samples
   from one trajectory across train and validation sets.
4. Train only actor mean layers, 2,000 Adam updates at 1e-4. Each batch has 256 HOLD
   and 256 RUN examples. Loss = HOLD action MSE + 2 x RUN action MSE. Freeze the
   estimator, critic, action standard deviation, and observation statistics during
   imitation. Save demonstrations, update history, held-out errors, and candidate.
5. Evaluate the actual learned actor without a zero-action override, on seed 4102.
   PPO starts only if all gates pass:
   - Source walking survival decreases by no more than 0.05 and command error
     increases by no more than 0.10 m/s, on nonzero commands.
   - Five-second success improves, or mean longest quiet hold improves by at least
     0.05 s. This is an initialization gate, not a stopping-success claim.
   - Held-out HOLD action MSE improves and RUN action MSE is at most 0.01.
6. If accepted, prefit the fresh critic using the existing 16-rollout procedure,
   then run the existing 2M-step standing PPO pilot. If rejected, retain results
   and end the job before PPO. Never silently bypass a failed gate.

STOP-02 smooth ramp, rewards, PPO settings, 75% initial HOLD resets, and physical
five-second acceptance criteria remain unchanged. PPO may adapt the estimator as
before; its freeze above applies only to supervised imitation. The policy receives
the normal RUN/HOLD task fields. There is no runtime stance override or base support.
This pilot isolates initialization; it does not yet train a complete moving STOP
transition or implement a speed/posture-based BRAKE-to-HOLD handoff.

## Budget and validation

One GPU, 10 CPUs, 90 GB memory, two-hour job limit. Demonstration steps (25,600),
supervised updates (2,000), critic-prefit samples, and PPO environment steps are
reported separately. PPO rounds its 2M target to complete rollout batches.
No automatic larger run or extra seed is scheduled.

Local validation: unittest suite and Python compilation. SCITAS job repeats the
tests and performs actual JAX/MJX collection, fitting, invariant checks, and physical
gates. GPU execution is not claimed validated until those stages finish.

Changed implementation: `config.py`, `train.py`, new `standing_imitation.py` and
`imitation_data.py`; stage labeling in `stop_eval.py`; automatic progress rendering,
source packaging and download support; new Slurm script and dataset/gate tests.

## Artifacts and monitoring

Each run writes `imitation.json`, `imitation_demonstrations.npz`,
`imitation_candidate.msgpack`, source `stop_baseline.json`, stage-tagged
`stop_evaluations.jsonl`, and `STOP_PROGRESS.md`. Accepted PPO additionally writes
normal checkpoints, `progress.csv`, `eval.csv`, and the final evaluation.
Reports refresh at evaluations and on batch exit. The original source baseline
remains the walking-retention reference, so forgetting cannot be hidden by resetting
the baseline after imitation. Snapshot manifests record SHA-256 hashes of inputs.

See [submission and measured status](experiments/stop03_20261007_v1/STATUS.md).
Submitted to SCITAS as job **3205920**, snapshot `stop03_20261007_v1`.
All 18 local and remote CPU tests passed before submission; uploaded archive hash
matched. No success measurement exists at submission time; fill results from
persisted metrics after the job runs. Single-seed improvements would still need
repetition. This submission entry was added locally after the source was frozen;
the packaged report preserves the pre-submission protocol.

## Completed pilot: measured results

Job 3205920 completed with exit code 0:0 in 26 min 28 s, reaching 2,031,616 PPO
steps. See the status file for verbatim scheduler timestamps and their discrepancy
with the client date context. Results below come from downloaded JSON/CSV files.

The stance teacher survived 8 s in all 16 demonstration HOLD environments with
0.03 rad reset joint noise. Retained samples: 9,600 HOLD training / 3,200 validation;
8,521 RUN training / 2,187 validation. Held-out HOLD action MSE fell from 0.192867
to 0.00005536; RUN MSE after fitting was 0.00003147. All initialization gates passed.
Teacher survival is distinct from the student's five-second physical holding result.

| Stage | Five-second success | Survived 7 s | Mean longest quiet hold | RUN survival | RUN error |
|---|---:|---:|---:|---:|---:|
| Source | 0/32 | 0/32 | 0.0184 s | 62.50% | 0.3706 m/s |
| Post-imitation | 21/32 | 22/32 | 4.360 s | 75.00% | 0.3538 m/s |
| PPO 524,288 | 21/32 | 22/32 | 4.434 s | 70.83% | 0.3252 m/s |
| PPO 1,048,576 | 14/32 | 30/32 | 4.158 s | 70.83% | 0.2089 m/s |
| PPO 1,572,864 | 14/32 | 30/32 | 3.646 s | 70.83% | 0.2182 m/s |
| PPO 2,031,616 | 14/32 | 31/32 | 3.929 s | 83.33% | 0.2245 m/s |

Every evaluated candidate retained walking against this run's pre-imitation
source reference. `best_stop.msgpack` selects step 524,288: tied with imitation on
success and survival, with lower residual speed (0.01407 versus 0.01461 m/s).
The final checkpoint has higher survival but lower strict holding success.
Final reward-floor fraction fell to 0.000826; improving that metric alone does
not imply improving the physical holding criterion.

Interpretation: these measurements support the hypothesis that feasible stance
demonstrations help initialize standing. Continued PPO did not improve quiet-hold
success: it fell from 65.625% to 43.75% while survival rose. Per-tick traces are
needed to determine which quietness or drift conditions the surviving failures
violate; aggregate metrics do not establish the mechanism. This is one training
seed and one repeated physical evaluation seed. Stopping from motion, restarting,
and hardware behavior are not yet validated. Preserve the best and post-imitation
checkpoints for analysis; no additional run was launched during this status check.

Evidence: [imitation record](experiments/stop03_20261007_v1/imitation.json),
[all physical evaluations](experiments/stop03_20261007_v1/stop_evaluations.jsonl),
[selected checkpoint metrics](experiments/stop03_20261007_v1/best_stop_eval.json),
[PPO progress](experiments/stop03_20261007_v1/progress.csv).
