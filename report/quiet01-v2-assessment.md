# QUIET-01 v2: what we have learned

[Inspect the selected policies in the video gallery](quiet01v2-videos.html).

23 November 2026. Inspection of archived baseline, selected best, final, per-level
evaluations, independent retention tests, promotion probes and learning curves.
No new simulation or training was run for this assessment.

## Main conclusion

We now have useful quiet-pose recovery policies in simulation, strongest on the
current V2 and X-shape test banks. They improve on the old standing policies under
the SAME corrected success rule. This is learning progress beyond fixing the old
hold-acceptance bug. It is not yet general push recovery, recovery stepping, walking,
or hardware validation.

| Model | Return: baseline to selected best /96 | Retention: baseline to same selected best /96 | Return from exact reference /16 | Retention from exact reference /16 |
|---|---:|---:|---:|---:|
| V1 | 59 to71 | 53 to62 | 16 | 16 |
| V2 | 37 to85 | 21 to78 | 16 | 15 |
| X-shape | 45 to89 | 40 to78 | 16 | 16 |

Return means complete a timely five-second reference hold and remain within safety
limits through the12s episode. Retention means a separate20s test with timely
initial completion AND a quiet final five-second window. It does not certify
uninterrupted stillness during all20s. All tests use fixed seed1234.

## Where recovery works

The96 tests comprise16 exact-reference trials and16 at each of five perturbation
levels. At the selected checkpoints:

| Initial-pose bank | V1 return /16 | V2 return /16 | X-shape return /16 |
|---|---:|---:|---:|
| Exact reference | 16 | 16 | 16 |
| 0.5deg requested | 16 | 16 | 16 |
| 1.5deg requested | 16 | 16 | 16 |
| 3deg requested | 11 | 16 | 16 |
| 5deg requested | 8 | 14 | 14 |
| 8deg requested | 4 | 7 | 11 |

The angles label requested noise BEFORE projection onto valid closed-linkage poses.
They are not exact motor deviations. For example, the hardest training banks have
mean actual motor RMS deviations3.53deg(V1),2.96deg(V2),2.98deg(X). Thus these scores
compare current policies on their model-specific banks, not an isolated mechanical
comparison of the CAD designs under identical physical displacement.

V2/X-shape return reliably on the smaller tested banks. Long retention is less
reliable: in the3deg-requested bank it is13/16(V2) and12/16(X), despite16/16 return.
Across all banks,8 V2 and11 X-shape retention failures reach the full20s with no
physical/task termination yet fail the final hold. Eliminating falls alone will
not finish the job; maintaining the target posture remains a separate limitation.

## Failure types and learning stability

At the selected checkpoints, mutually exclusive return failures are:

- V1:14 physical falls and11 other task violations out of96.
- V2:2 physical falls and9 other task violations out of96.
- X-shape:1 physical fall and6 other task violations out of96.

Task violations combine hold timeout and planted-foot movement limits; these
records do not separate those causes. Do not call all of them dragging or slips.
The original detailed hold diagnostics concern older policies, not these new weights.

X-shape's best return checkpoint is58.98M steps,89/96; final is85/96. V1 peaks at
66.36M,71/96; final is68/96. V2's final checkpoint is its best,85/96. The last four
return evaluations after75M steps are58–70/96 for V1,76–85/96 for V2 and79–87/96
for X. Training reward is not a sufficient convergence measure because the reset
curriculum also changes task difficulty.

The final frontier is level2(V1) and level3(V2/X). Last promotion probes at the
frontier achieve only8/16(V1),13/16(V2),13/16(X), below the required15/16. The
curriculum has stalled on measured task performance, not on a configured step cap
for those levels. No logged non-finite learning diagnostics are found.

## Interpretation and next decision

X-shape has the strongest selected return result, while V2 has the largest gain
and the same overall selected-checkpoint retention result. Keep both candidates;
V1 is currently weaker on its own bank. Keep the selected weights instead of
automatically using the last weights. A checkpoint selected for return need not
maximize retention: V1's final retention67/96 exceeds its selected-best62/96.

The next useful evidence would be videos of recovery and final holding, followed
by new noise seeds and genuinely new feasible starting poses that were not used
for checkpoint selection. The current fixed test bank was repeatedly used for
selection, so89/96 is not an untouched final-validation estimate or proof of100%
reliability. This assessment does not justify blindly adding more training time
or moving immediately to large pushes.

[Numeric assessment](data/quiet01v2/results/assessment-metrics.json),
[completed-run results](quiet01-v2-results.md),
[learning curves](figures/quiet01v2_convergence.png).
