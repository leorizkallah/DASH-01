# GAIT-08 reward audit: micro-steps

18 October 2026. Scope: inspect the reward used by the completed GAIT-08 run,
following the user's observation of extra small foot movements in the videos.
No reward, policy, configuration or training job changed.

## Main finding

The current reward encourages alternating **valid** steps, but does not reliably
discourage extra incomplete swings between them. Therefore 10 left / 10 right
valid steps and perfect valid-step alternation do not establish a clean gait.
Increasing the existing repeat-step weight alone would leave this loophole open.

SHA-256 checks confirm that local `config.py`, `env.py`, `bilateral_gait.py` and
`gait_selection.py` match the GAIT-08 training source manifest.

## Confirmed issues

| Issue | Current implementation | Consequence |
| --- | --- | --- |
| Repetition ignores incomplete steps | A valid touchdown requires at least 80 ms airborne, 1.5 cm peak whole-sole clearance and 4 cm forward body-relative travel. `bilateral_repeat` uses only these valid touchdowns. | A left valid step, several small left movements, then a right valid step can still look perfectly alternating to this metric. |
| Progress can restart before completing a turn | Forward-progress high-water mark resets on every airborne bout; partial lift and reach rewards do not require a valid landing. The next-foot mask restricts these rewards to the expected foot, but does not limit its retries. | Several incomplete attempts by the expected foot can repeatedly earn partial credit. Wrong-foot retries do not earn these two bonuses, but also escape the repeat penalty if invalid. |
| Stance-duration cost conflicts with slow cadence | The gait uses a 65% stance ratio. At 0.5 Hz, nominal stance is 1.3 s, but the slow-speed contact-duration penalty starts at 1.0 s. | The reward can encourage interrupting a stance that the phase schedule still requests. This is a plausible incentive for extra lifting, not proof of the video's cause. |
| Symmetry compares inconsistent event histories | `last_stance` updates on every debounced liftoff; `last_air` and `last_length` update only at valid touchdowns. | A small extra lift can replace the stance sample without replacing the corresponding air/length samples. The symmetry score does not consistently describe matched complete strides. |

Code: [step validity and history](../RLframework/bilateral_gait.py#L62),
[partial progress](../RLframework/bilateral_gait.py#L115),
[sequence rule](../RLframework/bilateral_gait.py#L145),
[stance duration and contact switching](../RLframework/env.py#L1323).

There **are** other costs on poor movement: clearance, phase/contact mismatch,
missing valid steps, a generic contact-switch penalty, and contact sliding.
Incomplete steps are not free overall. The generic switch weight is only 0.2
per change before shaping and timestep scaling, versus a valid repeat-event
weight of 2.0. Moreover, these use different contact definitions: the generic
path includes substep contact and a 5 mm near-ground threshold; the bilateral
path uses geometric contact with a two-control-tick debounce. These are not
interchangeable event counters.

## Reproduction

Run from the repository root:

```powershell
.venv/Scripts/python report/scripts/audit_gait08_microsteps.py
```

[Script](../report/scripts/audit_gait08_microsteps.py),
[saved results](../report/tables/gait08_microstep_audit.json).

The script calls the actual NumPy-compatible bilateral reward with synthetic
contact, height and position inputs. Left is set as the last valid foot; right
is therefore expected. Each micro-step has 120 ms airborne, 6 mm height and
2 cm forward travel, followed by 40 ms grounded.

| Input | Measured result |
| --- | --- |
| Three left micro-steps | Three debounced landings, zero valid steps, zero repeat cost. |
| Three right micro-steps | Three debounced landings, zero valid steps, zero repeat cost; raw accumulated lift bonus 3.96 and forward-progress bonus 3.60. |
| One left qualifying repeat: 6 cm travel, 2 cm height | One valid step and raw repeat cost -2.0. |
| Small lift following a short stance | Stance history changes while air/length history stays unchanged; synthetic stance mismatch becomes 1.8835 despite unchanged equal air/length samples. |

The micro-step sequence fits inside the 1.2 s next-foot grace at 0.5 Hz, so no
overdue penalty applies. Bonuses above are isolated sums before shaping and
timestep scaling, **not total return**. Other bilateral costs remain negative.
These checks establish bookkeeping behavior; they neither simulate a feasible
robot trajectory nor prove the cause or count of micro-steps in the videos.

## Other observations

- The weaker-foot quality average and pair-progress bonus also accept partial
  lifts, independently of the next-foot mask. This was useful for initial
  learning, but is insufficient as the final clean-step objective.
- `duty_sym` is a minimum-contact-duty guard, not a left/right equality penalty.
  Actual symmetry costs are in the bilateral module.
- The controller permits higher harmonics and per-joint residual actions.
  Its periodic clock does not mechanically enforce one swing per foot per cycle.
- The bilateral reward uses exactly half-cycle separation, while the controller
  permits a learned phase-offset knob. This is a possible additional conflict;
  the existing replay trace does not establish whether that knob is significant.
- Candidate ranking uses qualification, bilateral progress, survival and speed
  tracking. Among candidates failing qualification, symmetry has no separate
  ranking term. Thus "best candidate" is not synonymous with cleanest-looking gait.
- Contact drag already measures horizontal motion at active floor contact points,
  including forward, backward and sideways sliding. It cannot penalize a tiny
  airborne hop as sliding. More drag weight alone does not close the sequence gap.

## Recommended next change

1. Track all meaningful whole-foot lift/landing attempts separately from valid
   steps. Keep contact hysteresis/debounce so toe-to-heel rolling and numerical
   contact chatter are not treated as extra steps.
2. Add a bounded incomplete-step cost at landing, increasing smoothly with
   deficient clearance/travel. Penalize meaningful extra attempts by the previous
   foot while the opposite foot is due, even when those attempts are invalid.
   Keep the original deadline; an unsuccessful attempt must not reset it.
3. Budget partial progress over an entire expected-foot turn. Touching down and
   restarting must not replenish that budget. Retain some early-learning partial
   credit, then gradually increase the incomplete/extra-step cost as walking improves.
4. Make prolonged-stance allowance consistent with cadence and stance ratio,
   with bounded grace. Update symmetry from consistently paired stride events;
   report extra attempts separately rather than mixing incompatible samples.
5. Keep speed tracking, bilateral activity, clearance and all-direction contact
   drag. Avoid penalizing all small foot motion: controlled landing adjustments
   and legitimate shorter steps at lower speeds must remain possible.
6. Log all attempts, rejected-step reasons, travel/clearance distributions and
   extra same-foot attempts. Compare them with valid-step alternation and video.
   Check reward budgets before choosing weights or launching further training.

Priority is to fix event accounting and conflicting incentives before increasing
penalty weights or training duration. These are proposed changes, not implemented.
