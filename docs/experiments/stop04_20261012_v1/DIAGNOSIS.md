# STOP-04 paired checkpoint diagnosis

Pilot gate passed: **True**. Seed 4102, 32 near-stance trials.

| Removed condition | Best successful trials | Final successful trials |
|---|---:|---:|
| none | 22/32 | 13/32 |
| speed | 22/32 | 13/32 |
| angular | 23/32 | 31/32 |
| joints | 22/32 | 13/32 |
| tilt | 22/32 | 13/32 |
| support | 22/32 | 13/32 |
| drift | 22/32 | 13/32 |

Counterfactuals remove one measurement condition at a time; all other criteria
and first-episode survival remain required. They do not alter the policy.
Videos select the first successful best-policy trial and first surviving final-policy failure.
Rendering status and checkpoint hashes are in comparison.json.
