# STAND-04 videos, convergence and push strengths

[Open the media gallery](standing04-videos.html). Figures are available as PNG/PDF/SVG.

| Model | Final training Δv | Force at 100 ms | Level at best checkpoint | Best additional step |
|---|---:|---:|---:|---:|
| Original V1 | 0.1328 m/s | 19.06 N | 0.0998 m/s | 169.57M |
| V2 | 0.0200 m/s | 2.92 N | 0.0262 m/s | 58.98M |
| X-shape | 0.0200 m/s | 2.91 N | 0.0200 m/s | 132.71M |

Δv is impulse divided by robot mass. Forces illustrate the upper level for a 100 ms pulse; actual durations and strengths vary. The curriculum setting is not a measured maximum capability.

Best policies are replayed with the frozen training code, stage-3 disturbances, full sensor noise, seed 1234 and 80 first-episode trials. Quiet trial 0 and +X 0.20 m/s trial 48 were selected before outcomes. Videos show both independent trials side by side at 1× speed, with actual pulse timing. Early endings freeze visibly, without reset splicing.

## Original V1

Archived best: 60/80; successful trials with detected whole-foot landings: 0/60.

Independent replay: 59/80. Replay per-level rates: {'0.00': 0.9375, '0.10': 1.0, '0.15': 0.875, '0.20': 0.5625, '0.30': 0.3125}. Archived rates: {'0.00': 1.0, '0.10': 1.0, '0.15': 0.875, '0.20': 0.5625, '0.30': 0.3125}.

- Quiet trial: PASS, 20.00 s, 0 landings.
- Pushed trial: FAIL, 6.90 s, 0 landings.

[Video](videos/standing04_v1_best.mp4) · [learning curves](figures/standing04_v1_convergence.png).

## V2

Archived best: 52/80; successful trials with detected whole-foot landings: 0/52.

Independent replay: 52/80. Replay per-level rates: {'0.00': 1.0, '0.10': 0.9375, '0.15': 0.5625, '0.20': 0.5625, '0.30': 0.1875}. Archived rates: {'0.00': 1.0, '0.10': 0.9375, '0.15': 0.625, '0.20': 0.5, '0.30': 0.1875}.

- Quiet trial: PASS, 20.00 s, 0 landings.
- Pushed trial: FAIL, 6.80 s, 0 landings.

[Video](videos/standing04_v2_best.mp4) · [learning curves](figures/standing04_v2_convergence.png).

## X-shape

Archived best: 71/80; successful trials with detected whole-foot landings: 2/71.

Independent replay: 72/80. Replay per-level rates: {'0.00': 1.0, '0.10': 0.875, '0.15': 0.9375, '0.20': 0.9375, '0.30': 0.75}. Archived rates: {'0.00': 1.0, '0.10': 0.875, '0.15': 0.9375, '0.20': 0.9375, '0.30': 0.6875}.

- Quiet trial: PASS, 20.00 s, 0 landings.
- Pushed trial: PASS, 20.00 s, 0 landings.

[Video](videos/standing04_xshape_best.mp4) · [learning curves](figures/standing04_xshape_convergence.png).

## Evidence

Raw training/evaluation/probe logs, configs, best checkpoint weights, scheduler receipts, CSV tables and hashes are in `report/data/standing04/results/`. Replay job: 3248509; training array: 3247599. No new training submitted. Replay differences are retained rather than hidden. One seed per model; X-shape inertials remain provisional.
