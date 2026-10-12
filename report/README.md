# DASH-01 stopping-control report workspace

Updated **12 October 2026**. This folder gathers the evidence needed for the final
report and presentations. It describes the recorded STOP-01 to STOP-03 work; no
new training was launched to build it.

**Current result:** imitation enabled a five-second quiet hold in **21/32**
near-standing trials. The best recorded STOP-03 hold checkpoint is PPO step
**524,288**. The final checkpoint survived more often but held still successfully
in only **14/32** trials. Reliable stopping from motion remains unvalidated.

Approved next experiment: [STOP-04 diagnostic and preservation pilot](data/notes/stop04-experiment-report.md),
submitted as job **3215784**. Its results are pending; the figures below remain
the recorded STOP-03 evidence.

Start with [the visual gallery](index.html) for figures and playable videos, or
use these documents:

| Document | Purpose |
|---|---|
| [Experiment story](experiments.md) | What changed, why, and what happened |
| [Current control architecture](architecture.md) | Sensors, networks, gait, motors, STOP handling |
| [Measurement and interpretation](evaluation.md) | Exact success criteria and limits of the evidence |
| [Figure and video guide](assets.md) | Captions, selection rationale, provenance and missing assets |
| [Presentation outline](presentation.md) | Eight-slide narrative and speaker notes |
| [Work log and next questions](work-log.md) | Decisions, unresolved questions and update procedure |

![Standing performance](figures/stop03_outcomes.png)

## Folder conventions

- `figures/`: plots and architecture diagram, each as PNG, SVG and PDF. Use PDF/SVG
  in the final report, PNG in slides.
- `videos/`: copies of existing simulation clips, with explicit labels in the guide.
- `data/`: copied metrics, configuration, selected code and historical notes.
  Original experiment folders remain authoritative; copies make this collection portable.
- `tables/`: generated CSV summaries for report tables.
- `scripts/build_report.py`: copies inputs, rebuilds plots/gallery/tables and records hashes.
- `manifest.json`: source paths, SHA-256 hashes, video metadata and generated outputs.

From the repository root, regenerate the collection without connecting to SCITAS:

```powershell
.\.venv\Scripts\python.exe report/scripts/build_report.py
```

The builder uses the currently downloaded metrics. Updating remote results is a
separate, deliberate step. It does not change policy weights, rewards or training
code. Authored explanations and decisions should be updated when new evidence arrives.

After a fresh Git clone, use `--frozen` to rebuild from the committed report inputs.
Raw pose traces and transfer archives are kept locally/on SCITAS; see the
[artifact policy](../docs/experiments/README.md). Curated figures and videos remain
versioned so this gallery can be opened immediately.

Raw source records: [STOP-01 log](../docs/stop-training-report.md),
[STOP-02 log](../docs/stop02-experiment-report.md),
[STOP-03 log](../docs/stop03-experiment-report.md).
