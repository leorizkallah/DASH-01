# DASH-01 controller midterm

[PowerPoint deck — Tim’s template](DASH01_controller_midterm_Tim_template.pptx) · [PDF preview](DASH01_controller_midterm_Tim_template.pdf) ·
[Speaker notes](speaker-notes.md) · [Evidence and source references](sources.md)

Revised for Léo-Paul Rizkallah using **Tim Kobler’s actual PowerPoint template**:
the original COVER, LIGHT and DARK layouts, theme, masters, cover composition,
outline, four section dividers/navigation, Arial typography and closing slide.
Controller-specific results and embedded videos replace the mechanics content.
The first draft is retained locally as `DASH01_controller_midterm.pptx` and its
PDF; these superseded drafts and regenerable previews are ignored by Git.
Presentation date: 29 October 2026, as in the reference deck. Data reviewed through
26 October. Timeline uses relative milestones, per the user’s preference.

The **18 main slides are timed for about 9 minutes 35 seconds**; this includes
the outline, four brief section dividers and closing slide. Three hidden backup slides
cover reward definitions, the standalone replay failure and exact comparisons.
In PowerPoint, use Slide Show and click a video to play it. The four H.264 video
clips are **embedded**: the PPTX can be shared on its own. The PDF contains the
18 main slides with static video posters; videos do not play in the PDF.

The deck covers Nathann’s inherited controller, model adaptation and experiment
lessons, walking/flat-feet evolution, speed tracking and clearance measurements,
the original startup → commanded walking → STOP demo, early treadmill tests and
the sim-to-real gap. It credits the existing architecture and separates simulation
results from planned hardware capabilities. CLEARANCE-02 is only described as
submitted; no live training monitoring was performed to create this presentation.

Videos show nominal simulation, not hardware. Main clips are 12 s excerpts from
6–18 s of the archived replays, at 1× speed. Display headers/footers are cropped;
robot motion is unchanged. Both camera views show the same instant. The startup
failure is retained in backup slide 20. Tim’s original PPTX remains unchanged.

## Rebuild

Use the project Python environment with `python-pptx==1.0.2`, matplotlib, Pillow,
imageio and imageio-ffmpeg:

```powershell
.venv/Scripts/python -m pip install -r report/requirements.txt
.venv/Scripts/python report/scripts/build_midterm_controller.py
powershell -NoProfile -ExecutionPolicy Bypass -File report/scripts/export_midterm_controller.ps1
```

Each build loads `C:/Users/LeoPaul/Downloads/DASH01_midterm.pptx` to reuse the
actual template; use `--reference PATH` if it moves. Extracted assets are reused
from `assets/`. The delivered PowerPoint is self-contained and does not require
the reference file for viewing.
PowerPoint is required for the export script. It opens the generated deck without
a presentation window, exports previews/PDF, and closes only that deck.

Editable diagrams and text are native PowerPoint objects. Plot data are read from
the archived experiment JSON/JSONL; PNG and SVG versions are in `assets/`.
`manifest.json` records inputs and video hashes. `validation.json` records the
final slide/media/PDF checks. Previews are in `preview/`.
