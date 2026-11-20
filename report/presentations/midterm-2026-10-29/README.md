# DASH-01 controller midterm

**Latest final presentation, mechanical challenges revised:** [PowerPoint](DASH01_controller_midterm_final_mechanical_update.pptx).
Based on the user's `DASH01_controller_midterm_final.pptx`. Slides 14/15 now
connect Tim's observed Achilles/printed-part/leg-play problem to unmodeled effects,
the proposed mechanical repair, model updates and supported hardware validation.
The remaining slide parts/media are byte-identical. Embedded notes and
[speaker-notes_updated.md](speaker-notes_updated.md) include the revised scripts.
[Source findings and validation](mechanical-challenges-update.json) distinguish
reported mechanical observations from effects still requiring characterization.

**Selected version, 28 October:** [PowerPoint with synchronized speaker notes](DASH01_controller_midterm_speaker_notes_updated.pptx)
and [speaker-notes_updated.md](speaker-notes_updated.md). The user retained the
illustrated state-of-the-art slide 4 and removed the thesis alternative.
19 main slides + three hidden backups; planned speaking time **9 min 40 s**.
Notes follow the actual reordered deck: stopping videos at 9, walking at 10,
Challenges before Objectives and timeline. Embedded notes and manual footer numbers
are synchronized. Saved a separate matching copy because the source file could not be replaced.
[Update record](speaker-notes-update.json); previous deck/notes backed up under ignored `preview/notes-update-originals/`.

**Two state-of-the-art alternatives, 28 October:** [PowerPoint](DASH01_controller_midterm_state_of_art_options.pptx).
Inserted after the Context divider: **slide 4** is an illustrated literature overview
(Cassie, ANYmal, Digit); **slide 5** adapts Nathann's thesis review into three
controller design axes. Choose one and delete the other. Each is planned for
about 30 seconds. Sources and notes are embedded and collected in
[state-of-art-notes.md](state-of-art-notes.md); [provenance](state-of-art-update.json)
records paper figures, crop coordinates and hashes. Existing slides and videos
are preserved, with footer numbers shifted; stopping/walking clips are now at
slides 10/11 while both alternatives remain. The source deck is unchanged.

**Latest, 28 October:** [reordered deck with stopping failures](DASH01_controller_midterm_reordered_stopping.pptx).
Based on the user-edited `DASH01_controller_midterm_reordered.pptx`, which is preserved.
Slide 8 now shows two embedded failed stopping trials on the previous robot:
abrupt STOP (0.5x playback) and the same policy with a smooth reference (1x).
The previous slide 8 walking comparison is now slide 9. Other slide content,
ordering, masters and theme are preserved; the old speed-tracking slide is replaced.
[Changed-slide notes](stopping-slide-notes.md) · [validation and source hashes](stopping-slide-update.json).
The template PDF below predates this update.

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
