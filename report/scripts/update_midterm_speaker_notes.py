"""Synchronize user-edited speaker notes with the selected, reordered midterm deck.

Uses the original numbered user notes once; backups are kept under ignored preview/.
Preserves visual slide content except stale manual footer numbers.
"""

import hashlib
import json
import os
from pathlib import Path
import re
from zipfile import ZipFile

from lxml import etree as ET
from pptx import Presentation

FOLDER = Path(__file__).resolve().parents[1] / "presentations/midterm-2026-10-29"
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def title(slide):
    return next(sh.text for sh in slide.shapes if sh.has_text_frame and sh.text)


def encoded(tree):
    return ET.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    deck = FOLDER / "DASH01_controller_midterm_state_of_art_options.pptx"
    notes_file = FOLDER / "speaker-notes_updated.md"
    source_bytes, notes_bytes = deck.read_bytes(), notes_file.read_bytes()
    old_notes = notes_bytes.decode("utf-8")
    prs = Presentation(deck)
    assert len(prs.slides) == 22
    assert title(prs.slides[3]) == "State of the art: learned locomotion"
    assert not any("controller design choices" in title(s) for s in prs.slides)
    blocks = {int(n): (int(seconds), body.strip()) for n, seconds, body in re.findall(
        r"## Slide (\d+) \((\d+) s\)\s*\n(.*?)(?=\n## Slide |\Z)", old_notes, re.S)}
    if len(blocks) != 21:
        raise ValueError("This one-time migration expects the original 21 user-note blocks; use the saved backup to review later revisions.")
    source_numbers = [1, 2, 3, None, 4, 5, 6, 7, None, 8, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19, 20, 21]
    scripts = []
    for index, (slide, old_number) in enumerate(zip(prs.slides, source_numbers, strict=True), 1):
        if old_number is not None:
            seconds, body = blocks[old_number]
        else:
            embedded = slide.notes_slide.notes_text_frame.text
            seconds = int(re.search(r"Suggested time: (\d+) seconds", embedded).group(1))
            body = embedded.split("\n\n", 1)[1].split("Evidence / provenance:", 1)[0].strip()
            body = body.replace(" Choose this slide OR the following thesis-based alternative.", "")
        if index == 2:
            body = body.replace("context, what was done, objectives and timeline, and challenges",
                                "context, what was done, challenges, and objectives and timeline")
        elif index == 13:
            body = "Part 3: Challenges."
        elif index == 16:
            body = "Part 4: Objectives and timeline."
        scripts.append(dict(number=index, seconds=seconds, body=body, title=title(slide),
                            hidden=slide._element.get("show") == "0"))
    assert "failed stopping trials" in scripts[8]["body"]
    assert "These clips have the same target speed" in scripts[9]["body"]
    assert "numerical comparison is encouraging" not in "\n".join(s["body"] for s in scripts)
    assert all("Choose this slide" not in s["body"] for s in scripts)
    total = sum(s["seconds"] for s in scripts if not s["hidden"])
    assert total == 580
    backup = FOLDER / "preview/notes-update-originals"
    backup.mkdir(parents=True, exist_ok=True)
    for name, data in (("speaker-notes_before-selection.md", notes_bytes),
                       ("deck_before-notes-update.pptx", source_bytes)):
        path = backup / name
        if path.exists():
            assert path.read_bytes() == data, "Existing backup differs; do not overwrite it."
        else:
            path.write_bytes(data)
    with ZipFile(deck) as z:
        original = {n: z.read(n) for n in z.namelist()}
        infos = {i.filename: i for i in z.infolist()}
    changes, footer_updates = {}, []
    for slide, script in zip(prs.slides, scripts, strict=True):
        index = script["number"]
        part = str(slide.part.partname).lstrip("/")
        tree = ET.fromstring(original[part])
        changed = False
        for shape in slide.shapes:
            if not (shape.has_text_frame and shape.text.isdigit() and shape.top > 6.4 * 914400):
                continue
            if shape.text == str(index):
                continue
            xml_shape = next(s for s in tree.find("p:cSld/p:spTree", NS)
                             if s.xpath(f'.//p:cNvPr[@id="{shape.shape_id}"]', namespaces=NS))
            runs = xml_shape.findall(".//a:t", NS)
            runs[0].text = str(index)
            for run in runs[1:]:
                run.text = ""
            footer_updates.append(dict(slide=index, previous=shape.text, now=str(index)))
            changed = True
        if changed:
            changes[part] = encoded(tree)
        notes_part = str(slide.notes_slide.part.partname).lstrip("/")
        tree = ET.fromstring(original[notes_part])
        body = next(s for s in tree.findall(".//p:sp", NS)
                    if s.xpath('.//p:ph[@type="body"]', namespaces=NS)).find("p:txBody", NS)
        for paragraph in list(body.findall("a:p", NS)):
            body.remove(paragraph)
        _, separator, evidence = slide.notes_slide.notes_text_frame.text.partition("Evidence / provenance:")
        content = f'Suggested time: {script["seconds"]} seconds.\n\n{script["body"]}'
        if separator:
            content += "\n\nEvidence / provenance:\n" + evidence.strip()
        for line in content.splitlines():
            paragraph = ET.SubElement(body, f'{{{NS["a"]}}}p')
            run = ET.SubElement(paragraph, f'{{{NS["a"]}}}r')
            ET.SubElement(run, f'{{{NS["a"]}}}t').text = line
        changes[notes_part] = encoded(tree)
    assert sha(deck.read_bytes()) == sha(source_bytes)
    output = FOLDER / ".speaker-notes-update.tmp.pptx"
    with ZipFile(output, "w") as z:
        for name, data in original.items():
            z.writestr(infos[name], changes.get(name, data))
    with ZipFile(output) as z:
        assert z.testzip() is None
        for name, data in original.items():
            if name not in changes:
                assert z.read(name) == data, name
    updated = Presentation(output)
    assert len(updated.slides) == 22
    for slide, script in zip(updated.slides, scripts, strict=True):
        assert script["body"] in slide.notes_slide.notes_text_frame.text
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text.isdigit() and shape.top > 6.4 * 914400:
                assert shape.text == str(script["number"])
    for old_slide, new_slide in zip(prs.slides, updated.slides, strict=True):
        for old_shape, new_shape in zip(old_slide.shapes, new_slide.shapes, strict=True):
            if old_shape.has_text_frame and old_shape.text.isdigit() and old_shape.top > 6.4 * 914400:
                continue
            assert ET.tostring(old_shape._element) == ET.tostring(new_shape._element)
    try:
        os.replace(output, deck)
    except PermissionError:
        deck = FOLDER / "DASH01_controller_midterm_speaker_notes_updated.pptx"
        os.replace(output, deck)
    lines = ["# DASH-01 controller midterm - speaker notes", "",
             "Updated 28 October 2026. **State-of-the-art slide 4 retained; thesis alternative removed.**",
             "**19 main slides + three hidden backups; planned speaking time 9 min 40 s**, including the short video sections. Allow for playback and discussion pauses.",
             f"[Matching PowerPoint]({deck.name}). Slide numbers follow the current deck order: stopping failures at slide 9 and walking evolution at slide 10.",
             "Results cutoff: 26 October 2026. Relative roadmap; no new training results or hardware validation are claimed.", ""]
    divider_titles = {"PART 1": "Context", "PART 2": "What was done", "PART 3": "Challenges", "PART 4": "Objectives and timeline"}
    for script in scripts:
        name = divider_titles.get(script["title"], script["title"])
        lines += [f'## Slide {script["number"]} ({script["seconds"]} s)', "",
                  f"**{name}**" + (" - hidden backup" if script["hidden"] else ""), "", script["body"], ""]
    lines += ["## References for slide 4", "",
              "- [Siekmann et al., ICRA 2021 - periodic reward composition on Cassie](https://arxiv.org/abs/2011.01387).",
              "- [Rudin et al., CoRL 2021 / PMLR 2022 - massively parallel locomotion learning](https://proceedings.mlr.press/v164/rudin22a.html).",
              "- [Radosavovic et al., Science Robotics 2024 - real-world humanoid locomotion](https://doi.org/10.1126/scirobotics.adi9579).", ""]
    notes_file.write_text("\n".join(lines), encoding="utf-8")
    receipt = dict(date="2026-10-28", deck=deck.name, source_deck_sha256=sha(source_bytes),
                   final_deck_sha256=sha(deck.read_bytes()), original_notes_sha256=sha(notes_bytes),
                   notes_sha256=sha(notes_file.read_bytes()), selected_state_of_art_slide=4,
                   slides=22, main_slides=19, hidden_backups=3, planned_speaking_seconds=total,
                   footer_updates=footer_updates, slide_content_preserved_except_footer_numbers=True,
                   embedded_notes_match_markdown=True, backup_folder=str(backup.relative_to(FOLDER)))
    (FOLDER / "speaker-notes-update.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(deck=deck.name, slides=22, speaking_time="9 min 40 s",
                         footer_numbers_corrected=len(footer_updates), notes_match_embedded=True), indent=2))


if __name__ == "__main__":
    main()
