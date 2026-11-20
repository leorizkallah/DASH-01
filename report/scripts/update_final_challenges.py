"""Update the controller final deck's challenge slides from Tim's V2 findings.

Only slides 14/15 and their notes are patched; other user edits stay unchanged.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

from lxml import etree as ET
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "report/presentations/midterm-2026-10-29"
SOURCE = FOLDER / "DASH01_controller_midterm_final.pptx"
DEST = FOLDER / "DASH01_controller_midterm_final_mechanical_update.pptx"
REFERENCE = Path("C:/Users/LeoPaul/Downloads/DASH-01 midterm_V2.pptx")
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

NOTES = {
    14: (
        "Tim's first hardware tests revealed a problem that the simulation results could not establish. "
        "The Achilles rod is in tension at the heel. In Tim's report, the printed attachments cannot "
        "carry this load, and play in the leg makes that arrangement unreliable. The robot cannot yet "
        "establish stable standing with this foot design. For the controller, this is a concrete example "
        "of the sim-to-real gap: rigid links and ideal loop constraints do not validate attachment "
        "strength or reproduce real structural flex and joint play. Actuator and contact models exist, "
        "but their match to the assembled hardware still needs validation. We have identified a "
        "mechanical blocker; the remaining mismatch must be measured after repair. More policy "
        "training alone cannot repair the mechanism."
    ),
    15: (
        "With Tim, the proposed repair moves the Achilles to the toe and anchors it above the knee. "
        "Two rods are intended to resist foot torsion. This needs CAD changes, machining and assembly. "
        "On the control side, I then need to update the model, recheck the closed loops and starting "
        "pose, and log supported standing and motion. The hardware kill switch is a prerequisite. "
        "We can retrain and attempt walking only once the repaired robot can support itself reliably."
    ),
}


def set_text(slide, shape_id, value, size=None):
    shape = next(s for s in slide.shapes if s.shape_id == shape_id)
    body = shape.text_frame._txBody
    original = deepcopy(body.findall(f'{{{NS["a"]}}}p')[0])
    for paragraph in list(body.findall(f'{{{NS["a"]}}}p')):
        body.remove(paragraph)
    for line in value.splitlines():
        paragraph = ET.SubElement(body, f'{{{NS["a"]}}}p')
        props = original.find(f'{{{NS["a"]}}}pPr')
        if props is not None:
            paragraph.append(deepcopy(props))
        run = ET.SubElement(paragraph, f'{{{NS["a"]}}}r')
        run_props = original.find(f'{{{NS["a"]}}}r/{{{NS["a"]}}}rPr')
        if run_props is not None:
            run.append(deepcopy(run_props))
        if size is not None:
            rp = run.find(f'{{{NS["a"]}}}rPr')
            if rp is None:
                rp = ET.SubElement(run, f'{{{NS["a"]}}}rPr')
            rp.set("sz", str(round(size * 100)))
        ET.SubElement(run, f'{{{NS["a"]}}}t').text = line
    return shape


def textbox(slide, value, x, y, w, h, size=11, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = value
    paragraph.font.name = "Arial"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = RGBColor.from_string("26323F" if bold else "5A626D")


def xml_bytes(tree):
    return ET.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    reference_hash = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    prs = Presentation(SOURCE)
    assert len(prs.slides) == 22
    slide14, slide15 = prs.slides[13], prs.slides[14]
    assert next(s for s in slide14.shapes if s.shape_id == 3).text == "Where we are stuck"
    values = {
        5: "First restore reliable support, then measure the gap between the repaired robot and simulation.",
        7: "MECHANICAL BLOCKER", 8: "Achilles load path",
        9: "Rod in tension at the heel; printed parts cannot carry the load.",
        11: "REAL ASSEMBLY", 12: "Play changes the support",
        13: "Leg play makes this load path unreliable; stable standing is blocked.",
        15: "UNMODELED EFFECTS", 16: "Beyond rigid-body motion",
        17: "Play, structural flex and real contact response need characterization.",
        19: "CONTROL CONSEQUENCE", 20: "Update the robot model",
        21: "The Achilles redesign needs an updated model before policy transfer.",
    }
    for shape_id, value in values.items():
        set_text(slide14, shape_id, value, size=16 if shape_id in (9, 13, 17, 21) else None)
    next(s for s in slide14.shapes if s.shape_id == 9).width = Inches(5.27)
    # The original bottom cards had room for only one body line.
    for shape_id in (17, 21):
        sh = next(s for s in slide14.shapes if s.shape_id == shape_id)
        sh.top = Inches(5.10)
        sh.height = Inches(.58)
    textbox(slide14, "Mechanical findings: Tim Kobler, midterm V2, slides 14-15 (28 October 2026).",
            .889, 5.88, 11.55, .23, size=11)

    for shape_id, value in {
        5: "REPAIR WITH TIM",
        6: "Move Achilles to toe\nAnchor above knee\nTwo rods to resist torsion",
        7: "UPDATE THE MODEL",
        8: "Import the revised CAD\nCheck loops and workspace\nMatch the starting pose",
        9: "MEASURE, THEN TRANSFER",
        10: "Supported standing first\nLog motion and motor current\nRetrain, then test progressively",
    }.items():
        sh = set_text(slide15, shape_id, value, size=20 if shape_id in (6, 8, 10) else None)
        if shape_id in (6, 8, 10):
            sh.height = Inches(2.12)
    textbox(slide15, "Assembly and hardware kill switch first; walking remains conditional on reliable support.",
            .889, 5.28, 11.55, .63, size=18, bold=True)
    textbox(slide15, "Proposed mechanical repair: Tim Kobler, midterm V2, slide 15. Controller steps are our integration plan.",
            .889, 6.05, 11.55, .25, size=11)

    with ZipFile(SOURCE) as archive:
        original = {n: archive.read(n) for n in archive.namelist()}
        infos = {i.filename: i for i in archive.infolist()}
    changed = {}
    evidence = (
        "Evidence / provenance:\nTim Kobler, DASH-01 midterm_V2.pptx, slides 14-15 and speaker notes. "
        "Findings: heel Achilles tension, insufficient printed-part load capacity, leg play, inability to stand. "
        "Proposed repair: toe attachment, above-knee anchor, two rods, aluminium plates and a new dowel pin. "
        "Unmodeled-effects interpretation and controller sequence are our synthesis, not additional test results. "
        "No compliance/backlash/contact parameters have been measured or changed by this presentation edit."
    )
    for index, slide in ((14, slide14), (15, slide15)):
        changed[str(slide.part.partname).lstrip("/")] = xml_bytes(slide._element)
        part = str(slide.notes_slide.part.partname).lstrip("/")
        notes = ET.fromstring(original[part])
        body = next(s for s in notes.findall(".//p:sp", NS)
                    if s.xpath('.//p:ph[@type="body"]', namespaces=NS)).find("p:txBody", NS)
        for paragraph in list(body.findall("a:p", NS)):
            body.remove(paragraph)
        for line in (NOTES[index] + "\n\n" + evidence).splitlines():
            paragraph = ET.SubElement(body, f'{{{NS["a"]}}}p')
            run = ET.SubElement(paragraph, f'{{{NS["a"]}}}r')
            ET.SubElement(run, f'{{{NS["a"]}}}t').text = line
        changed[part] = xml_bytes(notes)
    with ZipFile(DEST, "w") as archive:
        for name, data in original.items():
            archive.writestr(infos[name], changed.get(name, data))
    with ZipFile(DEST) as archive:
        assert archive.testzip() is None
        for name, data in original.items():
            if name not in changed:
                assert archive.read(name) == data, name
    checked = Presentation(DEST)
    assert len(checked.slides) == 22
    assert all(NOTES[i] in checked.slides[i-1].notes_slide.notes_text_frame.text for i in (14, 15))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == reference_hash
    notes_path = FOLDER / "speaker-notes_updated.md"
    markdown = notes_path.read_text(encoding="utf-8")
    for index, heading in ((14, "Where we are stuck"), (15, "What we need")):
        pattern = rf'(## Slide {index} \(\d+ s\)\s*\n).*?(?=\n## Slide |\n## References|\Z)'
        replacement = rf'\g<1>\n**{heading}**\n\n' + NOTES[index] + "\n"
        markdown, count = re.subn(pattern, replacement, markdown, flags=re.S)
        assert count == 1
    markdown = re.sub(r'\[Matching PowerPoint\]\([^)]+\)', f'[Matching PowerPoint]({DEST.name})', markdown)
    notes_path.write_text(markdown, encoding="utf-8")
    receipt = dict(date="2026-10-28", source=SOURCE.name, source_sha256=source_hash,
                   reference=str(REFERENCE), reference_sha256=reference_hash, reference_slides=[14, 15],
                   output=DEST.name, output_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
                   changed_slides=[14, 15], changed_parts=list(changed), all_other_parts_byte_identical=True,
                   sources_preserved=True, observations_separated_from_unmeasured_effects=True)
    (FOLDER / "mechanical-challenges-update.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
