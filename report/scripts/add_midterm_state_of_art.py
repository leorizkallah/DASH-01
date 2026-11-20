"""Add two alternative literature slides without rewriting user-edited slide content.

Option A uses attributed figures from primary publications. Option B follows
Nathann Morand's thesis, Chapter 2. Source PDFs are local research-cache inputs.
"""

from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import posixpath
from zipfile import ZipFile

from lxml import etree as ET
from pptx import Presentation
import pymupdf

from build_midterm_controller import text, rect, picture, note, replace_template_text

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "report/presentations/midterm-2026-10-29"
ASSETS = FOLDER / "assets/state-of-art"
SOURCE = FOLDER / "DASH01_controller_midterm_reordered_stopping.pptx"
DEST = FOLDER / "DASH01_controller_midterm_state_of_art_options.pptx"
THESIS = Path("C:/Users/LeoPaul/Downloads/Master_Thesis__Nathann_Morand-5.pdf")
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"p": P, "a": A}
PAPERS = [
    dict(key="siekmann2021", title="Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition",
         citation="Siekmann et al., ICRA 2021", url="https://arxiv.org/abs/2011.01387",
         figure="Fig. 1: cropped Cassie photograph", page=0, crop=[487, 264, 556, 375]),
    dict(key="rudin2022", title="Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning",
         citation="Rudin et al., CoRL 2021 / PMLR 2022", url="https://proceedings.mlr.press/v164/rudin22a.html",
         figure="Fig. 1: parallel simulation", page=0, crop=[117.9, 432.3, 494.1, 577.3]),
    dict(key="radosavovic2024", title="Real-world humanoid locomotion with reinforcement learning",
         citation="Radosavovic et al., Science Robotics 2024", url="https://doi.org/10.1126/scirobotics.adi9579",
         figure="Fig. 1: cropped outdoor deployment examples", page=1, crop=[36, 61, 557, 323]),
]


def xml_bytes(element):
    return ET.tostring(element, xml_declaration=True, encoding="UTF-8", standalone=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_figures():
    for paper in PAPERS:
        pdf = ASSETS / (paper["key"] + ".pdf")
        with pymupdf.open(pdf) as doc:
            doc[paper["page"]].get_pixmap(matrix=pymupdf.Matrix(3, 3),
                                         clip=pymupdf.Rect(paper["crop"])).save(ASSETS / (paper["key"] + ".png"))
        paper["pdf_sha256"] = digest(pdf)
        paper["image_sha256"] = digest(ASSETS / (paper["key"] + ".png"))


def base(prs, title, number, subtitle):
    slide = prs.slides.add_slide(next(l for l in prs.slide_layouts if l.name == "LIGHT"))
    for shape in list(slide.shapes):
        slide.shapes._spTree.remove(shape._element)
    for original in (prs.slides[3].shapes[0], prs.slides[3].shapes[1]):
        slide.shapes._spTree.insert_element_before(deepcopy(original._element), "p:extLst")
    replace_template_text(slide.shapes[0], title)
    replace_template_text(slide.shapes[1], str(number))
    text(slide, subtitle, .889, 1.63, 11.55, .36, size=18, color="5A626D")
    return slide


def build_options(prs):
    first = base(prs, "State of the art: learned locomotion", 4,
                 "Selected advances: gait design, scalable learning and real-world transfer")
    cards = [
        ("Cassie", "Specify the gait", "Periodic force and velocity costs\nlearn different bipedal gaits.", "Siekmann et al. [1]", PAPERS[0]),
        ("ANYmal", "Scale the training", "Parallel simulation + curriculum\naccelerate locomotion learning.", "Rudin et al. [2]", PAPERS[1]),
        ("Digit", "Transfer to hardware", "Observation history + randomized\ntraining support outdoor walking.", "Radosavovic et al. [3]", PAPERS[2]),
    ]
    for i, (robot, heading, body, citation, paper) in enumerate(cards):
        x = .889 + i * 3.94
        rect(first, x, 2.16, 3.68, 3.48, color="FFFFFF", radius=True)
        text(first, heading, x + .16, 2.30, 3.36, .34, size=21, bold=True)
        rect(first, x + .15, 2.82, 3.38, 1.63, color="E8ECEA")
        picture(first, ASSETS / (paper["key"] + ".png"), x + .15, 2.82, 3.38, 1.63)
        text(first, robot, x + .16, 4.55, 3.36, .28, size=17, bold=True, color="E2001A")
        text(first, body, x + .16, 4.90, 3.36, .65, size=15)
        label = text(first, citation, x + .16, 5.72, 3.36, .25, size=12, color="5A626D")
        label.text_frame.paragraphs[0].runs[0].hyperlink.address = paper["url"]
    text(first, "DASH-01: adapt these principles to our morphology, actuators and demo sequence.",
         .889, 6.08, 11.55, .38, size=18, bold=True)
    notes_a = (
        "This slide gives three representative examples rather than a robot ranking. On Cassie, "
        "periodic costs on contact forces and foot velocities specify several gaits without prescribing "
        "every joint trajectory. On ANYmal, GPU-parallel simulation and a curriculum accelerate training. "
        "On Digit, a history-based transformer trained across randomized simulation environments transfers "
        "to outdoor walking. These are different robots and tasks: their performance numbers cannot be "
        "compared directly with DASH-01. The message for our project is to combine useful gait structure, "
        "enough training and early physical validation. Choose this slide OR the following thesis-based alternative."
    )
    note(first, 30, notes_a, [f'[{i+1}] {p["citation"]}. {p["title"]}. {p["url"]}. Figure: {p["figure"]}.'
                            for i, p in enumerate(PAPERS)])

    second = base(prs, "State of the art: controller design choices", 5,
                  "Literature framework from Nathann Morand's thesis, Chapter 2")
    text(second, "DESIGN AXIS", 1.05, 2.17, 2.30, .27, size=13, bold=True, color="5A626D")
    text(second, "CHOICES IN THE LITERATURE", 3.80, 2.17, 5.0, .27, size=13, bold=True, color="5A626D")
    text(second, "DASH-01 TODAY", 9.18, 2.17, 3.0, .27, size=13, bold=True, color="E2001A")
    rows = [
        ("Motor interface", "Torque / joint targets / impedance", "Joint targets\n+ local PD control"),
        ("Gait structure", "Reference motion / generator / free-form policy", "Periodic generator\n+ fast learned corrections"),
        ("Curriculum", "Commands / terrain / model randomization", "Gated speed commands\n+ domain randomization"),
    ]
    for i, (axis, choices, current) in enumerate(rows):
        y = 2.65 + 1.03 * i
        rect(second, .889, y, 11.55, .87, color="FFFFFF", radius=True)
        rect(second, .889, y, .055, .87, color="E2001A")
        text(second, axis, 1.06, y + .24, 2.45, .34, size=20, bold=True)
        text(second, choices, 3.80, y + .25, 5.13, .43, size=16)
        text(second, current, 9.18, y + .14, 3.06, .60, size=16, bold=True, color="176A55")
    text(second, "Key trade-off: useful gait structure simplifies the search, but limits the motion family.",
         .889, 5.99, 11.55, .53, size=18, bold=True)
    text(second, "Morand (2026), sections 2.3–2.4, pp. 9–12. Examples: Iscen et al. (2018); Rudin et al. (2022).",
         .889, 5.60, 11.55, .26, size=12, color="5A626D")
    notes_b = (
        "This alternative follows the way Nathann organizes his literature review. The first decision is "
        "what the policy commands: torque, position targets or impedance parameters. The second is how "
        "much gait structure we provide: a reference motion, a parameterized generator, or a policy that "
        "learns the motion directly. The third is the curriculum: how commands, terrain or model variation "
        "become harder. DASH-01 uses periodic gait parameters plus rapid learned corrections, joint targets "
        "tracked by local PD control, and gated command and randomization curricula. These choices narrow "
        "the search and connect to the controller architecture shown next. The green column describes "
        "our current code; it is not a hardware capability claim. This is an adaptation of the thesis review, "
        "not a copy of unavailable presentation slides. Choose this slide OR the preceding literature overview."
    )
    note(second, 30, notes_b, [
        "Nathann Morand, DASH-01 master's thesis (2026), Chapter 2, sections 2.3–2.4, PDF pp. 9–12.",
        "Iscen et al., Policies Modulating Trajectory Generators, CoRL 2018. https://proceedings.mlr.press/v87/iscen18a.html",
        "Rudin et al., Learning to Walk in Minutes, CoRL 2021 / PMLR 2022. https://proceedings.mlr.press/v164/rudin22a.html",
        "Current implementation: RLframework/gait.py, plant.py, config.py and ppo.py. Design axes are not mutually exclusive.",
    ])
    return first, second, notes_a, notes_b


def main():
    export_figures()
    source_hash = digest(SOURCE)
    with ZipFile(SOURCE) as z:
        original = {name: z.read(name) for name in z.namelist()}
        infos = {entry.filename: entry for entry in z.infolist()}
    prs = Presentation(SOURCE)
    first, second, notes_a, notes_b = build_options(prs)
    temporary = BytesIO()
    prs.save(temporary)
    with ZipFile(temporary) as z:
        generated = {name: z.read(name) for name in z.namelist()}
    # Import only the new parts; retain existing media, master and slide bytes.
    additions = {name: data for name, data in generated.items() if name not in original}
    # python-pptx may reuse filenames of orphaned images still present in the
    # original ZIP. Give every imported figure a unique name before grafting.
    first_name = str(first.part.partname).lstrip("/")
    first_rels_name = posixpath.join(posixpath.dirname(first_name), "_rels", posixpath.basename(first_name) + ".rels")
    first_rels = ET.fromstring(generated[first_rels_name])
    image_relations = [r for r in first_rels if r.get("Type").endswith("/image")]
    assert len(image_relations) == len(PAPERS)
    for relation, paper in zip(image_relations, PAPERS, strict=True):
        generated_image = posixpath.normpath(posixpath.join(posixpath.dirname(first_name), relation.get("Target")))
        payload = generated[generated_image]
        assert hashlib.sha256(payload).hexdigest() == paper["image_sha256"]
        unique_name = f'ppt/media/sota_{paper["key"]}.png'
        assert unique_name not in original
        additions[unique_name] = payload
        if generated_image not in original:
            additions.pop(generated_image, None)
        relation.set("Target", "../media/" + posixpath.basename(unique_name))
    additions[first_rels_name] = xml_bytes(first_rels)
    replacements = {}
    original_prs = Presentation(SOURCE)
    for index, slide in enumerate(original_prs.slides, 1):
        if index < 4:
            continue
        name = str(slide.part.partname).lstrip("/")
        tree = ET.fromstring(original[name])
        changed = False
        for s in tree.findall("p:cSld/p:spTree/p:sp", NS):
            position = s.find("p:spPr/a:xfrm/a:off", NS)
            if position is None or int(position.get("y", "0")) < 6.4 * 914400:
                continue
            for t in s.findall(".//a:t", NS):
                if t.text == str(index):
                    t.text = str(index + 2)
                    changed = True
        if changed:
            replacements[name] = xml_bytes(tree)
    # State-of-the-art belongs inside Context, immediately after its divider.
    presentation = ET.fromstring(original["ppt/presentation.xml"])
    ids = presentation.find(f"{{{P}}}sldIdLst")
    rels = ET.fromstring(original["ppt/_rels/presentation.xml.rels"])
    largest_id = max(int(node.get("id")) for node in ids)
    largest_rid = max(int(node.get("Id")[3:]) for node in rels if node.get("Id").startswith("rId") and node.get("Id")[3:].isdigit())
    for offset, slide in enumerate((first, second), 1):
        rid = f"rId{largest_rid + offset}"
        ET.SubElement(rels, f"{{{REL}}}Relationship", Id=rid, Type=R + "/slide",
                      Target=str(slide.part.partname).removeprefix("/ppt/"))
        node = ET.Element(f"{{{P}}}sldId", id=str(largest_id + offset))
        node.set(f"{{{R}}}id", rid)
        ids.insert(2 + offset, node)
    replacements["ppt/presentation.xml"] = xml_bytes(presentation)
    replacements["ppt/_rels/presentation.xml.rels"] = xml_bytes(rels)
    content = ET.fromstring(original["[Content_Types].xml"])
    generated_content = ET.fromstring(generated["[Content_Types].xml"])
    existing = {(n.tag, n.get("PartName") or n.get("Extension")) for n in content}
    for n in generated_content:
        key = (n.tag, n.get("PartName") or n.get("Extension"))
        if key not in existing:
            content.append(deepcopy(n))
    replacements["[Content_Types].xml"] = xml_bytes(content)
    # Keep document-level slide count consistent; the existing thumbnail is only a cover.
    app = ET.fromstring(original["docProps/app.xml"])
    app.find("{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Slides").text = "23"
    replacements["docProps/app.xml"] = xml_bytes(app)
    with ZipFile(DEST, "w") as z:
        for name, data in original.items():
            z.writestr(infos[name], replacements.get(name, data))
        for name, data in additions.items():
            z.writestr(name, data)
    with ZipFile(DEST) as z:
        assert z.testzip() is None
        for name, data in original.items():
            if name not in replacements:
                assert z.read(name) == data
        for name in z.namelist():
            if name.endswith(".rels"):
                base_dir = posixpath.dirname(posixpath.dirname(name))
                for rel in ET.fromstring(z.read(name)):
                    if rel.get("TargetMode") == "External":
                        continue
                    target = rel.get("Target")
                    resolved = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(base_dir, target))
                    assert resolved in z.namelist(), (name, resolved)
        for name in original:
            if name.startswith("ppt/media/"):
                assert z.read(name) == original[name]
    output = Presentation(DEST)
    assert len(output.slides) == 23
    assert output.slides[3].shapes[0].text == "State of the art: learned locomotion"
    assert output.slides[4].shapes[0].text == "State of the art: controller design choices"
    assert output.slides[9].shapes[0].text == "Stopping trials: early failures"
    assert output.slides[10].shapes[0].text == "Walking policies: visual evolution"
    imported_pictures = [s for s in output.slides[3].shapes if s.shape_type == 13]
    assert len(imported_pictures) == len(PAPERS)
    for imported, paper in zip(imported_pictures, PAPERS, strict=True):
        assert hashlib.sha256(imported.image.blob).hexdigest() == paper["image_sha256"]
    # All pre-existing slide content matches after ignoring manual footer renumbering.
    for index, slide in enumerate(original_prs.slides, 1):
        new_slide = output.slides[index - 1 if index < 4 else index + 1]
        for old_shape, new_shape in zip(slide.shapes, new_slide.shapes, strict=True):
            old, new = deepcopy(old_shape._element), deepcopy(new_shape._element)
            if old_shape.has_text_frame and old_shape.text == str(index) and old_shape.top > 6.4 * 914400:
                for node in old.findall(".//a:t", NS) + new.findall(".//a:t", NS):
                    node.text = "page"
            assert ET.tostring(old) == ET.tostring(new), (index, old_shape.shape_id)
    assert digest(SOURCE) == source_hash
    receipt = dict(date="2026-10-28", source=SOURCE.name, source_sha256=source_hash,
                   output=DEST.name, output_sha256=digest(DEST),
                   alternatives={"4": "Selected research advances: Cassie / ANYmal / Digit",
                                 "5": "Adapted from Morand's thesis: motor interface / gait prior / curriculum"},
                   instruction="Choose one alternative; delete the other. Each is planned for about 30 seconds.",
                   source_thesis=str(THESIS), thesis_sha256=digest(THESIS), thesis_pages=[9, 10, 11, 12],
                   papers=PAPERS, added_parts=list(additions), modified_parts=list(replacements),
                   original_media_byte_identical=True, original_slide_content_preserved_except_page_numbers=True,
                   source_presentation_preserved=True, all_internal_relationships_resolve=True)
    (FOLDER / "state-of-art-update.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    references = "\n".join(f'- [{p["citation"]}: {p["title"]}]({p["url"]}). Image adapted from {p["figure"]}.' for p in PAPERS)
    (FOLDER / "state-of-art-notes.md").write_text(
        "# State-of-the-art alternatives\n\nTwo alternative slides are inserted after the Context divider. "
        "Keep either slide 4 or slide 5; they are **alternatives, not two required talks**. "
        "Each adds about 30 seconds. The latest stopping and walking videos are preserved "
        "and are now at slides 10 and 11 while both alternatives remain.\n\n"
        "## Slide 4 — Selected research advances\n\n" + notes_a + "\n\n" + references +
        "\n\n## Slide 5 — Nathann's literature review\n\n" + notes_b +
        "\n\nSource: Nathann Morand, *DASH-01: RL Framework to Develop a Bipedal Humanoid Robot "
        "Aimed at Breaking the Speed Record* (2026), sections 2.3–2.4, PDF pp. 9–12. "
        "The DASH-01 column is our interpretation of the current code. No slides from his unavailable "
        "presentation are reproduced.\n\nSupporting primary reference: "
        "[Iscen et al., PMTG, CoRL 2018](https://proceedings.mlr.press/v87/iscen18a.html). "
        "These are selected relevant papers, not an exhaustive 2026 survey or cross-platform benchmark. "
        "No speed record or hardware capability is claimed for DASH-01.\n",
        encoding="utf-8")
    print(json.dumps({"output": DEST.name, "slides": 23, "alternatives": [4, 5],
                      "unchanged_source_media": True, "unchanged_original_slide_content": True}, indent=2))


if __name__ == "__main__":
    main()
