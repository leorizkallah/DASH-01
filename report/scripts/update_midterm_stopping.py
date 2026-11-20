"""Replace reordered slide 9 with STOP clips, then swap display slides 8/9.

Patch the Open XML package directly so unrelated user edits remain byte-identical.
The source presentation is preserved; output is a separate presentation.
"""

from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import posixpath
from zipfile import ZipFile

import imageio.v2 as imageio
from lxml import etree as ET
from PIL import Image
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "report/presentations/midterm-2026-10-29"
SOURCE = FOLDER / "DASH01_controller_midterm_reordered.pptx"
DEST = FOLDER / "DASH01_controller_midterm_reordered_stopping.pptx"
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def xml_bytes(element):
    return ET.tostring(element, xml_declaration=True, encoding="UTF-8", standalone=True)


def shape(slide, shape_id):
    return next(s for s in slide.find("p:cSld/p:spTree", NS)
                if s.xpath(f'.//p:cNvPr[@id="{shape_id}"]', namespaces=NS))


def text(slide, shape_id, value):
    nodes = shape(slide, shape_id).findall(".//a:t", NS)
    assert nodes
    nodes[0].text = value
    for node in nodes[1:]:
        node.text = ""


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with ZipFile(SOURCE) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
        info = {entry.filename: entry for entry in archive.infolist()}
    slide8 = ET.fromstring(original["ppt/slides/slide8.xml"])
    assert "Walking policies" in "".join(slide8.itertext())
    assert "Speed tracking" in original["ppt/slides/slide9.xml"].decode()
    stopping = deepcopy(slide8)
    text(stopping, 3, "Stopping trials: early failures")
    text(stopping, 4, "8")
    text(stopping, 5, "Reducing the speed command was not enough to obtain a stable five-second stand.")
    text(stopping, 6, "Abrupt STOP")
    text(stopping, 7, "STOP-01B final policy | 1.00 m/s | 10.1 s clip")
    text(stopping, 9, "Smooth braking reference")
    text(stopping, 10, "Same policy + STOP-02 ramp | 1.00 m/s | 7.8 s clip")
    text(stopping, 12, "Previous robot model | Simulation | Playback: left 0.5x, right 1x | Click to play")
    # 3:2 videos are fitted without stretching into the two existing columns.
    for shape_id, center_x in ((8, 3.696), (11, 9.628)):
        movie = shape(stopping, shape_id)
        transform = movie.find("p:spPr/a:xfrm", NS)
        transform.find("a:off", NS).set("x", str(round((center_x - 2.25) * 914400)))
        transform.find("a:off", NS).set("y", str(round(2.68 * 914400)))
        transform.find("a:ext", NS).set("cx", str(round(4.5 * 914400)))
        transform.find("a:ext", NS).set("cy", str(round(3.0 * 914400)))
    caption = shape(stopping, 12).find("p:spPr/a:xfrm/a:off", NS)
    caption.set("y", str(round(5.87 * 914400)))
    rels = ET.fromstring(original["ppt/slides/_rels/slide8.xml.rels"])
    replacements = {"ppt/slides/slide9.xml": xml_bytes(stopping)}
    additions, movies = {}, []
    for filename, media_ids, poster_id, shape_id, frame_index in (
        ("stop01_abrupt_failure.mp4", ("rId1", "rId2"), "rId7", 8, 186),
        ("stop02_ramp_failure.mp4", ("rId3", "rId4"), "rId8", 11, 96),
    ):
        video = ROOT / "report/videos" / filename
        reader = imageio.get_reader(str(video))
        try:
            meta = reader.get_meta_data()
            frame = reader.get_data(frame_index)
        finally:
            reader.close()
        assert meta["codec"] == "h264" and meta["size"] == (960, 640)
        poster = BytesIO()
        Image.fromarray(frame).save(poster, format="JPEG", quality=94)
        video_part = f"ppt/media/{filename}"
        poster_part = f"ppt/media/{Path(filename).stem}_poster.jpg"
        assert video_part not in original and poster_part not in original
        additions[video_part] = video.read_bytes()
        additions[poster_part] = poster.getvalue()
        for relation in rels:
            if relation.get("Id") in media_ids:
                relation.set("Target", "../media/" + filename)
            elif relation.get("Id") == poster_id:
                relation.set("Target", "../media/" + Path(poster_part).name)
            elif relation.get("Type").endswith("/notesSlide"):
                relation.set("Target", "../notesSlides/notesSlide9.xml")
        shape(stopping, shape_id).find("p:nvPicPr/p:cNvPr", NS).set("name", filename)
        movies.append(dict(source=video.relative_to(ROOT).as_posix(), embedded_part=video_part,
                           sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                           duration_s=meta["duration"], fps=meta["fps"],
                           recorded_playback=0.5 if shape_id == 8 else 1.0))
    replacements["ppt/slides/slide9.xml"] = xml_bytes(stopping)
    replacements["ppt/slides/_rels/slide9.xml.rels"] = xml_bytes(rels)
    text(slide8, 4, "9")
    replacements["ppt/slides/slide8.xml"] = xml_bytes(slide8)
    presentation = ET.fromstring(original["ppt/presentation.xml"])
    ids = presentation.find("p:sldIdLst", NS)
    previous8 = ids[7]
    ids.remove(previous8)
    ids.insert(8, previous8)
    replacements["ppt/presentation.xml"] = xml_bytes(presentation)
    notes = ET.fromstring(original["ppt/notesSlides/notesSlide9.xml"])
    body = next(s for s in notes.findall(".//p:sp", NS)
                if s.xpath('.//p:ph[@type="body"]', namespaces=NS)).find("p:txBody", NS)
    for paragraph in list(body.findall("a:p", NS)):
        body.remove(paragraph)
    script = (
        "Suggested time: 40 seconds.\n\n"
        "These failed stopping trials used the previous robot model. Both begin with a requested walking "
        "speed of one metre per second. On the left, the STOP-01B final policy receives an abrupt STOP. "
        "On the right, the same policy uses the STOP-02 smooth speed-reference adapter; this is a "
        "command-only diagnostic, not the trained STOP-02 standing policy. Neither clip demonstrates "
        "a successful five-second quiet stand. Slowing the command alone did not provide the missing "
        "standing and transition skills. The left recording is half speed; the right is normal speed. "
        "These are illustrative single trials, not a controlled estimate of success rates.\n\n"
        "Evidence: report/assets.md; report/videos/stop01_abrupt_failure.mp4; "
        "report/videos/stop02_ramp_failure.mp4; docs/stop02-experiment-report.md."
    )
    for line in script.splitlines():
        paragraph = ET.SubElement(body, f'{{{NS["a"]}}}p')
        run = ET.SubElement(paragraph, f'{{{NS["a"]}}}r')
        ET.SubElement(run, f'{{{NS["a"]}}}t').text = line
    replacements["ppt/notesSlides/notesSlide9.xml"] = xml_bytes(notes)
    with ZipFile(DEST, "w") as archive:
        for name, data in original.items():
            archive.writestr(info[name], replacements.get(name, data))
        for name, data in additions.items():
            archive.writestr(name, data)
    with ZipFile(DEST) as archive:
        assert archive.testzip() is None
        for name, data in original.items():
            if name not in replacements:
                assert archive.read(name) == data, name
        for movie in movies:
            assert hashlib.sha256(archive.read(movie["embedded_part"])).hexdigest() == movie["sha256"]
        # Check internal relationship destinations throughout the delivered package.
        for name in archive.namelist():
            if not name.endswith(".rels"):
                continue
            base = posixpath.dirname(posixpath.dirname(name))
            for relationship in ET.fromstring(archive.read(name)):
                if relationship.get("TargetMode") == "External":
                    continue
                target = relationship.get("Target")
                part = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(base, target))
                assert part in archive.namelist(), (name, part)
    prs = Presentation(DEST)
    assert len(prs.slides) == 21
    assert prs.slides[7].shapes[0].text == "Stopping trials: early failures"
    assert prs.slides[8].shapes[0].text == "Walking policies: visual evolution"
    assert len([s for s in prs.slides[7].shapes if s.shape_type == 16]) == 2
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    receipt = dict(date="2026-10-28", source=SOURCE.name, source_sha256=source_hash,
                   output=DEST.name, output_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
                   new_slide8="Failed stopping trials with two embedded videos",
                   new_slide9="Previous slide 8 walking comparison, unchanged except page number",
                   changed_package_parts=list(replacements), movies=movies,
                   unchanged_original_parts=len(original) - len(replacements),
                   all_internal_relationships_resolve=True, source_preserved=True)
    (FOLDER / "stopping-slide-update.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (FOLDER / "stopping-slide-notes.md").write_text(
        "# Reordered presentation: changed slides\n\n"
        "## Slide 8 — Stopping trials: early failures\n\n" + script +
        "\n\n## Slide 9 — Walking policies: visual evolution\n\n"
        "Previously slide 8; the two existing walking videos, text and embedded speaker notes are preserved.\n"
        "The former speed-tracking slide has been replaced. Other slides and their notes are unchanged.\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
