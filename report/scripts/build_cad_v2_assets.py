"""Build mass comparison and CAD inspection image from the recorded v2 audit."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dash_report_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="also render static CAD pose (requires OpenGL)")
    args = parser.parse_args()
    audit = json.loads((REPO / "dash-01CAD-v2/generated/audit.json").read_text())
    rows = audit["bodies"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": [1, 1.5]})
    for ax, selected, scale, unit in (
        (axes[0], [r for r in rows if r["corrected_mass_kg"] > 1], 1, "kg"),
        (axes[1], [r for r in rows if r["corrected_mass_kg"] <= 1], 1000, "g"),
    ):
        y = np.arange(len(selected))
        ax.barh(y-.18, [r["raw_mass_kg"]*scale for r in selected], .34, color="#9baab5", label="Raw export")
        ax.barh(y+.18, [r["corrected_mass_kg"]*scale for r in selected], .34, color="#17818a", label="PDF + torso payload")
        labels = [r["pdf_part"].replace("_", " ") if r["pdf_part"] != "Foot" else "Foot " + ("R" if r["body"].endswith("_2") else "L") for r in selected]
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"Mass ({unit})")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("DASH-01 v2: mass correction", fontsize=18)
    fig.text(.5, .02, f"Raw export: {audit['raw_mass_kg']:.3f} kg | PDF: {audit['pdf_mass_kg']:.3f} kg | Added torso payload: {audit['added_body_mass_kg']:.3f} kg | Final: {audit['corrected_mass_kg']:.3f} kg", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .05, 1, .94])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(REPORT / f"figures/cad_v2_mass_audit.{ext}", dpi=180)
    plt.close(fig)
    with (REPORT / "tables/cad_v2_mass_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["body", "pdf_part", "raw_mass_kg", "corrected_mass_kg"])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    if args.render:
        import mujoco
        from PIL import Image
        model = mujoco.MjModel.from_xml_path(str(REPO / "dash-01CAD-v2/generated/dash01_v2_corrected.xml"))
        model.vis.global_.offwidth = 1000
        model.vis.global_.offheight = 800
        model.vis.headlight.ambient[:] = [.55, .55, .55]
        model.vis.headlight.diffuse[:] = [.8, .8, .8]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, -.38]
        camera.distance = 1.65
        camera.azimuth = 130
        camera.elevation = -12
        with mujoco.Renderer(model, height=800, width=1000) as renderer:
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(REPORT / "figures/cad_v2_inspection.png")
    print("Updated v2 mass chart (PNG/SVG/PDF), CSV" + (" and static CAD image." if args.render else "."))


if __name__ == "__main__":
    main()
