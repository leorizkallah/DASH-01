"""Correct and audit the v2 CAD inertials; does not build a training-ready plant.

Run from the repository root: python RLframework/model/prepare_cad_v2.py
The raw CAD stays untouched. --report validates without writing outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from controller.identification.apply_identified import apply_body_inertials

CAD = REPO / "dash-01CAD-v2"
SOURCE = CAD / "EXPORT_assembly_closed/EXPORT_assembly-v34_closed.xml"
PROPERTIES = CAD / "properties/mass_properties.json"
MAPPING = CAD / "properties/body_mapping.json"
OUTPUT = CAD / "generated/dash01_v2_corrected.xml"
ORDER = ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor(values):
    xx, yy, zz, xy, xz, yz = (values[k] for k in ORDER)
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)


def tensor_dict(matrix):
    return dict(zip((k.lower() for k in ORDER),
                    (float(matrix[i, j]) for i, j in
                     ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))))


def rotation_x(degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def validate_inertia(mass, com, inertia):
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Mass must be finite and positive")
    if not np.all(np.isfinite(com)) or not np.all(np.isfinite(inertia)):
        raise ValueError("Non-finite COM/inertia")
    if not np.allclose(inertia, inertia.T, atol=1e-14, rtol=0):
        raise ValueError("Inertia tensor is not symmetric")
    principal = np.linalg.eigvalsh(inertia)
    if principal[0] <= 0 or principal[0] + principal[1] < principal[2] - 1e-12:
        raise ValueError(f"Nonphysical principal inertias: {principal}")
    return principal


def convert(part, frame, extra_kg=0.0):
    """g -> kg; mm -> m; g mm^2 -> kg m^2 (factor 1e-9).

    PDF tensors are already at COM, with signed off-diagonal matrix entries.
    An origin translation changes COM coordinates, not the COM inertia tensor.
    """
    if not np.isfinite(extra_kg) or extra_kg < 0:
        raise ValueError("Extra mass must be finite and nonnegative")
    mass = part["mass_g"] * 1e-3
    com = np.asarray(part["com_mm"], dtype=float) * 1e-3
    inertia = tensor(part["inertia_com_g_mm2"]) * 1e-9
    validate_inertia(mass, com, inertia)
    origin = tensor(part["inertia_origin_g_mm2"]) * 1e-9
    expected = inertia + mass * (com @ com * np.eye(3) - np.outer(com, com))
    residual = float(np.max(np.abs(origin - expected)) / np.max(np.abs(origin)))
    # The source rounds some entries to four significant digits and COM to .001 mm.
    if residual > 0.001:
        raise ValueError(f"PDF COM/origin tensors disagree (relative error {residual:g})")
    rotation = rotation_x(frame["rotation_x_deg"])
    com = rotation @ com + np.asarray(frame["translation_mm"], dtype=float) * 1e-3
    inertia = rotation @ inertia @ rotation.T
    inertia *= (mass + extra_kg) / mass  # explicitly assumed same mass distribution
    mass += extra_kg
    principal = validate_inertia(mass, com, inertia)
    return {"mass": mass, "com": com.tolist(), "inertia": tensor_dict(inertia)}, {
        "principal_inertia_kg_m2": principal.tolist(),
        "pdf_parallel_axis_relative_error": residual,
    }


def mesh_bounds(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    vertices = np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError(f"Non-finite mesh vertices: {path}")
    return vertices.min(axis=0).astype(float), vertices.max(axis=0).astype(float)


def loop_gaps(model, data, root):
    return {c.get("name"): float(np.linalg.norm(
        data.site(c.get("site1")).xpos - data.site(c.get("site2")).xpos) * 1000)
        for c in root.findall("equality/connect")}


def build(extra_body_kg=None):
    source = json.loads(PROPERTIES.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    if source["units"] != {"mass": "g", "com": "mm", "inertia": "g mm^2"}:
        raise ValueError("Unexpected source units")
    pdf = PROPERTIES.parent / source["source"]
    if digest(pdf) != source["sha256"]:
        raise ValueError("PDF changed; re-extract and review the properties first")
    root = ET.parse(SOURCE).getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}
    if len(bodies) != len(list(root.iter("body"))) or set(bodies) != set(mapping["bodies"]):
        raise ValueError("CAD body list changed; review the explicit PDF mapping")
    if len(root.findall("equality/connect")) != 4:
        raise ValueError("Expected all four mechanical loops")
    extra = mapping["extra_body_mass"]
    if extra["distribution"] != "same_as_torso" or extra["body"] not in bodies:
        raise ValueError("Unsupported extra-mass configuration")
    added_kg = extra["kg"] if extra_body_kg is None else extra_body_kg
    params, rows = {}, []
    for name, frame in mapping["bodies"].items():
        part = source["parts"][frame["part"]]
        corrected, checks = convert(part, frame, added_kg if name == extra["body"] else 0)
        params[name] = corrected
        old = bodies[name].find("inertial")
        mesh = SOURCE.parent / f"meshes/{name}.stl"
        lower, upper = mesh_bounds(mesh)
        com_mm = np.asarray(corrected["com"]) * 1000
        if np.any(com_mm < lower - 1) or np.any(com_mm > upper + 1):
            raise ValueError(f"Corrected COM is outside mesh bounding box: {name}")
        rows.append(dict(body=name, pdf_part=frame["part"], raw_mass_kg=float(old.get("mass")),
                         corrected_mass_kg=corrected["mass"], com_m=corrected["com"],
                         frame=frame, mesh_bbox_mm=(upper-lower).tolist(),
                         pdf_bbox_mm=part["bbox_mm"], **checks))
    apply_body_inertials(root, params, strict=True)
    compiler = root.find("compiler")
    compiler.set("inertiafromgeom", "false")
    compiler.set("balanceinertia", "false")
    compiler.set("meshdir", str(SOURCE.parent.resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gaps = loop_gaps(model, data, root)
    if max(gaps.values()) > 0.01:
        raise ValueError(f"Loop mismatch exceeds 0.01 mm: {gaps}")
    expected_mass = sum(v["mass"] for v in params.values())
    if not np.isclose(model.body_mass.sum(), expected_mass, atol=1e-9, rtol=0):
        raise ValueError("MuJoCo compiled mass differs from requested mass")
    # Verify the compiler preserved the full tensors, not only the total mass.
    for name, bp in params.items():
        b = model.body(name)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, b.iquat)
        rotation = rotation.reshape(3, 3)
        actual = rotation @ np.diag(b.inertia) @ rotation.T
        desired = tensor({k: bp["inertia"][k.lower()] for k in ORDER})
        # MuJoCo's eigensolver has finite precision, especially off diagonal.
        if np.max(np.abs(actual-desired)) > 1e-6 * np.max(np.abs(desired)) + 1e-12:
            raise ValueError(f"Compiled inertia mismatch: {name}")
        if not np.allclose(b.ipos, bp["com"], atol=1e-11, rtol=0):
            raise ValueError(f"Compiled COM mismatch: {name}")
    # Numerical check only: fixed-base zero-gravity test avoids claiming balance.
    model.opt.gravity[:] = 0
    for _ in range(100):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
        raise ValueError("MuJoCo smoke test produced a nonfinite state or warning")
    final_gaps = loop_gaps(model, data, root)
    if max(final_gaps.values()) > 0.01:
        raise ValueError("Loops drifted during zero-gravity smoke test")
    compiler.set("meshdir", "../EXPORT_assembly_closed")
    root.insert(0, ET.Comment(" Generated by prepare_cad_v2.py. PDF properties use export frames as confirmed by user; see audit.json. CAD inspection model, not a training plant. "))
    ET.indent(root, space="  ")
    xml = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
    inputs = [SOURCE, PROPERTIES, MAPPING, pdf, Path(__file__),
              REPO / "controller/identification/apply_identified.py",
              SOURCE.parent / "acdc4robot-export-report.json"]
    inputs += sorted((SOURCE.parent / "meshes").glob("*.stl"))
    audit = dict(schema_version=1, training_ready=False, mujoco_version=mujoco.__version__,
                 raw_mass_kg=sum(r["raw_mass_kg"] for r in rows),
                 pdf_mass_kg=expected_mass-added_kg, added_body_mass_kg=added_kg,
                 corrected_mass_kg=expected_mass, frame_confirmation=mapping["frame_confirmation"],
                 assembly_contents=mapping["assembly_contents"], extra_mass_assumption=extra["note"],
                 bodies=rows, initial_loop_gap_mm=gaps, smoke_test_final_loop_gap_mm=final_gaps,
                 smoke_test=dict(duration_s=float(data.time), gravity=[0,0,0], fixed_base=True,
                                 warnings=0, max_abs_qvel=float(np.max(np.abs(data.qvel)))),
                 model_dimensions=dict(nbody=model.nbody, nq=model.nq, nv=model.nv, nu=model.nu, neq=model.neq),
                 findings=[
                     "PDF and CAD component revision labels differ; user confirmed PDF properties use export frames.",
                     "Right cam COM/tensor resemble a 5 deg rotation of the mirrored left cam; right tibia mesh origins differ. No inferred transforms applied, per user confirmation.",
                     "PDF thigh width is 264.9 mm, exported mesh width is 61.5 mm; inspect source assembly contents.",
                     "Right thigh limits are not the sign-reversed left limits (about 4.53 deg offset); preserved for review.",
                     "PDF tibia masses differ by 3.757 g; preserved, not forced symmetric.",
                     "PDF supplies one Foot; applied to both identical exported foot meshes.",
                     "No extra motor bodies added: user confirmed PDF assembly masses include motors and battery.",
                     "Export is fixed-base with all collisions disabled, no floor, no actuators and no hardware joint-zero mapping.",
                     "Achilles links are rigid; no spring compliance is specified. Do not invent a stiffness.",
                     "Old flat-foot builder/policies are incompatible with the new articulated feet and four-loop topology.",
                 ], inputs={p.relative_to(REPO).as_posix(): digest(p) for p in inputs},
                 generated_xml_sha256=hashlib.sha256(xml.encode("utf-8")).hexdigest())
    return xml, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="validate without writing")
    parser.add_argument("--extra-body-kg", type=float, default=None, help="override configured torso payload (default 1 kg)")
    args = parser.parse_args(argv)
    xml, audit = build(args.extra_body_kg)
    if not args.report:
        OUTPUT.parent.mkdir(exist_ok=True)
        OUTPUT.write_bytes(xml.encode("utf-8"))
        (OUTPUT.parent / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(f"Wrote {OUTPUT.relative_to(REPO)} and audit.json")
    print(f"Mass: CAD {audit['raw_mass_kg']:.6f} -> PDF {audit['pdf_mass_kg']:.6f} + payload {audit['added_body_mass_kg']:.3f} = {audit['corrected_mass_kg']:.6f} kg")
    print(f"15 inertials checked; four loop gaps <= {max(audit['initial_loop_gap_mm'].values()):.6f} mm; numerical smoke test passed.")
    print("PDF frames and motor/battery inclusion confirmed by user. CAD inspection model only; see audit findings before training.")


if __name__ == "__main__":
    main()
