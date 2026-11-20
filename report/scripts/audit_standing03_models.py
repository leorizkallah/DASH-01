"""Read-only CAD audit and provisional mass-transfer proposal; no plant or training edits."""

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/data/standing03"
SOURCES = {
    "v1": ROOT / "RLframework/model/dash01_free.xml",
    "v2": ROOT / "RLframework/model/dash01_cad_v2_homing.xml",
    "v2_raw": ROOT / "dash-01CAD-v2/EXPORT_assembly_closed/EXPORT_assembly-v34_closed.xml",
    "xshape_raw": ROOT / "xshape_closed/EXPORT_assembly-v8_closed.xml",
}
CHANGED = {
    "EXPORT_thight_left-v2_1": "EXPORT_thigh_left-v3_1",
    "EXPORT_thight_right-v4_1": "EXPORT_thigh_right-v3_1",
    "EXPORT_foot-v2_2": "EXPORT_foot-v4_1",
    "EXPORT_foot-v2_1": "EXPORT_foot-v4_2",
    "EXPORT_achilles_left-v5_1": "EXPORT_achilles_left-v7_1",
    "EXPORT_achilles_right-v3_1": "EXPORT_achilles_right-v4_1",
}


def tensor(v):
    xx, yy, zz, xy, xz, yz = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def packed(a):
    return [float(a[i, j]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    input_paths = list(SOURCES.values()) + [ROOT / "dash-01CAD-v2/properties/body_mapping.json",
                                           ROOT / "dash-01CAD-v2/properties/mass_properties.json"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    models = {}
    for label, path in SOURCES.items():
        root = ET.parse(path).getroot()
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        loops = {c.get("name"): float(np.linalg.norm(data.site(c.get("site1")).xpos -
                  data.site(c.get("site2")).xpos) * 1000) for c in root.findall("equality/connect")}
        models[label] = dict(source=path.relative_to(ROOT).as_posix(), mass_kg=float(model.body_mass.sum()),
                             nq=model.nq, nv=model.nv, actuators=model.nu, equalities=model.neq,
                             keyframes=[model.key(i).name for i in range(model.nkey)],
                             max_loop_gap_mm=max(loops.values()), loops_mm=loops)
    old = {b.get("name"): b for b in ET.parse(SOURCES["v2_raw"]).getroot().findall(".//body")}
    new = ET.parse(SOURCES["xshape_raw"]).getroot()
    mapping = json.loads(input_paths[-2].read_text())["bodies"]
    props = json.loads(input_paths[-1].read_text())["parts"]
    rows, estimates = [], {}
    for b in new.findall(".//body"):
        name = b.get("name")
        old_name = CHANGED.get(name, name)
        ni, oi = b.find("inertial"), old[old_name].find("inertial")
        part = props[mapping[old_name]["part"]]
        if name not in CHANGED:
            assert np.isclose(float(ni.get("mass")), float(oi.get("mass")), rtol=1e-7)
            assert np.allclose(np.fromstring(ni.get("pos"), sep=" "), np.fromstring(oi.get("pos"), sep=" "), rtol=1e-7, atol=1e-8)
            assert np.allclose(np.fromstring(ni.get("fullinertia"), sep=" "), np.fromstring(oi.get("fullinertia"), sep=" "), rtol=1e-7, atol=1e-11)
            mass = part["mass_g"] * .001
            com = np.array(part["com_mm"]) * .001
            inertia = tensor([part["inertia_com_g_mm2"][k] for k in ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz")]) * 1e-9
            method = "v2 PDF transfer; raw local inertials match; frame/geometry check still required"
        else:
            factor = (part["mass_g"] * .001) / float(oi.get("mass"))
            mass = float(ni.get("mass")) * factor
            com = np.fromstring(ni.get("pos"), sep=" ")
            inertia = tensor(np.fromstring(ni.get("fullinertia"), sep=" ")) * factor
            method = "provisional v2 density-correction factor on new CAD mass/tensor; retain new COM"
        if name == "EXPORT_body-v4_1":
            inertia *= (mass + 1.0) / mass
            mass += 1.0
        eig = np.linalg.eigvalsh(inertia)
        assert eig.min() > 0 and eig[0] + eig[1] >= eig[2] - 1e-12
        estimates[name] = dict(mass_kg=mass, com_m=com.tolist(), fullinertia_kg_m2=packed(inertia), method=method)
        rows.append(dict(new_body=name, v2_body=old_name, v2_part=mapping[old_name]["part"],
                         raw_mass_kg=float(ni.get("mass")), proposed_mass_kg=mass, method=method))
    total = sum(r["proposed_mass_kg"] for r in rows)
    proposal = dict(status="DRAFT: no corrected XML written and no training authorized by this audit",
                    total_proposed_kg=total, extra_torso_mass_kg=1.0, motors_and_battery_added_again=False,
                    unchanged_raw_local_inertials=9, changed_components=6, bodies=estimates,
                    limitations=["V2 PDF frames were confirmed for v2, not automatically for X-shape",
                                 "Changed-component COM and inertia are CAD/density estimates, not new measurements",
                                 "Body stiffness, joint play and printed-part strength are not validated by this audit"])
    (OUT / "mass-transfer-draft.json").write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    with (OUT / "mass-transfer-draft.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = dict(date="2026-11-09", mujoco_version=mujoco.__version__, models=models,
                 proposed_xshape_mass_kg=total, source_sha256=hashes,
                 simulation_steps_run=0, training_jobs_submitted=0)
    (OUT / "model-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for rel, digest in hashes.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
    print(json.dumps(dict(models=models, proposed_xshape_mass_kg=total, changed_parts=6,
                          simulation_steps_run=0, source_files_preserved=True), indent=2))


if __name__ == "__main__":
    main()
