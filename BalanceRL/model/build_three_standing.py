"""Build isolated standing plants: one continuous V1 sole, two V2/X-shape pads."""
from pathlib import Path
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RLframework/model"))
import make_plant as legacy
import make_cad_v2_plant as v2

OUT = Path(__file__).parent
AUDIT = ROOT / "report/data/standing03"


def absolute_assets(root, source):
    compiler = root.find("compiler")
    meshdir = source.parent / compiler.attrib.pop("meshdir", "")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((meshdir / mesh.get("file")).resolve()))


def sole_patches(path, down_local):
    """Reuse the reviewed v2 surface detector with an explicit mesh input."""
    return v2.sole_patches(path=path, down_local=down_local)


def xshape():
    source = ROOT / "xshape_closed/EXPORT_assembly-v8_closed.xml"
    original_model = mujoco.MjModel.from_xml_path(str(source))
    original_data = mujoco.MjData(original_model)
    mujoco.mj_forward(original_model, original_data)
    root = ET.parse(source).getroot()
    absolute_assets(root, source)
    masses = json.loads((AUDIT / "mass-transfer-draft.json").read_text())["bodies"]
    rename = {"EXPORT_body-v4_1": "torso", "EXPORT_foot-v2_1": "foot_R", "EXPORT_foot-v2_2": "foot_L"}
    for raw, stem in (("hip", "hip"), ("cam", "cam"), ("pushrod", "rod"),
                      ("thight", "thigh"), ("tibia", "leg"), ("achilles", "achilles")):
        for body in root.findall(".//body"):
            name = body.get("name")
            if name.startswith("EXPORT_" + raw + "_"):
                rename[name] = stem + ("_L" if "_left-" in name else "_R")
    for body in root.findall(".//body"):
        raw = body.get("name")
        item = masses[raw]
        inertial = body.find("inertial")
        inertial.set("mass", str(item["mass_kg"]))
        inertial.set("pos", v2.fs(item["com_m"]))
        inertial.set("fullinertia", v2.fs(item["fullinertia_kg_m2"]))
        name = rename[raw]
        body.set("name", name)
        joint = body.find("joint")
        if joint is not None:
            stem, side = name.rsplit("_", 1)
            family = v2.JOINT[stem]
            joint.set("name", family + "_" + side)
            dyn = legacy.JOINT_DYN.get(family, dict(damping=.2, armature=.001))
            joint.set("damping", str(dyn["damping"]))
            joint.set("armature", str(dyn["armature"]))
            if family == "cam":
                joint.set("limited", "true")
                joint.set("range", "-0.6 0.6")
    for side in "LR":
        names = {f"loop_knee_{side}_link": f"rod_tip_{side}",
                 f"loop_knee_{side}_target": f"leg_anchor_{side}",
                 f"loop_achilles_{side}_link": f"achilles_tip_{side}",
                 f"loop_achilles_{side}_target": f"foot_anchor_{side}"}
        for site in root.iter("site"):
            if site.get("name") in names:
                site.set("name", names[site.get("name")])
        for eq in root.findall("equality/connect"):
            for attr in ("site1", "site2"):
                if eq.get(attr) in names:
                    eq.set(attr, names[eq.get(attr)])
            eq.set("solref", "0.002 1")
            eq.set("solimp", "0.999 0.9999 0.0001")
    root.remove(root.find("option"))
    legacy._scaffold(root)
    legacy._add_actuators_and_sensors(root)
    root.find("worldbody/geom[@name='floor']").set("conaffinity", "2")
    collision = root.find("default/default/default/geom")
    collision.set("contype", "2")
    collision.set("conaffinity", "1")
    for side in "LR":
        foot = root.find(f".//body[@name='foot_{side}']")
        meshname = foot.find("geom").get("mesh")
        meshpath = Path(root.find(f"asset/mesh[@name='{meshname}']").get("file"))
        raw_foot = "EXPORT_foot-v2_2" if side == "L" else "EXPORT_foot-v2_1"
        rotation = original_data.xmat[original_model.body(raw_foot).id].reshape(3, 3)
        patches = sole_patches(meshpath, rotation.T @ np.array([0., 0., -1.]))
        for tag, patch in zip(("foot", "heel"), patches):
            ET.SubElement(foot, "geom", name=f"{tag}_{side}_col", **{
                "class": "collision", "type": "box", "size": v2.fs(patch["size"]),
                "pos": v2.fs(patch["pos"]), "quat": v2.fs(patch["quat"])})
    return root


def build(name):
    if name == "xshape":
        root = xshape()
        model = v2.compile_root(root)
        qpos, _ = v2.solve_pose(model, root)
    else:
        source = ROOT / ("RLframework/model/dash01_free.xml" if name == "v1" else "RLframework/model/dash01_cad_v2_homing.xml")
        root = ET.parse(source).getroot()
        absolute_assets(root, source)
        model = v2.compile_root(root)
        qpos = model.key_qpos[0].copy()
        root.remove(root.find("keyframe"))
        if name == "v1":
            for side in "LR":
                toe = root.find(f".//geom[@name='foot_{side}_col']")
                heel = root.find(f".//geom[@name='heel_{side}_col']")
                quat = np.fromstring(toe.get("quat", "1 0 0 0"), sep=" ")
                assert np.allclose(quat, np.fromstring(heel.get("quat", "1 0 0 0"), sep=" "))
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, quat)
                rotation = rotation.reshape(3, 3)
                a, b = (np.fromstring(g.get("pos"), sep=" ") for g in (toe, heel))
                sa, sb = (np.fromstring(g.get("size"), sep=" ") for g in (toe, heel))
                assert np.allclose(sa, sb)
                offset = np.abs(rotation.T @ (a-b)) / 2
                # Exact union of contiguous equal boxes, not a new point or a wider foot.
                assert np.isclose(np.max(offset / sa), 1, atol=1e-4)
                toe.set("pos", v2.fs((a+b)/2))
                toe.set("size", v2.fs(sa+offset))
                parent = next(b for b in root.iter("body") if heel in list(b))
                parent.remove(heel)
    root.set("model", "standing_compare_" + name)
    model = v2.compile_root(root)
    # Recompute a genuinely contact-loaded, free-base reference after contact edits.
    seed = None
    numeric = root.find("custom/numeric[@name='nominal_cmd']")
    if numeric is not None:
        act_qadr = [model.joint(n).qposadr[0] for n in legacy.ACTUATED]
        seed = np.array(legacy.STANCE_KP) * (np.fromstring(numeric.get("data"), sep=" ") - qpos[act_qadr])
    qpos, command, tau, equilibrium = v2.solve_loaded_stance(model, qpos, initial_torque=seed)
    qadr = [model.joint(n).qposadr[0] for n in legacy.ACTUATED]
    custom = root.find("custom")
    for tag, value in (("nominal_ctrl", qpos[qadr]), ("nominal_cmd", command)):
        node = custom.find(f"numeric[@name='{tag}']")
        if node is None:
            node = ET.SubElement(custom, "numeric", name=tag)
        node.set("data", v2.fs(value))
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="stand", qpos=v2.fs(qpos), ctrl="0 0 0 0 0 0")
    # Portable paths inside the experiment snapshot.
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", os.path.relpath(mesh.get("file"), OUT).replace("\\", "/"))
    ET.indent(root, space="  ")
    path = OUT / f"stand_{name}.xml"
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = tau
    mujoco.mj_forward(model, data)
    pads = [[model.geom(f"foot_{s}_col").id] + ([] if name == "v1" else [model.geom(f"heel_{s}_col").id]) for s in "LR"]
    perfoot = []
    for ids in pads:
        force, count = 0., 0
        for k in range(data.ncon):
            c = data.contact[k]
            if model.geom("floor").id in c.geom and any(g in c.geom for g in ids):
                f = np.empty(6)
                mujoco.mj_contactForce(model, data, k, f)
                force += max(float(f[0]), 0)
                count += 1
        perfoot.append(dict(regions=len(ids), native_contact_points=count, normal_force_n=force))
    assert all(p["normal_force_n"] > 2 for p in perfoot)
    assert np.isclose(sum(p["normal_force_n"] for p in perfoot), model.body_mass.sum()*9.81, rtol=.01)
    audit = dict(model=name, mass_kg=float(model.body_mass.sum()), foot_contacts=perfoot,
                 equilibrium=equilibrium, holding_torque_nm=tau.tolist(),
                 xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 contact_definition="one continuous sole" if name == "v1" else "two separated mesh-derived pads",
                 all_base_dofs_free=True, mass_estimates_provisional=(name == "xshape"))
    return audit


if __name__ == "__main__":
    AUDIT.mkdir(parents=True, exist_ok=True)
    audits = []
    for name in (sys.argv[1:] or ["v1", "v2", "xshape"]):
        audit = build(name)
        audits.append(audit)
        (AUDIT / f"built-{name}.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit), flush=True)
