"""Build the articulated-foot v2 simulation plant from reviewed CAD inertials.

Coordinates remain CAD-relative: this model is for simulation, not deployment.
Motors unchanged and Achilles rigid: user confirmation, 2026-10-14.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

try:
    from . import make_plant as legacy
    from . import prepare_cad_v2 as cad
except ImportError:
    import make_plant as legacy
    import prepare_cad_v2 as cad

HERE = Path(__file__).resolve().parent
OUT = HERE / "dash01_cad_v2_free.xml"
ACT = legacy.ACTUATED
KP = np.array(legacy.STANCE_KP)
KD = np.array(legacy.STANCE_KD)
BODY = {"EXPORT_body-v4_1": "torso"}
for part, stem, lv, rv in (("hip", "hip", 2, 3), ("thigh", "thigh", 3, 3),
                          ("cam", "cam", 3, 3), ("pushrod", "rod", 3, 2),
                          ("tibia", "leg", 2, 2), ("achilles", "achilles", 7, 4)):
    for side, word, version in (("L", "left", lv), ("R", "right", rv)):
        BODY[f"EXPORT_{part}_{word}-v{version}_1"] = f"{stem}_{side}"
BODY.update({"EXPORT_foot-v4_1": "foot_L", "EXPORT_foot-v4_2": "foot_R"})
JOINT = {"hip": "hip_roll", "thigh": "thigh", "cam": "cam", "rod": "rod",
         "leg": "knee", "foot": "ankle", "achilles": "achilles"}


def fs(values):
    return " ".join(f"{v:.12g}" for v in values)


def sole_patches():
    """Find the two coplanar rectangular sole faces in the identical foot meshes.

    Collision boxes extend 2 mm inward from the actual sole, adding no pad below
    CAD geometry and no contact across the gap between heel and toe.
    """
    path = cad.SOURCE.parent / "meshes/EXPORT_foot-v4_1.stl"
    raw = path.read_bytes()
    triangles = np.frombuffer(raw, offset=84, dtype=np.dtype([
        ("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"].astype(float) * .001
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normals = cross / np.maximum(2 * area[:, None], 1e-20)
    groups = {}
    for i in np.flatnonzero(normals[:, 2] < -.8):
        key = tuple(np.round(np.r_[normals[i], normals[i] @ triangles[i, 0]], 5))
        groups.setdefault(key, []).append(i)
    indices = max(groups.values(), key=lambda ix: area[ix].sum())
    n = np.average(normals[indices], axis=0, weights=area[indices])
    n /= np.linalg.norm(n)
    w = -n
    u = np.array([1., 0, 0]) - w * w[0]
    u /= np.linalg.norm(u)
    rotation = np.column_stack([u, np.cross(w, u), w])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    # Connected triangles share vertices; do not fill the gap under the arch.
    components = []
    pending = set(indices)
    keys = {i: {tuple(p) for p in np.round(triangles[i], 7)} for i in indices}
    while pending:
        component = {pending.pop()}
        while True:
            vertices = set().union(*(keys[i] for i in component))
            adjacent = {i for i in pending if keys[i] & vertices}
            if not adjacent:
                break
            component |= adjacent
            pending -= adjacent
        components.append(sorted(component))
    if len(components) != 2:
        raise ValueError("Expected two sole patches; inspect changed foot geometry")
    patches = []
    for ix in components:
        vertices = triangles[ix].reshape(-1, 3)
        projected = vertices @ rotation
        low, high = projected.min(0), projected.max(0)
        if high[2]-low[2] > 1e-6:
            raise ValueError("Sole patch is not planar")
        expected_area = np.prod((high-low)[:2])
        if not np.isclose(area[ix].sum(), expected_area, rtol=1e-4):
            raise ValueError("Sole patch is not a filled rectangle")
        surface_center = rotation @ ((low+high)/2)
        patches.append(dict(surface_center=surface_center, pos=surface_center+w*.001,
                            size=np.r_[(high-low)[:2]/2, .001], quat=quaternion.copy(),
                            normal=n.copy(), area=float(expected_area)))
    return sorted(patches, key=lambda p: p["surface_center"][0], reverse=True)


def compile_root(root):
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def scaffold():
    xml, audit = cad.build()
    root = ET.fromstring(xml)
    root.find("compiler").set("meshdir", str(cad.SOURCE.parent.resolve()))
    root.remove(root.find("option"))
    for body in root.iter("body"):
        name = BODY[body.get("name")]
        body.set("name", name)
        joint = body.find("joint")
        if joint is not None:
            stem, side = name.rsplit("_", 1)
            family = JOINT[stem]
            joint.set("name", f"{family}_{side}")
            if family in ("hip_roll", "cam", "thigh"):
                dyn = legacy.JOINT_DYN[family]
                joint.set("damping", str(dyn["damping"]))
                joint.set("armature", str(dyn["armature"]))
            else:
                joint.set("damping", "0.2")
                joint.set("armature", "0.001")
            if family == "cam":
                # Export has no limit. Narrow exploratory travel about CAD zero;
                # not a claim that these are calibrated hardware stops.
                joint.set("limited", "true")
                joint.set("range", "-0.6 0.6")
    for side, word, version, av in (("L", "left", 3, 7), ("R", "right", 2, 4)):
        renames = {
            f"loop_EXPORT_pushrod_{word}-v{version}_1_s1": f"rod_tip_{side}",
            f"loop_EXPORT_pushrod_{word}-v{version}_1_s2": f"leg_anchor_{side}",
            f"loop_EXPORT_achilles_{word}-v{av}_1_s1": f"achilles_tip_{side}",
            f"loop_EXPORT_achilles_{word}-v{av}_1_s2": f"foot_anchor_{side}",
        }
        for site in root.iter("site"):
            if site.get("name") in renames:
                site.set("name", renames[site.get("name")])
        for connection in root.findall("equality/connect"):
            for attr in ("site1", "site2"):
                if connection.get(attr) in renames:
                    connection.set(attr, renames[connection.get(attr)])
            connection.set("solref", "0.002 1")
            connection.set("solimp", "0.999 0.9999 0.0001")
    legacy._scaffold(root)
    root.set("model", "dash01_cad_v2")
    legacy._add_actuators_and_sensors(root)
    root.find("worldbody/geom[@name='floor']").set("conaffinity", "2")
    collision = root.find("default/default/default/geom")
    collision.set("contype", "2")
    collision.set("conaffinity", "1")
    patches = sole_patches()
    for side in "LR":
        foot = root.find(f".//body[@name='foot_{side}']")
        for tag, patch in zip(("foot", "heel"), patches):
            ET.SubElement(foot, "geom", name=f"{tag}_{side}_col", **{
                "class": "collision", "type": "box", "size": fs(patch["size"]),
                "pos": fs(patch["pos"]), "quat": fs(patch["quat"])})
    return root, audit, patches


def solve_pose(model, root, com_offset=0.):
    data = mujoco.MjData(model)
    # Six unconstrained base joints, followed by 14 leg hinges.
    indices = np.arange(6, model.nq)
    low, high = np.full(len(indices), -2.), np.full(len(indices), 2.)
    for j in range(6, model.njnt):
        if model.jnt_limited[j]:
            low[j-6], high[j-6] = model.jnt_range[j] + np.array([.005, -.005])
    pairs = [(model.site(c.get("site1")).id, model.site(c.get("site2")).id)
             for c in root.findall("equality/connect")]
    gids = [model.geom(f"{tag}_{side}_col").id for side in "LR" for tag in ("foot", "heel")]

    def residual(values):
        data.qpos[indices] = values
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        loops = np.concatenate([data.site_xpos[a]-data.site_xpos[b] for a, b in pairs])
        normals = np.concatenate([data.geom_xmat[g].reshape(3, 3)[:2, 2] for g in gids[::2]])
        centers = data.geom_xpos[gids]
        # Upright torso, flat/equal-height soles, COM centred fore-aft. Tiny
        # regularization chooses the nearby assembly branch in this redundant solve.
        return np.r_[loops*1000, normals*10,
                     (centers[:, 2]-centers[:, 2].mean())*100,
                     (data.subtree_com[1, 0]-centers[:, 0].mean()-com_offset)*10, values*.001]

    solution = least_squares(residual, np.zeros(len(indices)), bounds=(low, high),
                             xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
    residual(solution.x)
    z = model.joint("base_z").qposadr[0]
    gaps = [legacy._geom_floor_gap(model, data, g) for g in gids]
    data.qpos[z] -= max(gaps) + .0001
    mujoco.mj_forward(model, data)
    closure = max(np.linalg.norm(data.site_xpos[a]-data.site_xpos[b]) for a, b in pairs)
    tilt = max(np.degrees(np.arccos(np.clip(data.geom_xmat[g].reshape(3, 3)[2, 2], -1, 1))) for g in gids)
    spread = np.ptp([legacy._geom_floor_gap(model, data, g) for g in gids])
    if not solution.success or closure > 1e-5 or tilt > .2 or spread > .0002:
        raise ValueError(f"Stance solve failed: loop={closure:g}, tilt={tilt:g}, spread={spread:g}")
    return data.qpos.copy(), dict(loop_gap_m=float(closure), sole_tilt_deg=float(tilt),
                                 sole_height_spread_m=float(spread), nfev=solution.nfev,
                                 height_m=float(data.qpos[z]), com_world_m=data.subtree_com[1].tolist())


def solve_loaded_stance(model, qpos):
    """Find gravity/contact equilibrium, instead of sampling a moving settle.

    Translational XY and yaw gauges are fixed. Remaining coordinates and six
    motor torques minimize acceleration; weak pose regularization keeps the
    intended stance. No support force is added to the simulated robot.
    """
    data = mujoco.MjData(model)
    qadr = np.array([model.joint(n).qposadr[0] for n in ACT])
    indices = np.array([i for i in range(model.nq) if i not in (0, 1, 5)])
    pose = qpos[qadr].copy()

    def residual(values):
        data.qpos[:] = qpos
        data.qpos[indices] = values[:-6]
        data.qvel[:] = 0
        data.ctrl[:] = values[-6:]
        data.qacc_warmstart[:] = 0
        mujoco.mj_forward(model, data)
        return np.r_[data.qacc/100, (data.qpos[qadr]-pose)*.01, data.qpos[3:5]*.01]

    initial = np.r_[qpos[indices], [-.3, 12, 3.5, .3, -12, -3.5]]
    lower, upper = np.r_[qpos[indices]-.05, -model.actuator_forcerange[:, 1]], np.r_[qpos[indices]+.05, model.actuator_forcerange[:, 1]]
    result = least_squares(residual, initial, bounds=(lower, upper), x_scale="jac",
                           diff_step=1e-5, xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=700)
    residual(result.x)
    if np.max(np.abs(data.qacc)) > .01 or np.max(np.abs(data.qpos[qadr]-pose)) > .01:
        raise ValueError(f"Static equilibrium solve failed: acceleration {data.qacc}")
    settled = data.qpos.copy()
    torque = result.x[-6:].copy()
    return settled, settled[qadr]+torque/KP, torque, dict(
        nfev=result.nfev, max_acceleration=float(np.max(np.abs(data.qacc))),
        max_pose_adjustment_rad=float(np.max(np.abs(settled[qadr]-pose))))


def run_hold(model, qpos, command, seconds=5):
    """Unconstrained-base test with the nominal drive gains and torque-speed limits."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    qadr = np.array([model.joint(n).qposadr[0] for n in ACT])
    dadr = np.array([model.joint(n).dofadr[0] for n in ACT])
    peaks = model.actuator_forcerange[:, 1]
    kt = np.array([4.655, 2.176, 2.176]*2)
    resistance = np.array([.665, .229, .229]*2)
    start = data.xpos[1].copy()
    trace, max_tau, max_gap, peak_tilt, min_height, sat_steps = [], np.zeros(6), 0., 0., start[2], 0
    pairs = [(model.site(f"{a}_{s}").id, model.site(f"{b}_{s}").id)
             for s in "LR" for a, b in (("rod_tip", "leg_anchor"), ("achilles_tip", "foot_anchor"))]
    for tick in range(round(seconds/model.opt.timestep)):
        # Constant target is already present in the command buffer at reset;
        # delaying that constant cannot change it. Velocity feedback is local.
        speed = data.qvel[dadr]
        requested = KP*(command-data.qpos[qadr])-KD*speed
        limit = np.minimum(peaks, kt*np.maximum(48-kt*np.abs(speed), 0)/resistance)
        tau = np.clip(requested, -limit, limit)
        sat_steps += int(np.any(np.abs(requested) > limit+1e-6))
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            raise ValueError("Nonfinite state/MuJoCo warning during hold")
        mujoco.mj_forward(model, data)
        tilt = np.degrees(np.arccos(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1, 1)))
        peak_tilt = max(peak_tilt, float(tilt))
        min_height = min(min_height, float(data.xpos[1, 2]))
        max_tau = np.maximum(max_tau, np.abs(tau))
        max_gap = max(max_gap, max(np.linalg.norm(data.site_xpos[a]-data.site_xpos[b]) for a, b in pairs))
        if tick % 10 == 0:
            trace.append(np.r_[data.time, data.qpos, data.qvel, tau])
        if tilt > 30 or data.xpos[1, 2] < .7*start[2]:
            break
    drift = np.linalg.norm(data.xpos[1, :2]-start[:2])
    result = dict(duration_s=float(data.time), max_tilt_deg=peak_tilt,
                  height_drop_m=float(start[2]-min_height), drift_xy_m=float(drift),
                  final_base_speed_mps=float(np.linalg.norm(data.qvel[:3])),
                  peak_torque_nm=max_tau.tolist(), max_loop_gap_m=float(max_gap),
                  saturation_fraction=sat_steps/(tick+1), warnings=0,
                  passed=bool(data.time >= seconds-.001 and peak_tilt < 5 and
                              start[2]-min_height < .02 and drift < .03 and max_gap < .001))
    return result, np.asarray(trace), data


def build():
    root, source_audit, patches = scaffold()
    model = compile_root(root)
    qpos, stance = solve_pose(model, root)
    qadr = np.array([model.joint(n).qposadr[0] for n in ACT])
    pose = qpos[qadr]
    qpos, command, tau, equilibrium = solve_loaded_stance(model, qpos)
    pose = qpos[qadr]
    hold, trace, _ = run_hold(model, qpos, command)
    disturbed_holds = []
    for pitch in (-.001, .001):
        disturbed = qpos.copy()
        disturbed[model.joint("base_pitch").qposadr[0]] += pitch
        result, _, _ = run_hold(model, disturbed, command)
        disturbed_holds.append(dict(pitch_perturbation_rad=pitch, **result))
    audit = dict(model="dash01_cad_v2", mass_kg=float(model.body_mass.sum()),
                 motors="unchanged; user confirmed 2026-10-14", achilles="rigid; no spring; user confirmed",
                 joint_zero="CAD-relative simulation coordinates; hardware homing not calibrated",
                 cam_range="provisional simulation guard +/-0.6 rad about CAD pose",
                 contact_model="two mesh-derived rectangular sole faces per foot, 2 mm inward; friction 0.7 provisional",
                 stance=stance, equilibrium=equilibrium, nominal_pose=pose.tolist(), nominal_command=command.tolist(),
                 holding_torque_nm=tau.tolist(), hold=hold,
                 dimensions=dict(nq=model.nq, nv=model.nv, nu=model.nu, neq=model.neq),
                 corrected_cad_sha256=source_audit["generated_xml_sha256"],
                 builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 cpu_nominal_hold_passed=hold["passed"], perturbed_holds=disturbed_holds,
                 fixed_target_balance_passed=all(r["passed"] for r in disturbed_holds),
                 training_ready=False,
                 next_gate="CPU/MJX physics comparison and finite PPO update; train walking directly",
                 deployment_ready=False)
    root.find("custom/numeric[@name='nominal_ctrl']").set("data", fs(pose))
    ET.SubElement(root.find("custom"), "numeric", name="nominal_cmd", data=fs(command))
    ET.SubElement(root.find("custom"), "numeric", name="cad_v2", data="1")
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="stand", qpos=fs(qpos), ctrl="0 0 0 0 0 0")
    root.find("compiler").set("meshdir", "../../dash-01CAD-v2/EXPORT_assembly_closed")
    ET.indent(root, space="  ")
    xml = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
    return xml, audit, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    xml, audit, trace = build()
    print(json.dumps(audit, indent=2))
    if not args.report:
        OUT.write_bytes(xml.encode("utf-8"))
        folder = cad.CAD / "generated"
        (folder / "plant_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
        np.savez_compressed(folder / "standing_trace.npz", trace=trace)
    if not audit["hold"]["passed"]:
        raise SystemExit("Standing gate failed: training must not launch.")


if __name__ == "__main__":
    main()
