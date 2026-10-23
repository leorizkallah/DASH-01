"""Recover the supplied homing assembly pose in the calibrated CAD-v2 coordinates."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

try:
    from . import make_cad_v2_plant as builder
except ImportError:
    import make_cad_v2_plant as builder

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "dash-01CAD-v2/homing/EXPORT_assembly_stable_closed/EXPORT_assembly-v36_closed.xml"
)
BASE = ROOT / "RLframework/model/dash01_cad_v2_free.xml"
OUT = BASE.with_name("dash01_cad_v2_homing.xml")
AUDIT = ROOT / "report/data/homing01/audit.json"


def build():
    # Same part meshes: use the new export ONLY as a pose reference. Retain reviewed
    # mass/inertia, 1 kg torso allowance, joints, contacts and rigid link constraints.
    for p in (SOURCE.parent / "meshes").glob("*.stl"):
        old = ROOT / "dash-01CAD-v2/EXPORT_assembly_closed/meshes" / p.name
        if p.read_bytes() != old.read_bytes():
            raise ValueError(f"Changed mesh needs a new model audit: {p.name}")
    target = mujoco.MjModel.from_xml_path(str(SOURCE))
    td = mujoco.MjData(target)
    mujoco.mj_forward(target, td)
    m = mujoco.MjModel.from_xml_path(str(BASE))
    d = mujoco.MjData(m)
    ids = [(m.body(name).id, target.body(raw).id) for raw, name in builder.BODY.items()]
    q = m.key_qpos[0].copy()
    q[:6] = 0.0
    tids = np.array([t for _, t in ids])
    mids = np.array([i for i, _ in ids])
    target_pos = td.xpos[tids] - td.xpos[target.body("EXPORT_body-v4_1").id]
    target_rot = td.xmat[tids].reshape(-1, 3, 3)
    low, high = np.full(m.nq - 6, -3.0), np.full(m.nq - 6, 3.0)
    for j in range(6, m.njnt):
        if m.jnt_limited[j]:
            low[j - 6], high[j - 6] = m.jnt_range[j]

    def residual(x):
        d.qpos[:] = q
        d.qpos[6:] = x
        mujoco.mj_forward(m, d)
        rot = d.xmat[mids].reshape(-1, 3, 3)
        angle = Rotation.from_matrix(rot @ target_rot.transpose(0, 2, 1)).as_rotvec()
        return np.r_[(d.xpos[mids] - target_pos).ravel(), angle.ravel()]

    fit = least_squares(
        residual,
        np.clip(q[6:], low + 1e-8, high - 1e-8),
        bounds=(low, high),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    residual(fit.x)
    pos_error = float(np.max(np.linalg.norm(d.xpos[mids] - target_pos, axis=1)))
    angle_error = float(np.max(np.abs(residual(fit.x)[len(ids) * 3 :])))
    if not fit.success or pos_error > 1e-5 or angle_error > 1e-4:
        raise ValueError(
            f"Homing geometry mismatch: position={pos_error}, angle={angle_error}"
        )
    q = d.qpos.copy()
    pads = [m.geom(f"{tag}_{s}_col").id for s in "LR" for tag in ("foot", "heel")]
    gaps = [builder.legacy._geom_floor_gap(m, d, g) for g in pads]
    q[2] -= min(gaps) + 0.0001  # Seed 0.1 mm contact compression for the loaded solve.
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    pairs = [
        (m.site(f"{a}_{s}").id, m.site(f"{b}_{s}").id)
        for s in "LR"
        for a, b in (("rod_tip", "leg_anchor"), ("achilles_tip", "foot_anchor"))
    ]
    closure = max(np.linalg.norm(d.site_xpos[a] - d.site_xpos[b]) for a, b in pairs)
    if closure > 1e-5:
        raise ValueError(f"Homing linkage gap {closure}")
    # Contact-loaded equilibrium may make a small explicit adjustment to the CAD
    # pose; retain exact source pose in the audit for review.
    q_source = q.copy()
    print(
        "Homing fit",
        dict(
            position_error=pos_error,
            rotation_error=angle_error,
            gap_spread=float(np.ptp(gaps)),
            qpos=q.tolist(),
        ),
        flush=True,
    )
    q, command, tau, equilibrium = builder.solve_loaded_stance(m, q)
    if np.max(np.abs(q[6:] - q_source[6:])) > 0.01:
        raise ValueError(
            "Equilibrium would alter homing leg pose by more than 0.01 rad"
        )
    qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    if np.any(command < m.jnt_range[m.actuator_trnid[:, 0], 0]) or np.any(
        command > m.jnt_range[m.actuator_trnid[:, 0], 1]
    ):
        raise ValueError("Homing PD target is outside motor range")
    hold, trace, _ = builder.run_hold(m, q, command)
    root = ET.parse(BASE).getroot()
    root.set("model", "dash01_cad_v2_homing")
    key = root.find("keyframe/key")
    key.set("name", "homing")
    key.set("qpos", builder.fs(q))
    root.find("custom/numeric[@name='nominal_ctrl']").set("data", builder.fs(q[qadr]))
    root.find("custom/numeric[@name='nominal_cmd']").set("data", builder.fs(command))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(OUT, encoding="utf-8", xml_declaration=True)
    audit = dict(
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
        max_body_position_fit_error_m=pos_error,
        max_rotation_fit_error_rad=angle_error,
        source_qpos=q_source.tolist(),
        loaded_qpos=q.tolist(),
        sole_gap_spread_before_loading_m=float(np.ptp(gaps)),
        source_loop_gap_m=float(closure),
        equilibrium=equilibrium,
        hold=hold,
        nominal_command=command.tolist(),
        motor_pose=q[qadr].tolist(),
        holding_torque=tau.tolist(),
        mass_kg=float(m.body_mass.sum()),
        note="Homing export used only as pose reference. Existing calibrated inertials/geometry retained. "
        "Small contact-loading adjustment reported explicitly; no balance rail or spring.",
    )
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    np.savez_compressed(AUDIT.with_name("hold_trace.npz"), qpos=trace)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    build()
