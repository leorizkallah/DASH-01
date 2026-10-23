"""Supplementary CPU motor/contact diagnostics using Nathann's deployment Sim."""

import json
from types import SimpleNamespace
import numpy as np
import mujoco
from cpu_walk_preview import export, ROOT
from play_joystick import Sim
from walk01_diagnostics import ground_forces


def main():
    out = ROOT / "report/data/walk01-diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    for label in ["best", "final"]:
        bundle, cfg, export_meta = export(label)
        trials = []
        for target in [0.25, 0.75, 1.5]:
            sim = Sim(bundle)
            sim.set_stick(target / cfg["v_max"])
            sim.reset()
            sim.ctrl.set_speed(target, immediate=True)
            m, d = sim.model, sim.data
            p = SimpleNamespace(
                floor_gid=m.geom("floor").id,
                foot_gids=[m.geom("foot_" + s + "_col").id for s in "LR"],
                heel_gids=[m.geom("heel_" + s + "_col").id for s in "LR"],
            )
            rows = {
                k: []
                for k in [
                    "qpos",
                    "feet",
                    "clearance",
                    "asymmetry",
                    "done",
                    "thermal",
                    "speed",
                    "tau",
                    "qd",
                    "q",
                    "limit",
                    "grf",
                ]
            }
            real = mujoco.mj_step

            def tap(model, data, *a, **kw):
                qd = data.qvel[sim.act_d].copy()
                rows["qd"].append(qd)
                rows["q"].append(data.qpos[sim.act_q].copy())
                rows["limit"].append(
                    np.minimum(
                        sim.tau_peak,
                        sim.kt
                        * np.maximum(sim.v_bus - sim.kt * np.abs(qd), 0)
                        / sim.r_ohm,
                    )
                )
                real(model, data, *a, **kw)
                rows["tau"].append(data.actuator_force.copy())
                rows["grf"].append(ground_forces(data, p, np))

            mujoco.mj_step = tap
            thermal = np.zeros(6)
            try:
                for k in range(2500):
                    rows["qpos"].append(d.qpos.copy())
                    R = d.xmat[sim.base_bid].reshape(3, 3)
                    rows["feet"].append(
                        (d.geom_xpos[p.foot_gids] - d.xpos[sim.base_bid]) @ R
                    )
                    rows["clearance"].append(
                        [
                            min(
                                d.geom_xpos[g, 2]
                                - np.abs(d.geom_xmat[g].reshape(3, 3)[2])
                                @ m.geom_size[g]
                                for g in pair
                            )
                            for pair in zip(p.foot_gids, p.heel_gids)
                        ]
                    )
                    rows["asymmetry"].append(float(sim.ctrl._spec[37]))
                    rows["thermal"].append(thermal.copy())
                    sim.control_tick()
                    if any(w.number for w in d.warning):
                        raise RuntimeError("CPU numerical warning")
                    fallen = bool(
                        not np.isfinite(d.qpos).all()
                        or sim._grav_body()[2] > cfg["term_gravity_z"]
                        or d.qpos[2] < cfg["term_height"]
                    )
                    rows["done"].append(fallen)
                    rows["speed"].append(sim.speed())
                    thermal += (
                        0.01
                        / cfg["thermal_tau_s"]
                        * (
                            np.mean(np.asarray(rows["tau"][-10:]) ** 2, axis=0)
                            / np.asarray(cfg["thermal_tau_cont"]) ** 2
                            - thermal
                        )
                    )
                    if fallen:
                        raise RuntimeError(
                            f"{label} CPU trial fell; inspect before aggregating"
                        )
            finally:
                mujoco.mj_step = real
            for key in ["tau", "qd", "q", "limit", "grf"]:
                v = np.asarray(rows[key])
                rows[key] = v.reshape((2500, 10) + v.shape[1:])
            trials.append(rows)
            print(label, target, "complete", flush=True)
        arrays = {
            k: np.stack([np.asarray(r[k]) for r in trials], axis=1) for k in trials[0]
        }
        meta = dict(
            backend="CPU deployment preview; independent of the MJX diagnostic replay",
            checkpoint_sha256=export_meta["checkpoint_sha256"],
            kt=cfg["motor_kt_joint"],
            resistance=cfg["motor_r_ohm"],
            bus_volts=cfg["motor_bus_volts"],
            torque_peak=sim.tau_peak.tolist(),
            torque_cont=cfg["thermal_tau_cont"],
            mass=float(m.body_mass.sum()),
            note="Native CPU solver 100/50. No training reward recomputation. Thermal reconstructed "
            "from torque history, cold start, nominal constants. Not a measurement on the MJX video trajectory.",
        )
        np.savez_compressed(out / (label + "_cpu.npz"), meta=json.dumps(meta), **arrays)


if __name__ == "__main__":
    main()
