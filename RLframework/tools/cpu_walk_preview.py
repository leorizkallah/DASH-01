"""WALK-01 CPU deployment previews while MJX replay waits; no learning or weight edits."""

import hashlib
import json
from pathlib import Path

import msgpack
import mujoco
import numpy as np
from play_joystick import Sim

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/experiments/walk01_20261019_v1"
OUT = ROOT / "RLframework/results/walk01_replay"


def unpack(code, blob):
    # Flax serialization's numeric extension format; reject unknown encodings.
    if code in (1, 3):
        shape, dtype, raw = msgpack.unpackb(blob, raw=True)
        a = np.frombuffer(raw, dtype=np.dtype(dtype.decode())).reshape(shape)
        return a if code == 1 else a[()]
    if code == 2:
        return complex(*msgpack.unpackb(blob))
    raise ValueError(f"Unsupported checkpoint extension {code}")


def export(label):
    cfg = json.loads((DATA / "resolved_config.json").read_text())["config"]
    checkpoint = OUT / f"{label}.msgpack"
    state = msgpack.unpackb(
        checkpoint.read_bytes(), ext_hook=unpack, raw=False, strict_map_key=False
    )
    p = state["params"]["params"]
    with np.load(
        ROOT / "controller/deploy/bundles/dash_joy3_lr_s2_180M.npz", allow_pickle=False
    ) as z:
        arrays = {k: z[k].copy() for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    # Same ordinary 384/47 interface; replace every network/statistic with this checkpoint.
    for prefix, module, count in [("est", "estimator", 3), ("pi", "policy_net", 2)]:
        for i in range(count):
            arrays[f"{prefix}_w{i}"] = p[module][f"Dense_{i}"]["kernel"].T.copy()
            arrays[f"{prefix}_b{i}"] = p[module][f"Dense_{i}"]["bias"].copy()
    arrays["act_w"], arrays["act_b"] = (
        p["action_net"]["kernel"].T.copy(),
        p["action_net"]["bias"].copy(),
    )
    arrays["log_std"] = p["log_std"].copy()
    arrays["obs_mean"], arrays["obs_var"] = (
        state["stats"]["mean"][:384],
        state["stats"]["var"][:384],
    )
    assert arrays["est_w0"].shape[1] == 384 and arrays["act_w"].shape[0] == 47
    for key in meta:
        if key in cfg:
            meta[key] = cfg[key]
    m = mujoco.MjModel.from_xml_path(str(ROOT / "RLframework" / cfg["model_path"]))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert names == meta["actuator_names"]
    qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    arrays["default_motor_pos"] = m.key_qpos[m.key(cfg["keyframe"]).id, qadr].copy()
    n = m.numeric("nominal_cmd")
    arrays["nominal_ctrl"] = n.data.copy()
    arrays["q_lo"], arrays["q_hi"] = m.jnt_range[m.actuator_trnid[:, 0]].T.copy()
    arrays["forcerange"] = m.actuator_forcerange[:, 1].copy()
    for key in ["motor_vel_limit", "drive_kp", "drive_kd"]:
        arrays[key] = np.asarray(cfg[key], float)
    arrays["stand_torque"] = arrays["drive_kp"] * (
        arrays["nominal_ctrl"] - arrays["default_motor_pos"]
    )
    arrays["hist_idx"] = np.arange(cfg["history_len"]) * cfg["history_stride"]
    arrays["latched_dims"] = np.r_[np.ones(41, bool), np.zeros(6, bool)]
    ep = json.loads((DATA / f"{label}.json").read_text())["env_params"]
    gait = meta["gait"]
    for key in gait:
        if key in cfg:
            gait[key] = cfg[key]
    gait.update(
        freq_lo=ep.get("gait_freq_lo") or cfg["gait_freq_hz"][0],
        freq_hi=cfg["gait_freq_hz"][1],
        o_max=[cfg["o_cam_max"], cfg["o_thigh_max"], cfg["o_hip_max"]],
    )
    meta.update(
        run="cad_v2_walk01_s0",
        checkpoint=checkpoint.name,
        step=int(state["step"]),
        gait=gait,
        env_params_at_checkpoint=ep,
        curriculum_source=f"{label}.json",
        height_stand=float(m.key_qpos[m.key(cfg["keyframe"]).id, 2]),
        policy_std_mean=float(np.exp(arrays["log_std"]).mean()),
        preview_only=True,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    out = OUT / f"{label}_cpu.npz"
    np.savez(out, meta=json.dumps(meta), **arrays)
    # Exact serialized actor/statistics correspondence (no old weights retained).
    assert np.array_equal(arrays["act_w"].T, p["action_net"]["kernel"])
    assert np.array_equal(arrays["obs_mean"], state["stats"]["mean"][:384])
    return out, cfg, meta


def main():
    for label in ["best", "final"]:
        bundle, cfg, meta = export(label)
        targets = [0.25, 0.75, 1.5]
        qs = []
        vs = []
        trials = []
        for target in targets:
            sim = Sim(bundle)
            # Keep classic MuJoCo's XML solver budget (100/50). MJX's 16/8 cap
            # is not interchangeable: it produced QACC warnings in CPU probes.
            sim.set_stick(target / cfg["v_max"])
            sim.reset()
            sim.ctrl.set_speed(target, immediate=True)
            q = []
            v = []
            fell = False
            for k in range(2500):
                q.append(sim.data.qpos.copy())
                sim.control_tick()
                v.append(sim.speed())
                if any(w.number for w in sim.data.warning):
                    raise RuntimeError(
                        "CPU physics warning; do not render this as a policy failure"
                    )
                fell = bool(
                    not np.isfinite(sim.data.qpos).all()
                    or sim._grav_body()[2] > cfg["term_gravity_z"]
                    or sim.data.qpos[2] < cfg["term_height"]
                )
                if fell:
                    break
            n = len(q)
            q.extend([q[-1]] * (2500 - n))
            v.extend([v[-1]] * (2500 - n))
            qs.append(q)
            vs.append(v)
            trials.append(
                dict(
                    target=target,
                    ticks=n,
                    seconds=n * 0.01,
                    fell=fell,
                    mean_speed_after_5s=float(np.mean(v[500:n])) if n > 500 else None,
                    speed_error_after_5s=float(
                        np.mean(np.abs(np.asarray(v[500:n]) - target))
                    )
                    if n > 500
                    else None,
                )
            )
        record = dict(
            step=meta["step"],
            dt=0.01,
            model=cfg["model_path"],
            checkpoint=meta["checkpoint"],
            checkpoint_sha256=meta["checkpoint_sha256"],
            label=f"WALK-01 {label} | CPU preview",
            trials=trials,
            note="CPU deployment simulation; not the queued MJX replay. Original trained weights and stats, "
            "nominal homing reset, 12 ms delay, no rail, XML CPU solver 100/50. Thermal evolution and full MJX termination "
            "checks are not reproduced. No randomization; single trial per target. Flight field is "
            "a rendering placeholder and must not be interpreted as a measurement.",
        )
        path = DATA / f"walk01_cpu_{label}.npz"
        np.savez_compressed(
            path,
            meta=json.dumps(record),
            qpos=np.swapaxes(qs, 0, 1),
            vx=np.asarray(vs).T,
            flight=np.zeros((2500, 3)),
        )
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
