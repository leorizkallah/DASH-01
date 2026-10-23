"""Speed-discovery reward contract and first-episode selection (NumPy/JAX)."""

import numpy as np

REWARD_KEYS = (
    "fwd_speed",
    "upright",
    "height",
    "heading",
    "lat_vel",
    "action_rate",
    "residual",
    "residual_rate",
    "torque",
    "motor_vel",
    "energy",
    "thermal",
)


def income(vx, upright, cfg, xp=np):
    raw = cfg.w_fwd_speed * xp.clip(vx, -cfg.v_ceiling, cfg.v_ceiling)
    gate = xp.clip(
        (upright - cfg.speed_upright_c0) / (1.0 - cfg.speed_upright_c0), 0.0, 1.0
    )
    return xp.where(raw > 0.0, raw * gate**cfg.speed_upright_k, raw)


def terms(existing, slip, shape_scale, cfg, xp=np):
    selected = {k: existing[k] for k in REWARD_KEYS}
    # RMS active contact-point velocity: forward, backward and sideways all count.
    selected["contact_slip"] = (
        -cfg.speed_slip_weight
        * shape_scale
        * xp.mean(xp.minimum((slip / 0.1) ** 2, 4.0))
    )
    return selected


def summarize(rows, dt, duration, warmup=5.0):
    active = np.asarray(rows["active"], bool)
    n = active.shape[1]
    count = np.maximum(active.sum(axis=0), 1)
    late = active & (np.arange(len(active))[:, None] * dt >= warmup)
    late_count = np.maximum(late.sum(axis=0), 1)
    average = lambda x: np.where(active, x, 0.0).sum(axis=0) / count
    survived = (~(rows["fallen"] & active).any(axis=0)) & (
        active.sum(axis=0) * dt >= duration - dt / 2
    )
    distance = np.where(active, rows["vx"], 0.0).sum(axis=0) * dt
    steady = np.where(late, rows["vx"], 0.0).sum(axis=0) / late_count
    joint = np.where(active, rows["joint_violation"], 0.0).max(axis=0)
    hot = np.where(active, rows["thermal"], 0.0).max(axis=0)
    qualified = survived & (joint <= 0.01) & (hot <= 1.0) & (steady > 0.02)
    m = dict(
        survived=survived,
        qualified=qualified,
        elapsed_s=active.sum(axis=0) * dt,
        distance_m=distance,
        speed_over_full_horizon=distance / duration,
        steady_speed=steady,
        steady_observed_s=late.sum(axis=0) * dt,
        heading_error=average(np.abs(rows["heading"])),
        slip=np.where(active[..., None], rows["slip"], 0.0).sum(axis=0)
        / count[:, None],
        clear_fraction=np.where(late[..., None], rows["clear"], 0.0).sum(axis=0)
        / late_count[:, None],
        contact_fraction=np.where(active[..., None], rows["contact"], 0.0).sum(axis=0)
        / count[:, None],
        flight_fraction=average(rows["flight"]),
        ceiling_fraction=average(rows["ceiling"]),
        floor_fraction=average(rows["floor"]),
        thermal_peak=hot,
        joint_violation=joint,
        reward=average(rows["reward"]),
    )
    eligible = bool(qualified.mean() >= 0.9)
    # Failures consume the full horizon; a brief fast fall cannot inflate speed.
    rank = [
        float(qualified.mean()),
        float(survived.mean()),
        float((distance / duration).mean()),
    ]
    return dict(
        trials=n,
        survival_fraction=float(survived.mean()),
        qualified_fraction=float(qualified.mean()),
        keeper_eligible=eligible,
        selection_score=float((distance / duration).mean()),
        candidate_rank=rank,
        per_trial={k: np.asarray(v).tolist() for k, v in m.items()},
        finite=bool(
            all(
                np.isfinite(np.asarray(rows[k])[active]).all()
                for k in ("reward", "vx", "thermal", "joint_violation")
            )
        ),
        note="Nominal speed feasibility, not clean-walking certification; first episode only. "
        "Failed trials retain elapsed time/distance and consume full ranking horizon.",
    )
