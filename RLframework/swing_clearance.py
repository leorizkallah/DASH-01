"""Phase-dependent whole-foot clearance shaping, with NumPy/JAX-compatible math."""

import numpy as np


def validate(c):
    if not c.swing_clearance_enable:
        return
    if not c.two_point_feet:
        raise ValueError("Swing clearance requires force-qualified two-point feet")
    if not 0 < c.swing_clearance_min_m < c.swing_clearance_bonus_cap_m:
        raise ValueError("Clearance heights must satisfy 0 < minimum < bonus cap")
    if not 0 < c.swing_clearance_ramp_fraction < 0.5:
        raise ValueError("Swing edge ramp must be between zero and half a swing")
    if (
        min(
            c.w_low_swing_clearance,
            c.w_extra_swing_clearance,
            c.w_excess_swing_clearance,
        )
        < 0
    ):
        raise ValueError("Clearance weights must be nonnegative")


def shaping(heights, phases, stance_ratio, grounded, walking, air_time, c, xp=np):
    """Configured height plateau, with ground-level swing endpoints ramped away.

    Charge missing height on the SCHEDULED swing even if the foot stays grounded.
    Bonus requires actual short-duration air, saturates at the cap, and is zero in
    stance. Sum both feet; never average away a weak foot or bill each pad twice.
    """
    phase = xp.mod(phases, 2 * np.pi) / (2 * np.pi)
    u = (phase - stance_ratio) / xp.maximum(1.0 - stance_ratio, 1e-6)
    swing = (u >= 0.0) & (u < 1.0) & walking
    ramp = c.swing_clearance_ramp_fraction
    envelope = xp.where(
        swing, xp.clip(xp.minimum(u / ramp, (1.0 - u) / ramp), 0.0, 1.0), 0.0
    )
    target = c.swing_clearance_min_m * envelope
    h = xp.maximum(heights, 0.0)
    deficit = xp.clip((target - h) / c.swing_clearance_min_m, 0.0, 1.0)
    low = -c.w_low_swing_clearance * xp.sum(deficit)
    extra = xp.clip(
        (h - c.swing_clearance_min_m)
        / (c.swing_clearance_bonus_cap_m - c.swing_clearance_min_m),
        0.0,
        1.0,
    )
    bonus_gate = swing & ~grounded & (air_time <= c.swing_fresh_s)
    extra = envelope * extra * bonus_gate
    bonus = c.w_extra_swing_clearance * xp.sum(extra)
    # A saturated bonus alone leaves excessive lift unpenalized. Keep this term
    # independent, zero by default for all previous presets, and swing-only.
    excess = xp.where(swing, xp.maximum(h - c.swing_clearance_bonus_cap_m, 0.0), 0.0)
    excess_cost = -c.w_excess_swing_clearance * xp.sum(
        excess / (c.swing_clearance_bonus_cap_m - c.swing_clearance_min_m)
    )
    metrics = dict(
        clearance=heights,
        clearance_swing=swing,
        clearance_middle=swing & (u >= ramp) & (u <= 1.0 - ramp),
        clearance_target=target,
        clearance_deficit=deficit,
        clearance_extra=extra,
        clearance_excess_cost=excess_cost,
    )
    return low, bonus, metrics
