"""Two-pad contact measurements and completed-cycle symmetry (NumPy/JAX pure functions)."""

import numpy as np


def validate(c):
    if not c.two_point_feet:
        if c.symmetry_enable or c.flat_support_enable:
            raise ValueError(
                "Flat support and completed-step symmetry require two_point_feet"
            )
        return
    if not (0 <= c.flat_stance_start_fraction < c.flat_stance_end_fraction <= 1):
        raise ValueError(
            "Flat-support stance fractions must satisfy 0 <= start < end <= 1"
        )
    if c.foot_contact_min_force_n <= 0 or c.symmetry_min_phase_s <= 0:
        raise ValueError("Contact force and minimum phase duration must be positive")
    if not (1 <= c.symmetry_min_steps_per_foot <= c.symmetry_window_steps):
        raise ValueError(
            "Symmetry window must cover the minimum number of completed steps"
        )
    if c.symmetry_persistent_pairs < 2:
        raise ValueError("Termination must require multiple completed step pairs")


def contact_measurements(data, p, threshold, xp=np):
    """Force-qualified toe/heel contact and force-weighted contact-point slip.

    Elliptic cone: the first constraint row is normal force. cvel is a world
    spatial velocity at the body's root-subtree COM, ordered angular, linear.
    The floor is static. Project each point velocity onto its contact tangent
    plane. Average within each FOOT, so extra heel/toe contact points do not
    double its penalty. Contact force, penetration and floor-pair membership
    are all required; height proximity alone never counts as support.
    """
    impl = getattr(data, "_impl", data)
    con = impl.contact
    adr = con.efc_address
    fn = xp.where(
        (adr >= 0) & (con.dist <= 0),
        xp.maximum(impl.efc_force[xp.maximum(adr, 0)], 0.0),
        0.0,
    )
    floor = (con.geom == p.floor_gid).any(axis=1)
    pads, forces, slips = [], [], []
    for i in range(2):
        masks = xp.stack(
            [
                floor & (con.geom == g).any(axis=1)
                for g in (p.foot_gids[i], p.heel_gids[i])
            ]
        )
        force = xp.sum(xp.where(masks, fn[None, :], 0.0), axis=1)
        down = force >= threshold
        active = xp.any(masks & down[:, None], axis=0)
        bid = p.foot_bids[i]
        cv = data.cvel[bid]
        origin = data.subtree_com[int(p.m.body_rootid[bid])]
        vel = cv[3:] + xp.cross(cv[:3], con.pos - origin)
        normal = con.frame.reshape((-1, 3, 3))[:, 0, :]
        tangent = vel - xp.sum(vel * normal, axis=1)[:, None] * normal
        weight = xp.where(active, fn, 0.0)
        slip2 = xp.sum(weight * xp.sum(tangent**2, axis=1)) / xp.maximum(
            weight.sum(), 1e-8
        )
        pads.append(down)
        forces.append(force)
        slips.append(xp.sqrt(slip2))
    pads = xp.stack(pads)
    return dict(
        toe=pads[:, 0],
        heel=pads[:, 1],
        grounded=pads.any(axis=1),
        flat=pads.all(axis=1),
        normal_force=xp.stack(forces),
        slip=xp.stack(slips),
    )


def midstance(phases, stance_ratio, start, end, xp=np):
    phase = xp.mod(phases, 2 * np.pi) / (2 * np.pi)
    return (phase >= stance_ratio * start) & (phase < stance_ratio * end)


def initial(grounded, c, xp=np):
    return dict(
        prev=xp.asarray(grounded, bool),
        stance=xp.zeros(2),
        swing=xp.zeros(2),
        last_stance=xp.zeros(2),
        lifted=xp.zeros(2, bool),
        stance_window=xp.zeros((2, c.symmetry_window_steps)),
        swing_window=xp.zeros((2, c.symmetry_window_steps)),
        count=xp.zeros(2, xp.int32),
        checked=xp.zeros(2, xp.int32),
        bad_pairs=xp.zeros((), xp.int32),
        walking_s=xp.zeros(()),
        moving_s=xp.zeros(()),
        recovery=xp.zeros(()),
    )


def update(s, grounded, walking, vx, disturbed, curriculum, c, dt, xp=np):
    """One valid step = touchdown after >=min swing and preceding >=min stance.

    Windows contain completed phases only. E is the MEAN of stance and swing
    relative differences, normalized by the L/R mean (range 0..2). New evidence
    is counted only when BOTH feet have completed a new cycle since comparison.
    An idle foot is handled separately after the other completes multiple cycles.
    """
    touchdown = grounded & ~s["prev"]
    liftoff = ~grounded & s["prev"]
    last = xp.where(liftoff, s["stance"], s["last_stance"])
    valid = (
        touchdown
        & s["lifted"]
        & (s["swing"] >= c.symmetry_min_phase_s)
        & (last >= c.symmetry_min_phase_s)
    )
    slots = (
        xp.arange(c.symmetry_window_steps)[None, :]
        == (s["count"] % c.symmetry_window_steps)[:, None]
    )
    sw = xp.where(valid[:, None] & slots, s["swing"][:, None], s["swing_window"])
    st = xp.where(valid[:, None] & slots, last[:, None], s["stance_window"])
    count = s["count"] + valid.astype(xp.int32)
    denom = xp.maximum(xp.minimum(count, c.symmetry_window_steps), 1)
    stance_mean, swing_mean = st.sum(axis=1) / denom, sw.sum(axis=1) / denom
    err = lambda a: (
        xp.abs(a[0] - a[1]) / xp.maximum(0.5 * (a[0] + a[1]), c.symmetry_min_phase_s)
    )
    eligible = (count >= c.symmetry_min_steps_per_foot).all()
    error = xp.where(eligible, 0.5 * (err(stance_mean) + err(swing_mean)), 0.0)
    stance = xp.where(grounded, s["stance"] + dt, 0.0)
    swing = xp.where(~grounded, s["swing"] + dt, 0.0)
    recovery = xp.where(
        disturbed, c.symmetry_recovery_s, xp.maximum(s["recovery"] - dt, 0.0)
    )
    active = walking & (recovery <= 0.0)
    walk_s = xp.where(active, s["walking_s"] + dt, 0.0)
    moving_s = xp.where(
        active & (vx >= c.symmetry_min_forward_mps), s["moving_s"] + dt, 0.0
    )
    # Several steps by the other leg, plus long actual stance, are required.
    planted = (
        grounded
        & (stance >= c.symmetry_planted_s)
        & (count[::-1] - count >= c.symmetry_planted_other_steps)
    ).any()
    planted = (
        planted
        & (walk_s >= c.symmetry_planted_grace_s)
        & (moving_s >= c.symmetry_motion_s)
    )
    pair = eligible & (count > s["checked"]).all()
    gate = (
        active
        & (walk_s >= c.symmetry_grace_s)
        & (curriculum >= c.symmetry_termination_shape_scale)
    )
    bad = pair & (error > c.symmetry_terminate_threshold)
    bad_pairs = xp.where(
        ~gate, 0, xp.where(pair, xp.where(bad, s["bad_pairs"] + 1, 0), s["bad_pairs"])
    ).astype(xp.int32)
    term = gate & ((bad_pairs >= c.symmetry_persistent_pairs) | planted)
    penalty = -c.w_symmetry * xp.where(
        active, xp.maximum(error, planted.astype(xp.float32)), 0.0
    )
    out = dict(
        prev=grounded,
        stance=stance,
        swing=swing,
        last_stance=last,
        lifted=xp.where(touchdown, False, s["lifted"] | liftoff),
        stance_window=st,
        swing_window=sw,
        count=count,
        checked=xp.where(pair, count, s["checked"]),
        bad_pairs=bad_pairs,
        walking_s=walk_s,
        moving_s=moving_s,
        recovery=recovery,
    )
    # Forget partial phases/history during STOP or recovery; restart grace afterwards.
    clean = initial(grounded, c, xp)
    out = {k: xp.where(active, v, clean[k]) for k, v in out.items()}
    out["recovery"] = recovery
    metrics = dict(
        error=error,
        valid_step=valid & active,
        touchdown=touchdown,
        liftoff=liftoff,
        stance_mean=stance_mean,
        swing_mean=swing_mean,
        count=count,
        planted=planted & active,
        gate=gate,
        bad_pairs=bad_pairs,
    )
    return out, penalty, term, metrics
