"""Standing contact/support math shared by NumPy validation and JAX training."""
import numpy as np


def sole_centers(data, p, xp=np):
    result = []
    for i in range(2):
        centers = []
        for g in sorted(set((int(p.foot_gids[i]), int(p.heel_gids[i])))):
            rotation = data.geom_xmat[g].reshape(3, 3)
            centers.append(data.geom_xpos[g] - rotation[:, 2] * p.m.geom_size[g, 2])
        result.append(xp.stack(centers).mean(0))
    return xp.stack(result)


def contacts(data, p, c, xp=np):
    impl = getattr(data, "_impl", data)
    con = impl.contact
    # Elliptic contact cone: the first constraint row is the normal force.
    force_vector = xp.concatenate([impl.efc_force, xp.zeros(1)])
    fn = xp.where((con.efc_address >= 0) & (con.dist <= 0),
                  xp.maximum(force_vector[xp.maximum(con.efc_address, 0)], 0.), 0.)
    floor = (con.geom == p.floor_gid).any(1)
    toes, heels, grounded, flat, slip, heights, point_masks, forces = [], [], [], [], [], [], [], []
    for i in range(2):
        first, second = int(p.foot_gids[i]), int(p.heel_gids[i])
        first_mask = floor & (con.geom == first).any(1)
        second_mask = floor & (con.geom == second).any(1)
        mask = first_mask | second_mask
        total = xp.where(mask, fn, 0.).sum()
        if c.contact_regions_per_foot == 1:
            axis = data.geom_xmat[first].reshape(3, 3)[:, 0]
            forward = ((con.pos - data.geom_xpos[first]) * axis).sum(1) >= 0
            toe = xp.where(first_mask & forward, fn, 0.).sum() >= c.standing_contact_force_n
            heel = xp.where(first_mask & ~forward, fn, 0.).sum() >= c.standing_contact_force_n
        else:
            toe = xp.where(first_mask, fn, 0.).sum() >= c.standing_contact_force_n
            heel = xp.where(second_mask, fn, 0.).sum() >= c.standing_contact_force_n
        down = total >= c.standing_contact_force_n
        active = mask & down & (fn > 1e-5)
        bid = int(p.foot_bids[i])
        cv = data.cvel[bid]
        origin = data.subtree_com[int(p.m.body_rootid[bid])]
        velocity = cv[3:] + xp.cross(cv[:3], con.pos - origin)
        weight = xp.where(active, fn, 0.)
        slip.append((weight * (velocity[:, :2]**2).sum(1)).sum() / xp.maximum(weight.sum(), 1e-8))
        gaps = []
        for g in sorted(set((first, second))):
            r = data.geom_xmat[g].reshape(3, 3)
            gaps.append(data.geom_xpos[g, 2] - (xp.abs(r[2]) * xp.asarray(p.m.geom_size[g])).sum())
        heights.append(xp.min(xp.stack(gaps)))
        toes.append(toe); heels.append(heel); grounded.append(down); flat.append(toe & heel)
        point_masks.append(active); forces.append(total)
    active = xp.stack(point_masks).any(0)
    any_sole = xp.zeros_like(floor)
    for g in set(map(int, np.r_[p.foot_gids, p.heel_gids])):
        any_sole = any_sole | (con.geom == g).any(1)
    violation = (floor & ~any_sole & (fn >= c.standing_contact_force_n)).any()
    violation = violation | (floor & any_sole & (con.dist < -.01)).any()
    return dict(toe=xp.stack(toes), heel=xp.stack(heels), grounded=xp.stack(grounded),
                flat=xp.stack(flat), slip2=xp.stack(slip), height=xp.stack(heights),
                points=con.pos[:, :2], active_points=active, normal_force=xp.stack(forces),
                floor_violation=violation)


def support_distance(point, points, active, xp=np):
    """Distance to convex hull of actual loaded points, including degenerate hulls."""
    if points.shape[0] == 0:
        return xp.asarray(1000.)
    delta = points[None, :, :] - points[:, None, :]
    length2 = (delta**2).sum(-1)
    # Oriented hull edges have every active point on their left (or collinear).
    offset = points[None, None, :, :] - points[:, None, None, :]
    cross = delta[:, :, None, 0] * offset[..., 1] - delta[:, :, None, 1] * offset[..., 0]
    edges = (active[:, None] & active[None, :] & (length2 > 1e-12)
             & ((cross >= -1e-8) | ~active[None, None, :]).all(-1))
    query = point - points
    side = delta[..., 0] * query[:, None, 1] - delta[..., 1] * query[:, None, 0]
    inside = edges.any() & ((side >= -1e-8) | ~edges).all()
    fraction = xp.clip((query[:, None, :] * delta).sum(-1) / xp.maximum(length2, 1e-12), 0., 1.)
    nearest = points[:, None, :] + fraction[..., None] * delta
    distance2 = xp.where(edges, ((nearest-point)**2).sum(-1), 1e6).min()
    vertex2 = xp.where(active, ((points-point)**2).sum(-1), 1e6).min()
    return xp.where(inside, 0., xp.sqrt(xp.minimum(distance2, vertex2)))
