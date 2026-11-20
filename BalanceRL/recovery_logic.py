"""Pure NumPy/JAX recovery-state logic; no simulator or policy dependencies."""
import numpy as np


def initial(xy, base, loaded, xp=np):
    z = xp.asarray(0., dtype=xp.float32)
    return dict(anchor=xy, candidate=xy, quiet_anchor=base, episode_feet=xy,
                episode_base=base, previous_xy=xy, contact_latched=loaded,
                previous_loaded=loaded, up_s=xp.zeros(2), down_s=xp.zeros(2),
                quiet_s=z, loaded_s=z, air_s=xp.zeros(2), quiet_start=z,
                recovering=xp.asarray(False), has_pushed=xp.asarray(False),
                pulse_end=z, sector=xp.asarray(0, dtype=xp.int32),
                steps=xp.asarray(0, dtype=xp.int32), total_steps=xp.asarray(0, dtype=xp.int32),
                total_path=z, recovery_path=z)


def advance(old, xy, base, normal_force, clearance, kinematic_quiet,
            started, duration, sector, now, dt, cfg, lateral_axis, xp=np,
            pose_ok=True, balance_supported=True):
    started = xp.asarray(started, dtype=bool)
    loaded = normal_force >= cfg.standing_contact_force_n
    up_s = xp.where(normal_force < cfg.recovery_contact_off_n, old['up_s']+dt, 0.)
    down_s = xp.where(loaded, old['down_s']+dt, 0.)
    dwell = cfg.recovery_contact_dwell_s-1e-6
    latched = xp.where(up_s >= dwell, False,
                       xp.where(down_s >= dwell, True, old['contact_latched']))
    liftoff = old['contact_latched'] & ~latched
    landing = ~old['contact_latched'] & latched
    switches = xp.sum(loaded != old['previous_loaded'])
    path = xp.linalg.norm(xy-old['previous_xy'], axis=1)
    foot_speed = path/dt
    air_s = xp.where(clearance > .003, old['air_s']+dt, 0.)
    recovering = old['recovering'] | started
    pulse_end = xp.where(started, now-dt+duration, old['pulse_end'])
    sector = xp.where(started, sector, old['sector'])
    steps = xp.where(started, 0, old['steps']) + xp.sum(landing, dtype=xp.int32)
    quiet_old = xp.where(started, 0., old['quiet_s'])
    pose_mode = cfg.recovery_pose_return
    balance_event = xp.asarray(False)
    if pose_mode:
        was_returning = old['returning'] & ~started
        stable_now = (recovering & (now > pulse_end+1e-6) & kinematic_quiet & loaded.all()
            & (foot_speed < cfg.recovery_foot_speed_mps).all() & (clearance <= .003).all()
            & balance_supported & ~liftoff.any() & ~landing.any())
        balance_stable_s = xp.where(stable_now, xp.where(started, 0., old['balance_stable_s'])+dt, 0.)
        balance_event = (recovering & ~was_returning
            & (balance_stable_s >= cfg.return_balance_dwell_s-1e-5)
            & (now <= pulse_end+cfg.recovery_begin_s+1e-5))
        returning = was_returning | balance_event
        balanced_at = xp.where(balance_event, now, xp.where(started, -1., old['balanced_at']))
    candidate = xp.where((quiet_old <= 0.), xy, old['candidate'])
    quiet_anchor = xp.where(quiet_old <= 0., base, old['quiet_anchor'])
    candidate_drift = xp.linalg.norm(xy-candidate, axis=1)
    quiet_now = (kinematic_quiet & (foot_speed < cfg.recovery_foot_speed_mps).all()
        & (clearance <= .003).all() & (candidate_drift <= .01).all()
        & (xp.linalg.norm(base-quiet_anchor) <= .03)
        & ~(recovering & (now <= pulse_end+1e-6)) & ~liftoff.any() & ~landing.any())
    if pose_mode:
        quiet_now = quiet_now & pose_ok & (~recovering | returning)
    quiet_s = xp.where(quiet_now, quiet_old+dt, 0.)
    loaded_s = xp.where(quiet_now, xp.where(started, 0., old['loaded_s'])+dt*loaded.all(), 0.)
    quiet_start = xp.where(quiet_now & (quiet_old <= 0.), now, old['quiet_start'])
    quiet_ok = (quiet_s >= cfg.recovery_quiet_s-1e-4) & (loaded_s >= .95*quiet_s)
    drift = xp.linalg.norm(xy-old['anchor'], axis=1)
    quiet_violation = ~recovering & ((drift > .020).any() | (air_s > .05).any()
                                    | liftoff.any() | landing.any())
    # A hard workspace envelope prevents cumulative walking across successive pushes.
    workspace_bad = ((xp.linalg.norm(xy-old['episode_feet'], axis=1) > cfg.recovery_episode_radius_m).any()
        | (xp.linalg.norm(base-old['episode_base']) > cfg.recovery_episode_radius_m)
        | (xp.linalg.norm(xy[0]-xy[1]) > cfg.recovery_stance_span_m)
        | (xp.sum((xy[0]-xy[1])*lateral_axis) < .12))
    timely = quiet_start <= pulse_end+cfg.recovery_begin_s+1e-5
    if pose_mode:
        timely = returning & (quiet_start <= balanced_at+cfg.return_begin_s+1e-5)
    success = recovering & quiet_ok & timely & ~workspace_bad & ~quiet_violation
    expired = recovering & (now > pulse_end+cfg.recovery_begin_s+cfg.recovery_quiet_s+.05) & ~success
    if pose_mode:
        expired = recovering & ~success & (
            (~returning & (now > pulse_end+cfg.recovery_begin_s+.05))
            | (returning & (now > balanced_at+cfg.return_begin_s+cfg.recovery_quiet_s+.05)))
    task_bad = quiet_violation | workspace_bad | expired
    new = dict(old, anchor=xp.where(success, candidate, old['anchor']), candidate=candidate,
        quiet_anchor=quiet_anchor, quiet_s=quiet_s, loaded_s=loaded_s, quiet_start=quiet_start,
        previous_xy=xy, contact_latched=latched, previous_loaded=loaded, up_s=up_s,
        down_s=down_s, air_s=air_s, recovering=recovering & ~success & ~task_bad,
        has_pushed=old['has_pushed'] | started, pulse_end=pulse_end, sector=sector,
        steps=steps, total_steps=old['total_steps']+xp.sum(landing, dtype=xp.int32),
        total_path=old['total_path']+path.sum(),
        recovery_path=xp.where(started, 0., old['recovery_path'])+xp.where(recovering, path.sum(), 0.))
    if pose_mode:
        new.update(balance_stable_s=balance_stable_s, returning=returning & ~success & ~task_bad,
                   balanced_at=balanced_at)
    metrics = dict(quiet_ok=quiet_ok, success=success, task_bad=task_bad, expired=expired,
        quiet_violation=quiet_violation, workspace_bad=workspace_bad, recovering=recovering,
        drift=drift, path=path.sum(), foot_speed=foot_speed, liftoff=liftoff,
        landing=landing, switches=switches, steps=steps, sector=sector,
        settling_time=xp.maximum(quiet_start-pulse_end, 0.),
        excess_cost=xp.maximum(steps-cfg.recovery_soft_steps, 0)*landing.sum())
    if pose_mode:
        metrics.update(balance_recovered=balance_event, returning=returning & recovering,
            balance_time=xp.maximum(balanced_at-pulse_end, 0.),
            pose_return_time=xp.maximum(quiet_start-balanced_at, 0.))
    return new, metrics
