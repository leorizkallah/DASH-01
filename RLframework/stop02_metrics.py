"""Five-second physical holding checks, independent of reward and JAX."""
import numpy as np


def timeout_allowed(stop_requested, elapsed, ramp_seconds, hold_seconds=7.):
    """A late STOP may extend the episode cap only through its bounded hold window."""
    return (stop_requested == False) | (elapsed >= ramp_seconds + hold_seconds)


def protocol(n, dt, cfg):
    speeds = np.resize(np.asarray(cfg.stop_eval_speeds, dtype=float), n)
    starts = np.rint((3 + (np.arange(n)//4 % 8)*.13) / dt).astype(int)
    maximum = max(cfg.stop_ramp_min_s, 1.5*cfg.v_max/cfg.stop_ramp_decel)
    if cfg.stop_standing_stage:
        speeds, starts, maximum = np.zeros(n), np.zeros(n, int), 0.0
    window = 7. if cfg.stop_standing_stage else cfg.stop_hold_window_s
    ends = starts + int(np.ceil((maximum + window)/dt))
    return speeds, starts, ends, int(ends.max() + round(3/dt))


def hold_window(quiet, support, xy, start, deadline, dt):
    """Longest continuous quiet period and first supported, non-drifting 5 s window."""
    count, longest, first = 0, 0, None
    length = round(5/dt)
    for k in range(start, deadline):
        count = count + 1 if quiet[k] else 0
        longest = max(longest, count)
        if count >= length:
            a = k + 1 - length
            drift = np.linalg.norm(xy[a:k+1] - xy[a], axis=-1).max()
            if first is None and support[a:k+1].mean() >= .9 and drift <= .10:
                first = a
    return longest*dt, first


def summarize(trace, speeds, starts, ends, dt, standing=False, hold_seconds=7.):
    rows = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        live = ~np.maximum.accumulate(trace['done'][:, i].astype(bool))
        quiet = ((trace['speed'][:, i] < .10) & (trace['gyro'][:, i] < .30)
                 & (trace['joint_rms'][:, i] < .50)
                 & (trace['upright'][:, i] > np.cos(np.deg2rad(10))) & live)
        support = trace['supported'][:, i] & live
        ramp = 0.0 if standing else float(trace['ramp_duration'][a, i])
        hold_start = a + int(np.ceil(ramp/dt - 1e-5))
        deadline = min(b, hold_start + round((7. if standing else hold_seconds)/dt))
        if 'hold_active' in trace:
            quiet &= trace['hold_active'][:,i].astype(bool)
        longest, first = hold_window(quiet, support, trace['xy'][:, i], hold_start, deadline, dt)
        incoming = float(trace['vx'][max(0, a-round(1/dt)):a, i].mean()) if a else 0.0
        approached = bool(standing or (live[a-1] and incoming >= max(.15, .5*speeds[i])))
        survived = bool(live[b-1])
        success = bool(approached and survived and first is not None)
        term = np.flatnonzero(trace['done'][:, i])
        end = int(term[0]) if len(term) else len(live)-1
        travel_end = first if first is not None else min(end,b-1)
        path = trace['xy'][max(0,a-1):travel_end+1,i]
        distance = float(np.linalg.norm(np.diff(path,axis=0),axis=-1).sum())
        braking_error = None
        if 'reference' in trace and hold_start > a:
            err = np.abs(trace['vx'][a:hold_start,i]-trace['reference'][a:hold_start,i])
            valid = live[a:hold_start] & np.isfinite(err)
            braking_error = float(err[valid].mean()) if valid.any() else None
        reason = []
        if 'reasons' in trace and len(term):
            reason = [name for name, yes in zip(('low','tip','floor','workspace','nan'), trace['reasons'][end,i]) if yes]
        entries = np.flatnonzero(trace['hold_active'][a:b,i] & live[a:b]) if 'hold_active' in trace else []
        entry_s = float(entries[0]*dt) if len(entries) else None
        readiness = ((trace['speed'][:,i] < .15) & (trace['gyro'][:,i] < .5)
                     & (trace['upright'][:,i] > np.cos(np.deg2rad(15))) & live)
        ready_count, ready_at = 0, None
        for tick in range(hold_start,deadline):
            ready_count = ready_count+1 if readiness[tick] else 0
            if ready_count >= round(.3/dt):
                ready_at = (tick+1-a)*dt
                break
        phase = None
        if len(term):
            phase = ('WALK' if end<a else ('AFTER_HOLD' if standing else 'RESTART') if end>=b
                     else ('HOLD' if trace['hold_active'][end,i] else 'BRAKE') if 'hold_active' in trace
                     else 'BRAKE' if end<hold_start else 'HOLD')
        restart_speed = float(trace['vx'][b+round(2/dt):b+round(3/dt),i].mean()) if live[b+round(3/dt)-1] else None
        rows.append(dict(command_mps=float(speeds[i]), stop_at_s=float(a*dt), approach_ok=approached,
            incoming_mps=incoming if np.isfinite(incoming) else None,
            controller_hold_entry_s=entry_s,
            physical_readiness_at_s=ready_at,
            hold_gate_timed_out=bool(np.any(trace['gate_timeout'][a:b,i] & live[a:b])) if 'gate_timeout' in trace else False,
            stop_success=success, survived_stop=survived, ramp_s=ramp,
            braking_tracking_error_mps=braking_error,
            distance_to_settle_or_failure_m=distance if np.isfinite(distance) else None,
            hold_max_displacement_m=(float(np.linalg.norm(trace['xy'][first:first+round(5/dt),i]
                                    -trace['xy'][first,i],axis=-1).max()) if first is not None else None),
            longest_quiet_hold_s=longest, hold_start_s=None if first is None else (first-a)*dt,
            stop_time_s=(first-a)*dt if success else None,
            final_stop_speed_mps=float(trace['speed'][b-round(5/dt):b,i].mean()) if survived else None,
            support_fraction=float(support[max(hold_start,b-round(5/dt)):b].mean()),
            termination_s=(end+1)*dt if len(term) else None, termination_reasons=reason,
            termination_phase=phase,
            restart_ok=bool(success and not standing and restart_speed is not None and restart_speed >= .5*speeds[i]),
            restart_speed_mps=restart_speed))
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    residual = [r['final_stop_speed_mps'] for r in rows if r['final_stop_speed_mps'] is not None]
    times = [r['stop_time_s'] for r in rows if r['stop_time_s'] is not None]
    return dict(protocol='STOP-02-standing' if standing else 'STOP-02-ramp', n=len(rows), episodes=rows,
        post_ramp_window_s=7. if standing else hold_seconds,
        success_fraction=mean('stop_success'), approach_fraction=mean('approach_ok'), restart_evaluated=not standing,
        survive_fraction=mean('survived_stop'), restart_fraction=mean('restart_ok'),
        longest_quiet_hold_mean_s=mean('longest_quiet_hold_s'),
        stop_time_mean_s=float(np.mean(times)) if times else None,
        residual_speed_mean_mps=float(np.mean(residual)) if residual else None,
        per_speed=[dict(command_mps=float(s), n=int(sum(r['command_mps']==s for r in rows)),
            successes=sum(r['stop_success'] for r in rows if r['command_mps']==s),
            survivors=sum(r['survived_stop'] for r in rows if r['command_mps']==s)) for s in sorted(set(speeds))])
