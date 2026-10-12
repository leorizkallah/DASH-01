"""STOP-01 reward shape and evaluation metrics; numpy-only, also usable with jax.numpy."""
import numpy as np


def brake_quality(speed, upright, tall, xp=np):
    """Broad, monotone pull toward rest without a large negative speed bill."""
    return xp.clip((upright - 0.5) / 0.5, 0.0, 1.0) * tall / (1.0 + speed / 0.5)


def hold_quality(speed, gyro, joint_rms, upright, supported, tall,
                 speed_width=0.15, gyro_width=0.5, joint_width=1.0, xp=np):
    """Bounded [0,1] income: unsupported or fallen robots cannot earn a hold bonus."""
    quiet = xp.exp(-((speed / speed_width) ** 2 + (gyro / gyro_width) ** 2
                     + (joint_rms / joint_width) ** 2))
    posture = xp.clip((upright - 0.9) / 0.1, 0.0, 1.0)
    return quiet * posture * supported * tall


def protocol(n, dt=0.01):
    """Fixed speeds and phase-varied STOP times, independent of training commands."""
    speeds = np.resize(np.array([0.5, 1.0, 1.5, 2.0]), n)
    # Each speed sees the same eight timing offsets when n=32.
    start = np.rint((3.0 + (np.arange(n) // 4 % 8) * 0.13) / dt).astype(int)
    end = start + round(8.0 / dt)
    return speeds, start, end, int(end.max() + round(3.0 / dt))


def summarize(trace, speeds, starts, ends, dt):
    """Score first episodes only. Failed approach/early fall never counts as a stop.

    Success requires a moving approach, then 2 s continuously quiet, and a quiet
    final 5 s under STOP. Contact support must occupy >=90% of the final 5 s.
    """
    n = len(speeds)
    rows = []
    for i in range(n):
        a, b = int(starts[i]), int(ends[i])
        v = np.asarray(trace['speed'])[:, i]
        vx = np.asarray(trace['vx'])[:, i]
        ended = np.asarray(trace['done'])[:, i].astype(bool)
        live = ~np.maximum.accumulate(ended)
        quiet = ((v < 0.1) & (np.asarray(trace['gyro'])[:, i] < 0.3)
                 & (np.asarray(trace['joint_rms'])[:, i] < 0.5)
                 & (np.asarray(trace['upright'])[:, i] > np.cos(np.deg2rad(10)))
                 & live)
        approach = slice(max(0, a - round(1 / dt)), a)
        incoming = float(vx[approach].mean())
        approached = bool(live[a - 1] and np.isfinite(incoming)
                          and incoming >= max(0.15, 0.5 * speeds[i]))
        consecutive, settled_at = 0, None
        for k in range(a, b):
            consecutive = consecutive + 1 if quiet[k] else 0
            if consecutive >= round(2 / dt):
                settled_at = k + 1 - consecutive
                break
        tail = slice(b - round(5 / dt), b)
        supported = float((np.asarray(trace['supported'])[tail, i] & live[tail]).mean())
        success = bool(approached and quiet[tail].all() and supported >= 0.9
                       and settled_at is not None)
        xy = np.asarray(trace['xy'])[:, i]
        finish = next((k for k in range(a, b) if not live[k]), b - 1)
        travel_end = settled_at if settled_at is not None else finish
        distance = float(np.linalg.norm(np.diff(xy[a - 1:travel_end + 1], axis=0), axis=1).sum())
        final_speed = float(v[tail].mean()) if live[b - 1] else None
        restart_end = b + round(3 / dt)
        restart_live = live[restart_end - 1]
        restart_speed = float(vx[restart_end - round(1 / dt):restart_end].mean()) if restart_live else None
        rows.append(dict(command_mps=float(speeds[i]), stop_at_s=a * dt,
                         incoming_mps=incoming if np.isfinite(incoming) else None, approach_ok=approached,
                         stop_success=success, survived_stop=bool(live[b - 1]),
                         stop_time_s=(settled_at - a) * dt if success else None,
                         distance_to_settle_or_failure_m=distance if np.isfinite(distance) else None,
                         final_stop_speed_mps=final_speed, support_fraction=supported,
                         restart_speed_mps=restart_speed,
                         restart_ok=bool(success and restart_live
                                         and restart_speed >= 0.5 * speeds[i])))
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    per_speed = []
    for speed in sorted(set(speeds)):
        group = [r for r in rows if r['command_mps'] == speed]
        per_speed.append(dict(command_mps=float(speed), n=len(group),
                              successes=sum(r['stop_success'] for r in group),
                              survivors=sum(r['survived_stop'] for r in group)))
    times = [r['stop_time_s'] for r in rows if r['stop_time_s'] is not None]
    tails = [r['final_stop_speed_mps'] for r in rows if r['final_stop_speed_mps'] is not None]
    return dict(n=n, success_fraction=mean('stop_success'),
                approach_fraction=mean('approach_ok'), survive_fraction=mean('survived_stop'),
                restart_fraction=mean('restart_ok'),
                stop_time_mean_s=float(np.mean(times)) if times else None,
                residual_speed_mean_mps=float(np.mean(tails)) if tails else None,
                per_speed=per_speed, episodes=rows)
