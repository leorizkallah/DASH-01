"""Diagnostic condition ablations, rehearsal loss, and preservation guards."""
import numpy as np
from stop02_metrics import hold_window


def rehearsal_loss(pred, target, xp=np):
    """Balanced batch: first half HOLD, second half RUN. Fixed .12 action scale.

    This is a scaled action MSE, not a Gaussian KL or physical stopping metric.
    Estimator gradients are detached by actor_mean, as in the usual actor path.
    """
    n = pred.shape[0] // 2
    hold = xp.mean((pred[:n]-target[:n])**2)
    walk = xp.mean((pred[n:]-target[n:])**2)
    return (hold + 2.*walk)/(2.*.12**2), hold, walk


def preservation_failed(baseline, current):
    return (not current['walking_retained'] or not current.get('standing_retained', True) or
            current['success_fraction'] < baseline['success_fraction']-.10)


def enforce_preservation(agent, baseline, current):
    """Shared by periodic and final evaluation; retain evidence before stopping."""
    import json
    cfg = getattr(agent, 'cfg', None)
    patience = getattr(cfg, 'stop_guard_patience', 1)
    grace = getattr(cfg, 'stop_guard_grace_steps', 0)
    if patience > 1 or grace > 0:
        # Reconstruct from recorded evaluations, so a process restart cannot erase a streak.
        path = agent.run/'stop_evaluations.jsonl'
        records = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else [current]
        streak = 0
        seen = set()
        for r in reversed(records):
            step = r.get('step', 0)
            if step in seen:
                continue
            seen.add(step)
            if step < grace or not preservation_failed(baseline, r):
                break
            streak += 1
        (agent.run/'guard_monitor.json').write_text(json.dumps(dict(step=agent.step,
            grace_steps=grace,patience=patience,consecutive_failures=streak,
            failed_now=preservation_failed(baseline,current)),indent=2)+'\n')
        if agent.step < grace or streak < patience:
            return
    if preservation_failed(baseline, current):
        agent.save(agent.run / 'guard_rejected.msgpack')
        (agent.run/'guard_stop.json').write_text(json.dumps(dict(step=agent.step,
            reason='walking/standing retention failed or primary success fell >10 percentage points',
            result=current),indent=2)+'\n')
        raise SystemExit('Preservation guard stopped this pilot; rejected checkpoint saved')


def command_duration(stop, uniform, run_bounds, hold_bounds, xp=np):
    lo=xp.where(stop>.5,hold_bounds[0],run_bounds[0])
    hi=xp.where(stop>.5,hold_bounds[1],run_bounds[1])
    return lo+(hi-lo)*uniform


def diagnose(trace, dt=.01, seconds=7.):
    """Describe violations and recompute success with one criterion removed.

    Ablations are measurement counterfactuals on saved trajectories. They are not
    interventions on the controller and cannot prove a physical cause.
    """
    end = round(seconds/dt)
    live = ~np.maximum.accumulate(trace['done'].astype(bool), axis=0)
    conditions = dict(speed=trace['speed'] < .1, angular=trace['gyro'] < .3,
                      joints=trace['joint_rms'] < .5,
                      tilt=trace['upright'] > np.cos(np.deg2rad(10)))
    rows=[]
    for i in range(live.shape[1]):
        survived=bool(live[end-1,i])
        eligible=live[:end,i].copy();eligible[:round(2/dt)]=False
        fractions={k: float((~v[:end,i])[eligible].mean()) if eligible.any() else None
                   for k,v in conditions.items()}
        outcomes={}
        for removed in ['none', *conditions, 'support', 'drift']:
            quiet=live[:,i].copy()
            for k,v in conditions.items():
                if k!=removed: quiet &= v[:,i]
            support=np.ones_like(quiet) if removed=='support' else trace['supported'][:,i]
            xy=np.zeros_like(trace['xy'][:,i]) if removed=='drift' else trace['xy'][:,i]
            longest,first=hold_window(quiet,support,xy,0,end,dt)
            outcomes[removed]=bool(survived and first is not None)
        rows.append(dict(trial=i,survived=survived,success=outcomes['none'],
                         violation_fraction_after_2s=fractions,
                         success_if_criterion_removed=outcomes))
    return dict(n=len(rows),successes=sum(r['success'] for r in rows),
                survivors=sum(r['survived'] for r in rows),episodes=rows,
                successful_if_removed={k:sum(r['success_if_criterion_removed'][k] for r in rows)
                                       for k in rows[0]['success_if_criterion_removed']})
