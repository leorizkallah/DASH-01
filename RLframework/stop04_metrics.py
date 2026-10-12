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
    return (not current['walking_retained'] or
            current['success_fraction'] < baseline['success_fraction']-.10)


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
