"""Pure five-second posture-hold verdict and held-out reset curriculum."""
import numpy as np


def hold(previous_s,loaded_s,quiet,loaded,t,dt,cfg,xp=np):
    streak=xp.where(quiet,previous_s+dt,0.)
    loading=xp.where(quiet,loaded_s+dt*loaded,0.)
    ok=(streak>=cfg.quiet_pose_hold_s)&(loading>=.95*streak)&((t-streak)<=cfg.quiet_pose_max_start_s+dt)
    return streak,loading,ok


def accept(completed,streak,loading,cfg,xp=np):
    """A timely completed hold stays completed. Retention has its own final window.

    The caller must additionally require full duration and no physical/task failure.
    Keeping a fixed-duration training episode avoids paying extra return for
    deliberately postponing an early success termination.
    """
    result=xp.asarray(completed,dtype=bool)
    if cfg.quiet_pose_retention_eval:
        result=result&(streak>=cfg.quiet_pose_hold_s)&(loading>=.95*streak)
    return result


def validate_physics(evaluation,min_no_fall_rate=.75):
    """Preflight checks physical validity, not mastery of a new reward objective."""
    rates=list(evaluation['rates'].values())
    if not np.isfinite(rates+[evaluation['score']]).all():
        raise AssertionError('Non-finite quiet-pose evaluation')
    assert all(0.<=r<=1. for r in rates), 'Invalid evaluation rates'
    n=int(evaluation['trials'])//6
    falls=evaluation['per_trial']['physical_fall'][:n]
    durations=evaluation['per_trial']['duration_s'][:n]
    assert n>=16 and len(falls)==n, 'Incomplete reference preflight'
    assert np.isfinite(durations).all() and min(durations)>=1., 'Invalid/very early reference termination'
    no_fall=1.-sum(falls)/n
    assert no_fall>=min_no_fall_rate, f'Reference physical stability failed: {no_fall}'
    return dict(reference_no_fall_rate=no_fall,reference_task_rate=evaluation['quiet_rate'],
                note='Task performance recorded; no minimum mastery of the new pose objective required.')


def promote(cur,probe):
    level=int(cur.get('pose_level',0))
    rates=probe['rates']
    passed=(rates['reference']>=.95 and rates[str(level)]>=.90
            and all(rates[str(k)]>=.80 for k in range(level)))
    cur['pose_gate_streak']=cur.get('pose_gate_streak',0)+1 if passed else 0
    if cur['pose_gate_streak']>=2 and level<4:
        cur['pose_level']=level+1
        cur['pose_gate_streak']=0
        return f'initial-pose level {level} -> {level+1}; two independent passing probes'
    return None
