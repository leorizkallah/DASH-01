"""Small offline learning/settling report; never starts training or changes selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    path = run / 'stop_evaluations.jsonl'
    if not path.exists():
        return
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows:
        return
    lines=[f'# STOP-06 settling study: {run.name}','',
           'Ground-truth readiness: horizontal speed <0.15 m/s, angular speed <0.5 rad/s,',
           'tilt <15 degrees for 0.3 s after the ramp. This diagnostic is not five-second hold success.',
           'Times are relative to STOP. Missing readiness is a failure to observe the event, not a zero time.',
           '', '| Steps | Valid approaches | Stops | Standing | Walking retained | HOLD entry median (s) | Readiness events |',
           '|---:|---:|---:|---:|:---:|---:|---:|']
    trials=[]
    for r in rows:
        entries=[e['controller_hold_entry_s'] for e in r['episodes'] if e.get('controller_hold_entry_s') is not None]
        readiness=[e['physical_readiness_at_s'] for e in r['episodes'] if e.get('physical_readiness_at_s') is not None]
        median=f'{np.median(entries):.2f}' if entries else 'unobserved'
        lines.append(f"| {r['step']} | {r['approach_fraction']:.1%} | {r['success_fraction']:.1%} | {r['standing']['success_fraction']:.1%} | {r['walking_retained']} | {median} | {len(readiness)}/{r['n']} |")
        for i,e in enumerate(r['episodes']):
            trials.append(dict(step=r['step'],trial=i,**{k:e.get(k) for k in (
                'command_mps','approach_ok','controller_hold_entry_s','physical_readiness_at_s',
                'hold_gate_timed_out','hold_start_s','stop_success','termination_s','termination_phase',
                'distance_to_settle_or_failure_m')}))
    with (run/'settling_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=trials[0]);w.writeheader();w.writerows(trials)
    lines+=['','Keeper selection still requires standing/walking retention. Guard patience only',
            'changes when training stops; it does not relax success or selection thresholds.',
            'Equal maximum budgets do not guarantee equal realized budgets if a guard stops an arm.',
            'Single training seed: this comparison is exploratory, not a repeatability claim.','']
    (run/'SETTLING_REPORT.md').write_text('\n'.join(lines))
    comparison=['# STOP-06 comparison snapshot','',
        'Same source keeper, training seed, reward, learning rate and maximum budget; only the HOLD gate differs.',
        'Metrics below are selection-seed results. Consult each motion assessment for held-out seeds.',
        '', '| Arm | Latest steps | Latest stops | Selected steps | Selected stops |',
        '|---|---:|---:|---:|---:|']
    for other in sorted(run.parent.glob('dash_stop06_*_s0')):
        ep=other/'stop_evaluations.jsonl';bp=other/'best_stop_eval.json'
        if not ep.exists() or not bp.exists():continue
        last=json.loads(ep.read_text().splitlines()[-1]);best=json.loads(bp.read_text())
        comparison.append(f"| {other.name} | {last['step']} | {last['success_fraction']:.1%} | {best['step']} | {best['success_fraction']:.1%} |")
    (run.parent/'STOP06_COMPARISON.md').write_text('\n'.join(comparison)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for name,label in [('success_fraction','Moving stops'),('standing','Standing holds'),('walking_run_only','Walking survival')]:
        vals=[r[name]['success_fraction'] if name=='standing' else r[name]['alive_frac'] if name=='walking_run_only' else r[name] for r in rows]
        ax.plot([r['step'] for r in rows],vals,'o-',label=label)
    ax.set(xlabel='New PPO steps',ylabel='Fraction',ylim=(-.03,1.03),title=run.name)
    ax.legend();fig.tight_layout();fig.savefig(run/'learning.png',dpi=160);plt.close(fig)
    trace=run/'replay/motion.npz'
    selection=run/'replay/selection.json'
    if trace.exists() and selection.exists():
        sel=json.loads(selection.read_text())
        if 'trial' not in sel:return
        i=sel['trial']
        with np.load(trace,allow_pickle=False) as z:
            meta=json.loads(str(z['meta'].item()));dt=meta['control_dt'];stop=meta['stop_ticks'][i]
            done=np.flatnonzero(z['done'][:,i]);end=int(done[0])+1 if len(done) else len(z['done'])
            t=(np.arange(end)-stop)*dt
            fig,axes=plt.subplots(4,1,figsize=(9,9),sharex=True)
            for key,label in [('speed','Actual horizontal speed'),('reference','Forward speed reference'),('estimated_speed_xy','Estimated horizontal speed')]:
                axes[0].plot(t,z[key][:end,i],label=label)
            axes[0].set_ylabel('m/s');axes[0].legend(fontsize=8)
            axes[1].plot(t,z['gyro'][:end,i]);axes[1].axhline(.5,ls=':',color='gray');axes[1].set_ylabel('Rotation (rad/s)')
            axes[2].plot(t,np.rad2deg(np.arccos(np.clip(z['upright'][:end,i],-1,1))));axes[2].axhline(15,ls=':',color='gray');axes[2].set_ylabel('Tilt (degrees)')
            axes[3].step(t,z['phase'][:end,i],where='post',label='Phase: WALK=0, BRAKE=1, HOLD=2')
            axes[3].step(t,z['supported'][:end,i],where='post',alpha=.5,label='Both feet supported')
            axes[3].legend(fontsize=8);axes[3].set_xlabel('Seconds after STOP')
            for ax in axes:ax.axvline(0,color='gray',ls='--');ax.grid(alpha=.2)
            fig.suptitle(f"{run.name}, replay trial {i}\n{sel['criterion']}",fontsize=10)
            fig.tight_layout();fig.savefig(run/'replay/settling.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
