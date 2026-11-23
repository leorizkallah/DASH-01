"""Summarize exploratory push trials; distinguish falling, recovery and final hold."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quiet_push_protocol import FORCES,DIRECTIONS,PUSH_TICK,PULSE_TICKS,DT

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'report/data/quiet_push01'


def analyze(model):
    folder=BASE/model
    with np.load(folder/'trace.npz',allow_pickle=False) as z:
        meta=json.loads(str(z['meta'].item()));data={k:z[k].copy() for k in z.files if k!='meta'}
    rows=[]
    for c in meta['cases']:
        i=c['index'];valid=data['alive'][:,i].astype(bool);idx=np.flatnonzero(valid);last=idx[-1]
        fell=bool(data['body_fall'][valid,i].any());invalid=bool(data['invalid'][valid,i].any());contact=bool(data['contact_bad'][valid,i].any())
        survived=bool(last==1999 and not(fell or invalid or contact))
        start=min(PUSH_TICK,len(valid)-1);pre=min(PUSH_TICK-1,last)
        after=valid&(np.arange(len(valid))>=PUSH_TICK)
        mask=after if after.any() else valid
        delivered=float(np.linalg.norm(data['impulse'][last,i]));nominal=c['force_n']*.1
        row=dict(model=model,**c,duration_s=(last+1)*DT,survived=survived,physical_fall=fell,
            invalid_physics=invalid,contact_failure=contact,push_reached=bool(last>=PUSH_TICK),
            prepush_ready=bool(data['pre_quiet_s'][pre,i]>=.5 and data['pose_ok'][pre,i] and (data['foot_force'][pre,i]>=2).all()),
            held_once=bool(data['hold_seen'][last,i]),recovered=bool(survived and data['hold_seen'][last,i]),
            retained=bool(survived and data['retained'][last,i]),
            recovery_streak_start_after_pulse_s=float(data['hold_start'][last,i]),
            recovery_hold_completed_after_pulse_s=float(data['hold_complete'][last,i]),
            delivered_impulse_ns=delivered,nominal_impulse_ns=nominal,
            nominal_delta_v_mps=nominal/meta['mass_kg'],
            peak_tilt_deg=float(np.rad2deg(data['tilt'][mask,i].max())),
            peak_foot_drift_m=float(data['foot_drift'][mask,i].max()),
            peak_clearance_mm=float(1000*data['clearance'][mask,i].max()),
            peak_loaded_slip_mps=float(data['slip'][mask,i].max()),
            integrated_loaded_slip_mm_left=float(data['slip'][mask,i,0].sum()*DT*1000),
            integrated_loaded_slip_mm_right=float(data['slip'][mask,i,1].sum()*DT*1000),
            peak_torque_util=float(data['torque_util'][mask,i].max()),
            liftoff_events=int(data['liftoffs'][last,i]),landing_events=int(data['landings'][last,i]),
            final_sole_relative_error_mm=float(1000*data['foot_error'][last,i]),
            final_quiet_s=float(data['quiet_s'][last,i]))
        if c['force_n']>0 and last>=PUSH_TICK+PULSE_TICKS:
            assert abs(delivered-nominal)<1e-4*max(nominal,1.),(i,delivered,nominal)
        rows.append(row)
    controls={r['repeat']:r for r in rows if r['force_n']==0.}
    for r in rows:
        control=controls[r['repeat']]
        r['paired_control_survived']=control['survived']
        r['paired_control_retained']=control['retained']
        r['lost_survival_vs_control']=bool(control['survived'] and not r['survived'])
        r['lost_retention_vs_control']=bool(control['retained'] and not r['retained'])
    # Paired conditions must have identical states before the force, within float32 noise.
    pre_q=data['qpos_video'][119]  # state at6.00s, immediately before pulse begins
    spreads={}
    for repeat in (0,1):
        ids=[r['index'] for r in rows if r['repeat']==repeat]
        ready=[i for i in ids if data['alive'][599,i]]
        spreads[str(repeat)]=float(np.max(np.abs(pre_q[ready]-pre_q[ready[0]]))) if ready else None
        if ready:assert spreads[str(repeat)]<1e-4,spreads
    summary=dict(model=model,checkpoint_step=meta['checkpoint_step'],mass_kg=meta['mass_kg'],
        no_learning=meta['learning_updates']==0 and meta['weights_unchanged'],paired_prepush_max_qpos_difference=spreads,
        controls=[r for r in rows if r['force_n']==0],cases=rows,
        counts_by_direction={})
    for direction,_ in DIRECTIONS:
        entries=[]
        for force in FORCES:
            group=[r for r in rows if r['force_n']==force and (force==0 or r['direction']==direction)]
            entries.append(dict(force_n=force,trials=len(group),ready=sum(r['prepush_ready'] for r in group),
                survived=sum(r['survived'] for r in group),recovered=sum(r['recovered'] for r in group),
                retained=sum(r['retained'] for r in group),falls=sum(r['physical_fall'] for r in group)))
        summary['counts_by_direction'][direction]=entries
    with (folder/'trials.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['v1','v2','xshape']);a=ap.parse_args()
    summaries=[analyze(m) for m in a.models if (BASE/m/'trace.npz').exists()]
    if not summaries:return
    fig,axes=plt.subplots(len(summaries),3,figsize=(15,3.7*len(summaries)),squeeze=False,layout='constrained')
    for row,s in enumerate(summaries):
        for col,(key,label) in enumerate([('survived','Survived20s (no failure termination)'),('recovered','Regained5s quiet hold + survived'),('retained','Quiet during final5s')]):
            grid=np.array([[entry[key]/entry['trials'] for entry in s['counts_by_direction'][d]] for d,_ in DIRECTIONS])
            ax=axes[row,col];ax.imshow(grid,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
            for j in range(4):
                for k in range(len(FORCES)):ax.text(k,j,f'{round(grid[j,k]*2)}/2',ha='center',va='center',fontsize=9)
            ax.set(title=s['model'].upper()+': '+label,xlabel='Force(N),100ms pulse at6s',xticks=range(len(FORCES)),xticklabels=[f'{f:g}' for f in FORCES],yticks=range(4),yticklabels=[d for d,_ in DIRECTIONS])
    fig.suptitle('Frozen quiet-pose policies under pushes: two paired noise realizations per condition')
    for ext in ('png','pdf','svg'):fig.savefig(ROOT/f'report/figures/quiet_push01_outcomes.{ext}',dpi=180)
    plt.close(fig)
    response,axs=plt.subplots(len(summaries),2,figsize=(12,3.5*len(summaries)),squeeze=False,layout='constrained')
    for j,s in enumerate(summaries):
        with np.load(BASE/s['model']/'trace.npz',allow_pickle=False) as z:
            t=(np.arange(2000)+1)*DT
            for col,direction in enumerate(('forward','left')):
                ax=axs[j,col]
                for force in (0,5,20,60):
                    i=next(r['index'] for r in s['cases'] if r['force_n']==force and r['repeat']==0 and (force==0 or r['direction']==direction))
                    valid=z['alive'][:,i]&np.isfinite(z['tilt'][:,i])
                    ax.plot(t[valid],np.rad2deg(z['tilt'][valid,i]),label=f'{force} N')
                ax.axvspan(6,6.1,color='red',alpha=.18,label='100ms pulse')
                ax.axhline(45,color='black',ls='--',lw=.8,label='Tilt fall criterion')
                ax.set(title=s['model'].upper()+' '+direction,xlabel='Time(s)',ylabel='Torso tilt(deg)',xlim=(0,20))
                ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.2)
    response.suptitle('Captured torso response: repeat0, fixed best policy, no reset splicing')
    for ext in ('png','pdf','svg'):response.savefig(ROOT/f'report/figures/quiet_push01_tilt_response.{ext}',dpi=180)
    plt.close(response)
    (BASE/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    report=['# Push-only simulations of the best quiet-pose policies','',
        'No training: the saved best checkpoint for each model is fixed. Independent20s',
        'trials start from its quiet reference, with full sensor noise, no physical DR.',
        'One100ms horizontal pulse acts at the torso COM at6s. Forces0/2/5/10/15/20/30/40/60N;',
        'four body-horizontal directions, two paired noise realizations,66trials/model.',
        'Each force/direction uses the same pre-push noise stream within a repeat.','',
        'Foot lifts, contact changes and drift do not terminate this diagnostic. Physical',
        'fall(45deg tilt or low torso), invalid numerics or serious floor-contact/penetration',
        'failure terminate. Post-push quiet must last five continuous seconds with sufficient',
        'loaded contact; posture tolerances stay identical to QUIET-01v2. No fixed return deadline.',
        'Survival, a regained hold and a final quiet window are separate results.','',
        '![Push outcomes](figures/quiet_push01_outcomes.png)','',
        '![Torso response over time](figures/quiet_push01_tilt_response.png)','',
        'Cells show observed counts out of two, not a statistically established safety limit.',
        'Zero-force controls are reused in the four direction rows, not four independent sets.',
        'Check pre-push readiness and controls before attributing a failure to the force.',
        'A contact landing event alone does not establish a useful recovery step.','',
        '| Model | Selected checkpoint(M additional) | Zero-force survival | Zero-force final hold | Weights unchanged |',
        '|---|---:|---:|---:|---|']
    for s in summaries:
        report.append(f"| {s['model'].upper()} | {s['checkpoint_step']/1e6:.2f} | {sum(r['survived'] for r in s['controls'])}/2 | {sum(r['retained'] for r in s['controls'])}/2 | {s['no_learning']} |")
    report+=['','A zero-force trial can already miss the strict posture hold. CSV columns',
        '`paired_control_retained` and `lost_retention_vs_control` distinguish that',
        'baseline behavior from a new loss under a push. These are paired observations,',
        'not significance estimates. The same applies to upright survival.','',
        'Observed first survival misses on this coarse ladder (counts include numerical/',
        'contact failures separately recorded in the CSV):','',
        '| Model | Forward | Backward | Left | Right |', '|---|---|---|---|---|']
    for s in summaries:
        cells=[]
        for direction,_ in DIRECTIONS:
            entries=s['counts_by_direction'][direction]
            if entries[0]['survived']<2:cells.append('Control not2/2');continue
            last=0.;label='Both survive through60N'
            for entry in entries[1:]:
                if entry['survived']<2:
                    label=f'{last:g}N both survive; first miss{entry["force_n"]:g}N';break
                last=entry['force_n']
            cells.append(label)
        report.append('| '+s['model'].upper()+' | '+' | '.join(cells)+' |')
    report+=['','## Behavioral interpretation','',
        'Upright survival and successful posture recovery are different capabilities.',
        'V2 is comparatively consistent at returning to the reference under moderate',
        'pushes. X-shape survives large lateral pushes, but at60N neither lateral',
        'direction regains the quiet reference hold in either repeat. The same happens',
        'for V2 with60N rightward pushes. This is survival, not full recovery.','',
        'These surviving60N lateral cases have only about0.0–0.1mm whole-foot clearance',
        '(minimum sole gap) while sole centers move roughly74–109mm. Loaded contact',
        'slip peaks about0.46–0.81m/s in the affected foot. Grounded slipping/shifting',
        'is therefore present; contact landing counts do not demonstrate a clean',
        'recovery step. This test permits such motion so the behavior remains visible.','',
        'The45deg tilt criterion ends a trial before literal ground impact; interpret',
        'the videos as reaching the stated fall criterion, not a measurement of',
        'physical ground contact for every fall. No hardware or general safety limit',
        'is inferred from two noise realizations and a single pulse duration.','']
    report+=['','[Video gallery](quiet-push01-videos.html). Raw per-trialCSV andJSON, measured delivered',
        'impulse, torso tilt, foot displacement, clearance, contact events and torque utilization',
        'are under`report/data/quiet_push01/{model}/`. CPU-MJX backend, float32; no policy updates.','']
    (ROOT/'report/quiet-push01-results.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({s['model']:{d:[r['retained'] for r in entries] for d,entries in s['counts_by_direction'].items()} for s in summaries}),flush=True)


if __name__=='__main__':main()
