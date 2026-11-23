"""Turn frozen hold traces into per-trial CSVs, paired-control tables and plots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def longest_streak(mask,dt):
    longest=now=0.
    for value in mask:
        now=now+dt if value else 0.
        longest=max(longest,now)
    return longest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=ROOT/'report/data/quiet01/hold-diagnostics')
    a=ap.parse_args();rows=[];out={};traces={}
    for model in ('v1','v2','xshape'):
        folder=a.data/'results'/model
        summary=json.loads((folder/'summary.json').read_text())
        dt=summary['dt'];names=summary['blocker_names'];out[model]={}
        for case,verdict in summary['cases'].items():
            with np.load(folder/f'{case}.npz',allow_pickle=False) as z:
                trace={key:z[key].copy() for key in z.files}
            if case=='reference_noise':traces[model]=(summary,trace)
            records=[]
            for trial in range(16):
                valid=trace['alive'][:,trial].astype(bool)
                index=np.flatnonzero(valid);last=index[-1]
                quiet=trace['quiet_s'][:,trial]
                breaks=valid&(np.r_[0.,quiet[:-1]]>0)&(quiet==0)
                flags=trace['blockers'][:,trial]
                reasons={name:int((breaks&flags[:,k]).sum()) for k,name in enumerate(names)}
                quiet_kinematic=valid&~flags[:,5:].any(1)&~trace['fallen'][:,trial]&~trace['task_bad'][:,trial]
                loaded=trace['loaded'][:,trial].astype(bool)
                rec=dict(model=model,case=case,trial=trial,passed=verdict['passed'][trial],
                    duration_s=float((last+1)*dt),physical_fall=verdict['physical_fall'][trial],
                    task_violation=verdict['task_violation'][trial],timeout=bool(trace['timeout'][last,trial]),
                    max_quiet_streak_s=float(quiet[valid].max()),final_quiet_streak_s=float(quiet[last]),
                    first_success_hold_start_s=float(trace['settling_s'][last,trial]),
                    final_streak_start_s=float(trace['quiet_start'][last,trial]),
                    final_foot_error_mm=float(trace['foot_error'][last,trial]*1000),
                    peak_foot_error_mm=float(trace['foot_error'][valid,trial].max()*1000),
                    final_joint_rms_deg=float(np.rad2deg(trace['joint_rms'][last,trial])),
                    final_tilt_error_deg=float(np.rad2deg(trace['tilt_error'][last,trial])),
                    hold_breaks=int(breaks.sum()),hold_breaks_after_5s=int((breaks&(np.arange(len(quiet))*dt>=5.)).sum()),
                    loaded_tick_fraction=float(loaded[valid].mean()),
                    max_kinematically_quiet_streak_s=float(longest_streak(quiet_kinematic,dt)),
                    final_pose_ok=bool(~flags[last,:5].any()),final_kinematic_ok=bool(~flags[last,5:].any()),
                    final_deadline_ok=bool(trace['quiet_start'][last,trial]<=6.+dt),
                    final_loaded=bool(loaded[last]),
                    any_observed_five_second_streak=bool(quiet[valid].max()>=5.-1e-4))
                rec.update({'break_'+name:count for name,count in reasons.items()})
                rec.update({'violation_fraction_'+name:float(flags[valid,k].mean()) for k,name in enumerate(names)})
                records.append(rec);rows.append(rec)
            out[model][case]=dict(successes=verdict['successes'],trials=16,
                failed_trials=[r['trial'] for r in records if not r['passed']],
                trials_with_observed_5s=sum(r['any_observed_five_second_streak'] for r in records),
                physical_falls=sum(r['physical_fall'] for r in records),
                task_violations=sum(r['task_violation'] for r in records),
                timeouts=sum(r['timeout'] for r in records),
                break_counts={name:sum(r['break_'+name] for r in records) for name in names},
                failed_records=[r for r in records if not r['passed']])
            if case=='reference_noise':
                out[model][case]['same_verdicts_as_archive']=verdict['same_verdicts_as_archive']
        print(model,{k:v['successes'] for k,v in out[model].items()})
    a.data.mkdir(parents=True,exist_ok=True)
    with (a.data/'per-trial.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.data/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for col,model in enumerate(('v2','xshape')):
        summary,trace=traces[model]
        failed=out[model]['reference_noise']['failed_trials']
        for i in failed:
            alive=trace['alive'][:,i].astype(bool)
            t=(np.arange(len(alive))+1)*summary['dt']
            axes[0,col].plot(t[alive],trace['foot_error'][alive,i]*1000,lw=1,label=f'Trial {i}')
            axes[1,col].plot(t[alive],trace['quiet_s'][alive,i],lw=1)
        axes[0,col].axhline(20,color='black',ls='--',label='20 mm limit')
        axes[1,col].axhline(5,color='black',ls='--',label='5 s hold')
        axes[0,col].set(title=f'{model.upper()}: failed reference-start trials',ylabel='Sole position error (mm)')
        axes[1,col].set(ylabel='Continuous quiet streak (s)')
        axes[0,col].legend(ncol=2,fontsize=8)
        names=summary['blocker_names']
        counts={name:sum(r['break_'+name] for r in out[model]['reference_noise']['failed_records']) for name in names}
        selected=[name for name in names if counts[name]>0]
        axes[2,col].barh(selected,[counts[n] for n in selected],color='#447cab')
        axes[2,col].set(xlabel='Hold interruptions (multiple simultaneous causes may count)',ylabel='Failed condition')
        for ax in axes[:2,col]:ax.set(xlabel='Time (s)',xlim=(0,12));ax.grid(alpha=.2)
    fig.suptitle('QUIET-01: detailed baseline hold failures (frozen parents, seed 1234, full noise)')
    folder=ROOT/'report/figures';folder.mkdir(exist_ok=True,parents=True)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'quiet01_hold_failures.{ext}',dpi=180)
    # Replace only this generated section; preserve the archive/code audit above it.
    report=ROOT/'report/quiet01-hold-diagnosis.md'
    marker='<!-- detailed-diagnostic-results -->'
    existing=report.read_text(encoding='utf-8').split(marker)[0].rstrip()
    text=[marker,'','## Detailed matched-seed diagnostic replay','',
        'Frozen-parent replay array **3263170**, seed1234, original96-member layout.',
        'There are no training updates. Controls change the initial pose and sensor',
        'noise only; model dynamics, policy weights and reference target are fixed.','',
        '| Model | Reference + noise | Reference, no noise | Keyframe + noise | Keyframe, no noise |',
        '|---|---:|---:|---:|---:|']
    for model,cases in out.items():
        values=[str(cases[k]['successes'])+'/16' for k in
                ('reference_noise','reference_clean','keyframe_noise','keyframe_clean')]
        text.append('| '+model.upper()+' | '+' | '.join(values)+' |')
    text+=['','Replay parity with archived baseline outcomes:']
    for model,cases in out.items():
        text.append('- '+model.upper()+': '+str(cases['reference_noise']['same_verdicts_as_archive'])+'.')
    text+=['','Exact per-condition hold interruptions for failed reference trials:',
           '(multiple conditions can fail on the same tick).']
    for model,cases in out.items():
        failures=cases['reference_noise']['failed_records']
        counts={name:sum(r['break_'+name] for r in failures) for name in traces[model][0]['blocker_names']}
        description=', '.join(k+'='+str(v) for k,v in counts.items() if v) or 'no interruptions counted in failed trials'
        text.append('- '+model.upper()+': '+description+'.')
    text+=['','![Hold traces and interruption causes](figures/quiet01_hold_failures.png)','',
        '[Per-trial numeric measurements](data/quiet01/hold-diagnostics/per-trial.csv),',
        '[complete analysis](data/quiet01/hold-diagnostics/analysis.json).',
        'No claim about improved policy weights follows from these controls; they are',
        'diagnostic rollouts with the original parents. This section is generated from',
        'the saved numerical traces. This diagnostic job does not train; see the',
        '[v2 correction and resubmission](quiet01-v2-pose-tracking.md) for subsequent work.','']
    report.write_text(existing+'\n\n'+'\n'.join(text),encoding='utf-8')


if __name__=='__main__':main()
