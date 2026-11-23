"""Archive-only hold diagnosis; does not infer missing per-tick measurements."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'BalanceRL'))
from quiet_tracking_math import hold
from config import get_config


def main():
    folder=ROOT/'report/data/quiet01'
    evidence=json.loads((folder/'reference-hold-evidence.json').read_text())
    rows=[];summary={}
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for model,data in evidence['models'].items():
        p=data['trials']
        summary[model]=dict(accepted=sum(p['passed']),completed_hold=sum(t>=0 for t in p['settling_s']),
            failed_after_completed_hold=sum(not ok and t>=0 for ok,t in zip(p['passed'],p['settling_s'])),
            never_completed_hold=sum(t<0 for t in p['settling_s']),physical_falls=sum(p['physical_fall']))
        for i in range(16):
            row=dict(model=model,trial=i,accepted=p['passed'][i],completed_hold=p['settling_s'][i]>=0,
                first_hold_start_s=p['settling_s'][i],duration_s=p['duration_s'][i],
                physical_fall=p['physical_fall'][i],task_violation=p['task_violation'][i],
                final_foot_error_mm=p['final_foot_error_m'][i]*1000,
                final_joint_rms_deg=float(np.rad2deg(p['final_joint_rms_rad'][i])),
                final_tilt_error_deg=float(np.rad2deg(p['final_tilt_error_rad'][i])))
            rows.append(row)
    for col,model in enumerate(('v2','xshape')):
        p=evidence['models'][model]['trials'];ax=axes[0,col]
        colors=['#2d906a' if ok else '#c56148' for ok in p['passed']]
        ax.bar(np.arange(16),np.array(p['final_foot_error_m'])*1000,color=colors)
        ax.axhline(20,color='black',ls='--',label='20 mm tolerance')
        ax.set(title=model.upper()+' terminal sole-relative error',xlabel='Reference trial (zero-based)',
               ylabel='Error relative to torso (mm)',xticks=np.arange(16))
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    models=list(summary);x=np.arange(len(models));ax=axes[1,0]
    ax.bar(x-.18,[summary[m]['completed_hold'] for m in models],.36,label='Completed a valid 5 s hold',color='#447cab')
    ax.bar(x+.18,[summary[m]['accepted'] for m in models],.36,label='Accepted at end of episode',color='#d78342')
    ax.set(xticks=x,xticklabels=[m.upper() for m in models],ylim=(0,18),ylabel='Trials out of 16')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    c=get_config('quiet_v2');s=l=0.;seen=False;time=[];streak=[];verdict=[]
    for k in range(1200):
        t=(k+1)*.01
        # One 10 ms interruption at 6.50 s, then a completely valid quiet state.
        quiet=k!=649
        s,l,ok=hold(s,l,quiet,True,t,.01,c)
        seen=seen or bool(ok);time.append(t);streak.append(float(s));verdict.append(bool(ok))
    audit=dict(scenario='Single 10ms quiet interruption at6.50s; otherwise quiet and loaded throughout12s',
        ever_accepted=seen,final_quiet_s=float(s),final_streak_start_s=12.-float(s),final_accepted=bool(ok))
    assert seen and not ok and s>5.
    ax=axes[1,1];ax.plot(time,streak,color='#447cab',label='Continuous quiet streak')
    ax.axhline(5,color='black',ls='--',label='Five seconds');ax.axvline(6.5,color='#c56148',ls=':')
    ax.set(title='Code-only timing example: 5.5 s final hold still rejected',xlabel='Time (s)',ylabel='Streak (s)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('QUIET-01 acceptance audit: green = accepted, red = failed; no reference-trial falls')
    target=ROOT/'report/figures'
    for ext in ('png','pdf','svg'):fig.savefig(target/f'quiet01_hold_acceptance_audit.{ext}',dpi=180)
    (folder/'hold-rule-audit.json').write_text(json.dumps(dict(archived_summary=summary,synthetic_timing_test=audit),indent=2)+'\n')
    with (folder/'reference-hold-audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(archived_summary=summary,synthetic_timing_test=audit),indent=2))


if __name__=='__main__':main()
