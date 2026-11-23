"""Plot measured reset difficulty; this is an input audit, not a learning curve."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'BalanceRL'))
from quiet_reset import Generator,validate_bank,LEVEL_DEG
from pose_reference import error,features
from standing_math import sole_centers
from config import get_config


def main():
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    out={}
    for model,color in [('v1','#3673b4'),('v2','#df8226'),('xshape','#2b946e')]:
        c=get_config('quiet_'+model)
        bank,q0,meta=validate_bank(ROOT/'BalanceRL'/c.quiet_pose_bank,ROOT/'BalanceRL'/c.model_path,
                                 ROOT/'BalanceRL'/c.quiet_reference_path)
        g=Generator(ROOT/'BalanceRL'/c.model_path,ROOT/'BalanceRL'/c.quiet_reference_path)
        rows=[];qm=[];tm=[]
        for level,poses in enumerate(bank['train']):
            joint=[];tilt=[];foot=[]
            for q in poses:
                g.set(q)
                f=features(q[g.act],g.d.xmat[g.bid].reshape(3,3),g.d.xpos[g.bid],sole_centers(g.d,g.p))
                e=error(f,g.reference['features'],c)
                joint.append(np.rad2deg(e['joint_rms']));tilt.append(np.rad2deg(e['tilt']));foot.append(e['feet_max']*1000)
            rows.append(dict(level=level,requested_deg=LEVEL_DEG[level],joint_rms_mean_deg=float(np.mean(joint)),
                joint_rms_p10_p90_deg=np.percentile(joint,[10,90]).tolist(),
                tilt_error_mean_deg=float(np.mean(tilt)),foot_error_mean_mm=float(np.mean(foot))))
            qm.append(joint);tm.append(tilt)
        for ax,values in zip(axes,(qm,tm)):
            means=np.mean(values,axis=1)
            ax.plot(LEVEL_DEG,means,'o-',label=model.upper(),color=color)
            bounds=np.percentile(values,[10,90],axis=1)
            ax.fill_between(LEVEL_DEG,*bounds,color=color,alpha=.13)
        out[model]=dict(rows=rows,reference_audit=meta['reference_audit'],
            split_shapes={k:list(v.shape) for k,v in bank.items()},attempts=meta['attempts'])
    for ax,title in zip(axes,('Actual motor RMS deviation','Actual torso-gravity deviation')):
        ax.set(xlabel='Requested angular noise (deg)',ylabel='Deviation from quiet reference (deg)',title=title)
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('QUIET-01 initial-pose curriculum: measured projection results (mean and 10–90%)')
    folder=ROOT/'report/figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'): fig.savefig(folder/f'quiet01_reset_difficulty.{ext}',dpi=180)
    data=ROOT/'report/data/quiet01';data.mkdir(parents=True,exist_ok=True)
    (data/'reset-audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({m:[round(r['joint_rms_mean_deg'],2) for r in d['rows']] for m,d in out.items()}))


if __name__=='__main__': main()
