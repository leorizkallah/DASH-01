"""Build completed-run evidence tables and learning curves without new simulation."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'report/data/quiet01v2/results'


def records(path):return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained');summaries=[]
    for model,color in [('v1','#3673b4'),('v2','#dc862b'),('xshape','#27946b')]:
        folder=DATA/model
        load=lambda name:json.loads((folder/(name+'.json')).read_text())
        rows=records(folder/'progress.jsonl');ev=records(folder/'quiet_pose_evaluations.jsonl')
        retained=records(folder/'quiet_pose_retention.jsonl')
        best=load('best');final=load('final');baseline=load('baseline_eval');end=load('final_eval')
        done=load('completion');b=best['eval']
        x=np.array([r['time/env_steps'] for r in rows])/1e6
        assert np.all(np.diff(x)>0) and done['reason']=='step_limit' and done['steps']==100012032
        bad={k:sum(not np.isfinite(r[k]) for r in rows) for k in
             ('rollout/reward_mean','train/kl','train/vf','train/est_loss','diag/pose_joint_rms','diag/pose_foot_error')}
        assert not any(bad.values()),bad
        reward=np.array([r['rollout/reward_mean'] for r in rows]);blocks=np.arange(0,len(rows),50)
        axes[0,0].plot([x[j:j+50].mean() for j in blocks],
                       [reward[j:j+50].mean() for j in blocks],label=model.upper(),color=color)
        ex=np.array([e['step'] for e in ev])/1e6;rx=np.array([e['step'] for e in retained])/1e6
        axes[0,1].plot(ex,[100*e['score'] for e in ev],'.-',color=color,label=model.upper())
        axes[0,1].plot(best['step']/1e6,100*b['score'],'*',ms=13,color=color)
        axes[1,0].plot(rx,[100*e['score'] for e in retained],'.-',color=color,label=model.upper())
        axes[1,1].plot(rx,[100*e['quiet_rate'] for e in retained],'.-',color=color,label=model.upper())
        axes[2,0].step(x,[r['curriculum/pose_level'] for r in rows],where='post',color=color,label=model.upper())
        err=np.array([r['diag/pose_foot_error']*1000 for r in rows])
        axes[2,1].plot([x[j:j+50].mean() for j in blocks],
                       [err[j:j+50].mean() for j in blocks],color=color,label=model.upper())
        item=dict(model=model,steps=done['steps'],elapsed_h=done['elapsed_s']/3600,
            baseline_passes=round(96*baseline['score']),best_passes=round(96*b['score']),
            final_passes=round(96*end['score']),best_step=best['step'],
            best_reference_passes=round(16*b['quiet_rate']),
            best_retention_reference_passes=round(16*b['retention_reference_rate']),
            best_retention_passes=round(96*b['retention_score']),
            final_retention_passes=round(96*end['retention_score']),
            best_rates=b['rates'],final_level=final['cur']['pose_level'],nonfinite_diagnostics=bad)
        summaries.append(item)
        f,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for key,label in [('reference','Exact reference'),('0','0.5 deg requested'),('1','1.5 deg requested'),
                          ('2','3 deg requested'),('3','5 deg requested'),('4','8 deg requested')]:
            axs[0].plot(ex,[100*e['rates'][key] for e in ev],'.-',label=label)
        axs[1].plot(ex,[100*e['score'] for e in ev],'.-',label='12 s return task')
        axs[1].plot(rx,[100*e['score'] for e in retained],'.-',label='20 s retention task')
        for ax in axs:ax.set(xlabel='Additional transitions (millions)',ylabel='Passing trials (%)',ylim=(-3,103));ax.legend(fontsize=8);ax.grid(alpha=.2)
        f.suptitle(model.upper()+': quiet-pose recovery and independent retention')
        for ext in ('png','pdf','svg'):f.savefig(ROOT/f'report/figures/quiet01v2_{model}_convergence.{ext}',dpi=180)
        plt.close(f)
    titles=['Training reward (50-update block mean)','12 s task success, all96 tests (* saved best)',
            '20 s retention success, all96 tests','20 s retention from exact reference (16 tests)',
            'Training reset-difficulty frontier','Training mean sole-relative error (50-update mean)']
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlabel='Additional transitions (millions)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    for ax in (axes[0,1],axes[1,0],axes[1,1]):ax.set(ylabel='Passing trials (%)',ylim=(-3,103))
    axes[2,0].set(yticks=range(5),yticklabels=['0: 0.5 deg','1: 1.5 deg','2: 3 deg','3: 5 deg','4: 8 deg'],ylabel='Requested noise before projection')
    axes[2,1].set_ylabel('Sole-relative error (mm)')
    fig.suptitle('QUIET-01 v2: 100M additional transitions per model; fixed seed1234 evaluations')
    for ext in ('png','pdf','svg'):fig.savefig(ROOT/f'report/figures/quiet01v2_convergence.{ext}',dpi=180)
    (DATA/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (DATA/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        fields=[k for k in summaries[0] if k not in ('best_rates','nonfinite_diagnostics')]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    lines=['# QUIET-01 v2 completed results','',
        '[Best-policy video gallery](quiet01v2-videos.html), [media provenance/status](quiet01v2-media.md).','',
        '[Interpretation: what we have and what remains uncertain](quiet01-v2-assessment.md).','',
        'Checked23November2026. GPU preflights3263184 and training3263185 all complete exit0.',
        'Each model reaches100,012,032 additional transitions (step limit). No jobs remain running.','',
        '| Model | Baseline /96 | Best /96 | Final /96 | Best step (M) | Reference /16 | Reference retention /16 | Runtime |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in summaries:
        lines.append(f"| {d['model'].upper()} | {d['baseline_passes']} | {d['best_passes']} | {d['final_passes']} | {d['best_step']/1e6:.2f} | {d['best_reference_passes']} | {d['best_retention_reference_passes']} | {d['elapsed_h']:.2f} h |")
    lines+=['','Best selection optimizes the12s return task with quiet-reference qualification.',
        'Reference retention means the separate20s test, including its final five-second quiet window.',
        'Both tests use the same initial-pose test split and sensor-noise seed1234.','',
        '![Learning curves](figures/quiet01v2_convergence.png)','',
        'V2 and X-shape improve substantially on these tests. All selected policies pass16/16',
        'quiet-reference return trials. Harder initial deviations remain less reliable:',
        'the hardest requested8deg bank passes4/16(V1),7/16(V2),11/16(X) at the selected checkpoints.',
        'Requested angles are projected onto feasible poses; they are not achieved motor RMS deviations.',
        'Final training frontier is level2(V1,3deg requested) and level3(V2/X,5deg requested).','',
        'X-shape has the best saved return score, but its final score is lower than its earlier best.',
        'V1 also ends below its best. These single-seed runs do not prove convergence or general',
        'robustness. There are no pushes or initial-velocity disturbances in this experiment.',
        'No non-finite values occur in logged reward, KL, value loss, estimator loss or posture-error',
        'diagnostics. The disabled push-gate metric intentionally remains NaN and is not a loss.','',
        '[Numeric summary](data/quiet01v2/results/summary.csv),',
        '[complete summary](data/quiet01v2/results/summary.json). Raw progress, evaluations,',
        'probes, preflight receipts and best/final checkpoints are archived under',
        '`report/data/quiet01v2/results/`. Individual plots:',
        '[V1](figures/quiet01v2_v1_convergence.png), [V2](figures/quiet01v2_v2_convergence.png),',
        '[X-shape](figures/quiet01v2_xshape_convergence.png).','',
        'Historical hold diagnostic3263170 also completed; see',
        '[the v1 hold diagnosis](quiet01-hold-diagnosis.md) for its independently labeled controls.',
        'No new training or video-generation jobs were submitted in this status check.','']
    (ROOT/'report/quiet01-v2-results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([{k:v for k,v in d.items() if k not in ('best_rates','nonfinite_diagnostics')} for d in summaries],indent=2))


if __name__=='__main__':main()
