"""STAND-04 report gallery, learning curves and explicit final/best curriculum levels."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from build_standing03_assets import block, save, ROOT, REPORT, FIG, MODELS, NAMES, COLORS

DATA = REPORT/'data/standing04/results'
STAGES = ['Horizontal COM pulses', 'Random timing/duration', 'Add elevation', 'Add COM offsets']


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    fig.suptitle('STAND-04: 200M additional transitions per model',fontsize=17)
    summary=[]
    for model in MODELS:
        folder=DATA/model
        rows=records(folder/'progress.jsonl')
        ev=records(folder/'recovery_evaluations.jsonl')
        probes=records(folder/'recovery_curriculum_probes.jsonl')
        best=json.loads((folder/'best.json').read_text())
        final=json.loads((folder/'final.json').read_text())
        x=np.array([r['time/env_steps'] for r in rows])/1e6
        ex=np.array([r['step'] for r in ev])/1e6
        px=np.array([r['step'] for r in probes])/1e6
        sx=best['step']/1e6
        assert np.all(np.diff(x)>0) and x[-1] == 200.005632
        assert all(np.isfinite(r['rollout/reward_mean']) for r in rows)
        bx,by=block(x,[r['rollout/reward_mean'] for r in rows])
        label=NAMES[model]; color=COLORS[model]
        levels=np.array([r['curriculum/push_level'] for r in rows])
        stages=np.array([r['curriculum/push_stage'] for r in rows])
        end=float(final['cur']['push_level'])
        mass=json.loads((REPORT/f'data/standing03/built-{model}.json').read_text())['mass_kg']
        per_trial=best['eval']['per_trial']
        pass_mask=np.asarray(per_trial['passed'])
        successful_steps=np.asarray(per_trial['steps'])[pass_mask]
        item=dict(model=model,best_step=best['step'],best_passes=round(80*best['eval']['score']),
            best_quiet_passes=round(16*best['eval']['quiet_rate']),
            baseline_passes=round(80*ev[0]['score']),final_passes=round(80*ev[-1]['score']),
            final_push_level=end,final_stage=final['cur']['push_stage'],
            best_checkpoint_training_level=best['cur']['push_level'],
            best_checkpoint_stage=best['cur']['push_stage'],
            max_logged_training_level=float(levels.max()),max_probed_level=max(r['level'] for r in probes),
            best_rates=best['eval']['survival'],mass_kg=mass,final_impulse_ns=mass*end,
            final_force_100ms_n=mass*end/.1,
            passing_trials_with_landings=int((successful_steps>0).sum()))
        replay_path=folder/'replay.json'
        if replay_path.exists():
            replay=json.loads(replay_path.read_text())
            item.update(replay_passes=round(80*replay['replay_score']),replay_rates=replay['rates'],
                        video_trials=[{k:replay['per_trial'][k][i] for k in ('passed','duration_s','steps','physical_fall','task_violation')}
                                      for i in replay['selected_trials']])
        summary.append(item)
        axes[0,0].plot(bx,by,color=color,label=label)
        axes[0,1].plot(ex,[100*r['score'] for r in ev],'.-',color=color,label=label)
        axes[0,1].plot(sx,100*best['eval']['score'],'*',color=color,ms=13)
        axes[1,0].plot(ex,[100*r['quiet_rate'] for r in ev],'.-',color=color,label=label)
        axes[1,1].plot(x,levels,color=color,label=f'{label}: ends {end:.4f}')
        axes[1,1].plot(x[-1],end,'o',color=color)
        f,a=plt.subplots(3,2,figsize=(13,11))
        f.suptitle(f'{label}: recovery fine-tuning and curriculum',fontsize=17)
        a[0,0].plot(x,[r['rollout/reward_mean'] for r in rows],alpha=.14,lw=.5,color=color,rasterized=True)
        a[0,0].plot(bx,by,color=color)
        a[0,0].set(title='Training reward: raw + 100-update block mean',ylabel='Reward / control transition')
        for level in ('0.00','0.10','0.15','0.20','0.30'):
            a[0,1].plot(ex,[100*r['survival'][level] for r in ev],'.-',ms=3,label=level+' m/s')
        a[0,1].set(title='Fixed full-variety task pass by impulse',ylabel='Passing trials [%]',ylim=(-3,103))
        a[0,1].legend(fontsize=8,ncol=3)
        for key,desc in [('score','Combined task pass'),('quiet_rate','Quiet-standing pass'),
                         ('full_duration_fraction','Full 20 s duration')]:
            a[1,0].plot(ex,[100*r[key] for r in ev],'.-',ms=3,label=desc)
        a[1,0].plot(sx,100*best['eval']['score'],'k*',ms=13,label='Saved best')
        a[1,0].set(title='Standing retention and task outcome',ylabel='Trials [%]',ylim=(-3,103))
        a[1,0].legend(fontsize=8)
        a[1,1].plot(x,levels,color=color,label='Training impulse ceiling')
        a[1,1].plot(px,[r['level'] for r in probes],'.',color='black',ms=2,label='Independent probe level')
        a[1,1].set(title=f'Curriculum: final {end:.4f} m/s-equivalent',ylabel='Impulse / mass [m/s]',ylim=(0,.32))
        a[1,1].legend(fontsize=8,loc='upper left')
        stage_axis=a[1,1].twinx()
        stage_axis.step(x,stages,where='post',color='gray',ls='--',alpha=.6)
        stage_axis.set(ylabel='Disturbance variety stage',ylim=(-.1,3.2),yticks=[0,1,2,3])
        for key,desc in [('hard_rate','Near-level pushes'),('quiet_rate','No-push retention')]:
            a[2,0].plot(px,[100*r[key] for r in probes],'.-',label=desc,ms=3)
        a[2,0].plot(px,[100*min(r['sector_rates']) for r in probes],'.-',ms=3,label='Worst direction sector')
        for value in (80,90,95): a[2,0].axhline(value,color='gray',lw=.6,ls=':',alpha=.5)
        a[2,0].set(title='Independent curriculum probes (unsmoothed)',ylabel='Pass rate [%]',ylim=(-3,103))
        a[2,0].legend(fontsize=8)
        rate=np.array([r['rollout/recovery_steps']/(512*36)*1000 for r in rows])
        a[2,1].plot(*block(x,rate),label='Whole-foot landings',color=color)
        a[2,1].set(title='Actual stepping during training',ylabel='Landings / 1000 control transitions',ylim=(0,None))
        for aa in a.flat:
            aa.axvline(sx,color='gray',ls=':',alpha=.7)
            aa.set(xlabel='Additional training transitions [millions]',xlim=(0,200.01))
            aa.grid(alpha=.2)
        f.text(.02,.012,f'Dotted line: saved best at +{sx:.2f}M. Fixed evaluation: 80 first episodes, 20 s, full variety/noise, final 5 s quiet.\n'
               'Curriculum probes: 40 quiet + 200 near-level pushes, fresh seeds. Training ceiling does not establish maximum recoverable impulse.',fontsize=9)
        save(f,f'standing04_{model}_convergence')
        with (folder/'evaluation-table.csv').open('w',newline='') as csvfile:
            fields=['step','score','quiet_rate','full_duration_fraction']
            writer=csv.DictWriter(csvfile,fieldnames=fields+['pass_'+k for k in best['eval']['survival']])
            writer.writeheader()
            writer.writerows({**{k:r[k] for k in fields},**{'pass_'+k:v for k,v in r['survival'].items()}} for r in ev)
    for aa,title,ylabel in zip(axes.flat,['Training reward','Fixed full-variety task pass',
          'Fixed quiet-standing pass','Training push strength'],
          ['Reward / transition','Passing trials [%]','Passing trials [%]','Impulse / mass [m/s]']):
        aa.set(title=title,ylabel=ylabel,xlabel='Additional training transitions [millions]',xlim=(0,200.01))
        aa.grid(alpha=.2); aa.legend(fontsize=9)
    axes[0,1].set_ylim(-3,103); axes[1,0].set_ylim(-3,103); axes[1,1].set_ylim(0,.32)
    fig.text(.02,.012,'Stars: saved best checkpoints; additional steps start at the STAND-03 parent. Reward uses 100-update (~1.84M step) means.\n'
             'Evaluation curves are raw. Push strength is a training input, not a measured limit. One seed per model; X-shape inertials provisional.',fontsize=9)
    save(fig,'standing04_convergence_comparison')
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    notes=['# STAND-04 videos, convergence and push strengths', '',
        '[Open the media gallery](standing04-videos.html). Figures are available as PNG/PDF/SVG.', '',
        '| Model | Final training Δv | Force at 100 ms | Level at best checkpoint | Best additional step |',
        '|---|---:|---:|---:|---:|']
    notes += [f'| {NAMES[r["model"]]} | {r["final_push_level"]:.4f} m/s | {r["final_force_100ms_n"]:.2f} N | {r["best_checkpoint_training_level"]:.4f} m/s | {r["best_step"]/1e6:.2f}M |' for r in summary]
    notes += ['', 'Δv is impulse divided by robot mass. Forces illustrate the upper level for a 100 ms pulse; actual durations and strengths vary. The curriculum setting is not a measured maximum capability.', '',
        'Best policies are replayed with the frozen training code, stage-3 disturbances, full sensor noise, seed 1234 and 80 first-episode trials. Quiet trial 0 and +X 0.20 m/s trial 48 were selected before outcomes. Videos show both independent trials side by side at 1× speed, with actual pulse timing. Early endings freeze visibly, without reset splicing.', '']
    for r in summary:
        m=r['model']
        notes += [f'## {NAMES[m]}', '', f'Archived best: {r["best_passes"]}/80; successful trials with detected whole-foot landings: {r["passing_trials_with_landings"]}/{r["best_passes"]}.', '']
        if 'replay_passes' in r:
            notes += [f'Independent replay: {r["replay_passes"]}/80. Replay per-level rates: {r["replay_rates"]}. Archived rates: {r["best_rates"]}.', '']
            notes += [f'- {label}: {"PASS" if t["passed"] else "FAIL"}, {t["duration_s"]:.2f} s, {t["steps"]} landings.' for label,t in zip(['Quiet trial','Pushed trial'],r['video_trials'])]
            notes += ['',f'[Video](videos/standing04_{m}_best.mp4) · [learning curves](figures/standing04_{m}_convergence.png).','']
        else:
            notes += ['Recording/rendering pending. [Pipeline state](data/standing04/video-pipeline.json).','']
    notes += ['## Evidence', '', 'Raw training/evaluation/probe logs, configs, best checkpoint weights, scheduler receipts, CSV tables and hashes are in `report/data/standing04/results/`. Replay job: 3248509; training array: 3247599. No new training submitted. Replay differences are retained rather than hidden. One seed per model; X-shape inertials remain provisional.']
    (REPORT/'standing04-media.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    with (DATA/'push-strengths.csv').open('w',newline='') as f:
        fields=['model','final_push_level','final_stage','final_impulse_ns','final_force_100ms_n',
                'best_checkpoint_training_level','best_checkpoint_stage','max_probed_level']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in summary)
    inputs=[p for m in MODELS for p in (DATA/m).glob('*.jsonl')]
    (DATA/'plot-provenance.json').write_text(json.dumps(dict(training_array=3247599,replay_array=3248509,
        inputs_sha256={p.relative_to(REPORT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        smoothing='Training curves: nonoverlapping 100-update means. Evaluation/probe points: raw.'),indent=2)+'\n')
    table=''.join(f'<tr><td>{NAMES[r["model"]]}</td><td>{r["final_push_level"]:.4f} m/s</td><td>{r["final_impulse_ns"]:.3f} N·s / {r["final_force_100ms_n"]:.2f} N</td><td>{r["final_stage"]}: {STAGES[r["final_stage"]]}</td><td>{r["best_checkpoint_training_level"]:.4f} m/s</td><td>{r["best_passes"]}/80</td></tr>' for r in summary)
    cards=[]
    for r in summary:
        m=r['model'];video=f'videos/standing04_{m}_best.mp4'
        markup=f'<video controls preload="metadata" poster="figures/standing04_{m}_preview.jpg" src="{video}"></video>' if (REPORT/video).exists() else '<p>Replay queued / rendering.</p>'
        detail=''
        stepping=f'<p>Archived passing trials with at least one detected whole-foot landing: {r["passing_trials_with_landings"]}/{r["best_passes"]}. Allowing recovery steps does not establish that a policy learned them.</p>'
        if 'replay_passes' in r:
            outcomes=[]
            for label,t in zip(['quiet','pushed'],r['video_trials']):
                outcomes.append(f'{label}: {"PASS" if t["passed"] else "FAIL"}, {t["duration_s"]:.2f} s, {t["steps"]} landings')
            detail=f'<p>Independent replay: {r["replay_passes"]}/80 (archive {r["best_passes"]}/80). Selected trials: '+ '; '.join(outcomes)+'.</p>'
        cards.append(f'<section><h2>{NAMES[m]} — best +{r["best_step"]/1e6:.2f}M</h2><p>Starting policy {r["baseline_passes"]}/80; saved best {r["best_passes"]}/80; final {r["final_passes"]}/80. Quiet standing at saved best: {r["best_quiet_passes"]}/16.</p>{markup}{detail}{stepping}<p>Two independent, preselected trials: quiet standing (left), +X 0.20 m/s-equivalent impulse (right), with stage-3 random onset, duration, elevation and torso offsets. Actual pulse timing is shown. Blue/orange identify left/right feet; 20 s at 1×. An ended trial freezes with an explicit warning; no resets are spliced.</p><a href="figures/standing04_{m}_convergence.pdf">Vector PDF</a><img src="figures/standing04_{m}_convergence.png" alt="{NAMES[m]} convergence"></section>')
    page='''<!doctype html><meta charset="utf-8"><title>STAND-04: recovery videos and learning curves</title>
<style>body{font:17px system-ui;max-width:1280px;margin:30px auto;padding:0 20px;background:#f3f5f8;color:#182635}section{background:white;padding:24px;margin:28px 0;border-radius:10px}video,img{width:100%;height:auto}p{line-height:1.5}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#1766a5}</style>
<h1>STAND-04: recovery stepping</h1><p>All three runs completed 200,005,632 additional transitions. Videos use saved best checkpoints; the final policies are weaker. Scores use the new final-standing protocol, not STAND-03's earlier protocol. Full raw results and provenance are retained in the report folder.</p>
<h2>Push strength at training end</h2><p><strong>Units are equivalent velocity impulse Δv = impulse / robot mass, not commanded walking speed.</strong> Final curriculum strength differs from the level at the earlier best checkpoint. Neither value is a demonstrated maximum recoverable push.</p>
<table><tr><th>Model</th><th>Final training level</th><th>Impulse / force for 100 ms</th><th>Final variety stage</th><th>Level at saved best</th><th>Best full-variety test</th></tr>'''+table+'''</table><p>Forces are illustrative for a 100 ms pulse at the final upper level; actual pulse duration is randomized and training also samples easier pushes. Stages: 0 horizontal COM pulses; 1 random timing/duration; 2 adds elevation; 3 adds COM offsets. Fixed checkpoint tests always use stage 3, irrespective of the training stage reached.</p>
<p><a href="standing04-progress.md">Experiment and metric definitions</a> · <a href="data/standing04/results/push-strengths.csv">Push-level CSV</a> · <a href="figures/standing04_convergence_comparison.pdf">Comparison PDF</a></p><img src="figures/standing04_convergence_comparison.png" alt="Comparison learning curves">'''+''.join(cards)
    (REPORT/'standing04-videos.html').write_text(page,encoding='utf-8')
    complete=all((REPORT/f'videos/standing04_{m}_best.mp4').exists() and
                 (DATA/m/'video-provenance.json').exists() for m in MODELS)
    if not complete:
        pipeline=REPORT/'data/standing04/video-pipeline.json'
        ps=json.loads(pipeline.read_text()) if pipeline.exists() else {'status':'queued'}
        banner=f'<section><strong>Videos are not ready yet.</strong> SCITAS replay array 3248509 is queued/running; local pipeline: {ps["status"]}. Current scheduler estimate: around 19:38 CET on 11 November, subject to change. Automatic fetching and rendering require this computer to remain awake and connected. This page refreshes every 120 seconds until the three videos are published. Convergence graphs and the push table below are ready.</section>'
        page=page.replace('<meta charset="utf-8">','<meta charset="utf-8"><meta http-equiv="refresh" content="120">')
        page=page.replace('<h1>STAND-04: recovery stepping</h1>','<h1>STAND-04: recovery stepping</h1>'+banner)
        (REPORT/'standing04-videos.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
