"""Build a portable, evidence-based report collection without cluster access.

Run from the repository root; --frozen rebuilds from the already copied inputs.
Use --manifest-only after editing notes to retain existing figures and input provenance.
Authored Markdown is maintained separately and is never overwritten here.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'dash_report_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FIG = ROOT / 'figures'
DATA = ROOT / 'data'
EXPERIMENTS = ['stop01_20261002_v2', 'stop02_20261006_v3', 'stop03_20261007_v1']
MEDIA = [
    ('source_walking.mp4', 'RLframework/results/dash_joy3_lr_s2_180M_ladder.mp4',
     'Original walking policy', 'Historical 180M walking-source evaluation. Background locomotion capability, not a STOP-03 result; a pre-existing command-ladder recording.'),
    ('stop01_abrupt_failure.mp4', 'RLframework/results/stop01_replay/final_stop_1mps.mp4',
     'Abrupt STOP failure', 'STOP-01B final policy, requested 1 m/s. Historical trace replay illustrating failure after STOP; not a successful hold.'),
    ('stop02_ramp_failure.mp4', 'RLframework/results/stop02_checks/final_ramp_1mps.mp4',
     'Smooth reference, continued failure', 'STOP-01B final policy with the STOP-02 command adapter, trial 1 at 1 m/s. This is a command-only diagnostic, not the trained STOP-02 standing policy.'),
    ('stop02_longest_survival.mp4', 'RLframework/results/stop02_checks/longest_upright_stop_trial23.mp4',
     'Longest recorded survival after STOP', 'Selected for survival, not success: trial 23, requested 2 m/s, STOP at 3.65 s, ramp 5.151 s, tip at 9.99 s. Zero quiet hold. Half-speed replay.'),
    ('nominal_stance_teacher.mp4', 'RLframework/results/stop02_checks/stance_feasibility.mp4',
     'Fixed nominal stance feasibility', 'Zero-action controller: nominal PD targets and gains, not zero torque. This illustrates a feasible teacher stance, not a learned STOP policy. Probe resets disable joint noise; corrected aggregate diagnostic is 31/32 holds.'),
]
CAPTIONS = {
    'control_architecture': ('Current control architecture', 'Current STOP-03 policy path: 384 actor observations, estimated velocity, 47 actions, gait decoding and motor feedback. Critic and privileged inputs are training-only. The runtime path is implemented; STOP-03 hardware use is not validated.'),
    'stop03_outcomes': ('Imitation enables holding; later PPO trades stillness for survival', '32 repeated near-stance trials per checkpoint, evaluation seed 4102. Source 0/32, post-imitation and 524k PPO 21/32, final 14/32. Survival alone is not five-second hold success. Reward-floor fraction uses live STOP ticks.'),
    'standing_comparison': ('Standing PPO with and without imitation initialization', 'STOP-02 and STOP-03 use the standing protocol. STOP-03 step zero here is post-imitation; STOP-02 is its source baseline. One training seed per run; baseline differences and repeated evaluations preclude a multi-seed causal claim.'),
    'imitation_fit': ('Held-out action fitting', '2,000 actor-only Adam updates. Whole demonstration environments are reserved for validation. MSE concerns action matching, not physical success; the physical gate is reported separately.'),
    'walking_retention': ('Walking retained relative to the source', 'Nonzero-command means from the walking ladder. Dashed thresholds are this run\'s source survival minus 0.05 and source error plus 0.10 m/s. Aggregate retention does not guarantee every speed succeeds.'),
    'quiet_periods': ('Every trial: longest continuous quiet interval', 'All 32 trials at each selected stage. Green points pass the full hold criterion; red points fail. The five-second line alone does not capture support, drift and full-window survival requirements.'),
    'ramp_trace': ('A ramp changes the command, not the measured motion directly', 'Recorded STOP-01B final-policy ramp diagnostic, trial 1, 1 m/s request. Trace stops at first termination; no auto-reset trajectory is shown. This is an earlier diagnostic, not STOP-03.'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(name):
    return [json.loads(x) for x in (DATA/name/'stop_evaluations.jsonl').read_text().splitlines() if x.strip()]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(FIG/f'{name}.{ext}', dpi=180, bbox_inches='tight')
        if ext == 'svg':
            svg = FIG/f'{name}.svg'
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set(xlim=(0, 13), ylim=(0, 8)); ax.axis('off')
    def box(x, y, w, h, text, color='#e8f1fa'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',
                                   facecolor=color,edgecolor='#37617e',linewidth=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b, label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#476074'))
        if label:
            ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.14,label,ha='center',fontsize=8)
    ax.text(.1,7.7,'DASH-01 | current STOP-03 control path',fontsize=20,weight='bold')
    box(.2,6,2.6,1,'Operator request\nSpeed + STOP')
    box(3.4,6,4.2,1,'Shared command adapter\nWALK -> BRAKE -> HOLD\nFiltered estimated speed; smooth reference')
    box(8.3,6,4.2,1,'Training only: critic\n413 inputs -> 256 -> 256 -> value\nIncludes 29 privileged inputs', '#fff0d9')
    box(.2,3.8,2.6,1.25,'Observation history\nJoints, torque, IMU,\nphase, prior residual, heading\n+ latched gait and task = 384')
    box(3.4,3.8,2.2,1.25,'Velocity estimator\n384 -> 128 -> 64 -> 3')
    box(6.2,3.8,2.7,1.25,'Actor, 100 Hz\n387 -> 256 -> 256 -> 47\nClipped mean in evaluation')
    box(9.5,3.8,3,1.25,'Action split\n41 gait values: cycle latch\n6 residuals: every tick')
    box(8.8,1.55,3.7,1.25,'Fourier gait + residual corrections\n6 position targets, Kp and Kd\nJoint / slew limits')
    box(4.3,1.55,3.7,1.25,'Actuation\nSimulation: delay + PD + limits\nRuntime: safety, joint map, CAN')
    box(.2,1.55,3.2,1.25,'Robot / simulated plant\nSix actuated joints\nMeasurements feed back')
    arrow((2.8,6.5),(3.3,6.5)); arrow((4.5,5.1),(4.5,5.9))
    arrow((3.8,5.9),(1.6,5.1),'effective task')
    arrow((2.9,4.4),(3.3,4.4)); arrow((5.7,4.4),(6.1,4.4))
    ax.plot([2.85,2.95,2.95,6.0,6.0],[3.9,3.9,3.25,3.25,4.0],color='#476074',lw=1)
    arrow((6.0,4.0),(6.1,4.0)); ax.text(4.5,3.03,'normalized observation also enters actor',fontsize=8,ha='center')
    arrow((9.0,4.4),(9.4,4.4)); arrow((11,3.7),(11,2.9))
    arrow((8.7,2.2),(8.1,2.2)); arrow((4.2,2.2),(3.5,2.2)); arrow((1.6,2.9),(1.6,3.7))
    ax.text(.2,.75,'Policy selects gait parameters and joint corrections; it does not directly output motor torques.',fontsize=12)
    ax.text(.2,.30,'Implemented runtime path does not imply hardware validation. STOP-03 results start near stance, directly in HOLD.',fontsize=10,color='#5d6470')
    save(fig,'control_architecture')


def plots(rows, previous, imitation):
    labels=['Source','Imitation','PPO\n524k','PPO\n1.049M','PPO\n1.573M','PPO\n2.032M']
    assert len(rows)==len(labels), 'Update stage labels for the new evaluation sequence'
    x=np.arange(len(rows))
    fig, axs=plt.subplots(3,1,figsize=(10,10),sharex=True,layout='constrained')
    axs[0].plot(x,[r['success_fraction']*100 for r in rows],'o-',label='Five-second hold')
    axs[0].plot(x,[r['survive_fraction']*100 for r in rows],'s--',label='Survived seven seconds')
    axs[0].set(ylabel='Trials (%)',ylim=(-3,105),title=CAPTIONS['stop03_outcomes'][0]);axs[0].legend()
    axs[1].plot(x,[r['longest_quiet_hold_mean_s'] for r in rows],'o-',color='#23947e')
    axs[1].set(ylabel='Mean longest quiet interval (s)',ylim=(0,7))
    axs[2].plot(x,[100*r['stop_reward_floor_fraction'] for r in rows],'o-',color='#b85c38')
    axs[2].set(ylabel='Reward-floor ticks (%)',xticks=x,xticklabels=labels)
    save(fig,'stop03_outcomes')
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for rr,label in [(previous,'STOP-02: standing PPO'),(rows[1:],'STOP-03: imitation + standing PPO')]:
        ax.plot([r['step']/1e6 for r in rr],[100*r['success_fraction'] for r in rr],'o-',label=label)
    ax.set(xlabel='PPO environment steps (millions)',ylabel='Five-second hold success (%)',ylim=(-3,103),title=CAPTIONS['standing_comparison'][0]);ax.legend()
    save(fig,'standing_comparison')
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    h=imitation['history']
    for key,label in [('hold_mse','HOLD'),('walk_mse','RUN')]:
        ax.semilogy([r['update'] for r in h],[r[key] for r in h],'o-',label=label)
    ax.set(xlabel='Supervised updates (not PPO steps)',ylabel='Held-out action MSE',title='Actor imitation: fit improves on held-out trajectories');ax.legend()
    save(fig,'imitation_fit')
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    base=rows[0]['walking_run_only']
    axs[0].plot(x,[r['walking_run_only']['alive_frac']*100 for r in rows],'o-')
    axs[0].axhline(100*(base['alive_frac']-.05),ls='--',color='#b85c38',label='Minimum retained survival')
    axs[0].set(ylabel='RUN survival (%)',ylim=(0,100),title=CAPTIONS['walking_retention'][0]);axs[0].legend()
    axs[1].plot(x,[r['walking_run_only']['track_err_mean'] for r in rows],'o-')
    axs[1].axhline(base['track_err_mean']+.1,ls='--',color='#b85c38',label='Maximum retained tracking error')
    axs[1].set(ylabel='RUN tracking error (m/s)',xticks=x,xticklabels=labels,ylim=(0,.55));axs[1].legend()
    save(fig,'walking_retention')
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    for i,rr in enumerate([rows[0],rows[1],rows[2],rows[-1]]):
        episodes=rr['episodes']; offset=np.linspace(-.20,.20,len(episodes))
        ax.scatter(i+offset,[r['longest_quiet_hold_s'] for r in episodes],
                   c=['#168568' if r['stop_success'] else '#be5943' for r in episodes],s=28,alpha=.8)
    ax.axhline(5,ls='--',color='#333333',label='Five-second duration threshold')
    ax.set(xticks=range(4),xticklabels=['Source','Imitation','Best PPO 524k','Final PPO'],ylabel='Longest quiet interval (s)',ylim=(-.15,7.3),title=CAPTIONS['quiet_periods'][0]);ax.legend()
    save(fig,'quiet_periods')
    trace=np.genfromtxt(DATA/'ramp_trial_1.csv',delimiter=',',names=True)
    meta=read(DATA/'ramp_trial_1.json')
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    axs[0].plot(trace['time_s'],trace['vx_mps'],label='Measured forward speed')
    axs[0].plot(trace['time_s'],trace['reference_mps'],'--',label='Reference speed')
    axs[0].set(ylabel='Speed (m/s)',title=CAPTIONS['ramp_trace'][0]);axs[0].legend()
    axs[1].plot(trace['time_s'],trace['tilt_deg']);axs[1].axhline(10,ls='--',color='#999');axs[1].set(ylabel='Tilt (degrees)')
    axs[2].plot(trace['time_s'],trace['joint_rms_radps']);axs[2].axhline(.5,ls='--',color='#999');axs[2].set(ylabel='Joint velocity RMS (rad/s)',xlabel='Simulation time (s)')
    for ax in axs:
        ax.axvline(meta['stop_s'],ls=':',color='#b85c38')
        ax.axvspan(meta['stop_s'],min(meta['hold_s'],trace['time_s'][-1]),alpha=.10,color='#e5a33d')
    save(fig,'ramp_trace')
    architecture()


def refresh_manifest(metadata=None):
    """Refresh output hashes without regenerating figures or changing source records."""
    path = ROOT / 'manifest.json'
    if metadata is None:
        metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata['outputs'] = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts
    }
    metadata['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--frozen', action='store_true')
    mode.add_argument('--manifest-only', action='store_true',
                      help='refresh output hashes, preserving figures and input provenance')
    args = ap.parse_args()
    if args.manifest_only:
        refresh_manifest()
        print('Refreshed report output hashes; preserved figures and source records.')
        return
    for folder in (FIG,DATA,ROOT/'videos',ROOT/'tables'):
        folder.mkdir(parents=True,exist_ok=True)
    provenance=[]
    def copy(source,destination):
        src=REPO/source;dst=ROOT/destination
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        provenance.append(dict(source=source,copy=destination,sha256=digest(dst)))
    if not args.frozen:
        for name in EXPERIMENTS:
            src=REPO/'docs/experiments'/name
            for path in sorted(src.iterdir()):
                if path.suffix in ('.json','.jsonl','.csv','.md'):
                    copy(path.relative_to(REPO).as_posix(),f'data/{name}/{path.name}')
        for name in ['stop-training-report.md','stop02-training-plan.md','stop02-experiment-report.md','stop03-experiment-report.md','stop04-experiment-report.md','stop05-experiment-report.md','stop06-experiment-report.md']:
            copy('docs/'+name,'data/notes/'+name)
        for path in sorted((REPO/'RLframework/results/stop02_checks').glob('*.json')):
            copy(path.relative_to(REPO).as_posix(),'data/diagnostics/'+path.name)
        for name in ['env.py','config.py','gait.py','drive.py','networks.py','ppo.py','standing_imitation.py','stop_eval.py','stop02_metrics.py','stop_adapter.py']:
            copy('RLframework/'+name,'data/code/RLframework/'+name)
        for name in ['controller_v2.py','stop_schedule.py']:
            copy('controller/deploy/'+name,'data/code/controller/deploy/'+name)
        for out,src,_,_ in MEDIA:
            copy(src,'videos/'+out)
        trace_path=REPO/'RLframework/results/stop02_checks/final_ramp.npz'
        with np.load(trace_path,allow_pickle=False) as z:
            meta=json.loads(str(z['meta'].item()));i=1
            done=np.flatnonzero(z['done'][:,i]);end=int(done[0]) if len(done) else len(z['done'])-1
            tt=np.arange(end+1)*meta['control_dt']
            vals=np.column_stack([tt,z['vx'][:end+1,i],z['reference'][:end+1,i],
                np.degrees(np.arccos(np.clip(z['upright'][:end+1,i],-1,1))),z['joint_rms'][:end+1,i]])
            np.savetxt(DATA/'ramp_trial_1.csv',vals,delimiter=',',comments='',header='time_s,vx_mps,reference_mps,tilt_deg,joint_rms_radps')
            stop=meta['stop_ticks'][i]*meta['control_dt']
            (DATA/'ramp_trial_1.json').write_text(json.dumps(dict(trial=i,source=trace_path.relative_to(REPO).as_posix(),
                source_sha256=digest(trace_path),stop_s=stop,hold_s=stop+float(z['ramp_duration'][meta['stop_ticks'][i],i]),
                first_termination_s=float(tt[-1])),indent=2)+'\n')
        provenance.append(read(DATA/'ramp_trial_1.json'))
    else:
        provenance=read(ROOT/'manifest.json')['inputs']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'svg.fonttype':'none'})
    rows=records(EXPERIMENTS[-1]);previous=records(EXPERIMENTS[1]);imitation=read(DATA/EXPERIMENTS[-1]/'imitation.json')
    plots(rows,previous,imitation)
    summary=[]
    for name in EXPERIMENTS:
        for r in records(name):
            summary.append(dict(experiment=name,protocol=r.get('protocol','STOP-01'),
                stage=r.get('stage','source' if r['step']==0 else 'ppo'),ppo_steps=r['step'],
                success_fraction=r['success_fraction'],survive_fraction=r['survive_fraction'],
                longest_quiet_hold_mean_s=r.get('longest_quiet_hold_mean_s',''),walking_retained=r.get('walking_retained','')))
    table(ROOT/'tables/experiment_evaluations.csv',summary)
    table(ROOT/'tables/imitation_updates.csv',imitation['history'])
    table(ROOT/'tables/stop03_trials.csv',[dict(stage=r['stage'],ppo_steps=r['step'],trial=i,**e) for r in rows for i,e in enumerate(r['episodes'])])
    import imageio_ffmpeg
    video_info=[]
    for out,source,title,caption in MEDIA:
        reader=imageio_ffmpeg.read_frames(str(ROOT/'videos'/out))
        meta=next(reader);reader.close()
        video_info.append(dict(file='videos/'+out,source=source,title=title,caption=caption,
                               duration_s=meta['duration'],fps=meta['fps'],size=meta['size']))
    guide=['# Figure and video guide','','All videos are existing simulation recordings. No STOP-03 replay is available',
           'locally in this collection. Do not present earlier clips as STOP-03 evidence.','',
           '## Figures','']
    for name,(title,caption) in CAPTIONS.items():
        guide += [f'### {title}','',caption,'',f'[PNG](figures/{name}.png) | [SVG](figures/{name}.svg) | [PDF](figures/{name}.pdf)','']
    guide += ['## Videos','','Playback rate overlays, when present, refer to simulation time. File durations',
              'below are wall-clock video durations. Trial-selection information is limited to',
              'what existing records establish; historical clips are not new matched evaluations.','']
    for v in video_info:
        guide += [f"### {v['title']}",'',v['caption'],'',f"[Play / copy video]({v['file']}) ({v['duration_s']:.1f} s, {v['fps']:g} fps).",
                  f"Original: `{v['source']}`.",'']
    guide += ['## Evidence and provenance','','`manifest.json` records hashes of copies and generated files. Metrics under',
              '`data/` are copied from the downloaded experiment folders. Selected current',
              'source files under `data/code/` document the architecture; original training',
              'source manifests remain in each copied experiment directory. Historical notes',
              'under `data/notes/` preserve their original relative links; use this report\'s',
              'top-level documents and manifest to navigate the portable evidence.','',
              'Regenerate from the frozen local report inputs:','',
              '```powershell','.\\.venv\\Scripts\\python.exe report/scripts/build_report.py --frozen','```','']
    (ROOT/'assets.md').write_text('\n'.join(guide),encoding='utf-8')
    esc=html.escape
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
          '<title>DASH-01 stopping research</title><style>body{font:17px/1.6 system-ui,sans-serif;background:#edf2f6;color:#172c3b;margin:0}main{max-width:1080px;margin:auto;padding:36px 24px}h1{font-size:38px;line-height:1.15}h2{margin-top:44px}a{color:#12668c}figure,section{background:white;padding:24px;border-radius:12px;margin:24px 0}img,video{width:100%;height:auto}figcaption{color:#435765}nav a{display:inline-block;margin-right:18px}.badge{color:#126957;font-weight:700}small{color:#536677}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}</style><main>',
          '<p class="badge">DASH-01 / RESEARCH RECORD / 12 OCTOBER 2026</p><h1>Teaching a walking robot to stop</h1>',
          '<p>Imitation enabled sustained standing. Later reinforcement learning improved survival but reduced strict stillness. Stopping from motion remains to be validated.</p>',
          '<nav><a href="README.md">Report index</a><a href="architecture.md">Architecture</a><a href="experiments.md">Experiment history</a><a href="evaluation.md">Measurement</a><a href="presentation.md">Presentation outline</a><a href="work-log.md">Work log</a><a href="assets.md">Asset guide</a></nav>',
          '<section><h2>Recorded STOP-03 outcome</h2><table><tr><th>Stage</th><th>Five-second holds</th><th>Survived 7 seconds</th></tr><tr><td>Source</td><td>0/32</td><td>0/32</td></tr><tr><td>After imitation</td><td>21/32</td><td>22/32</td></tr><tr><td>Best PPO, 524k</td><td>21/32</td><td>22/32</td></tr><tr><td>Final PPO, 2.03M</td><td>14/32</td><td>31/32</td></tr></table><p>All STOP-03 walking-retention checks passed. These are near-standing resets in nominal simulation, with one fixed evaluation seed.</p></section><h2>Figures for the report</h2>']
    for name,(title,caption) in CAPTIONS.items():
        page.append(f'<figure><h3>{esc(title)}</h3><img loading="lazy" src="figures/{name}.png" alt="{esc(title)}"><figcaption>{esc(caption)} <a href="figures/{name}.svg">SVG</a> · <a href="figures/{name}.pdf">PDF</a></figcaption></figure>')
    page += ['<h2>Simulation video collection</h2><p><strong>No STOP-03 replay is included yet.</strong> These clips explain the earlier failures and teacher feasibility. They are not videos of the 21/32 result.</p>']
    for v in video_info:
        page.append(f'<figure><h3>{esc(v["title"])}</h3><video controls preload="none" src="{v["file"]}">Video playback unavailable; use the download link.</video><figcaption>{esc(v["caption"])} <a href="{v["file"]}">Open video</a></figcaption><small>Source: {esc(v["source"])}</small></figure>')
    page += ['<p>Static local gallery. No external services, training jobs or live polling. Detailed provenance: <a href="manifest.json">manifest</a>.</p></main></html>']
    (ROOT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    refresh_manifest(dict(collection_date='2026-10-12', inputs=provenance, videos=video_info))
    print(f'Built {len(CAPTIONS)} figures (PNG/SVG/PDF), {len(video_info)} videos, 3 CSV tables, gallery and manifest.')
    print(f'Total collection: {sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())/1e6:.1f} MB')


if __name__=='__main__':
    main()
