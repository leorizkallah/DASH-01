"""Paired diagnostics and held-out-seed assessment for the STOP-04 pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import numpy as np
from evaluate import load_run
from stop_eval import StopEvaluator
from stop02_metrics import summarize
from stop04_metrics import diagnose


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def diagnostics(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    cfg,env,agent=load_run(args.source_run,args.source_run/'best_stop.msgpack',n_envs=32)
    evaluator=StopEvaluator(agent,env,capture=True)
    results={};videos=[]
    for name,filename in [('best','best_stop.msgpack'),('final','final.msgpack')]:
        checkpoint=args.source_run/filename
        agent.load(checkpoint)
        trace=jax.tree_util.tree_map(np.asarray,evaluator._run(agent.params,agent.stats,4102))
        summary=summarize(trace,evaluator.speeds,evaluator.starts,evaluator.ends,env.control_dt,True)
        diagnosis=diagnose(trace,env.control_dt)
        assert diagnosis['successes']==sum(r['stop_success'] for r in summary['episodes'])
        result=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    step=agent.step,summary=summary,diagnosis=diagnosis)
        results[name]=result;write(out/f'{name}.json',result)
        terms=list(trace['terms'])
        trace['reward_terms']=np.stack([trace['terms'][k] for k in terms],axis=-1)
        del trace['terms']
        meta=dict(run=str(args.source_run),checkpoint=str(checkpoint),step=agent.step,
                  control_dt=env.control_dt,model_path=cfg.model_path,n_envs=32,seed=4102,
                  speeds=evaluator.speeds.tolist(),stop_ticks=evaluator.starts.tolist(),
                  restart_ticks=evaluator.ends.tolist(),term_names=terms,
                  reason_names=['term_low','term_tip','term_floor','term_ws','term_nan'],
                  ramp=True,standing=True,sampling='qpos/qvel before tick; diagnostics after tick')
        np.savez_compressed(out/f'{name}.npz',meta=np.array(json.dumps(meta)),**trace)
        trials=summary['episodes']
        chosen=next((i for i,r in enumerate(trials) if (r['stop_success'] if name=='best'
                     else r['survived_stop'] and not r['stop_success'])),None)
        video=dict(checkpoint=name,trial=chosen,selection='lowest-index successful trial' if name=='best'
                   else 'lowest-index surviving failure',seed=4102,playback_rate=.5)
        if chosen is not None:
            cmd=[sys.executable,str(Path(__file__).with_name('replay_stop_trace.py')),
                 '--trace',str(out/f'{name}.npz'),'--env',str(chosen),
                 '--video',str(out/f'{name}.mp4'),'--rate','0.5']
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (out/f'{name}_render.log').write_text(proc.stdout)
            video['render_exit_code']=proc.returncode
        videos.append(video)
        print('[diagnostic]',name,'success',diagnosis['successes'],'survived',diagnosis['survivors'],
              'removed-criterion counts',diagnosis['successful_if_removed'],flush=True)
    b,f=results['best']['summary'],results['final']['summary']
    gate=bool(b['success_fraction']>=.5 and
              (f['success_fraction']<b['success_fraction'] or
               f['longest_quiet_hold_mean_s']<b['longest_quiet_hold_mean_s']-.1))
    write(out/'comparison.json',dict(results=results,videos=videos,allow_pilot=gate,
          rule='best >=50% holds and final worse in holds or mean longest quiet by >0.1 s',
          caution='Removing criteria is a metric ablation on fixed trajectories, not proof of physical causation'))
    lines=['# STOP-04 paired checkpoint diagnosis','',f'Pilot gate passed: **{gate}**. Seed 4102, 32 near-stance trials.',
           '', '| Removed condition | Best successful trials | Final successful trials |','|---|---:|---:|']
    for k,v in results['best']['diagnosis']['successful_if_removed'].items():
        lines.append(f"| {k} | {v}/32 | {results['final']['diagnosis']['successful_if_removed'][k]}/32 |")
    lines+=['','Counterfactuals remove one measurement condition at a time; all other criteria',
            'and first-episode survival remain required. They do not alter the policy.',
            'Videos select the first successful best-policy trial and first surviving final-policy failure.',
            'Rendering status and checkpoint hashes are in comparison.json.','']
    (out/'DIAGNOSIS.md').write_text('\n'.join(lines))
    if not gate: raise SystemExit('Diagnosis gate did not reproduce the expected standing regression; PPO not started')


def assess(args):
    run=args.run
    cfg,env,agent=load_run(run,run/'best_stop.msgpack',n_envs=32)
    evaluator=StopEvaluator(agent,env)
    records=[]
    for label,checkpoint in [('source_best',args.source_run/'best_stop.msgpack'),('pilot_best',run/'best_stop.msgpack')]:
        agent.load(checkpoint,warm_start=True)
        for seed in (4102,5102,6102):
            result=evaluator.evaluate(agent,seed)
            records.append(dict(label=label,checkpoint=str(checkpoint),seed=seed,result=result))
    selected=json.loads((run/'best_stop_eval.json').read_text())
    ready=bool(selected['walking_retained'] and all(r['result']['success_fraction']>=.9
                  for r in records if r['label']=='pilot_best'))
    write(run/'multiseed.json',dict(records=records,ready_for_low_speed_review=ready,
          threshold='>=90% five-second holds on EACH of three seeds; standard walking retention passes',
          selection_seed=4102,held_out_seeds=[5102,6102],automatic_next_stage=False))
    lines=['# STOP-04 multi-seed assessment','',f'Low-speed stage review gate: **{ready}**.',
           'No transition training is launched automatically.','',
           '| Policy | Seed | Successful holds | Survived |','|---|---:|---:|---:|']
    for r in records:
        s=r['result'];lines.append(f"| {r['label']} | {r['seed']} | {round(32*s['success_fraction'])}/32 | {round(32*s['survive_fraction'])}/32 |")
    lines+=['','4102 is the checkpoint selection seed; 5102 and 6102 are held out from selection.',
            'Walking retention uses the standard ladder evaluation, not a multi-seed walking test.','']
    (run/'MULTISEED.md').write_text('\n'.join(lines))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['diagnose','assess'])
    ap.add_argument('--source-run',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('RLframework/results/stop04_diagnosis'))
    ap.add_argument('--run',type=Path,default=Path('RLframework/runs/dash_stop04_rehearsal_s0'))
    a=ap.parse_args();diagnostics(a) if a.stage=='diagnose' else assess(a)
