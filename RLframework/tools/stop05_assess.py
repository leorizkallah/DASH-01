"""Automatically assess and record the selected low-speed moving-stop keeper."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate import load_run
from stop_eval import StopEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--replay-only', action='store_true')
    args = ap.parse_args()
    run = args.run
    if args.replay_only:
        record_replay(run)
        return
    keeper=run/'best_stop.msgpack'
    cfg,env,agent=load_run(run,keeper,n_envs=32)
    motion=StopEvaluator(agent,env);standing=StopEvaluator(agent,env,standing=True)
    records=[]
    for seed in (4102,5102,6102):
        records.append(dict(seed=seed,motion=motion.evaluate(agent,seed),standing=standing.evaluate(agent,seed)))
    chosen=json.loads((run/'best_stop_eval.json').read_text())
    payload=dict(checkpoint=str(keeper),step=agent.step,records=records,
        selection_seed=4102,held_out_seeds=[5102,6102],
        walking_retained_at_selection=chosen['walking_retained'],
        automatic_next_stage=False)
    (run/'motion_assessment.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=[f'# Low-speed motion assessment: {run.name}','',f'Selected checkpoint: {agent.step:,} new PPO steps.',
           '', '| Seed | Qualified moving approaches | Successful stops | Restart successes | Standing holds |',
           '|---:|---:|---:|---:|---:|']
    for r in records:
        m,s=r['motion'],r['standing']
        lines.append(f"| {r['seed']} | {round(32*m['approach_fraction'])}/32 | {round(32*m['success_fraction'])}/32 | {round(32*m['restart_fraction'])}/32 | {round(32*s['success_fraction'])}/32 |")
    lines+=['','Per-speed and per-trial results are in motion_assessment.json.',
            '4102 is the keeper-selection seed; 5102 and 6102 are held out from selection.',
            'Walking retention uses the standard ladder, not a three-seed walking test.',
            'No automatic increase in speed or additional training is scheduled.','']
    (run/'MOTION_ASSESSMENT.md').write_text('\n'.join(lines))
    # Release evaluation GPU allocations before compiling the recording graph.
    os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),
                            '--run',str(run),'--replay-only'])


def record_replay(run):
    keeper=run/'best_stop.msgpack'
    # Video recording is a separate graph; retain its own measured result and trial selection.
    out=run/'replay';out.mkdir(exist_ok=True)
    trace=out/'motion.npz'
    tools=Path(__file__).parent
    proc=subprocess.run([sys.executable,str(tools/'trace_stop_mjx.py'),'--run',str(run),
        '--checkpoint',str(keeper),'--out',str(trace)],capture_output=True,text=True)
    (out/'record.log').write_text(proc.stdout+proc.stderr)
    selection=dict(record_exit_code=proc.returncode)
    if proc.returncode==0:
        result=json.loads(trace.with_suffix('.json').read_text())
        trials=result['summary']['episodes']
        trial=next((i for i,r in enumerate(trials) if r['stop_success']),None)
        criterion='lowest-index successful moving stop'
        if trial is None:
            trial=next((i for i,r in enumerate(trials) if r['approach_ok']),0)
            criterion='lowest-index qualifying approach; trial zero if none qualify; NOT a success selection'
        selection.update(trial=trial,criterion=criterion,recorded_successes=sum(r['stop_success'] for r in trials),
                         playback_rate=.5,seed=4102)
        proc=subprocess.run([sys.executable,str(tools/'replay_stop_trace.py'),'--trace',str(trace),
            '--env',str(trial),'--video',str(out/'motion.mp4'),'--rate','0.5'],capture_output=True,text=True)
        selection['render_exit_code']=proc.returncode
        (out/'render.log').write_text(proc.stdout+proc.stderr)
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')


if __name__ == '__main__':
    main()
