"""Bounded fetch/render pipeline for replay3263991; never trains or resubmits."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'report/data/quiet01v2/videos'
REMOTE='/home/rizkalla/running_robot/reviews/quiet01v2_videos_20261123'
HOST='rizkalla@izar.hpc.epfl.ch'
FLAGS=['-i','C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519','-o','IdentitiesOnly=yes',
       '-o','BatchMode=yes','-o','ConnectTimeout=15']


def call(args,timeout=90,**kw):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=timeout,**kw)
    if p.returncode:raise RuntimeError(str(args[0])+': '+p.stderr[-1200:]+' '+p.stdout[-1200:])
    return p.stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-hours',type=float,default=24.);ap.add_argument('--interval',type=int,default=120)
    ap.add_argument('--job',type=int,default=3263991)
    a=ap.parse_args();BASE.mkdir(parents=True,exist_ok=True)
    path=BASE/'pipeline.json'
    old=json.loads(path.read_text()) if path.exists() else {}
    if old.get('status')=='complete':return
    state=dict(job=a.job,status='waiting',completed_models=old.get('completed_models',[]),
        requires='Local computer must remain awake/connected for automatic fetch/render.',max_hours=a.max_hours)
    def save():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print(json.dumps(state),flush=True)
    def gallery():call([sys.executable,'report/scripts/build_quiet01v2_gallery.py'])
    deadline=time.monotonic()+a.max_hours*3600;errors=0
    remote_code='''
import json,subprocess
from pathlib import Path
p=Path(REMOTE)
ready=[m for m in ('v1','v2','xshape') if (p/('results/'+m+'/replay.json')).exists()]
account=subprocess.check_output(['sacct','-j','JOB_ID','--format=JobID,State,ExitCode','-P'],text=True)
print(json.dumps(dict(ready=ready,accounting=account)))
'''.replace('REMOTE',repr(REMOTE)).replace('JOB_ID',str(a.job))
    save();gallery()
    while time.monotonic()<deadline:
        try:
            remote=json.loads(call(['ssh',*FLAGS,HOST,'python3','-'],input=remote_code,timeout=45))
            state['accounting']=remote['accounting']
            for model in remote['ready']:
                if model in state['completed_models']:continue
                state['status']='rendering_'+model;save()
                folder=BASE/model;folder.mkdir(parents=True,exist_ok=True)
                call(['scp',*FLAGS,HOST+':'+REMOTE+f'/results/{model}/trace.npz',
                      HOST+':'+REMOTE+f'/results/{model}/replay.json',str(folder)],timeout=180)
                meta=json.loads((folder/'replay.json').read_text())
                assert meta['selected_trials']==[0,48,64,80] and meta['seed_source_batch_size']==96 and meta['batch_size']==4
                # Render with direct output so progress is visible in worker.out.log.
                subprocess.run([sys.executable,'report/scripts/render_quiet01v2_videos.py','--models',model],
                               cwd=ROOT,check=True,timeout=900)
                state['completed_models'].append(model);save();gallery()
            if len(state['completed_models'])==3:
                state['status']='complete';save();gallery();return
            if any(word in remote['accounting'] for word in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL')):
                state['status']='replay_failed';save();return
            errors=0;state['status']='waiting';save()
        except Exception as exc:
            errors+=1;state.update(status='retrying',error=str(exc),consecutive_errors=errors);save()
            if errors>=5:state['status']='stopped_after_errors';save();return
        time.sleep(max(30,a.interval))
    state['status']='fetch_timeout';save()


if __name__=='__main__':main()
