"""Bounded completion pipeline for screen3264033, with no training/resubmission."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'report/data/quiet_push01'
REMOTE='/home/rizkalla/running_robot/reviews/quiet_push01_20261123_v1';HOST='rizkalla@izar.hpc.epfl.ch'
FLAGS=['-i','C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15']


def call(args,timeout=90,**kw):
    p=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=timeout,**kw)
    if p.returncode:raise RuntimeError(p.stderr[-1500:]+' '+p.stdout[-1500:])
    return p.stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-hours',type=float,default=2.);ap.add_argument('--interval',type=int,default=60)
    a=ap.parse_args();BASE.mkdir(parents=True,exist_ok=True);path=BASE/'pipeline.json'
    old=json.loads(path.read_text()) if path.exists() else {}
    if old.get('status')=='complete':return
    state=dict(job=3264033,status='waiting',completed_models=old.get('completed_models',[]))
    def save():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print(json.dumps(state),flush=True)
    code='''
import json,subprocess
from pathlib import Path
p=Path(REMOTE)
ready=[m for m in ('v1','v2','xshape') if (p/('results/'+m+'/metadata.json')).exists()]
account=subprocess.check_output(['sacct','-j','3264033','--format=JobID,State,ExitCode','-P'],text=True)
print(json.dumps(dict(ready=ready,accounting=account)))
'''.replace('REMOTE',repr(REMOTE))
    deadline=time.monotonic()+a.max_hours*3600;errors=0;save()
    call([sys.executable,'report/scripts/build_quiet_push_gallery.py'])
    while time.monotonic()<deadline:
        try:
            r=json.loads(call(['ssh',*FLAGS,HOST,'python3','-'],input=code,timeout=45));state['accounting']=r['accounting']
            for model in r['ready']:
                if model in state['completed_models']:continue
                state['status']='processing_'+model;save();folder=BASE/model;folder.mkdir(parents=True,exist_ok=True)
                call(['scp',*FLAGS,HOST+':'+REMOTE+f'/results/{model}/trace.npz',HOST+':'+REMOTE+f'/results/{model}/metadata.json',str(folder)],timeout=180)
                subprocess.run([sys.executable,'report/scripts/analyze_quiet_push_screen.py'],cwd=ROOT,check=True,timeout=180)
                call([sys.executable,'report/scripts/build_quiet_push_gallery.py'])
                subprocess.run([sys.executable,'report/scripts/render_quiet_push_screen.py','--models',model],cwd=ROOT,check=True,timeout=900)
                state['completed_models'].append(model);save();call([sys.executable,'report/scripts/build_quiet_push_gallery.py'])
            if len(state['completed_models'])==3:
                state['status']='complete';state.pop('error',None);state.pop('consecutive_errors',None);save();return
            if any(word in r['accounting'] for word in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL')):
                state['status']='screen_failed';save();return
            state['status']='waiting';errors=0;save()
        except Exception as exc:
            errors+=1;state.update(status='retrying',error=str(exc),consecutive_errors=errors);save()
            if errors>=5:state['status']='stopped_after_errors';save();return
        time.sleep(max(30,a.interval))
    state['status']='fetch_timeout';save()


if __name__=='__main__':main()
