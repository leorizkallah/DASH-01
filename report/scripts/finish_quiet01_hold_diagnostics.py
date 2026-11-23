"""Bounded local fetcher for diagnostic array3263170; never trains or resubmits.

Polls every120s for up to8h, archives results and builds the detailed report section.
"""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'report/data/quiet01/hold-diagnostics'
REMOTE='/home/rizkalla/running_robot/reviews/quiet01_holds_20261122'
HOST='rizkalla@izar.hpc.epfl.ch'
KEY='C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519'
FLAGS=['-i',KEY,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15']


def state(status,**extra):
    OUT.mkdir(parents=True,exist_ok=True)
    d=dict(status=status,job='3263170',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    (OUT/'pipeline.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(d),flush=True)


def main():
    path=OUT/'pipeline.json'
    if path.exists() and json.loads(path.read_text())['status']=='complete': return
    deadline=time.monotonic()+8*3600
    remote_code='''
import json,subprocess,tarfile
from pathlib import Path
p=Path(REMOTE)
ready=all((p/('results/'+m+'/summary.json')).exists() for m in ('v1','v2','xshape'))
account=subprocess.check_output(['sacct','-j','3263170','--format=JobID,State','-P'],text=True)
if ready:
    archive=p/'diagnostic-results.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:
            tar.add(p/'results',arcname='results')
            tar.add(p/'submission.json',arcname='submission.json')
            for log in p.glob('quiet01-diag-3263170_*.out'): tar.add(log,arcname=log.name)
print(json.dumps(dict(ready=ready,accounting=account)))
'''.replace('REMOTE',repr(REMOTE))
    while time.monotonic()<deadline:
        try:
            p=subprocess.run(['ssh',*FLAGS,HOST,'python3','-'],input=remote_code,text=True,capture_output=True,timeout=45,check=True)
            d=json.loads(p.stdout)
            if d['ready']:
                state('fetching',accounting=d['accounting'])
                archive=OUT/'diagnostic-results.tar.gz'
                subprocess.run(['scp',*FLAGS,HOST+':'+REMOTE+'/diagnostic-results.tar.gz',str(archive)],check=True,timeout=180)
                with tarfile.open(archive) as tar:
                    for member in tar.getmembers():
                        assert OUT.resolve() in (OUT/member.name).resolve().parents
                        assert member.isfile() or member.isdir()
                    tar.extractall(OUT,filter='data')
                subprocess.run([sys.executable,str(ROOT/'report/scripts/analyze_quiet01_holds.py')],cwd=ROOT,check=True,timeout=120)
                state('complete',accounting=d['accounting'],report='report/quiet01-hold-diagnosis.md')
                return
            failure=any(word in d['accounting'] for word in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL'))
            if failure:
                state('diagnostic_job_failed',accounting=d['accounting'])
                return
            state('waiting',accounting=d['accounting'])
        except Exception as exc:
            state('retrying',error=str(exc))
        time.sleep(120)
    state('fetch_timeout',note='No training/resubmission occurred. Remote diagnostic jobs are unaffected.')


if __name__=='__main__': main()
