"""Bounded unattended fetch/render after SCITAS replay; no training or resubmission."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'report'
STATUS = REPORT/'data/standing04/video-pipeline.json'
REMOTE = '/home/rizkalla/running_robot/experiments/stand04_20261110_v1'
HOST = 'rizkalla@izar.hpc.epfl.ch'
SSH = ['-i','C:/Users/LeoPaul/.ssh/dash_scitas_training_ed25519',
       '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15']
MODELS = ('v1','v2','xshape')


def call(args, timeout=90):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exit {result.returncode}: {result.stderr[-2000:]} {result.stdout[-1000:]}')
    return result.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job',type=int,default=3248509)
    ap.add_argument('--max-hours',type=float,default=8.)
    ap.add_argument('--interval',type=int,default=120)
    args = ap.parse_args()
    start = time.monotonic()
    state = dict(job=args.job,status='waiting_for_scitas',completed_models=[],
                 expected_start_cet='11 November, around 19:38 CET (scheduler estimate, may change)',
                 max_hours=args.max_hours,requires='This computer must stay awake and connected for local automatic fetching/rendering.')
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        STATUS.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(state),flush=True)
    save()
    errors = 0
    while time.monotonic()-start < args.max_hours*3600:
        try:
            command = ('cd '+REMOTE+'; for m in v1 v2 xshape; do '
                'if test -s report_replays_20261111/$m/replay.json; then echo $m; fi; done; '
                f'sacct -j {args.job} --parsable2 --noheader --format=JobID,State,ExitCode')
            output = call(['ssh',*SSH,HOST,command])
            ready = [m for m in MODELS if m in output.splitlines()]
            state['scheduler'] = [s for s in output.splitlines() if s.startswith(str(args.job))]
            for m in ready:
                if m in state['completed_models']:
                    continue
                state['status'] = 'rendering_'+m
                save()
                folder = REPORT/'data/standing04/results'/m
                call(['scp',*SSH,HOST+':'+REMOTE+f'/report_replays_20261111/{m}/trace.npz',
                      HOST+':'+REMOTE+f'/report_replays_20261111/{m}/replay.json',str(folder)],timeout=180)
                meta = json.loads((folder/'replay.json').read_text())
                assert meta['model']==m and meta['selected_trials']==[0,48]
                call([sys.executable,'report/scripts/render_standing03_best.py',
                      '--campaign','standing04','--replay-job',str(args.job),'--models',m],timeout=240)
                state['completed_models'].append(m)
                state['status'] = 'waiting_for_scitas'
                save()
                call([sys.executable,'report/scripts/build_standing04_assets.py'],timeout=180)
            if len(state['completed_models'])==3:
                state['status']='complete'
                save()
                call([sys.executable,'report/scripts/build_standing04_assets.py'],timeout=180)
                return
            failed=[]
            for i,m in enumerate(MODELS):
                prefix=f'{args.job}_{i}|'
                failed.extend(s for s in state['scheduler'] if s.startswith(prefix)
                    and any(v in s for v in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL')))
            if failed:
                state.update(status='replay_failed',error=failed)
                save()
                call([sys.executable,'report/scripts/build_standing04_assets.py'],timeout=180)
                return
            errors=0
            save()
        except Exception as exc:
            errors += 1
            state.update(status='retrying',error=str(exc),consecutive_errors=errors)
            save()
            if errors>=5:
                state['status']='stopped_after_errors'
                save()
                return
        time.sleep(max(30,args.interval))
    state['status']='stopped_at_time_limit'
    save()


if __name__=='__main__':
    main()
