"""Freeze the source/assets for a new STOP-02 snapshot; never overwrite a package."""
import argparse
import hashlib
import json
import tarfile
from datetime import date
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',required=True)
    ap.add_argument('--experiment',default='STOP-02',choices=['STOP-02','STOP-03','STOP-04','STOP-05','STOP-06'])
    args=ap.parse_args()
    if not args.snapshot.replace('_','').isalnum():
        ap.error('snapshot must contain only letters, numbers and underscores')
    root=Path(__file__).resolve().parents[2]
    files=set(json.loads((root/'docs/stop01b-source-manifest.json').read_text())['files'])
    for pattern in ('RLframework/*.py','RLframework/tests/*.py','controller/deploy/*.py','RLframework/slurm/*stop*'):
        files.update(p.relative_to(root).as_posix() for p in root.glob(pattern) if p.is_file())
    files.update('RLframework/tools/'+n for n in ('check_stop02.py','replay_stop_trace.py','trace_stop_mjx.py','package_stop02.py','stop_report.py'))
    files.update(['RLframework/results/stop01_replay/dash_stop01b_final.npz',
                  'docs/stop02-training-plan.md','docs/stop02-experiment-report.md'])
    if args.experiment == 'STOP-03':
        files.add('docs/stop03-experiment-report.md')
    if args.experiment == 'STOP-04':
        files.update(['docs/stop04-experiment-report.md','RLframework/tools/stop04_pipeline.py'])
    if args.experiment == 'STOP-05':
        files.update(['docs/stop05-experiment-report.md','RLframework/tools/stop05_assess.py'])
    if args.experiment == 'STOP-06':
        files.update(['docs/stop06-experiment-report.md','RLframework/tools/stop05_assess.py',
                      'RLframework/tools/stop06_report.py'])
    out=root/'docs/experiments'/args.snapshot
    out.mkdir(parents=True,exist_ok=True)
    archive=out/'source.tar.gz'
    if archive.exists():
        raise SystemExit('Package already exists; use a new snapshot name')
    manifest=dict(experiment=args.experiment,snapshot=args.snapshot,created_date=date.today().isoformat(),
        files={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(files)})
    (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files): tar.add(root/p,arcname=p)
        tar.add(out/'source-manifest.json',arcname='source-manifest.json')
    print(archive)
    print('SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__=='__main__': main()
