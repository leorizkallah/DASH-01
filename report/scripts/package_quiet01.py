"""Freeze quiet-pose experiment and each model's own STAND-04 best parent."""
import hashlib
import argparse
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True)
    ap.add_argument('--report',default='report/quiet01-v2-pose-tracking.md')
    args=ap.parse_args()
    import re
    assert re.fullmatch(r'quiet01_\d{8}_v\d+',args.name), 'Invalid experiment directory'
    OUT=ROOT/'docs/experiments'/args.name
    if (OUT/'source-manifest.json').exists() or (OUT/'source.tar.gz').exists():
        raise FileExistsError('Refusing to overwrite a previously frozen experiment')
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'docs/experiments/stand03_20261109_v1/runtime-manifest.json').read_text())
    paths={name:ROOT/name for name in old['files']}
    for p in (ROOT/'BalanceRL').rglob('*'):
        if '__pycache__' not in p.parts and p.suffix in ('.py','.sbatch'):
            paths[p.relative_to(ROOT).as_posix()]=p
    for pat in ('quiet_*.json','quiet_resets_*.npz'):
        for p in (ROOT/'BalanceRL/model').glob(pat): paths[p.relative_to(ROOT).as_posix()]=p
    for name in (args.report,'report/scripts/package_quiet01.py',
                 'report/scripts/check_quiet_tracking_trace.py','report/scripts/build_quiet01_assets.py'):
        paths[name]=ROOT/name
    parents={}
    for model in ('v1','v2','xshape'):
        folder=ROOT/'report/data/standing04/results'/model
        for ext in ('json','msgpack'): paths[f'parents/{model}/best.{ext}']=folder/f'best.{ext}'
        side=json.loads((folder/'best.json').read_text())
        parents[model]=dict(step=side['step'],weights_sha256=hashlib.sha256((folder/'best.msgpack').read_bytes()).hexdigest())
    manifest=dict(experiment=args.name,date='2026-11-22',additional_steps_per_model=100_000_000,
        protocol='quiet-pose-v2: latched timely hold; independent20s retention; matched-seed numerical/physical preflight',
        seed=2,parents=parents,initialization='Own STAND-04 best weights + normalizer; fresh optimizers/curriculum/counters',
        files={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(paths.items())})
    receipt=OUT/'source-manifest.json'
    receipt.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=OUT/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(receipt,arcname='source-manifest.json')
        for name,p in sorted(paths.items()): tar.add(p,arcname=name)
    summary=dict(files=len(paths),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (OUT/'packaging.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__': main()
