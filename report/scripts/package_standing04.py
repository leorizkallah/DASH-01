"""Freeze STAND-04 source, existing model assets and model-specific STAND-03 parents."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/experiments/stand04_20261110_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT/'docs/experiments/stand03_20261109_v1/runtime-manifest.json').read_text())
    paths = {n:ROOT/n for n in old['files']}
    for p in (ROOT/'BalanceRL').rglob('*.py'):
        if '__pycache__' not in p.parts:
            paths[p.relative_to(ROOT).as_posix()] = p
    for name in ('BalanceRL/slurm/izar_recovery_standing.sbatch', 'report/standing04-plan.md'):
        paths[name] = ROOT/name
    parents = {}
    for m in ('v1', 'v2', 'xshape'):
        folder = ROOT/'report/data/standing03/results'/m
        for ext in ('json', 'msgpack'):
            paths[f'parents/{m}/best.{ext}'] = folder/f'best.{ext}'
        b = json.loads((folder/'best.json').read_text())
        parents[m] = dict(step=b['step'], weights_sha256=hashlib.sha256((folder/'best.msgpack').read_bytes()).hexdigest(),
                          archived_task_score=b['eval']['score'])
    manifest = dict(experiment=OUT.name, date='2026-11-10',
        initialization='model-specific best weights + normalizer; fresh optimizers/counters/curriculum',
        additional_steps_per_model=200_000_000, seed=1, parents=parents,
        files={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sorted(paths.items())})
    receipt = OUT/'source-manifest.json'
    receipt.write_text(json.dumps(manifest, indent=2)+'\n')
    archive = OUT/'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(receipt, arcname='source-manifest.json')
        for n,p in sorted(paths.items()):
            tar.add(p, arcname=n)
    packaging = dict(files=len(paths), bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (OUT/'packaging.json').write_text(json.dumps(packaging, indent=2)+'\n')
    print(json.dumps(packaging))


if __name__ == '__main__':
    main()
