"""Freeze a gait snapshot using a previous manifest plus explicit changed files.

Only reads selected sources/assets; never copies runs, videos or credentials.
Refuses overwriting an existing frozen archive or manifest.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--base-manifest', type=Path, required=True)
    ap.add_argument('--preset', required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--parent')
    source.add_argument('--fresh', action='store_true')
    ap.add_argument('--parent-sha256')
    ap.add_argument('--include', nargs='+', required=True)
    args = ap.parse_args()
    if not args.snapshot.replace('_', '').isalnum():
        ap.error('snapshot must contain letters, digits and underscores only')
    if args.parent and (not args.parent_sha256 or len(args.parent_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.parent_sha256)):
        ap.error('parent hash must be a lowercase SHA-256 digest')
    if args.fresh and args.parent_sha256:
        ap.error('fresh initialization has no parent checkpoint hash')
    root = Path(__file__).resolve().parents[2]
    out = root/'docs/experiments'/args.snapshot
    archive, manifest_path = out/'source.tar.gz', out/'source-manifest.json'
    if archive.exists() or manifest_path.exists():
        ap.error('snapshot already frozen; choose a new snapshot name')
    base = json.loads(args.base_manifest.read_text())
    files = {p for p in base['files'] if not p.endswith('.sbatch') and not p.startswith('docs/')}
    files.update(args.include)
    files.add('RLframework/tools/package_gait.py')
    for name in files:
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            ap.error(f'not a workspace file: {name}')
    manifest = dict(experiment=args.snapshot, parent_design=base['experiment'], preset=args.preset,
                    initialization='fresh' if args.fresh else 'warm_start',
                    source_checkpoint=args.parent, source_checkpoint_sha256=args.parent_sha256,
                    files={n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sorted(files)})
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    with tarfile.open(archive, 'w:gz') as tar:
        for name in sorted(files):
            tar.add(root/name, arcname=name)
        tar.add(manifest_path, arcname='source-manifest.json')
    bundle = dict(archive=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  files=len(files), bytes=archive.stat().st_size)
    (out/'bundle.json').write_text(json.dumps(bundle, indent=2)+'\n')
    print(json.dumps(bundle))


if __name__ == '__main__':
    main()
