"""Read-only Git archaeology; pin Nathann's branch sources and saved run configs.

Run after `git fetch upstream`. Writes small provenance and historical run records;
does not check out, merge, execute old trainers, or alter training configurations.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'report/data/speed-discovery-reference'
GIT = ['git', '-c', 'safe.directory='+ROOT.as_posix()]
BRANCHES = ['main', 'balance-rl', 'loop-rate', 'feat/dynamic-parameter-identification',
            'research/ankle-spring-study', 'moteus_pi_controller_working', 'pendulum_sim2real']
CANDIDATES = ['RLframework/config.py', 'RLframework/env.py', 'BalanceRL/config.py',
              'sprint_runner/walk_mit/config.py', 'sprint_runner/walk_mit/env.py',
              'legacy_trainer_to_port/config.py', 'legacy_trainer_to_port/env.py',
              'training/config.py', 'training/env.py', 'RL/config.py']


def git(*args):
    return subprocess.check_output(GIT+list(args), cwd=ROOT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    branches = []
    for name in BRANCHES:
        ref = 'upstream/'+name
        commit = git('rev-parse', ref).decode().strip()
        paths = set(git('ls-tree', '-r', '--name-only', commit).decode().splitlines())
        sources = []
        for path in CANDIDATES:
            if path not in paths:
                continue
            blob = git('show', commit+':'+path)
            sources.append(dict(path=path, git_blob=git('rev-parse', commit+':'+path).decode().strip(),
                                sha256=hashlib.sha256(blob).hexdigest(),
                                url=f'https://github.com/nathmo/running_robot/blob/{commit}/{path}'))
        branches.append(dict(branch=ref, commit=commit, sources=sources))
    archived = git('rev-parse', 'f11eed8').decode().strip()
    evidence = []
    for run in ('sprint_m3_mit_s0', 'sprint_m6_lim2_s0', 'sprint_m6_lim2_s1'):
        for path, filename in [
            (f'walk_mit/runs/{run}/resolved_config.json', f'{run}_config.json'),
            (f'walk_mit/runs/{run}/curriculum.json', f'{run}_curriculum.json'),
            (f'walk_mit/monitor/gait_diag_{run}.json', f'{run}_diagnostics.json')]:
            blob = git('show', archived+':'+path)
            (OUT/filename).write_bytes(blob)
            evidence.append(dict(file=filename, commit=archived, path=path,
                                 sha256=hashlib.sha256(blob).hexdigest()))
    manifest = dict(note='Source/configuration evidence, not newly reproduced performance.',
                    archive_branch='dash-01-v2', archive_commit=git('rev-parse', 'c035774').decode().strip(),
                    work_branch='dash-01-v2-speed-discovery', branches=branches, evidence=evidence)
    (OUT/'source-audit.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Inventoried {len(branches)} upstream branches and preserved {len(evidence)} historical records.')


if __name__ == '__main__':
    main()
