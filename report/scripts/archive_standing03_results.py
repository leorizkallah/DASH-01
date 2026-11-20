"""Run in the frozen SCITAS experiment root; archive report inputs only."""
from pathlib import Path
import json
import subprocess
import tarfile

root = Path.cwd()
receipt = root/'report_scheduler_20261110.txt'
receipt.write_text(subprocess.check_output([
    'sacct', '-j', '3246816,3246822', '--parsable2',
    '--format=JobID,JobName,State,Start,End,Elapsed,ExitCode,NodeList'
], text=True))
with tarfile.open(root/'report_assets_20261110.tar.gz', 'w:gz') as tar:
    tar.add(receipt, arcname='scheduler.txt')
    tar.add(root/'source-manifest.json', arcname='source-manifest.json')
    for model in ('v1', 'v2', 'xshape'):
        run = root/'BalanceRL/runs'/f'stand03_{model}_s0'
        for name in ('progress.jsonl', 'standing_evaluations.jsonl', 'best.json', 'best.msgpack',
                     'final.json', 'final_eval.json', 'completion.json', 'config.json'):
            tar.add(run/name, arcname=f'{model}/{name}')
        tar.add(root/f'validation/preflight_{model}.json', arcname=f'{model}/preflight.json')
    for i in range(3):
        tar.add(root/f'stand03-3246822_{i}.out', arcname=f'training_{i}.out')
print(json.dumps({'archive': 'report_assets_20261110.tar.gz',
                  'bytes': (root/'report_assets_20261110.tar.gz').stat().st_size}))
