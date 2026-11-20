"""Archive completed STAND-04 evidence in its isolated SCITAS root."""
import json
from pathlib import Path
import subprocess
import tarfile

root = Path.cwd()
receipt = root/'report_scheduler_20261111.txt'
receipt.write_text(subprocess.check_output(['sacct', '-j', '3247598,3247599', '--parsable2',
    '--format=JobID,JobName,State,Start,End,Elapsed,ExitCode,NodeList'], text=True))
with tarfile.open(root/'report_assets_20261111.tar.gz', 'w:gz') as tar:
    tar.add(receipt, arcname='scheduler.txt')
    tar.add(root/'source-manifest.json', arcname='source-manifest.json')
    for model in ('v1', 'v2', 'xshape'):
        run = root/'BalanceRL/runs'/f'stand04_{model}_s1'
        for name in ('progress.jsonl', 'recovery_evaluations.jsonl', 'recovery_curriculum_probes.jsonl',
                     'best.json', 'best.msgpack', 'baseline_eval.json', 'final.json', 'final_eval.json',
                     'completion.json', 'config.json', 'parent.json'):
            tar.add(run/name, arcname=f'{model}/{name}')
        tar.add(root/f'validation/recovery_{model}.json', arcname=f'{model}/preflight.json')
    for i in range(3):
        tar.add(root/f'stand04-3247599_{i}.out', arcname=f'training_{i}.out')
print(json.dumps({'archive':'report_assets_20261111.tar.gz', 'bytes':(root/'report_assets_20261111.tar.gz').stat().st_size}))
