"""Archive completed QUIET-01v2 outputs; run in its frozen SCITAS experiment root."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    root=Path.cwd()
    assert root.name=='quiet01_20261122_v2'
    receipt=root/'validation/report_scheduler_20261123.txt'
    receipt.write_text(subprocess.check_output(['sacct','-j','3263184,3263185','-P',
        '--format=JobID,JobName,State,Start,End,Elapsed,ExitCode,NodeList'],text=True))
    archive=root/'report_results_20261123.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ('source-manifest.json','submission.json'):
            tar.add(root/name,arcname=name)
        tar.add(receipt,arcname='scheduler.txt')
        for i,model in enumerate(('v1','v2','xshape')):
            run=root/'BalanceRL/runs'/f'quiet01v2_{model}_s2'
            for name in ('config.json','parent.json','completion.json','progress.jsonl',
                         'baseline_eval.json','final_eval.json','baseline.json',
                         'best.json','best.msgpack','final.json','final.msgpack',
                         'quiet_pose_evaluations.jsonl','quiet_pose_probes.jsonl','quiet_pose_retention.jsonl'):
                tar.add(run/name,arcname=f'{model}/{name}')
            tar.add(root/f'validation/quiet_{model}.json',arcname=f'{model}/preflight.json')
            tar.add(root/f'quiet01v2-3263185_{i}.out',arcname=f'training_{model}.out')
    print(json.dumps(dict(archive=archive.name,bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))


if __name__=='__main__':main()
