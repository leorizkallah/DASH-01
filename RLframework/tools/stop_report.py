"""Regenerate a factual progress summary from persisted experiment metrics."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def generate(run):
    run = Path(run)
    stamp = datetime.now(ZoneInfo('Europe/Zurich')).isoformat(timespec='seconds')
    lines = ['# STOP measured progress', '', f'Updated: {stamp}', '', f'Run: `{run.name}`', '']
    warm = run / 'warmstart.json'
    if warm.exists():
        d = json.loads(warm.read_text())
        lines += [f"Source: `{d['source_run']}` at {d['source_step']:,} steps.",
                  f"Bundle SHA-256: `{d['sha256']}`.",
                  f"Actor restoration maximum absolute error: {d['actor_max_abs_error']:.3g}.", '']
    imitation = run / 'imitation.json'
    rehearsal = run / 'rehearsal.json'
    if rehearsal.exists():
        d = json.loads(rehearsal.read_text())
        lines += ['Continued actor rehearsal is enabled within the PPO objective.',
                  f"Dataset SHA-256: `{d['dataset_sha256']}`.",
                  f"Warm checkpoint SHA-256: `{d['checkpoint_sha256']}`.",
                  'Each minibatch adds 128 HOLD and 128 RUN training examples; validation examples are excluded.', '']
    if (run / 'guard_stop.json').exists():
        g = json.loads((run / 'guard_stop.json').read_text())
        lines += [f"Preservation guard stopped training at {g['step']:,} steps: {g['reason']}.", '']
    if (run / 'multiseed.json').exists():
        d = json.loads((run / 'multiseed.json').read_text())
        lines += [f"Three-seed low-speed review gate: **{d['ready_for_low_speed_review']}**.",
                  'See MULTISEED.md. No next training stage is launched automatically.', '']
    if imitation.exists():
        d = json.loads(imitation.read_text())
        lines += [f"Standing imitation stage: **{d['stage']}**.",
                  'Imitation updates and demonstration steps are separate from PPO steps.', '']
        if 'validation' in d:
            v = d['validation']
            lines += [f"Held-out HOLD action MSE: {v['hold_before']:.6f} -> {v['hold_after']:.6f}.",
                      f"Held-out RUN action MSE: {v['walk_after']:.6f}.", '']
        for name, passed in d.get('gates', {}).items():
            lines.append(f'- {name}: {passed}')
        lines.append('')
    progress = run / 'progress.csv'
    if progress.exists():
        with progress.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            lines += ['Latest PPO row (training reward is not stopping success):', '',
                      '| Metric | Value |', '|---|---|']
            for k in ['time/env_steps', 'rollout/ep_len_mean', 'rollout/reward_mean',
                      'train/approx_kl', 'train/std_mean', 'time/sps']:
                if k in last:
                    lines.append(f'| {k} | {last[k]} |')
            lines.append('')
    evaluations = run / 'stop_evaluations.jsonl'
    if evaluations.exists():
        records = [json.loads(s) for s in evaluations.read_text().splitlines() if s.strip()]
        protocol = records[-1].get('protocol', 'STOP-01')
        description = ('Near-stance starts; 7 s HOLD, including at least 5 s quiet with drift <=0.10 m.'
                       if protocol == 'STOP-02-standing' else
                       f"Ramped STOP, {records[-1].get('post_ramp_window_s',7.)-5:g} s settling allowance, 5 s quiet hold; approach speeds listed below."
                       if protocol == 'STOP-02-ramp' else
                       'Four approach speeds; 8 s STOP, then 3 s restart.')
        lines += [f'Nominal MJX benchmark: {protocol}. {description}', '',
                  '| Stage | PPO steps | Stop success | Survived STOP | Restart success | Walking retained | Reward-floor fraction |',
                  '|---|---:|---:|---:|---:|:---:|---:|']
        for r in records:
            floor = r.get('stop_reward_floor_fraction')
            floor_text = f'{floor:.1%}' if floor is not None else 'n/a'
            restart_text = f"{r['restart_fraction']:.1%}" if r.get('restart_evaluated',True) else 'n/a'
            lines.append(f"| {r.get('stage', 'source' if r['step'] == 0 else 'ppo')} | {r['step']:,} | {r['success_fraction']:.1%} | {r['survive_fraction']:.1%} | "
                         f"{restart_text} | {r.get('walking_retained')} | {floor_text} |")
        lines += ['', 'Per-speed results from the latest evaluation:', '',
                  '| Command (m/s) | Successful stops | Survived STOP |', '|---:|---:|---:|']
        for r in records[-1]['per_speed']:
            lines.append(f"| {r['command_mps']} | {r['successes']}/{r['n']} | {r['survivors']}/{r['n']} |")
        if 'longest_quiet_hold_mean_s' in records[-1]:
            lines += ['', f"Mean longest quiet hold: {records[-1]['longest_quiet_hold_mean_s']:.3f} s."]
        if 'standing' in records[-1]:
            r=records[-1]
            lines += ['', f"Separate near-stance hold success: {r['standing']['success_fraction']:.1%}.",
                      f"Standing retained: {r.get('standing_retained', True)}.",
                      f"Moving approach qualification: {r['approach_fraction']:.1%}."]
    else:
        lines += ['No completed transition evaluation has been recorded yet.']
    lines += ['', 'These are simulation results. Hardware stopping has not been validated.', '']
    (run / 'STOP_PROGRESS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    generate(parser.parse_args().run)
