"""Rebuild GAIT-11 learning figures, replay gallery and asset provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'report'
DATA = ROOT/'docs/experiments/gait11_20261019_v1'


def main():
    records = [json.loads(s) for s in (DATA/'gait_evaluations.jsonl').read_text().splitlines()]
    x = np.array([r['step']/1e6 for r in records])
    mean = lambda key: np.array([np.asarray(r['per_trial'][key], float).mean(axis=0) for r in records])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes[0, 0].plot(x, [100*r['survival_fraction'] for r in records], label='Survived')
    axes[0, 0].plot(x, [100*r['qualified_fraction'] for r in records], label='Qualified')
    axes[0, 0].set(ylabel='Trials (%)', title='0.15 m/s: survival is not gait quality', ylim=(-3, 103))
    axes[0, 0].legend()
    incomplete = 100*mean('incomplete_fraction')
    incomplete[mean('attempt').sum(axis=1) == 0] = np.nan
    axes[0, 1].plot(x, incomplete)
    axes[0, 1].axhline(10, color='grey', ls='--', label='Acceptance limit')
    axes[0, 1].set(ylabel='Incomplete attempts (%)', title='Extra attempts persist')
    axes[0, 1].legend()
    for i, (color, label) in enumerate([('tab:blue', 'Left'), ('tab:orange', 'Right')]):
        axes[1, 0].plot(x, mean('repeated_attempt')[:, i], color=color, label=label)
        axes[1, 1].plot(x, 100*mean('clear_fraction')[:, i], color=color, label=label)
    axes[1, 0].set(ylabel='Repeated attempts', title='All meaningful attempts, after 5 s')
    axes[1, 1].axhline(8, color='grey', ls='--', label='Required per foot')
    axes[1, 1].set(ylabel='Whole-sole clear time (%)', title='Clearance above 15 mm, after 5 s')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.set_xlabel('Environment steps (millions)')
        ax.legend()
    fig.suptitle('GAIT-11: 12 nominal greedy trials per checkpoint, fixed evaluation seed')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(REPORT/f'figures/gait11_progress.{ext}', dpi=150)
    plt.close(fig)
    traces = [(DATA/f'gait11_trace_{name}.json') for name in ('best_candidate', 'final')]
    if not all(p.exists() for p in traces):
        print('Progress figures saved; replay metadata not yet downloaded.')
        return
    meta = [json.loads(p.read_text()) for p in traces]
    sections = []
    for target in (.15, .10, .20):
        for label, prefix, m in zip(('Saved diagnostic candidate', 'Final policy'), ('candidate', 'final'), meta):
            trial = next(t for t in m['trials'] if abs(t['target']-target) < .001)
            path = f'videos/gait11_{prefix}_{target:.2f}mps.mp4'
            assert (REPORT/path).is_file(), path
            outcome = 'fell' if trial['fell'] else 'completed'
            speed = trial['mean_speed_after_5s']
            speed_text = f'{speed:.3f} m/s' if speed is not None else 'no post-5-second window'
            sections.append(f'''<section><h2>{label}: {target:.2f} m/s</h2>
<video controls playsinline preload="metadata" src="{path}"></video>
<p><a href="{path}">Open / download video</a> · {m['step']/1e6:.2f}M steps.
Replay {outcome} at {trial['seconds']:.2f} s; mean forward speed after 5 s: {speed_text}.</p>
<p>Valid steps L/R: {trial['valid_steps_after_5s']}; all attempts: {trial['attempt_after_5s']};
repeated attempts: {trial['repeated_attempt_after_5s']} (after 5 s).</p></section>''')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GAIT-11 walking videos</title><style>
body{font:17px/1.5 system-ui,sans-serif;background:#111925;color:#eef3fa;max-width:1280px;margin:32px auto;padding:0 20px}
a{color:#8dccff}video,img{width:100%;background:#000;border-radius:8px}section{margin:36px 0}
button{font:inherit;background:#24364d;color:white;border:1px solid #7290b5;border-radius:5px;padding:6px 14px;cursor:pointer}
</style><h1>GAIT-11: judge the walking, not just survival</h1>
<p>Training completed: 65.65M steps in 7h52m. Start with the two <strong>0.15 m/s</strong>
clips below. Left foot is blue; right is orange. Both views show the same instant.</p>
<p>Watch the extra right-foot lifts, foot clearance and forward sliding, and whether
forward motion stays fluid during foot exchanges. Equal accepted-step counts are
partly enforced by the supervisor and do not prove clean walking.</p>
<p>Playback: <button onclick="rate(1)">Normal</button> <button onclick="rate(.5)">Half speed</button>
<button onclick="rate(.25)">Quarter speed</button> <span id="rate">1x</span></p>
<p>Separate final 12-trial evaluation at 0.15 m/s: all survived, 24.44% incomplete
attempts, zero qualified gaits. Candidate is the saved diagnostic ranking winner;
the final has fewer incomplete attempts and lower slip. Neither is certified.</p>
''' + '\n'.join(sections) + '''
<h2>Training progress</h2><img src="figures/gait11_progress.png" alt="Survival, incomplete attempts, repeated attempts and foot clearance during training">
<p>Fresh nominal greedy replays, seed 1000; all three requested speeds shown for
both checkpoints, without selecting successful trials. Clips stop at first
termination or 25 seconds; no auto-resets. Evaluation uses seed 3103. Transfer
clips at 0.10 and 0.20 m/s are outside the sole training target.</p>
<p><a href="../docs/gait11-experiment-report.md">Experiment report</a> ·
<a href="../docs/experiments/gait11_20261019_v1/replay-provenance.json">Provenance</a> ·
<a href="gait09-videos.html">Earlier GAIT-09 videos</a></p>
<script>function rate(r){document.querySelectorAll('video').forEach(v=>v.playbackRate=r);document.getElementById('rate').textContent=r+'x';}</script></html>'''
    (REPORT/'gait11-videos.html').write_text(html, encoding='utf-8')
    files = (list((REPORT/'videos').glob('gait11_*.mp4')) + traces
             + [p.with_suffix('.npz') for p in traces]
             + [ROOT/'RLframework/tools/gait_video.py', Path(__file__).resolve(),
                ROOT/'RLframework/model/dash01_cad_v2_free.xml'])
    provenance = dict(training_job=3223293, replay_job=3223517, replays=meta,
                      files={str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (DATA/'replay-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Gallery, figures and replay provenance saved.')


if __name__ == '__main__':
    main()
