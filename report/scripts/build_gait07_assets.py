"""Rebuild GAIT-07 training figures from downloaded evaluation records."""
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'dash-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(s) for s in (ROOT/'docs/experiments/gait07_20261017_v1/gait_evaluations.jsonl').read_text().splitlines()]
x = np.array([r['step'] for r in rows])/1e6
fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True, layout='constrained')
ax[0].plot(x, [sum(r['per_trial']['survived']) for r in rows], 'o-', label='Survived 25 s')
ax[0].plot(x, [sum(r['passed']) for r in rows], 's-', label='Qualified gait')
ax[0].set(ylabel='Trials out of 12', ylim=(-.5, 12.5))
for i, side, color in [(0, 'Left', '#2582db'), (1, 'Right', '#e76e20')]:
    ax[1].plot(x, [np.mean(r['per_trial']['steps'], axis=0)[i] for r in rows], 'o-', color=color, label=side)
    ax[2].plot(x, [np.mean(r['per_trial']['slip_speed_mean'], axis=0)[i] for r in rows], 'o-', color=color, label=side)
ax[1].set_ylabel('Valid steps after 5 s')
ax[2].set_ylabel('Contact slip (m/s)')
ax[3].plot(x, [r['drag_curriculum']['drag_scale'] for r in rows], 'o-', label='Extra drag scale')
ax[3].plot(x, [r['walking_curriculum']['shape_scale'] for r in rows], 's-', label='Other gait shaping')
ax[3].set(ylabel='Reward scale', xlabel='Additional GAIT-07 steps (millions)', ylim=(-.05, 1.05))
for a in ax:
    a.grid(alpha=.2)
    a.legend(loc='upper left', ncol=2)
fig.suptitle('GAIT-07 at 0.25 m/s: survival improves; stepping remains asymmetric')
fig.supxlabel('Nominal evaluation means, not independent hardware trials.\nSlip includes all first-episode ticks; early falls shorten measurement windows.', fontsize=9)
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(ROOT/f'report/figures/gait07_progress.{ext}', dpi=150)
print('Saved GAIT-07 progress figures.')
