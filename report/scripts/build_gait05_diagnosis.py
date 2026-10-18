"""Plot GAIT-05 learning conditions from the downloaded training CSV."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
with (ROOT/'docs/experiments/gait05_20261015_v1/progress.csv').open() as f:
    rows = list(csv.DictReader(f))
def col(name):
    return np.array([float(r[name]) for r in rows])
x = col('time/env_steps')/1e6
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout='constrained')
axes[0].plot(x, col('rollout/ep_len_mean')/100, color='#2269ae', linewidth=1)
axes[0].axhline(5, color='gray', linestyle='--', label='Gait measurement begins at 5 s')
axes[0].set_ylabel('Mean training episode (s)')
axes[0].legend(loc='upper left')
axes[1].plot(x, 100*col('gait/reward_floor_fraction'), color='#c65c28', linewidth=1)
axes[1].set_ylabel('Training ticks at reward floor (%)')
axes[2].plot(x, col('curriculum/shape_scale'), label='Gait shaping multiplier', color='#a83b6b')
axes[2].plot(x, col('train/std_mean'), label='Mean action standard deviation', color='#35754b')
axes[2].set_ylabel('Multiplier / action std')
axes[2].legend(loc='upper right')
axes[2].set_xlabel('GAIT-05 training steps (millions)')
for ax in axes:
    ax.grid(alpha=.2)
    ax.axvline(3, color='gray', alpha=.35, linestyle=':')
fig.suptitle('GAIT-05: strong gait shaping before reliable support developed')
fig.supxlabel('Training-rollout aggregates, not first-fall times from greedy evaluation.\n'
              'The association between shaping, clipping and survival does not establish causation.', fontsize=9)
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(ROOT/f'report/figures/gait05_learning_diagnosis.{ext}', dpi=160)
print('Saved GAIT-05 learning diagnosis figure.')
