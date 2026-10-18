"""Plot the recorded GAIT-04 evaluations without rerunning simulation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
source = ROOT/'docs/experiments/gait04_20261015_v1/gait_evaluations.jsonl'
rows = [json.loads(line) for line in source.read_text().splitlines()]
x = np.array([r['step'] for r in rows])/1e6
steps = np.array([np.mean(r['per_trial']['steps'], axis=0) for r in rows])
clear = 100*np.array([np.mean(r['per_trial']['clear_fraction'], axis=0) for r in rows])
survival = [sum(r['per_trial']['survived']) for r in rows]

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout='constrained')
for i, (side, color) in enumerate([('Left', '#2582db'), ('Right', '#e76e20')]):
    axes[0].plot(x, steps[:, i], 'o-', color=color, label=side)
    axes[1].plot(x, clear[:, i], 'o-', color=color, label=side)
axes[0].set_ylabel('Mean valid steps')
axes[0].legend(loc='upper left', ncol=2)
axes[1].axhline(8, color='gray', linestyle='--', label='Per-foot minimum')
axes[1].set_ylabel('Mean clear time (%)')
axes[1].legend(loc='upper left', ncol=3)
axes[2].plot(x, survival, 'o-', color='#307f55', label='Survived')
axes[2].plot(x, [sum(r['passed']) for r in rows], 's-', color='#ab3355', label='Passed all gait criteria')
axes[2].set_ylabel('Trials out of 12')
axes[2].set_ylim(-.5, 12.8)
axes[2].legend(loc='center left')
axes[2].set_xlabel('Additional GAIT-04 training steps (millions)')
for ax in axes:
    ax.grid(alpha=.2)
    ax.axvline(14.680064, color='gray', alpha=.4, linestyle=':')
fig.suptitle('GAIT-04: right-foot stepping emerged late; final policy regressed')
fig.supxlabel('Quality: seconds 5–25 before termination; early falls shorten the window.\n'
              'Clearance threshold: 1.5 cm. These are evaluation means, not video measurements.', fontsize=9)
out = ROOT/'report/figures'
out.mkdir(exist_ok=True)
for extension in ('png', 'svg', 'pdf'):
    fig.savefig(out/f'gait04_progress.{extension}', dpi=160)
print('Saved GAIT-04 progress figure (PNG, SVG, PDF).')
