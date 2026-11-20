"""Plot the actual native-MuJoCo collision regions and loaded contact points."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'BalanceRL/tests'))
from test_standing import load


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), sharex=True, sharey=True)
    for ax, name, title in zip(axes, ('v1', 'v2', 'xshape'),
                               ('Original: continuous soles', 'V2: two pads per foot', 'X-shape: two pads per foot')):
        _, m, d, p = load(name)
        center = d.xpos[m.body('torso').id, :2]
        for side, color in enumerate(('#3377b9', '#dc7b31')):
            for gid in sorted(set((int(p.foot_gids[side]), int(p.heel_gids[side])))):
                sx, sy, sz = m.geom_size[gid]
                local = np.array([[-sx, -sy, -sz], [sx, -sy, -sz],
                                  [sx, sy, -sz], [-sx, sy, -sz]])
                world = local @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
                ax.add_patch(Polygon((world[:, :2]-center)*1000, facecolor=color, alpha=.35,
                                     edgecolor=color, linewidth=2))
            ids = {int(p.foot_gids[side]), int(p.heel_gids[side])}
            for con in d.contact:
                if p.floor_gid in con.geom and any(g in ids for g in con.geom) and con.dist <= 0:
                    ax.plot(*(1000*(con.pos[:2]-center)), '.', color=color, markersize=7)
        ax.plot(*(1000*(d.subtree_com[0, :2]-center)), 'k+', ms=12, mew=2, label='Projected CoM')
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
        ax.set_xlabel('World X relative to torso [mm]')
    axes[0].set_ylabel('World Y relative to torso [mm]')
    axes[-1].legend(loc='center', fontsize=8)
    fig.suptitle('Standing reference: physical sole regions and actual loaded contacts')
    fig.text(.5, .02, 'Filled regions = collision boxes; dots = native MuJoCo contacts. '
             'No foot is attached to the floor. Reference equilibrium, not a learned result.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .94))
    out = ROOT/'report/figures'
    out.mkdir(exist_ok=True)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(out/f'standing03_contact_layout.{suffix}', dpi=180)


if __name__ == '__main__':
    main()
