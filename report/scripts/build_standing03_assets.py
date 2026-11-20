"""Generate learning curves, report tables and gallery from archived STAND-03 data."""
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'report'
DATA = REPORT/'data/standing03/results'
FIG = REPORT/'figures'
MODELS = ('v1', 'v2', 'xshape')
NAMES = {'v1': 'Original V1', 'v2': 'V2', 'xshape': 'X-shape'}
COLORS = {'v1': '#2072b4', 'v2': '#d67727', 'xshape': '#279565'}


def read(model, name):
    return json.loads((DATA/model/name).read_text())


def block(x, y, size=100):
    return ([np.mean(x[i:i+size]) for i in range(0, len(x), size)],
            [np.mean(y[i:i+size]) for i in range(0, len(y), size)])


def save(fig, stem):
    fig.tight_layout(rect=(0, .065, 1, .94))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(FIG/f'{stem}.{ext}', dpi=170)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    overview, ax = plt.subplots(2, 2, figsize=(13, 8.6))
    overview.suptitle('STAND-03: three fresh policies, 200M transitions each', fontsize=17)
    summary = []
    for model in MODELS:
        rows = [json.loads(s) for s in (DATA/model/'progress.jsonl').read_text().splitlines()]
        ev = [json.loads(s) for s in (DATA/model/'standing_evaluations.jsonl').read_text().splitlines()]
        best, final, completed = read(model, 'best.json'), read(model, 'final_eval.json'), read(model, 'completion.json')
        x = np.array([r['time/env_steps'] for r in rows])/1e6
        ex = np.array([r['step'] for r in ev])/1e6
        sx = best['step']/1e6
        assert completed['steps'] == 200005632 and np.all(np.diff(x) > 0)
        assert all(np.isfinite(r['rollout/reward_mean']) for r in rows)
        fig, a = plt.subplots(3, 2, figsize=(13, 11))
        fig.suptitle(f'{NAMES[model]}: learning curves and fixed evaluation', fontsize=17)
        y = np.array([r['rollout/reward_mean'] for r in rows])
        bx, by = block(x, y)
        a[0, 0].plot(x, y, alpha=.13, lw=.5, color=COLORS[model], rasterized=True)
        a[0, 0].plot(bx, by, color=COLORS[model])
        a[0, 0].set(title='Training reward (100-update block mean)', ylabel='Unscaled reward / control transition')
        ey = np.array([r['rollout/ep_len_mean'] for r in rows])*.01
        # Mean duration is undefined when an update finishes no episodes.
        valid = np.array([r['rollout/episodes'] > 0 for r in rows])
        a[0, 1].plot(x[valid], ey[valid], '.', alpha=.2, ms=2, rasterized=True)
        a[0, 1].axhline(30, color='gray', ls='--', label='30 s episode cap')
        a[0, 1].set(title='Completed training episodes', ylabel='Mean duration [s]', ylim=(0, 31))
        for level in ('0.00', '0.02', '0.05', '0.10', '0.15'):
            a[1, 0].plot(ex, [100*r['survival'][level] for r in ev], '.-', label=f'{level} m/s', ms=4)
        a[1, 0].set(title='Fixed evaluation: full task pass by push level', ylabel='Passing trials [%]', ylim=(-3, 103))
        a[1, 0].legend(fontsize=8, ncol=3)
        a[1, 1].plot(ex, [100*r['score'] for r in ev], 'o-', ms=3, label='Combined task pass')
        a[1, 1].plot(ex, [100*r['full_duration_fraction'] for r in ev], '--', label='Complete 20 s duration')
        a[1, 1].plot(ex, [100*r['step_violation_fraction'] for r in ev], ':', label='Planted-rule violations')
        a[1, 1].plot(sx, 100*best['eval']['score'], '*', color='black', ms=14, label='Saved best')
        a[1, 1].set(title='Fixed evaluation: task success and duration', ylabel='Trials [%]', ylim=(-3, 103))
        a[1, 1].legend(fontsize=8)
        a[2, 0].plot(x, [r['curriculum/push_level'] for r in rows], label='Maximum sampled push level')
        a[2, 0].set(title='Training curriculum (not measured robustness)', ylabel='Equivalent impulse Δv [m/s]', ylim=(0, .16))
        noise = a[2, 0].twinx()
        noise.plot(x, [r['curriculum/plant_scale'] for r in rows], color='#888888', alpha=.65, ls='--')
        noise.set(ylabel='Sensor-noise ramp [0–1]', ylim=(0, 1.05))
        for term in ('quiet', 'support', 'torque', 'slip', 'flat_missing', 'foot_drift'):
            tx, ty = block(x, [r['reward_terms/'+term] for r in rows])
            a[2, 1].plot(tx, ty, label=term)
        a[2, 1].set(title='Selected weighted reward contributions', ylabel='Reward / transition')
        a[2, 1].legend(fontsize=8, ncol=3)
        for aa in a.flat:
            aa.axvline(sx, color='gray', ls=':', alpha=.7)
            aa.set(xlabel='Training transitions [millions]', xlim=(0, 200.01))
            aa.grid(alpha=.2)
        fig.text(.02, .012, f'Dotted line: saved best at {sx:.2f}M. Evaluation: 80 first episodes, 20 s, full sensor noise, nominal plant.\n'
                 '16 directions per level, fixed seed. 0.00 = quiet standing; push trials require recovery + 5 s quiet. One training seed; no convergence guarantee.', fontsize=9)
        save(fig, f'standing03_{model}_convergence')
        ax[0, 0].plot(bx, by, color=COLORS[model], label=NAMES[model])
        ax[0, 1].plot(ex, [100*r['score'] for r in ev], '.-', color=COLORS[model], label=NAMES[model])
        ax[0, 1].plot(sx, 100*best['eval']['score'], '*', color=COLORS[model], ms=13)
        ax[1, 0].plot(ex, [100*r['survival']['0.00'] for r in ev], '.-', color=COLORS[model], label=NAMES[model])
        ax[1, 1].plot(x, [r['curriculum/push_level'] for r in rows], color=COLORS[model], label=NAMES[model])
        item = dict(model=model, training_steps=completed['steps'], wall_hours=completed['elapsed_s']/3600,
                    best_step=best['step'], best_passes=round(80*best['eval']['score']),
                    final_passes=round(80*final['score']), best_rates=best['eval']['survival'],
                    best_full_duration=best['eval']['full_duration_fraction'])
        replay = DATA/model/'replay.json'
        if replay.exists():
            rr = json.loads(replay.read_text())
            item.update(replay_passes=round(80*rr['replay_score']), replay_rates=rr['rates'])
        summary.append(item)
        with (DATA/model/'evaluation-table.csv').open('w', newline='') as f:
            fields = ['step', 'score', 'full_duration_fraction', 'step_violation_fraction']
            writer = csv.DictWriter(f, fieldnames=fields+['pass_'+k for k in best['eval']['survival']])
            writer.writeheader()
            writer.writerows({**{k:r[k] for k in fields}, **{'pass_'+k:v for k,v in r['survival'].items()}} for r in ev)
    for aa, title, label in zip(ax.flat, ('Training reward', 'Fixed evaluation: combined task pass',
             'Fixed evaluation: quiet standing', 'Training push curriculum'),
             ('Reward / transition', 'Passing trials [%]', 'Passing trials [%]', 'Equivalent impulse Δv [m/s]')):
        aa.set(title=title, ylabel=label, xlabel='Training transitions [millions]', xlim=(0, 200.01))
        aa.grid(alpha=.2)
        aa.legend()
    ax[0, 1].set_ylim(-3, 103)
    ax[1, 0].set_ylim(-3, 103)
    ax[1, 1].set_ylim(0, .16)
    overview.text(.02, .014, 'Stars: saved best checkpoint. Reward: 100-update block means (~1.84M transitions); evaluations are unsmoothed.\n'
                  'Each evaluation uses the same 80 directions/noise seeds. Curriculum level is a training input, not a success rate. One training seed per model.', fontsize=9)
    save(overview, 'standing03_convergence_comparison')
    (DATA/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    inputs = list(DATA.glob('*/progress.jsonl')) + list(DATA.glob('*/standing_evaluations.jsonl')) + list(DATA.glob('*/best.json'))
    provenance = dict(date='2026-11-10', training_array=3246822, replay_array=3247563,
                      smoothing='100-update nonoverlapping block means; unsmoothed evaluation points',
                      inputs_sha256={p.relative_to(REPORT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (DATA/'plot-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    cards = []
    for r in summary:
        m = r['model']
        video = f'videos/standing03_{m}_best.mp4'
        video_markup = f'<video controls preload="metadata" poster="figures/standing03_{m}_preview.jpg" src="{video}"></video>' if (REPORT/video).exists() else '<p>Best-policy replay is queued / rendering.</p>'
        if m == 'v2' and 'replay_passes' in r:
            video_markup += '<p><strong>The fixed forward-push trial falls at 6.50 s.</strong> Its last pre-terminal pose is held with a warning; the quiet trial completes and passes.</p>'
        if m == 'xshape' and 'replay_passes' in r:
            video_markup += '<p>Replay total matches the archive, with quiet passes changing 12/16 to 13/16 and 0.15 m/s passes changing 12/16 to 11/16. This per-level variation remains unexplained.</p>'
        replay_text = f"Independent replay: {r['replay_passes']}/80." if 'replay_passes' in r else 'Independent replay pending.'
        cards.append(f'<section><h2>{NAMES[m]}</h2><p>Best checkpoint: {r["best_step"]/1e6:.2f}M transitions; archived task pass {r["best_passes"]}/80. Final: {r["final_passes"]}/80. {replay_text}</p>{video_markup}<p>Two independent trials, fixed before outcomes: quiet standing (left), +X 0.10 m/s-equivalent push at 5 s (right). Blue = left foot, orange = right. 1× speed. Any ended trial freezes with a visible warning; no reset splicing.</p><a href="figures/standing03_{m}_convergence.pdf">Vector PDF</a><img src="figures/standing03_{m}_convergence.png" alt="{NAMES[m]} learning curves"></section>')
    page = '<!doctype html><meta charset="utf-8"><title>STAND-03 — standing comparison</title><style>body{font:17px system-ui;max-width:1280px;margin:30px auto;padding:0 20px;background:#f3f5f8;color:#182635}section{background:white;padding:24px;margin:28px 0;border-radius:10px}video,img{width:100%;height:auto}p{line-height:1.5}a{color:#1766a5}</style><h1>STAND-03: standing with planted feet</h1><p>All three runs completed 200,005,632 steps, exit 0. Original V1: 7h11; V2: 9h33; X-shape: 9h31. Best checkpoints selected by fixed 80-trial task score; the final policies are weaker. These are simulation results with one training seed per model; X-shape mass correction remains provisional.</p><p><strong>How to read success:</strong> quiet trials must complete 20 s and finish with five seconds of quiet standing. Pushed trials must complete 20 s, start a five-second quiet window within two seconds after the pulse. A full-duration trial can still fail the quiet criterion. Push tests score the completed recovery window; they do not require the last five seconds to be quiet. See <a href="standing03-results.md">results and metric limitations</a>.</p><a href="figures/standing03_convergence_comparison.pdf">Comparison PDF</a><img src="figures/standing03_convergence_comparison.png" alt="Three-policy comparison">'+''.join(cards)
    (REPORT/'standing03-videos.html').write_text(page, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
