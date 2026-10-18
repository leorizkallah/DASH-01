"""Record a trained gait on SCITAS, then render the saved MJX poses locally."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def record(args):
    sys.path.insert(0, str(args.run.resolve().parents[1]))
    import jax
    import jax.numpy as jnp
    from evaluate import load_run
    from env import EnvParams
    from ppo import ObsStats
    checkpoints = args.checkpoints or ['best.msgpack']
    cfg, env, agent = load_run(args.run, args.run / checkpoints[0], n_envs=len(args.targets))
    params = EnvParams.final(cfg)._replace(dr_scale=0., ctrl_jitter_ms=0., ctrl_drop_prob=0.)
    targets = jnp.array(args.targets)

    @jax.jit
    def simulate(weights, mean, var, count):
        stats = ObsStats(mean=mean, var=var, count=count)
        state, obs = env.reset(jax.random.PRNGKey(args.seed), params)
        # Rebuild command-dependent reset features to match the pinned target.
        state = state.replace(v_cmd=targets, cmd_left=jnp.full_like(state.cmd_left, 1e9),
                              stop_cmd=jnp.zeros_like(state.stop_cmd))
        once = jax.vmap(env._once, in_axes=(0, None, 0, 0))(
            state, params, state.phase, state.commit.astype(jnp.float32))
        # The history width is the reset history flattened after subsampling.
        start = state.hist[:, env.hist_idx].reshape((env.n_envs, -1)).shape[1]
        obs = obs.at[:, start:start+once.shape[1]].set(once)
        def tick(carry, _):
            state, obs = carry
            action = jnp.clip(agent._act_greedy(weights, stats.normalize(obs)), -1., 1.)
            nxt, obs2, _, done, info = env.step(state, action, params)
            row = dict(qpos=state.data.qpos, done=done, fallen=info['fallen'],
                       vx=info['v_body_x'], flight=info['flight_fraction'],
                       upright=info['stop_upright'], freq=info['freq_hz'])
            if cfg.bilateral_enable:
                row['valid_step'] = info['bilateral']['valid_step']
                row['clear'] = info['bilateral']['clear']
            if getattr(cfg, 'bilateral_attempt_enable', False):
                for key in ('attempt', 'incomplete_step', 'repeated_attempt', 'contact',
                            'clearance', 'attempt_length', 'attempt_height', 'attempt_air'):
                    row[key] = info['bilateral'][key]
            if getattr(cfg, 'bilateral_repeat_air_weight', 0.) > 0:
                for key in ('repeat_air', 'reward_swing', 'controller_swing',
                            'expected_foot', 'lift_budget_used', 'turn_reach'):
                    row[key] = info['bilateral'][key]
            return (nxt, obs2), row
        return jax.lax.scan(tick, (state, obs), None, length=2500)[1]

    for checkpoint in checkpoints:
        agent.load(args.run / checkpoint)
        rows = jax.device_get(simulate(agent.params, agent.stats.mean, agent.stats.var, agent.stats.count))
        trace = args.trace if len(checkpoints) == 1 else args.trace.with_name(args.trace.stem+'_'+Path(checkpoint).stem+'.npz')
        save_record(args, cfg, env, agent, targets, rows, checkpoint, trace)


def save_record(args, cfg, env, agent, targets, rows, checkpoint, trace):
    trials = []
    for i, target in enumerate(np.asarray(targets)):
        ends = np.flatnonzero(rows['done'][:, i])
        n = int(ends[0]+1) if len(ends) else len(rows['done'])
        mask = slice(min(500, n), n)
        trials.append(dict(target=float(target), ticks=n, seconds=n*env.control_dt,
                           fell=bool(rows['fallen'][:n, i].any()),
                           mean_speed_after_5s=float(rows['vx'][mask, i].mean()) if n > 500 else None,
                           speed_error_after_5s=float(np.abs(rows['vx'][mask, i]-target).mean()) if n > 500 else None,
                           flight_fraction=float(rows['flight'][:n, i].mean())))
        if 'valid_step' in rows:
            trials[-1]['valid_steps_after_5s'] = rows['valid_step'][mask, i].sum(axis=0).tolist()
            trials[-1]['clear_fraction_after_5s'] = rows['clear'][mask, i].mean(axis=0).tolist() if n > 500 else None
        for key in ('attempt', 'incomplete_step', 'repeated_attempt'):
            if key in rows:
                trials[-1][key+'_after_5s'] = rows[key][mask, i].sum(axis=0).tolist()
    meta = dict(step=agent.step, seed=args.seed, dt=env.control_dt, model=cfg.model_path,
                checkpoint=checkpoint, label=args.label or f'{args.run.name} {Path(checkpoint).stem}',
                checkpoint_sha256=hashlib.sha256((args.run/checkpoint).read_bytes()).hexdigest(),
                trials=trials, note='Fresh nominal greedy replay trials, not the multi-trial gait evaluation. '
                'All trials retained; videos end at first termination, without showing auto-resets.')
    trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace, meta=json.dumps(meta), **rows)
    trace.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2), flush=True)


def render(args):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    with np.load(args.trace, allow_pickle=False) as z:
        meta = json.loads(str(z['meta'].item()))
        rows = {k: z[k].copy() for k in ('qpos', 'vx', 'flight')}
    model = mujoco.MjModel.from_xml_path(str(ROOT/'RLframework'/meta['model']))
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    model.vis.headlight.ambient[:] = [.5, .5, .5]
    if args.two_views:
        # Display-only colors identify anatomical sides; physics/XML are unchanged.
        for side, rgba in [('L', [.15, .55, 1., 1.]), ('R', [1., .40, .08, 1.])]:
            foot_id = model.body('foot_'+side).id
            for geom_id in np.flatnonzero(model.geom_bodyid == foot_id):
                model.geom_rgba[geom_id] = rgba
                model.geom_matid[geom_id] = -1
    data = mujoco.MjData(model)
    torso = model.body('torso').id
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.1, 115, -12
    font = ImageFont.load_default(size=21)
    args.out.mkdir(parents=True, exist_ok=True)
    width, height = (640, 596) if args.two_views else (960, 720)
    canvas_width = 1280 if args.two_views else 960
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i, trial in enumerate(meta['trials']):
            path = args.out / f"{args.prefix}_{trial['target']:.2f}mps.mp4"
            with imageio.get_writer(path, fps=30, codec='libx264', quality=8) as writer:
                for t in np.arange(0, trial['seconds'], 1/30):
                    k = min(round(t/meta['dt']), trial['ticks']-1)
                    data.qpos[:] = rows['qpos'][k, i]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = data.xpos[torso]
                    camera.lookat[2] = max(.4, camera.lookat[2]-.25)
                    if args.two_views:
                        frame = Image.new('RGB', (1280, 720), (19, 25, 36))
                        for j, azimuth in enumerate((115, 295)):
                            camera.azimuth = azimuth
                            renderer.update_scene(data, camera=camera)
                            frame.paste(Image.fromarray(renderer.render()), (j*640, 80))
                    else:
                        renderer.update_scene(data, camera=camera)
                        frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, canvas_width, 79), fill=(19, 25, 36))
                    label = args.label or meta.get('label', 'GAIT-02 best policy')
                    draw.text((15, 8), f"{label} | target {trial['target']:.2f} m/s | t={t:.2f} s | 1x speed", font=font, fill='white')
                    draw.text((15, 42), f"Forward speed {rows['vx'][max(0,k-1),i]:.2f} m/s | learned motion, no manual joint animation", font=font, fill='white')
                    if args.two_views:
                        draw.text((15, 84), 'Left foot: BLUE | Right foot: ORANGE', font=font, fill='white')
                        draw.text((655, 84), 'Same instant, opposite camera', font=font, fill='white')
                    draw.rectangle((0, 676, canvas_width, 720), fill=(19, 25, 36))
                    ending = 'fall detected' if trial['fell'] else '25 s completed'
                    draw.text((15, 687), f"Checkpoint {meta['step']:,} | MJX replay | trial outcome: {ending}", font=font, fill='white')
                    writer.append_data(np.asarray(frame))
            print(f'Saved {path}', flush=True)
    print(json.dumps(meta['trials'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--render', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT/'report/videos')
    p.add_argument('--checkpoints', nargs='+', help='Checkpoint filenames within --run; default best.msgpack')
    p.add_argument('--targets', nargs='+', type=float, default=[.2, .35, .5])
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--prefix', default='gait02_best')
    p.add_argument('--label')
    p.add_argument('--two-views', action='store_true')
    a = p.parse_args()
    if a.render:
        render(a)
    elif a.run is None:
        p.error('--run is required for recording')
    else:
        record(a)


if __name__ == '__main__':
    main()
