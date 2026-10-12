"""Watch/render recorded MJX poses; this does not run a different physics simulation.

python RLframework/tools/replay_stop_trace.py --trace ...npz --env 1 --viewer
python RLframework/tools/replay_stop_trace.py --trace ...npz --env 1 --video ...mp4
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(trace, env_id):
    with np.load(trace, allow_pickle=False) as f:
        meta = json.loads(str(f['meta'].item()))
        if not 0 <= env_id < meta['n_envs']:
            raise ValueError(f"env must be between 0 and {meta['n_envs'] - 1}")
        arrays = {k: f[k][:, env_id].copy() for k in
                  ('qpos', 'qvel', 'speed', 'vx', 'upright', 'joint_rms', 'freq', 'done', 'reasons')}
        for k in ('phase', 'reference', 'gyro', 'supported'):
            if k in f:
                arrays[k] = f[k][:, env_id].copy()
    terminated = np.flatnonzero(arrays['done'])
    end = int(terminated[0]) if terminated.size else len(arrays['qpos']) - 1
    reasons = [r.replace('term_', '') for r, hit in zip(meta['reason_names'], arrays['reasons'][end]) if hit]
    return meta, arrays, end, reasons


def setup(meta):
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'RLframework' / meta['model_path']))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    # Presentation only: brighten the dark CAD meshes without altering physics.
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth = 90
    camera.elevation = -12
    camera.distance = 3.3
    return model, data, camera


def pose(model, data, camera, arrays, k):
    data.qpos[:] = arrays['qpos'][k]
    data.qvel[:] = arrays['qvel'][k]
    mujoco.mj_forward(model, data)
    camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.55]


def render(meta, arrays, end, reasons, env_id, out, rate):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    model, data, camera = setup(meta)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 18) if font_path.exists() else font
    dt, fps = meta['control_dt'], 30
    stop_t = meta['stop_ticks'][env_id] * dt
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # End at first environment termination, then freeze for 2 s of video.
    frames = int(np.ceil((end * dt / rate + 2) * fps))
    with mujoco.Renderer(model, height=640, width=960) as renderer, \
            imageio.get_writer(out, fps=fps, codec='libx264', quality=8, macro_block_size=16) as writer:
        for frame in range(frames):
            k = min(end, int(round(frame / fps * rate / dt)))
            pose(model, data, camera, arrays, k)
            renderer.update_scene(data, camera=camera)
            im = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, 960, 116), fill=(19, 25, 36))
            stopped = k >= meta['stop_ticks'][env_id]
            label = 'STOP' if stopped else 'WALK'
            if meta.get('ramp') and 'phase' in arrays:
                label = ('WALK', 'STOP / BRAKE', 'STOP / HOLD')[int(arrays['phase'][k])]
            color = (255, 158, 75) if stopped else (104, 230, 168)
            draw.text((18, 10), f"{label}   |   requested walk speed {meta['speeds'][env_id]:.1f} m/s", font=font, fill=color)
            draw.text((18, 43), f"t = {k * dt:.2f} s     STOP at {stop_t:.2f} s     playback {rate:g}x", font=small, fill='white')
            j = max(0, k - 1)  # physical readouts follow each tick; poses precede it
            tilt = np.degrees(np.arccos(np.clip(arrays['upright'][j], -1, 1)))
            draw.text((18, 74), f"horizontal speed {arrays['speed'][j]:.2f} m/s   tilt {tilt:.1f} deg   gait {arrays['freq'][j]:.1f} Hz", font=small, fill='white')
            draw.rectangle((0, 576, 960, 640), fill=(19, 25, 36))
            caption = f"Checkpoint: {meta['step']:,} steps | Recorded MJX trajectory, trial {env_id}"
            if meta.get('zero_action'):
                caption = f"Nominal stance feasibility probe (fixed zero action) | Trial {env_id}"
            draw.text((18, 585), caption, font=small, fill='white')
            if k == end and reasons:
                draw.text((18, 610), 'Environment ended: ' + ', '.join(reasons) + ' (last pre-termination pose frozen)', font=small, fill=(255, 158, 75))
            else:
                detail = 'Actual recorded poses; no re-simulation and no automatic reset'
                if meta.get('ramp') and 'reference' in arrays:
                    detail = f"Speed reference {arrays['reference'][k]:.2f} m/s | {max(0,k*dt-stop_t):.2f} s since STOP"
                draw.text((18, 610), detail, font=small, fill=(180, 190, 210))
            writer.append_data(np.asarray(im))
    print(f'[replay] saved {out}', flush=True)


def viewer(meta, arrays, end, reasons, env_id, rate):
    import mujoco.viewer
    model, data, camera = setup(meta)
    control = dict(paused=False, restart=False, quit=False)
    def key(code):
        if code == 32:
            control['paused'] = not control['paused']
        elif code in (82, 114):
            control['restart'] = True
        elif code in (81, 113, 256):
            control['quit'] = True
    print('Recorded final-policy trajectory. Space = pause; R = replay; Q = close.', flush=True)
    print(f"Trial {env_id}, command {meta['speeds'][env_id]} m/s, STOP at "
          f"{meta['stop_ticks'][env_id] * meta['control_dt']:.2f} s; termination: {reasons}", flush=True)
    dt = meta['control_dt']
    with mujoco.viewer.launch_passive(model, data, key_callback=key, show_left_ui=False,
                                      show_right_ui=False) as view:
        view.cam.azimuth, view.cam.elevation, view.cam.distance = camera.azimuth, camera.elevation, camera.distance
        clock_t, previous, announced = 0.0, time.perf_counter(), False
        while view.is_running() and not control['quit']:
            now = time.perf_counter()
            if control['restart']:
                clock_t, control['restart'], announced = 0.0, False, False
            elif not control['paused']:
                clock_t = min(end * dt, clock_t + (now - previous) * rate)
            previous = now
            k = min(end, int(clock_t / dt))
            pose(model, data, view.cam, arrays, k)
            if not announced and k >= meta['stop_ticks'][env_id]:
                print('STOP requested', flush=True)
                announced = True
            view.sync()
            time.sleep(0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--env', type=int, default=1)
    ap.add_argument('--rate', type=float, default=0.5)
    ap.add_argument('--video')
    ap.add_argument('--viewer', action='store_true')
    args = ap.parse_args()
    if args.rate <= 0 or not (args.video or args.viewer):
        ap.error('provide --video or --viewer, and a positive --rate')
    meta, arrays, end, reasons = load(args.trace, args.env)
    if args.video:
        render(meta, arrays, end, reasons, args.env, args.video, args.rate)
    if args.viewer:
        viewer(meta, arrays, end, reasons, args.env, args.rate)


if __name__ == '__main__':
    main()
