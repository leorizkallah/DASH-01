"""Render two preselected independent MJX trials per model, with no reset splicing."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'report'
DATA = REPORT/'data/standing03/results'


def render(model_name, campaign='standing03', replay_job=3247563):
    folder = REPORT/'data'/campaign/'results'/model_name
    with np.load(folder/'trace.npz', allow_pickle=False) as z:
        meta = json.loads(str(z['meta'].item()))
        trace = {k:z[k].copy() for k in ('qpos', 'qvel', 'force', 'alive', 'quiet_s')}
    model_path = ROOT/'BalanceRL'/meta['model_path']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == meta['model_sha256']
    assert hashlib.sha256((folder/'best.msgpack').read_bytes()).hexdigest() == meta['checkpoint_sha256']
    m = mujoco.MjModel.from_xml_path(str(model_path))
    m.vis.global_.offwidth, m.vis.global_.offheight = 640, 592
    m.vis.quality.offsamples = 0
    m.vis.headlight.ambient[:] = [.5, .5, .5]
    for side, color in [('L', [.15, .55, 1., 1.]), ('R', [1., .4, .08, 1.])]:
        gid = m.geom(f'foot_{side}_col').id
        bid = m.geom_bodyid[gid]
        for g in np.flatnonzero(m.geom_bodyid == bid):
            m.geom_rgba[g] = color
            m.geom_matid[g] = -1
    d = mujoco.MjData(m)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.9, 115, -15
    camera.lookat[:] = [0., 0., .48]
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=17)
    path = REPORT/f'videos/{campaign}_{model_name}_best.mp4'
    path.parent.mkdir(exist_ok=True)
    last_valid = [np.flatnonzero(trace['alive'][:, i])[-1] for i in range(2)]
    with mujoco.Renderer(m, width=640, height=592) as renderer, imageio.get_writer(
            path, fps=20, codec='libx264', quality=8, macro_block_size=16) as writer:
        for frame_no in range(400):
            t = frame_no/20
            k = round(t/.01)
            frame = Image.new('RGB', (1280, 720), (19, 25, 36))
            for i, trial in enumerate(meta['selected_trials']):
                kk = min(k, last_valid[i])
                d.qpos[:] = trace['qpos'][kk, i]
                d.qvel[:] = trace['qvel'][kk, i]
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                frame.paste(Image.fromarray(renderer.render()), (640*i, 80))
            draw = ImageDraw.Draw(frame)
            checkpoint_label = f'+{meta["checkpoint_step"]/1e6:.2f}M additional' if campaign == 'standing04' else f'{meta["checkpoint_step"]/1e6:.2f}M'
            draw.text((14, 8), f'{model_name.upper()} | best {checkpoint_label} | t={t:.2f} s | 1x speed | MJX replay', font=font, fill='white')
            draw.text((14, 44), 'Quiet standing | fixed trial 0', font=font, fill='white')
            level = meta['selected_levels'][1]
            pulse = meta.get('selected_pulses', [{},dict(onset_s=5.,sampled_duration_s=.1)])[1]
            onset = pulse['onset_s']
            onset_text = f'{onset:.2f}s' if onset is not None else 'not reached'
            draw.text((654, 44), f'+X push {level:.2f} m/s | onset {onset_text} | trial 48', font=font, fill='white')
            for i, trial in enumerate(meta['selected_trials']):
                passed = meta['per_trial']['passed'][trial]
                draw.text((14+640*i, 84), 'Left foot: BLUE | Right foot: ORANGE', font=small, fill='white', stroke_width=1, stroke_fill='black')
                if k > last_valid[i]:
                    duration = meta['per_trial']['duration_s'][trial]
                    cause = 'physical fall' if meta['per_trial']['physical_fall'][trial] else 'recovery task' if campaign == 'standing04' else 'planted-foot rule'
                    draw.rectangle((640*i+10, 300, 640*i+630, 375), fill=(125, 20, 25))
                    draw.text((640*i+20, 309), f'Trial ended at {duration:.2f} s: {cause}', font=font, fill='white')
                    draw.text((640*i+20, 342), 'Last pre-terminal pose held; no restart shown', font=small, fill='white')
                elif i == 1 and np.linalg.norm(trace['force'][min(k,last_valid[i]),i]) > 1e-5:
                    draw.rectangle((650, 120, 1260, 155), fill=(130, 20, 20))
                    magnitude = np.linalg.norm(trace['force'][min(k,last_valid[i]),i])
                    draw.text((660, 126), f'Torso push: {magnitude:.1f} N | {pulse["sampled_duration_s"]*1000:.0f} ms', font=font, fill='white')
                quiet = trace['quiet_s'][min(k, last_valid[i]), i]
                verdict = 'PASS' if passed else 'FAIL'
                draw.text((14+640*i, 681), f'Quiet streak {quiet:.2f} s | complete-trial outcome: {verdict}', font=small, fill='white')
            writer.append_data(np.asarray(frame))
            if frame_no == (220 if campaign == 'standing04' else 140):
                frame.save(REPORT/f'figures/{campaign}_{model_name}_preview.jpg', quality=90)
    reader = imageio.get_reader(path)
    info = reader.get_meta_data()
    reader.close()
    assert info['size'] == (1280, 720) and abs(info['duration']-20.) < .06
    provenance = dict(model=model_name, replay_job=replay_job, checkpoint_step=meta['checkpoint_step'],
        selected_trials=meta['selected_trials'], batch_size=80, video=dict(
            fps=info['fps'], size=info['size'], duration_s=info['duration'], codec=info['codec']),
        sha256={p.relative_to(REPORT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (path, folder/'trace.npz', folder/'replay.json', Path(__file__))},
        note='20 s at 1x. Two independent trials, fixed before outcomes. Native rendering of MJX states; no physics rerun or reset splicing.')
    (folder/'video-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'model':model_name, **provenance['video']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['v1', 'v2', 'xshape'])
    ap.add_argument('--campaign', choices=['standing03','standing04'], default='standing03')
    ap.add_argument('--replay-job', type=int, default=3247563)
    args = ap.parse_args()
    for model in args.models:
        render(model, args.campaign, args.replay_job)
