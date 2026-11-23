"""Render saved MJX states at1x; terminal poses held visibly, never reset-spliced."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'report'


def render(model):
    folder=REPORT/f'data/quiet01v2/videos/{model}'
    with np.load(folder/'trace.npz',allow_pickle=False) as z:
        meta=json.loads(str(z['meta'].item()));trace={k:z[k].copy() for k in z.files if k!='meta'}
    mp=ROOT/'BalanceRL'/meta['model_path']
    assert hashlib.sha256(mp.read_bytes()).hexdigest()==meta['model_sha256']
    best=REPORT/f'data/quiet01v2/results/{model}/best.msgpack'
    assert hashlib.sha256(best.read_bytes()).hexdigest()==meta['checkpoint_sha256']
    m=mujoco.MjModel.from_xml_path(str(mp))
    m.vis.global_.offwidth=640;m.vis.global_.offheight=320;m.vis.quality.offsamples=0
    m.vis.headlight.ambient[:]=[.5,.5,.5]
    m.vis.headlight.diffuse[:]=[.8,.8,.8]
    m.vis.headlight.specular[:]=[.1,.1,.1]
    for gid in range(m.ngeom):
        if m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
            m.geom_rgba[gid]=[.70,.74,.78,1.];m.geom_matid[gid]=-1
    for side,color in [('L',[.15,.55,1.,1.]),('R',[1.,.4,.08,1.])]:
        bid=m.geom_bodyid[m.geom('foot_'+side+'_col').id]
        for gid in np.flatnonzero(m.geom_bodyid==bid):m.geom_rgba[gid]=color;m.geom_matid[gid]=-1
    d=mujoco.MjData(m);cam=mujoco.MjvCamera()
    cam.distance=1.5;cam.azimuth=115;cam.elevation=-12;cam.lookat[:]=[0.,0.,.43]
    font=ImageFont.load_default(size=20);small=ImageFont.load_default(size=16)
    valid=trace['alive']&np.isfinite(trace['qpos']).all(-1)&np.isfinite(trace['qvel']).all(-1)
    last=[np.flatnonzero(valid[:,i])[-1] for i in range(4)]
    path=REPORT/f'videos/quiet01v2_{model}_best.mp4';path.parent.mkdir(exist_ok=True)
    labels=['Quiet reference','Moderate: 3deg requested','Larger: 5deg requested','Hard: 8deg requested']
    with mujoco.Renderer(m,width=640,height=320) as renderer,imageio.get_writer(
            path,fps=20,codec='libx264',quality=8,macro_block_size=16) as writer:
        for number in range(400):
            t=number/20;k=round(t/meta['control_dt'])-1
            frame=Image.new('RGB',(1280,960),(19,25,36));draw=ImageDraw.Draw(frame)
            draw.text((12,6),f'{model.upper()} | best +{meta["checkpoint_step"]/1e6:.2f}M | t={t:.2f}s | 1x speed',font=font,fill='white')
            draw.text((12,34),'20s quiet-pose test | no pushes | BLUE left foot / ORANGE right foot',font=small,fill='#d2dce8')
            for i,trial in enumerate(meta['selected_trials']):
                x=(i%2)*640;y=64+(i//2)*432;kk=min(max(k,0),last[i])
                q=trace['initial_qpos'][i] if k<0 else trace['qpos'][kk,i]
                v=trace['initial_qvel'][i] if k<0 else trace['qvel'][kk,i]
                d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=cam)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                frame.paste(Image.fromarray(renderer.render()),(x,y+52))
                draw=ImageDraw.Draw(frame)
                draw.text((x+10,y+3),labels[i]+f' | trial {trial}',font=font,fill='white')
                draw.text((x+10,y+28),f'Actual initial motor RMS: {meta["initial_motor_rms_deg"][i]:.2f}deg',font=small,fill='#d2dce8')
                quiet=0. if k<0 else trace['quiet_s'][kk,i]
                passed=meta['per_trial']['passed'][trial]
                draw.text((x+10,y+378),f'Quiet streak {quiet:.2f}s | complete 20s test: '+('PASS' if passed else 'FAIL'),font=small,fill='#a0e8ba' if passed else '#ffb0a5')
                if k>=0:
                    draw.text((x+10,y+402),f'Joint RMS {np.rad2deg(trace["joint_rms"][kk,i]):.2f}deg | sole-relative error {1000*trace["foot_error"][kk,i]:.1f}mm',font=small,fill='#d2dce8')
                if k>last[i]:
                    cause='physical fall' if meta['per_trial']['physical_fall'][trial] else 'task timeout / foot limit'
                    draw.rectangle((x+8,y+170,x+630,y+234),fill=(125,25,30))
                    draw.text((x+18,y+177),f'Trial ended {meta["per_trial"]["duration_s"][trial]:.2f}s: {cause}',font=small,fill='white')
                    draw.text((x+18,y+204),'Last valid pose held; no reset is shown',font=small,fill='white')
            draw.text((12,937),'Requested angles are projected onto feasible poses. Sole-relative error is not clearance.',font=small,fill='#d2dce8')
            writer.append_data(np.asarray(frame))
            if number==20:frame.save(REPORT/f'figures/quiet01v2_{model}_preview.jpg',quality=90)
            if number%100==0:print(model,'render',number,'/400',flush=True)
    reader=imageio.get_reader(path);info=reader.get_meta_data();reader.close()
    assert info['size']==(1280,960) and abs(info['duration']-20.)<.06
    provenance=dict(model=model,checkpoint_step=meta['checkpoint_step'],selected_trials=meta['selected_trials'],
        backend=meta.get('backend','cuda'),
        fps=info['fps'],duration_s=info['duration'],size=info['size'],codec=info['codec'],
        video_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest(),
        note='Native rendering of captured MJX states, no motion resimulation,1x, no reset splicing; fixed cases0/48/64/80.')
    (folder/'video-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(model,'VIDEO COMPLETE',str(path),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['v1','v2','xshape'])
    for model in ap.parse_args().models:render(model)
