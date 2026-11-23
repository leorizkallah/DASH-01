"""Push videos: fixed force panels, genuine captured states, force arrows and separate outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'report';BASE=REPORT/'data/quiet_push01'


def render(model,direction):
    folder=BASE/model
    with np.load(folder/'trace.npz',allow_pickle=False) as z:
        meta=json.loads(str(z['meta'].item()));data={k:z[k].copy() for k in z.files if k!='meta'}
    summary=json.loads((folder/'summary.json').read_text())
    ids=[next(c['index'] for c in meta['cases'] if c['force_n']==f and c['repeat']==0 and (f==0 or c['direction']==direction)) for f in (0,5,20,60)]
    path=ROOT/'BalanceRL'/meta['model_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['model_sha256']
    assert hashlib.sha256((REPORT/f'data/quiet01v2/results/{model}/best.msgpack').read_bytes()).hexdigest()==meta['checkpoint_sha256']
    m=mujoco.MjModel.from_xml_path(str(path));m.vis.global_.offwidth=640;m.vis.global_.offheight=320;m.vis.quality.offsamples=0
    m.vis.headlight.ambient[:]=[.5]*3;m.vis.headlight.diffuse[:]=[.8]*3;m.vis.headlight.specular[:]=[.1]*3
    for g in range(m.ngeom):
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:m.geom_rgba[g]=[.70,.74,.78,1.];m.geom_matid[g]=-1
    for side,color in [('L',[.15,.55,1.,1.]),('R',[1.,.4,.08,1.])]:
        for g in np.flatnonzero(m.geom_bodyid==m.geom_bodyid[m.geom('foot_'+side+'_col').id]):m.geom_rgba[g]=color;m.geom_matid[g]=-1
    d=mujoco.MjData(m);cam=mujoco.MjvCamera();cam.distance=1.5;cam.azimuth=115;cam.elevation=-12
    font=ImageFont.load_default(size=20);small=ImageFont.load_default(size=16)
    video=REPORT/f'videos/quiet_push01_{model}_{direction}.mp4'
    with mujoco.Renderer(m,width=640,height=320) as renderer,imageio.get_writer(video,fps=20,codec='libx264',quality=8,macro_block_size=16) as writer:
        for number in range(400):
            t=number/20;k=min(max(round(t/.01)-1,0),1999);vk=max(number-1,0)
            frame=Image.new('RGB',(1280,960),(19,25,36));draw=ImageDraw.Draw(frame)
            draw.text((12,6),f'{model.upper()} | {direction.upper()} push | t={t:.2f}s | 1x | frozen best policy',font=font,fill='white')
            draw.text((12,34),'Single100ms pulse at6s | BLUE leftfoot / ORANGE rightfoot | same noise in all panels',font=small,fill='#d2dce8')
            for j,i in enumerate(ids):
                x=(j%2)*640;y=64+(j//2)*432;r=summary['cases'][i];ended=t>=r['duration_s'] and r['duration_s']<20.
                if t==0:q=data['initial_qpos'][i];v=np.zeros(m.nv)
                elif ended:q=data['qpos_terminal'][i];v=data['qvel_terminal'][i]
                else:q=data['qpos_video'][vk,i];v=data['qvel_video'][vk,i]
                d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
                bid=m.body('torso').id;cam.lookat[:]=[d.xpos[bid,0],d.xpos[bid,1],.43]
                renderer.update_scene(d,camera=cam);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                force=data['force'][k,i] if not ended else np.zeros(3);magnitude=np.linalg.norm(force)
                if magnitude>1e-6:
                    g=renderer.scene.geoms[renderer.scene.ngeom];origin=d.xipos[bid].copy()
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.array([.01,.01,.1]),origin,np.eye(3).ravel(),np.array([1.,.05,.05,1.],np.float32))
                    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.015,origin,origin+force/magnitude*(.12+.006*magnitude));renderer.scene.ngeom+=1
                frame.paste(Image.fromarray(renderer.render()),(x,y+52));draw=ImageDraw.Draw(frame)
                draw.text((x+10,y+3),f'{r["force_n"]:g} N | impulse {r["nominal_impulse_ns"]:.2f} N s',font=font,fill='white')
                phase='WARM-UP' if t<6 else 'PUSH' if t<6.1 else 'POST-PUSH'
                draw.text((x+10,y+28),phase+(' | red arrow = applied force' if magnitude>0 else ''),font=small,fill='#ffb0a5' if magnitude>0 else '#d2dce8')
                draw.text((x+10,y+378),f'Survived20s: {"YES" if r["survived"] else "NO"} | regained5s: {"YES" if r["recovered"] else "NO"} | final5s: {"YES" if r["retained"] else "NO"}',font=small,fill='#d2dce8')
                quiet=data['quiet_s'][min(k,round(r['duration_s']/.01)-1),i]
                draw.text((x+10,y+402),f'Post-push quiet {quiet:.2f}s | peak tilt {r["peak_tilt_deg"]:.1f}deg | max foot drift {r["peak_foot_drift_m"]*1000:.0f}mm',font=small,fill='#d2dce8')
                if ended:
                    cause='fall criterion' if r['physical_fall'] else 'invalid numerics' if r['invalid_physics'] else 'contact/penetration limit'
                    draw.rectangle((x+8,y+170,x+630,y+234),fill=(125,25,30));draw.text((x+18,y+178),f'Ended {r["duration_s"]:.2f}s: {cause}',font=small,fill='white')
                    draw.text((x+18,y+205),'Terminal pose held; no restart is shown',font=small,fill='white')
            draw.text((12,937),'Foot movement is permitted in this diagnostic. No learning, no policy/physics changes.',font=small,fill='#d2dce8')
            writer.append_data(np.asarray(frame))
            if number==122:frame.save(REPORT/f'figures/quiet_push01_{model}_{direction}_preview.jpg',quality=90)
    reader=imageio.get_reader(video);info=reader.get_meta_data();reader.close()
    assert info['size']==(1280,960) and abs(info['duration']-20.)<.06
    (folder/f'video-{direction}-provenance.json').write_text(json.dumps(dict(case_indices=ids,repeat=0,
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest(),
        duration_s=info['duration'],fps=info['fps'],size=info['size'],backend='cpu',
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(model,direction,'VIDEO COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['v1','v2','xshape'])
    ap.add_argument('--directions',nargs='+',default=['forward','left']);a=ap.parse_args()
    for m in a.models:
        for d in a.directions:render(m,d)
