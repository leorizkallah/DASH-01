"""Frozen-policy push screen: custom evaluation only, no reward/optimizer updates."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from quiet_push_protocol import cases,force_vector,quiet_counts,DT,PUSH_TICK,PULSE_TICKS,END_TICK


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',choices=['v1','v2','xshape'],required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();a.out.mkdir(parents=True,exist_ok=True)
    os.environ['JAX_PLATFORMS']='cpu';sys.path.insert(0,str(root/'BalanceRL'))
    import jax
    import jax.numpy as jp
    import numpy as np
    from flax import serialization
    from config import config_from_dict
    from env import BalanceEnv,EnvParams
    from plant import Override
    from ppo import Trainer
    from pose_reference import features,error
    import standing,quiet_tracking
    begin=time.monotonic()
    assert all(d.platform=='cpu' for d in jax.devices())
    for name,digest in json.loads((root/'source-manifest.json').read_text())['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    ckpt=root/'BalanceRL/runs'/f'quiet01v2_{a.model}_s2/best'
    side=json.loads(ckpt.with_suffix('.json').read_text())
    cfg=dataclasses.replace(config_from_dict(side['config']),episode_s=20.,n_envs=4,n_steps=8,batch_size=16,est_batch=16)
    original_initial=standing.initial
    def initial(env,data,params,override,key):
        old=original_initial(env,data,params,override,key);zeros=jp.zeros((),jp.int32)
        old.update(push_quiet_ticks=zeros,push_loaded_ticks=zeros,pre_quiet_ticks=zeros,
            hold_seen=jp.asarray(False),hold_start=jp.asarray(-1.),hold_complete=jp.asarray(-1.),
            candidate=standing.sole_centers(data,env.plant)[:,:2],
            latch=standing.contacts(data,env.plant,env.cfg)['grounded'],up_ticks=jp.zeros(2,jp.int32),down_ticks=jp.zeros(2,jp.int32),
            landings=zeros,liftoffs=zeros,impulse=jp.zeros(3))
        return old
    standing.initial=initial
    def push(env,state,key,params):
        due=state.step_n==PUSH_TICK
        magnitude=state.push_fix[0]*env.m_total/(PULSE_TICKS*env.control_dt)
        direction=force_vector(env._R(state.data),magnitude,state.push_fix[1],jp)
        force=jp.where(due,direction,state.push_force)
        point=env._R(state.data).T@(state.data.xipos[env.plant.base_bid]-state.data.xpos[env.plant.base_bid])
        return dict(push_countdown=jp.zeros((),jp.int32),push_left=jp.where(due,PULSE_TICKS,state.push_left),
            push_force=force,push_point=point,push_dv=state.push_fix[0],push_hard=jp.asarray(True),
            push_pending=state.push_pending|due,push_since=jp.where(due,0.,state.push_since)),due
    BalanceEnv._push=push
    def finish(env,state,data,action,params,pf,started,target,tvel,kp,kd,cmd_buf,tau_last,tau_sq,key,k_frame,k_reset):
        p,c,dt=env.plant,env.cfg,env.control_dt;s=state.standing
        foot=standing.contacts(data,p,c);soles=standing.sole_centers(data,p);base=data.xpos[p.base_bid]
        pose=error(features(data.qpos[p.act_qadr],env._R(data),base,soles,jp),p.quiet_reference,c,jp)
        tilt=jp.arccos(jp.clip(-env._grav_body(data)[2],-1.,1.));v=env._vel_world(data);w=env._gyro(data)
        js=jp.sqrt(jp.mean(data.qvel[p.act_dadr]**2))
        finite=jp.isfinite(data.qpos).all()&jp.isfinite(data.qvel).all()&(jp.abs(data.qvel).max()<200.)
        body_fall=(base[2]<.65*p.height_stand)|(tilt>np.pi/4)
        contact_bad=foot['floor_violation'];bad=~finite|body_fall|contact_bad
        count=state.step_n+1
        still=(jp.linalg.norm(v)<.05)&(jp.linalg.norm(w)<.25)&(js<.30)&(jp.sqrt(foot['slip2']).max()<.02)
        anchor=jp.where(s['push_quiet_ticks']==0,base[:2],s['quiet_anchor'])
        candidate=jp.where(s['push_quiet_ticks']==0,soles[:,:2],s['candidate'])
        quiet=(pose['ok']&still&(foot['height']<=.003).all()&~bad
            &(jp.linalg.norm(base[:2]-anchor)<.03)&(jp.linalg.norm(soles[:,:2]-candidate,axis=1)<=.01).all())
        qt,lt,hold=quiet_counts(s['push_quiet_ticks'],s['push_loaded_ticks'],quiet,foot['grounded'].all(),state.step_n,jp)
        event=hold&~s['hold_seen'];seen=s['hold_seen']|hold
        hold_start=jp.where(event,count*dt-qt*dt-(PUSH_TICK+PULSE_TICKS)*dt,s['hold_start'])
        completed=jp.where(event,count*dt-(PUSH_TICK+PULSE_TICKS)*dt,s['hold_complete'])
        pre=jp.where((state.step_n<PUSH_TICK)&quiet,s['pre_quiet_ticks']+1,0)
        up=jp.where(foot['normal_force']<1.,s['up_ticks']+1,0)
        down=jp.where(foot['normal_force']>=2.,s['down_ticks']+1,0)
        latch=jp.where(up>=2,False,jp.where(down>=2,True,s['latch']))
        liftoff=s['latch']&~latch;landing=~s['latch']&latch;post=state.step_n>=PUSH_TICK
        nland=s['landings']+jp.where(post,jp.sum(landing,dtype=jp.int32),0)
        nlife=s['liftoffs']+jp.where(post,jp.sum(liftoff,dtype=jp.int32),0)
        force=jp.where(pf['push_left']>0,pf['push_force'],jp.zeros(3))
        impulse=s['impulse']+force*dt
        frame,ns,_,meas=env._frame(data,tau_last,action,state,k_frame)
        nt=dict(s,push_quiet_ticks=qt,push_loaded_ticks=lt,pre_quiet_ticks=pre,
            quiet_anchor=anchor,candidate=candidate,hold_seen=seen,hold_start=hold_start,hold_complete=completed,
            latch=latch,up_ticks=up,down_ticks=down,landings=nland,liftoffs=nlife,impulse=impulse)
        new=state.replace(data=data,key=key,step_n=count,t=state.t+dt,standing=nt,
            cmd_buf=cmd_buf,prev_target=target,prev_target_vel=tvel,prev_action=action,
            prev_vel_body=env._vel_body(data),prev_foot_xy=soles[:,:2],
            ground_ticks=jp.where(foot['grounded'],state.ground_ticks+1,0),hist=jp.concatenate([state.hist[1:],frame[None]],axis=0),
            push_left=jp.maximum(pf['push_left']-1,0),push_force=pf['push_force'],push_point=pf['push_point'],
            push_since=pf['push_since']+dt,push_dv=pf['push_dv'],push_hard=pf['push_hard'],push_pending=pf['push_pending'],
            calm_t=jp.where(pf['push_left']>0,0.,state.calm_t+dt),**env._ema(state,meas),**ns)
        obs=env._obs(new,data,foot['toe'],foot['heel'],env._vel_body(data))
        info=dict(qpos=data.qpos,qvel=data.qvel,force=force,body_fall=body_fall,invalid=~finite,contact_bad=contact_bad,
            quiet_s=qt*dt,hold_seen=seen,hold_start=hold_start,hold_complete=completed,retained=hold,
            pose_ok=pose['ok'],joint_rms=pose['joint_rms'],foot_error=pose['feet_max'],tilt_error=pose['tilt'],
            height_error=pose['height'],tilt=tilt,body_speed=jp.linalg.norm(v),angular_speed=jp.linalg.norm(w),
            foot_force=foot['normal_force'],clearance=foot['height'],slip=jp.sqrt(foot['slip2']),
            foot_drift=jp.linalg.norm(soles[:,:2]-s['anchor'],axis=1),body_xy=base[:2],
            liftoffs=nlife,landings=nland,torque_util=jp.max(jp.abs(tau_last)/jp.asarray(p.tau_peak)),
            torque=tau_last,impulse=impulse,pre_quiet_s=pre*dt)
        # Freeze a terminal state. No autoresets: every recorded trajectory is a single trial.
        return new,jp.nan_to_num(obs),jp.zeros((),jp.float32),bad|(count>=env.max_steps),info
    quiet_tracking.finish=finish
    agent=Trainer(cfg,a.out/'runtime',seed=2,n_devices=1,n_eval=4);agent.load(ckpt,with_opt=False)
    env=agent.eval_env;prm=EnvParams(plant_scale=1.,push_on=1.,pose_split=2.)
    assert abs(env.control_dt-DT)<1e-9 and cfg.hold_s_range==(0.,0.) and not cfg.reflex_enable and not cfg.dr_enable
    original_params=hashlib.sha256(serialization.to_bytes(agent.params)).hexdigest()
    allcases=cases()
    @jax.jit
    def run(forces,angles,repeats):
        keys=jax.vmap(lambda r:jax.random.PRNGKey(20261123+r))(repeats)
        ov=Override(pose_level=jp.full(4,-1.),push_dv=forces*(PULSE_TICKS*DT)/env.m_total,push_az=angles,push_el=0.)
        ov=jax.tree_util.tree_map(lambda x:jp.broadcast_to(jp.asarray(x,jp.float32),(4,)),ov)
        state,obs=env._reset_v(keys,prm,ov)
        initial_q=state.data.qpos
        def tick(carry,_):
            st,ob,alive=carry
            action=agent._greedy(agent.params,agent.stats.normalize(ob))
            candidate,ob2,_,done,info=env.step(st,jp.clip(action,-1.,1.),prm)
            frame=dict(info,alive=alive)
            # Stopping after the first failure protects traces from post-fall solver explosions.
            candidate=jax.tree_util.tree_map(lambda old,new:jp.where(alive.reshape((4,)+(1,)*(new.ndim-1)),new,old),st,candidate)
            ob2=jp.where(alive[:,None],ob2,ob)
            return (candidate,ob2,alive&~done),frame
        final,trace=jax.lax.scan(tick,(state,obs,jp.ones(4,bool)),None,length=END_TICK)
        return initial_q,trace
    traces=[];initials=[]
    for start in range(0,len(allcases),4):
        group=allcases[start:start+4];valid=len(group);group+=([allcases[0]]*(4-valid))
        initial_q,trace=jax.device_get(run(jp.array([c['force_n'] for c in group]),
            jp.array([c['azimuth_rad'] for c in group]),jp.array([c['repeat'] for c in group],jp.int32)))
        traces.append({k:v[:,:valid] for k,v in trace.items()});initials.append(initial_q[:valid])
        print(a.model,f'{min(start+4,len(allcases))}/{len(allcases)} trials',flush=True)
    trace={k:np.concatenate([t[k] for t in traces],axis=1) for k in traces[0]}
    assert hashlib.sha256(serialization.to_bytes(agent.params)).hexdigest()==original_params
    assert agent.step==side['step']
    metadata=dict(model=a.model,checkpoint_step=side['step'],checkpoint_sha256=hashlib.sha256(ckpt.with_suffix('.msgpack').read_bytes()).hexdigest(),
        model_path=cfg.model_path,model_sha256=hashlib.sha256((root/'BalanceRL'/cfg.model_path).read_bytes()).hexdigest(),
        mass_kg=env.m_total,dt=DT,push_onset_s=PUSH_TICK*DT,pulse_s=PULSE_TICKS*DT,duration_s=END_TICK*DT,
        cases=allcases,backend='cpu',weights_unchanged=True,learning_updates=0,
        protocol='Isolated fresh quiet-reference starts; identical noise per repeat across all strengths/directions; torso-COM horizontal pulse; only physical/numerical/contact failure terminates. Foot movement observed, not prohibited.',
        elapsed_s=time.monotonic()-begin,recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Scalars stay100Hz. State arrays are sampled20Hz for video/render provenance.
    q=trace.pop('qpos');v=trace.pop('qvel')
    last=[np.flatnonzero(trace['alive'][:,i]&np.isfinite(q[:,i]).all(1)&np.isfinite(v[:,i]).all(1))[-1] for i in range(len(allcases))]
    np.savez_compressed(a.out/'trace.npz',**trace,qpos_video=q[4::5],qvel_video=v[4::5],
                        qpos_terminal=q[last,np.arange(len(allcases))],qvel_terminal=v[last,np.arange(len(allcases))],
                        initial_qpos=np.concatenate(initials),meta=json.dumps(metadata))
    (a.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(a.model,'SCREEN COMPLETE',metadata['elapsed_s'],'seconds',flush=True)


if __name__=='__main__':main()
