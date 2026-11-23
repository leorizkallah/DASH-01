"""Return from a feasible, zero-velocity posture disturbance; no external pushes."""
import jax
import jax.numpy as jnp
import numpy as np

from plant import Override
from standing_math import support_distance
from pose_reference import features,error
from quiet_tracking_math import hold,accept


def reset_pose(env,key,params,override):
    p=env.plant
    kl,km,ki=jax.random.split(key,3)
    top=jnp.clip(jnp.asarray(params.pose_level,jnp.int32),0,4)
    u=jax.random.uniform(km)
    # 20% exact target, 30% easier levels, 50% current frontier.
    easier=jax.random.randint(kl,(),0,top+1)
    level=jnp.where(u<.2,-1,jnp.where(u<.5,easier,top))
    level=jnp.where(jnp.isfinite(override.pose_level),override.pose_level,level).astype(jnp.int32)
    split=jnp.clip(jnp.asarray(params.pose_split,jnp.int32),0,2)
    count=jnp.asarray(p.quiet_bank_counts)[split]
    index=jax.random.randint(ki,(),0,count)
    index=jnp.where(jnp.isfinite(override.pose_index),override.pose_index,index).astype(jnp.int32)%count
    q=jnp.asarray(p.quiet_bank)[split,jnp.clip(level,0,4),index]
    return jnp.where(level<0,jnp.asarray(p.quiet_start_qpos),q),level,index


def finish(env,state,data,action,params,pf,started,target,tvel,kp,kd,cmd_buf,
           tau_last,tau_sq,key,k_frame,k_reset):
    c,p,dt=env.cfg,env.plant,env.control_dt
    import standing
    foot=standing.contacts(data,p,c)
    soles=standing.sole_centers(data,p)
    track=state.standing
    base=data.xpos[p.base_bid]
    current=features(data.qpos[p.act_qadr],env._R(data),base,soles,jnp)
    pose=error(current,p.quiet_reference,c,jnp)
    v,w=env._vel_world(data),env._gyro(data)
    js=jnp.sqrt(jnp.mean(data.qvel[p.act_dadr]**2))
    tilt=jnp.arccos(jnp.clip(-env._grav_body(data)[2],-1.,1.))
    drift=jnp.linalg.norm(soles[:,:2]-track['anchor'],axis=1)
    air_s=jnp.where(foot['height']>.005,track['air_s']+dt,0.)
    step_bad=(air_s>.10).any()|(drift>c.quiet_pose_max_foot_drift_m).any()
    finite=jnp.isfinite(data.qpos).all()&jnp.isfinite(data.qvel).all()&(jnp.abs(data.qvel).max()<200.)
    fallen=~finite|(base[2]<.65*p.height_stand)|(tilt>np.pi/4)|foot['floor_violation']
    still=(jnp.linalg.norm(v)<.05)&(jnp.linalg.norm(w)<.25)&(js<.30)&(jnp.sqrt(foot['slip2']).max()<.02)
    anchor=jnp.where(track['quiet_s']<=0,base[:2],track['quiet_anchor'])
    quiet=pose['ok']&still&(foot['height']<=.003).all()&(jnp.linalg.norm(base[:2]-anchor)<.03)&~fallen&~step_bad
    t=state.t+dt
    quiet_s,loaded_s,quiet_ok=hold(track['quiet_s'],track['loaded_s'],quiet,foot['grounded'].all(),t,dt,c,jnp)
    new_success=quiet_ok&~track['pose_succeeded']
    succeeded=track['pose_succeeded']|quiet_ok
    settling=jnp.where(new_success,t-quiet_s,track['settling_s'])
    timeout=(t>c.quiet_pose_max_start_s+c.quiet_pose_hold_s+.05)&~succeeded
    violation=step_bad|timeout
    distance=support_distance(data.subtree_com[0,:2],foot['points'],foot['active_points'],jnp)
    support=jnp.where(foot['active_points'].any(),jnp.exp(-(distance/.02)**2),0.)
    q=data.qpos[p.act_qadr]; half=(env.q_hi-env.q_lo)/2; middle=(env.q_hi+env.q_lo)/2
    # Target attraction remains active while moving; stillness pays only near the target.
    terms=dict(alive=.5*jnp.asarray(~fallen,jnp.float32), upright=.5*jnp.exp(-(tilt/.2)**2),
        support=.5*support, pose=c.quiet_pose_weight*pose['score'],
        pose_error=-.5*jnp.mean(jnp.clip(jnp.array([
            (pose['joint_rms']/c.return_joint_scale_rad)**2,(pose['tilt']/c.return_tilt_scale_rad)**2,
            (pose['feet_max']/c.return_foot_scale_m)**2,(pose['height']/c.return_height_scale_m)**2]),0.,4.)),
        quiet=2.*pose['ok']*foot['grounded'].all()*jnp.exp(-jnp.sum(v*v)/.05**2-jnp.sum(w*w)/.25**2-js**2/.30**2),
        torque=-.5*jnp.mean(tau_sq/p.decimation/(jnp.asarray(p.tau_peak)*state.draw.torque_scale)**2),
        action_rate=-.02*jnp.sum((action-state.prev_action)**2),
        slip=-.5*jnp.mean(jnp.clip(foot['slip2']/.1**2,0.,4.)), gain=-.02*jnp.mean(kp/c.kp_hi),
        joint_limit=-jnp.mean(jnp.clip((jnp.abs(q-middle)/half-.95)/.05,0.,1.)),
        foot_drift=-.1*jnp.mean(jnp.clip(drift**2/.03**2,0.,4.)),
        flat_missing=-.1*pose['ok']*jnp.mean((~foot['flat']).astype(jnp.float32)))
    reward=jnp.where(finite,sum(terms.values())*dt/.01+2.*new_success-20.*fallen-5.*(violation&~fallen),-20.)
    count=state.step_n+1
    truncated=(count>=env.max_steps)&~fallen&~violation
    done=fallen|violation|truncated
    frame,ns,_,measurements=env._frame(data,tau_last,action,state,k_frame)
    nt=dict(track,quiet_anchor=anchor,quiet_s=quiet_s,loaded_s=loaded_s,air_s=air_s,
            pose_succeeded=succeeded,settling_s=settling)
    new=state.replace(data=data,key=key,step_n=count,t=t,cmd_buf=cmd_buf,prev_target=target,
        prev_target_vel=tvel,prev_action=action,prev_vel_body=env._vel_body(data),prev_foot_xy=soles[:,:2],
        ground_ticks=jnp.where(foot['grounded'],state.ground_ticks+1,0),
        hist=jnp.concatenate([state.hist[1:],frame[None]],axis=0),ep_return=state.ep_return+reward,
        ep_len=count,standing=nt,**env._ema(state,measurements),**ns)
    obs=env._obs(new,data,foot['toe'],foot['heel'],env._vel_body(data))
    success=truncated&accept(succeeded,quiet_s,loaded_s,c,jnp)
    info=dict(reward_terms=terms,fallen=fallen,task_violation=violation,truncated=truncated,
        quiet_fall=fallen|violation,push_start=jnp.asarray(False),push_ok=jnp.asarray(False),
        push_fail=jnp.asarray(False),push_dv=jnp.zeros(()),push_hard=jnp.asarray(False),
        ep_return=new.ep_return,ep_len=count,kp_mean=kp.mean(),kd_mean=kd.mean(),
        torque_util=jnp.mean(jnp.abs(tau_last)/jnp.asarray(p.tau_peak)),tilt_deg=jnp.rad2deg(tilt),
        com_err=distance,airborne=jnp.mean((~foot['grounded']).astype(jnp.float32)),held=jnp.zeros(()),
        com_shift=state.draw.com_shift,obs_final=jnp.nan_to_num(obs),meas=measurements,
        target=target,kp=kp,kd=kd,action=action,quiet_trial=done,quiet_success=success,
        quiet_ok=quiet_ok,quiet_s=quiet_s,settling_s=settling,pose_ok=pose['ok'],
        pose_joint_rms=pose['joint_rms'],pose_foot_error=pose['feet_max'],pose_tilt_error=pose['tilt'],
        pose_height_error=pose['height'],pose_level=track['pose_level'],pose_timeout=timeout,
        normal_force=foot['normal_force'],contact_slip=jnp.sqrt(foot['slip2']),foot_clearance=foot['height'])
    reset,reset_obs=env._reset_one(k_reset,params,Override())
    result=jax.tree_util.tree_map(lambda a,b:jnp.where(done,a,b),reset,new)
    return result,jnp.nan_to_num(jnp.where(done,reset_obs,obs)),reward.astype(jnp.float32),done,info
