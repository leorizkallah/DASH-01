"""Frozen-parent diagnostic replay; record hold blockers and paired reset/noise controls.

No optimizer steps, no changed reward/verdict, no writes to the frozen source.
The measurement wrapper adds telemetry to the original finish() return only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--model',required=True,choices=['v1','v2','xshape'])
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();root=a.root.resolve();a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'BalanceRL'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from config import get_config
    from env import EnvParams
    from plant import Override
    from ppo import Trainer
    from pose_reference import features,error
    import quiet_tracking as task
    import standing
    start=time.monotonic()
    assert all(d.platform=='gpu' for d in jax.devices())
    manifest=json.loads((root/'source-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name

    original_reset=task.reset_pose
    def diagnostic_reset(env,key,params,override):
        q,level,index=original_reset(env,key,params,override)
        return jnp.where(level==-2,jnp.asarray(env.plant.key_qpos),q),level,index
    task.reset_pose=diagnostic_reset  # -2 is used only by the explicit keyframe control.
    original_finish=task.finish
    def measured_finish(env,state,data,*args):
        result=original_finish(env,state,data,*args)
        c,p=env.cfg,env.plant
        foot=standing.contacts(data,p,c);soles=standing.sole_centers(data,p)
        base=data.xpos[p.base_bid];R=env._R(data)
        pose=error(features(data.qpos[p.act_qadr],R,base,soles,jnp),p.quiet_reference,c,jnp)
        v,w=env._vel_world(data),env._gyro(data)
        js=jnp.sqrt(jnp.mean(data.qvel[p.act_dadr]**2))
        anchor=jnp.where(state.standing['quiet_s']<=0,base[:2],state.standing['quiet_anchor'])
        drift=jnp.linalg.norm(soles[:,:2]-state.standing['anchor'],axis=1)
        values=dict(joint_rms=pose['joint_rms'],joint_max=pose['joint_max'],tilt_error=pose['tilt'],
            foot_error=pose['feet_max'],height_error=pose['height'],body_speed=jnp.linalg.norm(v),
            angular_speed=jnp.linalg.norm(w),motor_speed=js,slip_speed=jnp.sqrt(foot['slip2']).max(),
            clearance=foot['height'].max(),anchor_drift=jnp.linalg.norm(base[:2]-anchor),
            foot_drift=drift.max(),min_normal_force=foot['normal_force'].min())
        blockers=jnp.array([pose['joint_rms']>c.return_joint_rms_rad,
            pose['joint_max']>c.return_joint_max_rad,pose['tilt']>c.return_tilt_rad,
            pose['feet_max']>c.return_foot_error_m,pose['height']>c.return_height_error_m,
            jnp.linalg.norm(v)>=.05,jnp.linalg.norm(w)>=.25,js>=.30,
            jnp.sqrt(foot['slip2']).max()>=.02,(foot['height']>.003).any(),
            jnp.linalg.norm(base[:2]-anchor)>=.03])
        info=dict(result[4],diagnostics=dict(values,blockers=blockers,
            loaded=foot['grounded'].all(),qpos=data.qpos,qvel=data.qvel,
            quiet_start=state.t+env.control_dt-result[4]['quiet_s']))
        return (*result[:4],info)
    task.finish=measured_finish
    cfg=get_config('quiet_'+a.model,n_envs=32,n_steps=64,batch_size=1024,est_batch=1024,
                   n_epochs=2,est_epochs=1)
    agent=Trainer(cfg,a.out/'runtime',seed=2,n_eval=96)
    parent=agent.initialize_from(root/f'parents/{a.model}/best')
    env=agent.eval_env;n=96;ticks=env.max_steps;ids=jnp.arange(16)
    levels=jnp.repeat(jnp.arange(-1,5),16);members=jnp.tile(jnp.arange(16),6)

    @jax.jit
    def run(noise,reset_level):
        prm=EnvParams(plant_scale=noise,push_level=0.,push_on=0.,pose_split=2.)
        lv=levels.at[:16].set(reset_level)
        state,obs=env.reset(jax.random.PRNGKey(1234),prm,Override(pose_level=lv,pose_index=members))
        initial_q=state.data.qpos[ids]
        def tick(carry,_):
            state,obs,alive,passed,fallen,bad=carry
            action=agent._greedy(agent.params,agent.stats.normalize(obs))
            state,obs,_,done,info=env.step(state,jnp.clip(action,-1.,1.),prm)
            frame={k:v[ids] for k,v in info['diagnostics'].items()}
            frame.update(alive=alive[ids],done=done[ids],passed=info['quiet_success'][ids],
                quiet_s=info['quiet_s'][ids],quiet_ok=info['quiet_ok'][ids],
                timeout=info['pose_timeout'][ids],fallen=info['fallen'][ids],
                task_bad=info['task_violation'][ids],settling_s=info['settling_s'][ids])
            return (state,obs,alive&~done,passed|(alive&info['quiet_success']),
                    fallen|(alive&info['fallen']),bad|(alive&info['task_violation'])),frame
        initial=(state,obs,jnp.ones(n,bool),jnp.zeros(n,bool),jnp.zeros(n,bool),jnp.zeros(n,bool))
        final,trace=jax.lax.scan(tick,initial,None,length=ticks)
        return initial_q,final[3:],trace

    baseline=json.loads((root/f'validation/pilot_{a.model}/quiet_pose_evaluations.jsonl').read_text().splitlines()[0])
    summaries={}
    for label,noise,reset in [('reference_noise',1.,-1),('reference_clean',0.,-1),
                              ('keyframe_noise',1.,-2),('keyframe_clean',0.,-2)]:
        q,(passed,fallen,bad),trace=jax.device_get(run(jnp.asarray(noise),jnp.asarray(reset)))
        summary=dict(passed=passed[:16].tolist(),physical_fall=fallen[:16].tolist(),
                     task_violation=bad[:16].tolist(),successes=int(passed[:16].sum()),trials=16)
        if label=='reference_noise':
            summary['archived_passed']=baseline['per_trial']['passed'][:16]
            summary['same_verdicts_as_archive']=bool(np.array_equal(passed[:16],summary['archived_passed']))
        np.savez_compressed(a.out/f'{label}.npz',**trace,initial_qpos=q)
        summaries[label]=summary
        print(a.model,label,summary['successes'],'/16',flush=True)
    record=dict(model=a.model,parent=parent,seed=1234,batch_size=96,dt=env.control_dt,ticks=ticks,
        blocker_names=['joint_rms','joint_max','tilt_error','foot_error','height_error','body_speed',
            'angular_speed','motor_speed','slip_speed','clearance','anchor_drift'],
        limits=dict(joint_rms=cfg.return_joint_rms_rad,joint_max=cfg.return_joint_max_rad,
            tilt_error=cfg.return_tilt_rad,foot_error=cfg.return_foot_error_m,height_error=cfg.return_height_error_m,
            body_speed=.05,angular_speed=.25,motor_speed=.30,slip_speed=.02,clearance=.003,anchor_drift=.03),
        cases=summaries,elapsed_s=time.monotonic()-start,
        note='Frozen parent, no PPO. Same seed and 96-member layout; only first16 reset source/noise varied. Sensor-noise-off is a diagnostic, not a new training setting.')
    (a.out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    assert agent.step==0
    print(a.model,'DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__': main()
