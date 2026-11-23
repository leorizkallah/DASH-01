"""Frozen best-policy 20s replays; four trials fixed before replay outcomes."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',choices=['v1','v2','xshape'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--platform',choices=['cpu','cuda'],default='cuda')
    a=ap.parse_args();root=a.root.resolve();a.out.mkdir(parents=True,exist_ok=True)
    os.environ['JAX_PLATFORMS']=a.platform
    sys.path.insert(0,str(root/'BalanceRL'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from config import config_from_dict
    from env import EnvParams
    from plant import Override
    from ppo import Trainer
    import quiet_tracking
    start=time.monotonic()
    assert all(d.platform==('cpu' if a.platform=='cpu' else 'gpu') for d in jax.devices())
    for name,digest in json.loads((root/'source-manifest.json').read_text())['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    ckpt=root/'BalanceRL/runs'/f'quiet01v2_{a.model}_s2/best'
    side=json.loads(ckpt.with_suffix('.json').read_text())
    cfg=dataclasses.replace(config_from_dict(side['config']),quiet_pose_retention_eval=True,
        episode_s=20.,n_envs=16,n_steps=8,batch_size=64,est_batch=64)
    original_finish=quiet_tracking.finish
    def measured_finish(env,state,data,*args):
        result=original_finish(env,state,data,*args)
        return (*result[:4],dict(result[4],video_qpos=data.qpos,video_qvel=data.qvel))
    quiet_tracking.finish=measured_finish
    agent=Trainer(cfg,a.out/'runtime',seed=2,n_devices=1,n_eval=4)
    agent.load(ckpt,with_opt=False)
    env,n=agent.eval_env,4
    original_trials=jnp.array([0,48,64,80]);selected=jnp.arange(4)
    levels=jnp.array([-1,2,3,4]);members=jnp.zeros(4)
    prm=EnvParams(plant_scale=1.,push_level=0.,push_on=0.,pose_split=2.)
    # Retain the original per-trial PRNG keys from the96-member evaluator.
    # Inference/physics are independent per environment, so only four are replayed.
    keys=jax.random.split(jax.random.PRNGKey(1234),96)[original_trials]
    ov=Override(pose_level=levels,pose_index=members)
    ov=jax.tree_util.tree_map(lambda x:jnp.broadcast_to(jnp.asarray(x,jnp.float32),(4,)),ov)
    state,obs=env._reset_v(keys,prm,ov)
    initial_q=np.asarray(state.data.qpos[selected]);initial_v=np.asarray(state.data.qvel[selected])
    print(a.model,'best',agent.step,'replaying20s, selected0/48/64/80',flush=True)

    @jax.jit
    def replay(state,obs):
        def tick(carry,_):
            st,ob,alive,passed,fallen,bad,ended,settle=carry
            action=agent._greedy(agent.params,agent.stats.normalize(ob))
            st,ob,_,done,info=env.step(st,jnp.clip(action,-1.,1.),prm)
            frame=dict(qpos=info['video_qpos'][selected],qvel=info['video_qvel'][selected],
                alive=alive[selected],quiet_s=info['quiet_s'][selected],pose_ok=info['pose_ok'][selected],
                foot_error=info['pose_foot_error'][selected],joint_rms=info['pose_joint_rms'][selected])
            ended=jnp.where(alive,info['ep_len']*env.control_dt,ended)
            settle=jnp.where(alive,info['settling_s'],settle)
            return (st,ob,alive&~done,passed|(alive&info['quiet_success']),
                    fallen|(alive&info['fallen']),bad|(alive&info['task_violation']),ended,settle),frame
        initial=(state,obs,jnp.ones(n,bool),jnp.zeros(n,bool),jnp.zeros(n,bool),jnp.zeros(n,bool),
                 jnp.zeros(n),jnp.full(n,-1.))
        return jax.lax.scan(tick,initial,None,length=env.max_steps)
    final,trace=jax.device_get(replay(state,obs))
    _,_,_,passed,fallen,bad,ended,settle=final
    archive=[json.loads(s) for s in (ckpt.parent/'quiet_pose_retention.jsonl').read_text().splitlines()]
    archived=next(e for e in reversed(archive) if e['step']==side['step'])
    reference=np.asarray(env.plant.quiet_reference['motor_q'])
    rms=np.rad2deg(np.sqrt(np.mean((initial_q[:,env.plant.act_qadr]-reference)**2,axis=1)))
    def expand(values):
        result=[None]*96
        for i,trial in enumerate((0,48,64,80)):result[trial]=values[i].item()
        return result
    archived_selected=[archived['per_trial']['passed'][i] for i in (0,48,64,80)]
    meta=dict(model=a.model,checkpoint_step=side['step'],checkpoint_sha256=hashlib.sha256(ckpt.with_suffix('.msgpack').read_bytes()).hexdigest(),
        model_path=cfg.model_path,model_sha256=hashlib.sha256((root/'BalanceRL'/cfg.model_path).read_bytes()).hexdigest(),
        batch_size=4,seed_source_batch_size=96,seed=1234,control_dt=env.control_dt,ticks=env.max_steps,selected_trials=[0,48,64,80],
        selected_labels=['Reference start','Moderate start:3deg requested','Larger start:5deg requested','Hard start:8deg requested'],
        initial_motor_rms_deg=rms.tolist(),requested_noise_deg=[0.,3.,5.,8.],
        archived_retention_passes=sum(archived['per_trial']['passed']),replay_selected_passes=int(passed.sum()),
        parity=bool(np.array_equal(passed,archived_selected)),archived_selected_passed=archived_selected,
        per_trial=dict(passed=expand(passed),physical_fall=expand(fallen),task_violation=expand(bad),
                       duration_s=expand(ended),settling_s=expand(settle)),
        protocol='20s retention replay of saved best RETURN checkpoint. Four fixed trials with original96-member PRNG keys, seed1234/test bank/full sensor noise. No pushes. No outcome selection.',
        elapsed_s=time.monotonic()-start,backend=a.platform,devices=[str(d) for d in jax.devices()],
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for i in range(4):
        valid=trace['alive'][:,i]
        assert np.isfinite(trace['qpos'][valid,i]).all() and np.isfinite(trace['qvel'][valid,i]).all()
    np.savez_compressed(a.out/'trace.npz',**trace,initial_qpos=initial_q,initial_qvel=initial_v,meta=json.dumps(meta))
    (a.out/'replay.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(a.model,'replay',meta['replay_selected_passes'],'/4; selected parity',meta['parity'],flush=True)


if __name__=='__main__':main()
