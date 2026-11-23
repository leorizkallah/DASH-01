"""First-episode quiet-pose tests, independent from training starts and push tasks."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from env import EnvParams
from plant import Override


def _runner(agent,env=None):
    env=agent.eval_env if env is None else env
    n=env.n_envs
    assert n%6==0, 'quiet evaluation needs equal reference + five difficulty groups'
    group=n//6
    levels=jnp.repeat(jnp.arange(-1,5),group)
    members=jnp.tile(jnp.arange(group),6)

    def run(weights,stats,key,split):
        prm=EnvParams(plant_scale=1.,push_level=0.,push_on=0.,pose_split=split)
        state,obs=env.reset(key,prm,Override(pose_level=levels,pose_index=members))
        def tick(carry,_):
            st,ob,alive,passed,fallen,bad,settling,joint,feet,tilt,height,duration=carry
            action=agent._greedy(weights,stats.normalize(ob))
            st,ob,_,done,info=env.step(st,jnp.clip(action,-1.,1.),prm)
            passed=passed|(alive&info['quiet_success'])
            fallen=fallen|(alive&info['fallen'])
            bad=bad|(alive&info['task_violation'])
            duration=jnp.where(alive,info['ep_len']*env.control_dt,duration)
            settling=jnp.where(alive,info['settling_s'],settling)
            joint=jnp.where(alive,info['pose_joint_rms'],joint)
            feet=jnp.where(alive,info['pose_foot_error'],feet)
            tilt=jnp.where(alive,info['pose_tilt_error'],tilt)
            height=jnp.where(alive,info['pose_height_error'],height)
            return (st,ob,alive&~done,passed,fallen,bad,settling,joint,feet,tilt,height,duration),None
        zero=jnp.zeros(n)
        initial=(state,obs,jnp.ones(n,bool),jnp.zeros(n,bool),jnp.zeros(n,bool),jnp.zeros(n,bool),
                 jnp.full(n,-1.),zero,zero,zero,zero,zero)
        return jax.lax.scan(tick,initial,None,length=env.max_steps)[0][3:]
    return jax.jit(run)


def evaluate(agent,params=None,stats=None,seed=1234,probe=False,retention=False):
    if retention and probe:
        raise ValueError('Retention is separate from curriculum promotion probes')
    attr='_quiet_pose_retention_run' if retention else '_quiet_pose_eval_run'
    if not hasattr(agent,attr):
        env=agent.eval_env
        if retention:
            from dataclasses import replace
            from env import BalanceEnv
            env=BalanceEnv(replace(agent.cfg,quiet_pose_retention_eval=True,
                                  episode_s=agent.cfg.quiet_pose_retention_s),env.n_envs)
            agent._quiet_pose_retention_env=env
        setattr(agent,attr,_runner(agent,env))
    p=agent.params if params is None else params
    s=agent.stats if stats is None else stats
    result=jax.device_get(getattr(agent,attr)(p,s,jax.random.PRNGKey(seed),1. if probe else 2.))
    passed,fallen,bad,settling,joint,feet,tilt,height,duration=result
    group=len(passed)//6
    names=['reference']+[str(i) for i in range(5)]
    rates={name:float(passed[i*group:(i+1)*group].mean()) for i,name in enumerate(names)}
    level=int(agent.cur.get('pose_level',0))
    quiet_rate=rates['reference']
    score=float(passed.mean())
    protocol='quiet-pose-v2: 12 s, zero initial velocity, no pushes; timely five-second hold completion latched; survive full episode; no whole-foot lift >5 mm for >100 ms or drift >50 mm'
    if retention:
        protocol=f'quiet-pose-v2-retention: {agent.cfg.quiet_pose_retention_s:g} s; timely initial five-second hold AND independent final five-second quiet window; no original return deadline applied to final window'
    out=dict(step=agent.step,protocol=protocol,
        trials=len(passed),seed=seed,split='probe' if probe else 'test',level=level,
        rates=rates,survival=rates,quiet_rate=quiet_rate,quiet_falls=int(fallen[:group].sum()),
        score=score,qualified=quiet_rate>=.95,
        per_trial=dict(passed=passed.tolist(),physical_fall=fallen.tolist(),task_violation=bad.tolist(),
            settling_s=settling.tolist(),final_joint_rms_rad=joint.tolist(),final_foot_error_m=feet.tolist(),
            final_tilt_error_rad=tilt.tolist(),final_height_error_m=height.tolist(),duration_s=duration.tolist()))
    name='quiet_pose_retention.jsonl' if retention else 'quiet_pose_probes.jsonl' if probe else 'quiet_pose_evaluations.jsonl'
    with (agent.run/name).open('a') as f:
        f.write(json.dumps(out)+'\n')
    return out
