"""GPU reset/parent/evaluation/PPO checks; pilot weights never seed the real run."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=['v1','v2','xshape'])
    ap.add_argument('--parent',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from config import get_config
    from env import EnvParams
    from plant import Override
    from ppo import Trainer
    from quiet_tracking_math import validate_physics
    import quiet_tracking_eval
    start=time.monotonic()
    assert all(d.platform=='gpu' for d in jax.devices())
    cfg=get_config('quiet_'+a.model,n_envs=32,n_steps=64,batch_size=1024,est_batch=1024,
                   n_epochs=2,est_epochs=1)
    tr=Trainer(cfg,a.out.parent/('pilot_'+a.model),seed=2,n_eval=96)
    parent=tr.initialize_from(a.parent)
    assert tr.step==tr.rollout_n==tr.n_upd==0
    assert tr.env_params().push_on==0.
    for level in (-1,0,4):
        prm=EnvParams(plant_scale=1.,push_on=1.,pose_level=float(level))
        st,ob=tr.env.reset(jax.random.PRNGKey(303+level),prm,Override(pose_level=float(level)))
        assert np.max(np.abs(np.asarray(st.data.qvel)))==0.
        expected=(np.broadcast_to(tr.env.plant.quiet_start_qpos,np.shape(st.data.qpos)) if level==-1 else
                  tr.env.plant.quiet_bank[0,level,np.asarray(st.standing['pose_index'])])
        np.testing.assert_allclose(np.asarray(st.data.qpos),expected,atol=1e-6)
        _,_,r,_,info=tr.env.step(st,jnp.zeros((32,18)),prm)
        assert np.isfinite(np.asarray(r)).all() and np.isfinite(np.asarray(ob)).all()
        assert not np.asarray(info['push_start']).any(), 'quiet mode must suppress even forced push_on'
    print(a.model,'reset and no-push checks passed; parent evaluation',flush=True)
    # Integration regression: an earlier completed hold remains accepted at the
    # horizon even when the current streak is short. Safety failures still win.
    prm=EnvParams(plant_scale=1.,push_on=0.)
    st,_=tr.env.reset(jax.random.PRNGKey(71),prm,Override(pose_level=-1.))
    track=dict(st.standing,pose_succeeded=jnp.ones(32,bool),quiet_s=jnp.zeros(32),loaded_s=jnp.zeros(32))
    st=st.replace(step_n=jnp.full(32,tr.env.max_steps-1,jnp.int32),
                  t=jnp.full(32,(tr.env.max_steps-1)*tr.env.control_dt),standing=track)
    _,_,_,done,info=tr.env.step(st,jnp.zeros((32,18)),prm)
    safe=~info['fallen']&~info['task_violation']
    assert np.asarray(safe).any()
    assert np.asarray(info['quiet_success'])[np.asarray(safe)].all()
    assert not np.asarray(info['quiet_ok']).any()
    assert np.asarray(done).all()
    baseline=tr.evaluate()
    baseline_check=validate_physics(baseline)
    baseline_retention=quiet_tracking_eval.evaluate(tr,seed=1234,retention=True)
    logs=[]
    for i in range(24):
        row,_=tr.iterate()
        assert all(np.isfinite(row[k]) for k in ('rollout/reward_mean','train/kl','train/vf','train/est_loss'))
        assert row['rollout/pushes']==0
        logs.append({k:row[k] for k in ('time/env_steps','rollout/reward_mean','train/kl','train/vf',
                                       'diag/pose_joint_rms','rollout/quiet_successes')})
        if (i+1)%8==0: print(a.model,'pilot',i+1,'/24',flush=True)
    assert bool(tr._finite((tr.params,tr.opt_state,tr.est_opt_state,tr.stats)))
    probe=tr.evaluate(seed=1234)
    postpilot_check=validate_physics(probe)
    postpilot_retention=quiet_tracking_eval.evaluate(tr,seed=1234,retention=True)
    out=dict(passed=True,model=a.model,parent=parent,baseline=baseline,postpilot_eval=probe,
        baseline_physics=baseline_check,postpilot_physics=postpilot_check,
        baseline_retention=baseline_retention,postpilot_retention=postpilot_retention,
        matched_seed=1234,reference_task_rate_delta=probe['quiet_rate']-baseline['quiet_rate'],
        pilot_steps=tr.step,pilot_logs=logs,elapsed_s=time.monotonic()-start,
        devices=[str(d) for d in jax.devices()],note='Pilot weights/optimizer discarded. Full run reloads the STAND-04 parent.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(a.model,'PREFLIGHT PASSED',flush=True)


if __name__=='__main__': main()
