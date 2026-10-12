"""Load immutable STOP-03 demonstration training partitions for continued PPO imitation."""
import hashlib
import json
from pathlib import Path
import numpy as np


def attach(agent, path, source_checkpoint):
    import jax
    import jax.numpy as jnp
    from flax import serialization
    path=Path(path);source_checkpoint=Path(source_checkpoint)
    if agent.n_dev!=1 or not agent.cfg.freeze_obs_stats:
        raise ValueError('Rehearsal currently requires one device and frozen observation statistics')
    # Demonstrations were normalized under STOP-03 source statistics; reject mismatches.
    d=serialization.from_bytes(agent.state_dict(), source_checkpoint.read_bytes())
    for name in ('mean','var'):
        np.testing.assert_array_equal(np.asarray(getattr(agent.stats,name)),np.asarray(d['stats'][name]))
    with np.load(path,allow_pickle=False) as z:
        datasets=[]
        for task in ('hold','walk'):
            train=z[task+'_train'];valid=z[task+'_valid']
            if np.any(train & valid) or np.any(train.any(0) & valid.any(0)):
                raise ValueError('Demonstration train/validation leakage')
            x,y=z['obs'][train],z['target'][train]
            if len(x)<128 or x.shape[1]!=agent.env.obs_dim or y.shape[1]!=agent.env.action_dim:
                raise ValueError('Invalid rehearsal dimensions or insufficient examples')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite rehearsal data')
            datasets.append((jnp.asarray(x),jnp.asarray(y)))
    @jax.jit
    def sample(key):
        kh,kw=jax.random.split(key)
        batches=[]
        for k,(x,y) in zip((kh,kw),datasets):
            idx=jax.random.randint(k,(128,),0,len(x))
            batches.append((x[idx],y[idx]))
        return tuple(jnp.concatenate([batches[0][i],batches[1][i]]) for i in (0,1))
    agent.rehearsal_sample=sample
    record=dict(dataset=str(path),dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                warm_checkpoint=str(source_checkpoint),checkpoint_sha256=hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
                samples_per_minibatch=dict(hold=128,walk=128),training_samples=[len(x) for x,y in datasets],
                coefficient=agent.cfg.stop_rehearsal_coef,action_scale=.12,walking_weight=2.,
                optimizer='joint PPO objective; fresh optimizer on warm start',
                normalization='source checkpoint exact equality; frozen',held_out_used_for_updates=False)
    record_path=agent.run/'rehearsal.json'
    if record_path.exists():
        old=json.loads(record_path.read_text())
        for key in ('dataset_sha256','checkpoint_sha256','coefficient'):
            if old[key]!=record[key]: raise ValueError(f'Rehearsal resume mismatch: {key}')
    record_path.write_text(json.dumps(record,indent=2)+'\n')
