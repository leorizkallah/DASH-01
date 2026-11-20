"""Abstract validation only; no PPO updates or simulated physics trajectories."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'BalanceRL'))
import jax
import jax.numpy as jnp
from config import get_config
from env import BalanceEnv, EnvParams
from ppo import Trainer
from recovery_eval import _runner

for name in ('v1','v2','xshape'):
    cfg=get_config('return_'+name,n_envs=2,n_steps=2,batch_size=2,est_batch=2)
    env=BalanceEnv(cfg,2)
    prm=EnvParams(1.,.1,1.,3.)
    state,obs=jax.eval_shape(env.reset,jax.random.PRNGKey(1),prm)
    result=jax.eval_shape(env.step,state,jax.ShapeDtypeStruct((2,18),jnp.float32),prm)
    assert result[0].standing.keys()==state.standing.keys()
    assert result[4]['pose_ok'].shape==(2,)
    agent=Trainer(cfg,Path('abstract_validation')/name,seed=1,n_devices=1,n_eval=5)
    fn,_=_runner(agent,agent.eval_env,1,False)
    evaluated=jax.eval_shape(fn,agent.params,agent.stats.mean,agent.stats.var,agent.stats.count,
                            jax.random.PRNGKey(1234),.1,3.)
    assert evaluated[0].shape==(5,)
    print(name,'pose reference/reset/step/25s-evaluator abstract trace passed',flush=True)
