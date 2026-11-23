"""CPU abstract shape check only; GPU physical/numerical checks remain mandatory."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'BalanceRL'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+',default=['v1','v2','xshape'])
    a=ap.parse_args()
    import jax
    import jax.numpy as jnp
    from config import get_config
    from env import BalanceEnv,EnvParams
    from plant import Override
    for model in a.models:
        for retention in (False,True):
            env=BalanceEnv(get_config('quiet_'+model,quiet_pose_retention_eval=retention,
                                     episode_s=20. if retention else 12.),n_envs=2)
            prm=EnvParams(plant_scale=1.,push_on=0.,pose_level=4.)
            st,obs=jax.eval_shape(env.reset,jax.random.PRNGKey(1),prm,Override(pose_level=4.))
            result=jax.eval_shape(env.step,st,jnp.zeros((2,18)),prm)
            assert result[2].shape==(2,)
            assert result[4]['quiet_success'].shape==(2,)
            print(model,'retention' if retention else 'tracking','abstract reset/step OK',obs.shape,flush=True)


if __name__=='__main__': main()
