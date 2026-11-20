"""Abstract JAX tracing only: catch state dtype/shape errors without physics execution."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()/'BalanceRL'))
import jax
import jax.numpy as jnp
from config import get_config
from env import BalanceEnv, EnvParams

for name in ('v1', 'v2', 'xshape'):
    env = BalanceEnv(get_config('recover_'+name), 2)
    prm = EnvParams(1., .1, 1., 3.)
    state, obs = jax.eval_shape(env.reset, jax.random.PRNGKey(1), prm)
    result = jax.eval_shape(env.step, state, jax.ShapeDtypeStruct((2,18), jnp.float32), prm)
    assert result[0].standing.keys() == state.standing.keys()
    assert result[2].shape == (2,)
    print(name, 'reset/step abstract trace passed', flush=True)
