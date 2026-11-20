"""Fixed first-episode standing ladder, with settled outcomes and no reset credit."""
import json
import jax
import jax.numpy as jnp
import numpy as np

from env import EnvParams
from plant import Override

LADDER = (0., .02, .05, .10, .15)


def evaluate(agent, params=None, stats=None, seed=1234):
    env = agent.eval_env
    n = env.n_envs
    if n % len(LADDER):
        raise ValueError('standing evaluation count must be divisible by five')
    rungs = np.repeat(np.array(LADDER), n//5)
    angles = np.tile(np.arange(n//5)*2*np.pi/(n//5), 5)
    if not hasattr(agent, '_standing_eval_fn'):
        def run(weights, mean, var, count, key):
            from ppo import ObsStats
            normalization = ObsStats(mean, var, count)
            prm = EnvParams(plant_scale=1., push_level=.15, push_on=1.)
            override = Override(push_dv=jnp.asarray(rungs), push_az=jnp.asarray(angles), push_el=0.)
            state, obs = env.reset(key, prm, override)
            def tick(carry, _):
                state, obs, alive, succeeded, quiet, ended, fell, stepped = carry
                action = agent._greedy(weights, normalization.normalize(obs))
                state, obs, _, done, info = env.step(state, jnp.clip(action, -1, 1), prm)
                succeeded = succeeded | (alive & info['push_ok'])
                quiet = jnp.where(alive, info['quiet_ok'], quiet)
                ended = jnp.where(alive & done, info['ep_len'], ended)
                fell = fell | (alive & info['fallen'])
                stepped = stepped | (alive & info['task_violation'])
                return (state, obs, alive & ~done, succeeded, quiet, ended, fell, stepped), None
            initial = (state, obs, jnp.ones(n, bool), jnp.zeros(n, bool), jnp.zeros(n, bool),
                       jnp.full(n, 2000), jnp.zeros(n, bool), jnp.zeros(n, bool))
            return jax.lax.scan(tick, initial, None, length=2000)[0][2:]
        agent._standing_eval_fn = jax.jit(run)
    p = agent.params if params is None else params
    s = agent.stats if stats is None else stats
    alive, ok, quiet, ended, fell, stepped = jax.device_get(agent._standing_eval_fn(
        p, s.mean, s.var, s.count, jax.random.PRNGKey(seed)))
    passed = alive & np.where(rungs == 0, quiet, ok)
    rates = {f'{level:.2f}': float(passed[rungs == level].mean()) for level in LADDER}
    out = dict(step=agent.step, score=float(passed.mean()), survival=rates, quiet_falls=int(fell[rungs == 0].sum()),
               protocol='nominal plant, full sensor noise, 20 s first episodes, fixed directions, one push at 5 s',
               levels_mps=list(LADDER), trials=n, full_duration_fraction=float(alive.mean()),
               physical_fall_fraction=float(fell.mean()), step_violation_fraction=float(stepped.mean()),
               per_trial=dict(level=rungs.tolist(), duration_s=(ended*.01).tolist(),
                              passed=passed.tolist(), physical_fall=fell.tolist(), step_violation=stepped.tolist()))
    with (agent.run/'standing_evaluations.jsonl').open('a') as f:
        f.write(json.dumps(out)+'\n')
    return out
