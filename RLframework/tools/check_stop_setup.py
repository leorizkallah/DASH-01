"""GPU preflight: real plant/bundle parity and one RUN and STOP transition."""
import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from config import get_config
from train import make_eval_env
from ppo import PPO, initial_params
from bundle_warmstart import restore_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    args = ap.parse_args()
    cfg = get_config('dash_stop')
    cfg.n_envs = 8
    env = make_eval_env(cfg, 8)
    with TemporaryDirectory() as tmp:
        agent = PPO(cfg, env, Path(tmp), cfg.total_steps)
        restore_bundle(agent, args.bundle)
        params = initial_params(cfg)
        state, obs = env.reset(jax.random.PRNGKey(3), params)
        for stop in (0.0, 1.0):
            state = state.replace(v_cmd=jnp.ones(8), stop_cmd=jnp.full((8,), stop),
                                  cmd_left=jnp.full((8,), 100.0))
            obs = obs.at[:, env.actor_dim - 3].set(1 / cfg.v_max)
            obs = obs.at[:, env.actor_dim - 2].set(1 - stop)
            action = agent._act_greedy(agent.params, agent.stats.normalize(obs))
            state, obs, reward, _, info = env.step(state, jnp.clip(action, -1, 1), params)
            assert np.isfinite(np.asarray(obs)).all() and np.isfinite(np.asarray(reward)).all()
            assert np.all(np.asarray(info['stop_cmd']) == stop)
            if stop == 0:
                assert np.all(np.asarray(info['reward_terms']['stand_hold']) == 0)
            else:
                assert np.all(np.asarray(info['reward_terms']['air_time']) == 0)
        print('[preflight] bundle parity, finite stepping, command timing, STOP reward gates: PASS', flush=True)


if __name__ == '__main__':
    main()
