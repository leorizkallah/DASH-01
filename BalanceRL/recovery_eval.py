"""Independent first-episode recovery evaluations and direction-balanced curriculum probes."""
import json
import jax
import jax.numpy as jnp
import numpy as np

from env import BalanceEnv, EnvParams
from plant import Override
from recovery_curriculum import update as update_curriculum

LADDER = (0., .10, .15, .20, .30)


def _runner(agent, env, quiet_n, probing):
    pose_mode = agent.cfg.recovery_pose_return
    n, ticks = env.n_envs, (2500 if pose_mode else 2000)
    if probing:
        sectors = np.tile(np.arange(8), (n-quiet_n)//8)
        sectors = np.r_[np.zeros(quiet_n, int), sectors]
    else:
        sectors = np.tile(np.floor(np.arange(n//5)*8/(n//5)).astype(int), 5)

    def run(weights, mean, var, count, key, level, stage):
        from ppo import ObsStats
        normalization = ObsStats(mean, var, count)
        kr, ka, kd = jax.random.split(key, 3)
        if probing:
            angles = (jnp.asarray(sectors)+jax.random.uniform(ka, (n,)))*2*np.pi/8
            levels = jnp.where(jnp.arange(n) < quiet_n, 0., level*(.9+.1*jax.random.uniform(kd, (n,))))
        else:
            angles = jnp.asarray(np.tile(np.arange(n//5)*2*np.pi/(n//5), 5))
            levels = jnp.asarray(np.repeat(np.array(LADDER), n//5))
        prm = EnvParams(plant_scale=1., push_level=level, push_on=1., push_stage=stage)
        # NaN elevation means random elevation according to the staged task.
        state, obs = env.reset(kr, prm, Override(push_dv=levels, push_az=angles))
        def tick(carry, _):
            state, obs, alive, recovered, quiet, ended, fell, violated, steps, path, settle, peak_torque = carry[:12]
            action = agent._greedy(weights, normalization.normalize(obs))
            state, obs, _, done, info = env.step(state, jnp.clip(action, -1., 1.), prm)
            recovered = recovered | (alive & info['push_ok'])
            quiet = jnp.where(alive, info['quiet_ok'], quiet)
            ended = jnp.where(alive & done, info['ep_len'], ended)
            fell = fell | (alive & info['fallen'])
            violated = violated | (alive & info['task_violation'])
            steps = steps+jnp.where(alive, info['step_event'], 0)
            path = path+jnp.where(alive, info['path_increment'], 0.)
            settle = jnp.where(alive & info['push_ok'], info['settling_time'], settle)
            peak_torque = jnp.maximum(peak_torque, jnp.where(alive, info['torque_util'], 0.))
            next_carry = (state, obs, alive & ~done, recovered, quiet, ended, fell, violated,
                          steps, path, settle, peak_torque)
            if pose_mode:
                balanced, balance_time, return_time, joint_error, foot_error, tilt_error, height_error = carry[12:]
                balanced = balanced | (alive & info['balance_recovered'])
                balance_time = jnp.where(alive & info['balance_recovered'],info['balance_time'],balance_time)
                return_time = jnp.where(alive & info['push_ok'],info['pose_return_time'],return_time)
                joint_error = jnp.where(alive,info['pose_joint_rms'],joint_error)
                foot_error = jnp.where(alive,info['pose_foot_error'],foot_error)
                tilt_error = jnp.where(alive,info['pose_tilt_error'],tilt_error)
                height_error = jnp.where(alive,info['pose_height_error'],height_error)
                next_carry += (balanced,balance_time,return_time,joint_error,foot_error,tilt_error,height_error)
            return next_carry, None
        initial = (state, obs, jnp.ones(n, bool), jnp.zeros(n, bool), jnp.zeros(n, bool),
            jnp.full(n, ticks), jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32),
            jnp.zeros(n), jnp.full(n, -1.), jnp.zeros(n))
        if pose_mode:
            initial += (jnp.zeros(n,bool),jnp.full(n,-1.),jnp.full(n,-1.),
                        jnp.zeros(n),jnp.zeros(n),jnp.zeros(n),jnp.zeros(n))
        return jax.lax.scan(tick, initial, None, length=ticks)[0][2:]
    return jax.jit(run), sectors


def evaluate(agent, params=None, stats=None, seed=1234, probe=False):
    if probe:
        from dataclasses import replace
        if not hasattr(agent, '_recovery_probe_env'):
            agent._recovery_probe_env = BalanceEnv(replace(agent.cfg, push_interval_s=(100., 100.)), agent.cfg.recovery_probe_trials)
        env = agent._recovery_probe_env
        quiet_n = env.n_envs//6  # 40 quiet, 200 near-level pushes
        assert quiet_n >= 40 and (env.n_envs-quiet_n) % 8 == 0
        attr = '_recovery_probe_run'
    else:
        env = agent.eval_env
        assert env.n_envs % 5 == 0
        quiet_n = env.n_envs//5
        attr = '_recovery_eval_run'
    if not hasattr(agent, attr):
        setattr(agent, attr, _runner(agent, env, quiet_n, probe))
    runner, sectors = getattr(agent, attr)
    p = agent.params if params is None else params
    s = agent.stats if stats is None else stats
    stage = float(agent.cur.get('push_stage', 0)) if probe else 3.
    level = float(agent.cur['push_level'])
    values = jax.device_get(runner(p, s.mean, s.var, s.count, jax.random.PRNGKey(seed), level, stage))
    alive, recovered, quiet, ended, fell, violated, steps, path, settle, torque = values[:10]
    no_push = np.arange(env.n_envs) < quiet_n
    # Both quiet and pushed trials require a quiet final five seconds and no late wandering.
    passed = alive & quiet & (no_push | recovered)
    quiet_rate = float(passed[no_push].mean())
    out = dict(step=agent.step, protocol='recovery-v1: full 20 s + final 5 s quiet; pushed trials also settle within 3 s + 5 s after actual pulse end',
        score=float(passed.mean()), quiet_falls=int(fell[no_push].sum()), quiet_rate=quiet_rate,
        full_duration_fraction=float(alive.mean()), physical_fall_fraction=float(fell.mean()),
        step_violation_fraction=float(violated.mean()), stage=stage, level=level, seed=seed,
        trials=env.n_envs, per_trial=dict(passed=passed.tolist(), duration_s=(ended*.01).tolist(),
            physical_fall=fell.tolist(), task_violation=violated.tolist(), steps=steps.tolist(),
            sole_path_m=path.tolist(), settling_s=settle.tolist(), max_mean_torque_util=torque.tolist()))
    if agent.cfg.recovery_pose_return:
        balanced,balance_time,return_time,joint_error,foot_error,tilt_error,height_error = values[10:]
        out['protocol'] = 'pose-return-v1: first 25 s; regain support within 3 s, start reference-posture hold within 4 s after that, hold 5 s; final 5 s quiet at reference posture; recovered floor location allowed'
        out['per_trial'].update(balance_recovered=balanced.tolist(),balance_time_s=balance_time.tolist(),
            pose_return_time_s=return_time.tolist(),final_joint_rms_rad=joint_error.tolist(),
            final_foot_error_m=foot_error.tolist(),final_tilt_error_rad=tilt_error.tolist(),
            final_height_error_m=height_error.tolist())
    if probe:
        out.update(hard_trials=int((~no_push).sum()), hard_rate=float(passed[~no_push].mean()),
                   sector_rates=[float(passed[(sectors == i) & ~no_push].mean()) for i in range(8)])
        name = 'recovery_curriculum_probes.jsonl'
    else:
        out['survival'] = {f'{v:.2f}':float(passed[i*quiet_n:(i+1)*quiet_n].mean()) for i,v in enumerate(LADDER)}
        out['qualified'] = quiet_rate >= .95
        name = 'recovery_evaluations.jsonl'
    with (agent.run/name).open('a') as f:
        f.write(json.dumps(out)+'\n')
    return out

