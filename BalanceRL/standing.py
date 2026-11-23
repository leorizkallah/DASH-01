"""Three-CAD planted-standing reward, first-episode outcomes and perturbation gates."""
import jax
import jax.numpy as jnp
import numpy as np

from standing_math import contacts as measure_contacts, sole_centers as centers, support_distance
from plant import Override


def sole_centers(data, p):
    return centers(data, p, jnp)


def contacts(data, p, c):
    return measure_contacts(data, p, c, jnp)


def initial(env, data, params, override, key):
    forced = jnp.isfinite(override.push_dv)
    enabled = (params.push_on > 0) & (jax.random.uniform(jax.random.fold_in(key, 114)) >= .20)
    # Forced evaluation rungs bypass the 20% quiet-episode mixture.
    result = dict(anchor=sole_centers(data, env.plant)[:, :2],
                quiet_anchor=data.xpos[env.plant.base_bid, :2], quiet_s=jnp.zeros(()),
                loaded_s=jnp.zeros(()), air_s=jnp.zeros(2), quiet_start=jnp.zeros(()),
                noise_scale=jnp.asarray(params.plant_scale),
                episode_push_on=jnp.where(forced, override.push_dv > 0, enabled))
    if env.cfg.recovery_steps:
        from recovery_logic import initial as recovery_initial
        result.update(recovery_initial(result['anchor'], data.xpos[env.plant.base_bid, :2],
                                      contacts(data, env.plant, env.cfg)['grounded'], jnp))
    if env.cfg.recovery_pose_return:
        from pose_reference import initial_phase
        result.update(initial_phase(jnp))
    if env.cfg.quiet_pose_tracking:
        result.update(pose_succeeded=jnp.asarray(False),settling_s=jnp.asarray(-1.))
    return result


def finish(env, state, data, action, params, pf, started, target, tvel, kp, kd, cmd_buf,
           tau_last, tau_sq, key, k_frame, k_reset):
    if env.cfg.quiet_pose_tracking:
        import quiet_tracking
        return quiet_tracking.finish(env,state,data,action,params,pf,started,target,tvel,kp,kd,
                                     cmd_buf,tau_last,tau_sq,key,k_frame,k_reset)
    if env.cfg.recovery_steps:
        import recovery
        return recovery.finish(env, state, data, action, params, pf, started, target, tvel,
                               kp, kd, cmd_buf, tau_last, tau_sq, key, k_frame, k_reset)
    c, p, dt = env.cfg, env.plant, env.control_dt
    foot = contacts(data, p, c)
    track = state.standing
    xy = sole_centers(data, p)[:, :2]
    drift = jnp.linalg.norm(xy-track['anchor'], axis=1)
    air_s = jnp.where(foot['height'] > .003, track['air_s'] + dt, 0.)
    step_bad = (air_s > .05).any() | (drift > .020).any()
    base = data.xpos[p.base_bid]
    grav, omega, velocity = env._grav_body(data), env._gyro(data), env._vel_world(data)
    tilt = jnp.arccos(jnp.clip(-grav[2], -1., 1.))
    joint_speed = jnp.sqrt(jnp.mean(data.qvel[p.act_dadr]**2))
    finite = jnp.isfinite(data.qpos).all() & jnp.isfinite(data.qvel).all() & (jnp.abs(data.qvel).max() < 200)
    fallen = ~finite | (base[2] < .65*p.height_stand) | (tilt > np.pi/4) | foot['floor_violation']
    physical_quiet = ((jnp.linalg.norm(velocity[:2]) < .05) & (jnp.linalg.norm(omega) < .25)
                      & (joint_speed < .30) & (tilt < np.pi/18)
                      & (jnp.abs(base[2]-p.height_stand) < .05)
                      & (drift < .01).all() & (foot['height'] <= .003).all())
    quiet_anchor = jnp.where(track['quiet_s'] <= 0., base[:2], track['quiet_anchor'])
    physical_quiet = physical_quiet & (jnp.linalg.norm(base[:2]-quiet_anchor) <= .03)
    push_left = jnp.maximum(pf['push_left']-1, 0)
    since = pf['push_since'] + dt
    duration = .1
    in_recovery = pf['push_pending'] & (since <= duration)
    quiet_now = physical_quiet & ~in_recovery
    quiet_s = jnp.where(quiet_now, track['quiet_s'] + dt, 0.)
    loaded_s = jnp.where(quiet_now, track['loaded_s'] + dt * foot['grounded'].all(), 0.)
    quiet_start = jnp.where(quiet_now & (track['quiet_s'] <= 0), since-duration, track['quiet_start'])
    quiet_ok = (quiet_s >= 5.) & (loaded_s >= .95*quiet_s)
    pending = pf['push_pending']
    ok = pending & quiet_ok & (quiet_start <= 2.) & ~fallen & ~step_bad
    fail = pending & (fallen | step_bad | ((since >= 7.1) & ~ok))
    pending = pending & ~ok & ~fail
    calm_t = jnp.where(pf['push_left'] > 0, 0., state.calm_t + dt)
    calm = jnp.clip(calm_t, 0., 1.)
    distance = support_distance(data.subtree_com[0, :2], foot['points'], foot['active_points'], jnp)
    support = jnp.where(foot['active_points'].any(), jnp.exp(-(distance/.02)**2), 0.)
    q = data.qpos[p.act_qadr]
    half, middle = (env.q_hi-env.q_lo)/2, (env.q_hi+env.q_lo)/2
    terms = dict(
        alive=jnp.where(fallen, 0., 1.), upright=jnp.exp(-(tilt/.20)**2),
        height=.5*jnp.exp(-((base[2]-p.height_stand)/.04)**2), support=1.5*support,
        quiet=2.*calm*foot['grounded'].all()*jnp.exp(-jnp.sum(velocity[:2]**2)/.05**2
                -jnp.sum(omega**2)/.25**2-joint_speed**2/.30**2),
        pose=.2*calm*(support > .99)*jnp.exp(-jnp.mean((q-jnp.asarray(p.default_motor_pos))**2)/.50**2),
        torque=-.5*jnp.mean(tau_sq/p.decimation/(jnp.asarray(p.tau_peak)*state.draw.torque_scale)**2),
        action_rate=-.02*jnp.sum((action-state.prev_action)**2),
        slip=-.5*jnp.mean(jnp.clip(foot['slip2']/.10**2, 0., 4.)),
        gain=-.02*jnp.mean(kp/c.kp_hi),
        joint_limit=-jnp.mean(jnp.clip((jnp.abs(q-middle)/half-.95)/.05, 0., 1.)),
        foot_drift=-jnp.mean(jnp.clip(drift**2/.020**2, 0., 4.)),
        flat_missing=-.25*calm*jnp.mean((~foot['flat']).astype(jnp.float32)),
    )
    reward = sum(terms.values()) * dt/.01 - 20.*fallen - 5.*(step_bad & ~fallen)
    reward = jnp.where(finite, reward, -20.)
    count = state.step_n+1
    truncated = (count >= env.max_steps) & ~fallen & ~step_bad
    done = fallen | step_bad | truncated
    frame, ns, _, measurements = env._frame(data, tau_last, action, state, k_frame)
    history = jnp.concatenate([state.hist[1:], frame[None]], axis=0)
    new_track = dict(track, quiet_anchor=quiet_anchor, quiet_s=quiet_s, loaded_s=loaded_s,
                     quiet_start=quiet_start, air_s=air_s)
    new_state = state.replace(data=data, key=key, step_n=count, t=state.t+dt,
        cmd_buf=cmd_buf, prev_target=target, prev_target_vel=tvel, prev_action=action,
        prev_vel_body=env._vel_body(data), prev_foot_xy=xy, ground_ticks=jnp.where(foot['grounded'], state.ground_ticks+1, 0),
        hist=history, push_countdown=pf['push_countdown'], push_left=push_left,
        push_force=pf['push_force'], push_point=pf['push_point'], push_dv=pf['push_dv'],
        push_hard=pf['push_hard'], push_pending=pending, push_since=since, calm_t=calm_t,
        ep_return=state.ep_return+reward, ep_len=count, standing=new_track,
        **env._ema(state, measurements), **ns)
    obs = env._obs(new_state, data, foot['toe'], foot['heel'], env._vel_body(data))
    quiet_episode = ~track['episode_push_on']
    info = dict(reward_terms=terms, fallen=fallen, task_violation=step_bad, truncated=truncated,
        quiet_fall=(fallen | step_bad) & quiet_episode, push_start=started, push_ok=ok,
        push_fail=fail, push_dv=pf['push_dv'], push_hard=pf['push_hard'],
        ep_return=new_state.ep_return, ep_len=count, kp_mean=kp.mean(), kd_mean=kd.mean(),
        torque_util=jnp.mean(jnp.abs(tau_last)/jnp.asarray(p.tau_peak)), tilt_deg=tilt*180/np.pi,
        com_err=distance, airborne=jnp.mean((~foot['grounded']).astype(jnp.float32)),
        held=jnp.zeros(()), com_shift=state.draw.com_shift, obs_final=jnp.nan_to_num(obs),
        meas=measurements, target=target, kp=kp, kd=kd, action=action,
        quiet_trial=done & quiet_episode, quiet_success=truncated & quiet_episode & quiet_ok,
        quiet_ok=quiet_ok, quiet_s=quiet_s, foot_drift=drift, foot_loaded=foot['grounded'],
        normal_force=foot['normal_force'], contact_slip=jnp.sqrt(foot['slip2']),
        foot_clearance=foot['height'])
    reset, reset_obs = env._reset_one(k_reset, params, Override())
    result = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), reset, new_state)
    return result, jnp.nan_to_num(jnp.where(done, reset_obs, obs)), reward.astype(jnp.float32), done, info
