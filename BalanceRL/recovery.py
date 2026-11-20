"""STAND-04 physics reward adapter; bounded recovery with no stepping income."""
import jax
import jax.numpy as jnp
import numpy as np

from plant import Override
from standing import contacts, sole_centers
from standing_math import support_distance
from recovery_logic import advance


def finish(env, state, data, action, params, pf, started, target, tvel, kp, kd, cmd_buf,
           tau_last, tau_sq, key, k_frame, k_reset):
    c, p, dt = env.cfg, env.plant, env.control_dt
    foot = contacts(data, p, c)
    xy = sole_centers(data, p)[:, :2]
    base = data.xpos[p.base_bid]
    grav, omega, velocity = env._grav_body(data), env._gyro(data), env._vel_world(data)
    tilt = jnp.arccos(jnp.clip(-grav[2], -1., 1.))
    joint_speed = jnp.sqrt(jnp.mean(data.qvel[p.act_dadr]**2))
    finite = jnp.isfinite(data.qpos).all() & jnp.isfinite(data.qvel).all() & (jnp.abs(data.qvel).max() < 200)
    fallen = ~finite | (base[2] < .65*p.height_stand) | (tilt > np.pi/4) | foot['floor_violation']
    kinematic_quiet = ((jnp.linalg.norm(velocity[:2]) < .05) & (jnp.linalg.norm(omega) < .25)
        & (joint_speed < .30) & (tilt < np.pi/18) & (jnp.abs(base[2]-p.height_stand) < .05))
    distance = support_distance(data.subtree_com[0, :2], foot['points'], foot['active_points'], jnp)
    support = jnp.where(foot['active_points'].any(), jnp.exp(-(distance/.02)**2), 0.)
    phase_args = {}
    if c.recovery_pose_return:
        from pose_reference import features, error
        posture = error(features(data.qpos[p.act_qadr], env._R(data), base,
                                 sole_centers(data, p), jnp), p.quiet_reference, c, jnp)
        phase_args = dict(pose_ok=posture['ok'], balance_supported=(support > .99))
    track, m = advance(state.standing, xy, base[:2], foot['normal_force'], foot['height'],
        kinematic_quiet, started, pf['push_duration'], pf['push_sector'], state.t+dt,
        dt, c, env._R(data)[:2, 1], jnp, **phase_args)
    ok = m['success'] & ~fallen
    fail = pf['push_pending'] & (fallen | m['task_bad'])
    pending = pf['push_pending'] & ~ok & ~fail
    calm_t = jnp.where(pf['push_left'] > 0, 0., state.calm_t+dt)
    calm = jnp.clip(calm_t, 0., 1.)
    q = data.qpos[p.act_qadr]
    half, middle = (env.q_hi-env.q_lo)/2, (env.q_hi+env.q_lo)/2
    # Displacement is permitted during recovery; ground sliding is still penalized.
    drift_cost = jnp.where(m['recovering'], .05*jnp.mean(jnp.clip(m['drift']**2/.25**2, 0., 4.)),
                          jnp.mean(jnp.clip(m['drift']**2/.020**2, 0., 4.)))
    terms = dict(alive=jnp.where(fallen, 0., 1.), upright=jnp.exp(-(tilt/.20)**2),
        height=.5*jnp.exp(-((base[2]-p.height_stand)/.04)**2), support=1.5*support,
        quiet=2.*calm*foot['grounded'].all()*jnp.exp(-jnp.sum(velocity[:2]**2)/.05**2
            -jnp.sum(omega**2)/.25**2-joint_speed**2/.30**2-jnp.mean(m['foot_speed']**2)/.02**2),
        pose=.2*calm*(support > .99)*jnp.exp(-jnp.mean((q-jnp.asarray(p.default_motor_pos))**2)/.50**2),
        torque=-.5*jnp.mean(tau_sq/p.decimation/(jnp.asarray(p.tau_peak)*state.draw.torque_scale)**2),
        action_rate=-.02*jnp.sum((action-state.prev_action)**2),
        slip=-.5*jnp.mean(jnp.clip(foot['slip2']/.10**2, 0., 4.)), gain=-.02*jnp.mean(kp/c.kp_hi),
        joint_limit=-jnp.mean(jnp.clip((jnp.abs(q-middle)/half-.95)/.05, 0., 1.)),
        foot_drift=-drift_cost, flat_missing=-.25*calm*~m['recovering']*jnp.mean((~foot['flat']).astype(jnp.float32)),
        foot_travel=-c.recovery_foot_travel_weight*m['path'],
        contact_events=-c.recovery_event_cost*(m['liftoff'].sum()+m['landing'].sum()),
        contact_chatter=-c.recovery_switch_cost*m['switches'],
        extra_steps=-c.recovery_extra_step_cost*m['excess_cost'], recovered=2.*ok)
    if c.recovery_pose_return:
        # Regain support first; ramp the posture objective after that milestone.
        return_gate = jnp.where(m['recovering'], m['returning']*jnp.clip(
            (state.t+dt-track['balanced_at'])/.5, 0., 1.), 1.)
        terms['pose'] = c.return_pose_weight*return_gate*posture['score']
    reward = sum(terms.values())*dt/.01 - 20.*fallen - 5.*(m['task_bad'] & ~fallen)
    reward = jnp.where(finite, reward, -20.)
    count = state.step_n+1
    truncated = (count >= env.max_steps) & ~fallen & ~m['task_bad']
    done = fallen | m['task_bad'] | truncated
    frame, ns, _, measurements = env._frame(data, tau_last, action, state, k_frame)
    history = jnp.concatenate([state.hist[1:], frame[None]], axis=0)
    new_state = state.replace(data=data, key=key, step_n=count, t=state.t+dt,
        cmd_buf=cmd_buf, prev_target=target, prev_target_vel=tvel, prev_action=action,
        prev_vel_body=env._vel_body(data), prev_foot_xy=xy,
        ground_ticks=jnp.where(foot['grounded'], state.ground_ticks+1, 0), hist=history,
        push_countdown=pf['push_countdown'], push_left=jnp.maximum(pf['push_left']-1, 0),
        push_force=pf['push_force'], push_point=pf['push_point'], push_dv=pf['push_dv'],
        push_hard=pf['push_hard'], push_pending=pending, push_since=pf['push_since']+dt,
        calm_t=calm_t, ep_return=state.ep_return+reward, ep_len=count, standing=track,
        **env._ema(state, measurements), **ns)
    obs = env._obs(new_state, data, foot['toe'], foot['heel'], env._vel_body(data))
    quiet_episode = ~track['episode_push_on']
    info = dict(reward_terms=terms, fallen=fallen, task_violation=m['task_bad'], truncated=truncated,
        quiet_fall=(fallen | m['task_bad']) & quiet_episode, push_start=started,
        push_ok=ok, push_fail=fail, push_dv=pf['push_dv'], push_hard=pf['push_hard'],
        ep_return=new_state.ep_return, ep_len=count, kp_mean=kp.mean(), kd_mean=kd.mean(),
        torque_util=jnp.mean(jnp.abs(tau_last)/jnp.asarray(p.tau_peak)), tilt_deg=tilt*180/np.pi,
        com_err=distance, airborne=jnp.mean((~foot['grounded']).astype(jnp.float32)),
        held=jnp.zeros(()), com_shift=state.draw.com_shift, obs_final=jnp.nan_to_num(obs),
        meas=measurements, target=target, kp=kp, kd=kd, action=action,
        quiet_trial=done & quiet_episode, quiet_success=truncated & quiet_episode & m['quiet_ok'],
        quiet_ok=m['quiet_ok'], quiet_s=track['quiet_s'], foot_drift=m['drift'],
        foot_loaded=foot['grounded'], normal_force=foot['normal_force'],
        contact_slip=jnp.sqrt(foot['slip2']), foot_clearance=foot['height'],
        recovery_steps=track['total_steps'], step_event=m['landing'].sum(),
        path_increment=m['path'], settling_time=m['settling_time'],
        sector=m['sector'], recovery_timeout=m['expired'])
    if c.recovery_pose_return:
        info.update(balance_recovered=m['balance_recovered'], pose_return_time=m['pose_return_time'],
            balance_time=m['balance_time'], returning=m['returning'], pose_ok=posture['ok'],
            pose_joint_rms=posture['joint_rms'], pose_joint_max=posture['joint_max'],
            pose_tilt_error=posture['tilt'], pose_foot_error=posture['feet_max'],
            pose_height_error=posture['height'])
    reset, reset_obs = env._reset_one(k_reset, params, Override())
    result = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), reset, new_state)
    return result, jnp.nan_to_num(jnp.where(done, reset_obs, obs)), reward.astype(jnp.float32), done, info
