"""Backend-neutral bilateral step bookkeeping and bounded reward terms.

Legacy presets use reward/evaluation state only. GAIT-11 also exposes the
sequential supervisor's four command features to the actor.
Units are metres/seconds. Foot order is left, right. NumPy and JAX share this code.
"""
import numpy as np
if __package__:
    from . import sequential_steps
else:
    import sequential_steps


def initial(foot_x, xp=np, progress=False, sequence=False, attempts=False, sequential=False, first_foot=0):
    z = xp.zeros(2)
    state = dict(contact=xp.ones(2, dtype=bool), debounce=z,
                stance=z, air=z, start_x=xp.asarray(foot_x), peak=z,
                last_stance=z, last_air=z, last_length=z,
                seen=xp.zeros(2, dtype=bool), last_foot=xp.asarray(-1),
                age=z, clear_ema=z)
    if progress:
        state.update(progress_peak=z, quality_ema=z)
    if sequence:
        state.update(turn_wait=xp.asarray(0.), turn_grace=xp.asarray(0.))
    if attempts:
        state.update(bout_stance=z, turn_lift_used=z, turn_reach=z)
    if sequential:
        state.update(sequential_steps.initial(first_foot, xp))
    return state


def asymmetry(values, floor, xp=np):
    return xp.abs(values[0]-values[1]) / xp.maximum(.5*xp.sum(values), floor)


def point_velocity(cvel, point, root_com, xp=np):
    """MuJoCo world-oriented, root-COM-centred spatial velocity -> world point velocity."""
    return cvel[..., 3:] + xp.cross(cvel[..., :3], point-root_com)


def drag_cost(slip, reference=.1, xp=np):
    """Quadratic near zero, linear past reference; no high-slip plateau.

    Input is RMS horizontal speed of actual active floor-contact points per foot,
    already zero when airborne. Direction/sign is removed by that speed norm.
    Normalized cost is one at reference; slope is continuous there.
    """
    x = xp.abs(slip) / reference
    return xp.where(x <= 1., x*x, 2.*x-1.)


def next_turn(last_foot, wait, grace, valid, dt, frequency_hz, cfg, xp=np):
    """Only an opposite valid touchdown clears an outstanding turn.

    Repeating the same foot, scraping, lifting in place, or changing cadence
    cannot restart/extend the deadline. Either foot may take the first step.
    """
    expected = xp.where(last_foot >= 0, 1-last_foot, -1)
    accepted = (xp.sum(valid.astype(float)) == 1) & ((last_foot < 0) | valid[xp.maximum(expected, 0)])
    wait = xp.where(accepted | (last_foot < 0), 0., wait+dt)
    # Latch the allowance at accepted touchdown; later cadence changes cannot
    # postpone the obligation. Bound it to the supported walking cadence range.
    allowance = xp.clip(.5/xp.maximum(frequency_hz, .5) + cfg.bilateral_next_grace_s,
                        .2, 1.5)
    grace = xp.where(accepted, allowance, grace)
    overdue = xp.where((last_foot >= 0) & ~accepted,
                       xp.clip((wait-grace)/cfg.bilateral_next_ramp_s, 0., 1.), 0.)
    return wait, grace, overdue


def stance_allowance(frequency_hz, stance_ratio, cfg, xp=np):
    """Bounded cadence-aware limit; does not tax a normal scheduled stance."""
    return xp.clip(stance_ratio/xp.maximum(frequency_hz, .5)
                   + cfg.bilateral_stance_grace_s, .2, 1.5)


def reward_with_floor(total, terms, cfg, dt_scale, xp=np):
    """Preserve explicit repeat costs; floor only the remaining dense reward.

    Historical presets retain the exact old arithmetic. For GAIT-10 the bounded
    event/duration costs are charged once, after the floor, before fall penalties.
    The returned floor diagnostic refers to the part actually being clipped.
    """
    protected = 0.
    if getattr(cfg, 'bilateral_protect_repeat_cost', False):
        protected = terms.get('bilateral_repeat', 0.) + terms.get('bilateral_repeat_air', 0.)
    base = total-protected
    return (xp.maximum(base*dt_scale, -cfg.step_reward_floor*dt_scale)
            + protected*dt_scale), base < -cfg.step_reward_floor


def attempt_events(s, land, foot_x, start_x, length, valid, cfg, xp=np):
    """Completed meaningful bouts, distinct from valid steps or contact noise."""
    attempt = land & (s['air'] >= cfg.bilateral_attempt_min_s) & (
        (s['peak'] >= cfg.bilateral_attempt_height_m) |
        (xp.abs(foot_x-start_x) >= cfg.bilateral_attempt_travel_m))
    deficits = xp.stack([
        1.-xp.clip(s['air']/cfg.bilateral_min_swing_s, 0., 1.),
        1.-xp.clip(s['peak']/cfg.bilateral_clearance_m, 0., 1.),
        1.-xp.clip(length/cfg.bilateral_step_min_m, 0., 1.)])
    metrics = dict(attempt=attempt, incomplete_step=attempt & ~valid,
                   repeated_attempt=attempt & (xp.arange(2) == s['last_foot']),
                   attempt_length=xp.where(attempt, length, 0.),
                   attempt_height=xp.where(attempt, s['peak'], 0.),
                   attempt_air=xp.where(attempt, s['air'], 0.),
                   rejected_air=attempt & (deficits[0] > 0.),
                   rejected_clearance=attempt & (deficits[1] > 0.),
                   rejected_travel=attempt & (deficits[2] > 0.))
    terms = dict(
        bilateral_incomplete=-cfg.bilateral_incomplete_weight*xp.sum(
            metrics['incomplete_step']*xp.mean(deficits, axis=0)),
        bilateral_repeat=-cfg.bilateral_repeat_weight*xp.sum(metrics['repeated_attempt']))
    return terms, metrics


def extra_airtime(last_foot, contact, air, peak, travel, cfg, xp=np):
    """Soft cost while the previous foot lifts again before its partner lands.

    Both pads must be clear. Smooth time/motion gates tolerate short chatter;
    the foot is never constrained mechanically and the next-turn timer is intact.
    """
    wrong = (xp.arange(2) == last_foot) & ~contact
    time_gate = xp.clip((air-cfg.bilateral_repeat_air_grace_s)/
                        cfg.bilateral_repeat_air_ramp_s, 0., 1.)
    motion_gate = xp.clip(xp.maximum(peak/cfg.bilateral_attempt_height_m,
                                     xp.abs(travel)/cfg.bilateral_attempt_travel_m), 0., 1.)
    intensity = wrong*time_gate*motion_gate
    return -cfg.bilateral_repeat_air_weight*xp.sum(intensity), intensity


def update(s, contact_raw, height, foot_x, slip, swing_phase, dt, cfg, xp=np, frequency_hz=1.):
    """Debounce contacts, validate steps, then compute rewards and diagnostics.

    A simultaneous landing earns no alternating-step credit. A valid touchdown
    requires >=80 ms airborne, whole-sole clearance and positive body-relative
    travel; chatter and dragging cannot accumulate touchdown rewards.
    """
    attempts_on = getattr(cfg, 'bilateral_attempt_enable', False)
    counter = xp.where(contact_raw == s['contact'], 0., s['debounce']+1.)
    contact = xp.where(counter >= 2, contact_raw, s['contact'])
    lift = s['contact'] & ~contact
    land = ~s['contact'] & contact
    start_x = xp.where(lift, foot_x, s['start_x'])
    air = xp.where(contact, 0., s['air']+dt)
    stance = xp.where(contact, s['stance']+dt, 0.)
    peak = xp.where(lift, height, xp.where(~contact, xp.maximum(s['peak'], height), s['peak']))
    length = xp.maximum(foot_x-start_x, 0.)
    valid = land & (s['air'] >= cfg.bilateral_min_swing_s) & (s['peak'] >= cfg.bilateral_clearance_m) & (length >= cfg.bilateral_step_min_m)
    single = xp.sum(valid.astype(float)) == 1
    valid = valid & single
    strict = getattr(cfg, 'sequential_steps_enable', False)
    if strict:
        seq, valid, permitted_progress, seq_metrics = sequential_steps.advance(
            s, contact, lift, land, valid, dt, cfg.sequential_support_s, xp)
    alternate = valid & (xp.arange(2) != s['last_foot'])
    last_foot = xp.where(valid.any(), xp.argmax(valid), s['last_foot'])
    last_stance = xp.where(lift, s['stance'], s['last_stance'])
    last_air = xp.where(valid, s['air'], s['last_air'])
    last_length = xp.where(valid, length, s['last_length'])
    if attempts_on:
        bout_stance = xp.where(lift, s['stance'], s['bout_stance'])
        # Commit all three measurements from the SAME completed swing.
        last_stance = xp.where(valid, bout_stance, s['last_stance'])
    seen = s['seen'] | valid
    clear = (~contact) & (height >= cfg.bilateral_clearance_m)
    alpha = 1.-xp.exp(-dt/1.0)
    clear_ema = s['clear_ema'] + alpha*(clear.astype(float)-s['clear_ema'])
    age = xp.where(valid, 0., s['age']+dt)
    ratios = xp.stack([asymmetry(last_stance, .1, xp), asymmetry(last_air, .1, xp),
                       asymmetry(last_length, .04, xp)])
    deficit = xp.clip((cfg.bilateral_swing_frac-clear_ema.min())/cfg.bilateral_swing_frac, 0., 1.)
    missing = xp.mean(xp.clip((age-3.)/3., 0., 1.))
    mismatch = xp.mean(swing_phase*contact + (1.-swing_phase)*(~contact))
    clearance_deficit = xp.clip((cfg.bilateral_clearance_m-height)/cfg.bilateral_clearance_m, 0., 1.)
    terms = dict(
        bilateral_swing=-.4*deficit**2,
        bilateral_missing=-.2*missing,
        bilateral_step=3.*xp.sum(alternate),
        bilateral_repeat=-.5*xp.sum(valid & ~alternate),
        bilateral_symmetry=-.3*xp.mean(xp.clip((ratios-cfg.bilateral_tolerance)/(1.-cfg.bilateral_tolerance), 0., 1.)**2)*seen.all(),
        bilateral_phase=-.5*mismatch,
        bilateral_clearance=-.3*xp.mean(swing_phase*clearance_deficit**2),
        bilateral_slip=-.4*xp.mean(xp.minimum((slip/.1)**2, 1.)),
    )
    new = dict(contact=contact, debounce=xp.where(counter >= 2, 0., counter),
               stance=stance, air=air, start_x=start_x, peak=peak,
               last_stance=last_stance, last_air=last_air, last_length=last_length,
               seen=seen, last_foot=last_foot, age=age, clear_ema=clear_ema)
    metrics = dict(contact=contact, clearance=height, clear=clear,
                   valid_step=valid, alternating_step=alternate,
                   symmetry=ratios, pair_seen=seen.all(), slip=slip,
                   step_length=xp.where(valid, length, 0.))
    if getattr(cfg, 'bilateral_progress_enable', False):
        # GAIT-04: smooth credit below the valid-step threshold. The old preset
        # retains its original state and rewards for reproducible replay.
        height_fraction = xp.clip(height/cfg.bilateral_clearance_m, 0., 1.)
        progress_gate = 1. if getattr(cfg, 'bilateral_turn_progress', False) else swing_phase
        quality = progress_gate * height_fraction * (~contact)
        if attempts_on:
            expected = (s['last_foot'] < 0) | (xp.arange(2) != s['last_foot'])
            if strict:
                expected = permitted_progress
            # A landing cannot replenish partial-lift or pair-quality credit.
            quality = xp.minimum(quality*expected,
                                 xp.maximum(cfg.bilateral_lift_budget_s-s['turn_lift_used'], 0.)/dt)
        quality_ema = s['quality_ema'] + alpha*(quality-s['quality_ema'])
        weak = xp.clip(quality_ema.min()/cfg.bilateral_swing_frac, 0., 1.)
        # Only new forward reach in this airborne bout earns progress. Repeated
        # back-and-forth motion cannot collect the same forward distance again.
        previous_peak = xp.where(lift, 0., s['progress_peak'])
        reach = xp.clip(length/cfg.bilateral_step_min_m, 0., 1.)
        progress_peak = xp.where(~contact, xp.maximum(previous_peak, reach), 0.)
        advance = xp.where(~contact, xp.maximum(progress_peak-previous_peak, 0.), 0.)
        if attempts_on:
            # Best reach ACROSS attempts in this turn, not cumulative shuffling.
            turn_reach = xp.maximum(s['turn_reach'], xp.where(~contact & expected, reach, 0.))
            advance = xp.maximum(turn_reach-s['turn_reach'], 0.)
        forward = xp.clip(advance*.4/dt, 0., 1.) * quality
        cycle_cost = xp.mean(xp.clip((ratios-cfg.bilateral_tolerance)/
                                    (1.-cfg.bilateral_tolerance), 0., 1.)**2)
        terms.update(
            bilateral_swing=-1.0*deficit**2,
            bilateral_missing=-.6*missing,
            bilateral_phase=-2.0*xp.mean(swing_phase*contact)
                            -.5*xp.mean((1.-swing_phase)*(~contact)),
            # Missing bilateral history retains a cost after the existing grace.
            bilateral_symmetry=-.3*xp.maximum(xp.where(seen.all(), cycle_cost, 0.), missing),
            bilateral_lift_progress=.6*xp.mean(quality),
            bilateral_forward_progress=.6*xp.mean(forward),
            bilateral_pair_progress=.5*weak,
        )
        new.update(progress_peak=progress_peak, quality_ema=quality_ema)
        metrics.update(lift_progress=quality, forward_progress=forward,
                       quality_ema=quality_ema)
    if getattr(cfg, 'bilateral_sequence_enable', False):
        wait, grace, overdue = next_turn(s['last_foot'], s['turn_wait'], s['turn_grace'],
                                        valid, dt, frequency_hz, cfg, xp)
        terms['bilateral_next_wait'] = -cfg.bilateral_next_wait_weight * overdue
        terms['bilateral_repeat'] = -cfg.bilateral_repeat_weight * xp.sum(valid & ~alternate)
        # Dense swing rewards should help the required foot, not keep paying
        # the already-active leg while its partner remains grounded.
        expected_before = (s['last_foot'] < 0) | (xp.arange(2) != s['last_foot'])
        if strict:
            expected_before = permitted_progress
        if getattr(cfg, 'bilateral_progress_enable', False):
            terms['bilateral_lift_progress'] = .6*xp.mean(quality*expected_before)
            terms['bilateral_forward_progress'] = .6*xp.mean(forward*expected_before)
        expected_after = (last_foot >= 0) & (xp.arange(2) != last_foot)
        new.update(turn_wait=wait, turn_grace=grace)
        metrics.update(expected_next=expected_after.astype(float),
                       overdue_next=expected_after.astype(float)*overdue)
    if attempts_on:
        event_terms, event_metrics = attempt_events(s, land, foot_x, start_x, length, valid, cfg, xp)
        terms.update(event_terms)
        metrics.update(event_metrics)
        completed_turn = alternate.any()
        new.update(bout_stance=bout_stance,
                   turn_lift_used=xp.where(completed_turn, 0., s['turn_lift_used']+quality*dt),
                   turn_reach=xp.where(completed_turn, 0., turn_reach))
        if getattr(cfg, 'bilateral_repeat_air_weight', 0.) > 0:
            terms['bilateral_repeat_air'], intensity = extra_airtime(
                s['last_foot'], contact, air, peak, foot_x-start_x, cfg, xp)
            metrics.update(repeat_air=intensity, reward_swing=swing_phase,
                           expected_foot=expected_before.astype(float),
                           lift_budget_used=s['turn_lift_used'], turn_reach=s['turn_reach'])
    if strict:
        # Count repetition among ALL meaningful attempts, not only valid steps.
        repeated = metrics['attempt'] & s['seq_previous_attempt']
        seq['seq_previous_attempt'] = xp.where(metrics['attempt'].any(),
                                               metrics['attempt'], s['seq_previous_attempt'])
        terms['bilateral_repeat'] = -cfg.bilateral_repeat_weight*xp.sum(repeated)
        time_gate = xp.clip((air-cfg.bilateral_repeat_air_grace_s)/cfg.bilateral_repeat_air_ramp_s, 0., 1.)
        intensity = seq_metrics['unauthorized_air']*time_gate
        terms['bilateral_repeat_air'] = -cfg.bilateral_repeat_air_weight*xp.sum(intensity)
        # Event supervisor replaces the waveform clock as the gait objective.
        terms['bilateral_phase'] = xp.asarray(0.)
        terms['bilateral_clearance'] = -.3*xp.mean(permitted_progress*clearance_deficit**2)
        # Failed/out-of-order meaningful bouts have a cost even if kinematics pass.
        terms['bilateral_incomplete'] = -cfg.bilateral_incomplete_weight*xp.sum(
            metrics['incomplete_step']*xp.maximum(.25, 1.-xp.minimum(
                xp.clip(length/cfg.bilateral_step_min_m, 0., 1.),
                xp.clip(s['peak']/cfg.bilateral_clearance_m, 0., 1.))))
        metrics.update(seq_metrics, repeated_attempt=repeated, repeat_air=intensity)
        new.update(seq)
    return new, terms, metrics


def assess(metrics):
    """Per-trial acceptance, used by tests and the greedy evaluator."""
    return (metrics['survived'] & (metrics['track_error'] <= .1) &
            (metrics['heading_error'] <= np.deg2rad(15)) &
            (np.min(metrics['steps'], axis=-1) >= 3) &
            (np.min(metrics['clear_fraction'], axis=-1) >= .08) &
            (metrics['alternation'] >= .9) & (metrics['max_symmetry'] <= .2) &
            (metrics['flight_fraction'] <= .01) & (metrics['contact_slip'] <= .08) &
            (metrics.get('joint_limit_violation', 0.) <= .01))
