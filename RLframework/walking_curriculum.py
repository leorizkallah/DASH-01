"""First-episode diagnostics and a walking-earned shaping ramp (NumPy only)."""
import numpy as np


def early_metrics(rows, dt, window_s=6.):
    """Mask auto-resets; include the terminal transition in elapsed time."""
    active = np.asarray(rows['active'], dtype=bool)
    window = active & (np.arange(len(active))[:, None] < round(window_s/dt))
    denom = np.maximum(window.sum(axis=0), 1)
    elapsed = active.sum(axis=0)*dt
    return dict(
        first_episode_seconds=elapsed,
        early_forward_speed=np.where(window, rows['speed'], 0.).sum(axis=0)/denom,
        early_track_error=np.where(window, rows['error'], 0.).sum(axis=0)/denom,
        early_steps=(rows['quality']['valid_step']*window[..., None]).sum(axis=0),
        steps_from_reset=(rows['quality']['valid_step']*active[..., None]).sum(axis=0),
        **{name: (np.asarray(rows[name]) & active).any(axis=0)
           for name in ('term_low', 'term_tip', 'term_floor', 'term_ws', 'term_nan')})


def observe(state, result, require_steps=True, min_speed=.125):
    """Two successive evaluations must pass; one failure immediately pauses ramp.

    Gate is deliberately easier than gait acceptance: some stepping is needed,
    but perfect bilateral walking is the outcome we are shaping, not a prerequisite.
    Uses only the training-target trials, never the off-target transfer results.
    The independent drag gate sets require_steps=False: sliding is precisely the
    behavior it should improve, so valid steps cannot be its prerequisite.
    """
    step = int(result['step'])
    if state.get('last_eval_step') == step:
        return state
    m = result['per_trial']
    passed = ((np.asarray(m['first_episode_seconds']) >= 6.)
              & (np.asarray(m['early_forward_speed']) >= min_speed)
              & (np.asarray(m['early_track_error']) <= .15))
    if require_steps:
        passed &= np.asarray(m['early_steps']).sum(axis=-1) >= 2
    fraction = float(passed.mean())
    streak = state.get('streak', 0) + 1 if fraction >= .5 and result['finite'] else 0
    return dict(state, last_eval_step=step, success_fraction=fraction,
                streak=streak, advancing=streak >= 2)


def slip_metrics(rows, dt):
    """First episode only, per foot; slip already vanishes without active contact.

    Distance is an integral of contact-point RMS speed, not an individual material
    point's path length. Fractions/means use all active ticks, including swing.
    """
    active = np.asarray(rows['active'], bool)
    slip = np.where(active[..., None], rows['quality']['slip'], 0.)
    denominator = np.maximum(active.sum(axis=0)[:, None], 1)
    return dict(integrated_slip_m=slip.sum(axis=0)*dt,
                slip_speed_mean=slip.sum(axis=0)/denominator,
                sliding_fraction=(slip > .02).sum(axis=0)/denominator)


def advance(state, steps, ramp_steps):
    """Accumulate only steps trained after demonstrated competence; never jump."""
    progress = state.get('progress', 0.)
    if state.get('advancing', False):
        progress = min(1., progress + steps/max(ramp_steps, 1))
    return dict(state, progress=float(progress))
