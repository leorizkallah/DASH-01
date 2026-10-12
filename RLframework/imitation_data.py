"""First-episode demonstration filtering and the pre-PPO behavioral gate."""
import numpy as np


def dataset_masks(done, upright, hold, finite):
    """Never learn from auto-resets; keep HOLD examples within the stance basin.

    Whole environments, rather than neighboring time samples, are held out.
    `hold` is a fixed boolean vector, one entry per environment.
    """
    live = ~np.maximum.accumulate(np.asarray(done, bool), axis=0)
    valid = live & finite & (~hold[None, :] | (upright > np.cos(np.deg2rad(10))))
    heldout = (np.arange(len(hold)) % 4 == 0)[None, :]
    return {f'{task}_{split}': valid & group[None, :] & partition
            for task, group in [('hold', hold), ('walk', ~hold)]
            for split, partition in [('train', ~heldout), ('valid', heldout)]}


def initialization_gate(before, after, validation):
    """PPO may start only with improved quiet holding and retained source walking."""
    return dict(
        walking_retained=bool(after['walking_retained']),
        holding_improved=bool(
            after['success_fraction'] > before['success_fraction'] or
            after['longest_quiet_hold_mean_s'] >= before['longest_quiet_hold_mean_s'] + .05),
        hold_fit_improved=bool(np.isfinite(validation['hold_after']) and
                              validation['hold_after'] < validation['hold_before']),
        walking_fit_bounded=bool(np.isfinite(validation['walk_after']) and
                                validation['walk_after'] <= .01))
