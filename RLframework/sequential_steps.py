"""Explicit foot-turn supervisor. Backend-neutral; simulation contacts drive it.

The actor receives four commanded expected-foot/mode features. A real
deployment needs an equivalent supervisor with validated contact/step estimation;
this module is not a hardware contact estimator or a mechanical foot constraint.
"""
import numpy as np

COMMAND_DIM = 4


def initial(first_foot=0, xp=np):
    return dict(seq_expected=xp.asarray(first_foot, dtype=xp.int32),
                seq_active=xp.asarray(-1, dtype=xp.int32),
                seq_ready=xp.asarray(0.), seq_authorized=xp.zeros(2, dtype=bool),
                seq_previous_attempt=xp.zeros(2, dtype=bool))


def command(s, xp=np):
    return xp.concatenate([(xp.arange(2) == s['seq_expected']).astype(float),
                           xp.stack([(s['seq_active'] >= 0).astype(float),
                                     (s['seq_ready'] > 0).astype(float)])])


def advance(s, contact, lift, land, candidate, dt, support_s, xp=np):
    """Accept only an authorized, supported swing followed by valid touchdown.

    A failed expected-foot attempt can retry, with a cost, after support returns.
    Loss of the other foot's support invalidates a bout even if support returns.
    The next foot cannot steal a turn by making a small early lift.
    """
    expected = xp.arange(2) == s['seq_expected']
    may_start = (s['seq_active'] < 0) & (s['seq_ready'] >= support_s)
    permitted = lift & expected & contact[::-1] & may_start
    authorized = xp.where(lift, permitted, s['seq_authorized'])
    authorized = authorized & ~(~contact & ~contact[::-1])
    active = xp.where(permitted.any(), s['seq_expected'], s['seq_active'])
    accepted = candidate & authorized & expected & contact[::-1]
    gate = authorized & expected & (xp.arange(2) == active)
    unauthorized_air = (~contact) & ~gate
    active = xp.where((active >= 0) & land[xp.maximum(active, 0)], -1, active)
    ready = xp.where(contact.all(), xp.minimum(s['seq_ready']+dt, support_s), 0.)
    # A valid, debounced touchdown establishes support NOW. No settling dwell,
    # zero-speed requirement or pause before the opposite foot may swing.
    switch = accepted.any()
    ready = xp.where(switch, support_s, ready)
    state = dict(seq_expected=xp.where(switch, 1-s['seq_expected'], s['seq_expected']),
                 seq_active=active, seq_ready=ready,
                 seq_authorized=authorized, seq_previous_attempt=s['seq_previous_attempt'])
    metrics = dict(sequence_rejected=candidate & ~accepted,
                   unauthorized_air=unauthorized_air.astype(float),
                   expected_foot=expected.astype(float), sequence_command=command(state, xp))
    return state, accepted, gate, metrics
