"""Continuous supported alternation: no mandatory pause or velocity condition."""
import importlib.util
import unittest
import numpy as np
from RLframework import sequential_steps as seq, bilateral_gait as bg
from RLframework.config import get_config
from RLframework.tests.test_step_attempts import bout

C = get_config('cad_v2_gait11')


def ready():
    s = bg.initial(np.zeros(2), progress=True, sequence=True, attempts=True, sequential=True)
    for _ in range(3):
        s, _, _ = bg.update(s, np.ones(2, bool), np.zeros(2), np.zeros(2),
                            np.zeros(2), np.zeros(2), .01, C)
    return s


def advance(s, contact, lift=(False, False), land=(False, False), valid=(False, False)):
    return seq.advance(s, *[np.asarray(a, bool) for a in (contact, lift, land, valid)], .01, .02)


class SequentialTests(unittest.TestCase):
    def test_next_swing_starts_immediately_after_touchdown(self):
        for first in (0, 1):
            s = seq.initial(first)
            s['seq_ready'] = np.asarray(.02)
            lift = np.arange(2) == first
            s, _, _, _ = advance(s, ~lift, lift=lift)
            s, accepted, _, _ = advance(s, (True, True), land=lift, valid=lift)
            self.assertTrue(accepted[first])
            s, _, gate, _ = advance(s, lift, lift=~lift)
            self.assertTrue(gate[1-first])
            self.assertEqual(seq.command(s).shape, (4,))

    def test_support_loss_invalidates_bout_even_if_support_returns(self):
        s = seq.initial()
        s['seq_ready'] = np.asarray(.02)
        s, _, _, _ = advance(s, (False, True), lift=(True, False))
        s, _, _, _ = advance(s, (False, False), lift=(False, True))
        s, _, _, _ = advance(s, (False, True), land=(False, True))
        s, ok, _, _ = advance(s, (True, True), land=(True, False), valid=(True, False))
        self.assertFalse(ok.any())
        self.assertEqual(s['seq_expected'], 0)

    def test_full_reward_counts_invalid_retries_and_rejects_wrong_foot(self):
        s, _, m = bout(ready(), foot=0, cfg=C)
        self.assertEqual(m['valid_step'].sum(), 0)
        self.assertEqual(s['seq_expected'], 0)
        s, t, m = bout(s, foot=0, cfg=C)
        self.assertEqual(m['repeated_attempt'][0], 1)
        self.assertEqual(t['bilateral_repeat'], -20.)
        s, t, m = bout(s, foot=1, distance=.06, height=.02, cfg=C)
        self.assertEqual(m['valid_step'].sum(), 0)
        self.assertLess(t['bilateral_repeat_air'], 0.)
        self.assertEqual(s['seq_expected'], 0)

    def test_clean_alternation_earns_completion_without_repeat(self):
        s = ready()
        for foot in (0, 1, 0, 1):
            s, t, m = bout(s, foot=foot, distance=.06, height=.02, cfg=C)
            self.assertEqual(m['valid_step'][foot], 1)
            self.assertEqual(m['repeated_attempt'].sum(), 0)
            self.assertGreater(t['bilateral_step'], 0.)
            self.assertEqual(s['seq_expected'], 1-foot)

    def test_noise_does_not_switch_turn(self):
        s, _, m = bout(ready(), foot=0, ticks=2, cfg=C)
        self.assertEqual(m['attempt'].sum(), 0)
        self.assertEqual(s['seq_expected'], 0)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX runs on SCITAS')
    def test_jax_tree_and_commands(self):
        import jax
        import jax.numpy as jnp
        s = bg.initial(jnp.zeros(2), jnp, progress=True, sequence=True, attempts=True, sequential=True)
        def tick(s):
            return bg.update(s, jnp.ones(2, bool), jnp.zeros(2), jnp.zeros(2),
                             jnp.zeros(2), jnp.zeros(2), .01, C, jnp)
        n, t, m = jax.jit(tick)(s)
        self.assertEqual(jax.tree.structure(n), jax.tree.structure(s))
        self.assertTrue(np.isfinite(np.asarray(seq.command(n, jnp))).all())
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in t.values()))
