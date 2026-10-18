import unittest
from RLframework.walltime_budget import WalltimeBudget


class BudgetTests(unittest.TestCase):
    def test_absolute_allocation_includes_setup_and_reserves_save_time(self):
        now = [0.]
        b = WalltimeBudget(8, deadline_epoch=10000, epoch_clock=lambda: 4000,
                           clock=lambda: now[0], reserve_s=600)
        self.assertEqual(b.remaining(), 6000)
        now[0] = 5101
        self.assertTrue(b.training_allowed())
        self.assertFalse(b.evaluation_allowed())
        now[0] = 5400
        self.assertFalse(b.training_allowed())
        self.assertEqual(b.remaining(), 600)

    def test_signal_and_expired_deadline(self):
        b = WalltimeBudget(8)
        b.request_stop(15)
        self.assertFalse(b.training_allowed())
        self.assertEqual(b.summary()['reason'], 'signal')
        self.assertFalse(WalltimeBudget(8, deadline_epoch=0).training_allowed())

    def test_fresh_preset_has_no_step_cap(self):
        from RLframework.config import get_config
        c = get_config('cad_v2_gait11')
        self.assertEqual(c.total_steps, 0)
        self.assertEqual(c.walltime_hours, 8)
        self.assertGreater(c.lr_schedule_steps, 0)
        self.assertAlmostEqual(c.v_max*c.cmd_range[0], .15)
