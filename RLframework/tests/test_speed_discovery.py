import unittest
import numpy as np
from RLframework.config import get_config
from RLframework.speed_metrics import income, terms, summarize, REWARD_KEYS

C = get_config("cad_v2_speed01")


class SpeedTests(unittest.TestCase):
    def test_signed_monotone_speed_and_tilt(self):
        v = np.array([-2.0, -1.0, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0])
        self.assertTrue((np.diff(income(v, 1.0, C)) > 0).all())
        self.assertEqual(income(1.0, 0.5, C), 0.0)
        self.assertEqual(income(-1.0, 0.5, C), -2.0)
        self.assertEqual(income(0.0, 1.0, C), 0.0)

    def test_fall_is_worse_than_worst_discounted_continuation(self):
        # control_dt is 0.01: floor is -0.5, discounted infinite cost is -100.
        bound = C.step_reward_floor * 0.5 / (1 - C.gamma)
        self.assertGreater(C.fall_penalty, bound)
        for ticks in (1, 100, 3000):
            delayed = -bound * (1 - C.gamma**ticks) - C.fall_penalty * C.gamma**ticks
            self.assertGreater(delayed, -C.fall_penalty)

    def test_only_explicit_terms_survive_and_slip_is_direction_symmetric(self):
        existing = {k: 0.0 for k in REWARD_KEYS}
        existing.update(
            alive=100.0, bilateral_step=100.0, phase_contact=-100.0, air_time=100.0
        )
        a = terms(existing, np.array([0.1, 0.2]), 1.0, C)
        b = terms(existing, np.array([-0.1, -0.2]), 1.0, C)
        self.assertEqual(a, b)
        self.assertNotIn("alive", a)
        self.assertNotIn("bilateral_step", a)
        self.assertEqual(a["contact_slip"], -0.5)
        self.assertAlmostEqual(
            terms(existing, np.array([0.1, 0.2]), 0.05, C)["contact_slip"], -0.025
        )

    def test_fresh_preset_preserves_motors_without_step_rules(self):
        old = get_config("cad_v2_gait11")
        for key in (
            "motor_vel_limit",
            "motor_r_ohm",
            "drive_kp",
            "drive_kd",
            "thermal_enable",
        ):
            self.assertEqual(getattr(C, key), getattr(old, key))
        self.assertEqual(C.walltime_hours, 5.0)
        self.assertEqual(C.total_steps, 0)
        for flag in (
            "bilateral_enable",
            "sequential_steps_enable",
            "stop_adapter",
            "stop_flag",
            "bringup_enable",
            "hold_enable",
        ):
            self.assertFalse(getattr(C, flag), flag)
        self.assertEqual(C.w_alive, 0.0)
        self.assertEqual(C.objective, "speed")
        self.assertTrue(C.sprint_world_speed)

    def test_first_episode_masks_resets_and_fast_falls(self):
        shape = (3000, 2)
        r = {
            k: np.zeros(shape)
            for k in (
                "vx",
                "heading",
                "thermal",
                "flight",
                "ceiling",
                "floor",
                "joint_violation",
                "reward",
            )
        }
        r.update(
            active=np.ones(shape, bool),
            fallen=np.zeros(shape, bool),
            slip=np.zeros(shape + (2,)),
            clear=np.zeros(shape + (2,), bool),
            contact=np.ones(shape + (2,), bool),
        )
        r["vx"][:, 0] = 0.5
        r["vx"][:, 1] = 4.0
        r["active"][100:, 1] = False
        r["fallen"][99, 1] = True
        r["reward"][100:, 1] = 1e6
        s = summarize(r, 0.01, 30.0)
        self.assertEqual(s["per_trial"]["survived"], [True, False])
        self.assertEqual(s["per_trial"]["reward"], [0.0, 0.0])
        self.assertAlmostEqual(s["per_trial"]["speed_over_full_horizon"][1], 4 / 30)
        self.assertFalse(s["keeper_eligible"])

    def test_old_presets_opt_out(self):
        for p in ("dash_sprint", "cad_v2_gait09", "cad_v2_gait11"):
            self.assertFalse(get_config(p).speed_discovery)
