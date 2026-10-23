"""Guard the walking-only recipe and homing import against silent task changes."""

import json
from pathlib import Path
import unittest
from dataclasses import asdict

import mujoco
import numpy as np
from RLframework.config import get_config

ROOT = Path(__file__).resolve().parents[2]


class WalkingRecipeTests(unittest.TestCase):
    def test_retained_reward_and_learning_recipe(self):
        old, new = (asdict(get_config(k)) for k in ("dash_joy3_lr", "cad_v2_walk01"))
        removed = {"w_stop_speed", "w_stop_amp", "w_stand_pose", "w_stand_vel"}
        for key in old:
            if key.startswith("w_") and key not in removed:
                self.assertEqual(old[key], new[key], key)
        for key in (
            "gamma",
            "gae_lambda",
            "learning_rate",
            "lr_final",
            "lr_kl_adaptive",
            "cmd_curriculum_steps",
            "track_sigma_steps",
            "shape_curriculum_steps",
            "dr_curriculum_steps",
            "dr_scale_final",
            "gait_freq_hz",
            "drive_kp",
            "drive_kd",
        ):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual(old["n_envs"] * old["n_steps"], new["n_envs"] * new["n_steps"])

    def test_no_stop_or_previous_gait_task(self):
        c = get_config("cad_v2_walk01")
        for field in (
            "stop_flag",
            "stand_at_zero",
            "stop_adapter",
            "bilateral_enable",
            "sequential_steps_enable",
            "speed_discovery",
            "bringup_enable",
            "hold_enable",
        ):
            self.assertFalse(getattr(c, field), field)
        self.assertEqual(c.cmd_stop_frac, 0)
        self.assertEqual(c.cmd_zero_frac, 0)
        self.assertGreater(c.cmd_range[0], 0)
        self.assertEqual(c.total_steps, 200_000_000)
        self.assertEqual(c.reset_joint_noise, 0)
        self.assertEqual(c.dr_loop_site_m, 0)

    def test_homing_preserves_calibrated_plant(self):
        c = get_config("cad_v2_walk01")
        m = mujoco.MjModel.from_xml_path(str(ROOT / "RLframework" / c.model_path))
        old = mujoco.MjModel.from_xml_path(
            str(ROOT / "RLframework/model/dash01_cad_v2_free.xml")
        )
        for field in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "jnt_range",
            "actuator_forcerange",
            "jnt_stiffness",
        ):
            np.testing.assert_allclose(getattr(m, field), getattr(old, field))
        self.assertEqual((m.nq, m.nu, m.neq), (20, 6, 4))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("homing").id)
        mujoco.mj_forward(m, d)
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                self.assertTrue(
                    m.jnt_range[j, 0] <= d.qpos[m.jnt_qposadr[j]] <= m.jnt_range[j, 1]
                )
        a = json.loads((ROOT / "report/data/homing01/audit.json").read_text())
        self.assertLess(a["max_body_position_fit_error_m"], 1e-5)
        self.assertLess(a["equilibrium"]["max_pose_adjustment_rad"], 0.001)
        self.assertTrue(a["hold"]["passed"])


if __name__ == "__main__":
    unittest.main()
