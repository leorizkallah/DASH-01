import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from RLframework.config import get_config
from RLframework.model import make_cad_v2_plant as builder


class CadV2PlantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml, cls.audit, cls.trace = builder.build()
        cls.model = mujoco.MjModel.from_xml_path(str(builder.OUT))

    def test_full_topology_and_mass(self):
        m = self.model
        self.assertEqual((m.nq, m.nv, m.nu, m.neq), (20, 20, 6, 4))
        self.assertAlmostEqual(m.body_mass.sum(), 14.594154, places=8)
        self.assertTrue(np.all(m.jnt_stiffness == 0))
        for side in "LR":
            self.assertGreaterEqual(m.joint(f"ankle_{side}").id, 0)
            self.assertGreaterEqual(m.joint(f"achilles_{side}").id, 0)

    def test_real_heel_and_toe_patches_keep_arch_gap(self):
        toe, heel = builder.sole_patches()
        self.assertLess(toe["size"][0]*2, .06)
        self.assertLess(heel["size"][0]*2, .06)
        self.assertGreater(toe["surface_center"][0]-heel["surface_center"][0], .2)
        for patch in (toe, heel):
            self.assertAlmostEqual(patch["size"][1]*2, .035, places=5)
            self.assertAlmostEqual(np.dot(patch["pos"]-patch["surface_center"], patch["normal"]), -.001)

    def test_unassisted_five_second_standing_gate(self):
        result = self.audit["hold"]
        self.assertTrue(result["passed"], result)
        self.assertLess(result["max_tilt_deg"], 5)
        self.assertLess(result["max_loop_gap_m"], .001)
        self.assertEqual(result["saturation_fraction"], 0)
        self.assertFalse(self.audit["deployment_ready"])

    def test_keyframe_and_joint_limits(self):
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        mujoco.mj_forward(m, d)
        self.assertEqual(d.ncon, 16)
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                self.assertGreaterEqual(d.qpos[m.jnt_qposadr[j]], m.jnt_range[j, 0])
                self.assertLessEqual(d.qpos[m.jnt_qposadr[j]], m.jnt_range[j, 1])
        np.testing.assert_allclose(d.qvel, 0)

    def test_low_speed_pilot_does_not_inherit_stop_or_old_plant(self):
        c = get_config("cad_v2_gait01")
        self.assertEqual(c.model_path, "model/dash01_cad_v2_free.xml")
        np.testing.assert_allclose(np.array(c.cmd_range)*c.v_max, [.2, .5])
        self.assertFalse(c.stop_flag or c.stop_adapter or c.bringup_enable or c.hold_enable)
        self.assertFalse(c.dr_enable)
        self.assertEqual(c.reset_joint_noise, 0)  # avoid breaking four loops at reset
        self.assertGreater(c.w_flight, 0)
        self.assertEqual(c.w_sym, 0)


if __name__ == "__main__":
    unittest.main()
