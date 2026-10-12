"""Prevent silently using the last body's orientation after a failed torso lookup."""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'RLframework' / 'tools'))
from play_joystick import Sim


class PlayerFrameTests(unittest.TestCase):
    def test_base_measurements_use_torso_not_last_body(self):
        sim = Sim(ROOT / 'controller/deploy/bundles/dash_joy3_lr_s2_180M.npz')
        sim.data.qpos[4] = 0.2
        sim.data.qpos[sim.act_q[2]] += 0.4
        # A rotated motor housing must not rotate the torso's sensor frame.
        # In the current XML the last housing happens to align with the torso;
        # that coincidence previously hid the failed name lookup (index -1).
        sim.model.body_quat[-1] = [np.cos(0.2), 0, np.sin(0.2), 0]
        sim.data.qvel[:3] = [1, 0.3, 0.1]
        mujoco.mj_forward(sim.model, sim.data)
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        self.assertGreaterEqual(bid, 0)
        r = sim.data.xmat[bid].reshape(3, 3)
        np.testing.assert_allclose(sim._grav_body(), r.T @ [0, 0, -1])
        self.assertAlmostEqual(sim.speed(), (r.T @ sim.data.qvel[:3])[0])
        self.assertGreater(np.linalg.norm(sim._grav_body() -
                           sim.data.xmat[-1].reshape(3, 3).T @ [0, 0, -1]), 0.01)


if __name__ == '__main__':
    unittest.main()
