"""Native-physics checks of the three contact layouts and shared support math."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import get_config
from standing_math import contacts, sole_centers, support_distance


def load(name):
    cfg = get_config('stand_'+name)
    model = mujoco.MjModel.from_xml_path(str(PACKAGE/cfg.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    qadr = [model.joint(n).qposadr[0] for n in ('hip_roll_L', 'cam_L', 'thigh_L', 'hip_roll_R', 'cam_R', 'thigh_R')]
    nid = model.numeric('nominal_cmd').id
    command = model.numeric_data[model.numeric_adr[nid]:model.numeric_adr[nid]+6]
    data.ctrl[:] = np.array(cfg.drive_kp)*(command-data.qpos[qadr])
    mujoco.mj_forward(model, data)
    first = np.array([model.geom(f'foot_{side}_col').id for side in 'LR'])
    second = first.copy() if name == 'v1' else np.array([model.geom(f'heel_{side}_col').id for side in 'LR'])
    plant = SimpleNamespace(m=model, floor_gid=model.geom('floor').id, foot_gids=first,
                            heel_gids=second, foot_bids=model.geom_bodyid[first])
    return cfg, model, data, plant


class StandingContactTests(unittest.TestCase):
    def test_exact_physical_regions_and_force_without_double_count(self):
        for name, count in [('v1', 1), ('v2', 2), ('xshape', 2)]:
            with self.subTest(name=name):
                cfg, model, data, plant = load(name)
                ids = set(map(int, np.r_[plant.foot_gids, plant.heel_gids]))
                self.assertEqual(len(ids), 2*count)
                measured = contacts(data, plant, cfg)
                self.assertTrue(measured['grounded'].all())
                self.assertTrue(measured['flat'].all())
                self.assertAlmostEqual(measured['normal_force'].sum(), model.body_mass.sum()*9.81, delta=.1)
                self.assertLess(support_distance(data.subtree_com[0, :2], measured['points'], measured['active_points']), 1e-6)

    def test_hover_does_not_count_as_support(self):
        for name in ('v1', 'v2', 'xshape'):
            cfg, model, data, plant = load(name)
            data.qpos[model.joint('base_z').qposadr[0]] += .02
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            measured = contacts(data, plant, cfg)
            self.assertFalse(measured['grounded'].any())
            self.assertFalse(measured['active_points'].any())

    def test_sole_markers_are_at_floor_not_body_origin(self):
        for name in ('v1', 'v2', 'xshape'):
            _, _, data, plant = load(name)
            self.assertTrue((np.abs(sole_centers(data, plant)[:, 2]) < .005).all())

    def test_slip_uses_contact_velocity_not_body_origin_velocity(self):
        cfg, _, data, plant = load('v1')
        for bid in plant.foot_bids:
            origin = data.subtree_com[plant.m.body_rootid[bid]]
            data.cvel[bid, :3] = [0., 1., 0.]
            data.cvel[bid, 3:] = -np.cross(data.cvel[bid, :3], -origin)
        measured = contacts(data, plant, cfg)
        self.assertTrue((np.sqrt(measured['slip2']) < .005).all())

    def test_friction_and_free_base_match(self):
        for name in ('v1', 'v2', 'xshape'):
            cfg, m, _, _ = load(name)
            self.assertEqual(m.nu, 6)
            for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
                self.assertFalse(m.jnt_limited[m.joint('base_'+axis).id])
            np.testing.assert_allclose(m.geom_friction[m.geom('floor').id], [.7, .005, .0001])
            self.assertEqual(cfg.hold_s_range, (0., 0.))
            self.assertFalse(cfg.reflex_enable)
            self.assertEqual(cfg.dr_loop_site_m, 0.)

    def test_support_hull_and_degenerate_cases(self):
        square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        active = np.ones(4, bool)
        self.assertEqual(support_distance(np.array([.5, .5]), square, active), 0.)
        self.assertAlmostEqual(support_distance(np.array([2., .5]), square, active), 1.)
        self.assertGreater(support_distance(np.array([.5, .5]), square, ~active), 100.)
        self.assertAlmostEqual(support_distance(np.array([.5, .5]), square[:2], active[:2]), .5)


if __name__ == '__main__':
    unittest.main()
