"""Lightweight contact, rollover, timing and persistent-symmetry checks."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import mujoco
from RLframework import flatfeet as f
from RLframework.config import get_config


def fixture(force=(5.0, 5.0, 5.0, 5.0), dist=None):
    con = NS(
        efc_address=np.arange(4) * 3,
        dist=np.full(4, -0.001) if dist is None else np.asarray(dist),
        geom=np.array([[0, 1], [0, 2], [0, 3], [0, 4]]),
        pos=np.zeros((4, 3)),
        frame=np.tile(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), (4, 1, 1)),
    )
    efc = np.zeros(12)
    efc[::3] = force
    d = NS(
        contact=con, efc_force=efc, cvel=np.zeros((3, 6)), subtree_com=np.zeros((3, 3))
    )
    p = NS(
        floor_gid=0,
        foot_gids=[1, 3],
        heel_gids=[2, 4],
        foot_bids=[1, 2],
        m=NS(body_rootid=np.arange(3)),
    )
    return d, p


class FlatFeetTests(unittest.TestCase):
    def setUp(self):
        self.c = get_config("cad_v2_flatfeet_symmetry_v1")

    def test_contact_states_and_no_hover(self):
        for force, expected in [
            ([5, 0, 0, 0], [True, False]),
            ([0, 5, 0, 0], [True, False]),
            ([5, 5, 0, 0], [True, False]),
            ([0, 0, 0, 0], [False, False]),
        ]:
            d, p = fixture(force)
            r = f.contact_measurements(d, p, 2)
            np.testing.assert_array_equal(r["grounded"], expected)
            self.assertEqual(bool(r["flat"][0]), force[0] > 0 and force[1] > 0)
        d, p = fixture([1, 1, 1, 1])
        self.assertFalse(f.contact_measurements(d, p, 2)["grounded"].any())
        d, p = fixture(dist=[0.001] * 4)
        self.assertFalse(f.contact_measurements(d, p, 2)["flat"].any())

    def test_rollover_is_not_a_step(self):
        s = f.initial(np.array([True, True]), self.c)
        for force in ([0, 5, 5, 0], [5, 5, 5, 5], [5, 0, 0, 5]) * 20:
            d, p = fixture(force)
            g = f.contact_measurements(d, p, 2)["grounded"]
            s, _, term, m = f.update(s, g, True, 0.25, False, 1.0, self.c, 0.01)
            self.assertFalse(m["touchdown"].any() or m["liftoff"].any() or term)
        self.assertEqual(s["count"].sum(), 0)

    def test_slip_normalized_and_rollover(self):
        d, p = fixture()
        d.cvel[:, 3] = 0.2
        np.testing.assert_allclose(f.contact_measurements(d, p, 2)["slip"], 0.2)
        d.efc_force[3] = 0  # removing the heel does not halve/double slip
        np.testing.assert_allclose(f.contact_measurements(d, p, 2)["slip"], 0.2)
        d.cvel[:] = 0
        d.cvel[:, 1] = 1
        d.cvel[:, 5] = 1
        d.contact.pos[:, 0] = 1
        np.testing.assert_allclose(
            f.contact_measurements(d, p, 2)["slip"], 0, atol=1e-12
        )

    def test_contact_velocity_matches_mujoco_jacobian(self):
        root = Path(__file__).resolve().parents[2]
        m = mujoco.MjModel.from_xml_path(str(root / "RLframework" / self.c.model_path))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("homing").id)
        d.qpos[2] -= 0.002
        d.qvel[:] = np.linspace(-0.05, 0.05, m.nv)
        mujoco.mj_forward(m, d)
        p = NS(
            m=m,
            floor_gid=m.geom("floor").id,
            foot_gids=[m.geom("foot_" + s + "_col").id for s in "LR"],
            heel_gids=[m.geom("heel_" + s + "_col").id for s in "LR"],
        )
        p.foot_bids = [int(m.geom_bodyid[g]) for g in p.foot_gids]
        result = f.contact_measurements(d, p, 2)
        for side in range(2):
            bid = p.foot_bids[side]
            weights = []
            v2 = []
            for k in range(d.ncon):
                c = d.contact[k]
                if not (
                    p.floor_gid in c.geom
                    and any(g in c.geom for g in [p.foot_gids[side], p.heel_gids[side]])
                ):
                    continue
                jp = np.zeros((3, m.nv))
                jr = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, jr, c.pos, bid)
                vel = jp @ d.qvel
                n = c.frame.reshape(3, 3)[0]
                vt = vel - n * np.dot(vel, n)
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, k, wrench)
                if c.dist <= 0 and wrench[0] > 0:
                    weights.append(wrench[0])
                    v2.append(vt @ vt)
            self.assertGreater(sum(weights), 2)
            self.assertAlmostEqual(
                result["slip"][side], np.sqrt(np.average(v2, weights=weights)), places=8
            )

    def run_pattern(self, pattern, seconds=10, curriculum=1.0, walking=True):
        s = f.initial(np.array([True, True]), self.c)
        events = []
        for k in range(round(seconds / 0.01)):
            s, pen, term, metrics = f.update(
                s,
                np.asarray(pattern(k * 0.01)),
                walking,
                0.25,
                False,
                curriculum,
                self.c,
                0.01,
            )
            events.append((term, metrics["error"], s["bad_pairs"]))
        return s, events

    def test_ideal_alternating_cycles(self):
        s, e = self.run_pattern(lambda t: [t % 0.6 >= 0.2, not (0.3 <= t % 0.6 < 0.5)])
        self.assertGreaterEqual(min(s["count"]), 2)
        self.assertFalse(any(x[0] for x in e))
        self.assertLess(e[-1][1], 0.05)

    def test_persistence_cold_gate_and_zero_command(self):
        pattern = lambda t: [t % 1 >= 0.2, t % 1 >= 0.6]
        _, e = self.run_pattern(pattern)
        self.assertTrue(any(x[0] for x in e))
        self.assertTrue(all(not term for term, err, pairs in e if pairs < 2))
        _, cold = self.run_pattern(pattern, curriculum=0.15)
        self.assertFalse(any(x[0] for x in cold))
        s, stop = self.run_pattern(pattern, walking=False)
        self.assertFalse(any(x[0] for x in stop))
        self.assertEqual(s["count"].sum(), 0)

    def test_one_planted_leg(self):
        _, e = self.run_pattern(lambda t: [t % 0.6 >= 0.2, True])
        self.assertFalse(any(x[0] for x in e[:790]))
        self.assertTrue(any(x[0] for x in e[810:]))

    def test_recovery_discards_stale_evidence(self):
        s, _ = self.run_pattern(lambda t: [t % 1 >= 0.2, t % 1 >= 0.6])
        s, p, term, m = f.update(
            s, np.array([True, True]), True, 0.25, True, 1.0, self.c, 0.01
        )
        self.assertFalse(term)
        self.assertEqual(p, 0)
        self.assertEqual(s["count"].sum(), 0)

    def test_flat_support_phase_windows(self):
        phase = np.array([0.1, 0.2, 0.5, 0.79, 0.8, 0.99, 1.1]) * 2 * np.pi * 0.6
        np.testing.assert_array_equal(
            f.midstance(phase, 0.6, 0.2, 0.8),
            [False, True, True, True, False, False, False],
        )
        np.testing.assert_array_equal(
            f.midstance(phase, 0.6, 0.3, 0.7),
            [False, False, True, False, False, False, False],
        )
        f.validate(self.c)
        with self.assertRaises(ValueError):
            f.validate(replace(self.c, flat_stance_start_fraction=0.9))

    def test_legacy_recipe_preserved(self):
        old, new = map(asdict, [get_config("cad_v2_walk01"), self.c])
        changed = {k for k in old if old[k] != new[k]}
        self.assertEqual(
            changed,
            {"two_point_feet", "symmetry_enable", "flat_support_enable", "w_foot_flat"},
        )
        self.assertEqual(new["total_steps"], 200_000_000)
        self.assertFalse(old["two_point_feet"])


if __name__ == "__main__":
    unittest.main()
