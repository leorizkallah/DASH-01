"""Posture return is independent of floor location; recovery and return are distinct."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from config import get_config
from pose_reference import features, error, initial_phase
import test_recovery as helpers
from test_standing import load
from standing_math import sole_centers


class PoseFeatureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = get_config('return_v1')
        self.q = np.array([.01,.1,-.2,-.01,-.1,.2])
        self.base = np.array([0.,0.,.8])
        self.feet = np.array([[.01,.2,0.],[.01,-.2,0.]])
        self.ref = features(self.q,np.eye(3),self.base,self.feet)

    def test_floor_translation_and_heading_do_not_change_target(self):
        yaw = .7
        R = np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
        delta = np.array([.3,-.2,0.])
        current = features(self.q,R,R@self.base+delta,self.feet@R.T+delta)
        result = error(current,self.ref,self.cfg)
        self.assertTrue(result['ok'])
        self.assertAlmostEqual(result['score'],1.)

    def test_wrong_joint_or_foot_posture_cannot_pass_by_standing_still(self):
        q = self.q.copy(); q[0] += np.deg2rad(12)
        self.assertFalse(error(features(q,np.eye(3),self.base,self.feet),self.ref,self.cfg)['ok'])
        feet = self.feet.copy(); feet[0,0] += .08
        self.assertFalse(error(features(self.q,np.eye(3),self.base,feet),self.ref,self.cfg)['ok'])

    def test_each_saved_reference_is_an_actual_model_pose(self):
        for name in ('v1','v2','xshape'):
            cfg = get_config('return_'+name)
            record = json.loads((PACKAGE/cfg.quiet_reference_path).read_text())
            _,m,d,p = load(name)
            d.qpos[:] = record['reference_qpos_for_visualization_only']
            import mujoco
            mujoco.mj_forward(m,d)
            qadr = [m.joint(n).qposadr[0] for n in
                    ('hip_roll_L','cam_L','thigh_L','hip_roll_R','cam_R','thigh_R')]
            bid=m.body('torso').id
            f=features(d.qpos[qadr],d.xmat[bid].reshape(3,3),d.xpos[bid],sole_centers(d,p))
            self.assertTrue(error(f,record['features'],cfg)['ok'])
            self.assertEqual(record['candidate_match_fraction'],1.)


class PosePhaseTests(unittest.TestCase):
    tick = helpers.RecoveryTests.tick

    def setUp(self):
        helpers.RecoveryTests.setUp(self)
        self.cfg = get_config('return_v1')
        self.s.update(initial_phase())

    def test_recovering_in_an_unfamiliar_stance_is_not_success(self):
        self.tick(started=True,pose_ok=False)
        events=0
        for _ in range(400):
            m=self.tick(pose_ok=False)
            events+=int(m['balance_recovered'])
            self.assertFalse(m['success'])
            self.assertFalse(m['task_bad'])
            self.assertEqual(self.s['quiet_s'],0.)
        self.assertEqual(events,1)
        self.assertTrue(self.s['returning'])

    def test_steps_are_still_legal_while_returning_then_final_hold_commits(self):
        self.tick(started=True,pose_ok=False)
        moved=self.xy.copy(); moved[0,0]=.08
        for _ in range(330):
            m=self.tick(xy=moved,pose_ok=False)
            self.assertFalse(m['task_bad'])
        self.assertTrue(self.s['returning'])
        # Correction after 3 s is legal because balance was regained earlier.
        for _ in range(3):
            m=self.tick(xy=moved,pose_ok=False,normal_force=np.array([0.,70.]))
            self.assertFalse(m['task_bad'])
        shifted=self.xy+np.array([.08,0.])
        for _ in range(3):
            m=self.tick(xy=shifted,pose_ok=True)
            self.assertFalse(m['task_bad'])
        events=0
        for _ in range(510):
            m=self.tick(xy=shifted,pose_ok=True)
            events+=int(m['success'])
            self.assertFalse(m['task_bad'])
        self.assertEqual(events,1)
        np.testing.assert_allclose(self.s['anchor'],shifted)
        self.assertFalse(self.s['recovering'])

    def test_unreturned_pose_has_fixed_deadline(self):
        self.tick(started=True,pose_ok=False)
        failed=False
        for _ in range(1000):
            m=self.tick(pose_ok=False)
            if m['expired']:
                failed=True
                break
        self.assertTrue(failed)
        self.assertLess(self.t,9.6)

    def test_pose_match_without_regaining_support_cannot_complete(self):
        self.tick(started=True,pose_ok=True,balance_supported=False)
        for _ in range(320):
            m=self.tick(pose_ok=True,balance_supported=False)
            if m['expired']:
                break
        self.assertTrue(m['expired'])
        self.assertFalse(m['success'])
        self.assertFalse(self.s['returning'])


if __name__=='__main__':
    unittest.main()
