"""Quiet tracking: valid nontrivial starts, disjoint splits, sustained pose matching."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from config import get_config
from quiet_tracking_math import hold,promote,accept,validate_physics
from quiet_reset import Generator,validate_bank,LEVEL_DEG
from pose_reference import features,error
from standing_math import sole_centers


class QuietHoldTests(unittest.TestCase):
    def setUp(self):
        self.c=get_config('quiet_v1')

    def run_hold(self,quiet,loaded,start=0.,ticks=501):
        s=l=0.
        for k in range(ticks):
            s,l,ok=hold(s,l,quiet,loaded,start+(k+1)*.01,.01,self.c)
        return s,l,ok

    def test_five_seconds_and_real_loading_required(self):
        self.assertFalse(self.run_hold(True,True,ticks=499)[2])
        self.assertTrue(self.run_hold(True,True)[2])
        self.assertFalse(self.run_hold(True,False)[2])

    def test_motion_or_wrong_posture_resets_hold(self):
        s,l,_=self.run_hold(True,True,ticks=400)
        s,l,ok=hold(s,l,False,True,4.01,.01,self.c)
        self.assertEqual(s,0.)
        self.assertEqual(l,0.)
        self.assertFalse(ok)

    def test_late_return_cannot_extend_fixed_deadline(self):
        self.assertFalse(self.run_hold(True,True,start=6.1)[2])

    def test_promotion_needs_frontier_and_reference_retention_twice(self):
        cur={'pose_level':1}
        good={'rates':dict(reference=1.,**{'0':1.,'1':1.})}
        self.assertIsNone(promote(cur,good))
        poor={'rates':dict(reference=.8,**{'0':1.,'1':1.})}
        self.assertIsNone(promote(cur,poor))
        self.assertEqual(cur['pose_level'],1)
        self.assertIsNone(promote(cur,good))
        self.assertIsNotNone(promote(cur,good))
        self.assertEqual(cur['pose_level'],2)

    def test_completed_hold_survives_a_late_interruption(self):
        s=l=0.;completed=False
        for k in range(1200):
            s,l,ok=hold(s,l,k!=649,True,(k+1)*.01,.01,self.c)
            completed=completed or bool(ok)
        self.assertTrue(completed)
        self.assertFalse(ok)  # final streak begins at6.50s, later than first-return deadline
        self.assertGreater(s,5.)
        self.assertTrue(accept(completed,s,l,self.c))

    def test_partial_or_late_first_hold_is_not_accepted(self):
        self.assertFalse(accept(False,4.99,4.99,self.c))
        s,l,ok=self.run_hold(True,True,start=6.1)
        self.assertFalse(accept(bool(ok),s,l,self.c))

    def test_retention_has_independent_final_window(self):
        from dataclasses import replace
        c=replace(self.c,quiet_pose_retention_eval=True)
        # Initial timely hold was completed; a later final window can start after6s.
        self.assertTrue(accept(True,5.5,5.5,c))
        self.assertFalse(accept(False,5.5,5.5,c))
        self.assertFalse(accept(True,4.9,4.9,c))
        self.assertFalse(accept(True,5.5,4.,c))

    def test_preflight_does_not_require_mastery_of_new_pose_task(self):
        ev=dict(trials=96,score=0.,quiet_rate=0.,rates={'reference':0.},
                per_trial=dict(physical_fall=[False]*16,duration_s=[11.05]*16))
        self.assertEqual(validate_physics(ev)['reference_no_fall_rate'],1.)
        ev['per_trial']['physical_fall']=[True]*5+[False]*11
        with self.assertRaises(AssertionError): validate_physics(ev)
        ev['per_trial']['physical_fall']=[False]*16
        ev['score']=float('nan')
        with self.assertRaises(AssertionError): validate_physics(ev)


class QuietResetTests(unittest.TestCase):
    def test_banks_have_valid_linkages_floor_and_limits(self):
        for name in ('v1','v2','xshape'):
            c=get_config('quiet_'+name)
            model=PACKAGE/c.model_path;ref=PACKAGE/c.quiet_reference_path
            bank,q0,meta=validate_bank(PACKAGE/c.quiet_pose_bank,model,ref)
            g=Generator(model,ref)
            self.assertTrue(np.isfinite(q0).all())
            hashes=[]
            for split,arr in bank.items():
                seen=set()
                for level,poses in enumerate(arr):
                    for q in poses:
                        a=g.audit(q)
                        self.assertLess(a['loop_max_m'],.0001)
                        self.assertLess(max(np.abs(np.array(a['sole_gap_m'])+.0001)),.00015)
                        self.assertLess(a['foot_shape_error_m'],.015)
                        self.assertTrue(a['within_limits'])
                        self.assertGreaterEqual(a['joint_rms_deg'],.15*LEVEL_DEG[level])
                        seen.add(hashlib.sha256(q.tobytes()).hexdigest())
                self.assertEqual(len(seen),arr.shape[0]*arr.shape[1])
                hashes.append(seen)
            self.assertFalse(hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
            # Hardest bank must actually differ from the easiest, rather than project to zero.
            rms=lambda a:np.rad2deg(np.sqrt(np.mean((a[:,:,g.act]-q0[g.act])**2,axis=-1)))
            actual=rms(bank['train'])
            self.assertGreater(np.mean(actual[-1]),2*np.mean(actual[0]))

    def test_quiet_goal_is_not_the_model_keyframe_or_floor_location(self):
        for name in ('v1','v2','xshape'):
            c=get_config('quiet_'+name)
            self.assertFalse(c.recovery_steps or c.recovery_pose_return)
            self.assertEqual(c.hold_s_range,(0.,0.))
            g=Generator(PACKAGE/c.model_path,PACKAGE/c.quiet_reference_path)
            g.set(g.q0)
            f=features(g.q0[g.act],g.d.xmat[g.bid].reshape(3,3),g.d.xpos[g.bid],sole_centers(g.d,g.p))
            self.assertTrue(error(f,g.reference['features'],c)['ok'])


if __name__=='__main__': unittest.main()
