import sys
import json
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'RLframework'))
sys.path.insert(0, str(ROOT/'controller'/'deploy'))
from stop_schedule import advance, reference, duration
from stop02_metrics import hold_window, protocol, summarize, timeout_allowed
from config import get_config


class Stop02Tests(unittest.TestCase):
    def test_late_stop_gets_full_window(self):
        self.assertFalse(timeout_allowed(True,1.,3.))
        self.assertFalse(timeout_allowed(True,9.99,3.))
        self.assertTrue(timeout_allowed(True,10.,3.))
        self.assertTrue(timeout_allowed(False,0.,3.))
    def test_ramp_continuity_and_deceleration(self):
        t = np.linspace(0, 3, 3001)
        v = reference(1.0, t, duration(1.0))
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[-1], 0.0)
        self.assertTrue(np.all(np.diff(v) <= 0))
        self.assertLessEqual(np.max(-np.diff(v)/.001), .50001)
        self.assertLess(abs(v[1]-v[0])/.001, .001)

    def test_latched_stop_and_restart(self):
        x = advance(True, False, 2., 1., 1., 0., 0., 0., 5.)
        filt,v0,start,seconds,target,hold,phase = x
        self.assertEqual((start,seconds,target,hold,phase), (5.,3.,1.,False,1))
        y = advance(True, True, 2., 2., filt, v0, start, seconds, 8.)
        self.assertEqual(y[1],1.)
        self.assertEqual(y[2],5.)
        self.assertEqual(y[4],0.)
        self.assertTrue(y[5])
        z = advance(False, True, .5, 0., y[0],y[1],y[2],y[3],9.)
        self.assertEqual(z[4],.5)
        self.assertEqual(z[6],0)

    def test_five_seconds_and_reset(self):
        q = np.ones(700,bool); xy=np.zeros((700,2)); support=q.copy()
        self.assertIsNotNone(hold_window(q,support,xy,0,700,.01)[1])
        q[300] = False
        self.assertIsNone(hold_window(q,support,xy,0,700,.01)[1])
        self.assertIsNone(hold_window(np.ones(499,bool),support,xy,0,499,.01)[1])

    def test_drift_and_support(self):
        q=np.ones(700,bool); xy=np.zeros((700,2)); xy[:,0]=np.arange(700)*.0003
        self.assertIsNone(hold_window(q,q,xy,0,700,.01)[1])
        self.assertIsNone(hold_window(q,np.zeros(700,bool),xy*0,0,700,.01)[1])

    def test_protocol_and_old_preset(self):
        c=get_config('dash_stop02_stand'); _,s,e,_=protocol(32,.01,c)
        self.assertTrue(np.all(e-s==700))
        c.stop_standing_stage=False; _,s,e,_=protocol(32,.01,c)
        self.assertTrue(np.all(e-s>=1450))
        self.assertFalse(get_config('dash_stop').stop_adapter)

    def test_episode_reset_never_rescues_a_fall(self):
        n=1000
        trace={k:np.zeros((n,1)) for k in ('speed','gyro','joint_rms','vx','ramp_duration')}
        trace.update(upright=np.ones((n,1)),supported=np.ones((n,1),bool),
                     xy=np.zeros((n,1,2)),done=np.zeros((n,1),bool))
        r=summarize(trace,[0.],np.array([0]),np.array([700]),.01,True)
        self.assertEqual(r['success_fraction'],1.)
        json.dumps(r, allow_nan=False)
        trace['done'][30]=True
        r=summarize(trace,[0.],np.array([0]),np.array([700]),.01,True)
        self.assertEqual(r['success_fraction'],0.)
        self.assertLess(r['longest_quiet_hold_mean_s'],1.)

    def test_deployment_command_adapter(self):
        sys.path.insert(0,str(ROOT/'RLframework'/'tools'))
        from play_joystick import Sim
        sim=Sim(ROOT/'RLframework/results/stop01_replay/dash_stop01b_final.npz')
        sim.bundle.meta['stop_adapter']=dict(enabled=True,version=1,decel=.5,minimum_s=2.,
            velocity_mean=0.,velocity_std=1.,velocity_scale=1.)
        sim.reset(); sim.ctrl.set_speed(.8,immediate=True); sim.ctrl.set_run(True)
        sim.ctrl.net.estimate_velocity=lambda obs: np.array([.8,0,0],np.float32)
        q,qd,tau=sim._motor_state()
        for _ in range(200):
            sim.ctrl.step(q,qd,tau,sim._grav_body(),sim._gyro())
        sim.ctrl.set_run(False)
        sim.ctrl.step(q,qd,tau,sim._grav_body(),sim._gyro())
        self.assertAlmostEqual(sim.ctrl._stop_v0,.8,places=5)
        self.assertEqual(sim.ctrl._task()[1],1.)
        for _ in range(242):
            sim.ctrl.set_run(False)  # repeated STOP must not restart the clock
            sim.ctrl.step(q,qd,tau,sim._grav_body(),sim._gyro())
        self.assertEqual(sim.ctrl._task(),(0.,0.))


if __name__ == '__main__':
    unittest.main()
