import unittest
import numpy as np
import mujoco
from quiet_push_protocol import cases,force_vector,active,quiet_counts,PUSH_TICK,PULSE_TICKS,DT


class PushProtocolTests(unittest.TestCase):
    def test_paired_cases_and_unique_zero_controls(self):
        c=cases();self.assertEqual(len(c),66)
        self.assertEqual(sum(v['force_n']==0 for v in c),2)
        for force in (2,5,10,15,20,30,40,60):
            for direction in ('forward','backward','left','right'):
                self.assertEqual([v['repeat'] for v in c if v['force_n']==force and v['direction']==direction],[0,1])

    def test_exact_pulse_impulse_and_no_early_force(self):
        ticks=np.arange(2000);mask=active(ticks)
        self.assertEqual(mask.sum(),10)
        self.assertEqual(np.flatnonzero(mask).tolist(),list(range(600,610)))
        self.assertAlmostEqual(30*DT*mask.sum(),3.)

    def test_horizontal_body_relative_direction(self):
        yaw=.6;roll=.2
        rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
        rx=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        for angle in (0,np.pi/2,np.pi,-np.pi/2):
            f=force_vector(rz@rx,30.,angle)
            self.assertAlmostEqual(np.linalg.norm(f),30.)
            self.assertEqual(f[2],0.)
        np.testing.assert_allclose(force_vector(rz,10,0),10*rz[:,0],atol=1e-12)

    def test_warmup_cannot_pay_for_post_push_hold(self):
        s=l=0
        for tick in range(610):s,l,ok=quiet_counts(s,l,True,True,tick)
        self.assertFalse(ok);self.assertEqual(s,0)
        for tick in range(610,1109):s,l,ok=quiet_counts(s,l,True,True,tick)
        self.assertFalse(ok)
        s,l,ok=quiet_counts(s,l,True,True,1109)
        self.assertTrue(ok);self.assertEqual(s,500)

    def test_missing_support_or_interrupted_posture_does_not_pass(self):
        s=l=0
        for tick in range(610,1110):s,l,ok=quiet_counts(s,l,True,False,tick)
        self.assertFalse(ok)
        s,l,ok=quiet_counts(s,l,False,True,1110)
        self.assertEqual(s,0);self.assertEqual(l,0)

    def test_newtons_applied_at_com_give_expected_native_impulse(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.001" gravity="0 0 0"/><worldbody><body name="b"><freejoint/><geom type="sphere" size=".1" mass="12"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        d.xfrc_applied[1,:3]=[30.,0,0]
        for _ in range(100):mujoco.mj_step(m,d)
        np.testing.assert_allclose(d.qvel[:3],[30*.1/12,0,0],atol=1e-10)
        np.testing.assert_allclose(d.qvel[3:],0.,atol=1e-10)


if __name__=='__main__':unittest.main()
