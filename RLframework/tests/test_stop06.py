import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RLframework'))
sys.path.insert(0,str(ROOT/'controller/deploy'))
from config import get_config,config_to_dict
from stop_schedule import feedback_hold
from stop02_metrics import summarize
from stop04_metrics import enforce_preservation


class Stop06Tests(unittest.TestCase):
    def test_gate_waits_then_latches_and_clears_on_run(self):
        h,c,t=feedback_hold(True,False,True,False,0.,.4,.1,1.,0.)
        self.assertFalse(h)
        for i in range(30):
            h,c,t=feedback_hold(True,False,True,h,c,.05,.1,1.,i*.01)
        self.assertTrue(h);self.assertFalse(t)
        h,c,t=feedback_hold(True,False,True,h,c,.9,2.,.5,1.)
        self.assertTrue(h)
        self.assertFalse(feedback_hold(False,False,True,h,c,0.,0.,1.,2.)[0])

    def test_gate_interruptions_invalid_estimate_and_timeout(self):
        self.assertEqual(feedback_hold(True,False,True,False,.2,.05,2.,1.,1.)[1],0.)
        self.assertFalse(feedback_hold(True,False,True,False,.29,np.nan,0.,1.,1.)[0])
        h,c,t=feedback_hold(True,False,True,False,0.,.9,2.,.5,4.)
        self.assertTrue(h);self.assertTrue(t)
        self.assertFalse(feedback_hold(True,True,False,True,.4,0.,0.,1.,-2.)[0])

    def test_only_gate_differs_between_comparison_arms(self):
        a=config_to_dict(get_config('dash_stop06_timed'))
        b=config_to_dict(get_config('dash_stop06_feedback'))
        self.assertEqual([k for k in a if a[k]!=b[k]],['stop_feedback_hold'])
        self.assertEqual(a['total_steps'],5_000_000)
        self.assertEqual(a['stop_hold_window_s'],9.)
        self.assertEqual(get_config('dash_stop05_motion').stop_guard_patience,1)

    def test_late_hold_still_requires_five_seconds_and_survival(self):
        n=1700; shape=(n,1)
        trace={k:np.zeros(shape) for k in ('speed','gyro','joint_rms','vx')}
        trace.update(ramp_duration=np.full(shape,2.),upright=np.ones(shape),
                     supported=np.ones(shape,bool),xy=np.zeros(shape+(2,)),done=np.zeros(shape,bool),
                     hold_active=np.zeros(shape,bool),gate_timeout=np.ones(shape,bool))
        trace['vx'][:100]=.3
        trace['hold_active'][700:]=True  # 6s after STOP; five seconds until deadline 12s
        args=(trace,[.3],np.array([100]),np.array([1300]),.01)
        result=summarize(*args,hold_seconds=9.)
        self.assertEqual(result['success_fraction'],1.)
        self.assertEqual(result['episodes'][0]['controller_hold_entry_s'],6.)
        self.assertIsNone(result['episodes'][0]['termination_phase'])
        trace['hold_active'][700]=False
        self.assertEqual(summarize(*args,hold_seconds=9.)['success_fraction'],0.)
        trace['hold_active'][700]=True;trace['done'][1201]=True
        self.assertEqual(summarize(*args,hold_seconds=9.)['success_fraction'],0.)

    def test_persistent_guard_survives_resume_and_ignores_duplicate_steps(self):
        with TemporaryDirectory() as d:
            run=Path(d);cfg=SimpleNamespace(stop_guard_patience=3,stop_guard_grace_steps=100)
            agent=SimpleNamespace(run=run,cfg=cfg,step=50,save=lambda p:p.write_bytes(b'keeper'))
            baseline={'success_fraction':0.}
            rows=[]
            def check(step,ok):
                agent.step=step
                r=dict(step=step,walking_retained=ok,standing_retained=True,success_fraction=0.)
                rows.append(r)
                (run/'stop_evaluations.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                enforce_preservation(agent,baseline,r)
            check(50,False);check(100,False);check(100,False);check(200,True)
            check(300,False);check(400,False)
            with self.assertRaises(SystemExit):check(500,False)

    def test_jax_gate_matches_numpy(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            self.skipTest('JAX is validated on SCITAS')
        values=(True,False,True,False,.29,np.array([.05,.9]),.1,1.,.5)
        expected=feedback_hold(*values)
        actual=jax.jit(lambda speed:feedback_hold(*values[:5],speed,*values[6:],xp=jnp))(jnp.array(values[5]))
        for a,b in zip(expected,actual):np.testing.assert_allclose(a,b,atol=1e-6)

    def test_jitted_adapter_uses_imu_and_is_idempotent(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            self.skipTest('JAX is validated on SCITAS')
        from collections import namedtuple
        from stop_adapter import prepare
        cfg=get_config('dash_stop06_feedback')
        vals=dict(stop_cmd=1.,stop_previous=True,v_cmd=.4,stop_estimate=0.,stop_v0=.4,
                  stop_started=0.,stop_duration=2.,t=3.,effective_speed=0.,effective_hold=False,
                  stop_phase=1,stop_prepared_t=2.99,stop_ready_s=.1,stop_gate_timeout=False,stop_gate_speed=0.)
        vals={k:jnp.asarray([v]) for k,v in vals.items()}
        vals['hist']=jnp.zeros((1,19,34)).at[0,-1,20].set(-cfg.obs_scales['gravity'])
        State=namedtuple('TestState',vals);State.replace=State._replace
        state=State(**vals)
        net=SimpleNamespace(apply=lambda *a,**k:jnp.zeros((1,3)),estimate=None)
        stats=SimpleNamespace(mean=jnp.zeros(413),var=jnp.ones(413),normalize=lambda x:x)
        env=SimpleNamespace(cfg=cfg,actor_dim=384,control_dt=.01)
        run=jax.jit(lambda s,o:prepare(net,None,stats,env,s,o))
        s,o=run(state,jnp.zeros((1,413)))
        np.testing.assert_allclose(s.stop_ready_s,[.11],atol=1e-6)
        twice,_=run(s,o)
        np.testing.assert_allclose(twice.stop_ready_s,s.stop_ready_s)
        disturbed=state.replace(hist=state.hist.at[0,-1,21].set(cfg.obs_scales['ang_vel']))
        bad,_=run(disturbed,jnp.zeros((1,413)))
        np.testing.assert_allclose(bad.stop_ready_s,[0.])

    def test_feedback_deployment_waits_for_lateral_velocity(self):
        sys.path.insert(0,str(ROOT/'RLframework/tools'))
        from play_joystick import Sim
        sim=Sim(ROOT/'RLframework/results/stop01_replay/dash_stop01b_final.npz')
        sim.bundle.meta['stop_adapter']=dict(enabled=True,version=2,feedback=True,decel=.5,minimum_s=2.,
            velocity_mean=0.,velocity_std=1.,velocity_scale=1.,velocity_mean_xy=[0.,0.],velocity_std_xy=[1.,1.],
            ready_speed=.15,ready_gyro=.5,ready_tilt_deg=15.,ready_dwell_s=.3,ready_max_wait_s=4.)
        sim.reset();sim.ctrl.set_speed(.3,immediate=True);sim.ctrl.set_run(True)
        estimate=np.array([.3,.2,0.],np.float32)
        sim.ctrl.net.estimate_velocity=lambda obs:estimate
        q,qd,tau=sim._motor_state()
        for _ in range(100):sim.ctrl.step(q,qd,tau,np.array([0.,0.,-1.]),np.zeros(3))
        sim.ctrl.set_run(False);estimate[:]=[0.,.2,0.]
        for _ in range(250):sim.ctrl.step(q,qd,tau,np.array([0.,0.,-1.]),np.zeros(3))
        self.assertEqual(sim.ctrl._task()[1],1.)
        estimate[:]=[.02,.02,0.]
        for _ in range(40):sim.ctrl.step(q,qd,tau,np.array([0.,0.,-1.]),np.zeros(3))
        self.assertEqual(sim.ctrl._task()[1],0.)


if __name__=='__main__':unittest.main()
