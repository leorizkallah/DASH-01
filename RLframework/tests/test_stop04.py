import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stop04_metrics import diagnose,rehearsal_loss,preservation_failed


class Stop04Tests(unittest.TestCase):
    def trace(self):
        shape=(700,2)
        return dict(done=np.zeros(shape,bool),speed=np.zeros(shape),gyro=np.zeros(shape),
                    joint_rms=np.zeros(shape),upright=np.ones(shape),supported=np.ones(shape,bool),
                    xy=np.zeros(shape+(2,)))

    def test_joint_violation_and_fall_are_separate(self):
        t=self.trace();t['joint_rms'][250,0]=1.;t['done'][100,1]=True
        d=diagnose(t)
        self.assertEqual(d['successes'],0)
        self.assertEqual(d['successful_if_removed']['joints'],1)
        self.assertEqual(d['successful_if_removed']['speed'],0)
        self.assertFalse(d['episodes'][1]['success_if_criterion_removed']['joints'])

    def test_drift_and_support_remain_required(self):
        t=self.trace();t['xy'][:,0,0]=np.arange(700)*.001;t['supported'][:,1]=False
        d=diagnose(t)
        self.assertEqual(d['successes'],0)
        self.assertEqual(d['successful_if_removed']['drift'],1)
        self.assertEqual(d['successful_if_removed']['support'],1)

    def test_balanced_loss_and_preservation(self):
        p=np.zeros((4,47));y=p.copy();p[:2]=.12;p[2:]=.24
        loss,h,w=rehearsal_loss(p,y)
        self.assertAlmostEqual(loss,4.5)
        self.assertAlmostEqual(h,.0144)
        self.assertAlmostEqual(w,.0576)
        b=dict(success_fraction=21/32)
        self.assertFalse(preservation_failed(b,dict(success_fraction=20/32,walking_retained=True)))
        self.assertTrue(preservation_failed(b,dict(success_fraction=17/32,walking_retained=True)))
        self.assertTrue(preservation_failed(b,dict(success_fraction=1.,walking_retained=False)))

    def test_jax_actor_rehearsal_gradient_when_available(self):
        try:
            import jax
            import jax.numpy as jnp
            from networks import ActorCritic
        except ImportError:
            self.skipTest('JAX/Flax unavailable locally; exercised on SCITAS')
        net=ActorCritic(n_actor=384,n_priv=29,action_dim=47)
        obs=jnp.ones((4,413));p=net.init(jax.random.PRNGKey(9),obs)
        loss,grad=jax.value_and_grad(lambda p: rehearsal_loss(
            net.apply(p,obs,method=net.actor_mean),jnp.zeros((4,47)),jnp)[0])(p)
        self.assertTrue(np.isfinite(float(loss)))
        for key in ('estimator','value_net','value_head','log_std'):
            self.assertTrue(all(np.all(np.asarray(v)==0) for v in jax.tree_util.tree_leaves(grad['params'][key])))
        self.assertTrue(any(np.any(np.asarray(v)!=0) for v in jax.tree_util.tree_leaves(grad['params']['action_net'])))


if __name__=='__main__': unittest.main()
