import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import get_config
from stop02_metrics import protocol,summarize
from stop04_metrics import command_duration,enforce_preservation


class Stop05Tests(unittest.TestCase):
    def test_low_speed_protocol_requires_moving_approach(self):
        cfg=get_config('dash_stop05_motion')
        speeds,starts,ends,ticks=protocol(3,.01,cfg)
        np.testing.assert_allclose(speeds,[.25,.375,.5])
        self.assertFalse(cfg.stop_standing_stage)
        self.assertEqual(cfg.cmd_range,(.1,.2))
        shape=(ticks,3)
        trace=dict(done=np.zeros(shape,bool),speed=np.zeros(shape),vx=np.zeros(shape),
            gyro=np.zeros(shape),joint_rms=np.zeros(shape),upright=np.ones(shape),
            supported=np.ones(shape,bool),xy=np.zeros(shape+(2,)),ramp_duration=np.full(shape,2.))
        self.assertEqual(summarize(trace,speeds,starts,ends,.01)['success_fraction'],0.)
        for i,a in enumerate(starts): trace['vx'][:a,i]=speeds[i]
        self.assertEqual(summarize(trace,speeds,starts,ends,.01)['success_fraction'],1.)
        self.assertTrue(np.all(ends-starts >= 900))
        old=get_config('dash_stop04_rehearsal')
        self.assertFalse(old.stop_motion_training)

    def test_durations_follow_forced_command(self):
        result=command_duration(np.array([0.,1.,0.,1.]),np.array([0.,0.,1.,1.]),(3.,4.),(10.,12.))
        np.testing.assert_allclose(result,[3.,10.,4.,12.])

    def test_shared_guard_saves_rejection_for_standing_loss(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            agent=SimpleNamespace(run=root,step=100,save=lambda p:p.write_bytes(b'checkpoint'))
            before=dict(success_fraction=0.)
            current=dict(success_fraction=.75,walking_retained=True,standing_retained=False)
            with self.assertRaises(SystemExit): enforce_preservation(agent,before,current)
            self.assertTrue((root/'guard_rejected.msgpack').exists())
            self.assertEqual(json.loads((root/'guard_stop.json').read_text())['step'],100)


if __name__=='__main__': unittest.main()
