"""Native constrained reset generation; no simulation settling or policy actions."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares

from standing_math import sole_centers

ACT_NAMES = ('hip_roll_L', 'cam_L', 'thigh_L', 'hip_roll_R', 'cam_R', 'thigh_R')
LEVEL_DEG = (.5, 1.5, 3., 5., 8.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Generator:
    def __init__(self, model, reference):
        self.m = m = mujoco.MjModel.from_xml_path(str(model))
        self.d = mujoco.MjData(m)
        self.reference = json.loads(Path(reference).read_text())
        assert self.reference['model_sha256'] == digest(model)
        self.q0 = np.array(self.reference['reference_qpos_for_visualization_only'])
        self.act = np.array([m.joint(n).qposadr[0] for n in ACT_NAMES])
        self.base = np.array([m.joint('base_'+n).qposadr[0] for n in ('x','y','z','roll','pitch','yaw')])
        self.free = np.array([i for i in range(m.nq) if i not in self.base[[0,1,5]]])
        first = np.array([m.geom('foot_'+s+'_col').id for s in 'LR'])
        second = np.array([m.geom('heel_'+s+'_col').id if 'heel_'+s+'_col' in
                           [m.geom(i).name for i in range(m.ngeom)] else first[k]
                           for k,s in enumerate('LR')])
        self.p = SimpleNamespace(m=m, foot_gids=first, heel_gids=second)
        self.gids = np.unique(np.r_[first,second])
        self.sites = [(m.site(a+'_'+s).id, m.site(b+'_'+s).id) for s in 'LR'
                      for a,b in [('rod_tip','leg_anchor')] +
                      ([('achilles_tip','foot_anchor')] if m.neq == 4 else [])]
        self.bid = m.body('torso').id
        mujoco.mj_forward(m,self.d)
        self.set(self.q0)
        self.feet0 = sole_centers(self.d,self.p).copy()
        self.lo, self.hi = np.full(m.nq,-10.),np.full(m.nq,10.)
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                self.lo[m.jnt_qposadr[j]],self.hi[m.jnt_qposadr[j]] = m.jnt_range[j]+[1e-5,-1e-5]

    def set(self,q):
        self.d.qpos[:] = q
        # Kinematics only: never settle, hold, or inject corrective simulation impulses.
        mujoco.mj_kinematics(self.m,self.d)
        mujoco.mj_comPos(self.m,self.d)

    def gaps(self):
        gaps=[]
        for first,second in zip(self.p.foot_gids,self.p.heel_gids):
            gaps.append(min(self.d.geom_xpos[g,2]-np.abs(self.d.geom_xmat[g].reshape(3,3)[2])@
                            self.m.geom_size[g] for g in set((first,second))))
        return np.array(gaps)

    def loops(self):
        return np.ravel([self.d.site_xpos[a]-self.d.site_xpos[b] for a,b in self.sites])

    def audit(self,q):
        self.set(q)
        feet = sole_centers(self.d,self.p)
        R = self.d.xmat[self.bid].reshape(3,3)
        centered=feet[:,:2]-feet[:,:2].mean(0)
        centered0=self.feet0[:,:2]-self.feet0[:,:2].mean(0)
        return dict(loop_max_m=float(np.abs(self.loops()).max()),
            sole_gap_m=self.gaps().tolist(),
            foot_xy_shift_m=np.linalg.norm(feet[:,:2]-self.feet0[:,:2],axis=1).tolist(),
            foot_shape_error_m=float(np.linalg.norm(centered-centered0,axis=1).max()),
            joint_max_deg=float(np.rad2deg(np.abs(q[self.act]-self.q0[self.act]).max())),
            joint_rms_deg=float(np.rad2deg(np.sqrt(np.mean((q[self.act]-self.q0[self.act])**2)))),
            tilt_deg=float(np.rad2deg(np.arccos(np.clip(R[2,2],-1,1)))),
            within_limits=bool(((q>=self.lo)&(q<=self.hi)).all()))

    def sample(self,rng,level):
        width=np.deg2rad(LEVEL_DEG[level])
        wanted=self.q0.copy()
        bilateral=rng.uniform(-width,width,3)
        wanted[self.act] += np.r_[bilateral,-bilateral]+rng.uniform(-.2*width,.2*width,6)
        wanted[self.base[3:5]] += rng.uniform(-min(width,np.deg2rad(6)),min(width,np.deg2rad(6)),2)
        wanted[self.base[2]] += rng.uniform(-.01,.01)*width/np.deg2rad(8)
        def residual(x):
            q=self.q0.copy(); q[self.free]=x
            self.set(q)
            feet=sole_centers(self.d,self.p)
            return np.r_[self.loops()/1e-5, (self.gaps()+.0001)/1e-5,
                ((feet[:,:2]-feet[:,:2].mean(0))-(self.feet0[:,:2]-self.feet0[:,:2].mean(0))).ravel()/.01,
                (q[self.act]-wanted[self.act])/.02,
                (q[self.base[3:5]]-wanted[self.base[3:5]])/.02,
                (q[self.base[2]]-wanted[self.base[2]])/.02,
                (q[self.free]-self.q0[self.free])*.01]
        fit=least_squares(residual,np.clip(wanted[self.free],self.lo[self.free]+1e-6,self.hi[self.free]-1e-6),
                          bounds=(self.lo[self.free],self.hi[self.free]),
                          max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-8,x_scale='jac')
        q=self.q0.copy(); q[self.free]=fit.x
        a=self.audit(q)
        good=(a['within_limits'] and a['loop_max_m']<.0001
              and max(np.abs(np.array(a['sole_gap_m'])+.0001))<.00015
              and a['foot_shape_error_m']<.015
              and a['joint_rms_deg']>=.15*LEVEL_DEG[level]
              and a['tilt_deg']<10 and a['joint_max_deg']<LEVEL_DEG[level]*1.5+.5)
        return (q,a) if good else None


def validate_bank(path,model,reference):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['meta'].item()))
        assert meta['model_sha256']==digest(model), 'reset bank model mismatch'
        assert meta['reference_sha256']==digest(reference), 'reset bank reference mismatch'
        arrays={key:z[key].copy() for key in ('train','probe','test')}
        assert all(a.ndim==3 and a.shape[0]==len(LEVEL_DEG) and np.isfinite(a).all() for a in arrays.values())
        assert all(np.isfinite(z['reference_qpos']))
        return arrays, z['reference_qpos'].copy(),meta
