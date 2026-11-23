"""Deterministically build disjoint feasible training/probe/test posture banks."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from quiet_reset import Generator,LEVEL_DEG,digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+',default=['v1','v2','xshape'])
    ap.add_argument('--train',type=int,default=64)
    ap.add_argument('--heldout',type=int,default=16)
    a=ap.parse_args()
    for name in a.models:
        start=time.monotonic()
        model=PACKAGE/f'model/stand_{name}.xml'; reference=PACKAGE/f'model/quiet_{name}.json'
        g=Generator(model,reference)
        arrays={}; audits={}; attempts={}
        # Distinct seeds and independent solves; reference is a single exact baseline only.
        for split,seed,count in [('train',7001,a.train),('probe',8001,a.heldout),('test',9001,a.heldout)]:
            rng=np.random.default_rng(seed)
            arrays[split]=[]; audits[split]=[]; attempts[split]=[]
            for level in range(len(LEVEL_DEG)):
                poses=[]; metrics=[]; tried=0
                while len(poses)<count and tried<count*30:
                    tried+=1
                    sample=g.sample(rng,level)
                    if sample is not None:
                        q,info=sample; poses.append(q); metrics.append(info)
                assert len(poses)==count,(name,split,level,len(poses),tried)
                arrays[split].append(poses); audits[split].append(metrics); attempts[split].append(tried)
            print(name,split,'generated',flush=True)
        meta=dict(model=name,model_sha256=digest(model),reference_sha256=digest(reference),
            target_motor_noise_deg=list(LEVEL_DEG),attempts=attempts,
            description='Constrained native kinematic projection; zero velocities; floor gap -0.1 mm per foot; no settling.',
            audits=audits,reference_audit=g.audit(g.q0),elapsed_s=time.monotonic()-start)
        path=PACKAGE/f'model/quiet_resets_{name}.npz'
        np.savez_compressed(path,**{k:np.asarray(v) for k,v in arrays.items()},reference_qpos=g.q0,
                            meta=np.array(json.dumps(meta)))
        print(name,'wrote',path.stat().st_size,'bytes in',round(meta['elapsed_s'],1),'s',flush=True)


if __name__=='__main__': main()
