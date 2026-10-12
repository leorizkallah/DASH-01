"""Record matched command-only diagnostics and a standing feasibility probe on MJX."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from config import get_config
from train import make_eval_env
from env import EnvParams
from ppo import PPO
from bundle_warmstart import restore_bundle
from stop_adapter import prepare
from stop02_metrics import protocol, summarize


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--ramp', action='store_true')
    ap.add_argument('--standing', action='store_true')
    ap.add_argument('--dense', action='store_true')
    ap.add_argument('--zero-action', action='store_true')
    ap.add_argument('--n-envs', type=int, default=32)
    ap.add_argument('--seed', type=int, default=4102)
    args=ap.parse_args()
    cfg=get_config('dash_stop02_stand')
    cfg.stop_adapter=args.ramp
    cfg.stop_dense_hold=args.dense
    cfg.stop_standing_stage=args.standing
    cfg.stop_reset_frac=0.0
    if not args.dense:
        old=get_config('dash_stop')
        for k in ('w_stop_brake_income','w_stand_pose','w_stop_amp'):
            setattr(cfg,k,getattr(old,k))
    if args.zero_action:
        cfg.reset_joint_noise=0.0
    env=make_eval_env(cfg,args.n_envs)
    speeds,starts,ends,ticks=protocol(env.n_envs,env.control_dt,cfg)
    p=EnvParams.final(cfg)._replace(dr_scale=0.,bringup_scale=0.,bringup_p_drop=0.,
        bringup_p_held=0.,ctrl_jitter_ms=0.,ctrl_drop_prob=0.)
    reasons=('term_low','term_tip','term_floor','term_ws','term_nan')
    with TemporaryDirectory() as tmp:
        agent=PPO(cfg,env,Path(tmp),cfg.total_steps)
        restore_bundle(agent,args.bundle)
        def run(weights,stats):
            state,obs=env.reset(jax.random.PRNGKey(args.seed),p)
            state=state.replace(stop_previous=jnp.full_like(state.stop_previous,args.standing),
                effective_speed=jnp.asarray(speeds,jnp.float32),
                effective_hold=jnp.full_like(state.effective_hold,args.standing))
            obs=obs.at[:,env.actor_dim-3].set(speeds/cfg.v_max)
            obs=obs.at[:,env.actor_dim-2].set(float(not args.standing))
            def tick(carry,k):
                state,obs=carry
                stop=((k>=jnp.asarray(starts)) & (k<jnp.asarray(ends))).astype(jnp.float32)
                state=state.replace(v_cmd=jnp.asarray(speeds,jnp.float32),stop_cmd=stop,
                                    cmd_left=jnp.full_like(state.cmd_left,1e6))
                if cfg.stop_adapter:
                    state,obs=prepare(agent.net,weights,stats,env,state,obs)
                else:
                    obs=obs.at[:,env.actor_dim-3].set(speeds/cfg.v_max)
                    obs=obs.at[:,env.actor_dim-2].set(1-stop)
                action=jnp.clip(agent._act_greedy(weights,stats.normalize(obs)),-1,1)
                if args.zero_action:
                    action=jnp.zeros_like(action)
                ns,no,r,done,info=env.step(state,action,p)
                record=dict(qpos=state.data.qpos,qvel=state.data.qvel,spec=state.spec,action=action,
                    speed=info['stop_speed_xy'],vx=info['v_body_x'],gyro=info['stop_gyro'],
                    joint_rms=info['stop_joint_rms'],upright=info['stop_upright'],
                    supported=info['stop_supported'],xy=info['stop_xy'],done=done,reward=r,
                    floor=info['reward_floor_hit'],freq=info['freq_hz'],
                    ramp_duration=info['stop_duration'],reference=info['stop_reference'],
                    phase=info['stop_phase'],estimated_speed=state.stop_estimate,
                    actor_speed=obs[:,env.actor_dim-3]*cfg.v_max,actor_run=obs[:,env.actor_dim-2],
                    reasons=jnp.stack([info[x] for x in reasons],axis=-1),terms=info['reward_terms'])
                return (ns,no),record
            return jax.lax.scan(tick,(state,obs),jnp.arange(ticks))[1]
        print(f'[STOP02] record {ticks} ticks x {env.n_envs}',flush=True)
        data=jax.tree_util.tree_map(np.asarray,jax.jit(run)(agent.params,agent.stats))
    if cfg.stop_adapter:
        np.testing.assert_allclose(data['reference'],data['actor_speed'],atol=1e-6)
        np.testing.assert_array_equal(data['actor_run']==0,data['phase']==2)
        assert np.all(data['terms']['stand_hold'][data['phase']!=2]==0)
    assert np.isfinite(data['reward']).all()
    result=summarize(data,speeds,starts,ends,env.control_dt,args.standing)
    if not args.ramp and not args.standing:
        result['protocol']='STOP-02-abrupt-comparison'
    k=np.arange(ticks)[:,None]; live=~np.maximum.accumulate(data['done'],axis=0)
    mask=live & (k>=starts) & (k<ends)
    result['live_stop_reward_mean']=float(data['reward'][mask].mean()) if mask.any() else None
    result['reward_floor_fraction']=float(data['floor'][mask].mean()) if mask.any() else None
    result['reward_terms_mean']={n:float(v[mask].mean()) if mask.any() else None for n,v in data['terms'].items()}
    approach=live & (k<starts) & (k>=starts-100)
    result['estimated_speed_mae']=float(np.abs(data['estimated_speed']-data['vx'])[approach].mean()) if args.ramp and approach.any() else None
    names=list(data['terms']);data['reward_terms']=np.stack([data['terms'][n] for n in names],axis=-1);del data['terms']
    with np.load(args.bundle) as f:
        bundle_meta=json.loads(str(f['meta'].item()))
    meta=dict(step=bundle_meta['step'],run=bundle_meta['run'],checkpoint=args.bundle,
        control_dt=env.control_dt,model_path=cfg.model_path,n_envs=env.n_envs,seed=args.seed,
        speeds=speeds.tolist(),stop_ticks=starts.tolist(),restart_ticks=ends.tolist(),
        term_names=names,reason_names=list(reasons),ramp=args.ramp,standing=args.standing,
        zero_action=args.zero_action,dense=args.dense,adapter_version=1)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,meta=np.array(json.dumps(meta)),**data)
    args.out.with_suffix('.json').write_text(json.dumps(dict(metadata=meta,summary=result),indent=2)+'\n')
    print('[STOP02] '+json.dumps({k:v for k,v in result.items() if k not in ('episodes','reward_terms_mean')}),flush=True)


if __name__=='__main__':
    main()
