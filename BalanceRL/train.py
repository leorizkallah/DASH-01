"""Train the balance policy.

    python BalanceRL/train.py --preset balance --name bal_s0 --seed 0 --devices 2
    python BalanceRL/train.py --preset smoke --name smoke            # CPU, a few iterations

Writes to BalanceRL/runs/<name>/: progress.jsonl (one line per iteration), ckpt_<step>.{msgpack,json}
every cfg.ckpt_every iterations, best.{msgpack,json} by the greedy push-ladder score, final.*.
--resume auto picks up the newest checkpoint in the run directory (the SLURM chain relies on it).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                                    # noqa: E402

from config import get_config, config_to_dict                          # noqa: E402


def newest_ckpt(run):
    c = sorted(run.glob("ckpt_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if (run / "final.json").exists():
        return run / "final"
    return c[-1].with_suffix("") if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="balance")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--eval-envs", type=int, default=240)
    ap.add_argument("--resume", default=None, help="'auto' or a checkpoint path (no suffix)")
    ap.add_argument('--initialize-from', type=Path, default=None,
                    help='fine-tune weights + normalizer; reset optimizers/curriculum/step counters')
    ap.add_argument("--walltime-hours", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (python literal)")
    a = ap.parse_args()

    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        import ast
        over[k] = ast.literal_eval(v)
    if a.n_envs:
        over["n_envs"] = a.n_envs
    if a.n_steps:
        over["n_steps"] = a.n_steps
    if a.steps:
        over["total_steps"] = a.steps
    cfg = get_config(a.preset, **over)

    import jax
    from ppo import Trainer
    run = HERE / "runs" / a.name
    tr = Trainer(cfg, run, seed=a.seed, n_devices=a.devices, n_eval=a.eval_envs)
    if a.initialize_from is not None:
        if a.resume:
            raise ValueError('Choose resume or initialize-from, not both')
        if any(run.glob('ckpt_*.json')) or (run/'final.json').exists():
            raise ValueError('Fine-tuning initialization requires a new run directory')
        print('[train] fine-tuning', tr.initialize_from(a.initialize_from), flush=True)
    if a.resume:
        ck = newest_ckpt(run) if a.resume == "auto" else Path(a.resume)
        if ck is not None:
            side = tr.load(ck)
            print(f"[train] resumed {ck} at step {tr.step:,}, curriculum {tr.cur}")
            if ck.name == "final":
                print("[train] run already finished")
                return
    (run / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=1))
    print(f"[train] {a.name}: {jax.devices()} x{a.devices}, batch {tr.batch:,} "
          f"({cfg.n_envs} envs x {cfg.n_steps}), {tr.n_mb} minibatches, actor {tr.env.actor_dim} "
          f"obs {tr.env.obs_dim}, action {tr.env.action_dim}, budget {cfg.total_steps:,}")
    prog = open(run / "progress.jsonl", "a")
    t_start = time.time()
    def evaluate_task():
        ev=tr.evaluate()
        if cfg.quiet_pose_tracking:
            import quiet_tracking_eval
            retained=quiet_tracking_eval.evaluate(tr,seed=1234,retention=True)
            ev['retention_score']=retained['score']
            ev['retention_reference_rate']=retained['quiet_rate']
        return ev
    def consider_best(ev):
        if not (cfg.recovery_steps or cfg.quiet_pose_tracking):
            if ev['score'] > tr.cur['best_score']:
                tr.cur['best_score'] = ev['score']
                tr.save('best', extra=dict(eval=ev))
            return
        qualified = ev.get('qualified', True)
        previous_qualified = tr.cur.get('best_qualified', False)
        if ((qualified and not previous_qualified) or
            (qualified == previous_qualified and ev['score'] > tr.cur['best_score'])):
            tr.cur['best_score'] = ev['score']
            tr.cur['best_qualified'] = qualified
            tr.save('best', extra=dict(eval=ev))

    if (cfg.recovery_steps or cfg.quiet_pose_tracking) and a.initialize_from is not None:
        baseline = evaluate_task()
        (run/'baseline_eval.json').write_text(json.dumps(baseline, indent=2))
        tr.save('baseline', extra=dict(eval=baseline))
        consider_best(baseline)
    while tr.step < cfg.total_steps:
        log, moved = tr.iterate()
        n = tr.rollout_n
        if moved:
            print(f"[curriculum] @ {tr.step:,}: {moved.strip()}")
        if cfg.quiet_pose_tracking and n % cfg.quiet_pose_probe_every == 0:
            import quiet_tracking_eval
            from quiet_tracking_math import promote
            number=int(tr.cur.get('pose_probe_number',0))
            probe=quiet_tracking_eval.evaluate(tr,seed=920000+number,probe=True)
            tr.cur['pose_probe_number']=number+1
            changed=promote(tr.cur,probe)
            log.update({'probe/pose_rates':probe['rates'],'curriculum/pose_level':tr.cur['pose_level']})
            print(f"[pose probe] {tr.step:,}: {probe['rates']}; {changed}",flush=True)
        if cfg.recovery_steps and n % cfg.recovery_probe_every == 0:
            import recovery_eval
            number = int(tr.cur.get('probe_number', 0))
            probe = recovery_eval.evaluate(tr, seed=910000+number, probe=True)
            tr.cur['probe_number'] = number+1
            changed = recovery_eval.update_curriculum(tr, probe)
            log.update({'probe/hard_rate':probe['hard_rate'], 'probe/quiet_rate':probe['quiet_rate'],
                        'probe/worst_sector':min(probe['sector_rates']),
                        'curriculum/push_level':tr.cur['push_level'],
                        'curriculum/push_stage':tr.cur.get('push_stage', 0)})
            print(f"[probe] {tr.step:,}: hard {probe['hard_rate']:.3f}, quiet {probe['quiet_rate']:.3f}, "
                  f"worst sector {min(probe['sector_rates']):.3f}; {changed}", flush=True)
        if n % cfg.eval_every == 0 and tr.eval_env is not None:
            ev = evaluate_task()
            log["eval/score"] = ev["score"]
            log["eval/quiet_falls"] = ev["quiet_falls"]
            if cfg.quiet_pose_tracking:
                log['eval/retention_score']=ev['retention_score']
                log['eval/retention_reference_rate']=ev['retention_reference_rate']
            for k, v in ev["survival"].items():
                log[f"eval/surv_{k}"] = v
            consider_best(ev)
            print(f"[eval] @ {tr.step:,}: score {ev['score']:.3f} {ev['survival']} quiet {ev['quiet_falls']}"
                  f"  (best {tr.cur['best_score']:.3f})")
        log["curriculum/bins"] = tr.survival_bins()
        prog.write(json.dumps(log) + "\n")
        prog.flush()
        if n % 10 == 0 or n <= 3:
            print(f"[{tr.step/1e6:7.2f}M] sps {log['time/sps']:,.0f} ep_len {log['rollout/ep_len_mean']:.0f} "
                  f"falls {log['rollout/falls']} quiet {log['rollout/quiet_falls']} "
                  f"hard_surv {log['rollout/hard_survival']:.2f} | plant {tr.cur['plant_scale']:.2f} "
                  f"push {tr.cur['push_level']:.3f} | kl {log['train/kl']:.4f} lr {tr.lr:.1e} "
                  f"std {log['train/std_mean']:.3f} vf {log['train/vf']:.3f} kp {log['diag/kp_mean']:.0f} "
                  f"kd {log['diag/kd_mean']:.2f}", flush=True)
        if n % cfg.ckpt_every == 0:
            tr.save(f"ckpt_{tr.step}")
        if a.walltime_hours is not None and time.time()-t_start >= a.walltime_hours*3600:
            print('[train] wall-time save reserve reached', flush=True)
            break
    if cfg.standing_compare and tr.eval_env is not None:
        ev = evaluate_task()
        (run/'final_eval.json').write_text(json.dumps(ev, indent=2))
        consider_best(ev)
    tr.save("final")
    (run/'completion.json').write_text(json.dumps(dict(steps=tr.step,
        reason='step_limit' if tr.step >= cfg.total_steps else 'walltime_reserve',
        elapsed_s=time.time()-t_start), indent=2))
    print(f"[train] done: {tr.step:,} steps in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
