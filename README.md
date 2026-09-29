# DASH-01 — bipedal robot 100 m sprint

This repository continues the work from [nathmo/running_robot](https://github.com/nathmo/running_robot),
developed by [nathmo](https://github.com/nathmo) for his master's thesis on DASH-01,
a custom biped learning to sprint 100 m with reinforcement learning (RL).

I am continuing this work as part of my master's semester project at EPFL, building on
the RL training pipeline that nathmo already implemented. The original robot code,
training pipeline, and experiments are his work; this repository is where I develop
the continuation of that project.

The original Git history and [Mozilla Public License 2.0](LICENSE) are preserved.

Two self-contained halves:

- **[`training/`](training/README.md)** — the RL training stack (MuJoCo + PPO): teach DASH-01 to
  sprint 100 m from a standing start and stop. Clone, venv, `pip install -r
  training/requirements.txt`, train. Includes SLURM scripts for the EPFL Izar cluster
  ([training/slurm/README.md](training/slurm/README.md)).
- **[`robot/`](robot/README.md)** — the physical robot: CAD, simulation-model build, motor/CAN
  tools, the fixed-gait demo + web control UI. Will eventually move to its own repo.

## 30-second start

```bash
python -m venv .venv
# activate it — PowerShell: .venv\Scripts\Activate.ps1 | Git Bash: source .venv/Scripts/activate
#                | Linux/macOS: source .venv/bin/activate
pip install -r training/requirements.txt
python training/smoke_test.py
python training/train.py --preset m2_sprint --steps 240000000 --n-envs 20 --subproc
```

Everything else: [training/README.md](training/README.md).

## History

The previous experimentation stack (per-step PD policies, per-cycle Fourier gaits, the
experiments/framework/orchestrator machinery) was removed on 2026-07-17 after a literature
review — it lives in git history before that date. The current stack is the reviewed state of
the art: per-step re-parameterized Fourier gait + residuals (CPG-RL/PMTG hybrid), phase-gated
contact scheduling, dense-speed + clock-cost sprint reward, efficiency shaping, milestone
base-DOF curriculum.
