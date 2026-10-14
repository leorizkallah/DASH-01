# DASH-01 v2 — walking and autonomous stopping

This repository continues the work from [nathmo/running_robot](https://github.com/nathmo/running_robot),
developed by [nathmo](https://github.com/nathmo) for his master's thesis on DASH-01,
a custom biped learning to walk, stand and sprint with reinforcement learning (RL).

I am continuing this work as part of my master's semester project at EPFL, building on
the RL training pipeline that nathmo already implemented. The original robot code,
training pipeline, and experiments are his work; this repository is where I develop
the continuation of that project.

The original Git history and [Mozilla Public License 2.0](LICENSE) are preserved.

## Purpose of this branch

The `dash-01-v2` branch prepares for the new CAD model and a new policy-training
campaign. The model will be supplied later; its dedicated folder is
[`dash-01CAD-v2/`](dash-01CAD-v2/README.md).

Once the new model is available, we will integrate and validate it in simulation,
then redo training to find a policy for the revised robot. First, we will optimize
its walking gait and teach the policy to adapt that gait to different target
speeds. The demonstration will stay within walking speeds; transitions between
walking and running gaits are outside its scope.

We will then train and validate the complete Robotics Day demonstration:

1. **Automatic startup and standing.** From a defined initial position, the robot
   starts automatically, reaches an upright stance, and maintains balance while
   remaining robust to perturbations.
2. **START and joystick speed control.** After a `START` command, the robot walks.
   The operator can increase or decrease its target speed using joystick-style
   control, and the robot adapts its walking gait accordingly.
3. **STOP and autonomous standing.** After a `STOP` command, the goal is for the
   robot to decelerate, stop without physical assistance, and remain upright in a
   stable stance.

These are planned capabilities, not demonstrated results for the new model.
Training changes, evaluations, useful plots and videos will continue to be
documented in the [report workspace](report/README.md).

## Existing code and reference policies

The code, policies and results below come from the previous robot models. They
provide a starting point and historical references; they have not been validated
against the forthcoming v2 CAD model.

The current code includes the training stacks and the software that runs the policies on the
real robot:

| folder | what |
|---|---|
| [`controller/`](controller/README.md) | everything that runs on the robot's Raspberry Pi: the web control UI ([`fixed_gait/webui/`](controller/fixed_gait/webui/README.md)), the policy deployment stack ([`deploy/`](controller/deploy/README.md)), motor/CAN tools, identification, the hardware model |
| `RLframework/` | MJX/JAX PPO training for the flat-foot DASH-01 walker (joystick-commanded). Launch on Izar with `slurm/izar_dash.sh` |
| [`BalanceRL/`](BalanceRL/README.md) | push-recovery stander that writes the MIT force-control frame directly; a side quest off `RLframework/` |
| [`sprint_runner/`](sprint_runner/README.md) | the restored `walk_mit` stack and the 3.06 m/s runner (`sprint_m3_mit_s0`), kept so it can still be run, measured and filmed. Not deployable on the current flat-foot robot |
| `dash-01CAD/` | the current robot's CAD export (MJCF + meshes) that the training plants and the web UI's digital twin are built from |
| [`dash-01CAD-v2/`](dash-01CAD-v2/README.md) | reserved for the forthcoming CAD model; no model files have been supplied yet |

## What is deployed

The Pi runs the web UI (`controller/fixed_gait/webui/server.py`, systemd units in
`controller/fixed_gait/webui/systemd/`). Policies reach it as bundles in
`controller/deploy/bundles/`:

- `bal5_s0.npz` — the BalanceRL stander (v3 bundle).
- `dash_joy3_lr_s2_180M.npz` — the best RLframework walker.

Older bundles, the analytic in-air gait demo, the browser MJCF viewer and the URDF patch tools
were removed on 2026-09-30; they live in git history.

## Run the included policies locally

From the repository root in Windows PowerShell, install the sprint runner dependencies into
the existing project environment. MuJoCo **3.13.0** is required to reproduce this policy's
behavior; 3.14.0 caused an early fall in the local replay.

```powershell
.\.venv\Scripts\python.exe -m pip install -r sprint_runner/walk_mit/requirements.txt
.\.venv\Scripts\python.exe sprint_runner/walk_mit/evaluate.py --run sprint_runner/walk_mit/runs/sprint_m3_mit_s0 --viewer
```

The sprint runner uses the older robot model with lateral motion, roll and yaw constrained.
For the current flat-foot robot's walking policy, use the separate joystick player (no JAX
installation is needed for this player):

```powershell
.\.venv\Scripts\python.exe RLframework/tools/play_joystick.py --bundle controller/deploy/bundles/dash_joy3_lr_s2_180M.npz --start-stick 0.25
```

Focus the viewer and use **W/S** to change speed, **Space** to request a stop, and **R** to reset.
The walker is not a guaranteed stander at zero command; the balance policy is a separate bundle.

The stopping fine-tuning experiment, reward changes, evaluation protocol and SCITAS
progress are recorded in [the stopping training report](docs/stop-training-report.md).
For a concise overview, figures, videos and the current control architecture, start
with the [report workspace](report/README.md) or [visual gallery](report/index.html).
The [experiment index](docs/experiments/README.md) identifies each recorded run and
explains which artifacts are versioned.

The original SCITAS setup for `rizkalla` is archived in [docs/legacy-scitas.md](docs/legacy-scitas.md).
Current training uses `RLframework/` and its separate JAX/MJX requirements and `dash-v2`
cluster environment. The old `training/slurm/` commands no longer apply.

## Tests

```bash
python -m pytest controller/fixed_gait/webui/tests controller/deploy/tests
```

## History

Earlier stacks (`RL/`, `framework/`, `experiments/`, `orchestrator/`, `training/`, `robot/`,
`walk_v2`..`walk_v4`) were removed as each was superseded and live in git history. `walk_mit/`
survives only inside `sprint_runner/`.
