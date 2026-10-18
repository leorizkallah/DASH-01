#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?isolated source directory required}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export JAX_ENABLE_X64=0
# Local regression suite passed before packaging. The minimal cluster bundle
# excludes historical test models, policies and the source mass-property PDF.
# v1 exposed an unstable fixed-target equilibrium, also reproduced on CPU.
# Validate matched physics; active balance is part of the fresh policy's task.
# The original five-second standing result is still recorded independently.
python RLframework/tools/cad_v2_preflight.py --gate physics --out validation/gpu-standing.json
python RLframework/bench.py --preset cad_v2_gait01 --n-envs 512 --steps 32 --ppo --json validation/benchmark.json
python RLframework/train.py --preset cad_v2_gait01 --name cad_v2_gait01_s0 --seed 0 \
  --devices 1 --eval-envs 96 --resume auto \
  --description "GAIT-01: new articulated-foot CAD, PDF inertials + 1 kg torso; fresh PPO; 0.2-0.5 m/s walking pilot"
