#!/bin/bash -l
#SBATCH --job-name=quiet01-diag
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --exclude=i39
#SBATCH --output=quiet01-diag-%A_%a.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=0 JAX_PLATFORMS=cuda
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
cd "${DIAGNOSTICS:?isolated diagnostic folder required}"
models=(v1 v2 xshape)
model=${models[$SLURM_ARRAY_TASK_ID]}
timeout --signal=TERM --kill-after=20s 1080s python diagnose_quiet01_holds.py \
  --root /home/rizkalla/running_robot/experiments/quiet01_20261122_v1 \
  --model "$model" --out "results/$model"
