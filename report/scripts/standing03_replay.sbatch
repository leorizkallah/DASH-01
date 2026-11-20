#!/bin/bash -l
#SBATCH --job-name=stand03-video
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclude=i39
#SBATCH --output=stand03-video-%A_%a.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?frozen experiment root required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
models=(v1 v2 xshape)
model=${models[$SLURM_ARRAY_TASK_ID]}
timeout --signal=TERM --kill-after=20s 1500s python record_report_best.py \
    --root "$PWD" --model "$model" --out "report_replays_20261110/$model"
