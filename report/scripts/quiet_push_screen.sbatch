#!/bin/bash -l
#SBATCH --job-name=quiet-push-screen
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --qos=debug
#SBATCH --output=quiet-push-%A_%a.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${SCREEN:?isolated diagnostic directory required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m unittest test_quiet_push_protocol -v
models=(v1 v2 xshape)
model=${models[$SLURM_ARRAY_TASK_ID]}
timeout --signal=TERM --kill-after=20s 2580s python run_quiet_push_screen.py \
  --root /home/rizkalla/running_robot/experiments/quiet01_20261122_v2 \
  --model "$model" --out "results/$model"
