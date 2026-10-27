#!/bin/bash -l
#SBATCH --job-name=clearance01-video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=replay-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 JAX_ENABLE_X64=0
python RLframework/tools/gait_video.py \
  --run "$PWD/RLframework/runs/cad_v2_clearance01_s0" \
  --checkpoints final.msgpack --targets 1.50 --seed 1000 \
  --trace clearance01_trace_final.npz --label CLEARANCE-01
