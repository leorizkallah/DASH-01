#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait11-video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=replay-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 JAX_ENABLE_X64=0
python gait_video.py \
  --run /home/rizkalla/running_robot/experiments/gait11_20261019_v1/RLframework/runs/cad_v2_gait11_s0 \
  --checkpoints best_candidate.msgpack final.msgpack --targets 0.10 0.15 0.20 --seed 1000 \
  --trace gait11_trace.npz
