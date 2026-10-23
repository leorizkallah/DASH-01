#!/bin/bash -l
#SBATCH --job-name=flatfeet-video
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
python gait_video.py \
  --run /home/rizkalla/running_robot/experiments/dash01_flatfeet_symmetry_v1/RLframework/runs/dash01_flatfeet_symmetry_v1_s0 \
  --checkpoints best.msgpack final.msgpack --targets 0.25 0.75 1.50 --seed 1000 \
  --trace flatfeet_trace.npz --label Flatfeet-v1
