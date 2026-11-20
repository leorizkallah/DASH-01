#!/bin/bash -l
#SBATCH --job-name=clearance02-video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --exclude=i39
#SBATCH --output=clearance02-replay-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 JAX_ENABLE_X64=0 JAX_PLATFORMS=cuda
sha256sum RLframework/runs/cad_v2_clearance02_s0/best_clearance_candidate.msgpack
timeout --signal=TERM --kill-after=20s 900s python clearance02-replay-recorder.py \
  --run "$PWD/RLframework/runs/cad_v2_clearance02_s0" \
  --checkpoints best_clearance_candidate.msgpack \
  --targets 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 \
  --seed 1000 --keep-reset-command \
  --trace clearance02_trace_best_eval12.npz --label CLEARANCE-02
