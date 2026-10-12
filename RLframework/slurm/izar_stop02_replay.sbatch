#!/bin/bash -l
#SBATCH --job-name=stop02-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=RLframework/results/stop02_replay/replay-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python RLframework/results/stop02_replay/trace_stop_mjx.py --repo "$REPO" \
  --run RLframework/runs/dash_stop02_stand_s0 \
  --checkpoint RLframework/runs/dash_stop02_stand_s0/ckpt_1048576.msgpack \
  --out RLframework/results/stop02_replay/standing_1048576.npz
