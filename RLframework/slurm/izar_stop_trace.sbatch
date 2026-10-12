#!/bin/bash -l
# Read-only policy analysis: no optimizer updates or training checkpoint writes.
#SBATCH --job-name=stop-trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=45G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=RLframework/results/stop01_replay/trace-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?set the source snapshot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python RLframework/results/stop01_replay/trace_stop_mjx.py --repo "$REPO" \
  --run RLframework/runs/dash_stop01b_s0 \
  --checkpoint RLframework/runs/dash_stop01b_s0/final.msgpack \
  --out RLframework/results/stop01_replay/final_stop_trace.npz
