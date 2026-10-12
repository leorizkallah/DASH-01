#!/bin/bash -l
#SBATCH --job-name=stop02-stand
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
NAME="${NAME:-dash_stop02_stand_s0}"
BUNDLE="${BUNDLE:-RLframework/results/stop01_replay/dash_stop01b_final.npz}"
trap 'python RLframework/tools/stop_report.py --run "RLframework/runs/$NAME" || true' EXIT
python -m unittest discover -s RLframework/tests -v
python RLframework/train.py --preset dash_stop02_stand --name "$NAME" --seed 0 \
    --resume auto --warm-bundle "$BUNDLE" --eval-envs 32 --devices 1 \
    --description 'STOP-02 standing pilot: 75% near-stance STOP resets, dense balance rewards, shared smooth braking adapter, five-second hold with drift check; 2M steps, nominal plant'
