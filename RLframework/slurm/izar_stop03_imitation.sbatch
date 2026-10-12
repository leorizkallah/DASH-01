#!/bin/bash -l
#SBATCH --job-name=stop03-imitation
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
NAME="${NAME:-dash_stop03_imitation_s0}"
trap 'python RLframework/tools/stop_report.py --run "RLframework/runs/$NAME" || true' EXIT
python -m unittest discover -s RLframework/tests -v
python RLframework/train.py --preset dash_stop03_imitation --name "$NAME" --seed 0 \
    --resume auto --warm-bundle RLframework/results/stop01_replay/dash_stop01b_final.npz \
    --eval-envs 32 --devices 1 \
    --description 'STOP-03: actor-only stance and walking imitation, physical improvement and walking-retention gate, then 2M-step standing PPO; STOP-02 rewards/ramp unchanged'
