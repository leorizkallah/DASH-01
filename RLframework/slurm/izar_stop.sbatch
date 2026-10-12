#!/bin/bash -l
# STOP-01 pilot. Submit from an immutable experiment snapshot.
#SBATCH --job-name=dash-stop01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=04:30:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
module purge
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"
cd "${REPO:?set REPO to the experiment snapshot}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export JAX_ENABLE_X64=0
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
NAME="${NAME:-dash_stop01b_s0}"
BUNDLE=controller/deploy/bundles/dash_joy3_lr_s2_180M.npz
mkdir -p "RLframework/runs/$NAME"
trap 'python RLframework/tools/stop_report.py --run "RLframework/runs/$NAME" || true' EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c 'import jax; print("JAX", jax.__version__, jax.devices()); assert jax.default_backend() == "gpu"'
python -m unittest discover -s RLframework/tests -v
python RLframework/tools/check_stop_setup.py --bundle "$BUNDLE"
python RLframework/train.py --preset dash_stop --name "$NAME" --seed 0 \
    --resume auto --warm-bundle "$BUNDLE" --eval-envs 32 --devices 1 \
    --description 'STOP-01: bundle-initialized actor, new critic, supported hold reward and longer STOP intervals; nominal-plant pilot'
