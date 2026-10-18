#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?isolated source directory required}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl JAX_ENABLE_X64=0
# Same model/physics as the validated pilot. Compare candidates on the GPU,
# then preserve the selected weights/stats with fresh optimizer and schedules.
python RLframework/tools/select_gait_checkpoint.py --source "${SOURCE_RUN:?pilot directory required}" --out inputs
python RLframework/train.py --preset cad_v2_gait02 --name cad_v2_gait02_s0 --seed 0 \
  --devices 1 --eval-envs 96 --resume auto --warm-start inputs/selected.msgpack \
  --description "GAIT-02: 30M additional walking steps; GAIT-01 rewards/model; selected pilot checkpoint; longer schedules"
