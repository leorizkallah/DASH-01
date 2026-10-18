#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait05-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=diagnosis-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 JAX_ENABLE_X64=0
python diagnose_gait05.py \
 --run /home/rizkalla/running_robot/experiments/gait05_20261015_v1/RLframework/runs/cad_v2_gait05_s0 \
 --comparison /home/rizkalla/running_robot/experiments/gait04_20261015_v1/RLframework/runs/cad_v2_gait04_s0/ckpt_14221312.msgpack \
 --out output
