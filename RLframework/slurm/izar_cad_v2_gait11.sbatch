#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
export GAIT_JOB_DEADLINE=$(( $(date +%s) + 28800 ))
export PRESET=cad_v2_gait11 RUN_NAME=cad_v2_gait11_s0
export DESCRIPTION="GAIT-11: fresh continuous alternating walking at 0.15 m/s; no mandatory inter-step pause; eight-hour allocation with final-save reserve, no step cap"
source "${REPO:?isolated source directory required}/RLframework/slurm/run_gait_training.sh"
