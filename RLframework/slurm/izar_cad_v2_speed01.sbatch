#!/bin/bash -l
#SBATCH --job-name=cad-v2-speed01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
export GAIT_JOB_DEADLINE=$(( $(date +%s) + 18000 ))
export PRESET=cad_v2_speed01 RUN_NAME=cad_v2_speed01_s0
export DESCRIPTION="SPEED-01: fresh world-X speed discovery, no foot-order reward or alive bonus; calibrated free CAD-v2; five-hour allocation, no step cap"
source "${REPO:?isolated source directory required}/RLframework/slurm/run_gait_training.sh"
