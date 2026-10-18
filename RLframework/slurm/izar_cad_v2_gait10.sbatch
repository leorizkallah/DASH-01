#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
export PRESET=cad_v2_gait10 RUN_NAME=cad_v2_gait10_s0
export DESCRIPTION="GAIT-10: last 15M continuation from GAIT-09 final; heavy protected repeated-step event and duration costs; controller phase alignment and turn-budget progress"
source "${REPO:?isolated source directory required}/RLframework/slurm/run_gait_training.sh"
