#!/bin/bash -l
#SBATCH --job-name=stop06-compare
#SBATCH --array=0-1%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=RLframework/logs/%x-%A_%a.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
ARMS=(timed feedback)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
NAME=dash_stop06_${ARM}_s0
RUN=RLframework/runs/$NAME
trap 'if test -d "$RUN"; then python RLframework/tools/stop_report.py --run "$RUN" || true; python RLframework/tools/stop06_report.py --run "$RUN" || true; fi' EXIT
python -m unittest discover -s RLframework/tests -q
sha256sum inputs/stop04/best_stop.msgpack inputs/imitation_demonstrations.npz > inputs/source-sha256-${ARM}.txt
set +e
python RLframework/train.py --preset dash_stop06_${ARM} --name "$NAME" --seed 0 \
  --warm-start inputs/stop04/best_stop.msgpack --rehearsal-data inputs/imitation_demonstrations.npz \
  --resume auto --eval-envs 32 --devices 1 \
  --description "STOP-06 ${ARM}: matched 5M moving-stop comparison from STOP-04 keeper; bounded settling, rehearsal, persistent regression guard"
TRAIN_EXIT=$?
set -e
python RLframework/tools/stop_report.py --run "$RUN"
python RLframework/tools/stop06_report.py --run "$RUN"
if test -f "$RUN/best_stop.msgpack"; then
  python RLframework/tools/stop05_assess.py --run "$RUN"
fi
exit "$TRAIN_EXIT"
