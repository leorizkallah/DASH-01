#!/bin/bash -l
#SBATCH --job-name=stop05-motion
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
NAME=dash_stop05_motion_s0
SOURCE=inputs/stop04
trap 'if test -d "RLframework/runs/$NAME"; then python RLframework/tools/stop_report.py --run "RLframework/runs/$NAME" || true; fi' EXIT
python -m unittest discover -s RLframework/tests -v
sha256sum "$SOURCE/best_stop.msgpack" inputs/imitation_demonstrations.npz > inputs/source-sha256.txt
set +e
python RLframework/train.py --preset dash_stop05_motion --name "$NAME" --seed 0 \
  --warm-start "$SOURCE/best_stop.msgpack" --rehearsal-data inputs/imitation_demonstrations.npz \
  --resume auto --eval-envs 32 --devices 1 \
  --description 'STOP-05 low-speed 0.25-0.5 m/s moving-stop pilot from STOP-04 262k keeper; alternating commands, preserved imitation, separate standing/walking guards, 1M new steps'
TRAIN_EXIT=$?
set -e
if test -f "RLframework/runs/$NAME/best_stop.msgpack"; then
  python RLframework/tools/stop05_assess.py --run "RLframework/runs/$NAME"
fi
exit "$TRAIN_EXIT"
