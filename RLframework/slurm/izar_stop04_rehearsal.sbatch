#!/bin/bash -l
#SBATCH --job-name=stop04-rehearsal
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
NAME=dash_stop04_rehearsal_s0
SOURCE=inputs/stop03
trap 'if test -d "RLframework/runs/$NAME"; then python RLframework/tools/stop_report.py --run "RLframework/runs/$NAME" || true; fi' EXIT
python -m unittest discover -s RLframework/tests -v
sha256sum "$SOURCE/best_stop.msgpack" "$SOURCE/final.msgpack" "$SOURCE/imitation_demonstrations.npz" > inputs/source-sha256.txt
python RLframework/tools/stop04_pipeline.py diagnose --source-run "$SOURCE"
set +e
python RLframework/train.py --preset dash_stop04_rehearsal --name "$NAME" --seed 0 \
  --warm-start "$SOURCE/best_stop.msgpack" --rehearsal-data "$SOURCE/imitation_demonstrations.npz" \
  --resume auto --eval-envs 32 --devices 1 \
  --description 'STOP-04: continue from STOP-03 524k best hold keeper with joint PPO + standing/walking imitation; 1M-step pilot, unchanged reward/ramp, preservation guard'
TRAIN_EXIT=$?
set -e
if test -f "RLframework/runs/$NAME/best_stop.msgpack"; then
  python RLframework/tools/stop04_pipeline.py assess --source-run "$SOURCE" --run "RLframework/runs/$NAME"
fi
exit "$TRAIN_EXIT"
