#!/bin/bash -l
#SBATCH --job-name=stop02-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m unittest discover -s RLframework/tests -v
SOURCE=controller/deploy/bundles/dash_joy3_lr_s2_180M.npz
FINAL=RLframework/results/stop01_replay/dash_stop01b_final.npz
OUT=RLframework/results/stop02_checks
python RLframework/tools/check_stop02.py --bundle "$FINAL" --ramp --standing --dense --zero-action --out "$OUT/stance_probe.npz"
for POLICY in source final; do
    BUNDLE="$SOURCE"
    if [ "$POLICY" = final ]; then BUNDLE="$FINAL"; fi
    python RLframework/tools/check_stop02.py --bundle "$BUNDLE" --out "$OUT/${POLICY}_abrupt.npz"
    python RLframework/tools/check_stop02.py --bundle "$BUNDLE" --ramp --out "$OUT/${POLICY}_ramp.npz"
done
python RLframework/tools/check_stop02.py --bundle "$FINAL" --ramp --standing --dense --out "$OUT/final_standing.npz"
