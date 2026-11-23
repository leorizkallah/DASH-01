#!/bin/bash -l
#SBATCH --job-name=quiet01
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=i39
#SBATCH --output=quiet01-%A_%a.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?isolated experiment root required}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=0 JAX_PLATFORMS=cuda
export PYTHONFAULTHANDLER=1 MUJOCO_GL=egl
models=(v1 v2 xshape)
model=${models[$SLURM_ARRAY_TASK_ID]}
python - <<'PY'
import hashlib,json
from pathlib import Path
for name,digest in json.loads(Path('source-manifest.json').read_text())['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print('Frozen source, models, references, reset banks and parents verified',flush=True)
PY
mkdir -p validation
if [[ "${MODE:-train}" == preflight ]]; then
    timeout --signal=TERM --kill-after=20s 180s python RLframework/tools/gpu_startup_probe.py --output "validation/gpu_${model}.json"
    timeout --signal=TERM --kill-after=20s 300s env JAX_PLATFORMS=cpu python -m unittest discover -s BalanceRL/tests -v
    timeout --signal=TERM --kill-after=30s 3000s python BalanceRL/tools/quiet_tracking_preflight.py \
        --model "$model" --parent "parents/$model/best" --out "validation/quiet_${model}.json"
else
    python - <<'PY'
import json
from pathlib import Path
for model in ('v1','v2','xshape'):
    receipt=json.loads(Path(f'validation/quiet_{model}.json').read_text())
    assert receipt['passed'] and receipt['matched_seed']==1234,model
    assert receipt['baseline']['protocol'].startswith('quiet-pose-v2:'),model
print('All three GPU preflight gates passed',flush=True)
PY
    python BalanceRL/train.py --preset "quiet_${model}" --name "${EXPERIMENT:?run-name prefix required}_${model}_s2" \
        --initialize-from "parents/$model/best" --seed 2 --devices 1 \
        --eval-envs 96 --walltime-hours 10.5
fi
