#!/bin/bash -l
#SBATCH --job-name=stand03
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=i39
#SBATCH --output=stand03-%A_%a.out
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
m=json.loads(Path('source-manifest.json').read_text())
for name,digest in m['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print('Verified frozen source and all model/mesh inputs',flush=True)
PY
if [[ "${MODE:-train}" == preflight ]]; then
    mkdir -p validation
    timeout --signal=TERM --kill-after=20s 180s python RLframework/tools/gpu_startup_probe.py --output "validation/gpu_${model}.json"
    timeout --signal=TERM --kill-after=20s 300s env JAX_PLATFORMS=cpu python -m unittest discover -s BalanceRL/tests -v
    timeout --signal=TERM --kill-after=20s 1500s python BalanceRL/tools/standing_preflight.py --model "$model" --out "validation/preflight_${model}.json"
else
    python - "$model" <<'PY'
import json,sys
from pathlib import Path
for model in ('v1','v2','xshape'):
    r=json.loads(Path(f'validation/preflight_{model}.json').read_text())
    assert r['passed'],model
print('All three contact/physics/PPO gates passed',flush=True)
PY
    python BalanceRL/train.py --preset "stand_${model}" --name "stand03_${model}_s0" \
        --seed 0 --devices 1 --eval-envs 80 --walltime-hours 22.5
fi
