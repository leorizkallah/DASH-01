#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?isolated source directory required}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl JAX_ENABLE_X64=0
mkdir -p validation
python - <<'PY'
import hashlib, json
from pathlib import Path
m=json.loads(Path('source-manifest.json').read_text())
for name, expected in m['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Verified', len(m['files']), 'source hashes', flush=True)
PY
# Refuse accidental reuse: this experiment must start without any learned state.
test ! -e RLframework/runs/cad_v2_gait05_s0
python -m unittest RLframework.tests.test_bilateral_gait RLframework.tests.test_gait_selection -q
python RLframework/bench.py --preset cad_v2_gait05 --n-envs 512 --steps 32 --ppo --json validation/benchmark.json
python RLframework/train.py --preset cad_v2_gait05 --name cad_v2_gait05_s0 --seed 0 \
  --devices 1 --eval-envs 36 \
  --description "GAIT-05: fresh policy, 15M steps at 0.25 m/s; GAIT-04 reward unchanged; three-speed evaluation and every evaluated checkpoint retained"
