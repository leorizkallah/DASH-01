#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait03
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
mkdir -p inputs validation
test -s "${SOURCE_CHECKPOINT:?best gait02 checkpoint required}"
cp "$SOURCE_CHECKPOINT" inputs/source.msgpack
python - <<'PY'
from pathlib import Path
import hashlib, json, os
p=Path('inputs/source.msgpack')
actual=hashlib.sha256(p.read_bytes()).hexdigest()
assert actual == 'bd6aad4a148c86d26d3963f417c4a20778b8dd461d035d6e18feddde096a6ab7', actual
Path('inputs/source.json').write_text(json.dumps(dict(source=os.environ['SOURCE_CHECKPOINT'],
    sha256=actual, source_step=28311552),indent=2)+'\n')
PY
python -m unittest RLframework.tests.test_bilateral_gait -q
python RLframework/bench.py --preset cad_v2_gait03 --n-envs 512 --steps 32 --ppo --json validation/benchmark.json
python RLframework/train.py --preset cad_v2_gait03 --name cad_v2_gait03_s0 --seed 0 \
  --devices 1 --eval-envs 32 --resume auto --warm-start inputs/source.msgpack \
  --description "GAIT-03: bilateral alternating steps; contact-point slip; 5M-step pilot at 0.35 m/s; gait-quality-gated keeper"
