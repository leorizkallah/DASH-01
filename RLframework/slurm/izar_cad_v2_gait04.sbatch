#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait04
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
test -s "${SOURCE_CHECKPOINT:?gait03 final checkpoint required}"
cp "$SOURCE_CHECKPOINT" inputs/source.msgpack
python - <<'PY'
from pathlib import Path
import hashlib, json, os
p=Path('inputs/source.msgpack')
actual=hashlib.sha256(p.read_bytes()).hexdigest()
assert actual == '65cff0cda55d4b8fc547ad40585fcf7b3a77c1cb7e327826c70c56d90595f439', actual
Path('inputs/source.json').write_text(json.dumps(dict(source=os.environ['SOURCE_CHECKPOINT'],
    sha256=actual, source_step=5013504),indent=2)+'\n')
PY
python -m unittest RLframework.tests.test_bilateral_gait -q
python RLframework/bench.py --preset cad_v2_gait04 --n-envs 512 --steps 32 --ppo --json validation/benchmark.json
python RLframework/train.py --preset cad_v2_gait04 --name cad_v2_gait04_s0 --seed 0 \
  --devices 1 --eval-envs 32 --resume auto --warm-start inputs/source.msgpack \
  --description "GAIT-04: stronger swing-contact cost and dense bilateral progress; 15M additional steps at 0.35 m/s; unchanged gait acceptance"
