#!/bin/bash -l
#SBATCH --job-name=cad-v2-gait09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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
parent=Path(m['source_checkpoint'])
assert hashlib.sha256(parent.read_bytes()).hexdigest() == m['source_checkpoint_sha256'], 'parent checkpoint changed'
print('Verified', len(m['files']), 'source hashes and GAIT-08 candidate checkpoint', flush=True)
PY
test ! -e RLframework/runs/cad_v2_gait09_s0
python -m unittest RLframework.tests.test_bilateral_gait RLframework.tests.test_gait_selection RLframework.tests.test_walking_curriculum RLframework.tests.test_drag_curriculum RLframework.tests.test_step_sequence RLframework.tests.test_step_attempts -q
python RLframework/bench.py --preset cad_v2_gait09 --n-envs 512 --steps 32 --ppo --json validation/benchmark.json
python RLframework/train.py --preset cad_v2_gait09 --name cad_v2_gait09_s0 --seed 0 \
  --devices 1 --eval-envs 36 \
  --warm-start /home/rizkalla/running_robot/experiments/gait08_20261017_v1/RLframework/runs/cad_v2_gait08_s0/eval_25165824.msgpack \
  --description "GAIT-09: 32M from GAIT-08 best candidate at 0.25 m/s; penalize incomplete/repeated attempts; cap turn progress; cadence-aware stance; coherent stride history"
