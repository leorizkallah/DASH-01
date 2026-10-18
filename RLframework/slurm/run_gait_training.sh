#!/bin/bash
# Shared validation and training body for new, isolated gait experiments.
# Historical launchers remain frozen for reproducibility.
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?isolated source directory required}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl JAX_ENABLE_X64=0
mkdir -p validation
python - <<'PY'
import hashlib, json, os
from pathlib import Path
m=json.loads(Path('source-manifest.json').read_text())
assert m['preset'] == os.environ['PRESET']
for name, expected in m['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
if m.get('source_checkpoint'):
    assert hashlib.sha256(Path(m['source_checkpoint']).read_bytes()).hexdigest() == m['source_checkpoint_sha256']
print('Verified', len(m['files']), 'source hashes and parent checkpoint', flush=True)
PY
test ! -e "RLframework/runs/${RUN_NAME:?new run name required}"
python -m unittest RLframework.tests.test_bilateral_gait RLframework.tests.test_gait_selection RLframework.tests.test_walking_curriculum RLframework.tests.test_drag_curriculum RLframework.tests.test_step_sequence RLframework.tests.test_step_attempts RLframework.tests.test_heavy_repeats -q
if [[ "$PRESET" == cad_v2_gait11 ]]; then
  python -m unittest RLframework.tests.test_sequential_steps RLframework.tests.test_walltime_budget -q
fi
python RLframework/bench.py --preset "$PRESET" --n-envs 512 --steps 32 --ppo --json validation/benchmark.json
parent=$(python -c "import json; print(json.load(open('source-manifest.json')).get('source_checkpoint') or '')")
warm_args=()
if [[ -n "$parent" ]]; then warm_args=(--warm-start "$parent"); fi
time_args=()
if [[ -n "${GAIT_JOB_DEADLINE:-}" ]]; then time_args=(--deadline-epoch "$GAIT_JOB_DEADLINE"); fi
python RLframework/train.py --preset "$PRESET" --name "$RUN_NAME" --seed 0 \
  --devices 1 --eval-envs 36 "${warm_args[@]}" "${time_args[@]}" --description "${DESCRIPTION:?experiment description required}"
