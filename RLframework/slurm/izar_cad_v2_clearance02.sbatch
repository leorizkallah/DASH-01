#!/bin/bash -l
#SBATCH --job-name=cad-v2-clearance02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?isolated experiment directory required}"
deadline=$(( $(date +%s) + 86400 ))
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl JAX_ENABLE_X64=0
mkdir -p validation
python - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('source-manifest.json').read_text())
assert m['preset']=='cad_v2_clearance02' and m['initialization']=='warm_start'
for name,h in m['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
assert hashlib.sha256(Path(m['source_checkpoint']).read_bytes()).hexdigest()==m['source_checkpoint_sha256']
print('Verified source and parent checkpoint hashes',flush=True)
PY
test ! -e RLframework/runs/cad_v2_clearance02_s0
python -m unittest RLframework.tests.test_swing_clearance RLframework.tests.test_clearance_eval RLframework.tests.test_flatfeet RLframework.tests.test_walk_reproduction RLframework.tests.test_walltime_budget -q
parent=$(python -c "import json; print(json.load(open('source-manifest.json'))['source_checkpoint'])")
python RLframework/tools/clearance_preflight.py --preset cad_v2_clearance02 --parent "$parent"
python RLframework/train.py --preset cad_v2_clearance02 --name cad_v2_clearance02_s0 --seed 0 \
 --warm-start "$parent" --devices 1 --eval-envs 12 --deadline-epoch "$deadline" \
 --description "200M extra steps from flat-feet best 184.32M at 1.5 m/s; 10 mm mid-swing minimum, 25 mm bonus cap and explicit excess-height cost; inherited plant difficulty, conservative warm start; preserve symmetry/flat support/slip"
