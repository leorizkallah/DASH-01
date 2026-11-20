#!/bin/bash -l
#SBATCH --job-name=cad-v2-clearance02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=i39
#SBATCH --output=RLframework/logs/%x-%j.out
set -euo pipefail
module purge
source "$HOME/venvs/dash-v2/bin/activate"
cd "${REPO:?isolated experiment directory required}"
deadline=$(( $(date +%s) + 86400 ))
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda PYTHONFAULTHANDLER=1
stage=startup
trap 'status=$?; echo "[batch] stage=${stage} exit=${status} at=$(date -Is)"' EXIT
run_bounded() {
    stage=$1
    local seconds=$2
    shift 2
    echo "[batch] $(date -Is) stage=${stage} timeout=${seconds}s"
    timeout --signal=TERM --kill-after=20s "${seconds}s" "$@"
}
cache_root=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/dash-clearance-${SLURM_JOB_ID}-XXXXXX")
export CUDA_CACHE_PATH="$cache_root/cuda" JAX_COMPILATION_CACHE_DIR="$cache_root/jax"
mkdir -p "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR"
echo "[batch] host=$(hostname) job=${SLURM_JOB_ID} isolated_cache=${cache_root}"
mkdir -p validation
run_bounded nvidia_smi 60 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
run_bounded gpu_backend 180 python -u RLframework/tools/gpu_startup_probe.py --output validation/gpu-startup.json
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
run_bounded unit_tests 300 env JAX_PLATFORMS=cpu python -m unittest RLframework.tests.test_swing_clearance RLframework.tests.test_clearance_eval RLframework.tests.test_flatfeet RLframework.tests.test_walk_reproduction RLframework.tests.test_walltime_budget -q
parent=$(python -c "import json; print(json.load(open('source-manifest.json'))['source_checkpoint'])")
run_bounded gpu_smoke 1200 python -u RLframework/tools/clearance_preflight.py --preset cad_v2_clearance02 --parent "$parent"
stage=training
python RLframework/train.py --preset cad_v2_clearance02 --name cad_v2_clearance02_s0 --seed 0 \
 --warm-start "$parent" --devices 1 --eval-envs 12 --deadline-epoch "$deadline" \
 --description "200M extra steps from flat-feet best 184.32M at 1.5 m/s; 10 mm mid-swing minimum, 25 mm bonus cap and explicit excess-height cost; inherited plant difficulty, conservative warm start; preserve symmetry/flat support/slip"
