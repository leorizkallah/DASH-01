#!/bin/bash -l
#SBATCH --job-name=cad-v2-walk01
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
cd "${REPO:?isolated source directory required}"
deadline=$(( $(date +%s) + 86400 ))
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl JAX_ENABLE_X64=0
mkdir -p validation
python - <<'PY'
import hashlib, json
from pathlib import Path
m=json.loads(Path("source-manifest.json").read_text())
assert m["preset"] == "cad_v2_walk01" and m["initialization"] == "fresh"
assert not m.get("source_checkpoint")
for name, digest in m["files"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
print("Verified", len(m["files"]), "source hashes; fresh walking-only run", flush=True)
PY
test ! -e RLframework/runs/cad_v2_walk01_s0
python -m unittest RLframework.tests.test_walk_reproduction RLframework.tests.test_walltime_budget -q
python RLframework/bench.py --preset cad_v2_walk01 --n-envs 512 --steps 36 --ppo --json validation/benchmark.json
python RLframework/tools/walk01_preflight.py
python RLframework/train.py --preset cad_v2_walk01 --name cad_v2_walk01_s0 --seed 0 \
    --devices 1 --eval-envs 28 --deadline-epoch "$deadline" \
    --description "WALK-01: fresh 200M Nathann joystick recipe on calibrated CAD-v2 with supplied homing pose; no STOP; original walking reward/curricula; CAD action envelope and rigid-loop reset adaptations"
python RLframework/evaluate.py --run RLframework/runs/cad_v2_walk01_s0 \
    --checkpoint RLframework/runs/cad_v2_walk01_s0/final.msgpack \
    --episodes 28 --seconds 30 --json RLframework/runs/cad_v2_walk01_s0/final_nominal_eval.json
