"""Small GPU backend check; the Slurm launcher supplies the hard process timeout."""

import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    started = time.monotonic()
    print("[gpu-probe] importing JAX", flush=True)
    import jax
    import jax.numpy as jnp

    print("[gpu-probe] initializing CUDA backend", flush=True)
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("GPU allocation has no JAX GPU device")
    print(f"[gpu-probe] devices: {devices}", flush=True)
    with jax.default_device(devices[0]):
        value = jax.jit(lambda x: (x * x).sum())(jnp.arange(1024, dtype=jnp.float32))
        value.block_until_ready()
    if not math.isfinite(float(value)) or not float(value) > 0:
        raise RuntimeError("GPU kernel returned an invalid probe value")
    result = dict(
        passed=True,
        jax_version=jax.__version__,
        devices=[
            dict(platform=d.platform, kind=d.device_kind, id=d.id) for d in devices
        ],
        elapsed_s=time.monotonic() - started,
        job_id=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
