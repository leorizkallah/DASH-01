# STOP experiment index

Start with the [report workspace](../../report/README.md) for the narrative,
architecture, figures and videos. This directory contains the small source records
behind those summaries. Existing snapshots are kept under their original names so
report links and SCITAS paths remain stable.

| Snapshot | Purpose | Record |
|---|---|---|
| `stop01_20261002_v1` | Initial STOP pilot | [Progress](stop01_20261002_v1/STOP_PROGRESS.md) |
| `stop01_20261002_v2` | STOP-01B fine-tuning; 20M steps | [Progress](stop01_20261002_v2/STOP_PROGRESS.md) |
| `stop02_20261006_v1` | Initial feasibility diagnostic; output serialization issue | [Experiment log](../stop02-experiment-report.md) |
| `stop02_20261006_v2` | Corrected ramp/stance diagnostics | [Experiment log](../stop02-experiment-report.md) |
| `stop02_20261006_v3` | Standing PPO pilot; 0/32 holds | [Status](stop02_20261006_v3/STATUS.md) |
| `stop03_20261007_v1` | Imitation-assisted standing; best 21/32 holds | [Status](stop03_20261007_v1/STATUS.md) |
| `stop04_20261012_v1` | Paired diagnosis and conditional rehearsal pilot | [Status](stop04_20261012_v1/STATUS.md) |
| `stop05_20261012_v1` | Low-speed moving stops with standing/walking preservation | [Status](stop05_20261012_v1/STATUS.md) |
| `stop06_20261012_v1` | Longer timed-versus-feedback HOLD comparison | [Status](stop06_20261012_v1/STATUS.md) |

Status files are dated observations, not a live scheduler dashboard. The remote
root is `/home/rizkalla/running_robot_experiments/<snapshot>/`.

## Artifact policy

Versioned: training/evaluation code, Slurm scripts, experiment notes, resolved
configuration, small JSON/CSV metrics, source manifests, curated plots/videos, and
the compact STOP-01B warm-start bundle required by the packager.

Local/SCITAS only: raw pose traces (`RLframework/results/stop*/**/*.npz`, except
the named warm-start policy), full training checkpoints, source/transfer archives,
intermediate checkpoint sidecars, TensorBoard event files, scratch JPG previews,
caches and logs. Best/final checkpoint metadata remains versioned.
Ignoring these files does not delete them.
The immutable SCITAS snapshots retain the source packages and run artifacts.

`report/data/` intentionally contains frozen copies of selected metrics and code,
with hashes, so the report can be shared independently. `report/videos/` provides
the curated clips. Earlier paths under `RLframework/results/` remain for existing
links; identical copies share one Git object.

## Rebuilding after cloning

The committed report gallery works without SCITAS or raw pose traces. With the
local reporting dependencies available, rebuild its figures from committed inputs:

```powershell
.\.venv\Scripts\python.exe report/scripts/build_report.py --frozen
```

The default builder refreshes inputs from the working checkout and expects the
original local `final_ramp.npz` trace. Retrieve raw traces from their SCITAS
snapshot before performing that full refresh on a fresh clone. The generator's
manifest records original paths and hashes; it does not upload results remotely.

Append new experiments here and in `report/work-log.md`. Do not rename a frozen
snapshot, overwrite earlier measurements, or stage upload archives with source code.
