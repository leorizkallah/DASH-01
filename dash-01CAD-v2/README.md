# DASH-01 v2 CAD model

Imported **14 October 2026**. Mass/inertia correction and floating-base training
integration are complete. [GAIT-01](../docs/gait01-experiment-report.md) records
validation and the fresh walking pilot. Fixed motor targets do not robustly
balance this model; walking is trained directly, with homing/standing deferred.

- `EXPORT_assembly_closed/`: original supplied export and meshes, unchanged.
- `properties/inerties_masses.pdf`: preserved source document.
- `properties/mass_properties.json`: extracted values in the PDF's original units,
  with page references, COM and origin tensors, and the PDF hash.
- `properties/body_mapping.json`: explicit component mapping and extra torso mass.
  The user confirmed that PDF frames match the export and masses include motors
  and battery. No inferred rotation/translation or extra motor masses are applied.
- `generated/dash01_v2_corrected.xml`: corrected MuJoCo CAD model.
- `generated/audit.json`: per-link checks, mass totals, findings and input hashes.
- `generated/plant_audit.json`: training-plant topology, reset and perturbation checks.
- `../RLframework/model/dash01_cad_v2_free.xml`: floating-base plant with six
  motors, articulated feet, four rigid linkage loops and mesh-derived sole contacts.

**Total: 13.594154 kg from the PDF + 1 kg torso allowance = 14.594154 kg.**
The allowance shares the torso's COM and normalized mass distribution; its tensor
is scaled accordingly. This is an explicit approximation for cables/accessories.

From the repository root:

```powershell
.\.venv\Scripts\python.exe RLframework/model/make_plant.py --cad-v2
.\.venv\Scripts\python.exe RLframework/model/make_plant.py --cad-v2 --report
.\.venv\Scripts\python.exe -m unittest RLframework.tests.test_cad_v2 -v
```

The builder uses NumPy and MuJoCo; PDF software is not required to rebuild from
the reviewed JSON. `--extra-body-kg 0` can inspect the model without the allowance.
Repeated builds always start from the original export, so the allowance cannot
accumulate. `--report` writes no generated files.

Do not run the historical `apply_measured_masses.py` on this model: its assembly
weights and body mappings describe the old robot. The old plant outputs and
policies remain unchanged. The new articulated feet/Achilles linkages are
retained in the separate training plant. Hardware homing is still uncalibrated.

See [the detailed audit and figures](../report/cad-v2-import.md) for findings,
assumptions, validation limits and next steps.
