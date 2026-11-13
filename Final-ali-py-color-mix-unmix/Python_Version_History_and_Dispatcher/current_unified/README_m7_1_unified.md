# M7.1 corrected Python package

This package contains the M7.1 maintenance release reviewed on 7 November 2026.
The fixes are implemented in `m7_1_unified.py`. The runtime release ID is
`color-mixer-maintenance-r3-2026-11-07`.

## Contents

- `m7_1_unified.py`: corrected local prediction, inverse search and M4 route policy.
- `data/`: all eight original profile/target files, unchanged.
- `requirements.txt`: runtime dependencies (NumPy and pandas).
- `requirements-tested.txt`: exact versions used for verification.
- `test_recipe_invariants.py` and `test_core_revision.py`: offline regression tests for this release.

This follows the original single-module package layout. Run commands from the
extracted folder. Do not use `pip install .`; install the requirements instead.

## Windows quick start (Python 3.12)

In PowerShell or Command Prompt, from the extracted package folder:

```text
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m unittest test_recipe_invariants test_core_revision -v
.\.venv\Scripts\python.exe m7_1_unified.py m7-predict --pigments "CY,QM,UB" --parts "4,1,1"
.\.venv\Scripts\python.exe m7_1_unified.py m7-predict --pigments "CY,QM,UB" --parts "8,2,2"
```

All tests should pass. Both prediction commands should return `#6E622E` with
`nary_exact_observed_anchor` provenance. No API key is required. Dependency
installation needs package access; prediction and tests themselves are offline.

## Linux / macOS quick start

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt
.venv/bin/python -m unittest test_recipe_invariants test_core_revision -v
.venv/bin/python m7_1_unified.py m7-predict --pigments "CY,QM,UB" --parts "8,2,2"
```

For an existing compatible environment, use its Python and `requirements.txt`.
The recorded test environment is Python 3.12.14, NumPy 2.3.5 and pandas 2.2.3
on Linux; Windows instructions are supplied but were not executed here.

## Additional commands

Use the virtual environment's Python in place of `python` if needed:

```bash
python m7_1_unified.py m7-unmix --target-hex "#706A35" --top-n 5
python m7_1_unified.py batch-unmix --targets data/target_colors_H01_H24.csv --outdir outputs
python m7_1_unified.py diagnostics --outdir diagnostics
```

The module can also be imported by existing Python code:

```python
from pathlib import Path
import m7_1_unified as mixer

model = mixer.build_model(Path(mixer.__file__).resolve().parent / "data")
predicted_hex, metadata = model.predict_recipe(["CY", "QM", "UB"], [8, 2, 2])
```

## Corrections included

- Equivalent scaled ratios, component permutations, duplicate pigment entries
  and zero parts are handled consistently.
- Stable pair weighting corrects spurious color changes from tiny pigment amounts.
- Recipes, color values, model captures and search controls are validated.
- Capture matching and grouped diagnostics tolerate floating-point roundoff in scaled ratios.
- Batch CSV import preserves digit-only HEX strings, leading zeros in IDs and text labels.
- The M4 router accepts the historical pigment labels and consistently applies its cutoff rules.
- Output metadata distinguishes measured captures, palette endpoints and estimates.

The default 1,408-recipe grid and its ordinary predicted colors are preserved.
Invalid inputs now fail explicitly. Outputs retain the existing fields and add
release/provenance metadata. The offline `total_parts` parameter retains its
existing exact-sum meaning for multi-pigment candidates; the grid has a
1,000,000-candidate resource limit.

## Application deployment

Point the application at the updated module and its included data, then restart
the process or rebuild its model instance. If the application caches predicted
colors, use the new release ID in the cache key or clear the old predictions.
Applications that embed or translate the Python logic need the same fixes in
the code path that actually serves requests.

Before deploying, run these checks through the application's own prediction
endpoint with the M7.1 palette. Verify each pair of recipes returns the same
measured color. If the endpoint exposes provenance, both should report
`nary_exact_observed_anchor`.

| Pigments, in order | Ratios to try | Expected color |
| --- | --- | --- |
| CY, QM, UB | 4:1:1 and 8:2:2 | `#6E622E` |
| CY, UB, BK | 5:1:1 and 10:2:2 | `#3B4F2A` |

## Scope

This is the complete recovered unified M7.1 Python component, including its
required data. It is not the private GeoMagic web application or the separate
M7.2 API verifier. The `m4-route` command provides routing policy only.
The existing n-ary approximation remains; predictions near exact measured
anchors can still jump. The corrections do not establish a new general
accuracy level or complete the outstanding external validation.

Runtime SHA-256:
`c842779d7f72943d8304d65305c8ef38a852944c6c10bdce3b61ba485ebb62e3`
