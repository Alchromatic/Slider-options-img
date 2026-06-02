# M7.1 Unified Single-Python-File Package

This package answers the code-organization question directly: the current M7.1 logic is consolidated into one Python file:

```text
m7_1_unified.py
```

The measured Trycolors UI data is kept in `data/` as CSV/JSON, because those are model inputs rather than Python versions.

## What is inside the single file

- M4 route decision logic (`m4-route` command)
- M7.1 measured pairwise prediction (`m7-predict` command)
- M7.1 unmix/inverse search (`m7-unmix` and `batch-unmix` commands)
- Pair-curve diagnostics (`diagnostics` command)
- Color metrics including CIEDE2000

## How to run

Install requirements:

```bash
pip install -r requirements.txt
```

M4 route decision:

```bash
python m7_1_unified.py m4-route --pigments "CY,CR,BK" --parts "40,9,1"
```

Predict one recipe with M7.1:

```bash
python m7_1_unified.py m7-predict --pigments "CY,QM,UB" --parts "4,1,1"
```

Unmix one target:

```bash
python m7_1_unified.py m7-unmix --target-hex "#706A35" --top-n 5
```

Unmix target CSV:

```bash
python m7_1_unified.py batch-unmix --targets data/target_colors_H01_H24.csv --outdir outputs
```

Diagnostics:

```bash
python m7_1_unified.py diagnostics --outdir outputs_diagnostics
```

## Version map

- M4 = forward router, safest forward candidate.
- M7.1 = latest inverse/unmix closure candidate.
- The old M5/M6/M6.1 packages were intermediate research candidates and are no longer the recommended client-facing path.

## Note

The Trycolors API is not used by this package because tested API modes did not reproduce the Trycolors UI Pro Advanced engine.
