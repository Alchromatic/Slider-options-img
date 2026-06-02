# Version Map

## Current client-facing code

- `m7_1_unified.py` is the single consolidated Python file.

## Branch meanings

- **M4**: forward-routing policy. It decides whether a recipe should be evaluated by the baseline or dual-gate candidate.
- **M7.1**: latest inverse/unmix closure candidate. It uses measured Trycolors UI pair curves plus exact n-ary anchors.

## Older branches

- M3b, M4.1, M5, M6, M6.1 were intermediate research/validation branches.
- They remain useful historically but should not be treated as separate production Python versions.

## Data files

The CSV/JSON files in `data/` are required measured Trycolors UI inputs, not separate algorithm versions.
