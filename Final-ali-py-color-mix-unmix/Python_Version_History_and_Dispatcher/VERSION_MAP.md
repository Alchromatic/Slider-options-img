# Version Map

| Version | Type | Status | Included file |
|---|---|---|---|
| Baseline | forward/app | historical | `versions/01_frozen_baseline_app_trycolors_lab_final.py` |
| Pairwise Lab Residual | forward/model | historical | `versions/02_pairwise_lab_residual.py` |
| Dual-gate | forward/model | historical / important basis for M4 | `versions/03_dual_gate.py` |
| M4 | forward router | current forward candidate | `versions/04_m4_balanced_router.py` |
| M5 | forward/router | experimental | `versions/05_m5_candidate_router.py` |
| M6 | unmix/inverse | experimental historical | `versions/06_m6_guarded_unmix.py` |
| M6.1 | unmix/inverse | experimental historical | `versions/07_m6_1_tiered_confidence_unmix.py` |
| M7 | unmix/inverse | measured pairwise research candidate | `versions/08_m7_run_measured_pairwise_unmix.py` |
| M7.1 | unmix/inverse | current closure candidate | `current_unified/m7_1_unified.py` |

## Recommended current production variables

```python
forward_version = "m4"
unmix_version = "m7_1"
```

## API design note

Do not compare all versions under one universal function unless their input/output types match. Use two endpoint groups:

```text
forward-mix(recipe, version=...)
unmix(target_hex, version=...)
```
