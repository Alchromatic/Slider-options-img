# Python Version History and Versioned API Dispatcher

This package addresses the request to identify the different Python versions and to show how they can be exposed through one API-style interface.

## Important clarification

There are not six independent production versions. There are several historical / research branches. The two current client-facing components are:

- **M4**: current forward-router candidate.
- **M7.1**: latest inverse/unmix closure candidate.

Older branches are included for transparency and comparison, but they should not be treated as separate production candidates.

## Same inputs / same outputs?

Only branches of the same type can share the same input/output contract:

### Forward-mix versions

Input:

```text
recipe pigments + recipe parts
```

Output:

```text
predicted mixed color / selected route
```

Relevant versions:

```text
baseline, pairwise_lab_residual, dualgate, M4, M5
```

### Inverse/unmix versions

Input:

```text
target color
```

Output:

```text
proposed recipe + predicted color + confidence/metadata
```

Relevant versions:

```text
M6, M6.1, M7, M7.1
```

So, yes, an API can expose a `version` variable, but it should be scoped to endpoint type:

```text
/forward-mix?version=m4
/unmix?version=m7_1
```

Trying to compare a forward router and an unmix engine under one identical call would be misleading because the inputs are not the same.

## Files

- `current_unified/m7_1_unified.py`  
  Current consolidated Python file for M4/M7.1 functionality.

- `versioned_color_mixing_api.py`  
  Lightweight dispatcher/specification showing how version variables should be structured.

- `versions/`  
  Selected historical Python files, named by branch order.

- `data/`  
  Measured Trycolors UI data required by the current M7.1 model. These are model inputs, not separate code versions.

## Recommended production naming

```text
forward_version = "m4"
unmix_version   = "m7_1"
```

## Suggested API response fields

A clean service response could return:

```json
{
  "version": "m7_1",
  "target_hex": "#706A35",
  "recipe": {
    "pigments": ["Cadmium Yellow Light", "Quinacridone Magenta", "Ultramarine Blue"],
    "parts": [4, 1, 1]
  },
  "predicted_hex": "#6E622E",
  "confidence": "measured_anchor_or_high_confidence",
  "notes": "M7.1 closure candidate"
}
```

## Current recommendation

Use `m7_1_unified.py` as the primary code file. Use the historical files only if the goal is to compare outputs by version.
