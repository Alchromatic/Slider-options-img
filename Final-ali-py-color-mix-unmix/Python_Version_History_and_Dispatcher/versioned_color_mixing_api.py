#!/usr/bin/env python3
"""
versioned_color_mixing_api.py

Purpose
-------
A lightweight dispatcher showing how to expose several algorithm versions via
one API-style interface.

Important distinction
---------------------
Not every historical branch has the same input/output shape:

1. Forward-mix versions:
   input  = recipe pigments + parts
   output = predicted mixed color / chosen route
   examples: baseline, dual-gate, M4, M5

2. Inverse/unmix versions:
   input  = target color
   output = proposed recipe + predicted color + confidence
   examples: M6, M6.1, M7, M7.1

Therefore, a clean API should use a version variable within the appropriate
endpoint, e.g.:

    /forward-mix?version=m4
    /unmix?version=m7_1

For historical comparison, the same normalized input can be passed to multiple
versions and their outputs can be displayed side-by-side.

This file is intentionally a dispatcher/specification layer. The production
recommended implementation is `current_unified/m7_1_unified.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Any

ForwardVersion = Literal[
    "baseline",
    "dualgate",
    "m4",
    "m5",
]

UnmixVersion = Literal[
    "m6",
    "m6_1",
    "m7",
    "m7_1",
]

CURRENT_FORWARD_VERSION = "m4"
CURRENT_UNMIX_VERSION = "m7_1"

VERSION_REGISTRY: Dict[str, Dict[str, str]] = {
    "baseline": {
        "kind": "forward",
        "status": "historical",
        "file": "versions/01_frozen_baseline_app_trycolors_lab_final.py",
        "note": "Original Streamlit-era baseline app.",
    },
    "pairwise_lab_residual": {
        "kind": "forward",
        "status": "historical",
        "file": "versions/02_pairwise_lab_residual.py",
        "note": "Early residual correction branch.",
    },
    "dualgate": {
        "kind": "forward",
        "status": "historical/important",
        "file": "versions/03_dual_gate.py",
        "note": "Dual-gate runtime sidecar used by later M4 routing.",
    },
    "m4": {
        "kind": "forward",
        "status": "current forward candidate",
        "file": "versions/04_m4_balanced_router.py",
        "note": "Safest forward-router candidate: baseline vs dualgate routing.",
    },
    "m5": {
        "kind": "forward/router",
        "status": "experimental",
        "file": "versions/05_m5_candidate_router.py",
        "note": "Small experimental forward/router improvement candidate.",
    },
    "m6": {
        "kind": "unmix",
        "status": "historical experimental",
        "file": "versions/06_m6_guarded_unmix.py",
        "note": "First guarded inverse/unmix branch.",
    },
    "m6_1": {
        "kind": "unmix",
        "status": "historical experimental",
        "file": "versions/07_m6_1_tiered_confidence_unmix.py",
        "note": "Tiered-confidence unmix candidate; later surpassed by M7/M7.1.",
    },
    "m7": {
        "kind": "unmix",
        "status": "research candidate",
        "file": "versions/08_m7_run_measured_pairwise_unmix.py",
        "note": "Measured pairwise/trust-region unmix model.",
    },
    "m7_1": {
        "kind": "unmix",
        "status": "current closure candidate",
        "file": "current_unified/m7_1_unified.py",
        "note": "Latest and strongest closure candidate; unified single-file implementation.",
    },
}


def list_versions() -> Dict[str, Dict[str, str]]:
    """Return metadata for every included version."""
    return VERSION_REGISTRY


@dataclass
class RecipeInput:
    pigment_names: List[str]
    parts: List[float]


@dataclass
class TargetInput:
    target_hex: str
    max_colors: int = 4
    total_parts: int = 6
    top_n: int = 5


def forward_mix(recipe: RecipeInput, version: ForwardVersion = "m4") -> Dict[str, Any]:
    """
    Forward mix API shape.

    In a full API service, this function would call the selected module.
    The current recommended production path is:

        version='m4'

    For M7.1 package usage, forward recipe prediction is available through:

        python current_unified/m7_1_unified.py m7-predict \
            --pigments "CY,QM,UB" --parts "4,1,1"

    Returns a normalized dictionary shape:
        {
          'version': 'm4',
          'input': {...},
          'predicted_hex': '#...',
          'metadata': {...}
        }
    """
    if version not in ("baseline", "dualgate", "m4", "m5"):
        raise ValueError(f"Unsupported forward version: {version}")
    raise NotImplementedError(
        "This dispatcher is a versioning/spec layer. Use the files in versions/ "
        "or current_unified/m7_1_unified.py for runnable commands."
    )


def unmix(target: TargetInput, version: UnmixVersion = "m7_1") -> Dict[str, Any]:
    """
    Inverse/unmix API shape.

    The current recommended production path is:

        version='m7_1'

    Runnable CLI example:

        python current_unified/m7_1_unified.py m7-unmix \
            --target-hex "#706A35" --top-n 5

    Returns a normalized dictionary shape:
        {
          'version': 'm7_1',
          'target_hex': '#...',
          'proposals': [
             {'pigment_names': [...], 'parts': [...], 'predicted_hex': '#...', 'match': ...}
          ]
        }
    """
    if version not in ("m6", "m6_1", "m7", "m7_1"):
        raise ValueError(f"Unsupported unmix version: {version}")
    raise NotImplementedError(
        "This dispatcher is a versioning/spec layer. Use the files in versions/ "
        "or current_unified/m7_1_unified.py for runnable commands."
    )


if __name__ == "__main__":
    import json
    print(json.dumps(list_versions(), indent=2))
