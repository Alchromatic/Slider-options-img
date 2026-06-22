#!/usr/bin/env python3
"""
version_dispatch.py
===================

A single dispatcher that exposes every color-mixing / unmixing *version* behind
one programmatic interface, so the front-end can switch models from a dropdown
and compare how their outputs differ.

It follows the API shape described in the closure-package readme:

    forward_mix(recipe, version="m4")    -> predicted mixed color
    unmix(target_hex, version="m7_1")    -> proposed recipe(s)

Forward-mix versions  : baseline, dualgate, m4, m5, m7_1
Unmix / inverse versions: m6, m6_1, m7, m7_1   (km_baseline is served by the
                          existing /unmix endpoint and is listed here only for
                          discoverability)

Only the runnable versions are wired to live logic.  Historical research
branches whose source depends on modules that were not shipped in this package
(m6, m6_1) are reported as ``available: False`` with an explanation rather than
silently faked.

This module loads the version sources straight from the
``Final-ali-py-color-mix-unmix`` package by file path, so nothing has to be
copied or renamed.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "Final-ali-py-color-mix-unmix"
M71_PKG = PKG / "M7_1_Unified_Single_Py_Package"
M71_DATA = M71_PKG / "data"
VERSIONS_DIR = PKG / "Python_Version_History_and_Dispatcher" / "versions"


# ---------------------------------------------------------------------------
# Module loading helpers (load by absolute file path, optionally aliasing the
# module name so intra-package "from x import y" statements resolve).
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def _m71():
    """The unified M7.1 module (fully self-contained)."""
    return _load_module("m7_1_unified", M71_PKG / "m7_1_unified.py")


@lru_cache(maxsize=1)
def _m71_model():
    """Measured-pairwise model built from the shipped Trycolors UI data."""
    return _m71().MeasuredPairwiseModel.from_data_dir(M71_DATA)


@lru_cache(maxsize=1)
def _m4_router():
    return _load_module("v_m4_router", VERSIONS_DIR / "04_m4_balanced_router.py")


@lru_cache(maxsize=1)
def _m5_router():
    return _load_module("v_m5_router", VERSIONS_DIR / "05_m5_candidate_router.py")


@lru_cache(maxsize=1)
def _km_mixer():
    """Self-contained Kubelka-Munk / linear mixer from the M4 router file."""
    return _load_module("v_m4_balanced", ROOT / "m4_balanced_router.py")


@lru_cache(maxsize=1)
def _m7_model():
    """
    The M7 (pre-M7.1) measured-pairwise model.  Its source uses bare imports
    (``from color_metrics import ...``) so we register the dependency modules
    under the names it expects before loading it.
    """
    _load_module("color_metrics", VERSIONS_DIR / "08_m7_color_metrics.py")
    mpm = _load_module("measured_pairwise_model", VERSIONS_DIR / "08_m7_measured_pairwise_model.py")
    # M7 used three measured CSVs (no H24 n-ary rescue capture) -> different output.
    model = mpm.MeasuredPairwiseModel.from_csvs(
        M71_DATA / "palette_v1.json",
        M71_DATA / "trycolors_ui_pairwise_50_50_matrix_P01_P28.csv",
        M71_DATA / "trycolors_ui_phase4_fragile_pair_curves_C01_C16.csv",
        M71_DATA / "trycolors_ui_additional_binary_captures.csv",
    )
    return mpm, model


# ---------------------------------------------------------------------------
# Version registry
# ---------------------------------------------------------------------------

VERSION_REGISTRY: List[Dict[str, Any]] = [
    # forward-mix versions
    {"id": "baseline", "mode": "forward", "label": "Baseline (KM)",
     "status": "historical", "available": True,
     "note": "Frozen baseline: Kubelka-Munk subtractive mix of the measured palette."},
    {"id": "dualgate", "mode": "forward", "label": "Dual-gate (linear)",
     "status": "historical / basis for M4", "available": True,
     "note": "Dual-gate sidecar stand-in: linear-light mix of the measured palette."},
    {"id": "m4", "mode": "forward", "label": "M4 router (current)",
     "status": "current forward candidate", "available": True,
     "note": "Routes baseline vs dual-gate, then predicts with the chosen mixer."},
    {"id": "m5", "mode": "forward", "label": "M5 router (experimental)",
     "status": "experimental", "available": True,
     "note": "Experimental router (adds a P3 branch); predicts with the chosen mixer."},
    {"id": "m7_1", "mode": "forward", "label": "M7.1 measured predict",
     "status": "current closure candidate", "available": True,
     "note": "Measured-pairwise forward prediction (also the recommended unmix model)."},

    # unmix / inverse versions
    {"id": "km_baseline", "mode": "unmix", "label": "KM search (your palette)",
     "status": "app baseline", "available": True, "endpoint": "/unmix",
     "note": "The existing Kubelka-Munk recipe search; uses the palette you load."},
    {"id": "m6", "mode": "unmix", "label": "M6 guarded unmix",
     "status": "historical experimental", "available": False,
     "reason": "Source imports scripts/unmix_targets_m4_m5, which was not shipped in this package."},
    {"id": "m6_1", "mode": "unmix", "label": "M6.1 tiered-confidence unmix",
     "status": "historical experimental", "available": False,
     "reason": "Source imports miniapp_pairwise_lab_residual / miniapp_dual_residual_gate, not shipped in this package."},
    {"id": "m7", "mode": "unmix", "label": "M7 measured pairwise",
     "status": "research candidate", "available": True,
     "note": "Measured-pairwise unmix over the fixed 8-pigment palette (no H24 n-ary rescue anchors)."},
    {"id": "m7_1", "mode": "unmix", "label": "M7.1 measured pairwise (current)",
     "status": "current closure candidate", "available": True,
     "note": "Measured-pairwise unmix over the fixed 8-pigment palette, incl. observed n-ary anchors."},
    {"id": "m7_2", "mode": "unmix", "label": "M7.2 measured profile (custom palette)",
     "status": "current closure candidate", "available": True, "endpoint": "/unmix/custom",
     "note": "Same M7.1 algorithm over your own Color Library, driven by a measured palette "
             "profile (TryColors pro/2025) generated offline; falls back to the KM physical "
             "model for any recipe the profile does not cover. Served by POST /unmix/custom "
             "(pass palette_id); profiles are built by profile_generator.py / the admin endpoint."},
]


def list_versions() -> Dict[str, Any]:
    return {
        "current_forward_version": "m4",
        "current_unmix_version": "m7_1",
        "forward": [v for v in VERSION_REGISTRY if v["mode"] == "forward"],
        "unmix": [v for v in VERSION_REGISTRY if v["mode"] == "unmix"],
    }


def _registry_entry(version: str, mode: str) -> Dict[str, Any]:
    for v in VERSION_REGISTRY:
        if v["id"] == version and v["mode"] == mode:
            return v
    raise KeyError(f"Unknown {mode} version: {version}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _palette_hex_map() -> Dict[str, str]:
    """Full pigment name -> measured hex (from palette_v1.json)."""
    return dict(_m71_model().palette)


def _names_to_hexes(names: Sequence[str]) -> List[str]:
    pmap = _palette_hex_map()
    return [pmap.get(n, "#000000") for n in names]


def _abbr_for(names: Sequence[str]) -> List[str]:
    n2a = _m71().NAME_TO_ABBR
    return [n2a.get(n, n) for n in names]


# ---------------------------------------------------------------------------
# Forward mix
# ---------------------------------------------------------------------------

def forward_mix(pigments: Sequence[str], parts: Sequence[float], version: str = "m4") -> Dict[str, Any]:
    """
    Predict the mixed color for a recipe (pigment names/abbreviations + parts)
    using the requested forward version.
    """
    entry = _registry_entry(version, "forward")
    if not entry.get("available", False):
        return {"version": version, "mode": "forward", "available": False,
                "reason": entry.get("reason", "Version unavailable in this package.")}

    m71 = _m71()
    names = [m71.expand_pigment_name(x) for x in pigments]
    names, parts2 = m71.canonicalize_recipe(names, [float(p) for p in parts])
    weights = m71.normalize(parts2)

    out: Dict[str, Any] = {
        "version": version, "mode": "forward", "available": True,
        "pigment_names": names, "pigment_abbr": _abbr_for(names),
        "pigment_hexes": _names_to_hexes(names),
        "parts": parts2, "normalized_weights": weights,
        "note": entry.get("note", ""),
    }

    if version == "m7_1":
        pred, meta = _m71_model().predict_recipe(names, parts2)
        out.update({"predicted_hex": pred, "confidence_tier": meta.get("confidence_tier"),
                    "meta": meta})
        return out

    # baseline / dualgate / m4 / m5 -> physical mix of measured palette hexes
    km = _km_mixer()
    hexes = _names_to_hexes(names)

    def _mix(method: str) -> str:
        return km._mix_hex(hexes, list(weights), method=method)

    chosen_model = None
    route_reason = None
    if version == "baseline":
        method = "kubelka_munk"
    elif version == "dualgate":
        method = "linear"
    elif version == "m4":
        chosen_model, route_reason = _m4_router().choose_model(names, weights)
        method = "kubelka_munk" if chosen_model == "baseline" else "linear"
    else:  # m5
        chosen_model, route_reason = _m5_router().choose_model(names, weights)
        # p3 is an experimental correction branch; use KM as its physical stand-in.
        method = "linear" if chosen_model == "dualgate" else "kubelka_munk"

    predicted_hex = _mix(method)
    out.update({
        "predicted_hex": predicted_hex,
        "mix_method": method,
        "chosen_model": chosen_model,
        "route_reason": route_reason,
    })
    return out


# ---------------------------------------------------------------------------
# Unmix
# ---------------------------------------------------------------------------

def _proposal_row(m71, names, parts, pred_hex, target_hex, meta, score, penalty) -> Dict[str, Any]:
    weights = m71.normalize(parts)
    d = m71.de00_hex(pred_hex, target_hex)
    return {
        "pigment_names": list(names),
        "pigment_abbr": _abbr_for(names),
        "pigment_hexes": _names_to_hexes(names),
        "parts": list(parts),
        "percentages": [round(w * 100, 1) for w in weights],
        "predicted_hex": pred_hex,
        "delta_e": round(d, 2),
        "match_percentage": round(m71.match_pct(d), 1),
        "score_with_risk_penalty": round(score, 3),
        "risk_penalty": round(penalty, 2),
        "confidence_tier": meta.get("confidence_tier"),
        "anchor_trycolors_name": meta.get("anchor_trycolors_name", ""),
    }


def unmix(target_hex: str, version: str = "m7_1", max_colors: int = 4,
          total_parts: int = 6, top_n: int = 5) -> Dict[str, Any]:
    """
    Propose recipes for a target color using the requested unmix version.
    Measured models (m7/m7_1) use the fixed measured 8-pigment palette.
    """
    entry = _registry_entry(version, "unmix")
    if not entry.get("available", False):
        return {"version": version, "mode": "unmix", "available": False,
                "target_color": target_hex,
                "reason": entry.get("reason", "Version unavailable in this package.")}

    if version == "km_baseline":
        return {"version": version, "mode": "unmix", "available": True,
                "target_color": target_hex, "delegated_endpoint": "/unmix",
                "note": entry.get("note", ""),
                "proposals": []}

    m71 = _m71()
    target_hex = m71.clean_hex(target_hex)
    max_colors = max(1, min(4, int(max_colors)))
    total_parts = max(2, min(12, int(total_parts)))
    top_n = max(1, min(20, int(top_n)))

    proposals: List[Dict[str, Any]] = []

    if version == "m7_1":
        df = m71.m7_unmix(_m71_model(), target_hex, max_colors=max_colors,
                          total_parts=total_parts, top_n=top_n)
        import json as _json
        for _, r in df.iterrows():
            names = _json.loads(r["pigment_names"])
            parts = _json.loads(r["parts"])
            weights = m71.normalize(parts)
            proposals.append({
                "rank": int(r["rank"]),
                "pigment_names": names,
                "pigment_abbr": _abbr_for(names),
                "pigment_hexes": _names_to_hexes(names),
                "parts": parts,
                "percentages": [round(w * 100, 1) for w in weights],
                "predicted_hex": r["predicted_hex"],
                "delta_e": round(float(r["predicted_dE00"]), 2),
                "match_percentage": round(float(r["predicted_match_pct"]), 1),
                "score_with_risk_penalty": round(float(r["score_with_risk_penalty"]), 3),
                "risk_penalty": round(float(r["risk_penalty"]), 2),
                "confidence_tier": r["confidence_tier"],
                "anchor_trycolors_name": r.get("anchor_trycolors_name", ""),
            })
    elif version == "m7":
        mpm, model = _m7_model()
        PALETTE_ORDER = m71.PALETTE_ORDER
        scored = []
        for k in range(1, max_colors + 1):
            for combo in itertools.combinations(PALETTE_ORDER, k):
                parts_iter = [[1]] if k == 1 else m71.compositions(total_parts, k)
                for parts in parts_iter:
                    names, parts2 = model.canonicalize_recipe(combo, parts) \
                        if hasattr(model, "canonicalize_recipe") else mpm.canonicalize_recipe(combo, parts)
                    pred, meta = model.predict_recipe(names, parts2)
                    d = m71.de00_hex(pred, target_hex)
                    penalty = _m7_risk_penalty(mpm, meta)
                    scored.append((d + penalty, d, penalty, names, parts2, pred, meta))
        scored.sort(key=lambda x: x[0])
        for rank, (score, d, penalty, names, parts2, pred, meta) in enumerate(scored[:top_n], start=1):
            row = _proposal_row(m71, names, parts2, pred, target_hex, meta, score, penalty)
            row["rank"] = rank
            proposals.append(row)

    return {
        "version": version, "mode": "unmix", "available": True,
        "target_color": target_hex,
        "palette_mode": "fixed_measured_8",
        "palette": [{"name": n, "hex": h} for n, h in _palette_hex_map().items()],
        "note": entry.get("note", ""),
        "proposals": proposals,
    }


def _m7_risk_penalty(mpm, meta: dict) -> float:
    """M7's risk penalty (predates the n-ary-anchor zero tier added in M7.1)."""
    tier = meta.get("confidence_tier")
    if tier in ("single_anchor", "pair_exact_observed", "pair_dense_interpolation"):
        return 0.0
    if tier == "pair_medium_interpolation":
        return 1.0
    if tier == "pair_sparse_interpolation":
        return 2.0
    if tier == "nary_dense_pairwise_composition":
        return 1.5
    if tier == "nary_medium_pairwise_composition":
        return 2.5
    return 4.0


if __name__ == "__main__":
    import json
    print(json.dumps(list_versions(), indent=2))
    print("\n-- forward m4 --")
    print(json.dumps(forward_mix(["CY", "QM", "UB"], [4, 1, 1], "m4"), indent=2))
    print("\n-- unmix m7_1 --")
    print(json.dumps(unmix("#706A35", "m7_1", top_n=3), indent=2))
