#!/usr/bin/env python3
"""
measured_profile.py
===================

The **measured palette profile** -- the saved output of running a palette's
comparison recipes through TryColors ``pro``/``2025`` (see Color-engine.docx).

The doc review made the profile shape mandatory (rule R3): every profile carries
a palette ID, the color list, the engine/mode it was measured with, the generated
comparison recipes, the returned TryColors hexes, a timestamp, a profile version,
and a completeness block listing which comparisons are still missing.  The
completeness block is also what makes generation *incremental* (rule R4): adding a
new color only generates the comparisons that involve it.

This module also provides :class:`MeasuredProfileModel`, the profile-driven
predictor.  It is the same M7.1/M7.2 algorithm -- measured pairwise curves
interpolated in OKLab, composed pairwise for n-ary recipes -- but built from a
*custom palette's* profile instead of the fixed 8-pigment data set.  That is the
"measured mixing behavior" the doc describes (rule R1): same algorithm + a new
measured palette profile = palette-specific mix/unmix behavior.

Nothing here calls TryColors or reads the API key; that is
``trycolors_client.py``.  The app loads a saved profile and uses it offline
(rule R5).
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
_PROFILES_DIR = _ROOT / "profiles"
_M71_PATH = _ROOT / "Final-ali-py-color-mix-unmix" / "M7_1_Unified_Single_Py_Package" / "m7_1_unified.py"

SCHEMA_VERSION = 1
DEFAULT_ENGINE = "2025"
DEFAULT_MIXER_MODE = "pro"
# Pairwise ratios captured per color pair.  (1,1)=midpoint, (1,3)/(3,1)=quarter
# points -> a 3-point curve plus the two pure endpoints.
DEFAULT_RATIOS: List[Tuple[int, int]] = [(1, 1), (1, 3), (3, 1)]


# ---------------------------------------------------------------------------
# Reuse M7.1's color science so the measured math matches the shipped models.
# ---------------------------------------------------------------------------

def _m71():
    mod = sys.modules.get("m7_1_unified")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("m7_1_unified", str(_M71_PATH))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load m7_1_unified from {_M71_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m7_1_unified"] = mod
    spec.loader.exec_module(mod)
    return mod


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Palette identity + comparison helpers
# ---------------------------------------------------------------------------

def normalize_hex(h: str) -> str:
    return _m71().clean_hex(h)


def assign_color_ids(colors: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return [{id,name,hex}] with deterministic ids (c1..cN) by sorted hex.

    Dedupes by hex (first name wins).  Stable ids mean the same palette always
    produces the same comparison keys, so incremental updates line up.
    """
    seen: Dict[str, str] = {}
    for c in colors:
        hx = normalize_hex(c["hex"] if isinstance(c, dict) else c)
        name = (c.get("name") if isinstance(c, dict) else None) or hx
        seen.setdefault(hx, str(name))
    ordered = sorted(seen.items(), key=lambda kv: kv[0])
    return [{"id": f"c{i + 1}", "name": name, "hex": hx} for i, (hx, name) in enumerate(ordered)]


def palette_id_for(colors: Sequence[Dict[str, str]]) -> str:
    """Stable id from the sorted set of hexes (order/name independent)."""
    hexes = sorted({normalize_hex(c["hex"] if isinstance(c, dict) else c) for c in colors})
    digest = hashlib.sha1("|".join(hexes).encode("utf-8")).hexdigest()[:12]
    return f"pal_{digest}"


def _canon(ids: Sequence[str], parts: Sequence[float]) -> Tuple[List[str], List[int]]:
    items = sorted(zip([str(i) for i in ids], [float(p) for p in parts]), key=lambda x: x[0])
    cids = [i for i, _ in items]
    cparts = [int(round(p)) if abs(p - round(p)) < 1e-9 else float(p) for _, p in items]
    return cids, cparts


def comparison_key(ids: Sequence[str], parts: Sequence[float]) -> str:
    cids, cparts = _canon(ids, parts)
    return json.dumps({"ids": cids, "parts": cparts}, sort_keys=True)


# ---------------------------------------------------------------------------
# Profile schema (R3) + store
# ---------------------------------------------------------------------------

def _next_id_num(existing_cols: Sequence[Dict[str, str]]) -> int:
    nums = [int(c["id"][1:]) for c in existing_cols if str(c["id"]).startswith("c") and c["id"][1:].isdigit()]
    return max(nums, default=0) + 1


def merge_colors(existing_cols: Sequence[Dict[str, str]],
                 new_colors: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[str]]:
    """Merge new colors into an existing color list, preserving existing ids.

    Existing hexes keep their id (so prior comparison keys stay valid); genuinely
    new hexes get the next sequential id.  This is what makes adding a color an
    incremental update (rule R4) instead of a full re-id.
    """
    merged = [dict(c) for c in existing_cols]
    seen = {c["hex"] for c in merged}
    nid = _next_id_num(merged)
    added: List[str] = []
    for c in new_colors:
        hx = normalize_hex(c["hex"] if isinstance(c, dict) else c)
        name = (c.get("name") if isinstance(c, dict) else None) or hx
        if hx in seen:
            continue
        seen.add(hx)
        cid = f"c{nid}"
        nid += 1
        merged.append({"id": cid, "name": str(name), "hex": hx})
        added.append(cid)
    return merged, added


def update_profile_colors(profile: Dict, colors: Sequence[Dict[str, str]]) -> Tuple[Dict, List[str]]:
    merged, added = merge_colors(profile["colors"], colors)
    profile["colors"] = merged
    return profile, added


def new_profile(colors: Sequence[Dict[str, str]], *, palette_id: Optional[str] = None,
                palette_name: str = "", engine: str = DEFAULT_ENGINE,
                mixer_mode: str = DEFAULT_MIXER_MODE) -> Dict:
    cols = assign_color_ids(colors)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_version": 1,
        "palette_id": palette_id or palette_id_for(cols),
        "palette_name": palette_name,
        "engine": engine,
        "mixer_mode": mixer_mode,
        "source": None,            # set per comparison; summarized on save
        "generated_at": _now_iso(),
        "updated_at": _now_iso(),
        "colors": cols,
        "comparisons": [],
        "completeness": {"expected_count": 0, "present_count": 0, "missing": [], "complete": False},
    }


def profile_path(palette_id: str, profiles_dir: Optional[Path] = None) -> Path:
    return (Path(profiles_dir) if profiles_dir else _PROFILES_DIR) / f"{palette_id}.json"


def load_profile(palette_id: str, profiles_dir: Optional[Path] = None) -> Optional[Dict]:
    p = profile_path(palette_id, profiles_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_profile(profile: Dict, profiles_dir: Optional[Path] = None) -> Path:
    p = profile_path(profile["palette_id"], profiles_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    sources = {c.get("source") for c in profile.get("comparisons", []) if c.get("source")}
    profile["source"] = (sources.pop() if len(sources) == 1 else
                         ("mixed" if sources else None))
    profile["updated_at"] = _now_iso()
    p.write_text(json.dumps(profile, indent=2))
    return p


def bump_version(profile: Dict) -> Dict:
    profile["profile_version"] = int(profile.get("profile_version", 1)) + 1
    return profile


# ---------------------------------------------------------------------------
# Planning + incremental update (R4)
# ---------------------------------------------------------------------------

def expected_pairwise_comparisons(profile: Dict,
                                  ratios: Sequence[Tuple[int, int]] = DEFAULT_RATIOS) -> List[Dict]:
    """All pairwise comparison specs for the palette: every pair x every ratio."""
    ids = [c["id"] for c in profile["colors"]]
    out: List[Dict] = []
    for a, b in itertools.combinations(ids, 2):
        for pa, pb in ratios:
            cids, cparts = _canon([a, b], [pa, pb])
            out.append({"pigments": cids, "parts": cparts, "key": comparison_key(cids, cparts)})
    # de-dup (canonicalizing (1,3)/(3,1) keeps both because ids order differs)
    uniq: Dict[str, Dict] = {c["key"]: c for c in out}
    return list(uniq.values())


def existing_keys(profile: Dict) -> set:
    return {comparison_key(c["pigments"], c["parts"]) for c in profile.get("comparisons", [])}


def diff_missing(profile: Dict, expected: Optional[Sequence[Dict]] = None,
                 ratios: Sequence[Tuple[int, int]] = DEFAULT_RATIOS) -> List[Dict]:
    """Expected comparisons not yet present in the profile (R4)."""
    exp = list(expected) if expected is not None else expected_pairwise_comparisons(profile, ratios)
    have = existing_keys(profile)
    return [c for c in exp if c.get("key", comparison_key(c["pigments"], c["parts"])) not in have]


def add_results(profile: Dict, results: Sequence[Dict]) -> Dict:
    """Append measured comparison results (idempotent by canonical key).

    Each result: {pigments, parts, trycolors_result_hex, source, [trycolors_name]}.
    """
    have = existing_keys(profile)
    for r in results:
        cids, cparts = _canon(r["pigments"], r["parts"])
        key = comparison_key(cids, cparts)
        if key in have:
            continue
        profile["comparisons"].append({
            "pigments": cids,
            "parts": cparts,
            "trycolors_result_hex": normalize_hex(r["trycolors_result_hex"]),
            "trycolors_name": r.get("trycolors_name"),
            "engine": r.get("engine", profile.get("engine")),
            "mixer_mode": r.get("mixer_mode", profile.get("mixer_mode")),
            "source": r.get("source", "trycolors_api"),
            "generated_at": _now_iso(),
        })
        have.add(key)
    return profile


def recompute_completeness(profile: Dict,
                           ratios: Sequence[Tuple[int, int]] = DEFAULT_RATIOS) -> Dict:
    expected = expected_pairwise_comparisons(profile, ratios)
    missing = diff_missing(profile, expected, ratios)
    profile["completeness"] = {
        "expected_count": len(expected),
        "present_count": len(expected) - len(missing),
        "missing": [{"pigments": m["pigments"], "parts": m["parts"]} for m in missing],
        "complete": len(missing) == 0,
    }
    return profile


# ---------------------------------------------------------------------------
# MeasuredProfileModel -- the profile-driven (M7.2) predictor (R1)
# ---------------------------------------------------------------------------

class MeasuredProfileModel:
    """Predict mixed colors from a measured palette profile (custom palette).

    Mirrors m7_1_unified.MeasuredPairwiseModel, but keyed on the profile's color
    ids and built from the profile's measured comparisons.
    """

    def __init__(self, profile: Dict):
        self.m71 = _m71()
        self.profile = profile
        self.id_to_hex: Dict[str, str] = {c["id"]: normalize_hex(c["hex"]) for c in profile["colors"]}
        self.id_to_name: Dict[str, str] = {c["id"]: c.get("name", c["id"]) for c in profile["colors"]}
        self.hex_to_id: Dict[str, str] = {h: i for i, h in self.id_to_hex.items()}
        self._anchors: Dict[str, Dict] = {}
        self._curves: Dict[Tuple[str, str], object] = {}
        self._curve_observed: Dict[Tuple[str, str], bool] = {}
        self._build()

    # -- construction -------------------------------------------------------
    def _build(self) -> None:
        m71 = self.m71
        ids = list(self.id_to_hex.keys())
        points: Dict[Tuple[str, str], List[dict]] = {}
        observed: Dict[Tuple[str, str], bool] = {}

        # endpoints (pure colors) for every pair so interpolation has anchors
        for a, b in itertools.combinations(ids, 2):
            key = tuple(sorted((a, b)))
            points[key] = [
                {"t": 0.0, "hex": self.id_to_hex[key[0]], "sid": "endpoint"},
                {"t": 1.0, "hex": self.id_to_hex[key[1]], "sid": "endpoint"},
            ]
            observed[key] = False

        for comp in self.profile.get("comparisons", []):
            pigs = [str(p) for p in comp["pigments"]]
            parts = [float(p) for p in comp["parts"]]
            hexv = normalize_hex(comp["trycolors_result_hex"])
            if len(pigs) >= 3:
                cids, cparts = _canon(pigs, parts)
                self._anchors[comparison_key(cids, cparts)] = {
                    "hex": hexv,
                    "trycolors_name": comp.get("trycolors_name", ""),
                    "source": comp.get("source", ""),
                }
                continue
            if len(pigs) != 2:
                continue
            a, b = pigs
            if a not in self.id_to_hex or b not in self.id_to_hex:
                continue
            first, second = tuple(sorted((a, b)))
            w = m71.normalize(parts)
            wa = {a: w[0], b: w[1]}
            share_second = wa[second]
            points.setdefault((first, second), [])
            points[(first, second)].append({"t": float(share_second), "hex": hexv, "sid": comp.get("source", "obs")})
            observed[(first, second)] = True

        PairCurve = m71.PairCurve
        for key, pts in points.items():
            by_t: Dict[float, Tuple[int, dict]] = {}
            for p in pts:
                t = round(float(p["t"]), 12)
                priority = 1 if p["sid"] == "endpoint" else 0  # observed beats endpoint
                if t not in by_t or priority < by_t[t][0]:
                    by_t[t] = (priority, p)
            pts2 = [v[1] for _, v in sorted(by_t.items(), key=lambda kv: kv[0])]
            t_arr = np.array([p["t"] for p in pts2], dtype=float)
            hexes = [p["hex"] for p in pts2]
            oks = np.vstack([m71.hex_to_oklab(h) for h in hexes])
            self._curves[key] = PairCurve(first=key[0], second=key[1], t=t_arr, hexes=hexes,
                                          oklab=oks, source_ids=[p["sid"] for p in pts2])
            self._curve_observed[key] = observed.get(key, False)

    # -- coverage (R7 gate) -------------------------------------------------
    def covers(self, ids: Sequence[str], parts: Sequence[float]) -> bool:
        """True only when this exact recipe is backed by *measured* data.

        A single color is its own swatch; an exact n-ary anchor counts; otherwise
        every pair in the recipe must have at least one observed (non-endpoint)
        mix.  Endpoint-only curves do not count, so a recipe with no measured
        comparisons falls through to the KM physical model (rule R7).
        """
        ids = [str(i) for i in ids]
        if any(i not in self.id_to_hex for i in ids):
            return False
        if len(ids) == 1:
            return True
        if self.has_anchor(ids, parts):
            return True
        for a, b in itertools.combinations(sorted(ids), 2):
            if not self._curve_observed.get(tuple(sorted((a, b))), False):
                return False
        return True

    def has_anchor(self, ids: Sequence[str], parts: Sequence[float]) -> bool:
        cids, cparts = _canon(ids, parts)
        return comparison_key(cids, cparts) in self._anchors

    # -- prediction (mirrors M7.1 predict_recipe) --------------------------
    def predict_pair(self, a: str, b: str, wa: float, wb: float) -> Tuple[str, dict]:
        first, second = tuple(sorted((a, b)))
        curve = self._curves[(first, second)]
        weights = {a: float(wa), b: float(wb)}
        wf = weights[first]
        ws = weights[second]
        share_second = ws / (wf + ws) if (wf + ws) > 0 else 0.5
        hx = curve.predict(share_second)
        spacing = curve.nearest_spacing(share_second)
        exact = curve.has_exact_ratio(share_second)
        observed = self._curve_observed.get((first, second), False)
        if not observed:
            tier = "pair_endpoint_only"
        elif exact:
            tier = "pair_exact_observed"
        elif spacing <= 0.25:
            tier = "pair_dense_interpolation"
        elif spacing <= 0.50:
            tier = "pair_medium_interpolation"
        else:
            tier = "pair_sparse_interpolation"
        return hx, {"share_second": share_second, "curve_points": len(curve.t),
                    "nearest_spacing": spacing, "pair_confidence": tier}

    def predict_recipe(self, ids: Sequence[str], parts: Sequence[float]) -> Tuple[str, dict]:
        m71 = self.m71
        cids, cparts = _canon(ids, parts)
        akey = comparison_key(cids, cparts)
        if akey in self._anchors:
            return self._anchors[akey]["hex"], {
                "confidence_tier": "nary_exact_observed_anchor",
                "anchor_trycolors_name": self._anchors[akey].get("trycolors_name", ""),
            }
        weights = m71.normalize(cparts)
        if len(cids) == 1:
            return self.id_to_hex[cids[0]], {"confidence_tier": "single_anchor"}
        pair_oklabs = []
        pair_weights = []
        pair_meta = []
        for i, j in itertools.combinations(range(len(cids)), 2):
            hx, meta = self.predict_pair(cids[i], cids[j], weights[i], weights[j])
            pair_oklabs.append(m71.hex_to_oklab(hx))
            pair_weights.append(max(1e-12, weights[i] * weights[j]))
            pair_meta.append(meta)
        pw = np.array(pair_weights, dtype=float)
        pw /= pw.sum()
        ok = np.sum(np.vstack(pair_oklabs) * pw[:, None], axis=0)
        pred = m71.oklab_to_hex(ok)
        max_spacing = max(float(m["nearest_spacing"]) for m in pair_meta)
        min_points = min(int(m["curve_points"]) for m in pair_meta)
        if len(cids) == 2:
            conf = pair_meta[0]["pair_confidence"]
        elif max_spacing <= 0.25 and min_points >= 5:
            conf = "nary_dense_pairwise_composition"
        elif max_spacing <= 0.50:
            conf = "nary_medium_pairwise_composition"
        else:
            conf = "nary_sparse_pairwise_composition"
        return pred, {"confidence_tier": conf, "max_spacing": max_spacing, "min_pair_points": min_points}


def risk_penalty(tier: str) -> float:
    """Risk penalty by confidence tier (mirrors m7_1_unified.m7_risk_penalty)."""
    if tier in ("single_anchor", "nary_exact_observed_anchor", "pair_exact_observed",
                "pair_dense_interpolation"):
        return 0.0
    if tier == "pair_medium_interpolation":
        return 1.0
    if tier == "pair_sparse_interpolation":
        return 2.0
    if tier == "nary_dense_pairwise_composition":
        return 1.5
    if tier == "nary_medium_pairwise_composition":
        return 2.5
    if tier == "pair_endpoint_only":
        return 5.0
    return 4.0


__all__ = [
    "SCHEMA_VERSION", "DEFAULT_ENGINE", "DEFAULT_MIXER_MODE", "DEFAULT_RATIOS",
    "normalize_hex", "assign_color_ids", "palette_id_for", "comparison_key",
    "new_profile", "profile_path", "load_profile", "save_profile", "bump_version",
    "merge_colors", "update_profile_colors",
    "expected_pairwise_comparisons", "diff_missing", "add_results",
    "recompute_completeness", "MeasuredProfileModel", "risk_penalty",
]
