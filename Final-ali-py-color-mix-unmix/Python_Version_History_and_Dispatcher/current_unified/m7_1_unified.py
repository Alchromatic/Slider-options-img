#!/usr/bin/env python3
"""
M7.1 unified color-mixing / unmixing file
=========================================

This file is a reviewed offline maintenance candidate for the recovered logic:

1. M4 forward-router policy
   - conservative route selection between baseline and dual-gate
   - included here as route logic only, not as the full dual-gate runtime

2. M7.1 inverse/unmix candidate
   - measured Trycolors UI pairwise model
   - fragile pair-ratio curves
   - exact observed n-ary anchors, including the H24 rescue winner
   - CIEDE2000 scoring and sampling-coverage tiers

The file intentionally keeps all executable logic in one place.  The measured
Trycolors UI data remains in the `data/` folder as CSV/JSON inputs.

Main command examples
---------------------

M4 route only:
    python m7_1_unified.py m4-route --pigments "CY,CR,BK" --parts "40,9,1"

M7.1 forward prediction for one recipe:
    python m7_1_unified.py m7-predict --pigments "CY,QM,UB" --parts "4,1,1"

M7.1 unmix one target:
    python m7_1_unified.py m7-unmix --target-hex "#706A35" --top-n 5

M7.1 unmix a target CSV:
    python m7_1_unified.py batch-unmix --targets data/target_colors_H01_H24.csv --outdir outputs

Notes
-----
- This file is not identified as the deployed/private frontend backend.
- M4 here is route policy only; the baseline and dual-gate runtimes are separate.
- No API calls occur in this core. Optional verification belongs to M7.2.
- Coverage tiers and risk penalties are heuristics, not statistical guarantees.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import numbers
import re
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Shared palette definitions
# -----------------------------------------------------------------------------

PALETTE_ORDER = [
    "Cadmium Yellow Light",
    "India Yellow Hue",
    "Cadmium Red Light",
    "Quinacridone Magenta",
    "Ultramarine Blue",
    "Phthalo Green",
    "Titanium White",
    "Carbon Black",
]

ABBR = {
    "CY": "Cadmium Yellow Light",
    "IY": "India Yellow Hue",
    "CR": "Cadmium Red Light",
    "QM": "Quinacridone Magenta",
    "UB": "Ultramarine Blue",
    "PG": "Phthalo Green",
    "TW": "Titanium White",
    "BK": "Carbon Black",
}
NAME_TO_ABBR = {v: k for k, v in ABBR.items()}
RELEASE_ID = "color-mixer-maintenance-r3-2026-11-07"
MAX_CANDIDATES = 1_000_000


def expand_pigment_name(x: str) -> str:
    """Accept either abbreviation (CY) or full pigment name."""
    s = str(x).strip()
    return ABBR.get(s.upper(), {n.casefold(): n for n in PALETTE_ORDER}.get(s.casefold(), s))


def _sequence(values, label: str) -> list:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence, not a string")
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence") from exc


def _finite_number(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_integer(value, label: str, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    value = int(value)
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return value


def _split_text(text: str, pattern: str) -> List[str]:
    if not isinstance(text, str):
        raise ValueError("Recipe text must be a string")
    if not text.strip():
        return []
    tokens = [token.strip() for token in re.split(pattern, text.strip())]
    if any(not token for token in tokens):
        raise ValueError("Recipe text contains an empty field")
    return tokens


def parse_list_text(text: str) -> List[str]:
    return _split_text(text, r"[,|]")


def parse_number_list(text: str) -> List[float]:
    return [_finite_number(x, "Parts") for x in _split_text(text, r"[,|/:]")]


def normalize(parts: Sequence[float]) -> List[float]:
    vals = [_finite_number(x, "Parts") for x in _sequence(parts, "Parts")]
    if not vals or any(not math.isfinite(v) or v < 0 for v in vals):
        raise ValueError("Parts must be finite, nonnegative, and nonempty")
    largest = max(vals)
    if largest <= 0:
        raise ValueError("zero-sum parts")
    # Rescaling first avoids overflow for valid large positive counts.
    scaled = [v / largest for v in vals]
    total = sum(scaled)
    return [v / total for v in scaled]


def canonicalize_recipe(names: Sequence[str], parts: Sequence[float]) -> Tuple[List[str], List[float]]:
    """Validate, merge duplicate identities, and order a nonempty recipe.

    Parts retain their supplied scale; recipe_key separately normalizes ratios.
    Zero parts are discarded so they cannot introduce phantom pair interactions.
    """
    return _canonicalize_recipe(names, parts, measured_only=True)


def _canonicalize_recipe(names, parts, *, measured_only):
    names, parts = _sequence(names, "Pigment names"), _sequence(parts, "Parts")
    if len(names) != len(parts):
        raise ValueError("Pigment names and parts must have the same length")
    if len(names) == 0:
        raise ValueError("A recipe must contain at least one pigment")
    order = {n: i for i, n in enumerate(PALETTE_ORDER)}
    merged: Dict[str, Fraction] = {}
    for name, part in zip(names, parts):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A pigment name must be a nonempty string")
        full_name = expand_pigment_name(name)
        if measured_only and full_name not in order:
            raise ValueError(f"Unknown pigment: {full_name}")
        value = _finite_number(part, f"Parts for {full_name}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Parts must be finite and nonnegative: {full_name}")
        if value > 0:
            merged[full_name] = merged.get(full_name, Fraction(0)) + Fraction(str(value))
    if not merged:
        raise ValueError("A recipe must contain at least one positive part")
    try:
        items = sorted([(n, float(p)) for n, p in merged.items()], key=lambda x: order.get(x[0], len(order)))
    except OverflowError as exc:
        raise ValueError("Merged parts exceed the supported numeric range") from exc
    if any(not math.isfinite(p) for _, p in items):
        raise ValueError("Merged parts exceed the supported numeric range")
    names2 = [n for n, _ in items]
    parts2 = [int(p) if p.is_integer() else p for _, p in items]
    normalize(parts2)
    return names2, parts2


def recipe_key(names: Sequence[str], parts: Sequence[float]) -> str:
    names2, parts2 = canonicalize_recipe(names, parts)
    # Decimal fractions avoid a raw-count key; 14 significant digits absorb
    # floating-point scale noise (e.g. 3*0.1) without dropping tiny positive
    # weights. This is an indexing key, not a numeric equality test: capture
    # grouping and lookup independently verify ratios within arithmetic noise.
    # Common 4:1:1, 8:2:2, and 0.4:0.1:0.1 forms share the same index.
    values = [Fraction(str(p)) for p in parts2]
    total = sum(values, Fraction(0))
    weights = [v / total for v in values]
    return json.dumps({"names": names2, "weights": [format(float(v), ".14g") for v in weights]}, sort_keys=True)


def _ratio_vector(parts: Sequence[float]) -> Tuple[float, ...]:
    return tuple(normalize(parts))


def _same_ratio_vector(first: Sequence[float], second: Sequence[float]) -> bool:
    """Allow arithmetic roundoff, without treating decimal hash bins as equality."""
    return len(first) == len(second) and all(
        abs(a - b) <= 4 * max(math.ulp(a), math.ulp(b))
        for a, b in zip(first, second)
    )


def _capture_group_key(names: Sequence[str], weights: Sequence[float]) -> str:
    # Full numeric identity makes the coarse public key safe as an index only.
    return json.dumps({"names": list(names), "float_weights": [value.hex() for value in weights]}, sort_keys=True)


def parse_parts_text(s: str) -> List[float]:
    return parse_number_list(s)


def parse_pigments_full(s: str) -> List[str]:
    return parse_list_text(s)


# -----------------------------------------------------------------------------
# Color utilities: sRGB, CIELAB, OKLab, CIEDE2000
# -----------------------------------------------------------------------------

def clean_hex(h: str) -> str:
    if not isinstance(h, str):
        raise ValueError("HEX color must be a string")
    s = str(h).strip().upper()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4:
        s = "#" + "".join(c * 2 for c in s[1:])
    if not re.fullmatch(r"#[0-9A-F]{6}", s):
        raise ValueError(f"Invalid hex: {h}")
    return s


def _vector3(value, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must have three finite numeric components") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must have three finite numeric components")
    return result


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = float(np.clip(c, 0.0, 1.0))
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def hex_to_linear_rgb(h: str) -> np.ndarray:
    h = clean_hex(h).lstrip("#")
    vals = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return np.array([_srgb_to_linear(v) for v in vals], dtype=float)


def linear_rgb_to_hex(rgb: np.ndarray) -> str:
    rgb = _vector3(rgb, "Linear RGB")
    srgb = [_linear_to_srgb(float(x)) for x in np.clip(rgb, 0, 1)]
    vals = [max(0, min(255, int(round(x * 255)))) for x in srgb]
    return "#{:02X}{:02X}{:02X}".format(*vals)


_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=float)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=float)


def linear_rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return _vector3(_RGB_TO_XYZ @ _vector3(rgb, "Linear RGB"), "XYZ result")


def xyz_to_linear_rgb(xyz: np.ndarray) -> np.ndarray:
    return _vector3(_XYZ_TO_RGB @ _vector3(xyz, "XYZ"), "Linear RGB result")


def _f_lab(t: float) -> float:
    d = 6 / 29
    return t ** (1 / 3) if t > d ** 3 else t / (3 * d * d) + 4 / 29


def _finv_lab(f: float) -> float:
    d = 6 / 29
    return f ** 3 if f > d else 3 * d * d * (f - 4 / 29)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    x, y, z = _vector3(xyz, "XYZ") / _D65
    fx, fy, fz = _f_lab(float(x)), _f_lab(float(y)), _f_lab(float(z))
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dtype=float)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = _vector3(lab, "CIELAB")
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    try:
        result = np.array([_finv_lab(fx), _finv_lab(fy), _finv_lab(fz)], dtype=float) * _D65
    except OverflowError as exc:
        raise ValueError("CIELAB exceeds the supported numeric range") from exc
    return _vector3(result, "XYZ result")


def hex_to_lab(h: str) -> np.ndarray:
    return xyz_to_lab(linear_rgb_to_xyz(hex_to_linear_rgb(h)))


def lab_to_hex(lab: np.ndarray) -> str:
    return linear_rgb_to_hex(xyz_to_linear_rgb(lab_to_xyz(lab)))


_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6309787005],
], dtype=float)
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=float)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = _M1 @ _vector3(rgb, "Linear RGB")
    return _vector3(_M2 @ np.cbrt(lms), "OKLab result")


def oklab_to_linear_rgb(ok: np.ndarray) -> np.ndarray:
    lms_ = _M2_INV @ _vector3(ok, "OKLab")
    lms = lms_ ** 3
    return _vector3(_M1_INV @ lms, "Linear RGB result")


def hex_to_oklab(h: str) -> np.ndarray:
    return linear_rgb_to_oklab(hex_to_linear_rgb(h))


def oklab_to_hex(ok: np.ndarray) -> str:
    return linear_rgb_to_hex(oklab_to_linear_rgb(ok))


def _ciede2000_impl(lab1, lab2) -> float:
    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    avg_C = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if C1p != 0 else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if C2p != 0 else 0.0
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p <= h1p:
        dhp = h2p - h1p + 360
    else:
        dhp = h2p - h1p - 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    avg_Lp = (L1 + L2) / 2
    avg_Cpp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        avg_hp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        avg_hp = (h1p + h2p + 360) / 2
    else:
        avg_hp = (h1p + h2p - 360) / 2
    T = 1 - 0.17 * math.cos(math.radians(avg_hp - 30)) + 0.24 * math.cos(math.radians(2 * avg_hp)) + 0.32 * math.cos(math.radians(3 * avg_hp + 6)) - 0.20 * math.cos(math.radians(4 * avg_hp - 63))
    d_ro = 30 * math.exp(-((avg_hp - 275) / 25) ** 2)
    R_C = 2 * math.sqrt((avg_Cpp ** 7) / (avg_Cpp ** 7 + 25 ** 7))
    S_L = 1 + (0.015 * ((avg_Lp - 50) ** 2)) / math.sqrt(20 + ((avg_Lp - 50) ** 2))
    S_C = 1 + 0.045 * avg_Cpp
    S_H = 1 + 0.015 * avg_Cpp * T
    R_T = -math.sin(math.radians(2 * d_ro)) * R_C
    return float(math.sqrt((dLp / S_L) ** 2 + (dCp / S_C) ** 2 + (dHp / S_H) ** 2 + R_T * (dCp / S_C) * (dHp / S_H)))


def ciede2000(lab1, lab2) -> float:
    first, second = _vector3(lab1, "First CIELAB color"), _vector3(lab2, "Second CIELAB color")
    try:
        result = _ciede2000_impl(first, second)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("CIELAB values exceed the supported metric range") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("CIELAB values exceed the supported metric range")
    return result


def de00_hex(a: str, b: str) -> float:
    return ciede2000(hex_to_lab(a), hex_to_lab(b))


def match_pct(de: float) -> float:
    """Return the legacy display score 100−ΔE00, not an accuracy probability."""
    value = _finite_number(de, "Color difference")
    if value < 0:
        raise ValueError("Color difference must be nonnegative")
    return max(0.0, 100.0 - value)


# -----------------------------------------------------------------------------
# M4 forward-router policy
# -----------------------------------------------------------------------------

def m4_role_weights(pigment_names: Sequence[str], weights: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    pigment_names, weights = _canonicalize_recipe(pigment_names, weights, measured_only=False)
    weights = normalize(weights)
    rw = {k: 0.0 for k in ["yellow", "red", "blue", "green", "magenta", "white", "black", "other"]}
    flags = {k: False for k in ["india_yellow", "cad_yellow", "cad_red", "qm", "ub", "phg", "white", "black"]}
    for name, w in zip(pigment_names, weights):
        n = str(name).lower()
        if "india yellow" in n:
            rw["yellow"] += w; flags["india_yellow"] = True
        elif "yellow" in n:
            rw["yellow"] += w; flags["cad_yellow"] = True
        elif "red" in n:
            rw["red"] += w; flags["cad_red"] = True
        elif "ultramarine" in n or "blue" in n:
            rw["blue"] += w; flags["ub"] = True
        elif "phthalo green" in n or "green" in n or "killarney" in n:
            rw["green"] += w; flags["phg"] = True
        elif "quinacridone" in n or "magenta" in n or "wisteria" in n or "contessa" in n:
            rw["magenta"] += w; flags["qm"] = True
        elif "white" in n or "bon jour" in n:
            rw["white"] += w; flags["white"] = True
        elif "black" in n:
            rw["black"] += w; flags["black"] = True
        else:
            rw["other"] += w
    return rw, flags


def _at_route_boundary(share: float, threshold: float) -> float:
    # Equivalent floating-point ratios can land a few ULPs either side of a cutoff.
    if abs(share - threshold) <= 4 * math.ulp(threshold):
        return threshold
    return share


def _is_yrb_only(rw: Dict[str, float]) -> bool:
    return (
        _at_route_boundary(rw["yellow"], 0.70) >= 0.70 and rw["red"] > 0 and rw["black"] > 0 and
        rw["blue"] == 0 and rw["green"] == 0 and rw["magenta"] == 0 and rw["white"] == 0
    )


def m4_is_yrb_only(pigment_names: Sequence[str], weights: Sequence[float]) -> bool:
    return _is_yrb_only(m4_role_weights(pigment_names, weights)[0])


def m4_choose_model(pigment_names: Sequence[str], weights: Sequence[float]) -> Tuple[str, str]:
    """Return (chosen_model, reason), where chosen_model is 'baseline' or 'dualgate'."""
    rw, flags = m4_role_weights(pigment_names, weights)
    if _is_yrb_only(rw):
        b = rw["black"]
        r = rw["red"]
        if (_at_route_boundary(b, 0.015) > 0.015 and _at_route_boundary(b, 0.025) <= 0.025
                and _at_route_boundary(r, 0.12) >= 0.12):
            return "baseline", "Y/R/B mid-black warm-brown -> baseline"
        return "dualgate", "Y/R/B outside mid-black band -> dual-gate"
    if flags["qm"] and flags["ub"] and rw["white"] == 0 and rw["red"] == 0 and rw["black"] == 0:
        if _at_route_boundary(rw["magenta"], 0.30) >= 0.30:
            return "baseline", "QM+UB no-white -> baseline"
    return "dualgate", "default -> dual-gate"


# -----------------------------------------------------------------------------
# M7.1 measured pairwise model
# -----------------------------------------------------------------------------

def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    order = {n: i for i, n in enumerate(PALETTE_ORDER)}
    names = [expand_pigment_name(a), expand_pigment_name(b)]
    if any(name not in order for name in names):
        raise ValueError("Pair contains an unknown pigment")
    if names[0] == names[1]:
        raise ValueError("A pair curve requires two distinct pigments")
    x, y = sorted(names, key=lambda n: order[n])
    return x, y


@dataclass
class PairCurve:
    first: str
    second: str
    t: np.ndarray       # share of canonical second pigment, from 0 to 1
    hexes: List[str]
    oklab: np.ndarray
    source_ids: List[str]

    def __post_init__(self):
        original = (expand_pigment_name(self.first), expand_pigment_name(self.second))
        self.first, self.second = canonical_pair(self.first, self.second)
        if original != (self.first, self.second):
            raise ValueError("Pair curve identities must be in canonical order")
        self.t = np.asarray(self.t, dtype=float)
        self.oklab = np.asarray(self.oklab, dtype=float)
        if (self.t.ndim != 1 or len(self.t) < 2 or not np.isfinite(self.t).all()
                or np.any(np.diff(self.t) <= 0) or self.t[0] != 0 or self.t[-1] != 1):
            raise ValueError("Pair curve ratios must increase strictly from 0 to 1")
        if (len(self.hexes) != len(self.t) or len(self.source_ids) != len(self.t)
                or self.oklab.shape != (len(self.t), 3) or not np.isfinite(self.oklab).all()):
            raise ValueError("Pair curve data lengths and finite OKLab points must agree")
        self.hexes = [clean_hex(h) for h in self.hexes]

    @staticmethod
    def _share(second_share: float) -> float:
        value = _finite_number(second_share, "Pair ratio")
        if not 0 <= value <= 1:
            raise ValueError("Pair ratio must be between zero and one")
        return value

    def exact_index(self, second_share: float) -> int | None:
        """Recognize floating arithmetic noise, not an interpolation neighborhood."""
        x = self._share(second_share)
        matches = []
        for index, value in enumerate(self.t):
            tolerance = 4 * max(math.ulp(x), math.ulp(float(value)))
            if abs(float(value) - x) <= tolerance:
                matches.append((abs(float(value) - x), index))
        return min(matches)[1] if matches else None

    def predict(self, second_share: float) -> str:
        x = self._share(second_share)
        exact = self.exact_index(x)
        if exact is not None:
            return self.hexes[exact]
        vals = [float(np.interp(x, self.t, self.oklab[:, dim])) for dim in range(3)]
        return oklab_to_hex(np.array(vals, dtype=float))

    def nearest_spacing(self, second_share: float) -> float:
        x = self._share(second_share)
        idx = int(np.searchsorted(self.t, x))
        left = max(0, idx - 1)
        right = min(len(self.t) - 1, idx)
        if right == left:
            return 0.0
        return float(self.t[right] - self.t[left])

    def has_exact_ratio(self, second_share: float, tol: float | None = None) -> bool:
        x = self._share(second_share)
        if tol is None:
            return self.exact_index(x) is not None
        tolerance = _finite_number(tol, "Ratio tolerance")
        if tolerance < 0:
            raise ValueError("Ratio tolerance must be nonnegative")
        return bool(np.any(np.abs(self.t - x) <= tolerance))


class MeasuredPairwiseModel:
    def __init__(self, palette: Dict[str, str], observations: pd.DataFrame):
        if not isinstance(palette, dict) or not palette:
            raise ValueError("A nonempty measured palette mapping is required")
        self.palette = {}
        for key, value in palette.items():
            name = expand_pigment_name(key)
            if name not in PALETTE_ORDER:
                raise ValueError(f"Unknown palette pigment: {name}")
            if name in self.palette:
                raise ValueError(f"Duplicate palette pigment identity: {name}")
            self.palette[name] = clean_hex(value)
        self.excluded_observations = 0
        self.observations = self._validated_observations(observations)
        self.nary_anchors = self._build_nary_anchors()
        self.curves = self._build_curves()

    def _validated_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(observations, pd.DataFrame):
            raise ValueError("Observations must be a pandas DataFrame")
        required = {"pigments_full", "parts_text", "trycolors_result_hex"}
        if not observations.empty and not required.issubset(observations.columns):
            raise ValueError(f"Missing observation columns: {sorted(required - set(observations.columns))}")
        rows, groups_by_names = [], {}
        for index, row in observations.iterrows():
            try:
                names, parts = canonicalize_recipe(parse_pigments_full(row["pigments_full"]), parse_parts_text(row["parts_text"]))
                true_hex = clean_hex(row["trycolors_result_hex"])
                if any(name not in self.palette for name in names):
                    self.excluded_observations += 1
                    continue
                key = recipe_key(names, parts)
                weights = _ratio_vector(parts)
                groups = groups_by_names.setdefault(tuple(names), [])
                matches = [group for group in groups if _same_ratio_vector(weights, group["weights"])]
                if len(matches) > 1:
                    raise ValueError("Ambiguous captures at floating-point ratio precision")
                if matches:
                    group = matches[0]
                    if group["hex"] != true_hex:
                        raise ValueError("Conflicting HEX observations for the same normalized recipe")
                else:
                    group_key = key if not any(group["key"] == key for group in groups) else _capture_group_key(names, weights)
                    group = {"key": group_key, "weights": weights, "hex": true_hex}
                    groups.append(group)
                if len(names) == 1 and true_hex != self.palette[names[0]]:
                    raise ValueError("Pure-pigment capture conflicts with the palette endpoint")
                record = row.to_dict()
                record.update({"pigments_full": "|".join(names), "parts_text": "/".join(str(p) for p in parts),
                               "trycolors_result_hex": true_hex, "_recipe_key": key, "_recipe_group": group["key"]})
                if not str(record.get("trial_id", "")).strip() or pd.isna(record.get("trial_id")):
                    record["trial_id"] = f"capture_row:{index}"
                rows.append(record)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"Invalid observation at row {index}: {exc}") from exc
        return pd.DataFrame(rows, columns=list(dict.fromkeys(list(observations.columns) + sorted(required) + ["trial_id", "_recipe_key", "_recipe_group"])))

    @classmethod
    def from_data_dir(cls, data_dir: str | Path):
        data_dir = Path(data_dir)
        palette = json.loads((data_dir / "palette_v1.json").read_text())
        csvs = [
            data_dir / "trycolors_ui_pairwise_50_50_matrix_P01_P28.csv",
            data_dir / "trycolors_ui_phase4_fragile_pair_curves_C01_C16.csv",
            data_dir / "trycolors_ui_additional_binary_captures.csv",
            data_dir / "trycolors_ui_nary_anchor_captures_h24_rescue.csv",
        ]
        frames = []
        missing = [p.name for p in csvs if not p.is_file()]
        if missing:
            raise ValueError(f"Measured profile is incomplete; missing capture files: {', '.join(missing)}")
        for p in csvs:
            frames.append(pd.read_csv(p, dtype=str, keep_default_na=False))
        obs = pd.concat(frames, ignore_index=True)
        if obs.empty:
            raise ValueError("Measured profile contains no captured observations")
        return cls(palette, obs)

    def _build_nary_anchors(self) -> Dict[str, dict]:
        anchors: Dict[str, dict] = {}
        self._nary_names_index = {}
        for _, r in self.observations.iterrows():
            names = parse_pigments_full(r.get("pigments_full", ""))
            parts = parse_parts_text(r.get("parts_text", ""))
            if len(names) < 3 or len(names) != len(parts):
                continue
            true_hex = clean_hex(r["trycolors_result_hex"])
            key = r["_recipe_group"]
            if key not in anchors:
                anchors[key] = {
                    "hex": true_hex,
                    "names": tuple(names),
                    "normalized_weights": _ratio_vector(parts),
                    "trial_id": str(r.get("trial_id", "")),
                    "trycolors_name": str(r.get("trycolors_name", "")),
                    "source": str(r.get("source", "observed_nary_anchor")),
                    "notes": str(r.get("notes", "")),
                }
            if key not in self._nary_names_index.setdefault(tuple(names), []):
                self._nary_names_index[tuple(names)].append(key)
        return anchors

    def _find_nary_anchor(self, names: Sequence[str], parts: Sequence[float]) -> dict | None:
        candidates = self._nary_names_index.get(tuple(names), [])
        if not candidates or not self.nary_anchors:
            return None
        weights = _ratio_vector(parts)
        matches = [self.nary_anchors[key] for key in candidates
                   if key in self.nary_anchors and _same_ratio_vector(weights, self.nary_anchors[key]["normalized_weights"])]
        if len(matches) > 1:
            raise ValueError("Ambiguous n-ary capture at floating-point ratio precision")
        return matches[0] if matches else None

    def _build_curves(self) -> Dict[Tuple[str, str], PairCurve]:
        points: Dict[Tuple[str, str], List[dict]] = {}
        for a, b in itertools.combinations(PALETTE_ORDER, 2):
            if a in self.palette and b in self.palette:
                key = (a, b)
                points.setdefault(key, [])
                points[key].append({"t": 0.0, "hex": self.palette[a], "source_id": f"endpoint:{NAME_TO_ABBR[a]}"})
                points[key].append({"t": 1.0, "hex": self.palette[b], "source_id": f"endpoint:{NAME_TO_ABBR[b]}"})
        for _, r in self.observations.iterrows():
            names = parse_pigments_full(r.get("pigments_full", ""))
            parts = parse_parts_text(r.get("parts_text", ""))
            if len(names) != 2 or len(parts) != 2:
                continue
            n2, p2 = canonicalize_recipe(names, parts)
            a, b = n2
            w = normalize(p2)
            second_share = w[1]
            key = (a, b)
            points.setdefault(key, [])
            points[key].append({"t": float(second_share), "hex": clean_hex(r["trycolors_result_hex"]), "source_id": str(r.get("trial_id", ""))})
        curves = {}
        for key, pts in points.items():
            by_t = {}
            for p in pts:
                t = float(p["t"])
                priority = 0 if str(p["source_id"]).startswith("endpoint:") else 1
                if t in by_t and by_t[t][1]["hex"] != p["hex"]:
                    raise ValueError("Conflicting pair colors at indistinguishable numeric ratios")
                if t not in by_t or priority < by_t[t][0]:
                    by_t[t] = (priority, p)
            pts2 = [v[1] for _, v in sorted(by_t.items(), key=lambda kv: kv[0])]
            t = np.array([p["t"] for p in pts2], dtype=float)
            hexes = [p["hex"] for p in pts2]
            oks = np.vstack([hex_to_oklab(h) for h in hexes])
            curves[key] = PairCurve(first=key[0], second=key[1], t=t, hexes=hexes, oklab=oks, source_ids=[p["source_id"] for p in pts2])
        return curves

    def predict_pair(self, a: str, b: str, weight_a: float, weight_b: float) -> Tuple[str, dict]:
        names, parts = canonicalize_recipe([a, b], [weight_a, weight_b])
        if any(name not in self.palette for name in names):
            raise ValueError("Pair contains a pigment outside this model's measured palette")
        if len(names) == 1:
            return self.palette[names[0]], {"pair_key": names[0], "share_second": None,
                "curve_points": 1, "nearest_spacing": 0.0, "pair_confidence": "pair_endpoint_anchor",
                "prediction_kind": "palette_endpoint", "is_observed": False, "observed_source_id": ""}
        first, second = names
        key = (first, second)
        if key not in self.curves:
            raise KeyError(f"No curve for pair: {first} + {second}")
        curve = self.curves[key]
        share_second = normalize(parts)[1]
        hx = curve.predict(share_second)
        spacing = curve.nearest_spacing(share_second)
        exact = curve.exact_index(share_second)
        is_endpoint = exact in (0, len(curve.t) - 1)
        if exact is not None and is_endpoint:
            tier = "pair_endpoint_anchor"
        elif exact is not None:
            tier = "pair_exact_observed"
        elif not any(not sid.startswith("endpoint:") for sid in curve.source_ids):
            tier = "pair_unmeasured_interpolation"
        elif spacing <= 0.25:
            tier = "pair_dense_interpolation"
        elif spacing <= 0.50:
            tier = "pair_medium_interpolation"
        else:
            tier = "pair_sparse_interpolation"
        meta = {
            "pair_key": f"{first}|{second}",
            "share_second": share_second,
            "curve_points": len(curve.t),
            "nearest_spacing": spacing,
            "pair_confidence": tier,
            "prediction_kind": "numeric_endpoint_limit" if is_endpoint else ("observed_binary" if exact is not None else "interpolated_binary"),
            "is_observed": exact is not None and not is_endpoint,
            "observed_source_id": curve.source_ids[exact] if exact is not None and not is_endpoint else "",
        }
        return hx, meta

    def predict_recipe(self, names: Sequence[str], parts: Sequence[float]) -> Tuple[str, dict]:
        names, parts = canonicalize_recipe(names, parts)
        missing = [name for name in names if name not in self.palette]
        if missing:
            raise ValueError(f"Pigments are outside this model's measured palette: {', '.join(missing)}")
        anchor = self._find_nary_anchor(names, parts) if len(names) >= 3 else None
        if anchor is not None:
            return anchor["hex"], {
                "confidence_tier": "nary_exact_observed_anchor",
                "n_pairs": 0,
                "max_spacing": 0.0,
                "min_pair_points": 999,
                "pair_meta": [],
                "anchor_trial_id": anchor.get("trial_id", ""),
                "anchor_trycolors_name": anchor.get("trycolors_name", ""),
                "anchor_source": anchor.get("source", ""),
                "anchor_notes": anchor.get("notes", ""),
                "prediction_kind": "observed_nary", "is_observed": True,
            }
        if len(names) == 1:
            return self.palette[names[0]], {
                "confidence_tier": "single_anchor",
                "n_pairs": 0,
                "max_spacing": 0.0,
                "min_pair_points": 999,
                "pair_meta": [],
                "prediction_kind": "palette_endpoint", "is_observed": False,
            }
        pair_oklabs = []
        pair_logweights = []
        pair_meta = []
        for i, j in itertools.combinations(range(len(names)), 2):
            hx, meta = self.predict_pair(names[i], names[j], parts[i], parts[j])
            pair_oklabs.append(hex_to_oklab(hx))
            pair_logweights.append(math.log(parts[i]) + math.log(parts[j]))
            pair_meta.append(meta)
        logs = np.asarray(pair_logweights, dtype=float)
        pw = np.exp(logs - np.max(logs))
        pw /= pw.sum()
        ok = np.sum(np.vstack(pair_oklabs) * pw[:, None], axis=0)
        pred = oklab_to_hex(ok)
        max_spacing = max(float(m["nearest_spacing"]) for m in pair_meta) if pair_meta else 0.0
        min_points = min(int(m["curve_points"]) for m in pair_meta) if pair_meta else 999
        if len(names) == 2:
            conf = pair_meta[0]["pair_confidence"]
        elif max_spacing <= 0.25 and min_points >= 5:
            conf = "nary_dense_pairwise_composition"
        elif max_spacing <= 0.50:
            conf = "nary_medium_pairwise_composition"
        else:
            conf = "nary_sparse_pairwise_composition"
        return pred, {"confidence_tier": conf, "n_pairs": len(pair_meta), "max_spacing": max_spacing, "min_pair_points": min_points, "pair_meta": pair_meta,
                      "prediction_kind": pair_meta[0]["prediction_kind"] if len(names) == 2 else "heuristic_nary_composition",
                      "is_observed": pair_meta[0]["is_observed"] if len(names) == 2 else False}

    def curve_table(self) -> pd.DataFrame:
        rows = []
        for _, c in self.curves.items():
            for t, hx, sid in zip(c.t, c.hexes, c.source_ids):
                rows.append({"pair_first": c.first, "pair_second": c.second, "second_share": float(t), "hex": hx, "source_id": sid, "observed": not str(sid).startswith("endpoint")})
        return pd.DataFrame(rows)

    def leave_one_out_captured_pairs(self) -> pd.DataFrame:
        rows = []
        for _, r in self.observations.iterrows():
            names = parse_pigments_full(r.get("pigments_full", ""))
            parts = parse_parts_text(r.get("parts_text", ""))
            if len(names) != 2 or len(parts) != 2:
                continue
            n2, p2 = canonicalize_recipe(names, parts)
            true_hex = clean_hex(r["trycolors_result_hex"])
            trial_id = str(r.get("trial_id", ""))
            group = r["_recipe_group"]
            obs2 = self.observations[self.observations["_recipe_group"] != group].copy()
            model2 = MeasuredPairwiseModel(self.palette, obs2)
            pred, meta = model2.predict_recipe(n2, p2)
            d = de00_hex(pred, true_hex)
            rows.append({"trial_id": trial_id, "recipe_short": r.get("recipe_short", ""), "parts_text": r.get("parts_text", ""), "true_hex": true_hex, "pred_hex": pred, "dE00": d, "match_pct": match_pct(d), "confidence_tier": meta["confidence_tier"], "max_spacing": meta["max_spacing"], "min_pair_points": meta["min_pair_points"], "heldout_recipe_group_size": int((self.observations["_recipe_group"] == group).sum())})
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# M7.1 candidate generation and unmix scoring
# -----------------------------------------------------------------------------

def compositions(total: int, k: int) -> Iterable[List[int]]:
    total, k = _positive_integer(total, "Total parts"), _positive_integer(k, "Composition length")
    if k == 1:
        yield [total]
        return
    for first in range(1, total - k + 2):
        for rest in compositions(total - first, k - 1):
            yield [first] + rest


def generate_candidates(max_colors: int = 4, total_parts: int = 6, palette_names: Sequence[str] | None = None) -> pd.DataFrame:
    max_colors = _positive_integer(max_colors, "Max colors", len(PALETTE_ORDER))
    total_parts = _positive_integer(total_parts, "Total parts")
    if palette_names is None:
        candidate_names = list(PALETTE_ORDER)
    else:
        supplied = _sequence(palette_names, "Palette names")
        candidate_names, _ = canonicalize_recipe(supplied, [1] * len(supplied))
    max_k = min(max_colors, len(candidate_names), total_parts)
    candidate_count = len(candidate_names) + sum(math.comb(len(candidate_names), k) * math.comb(total_parts - 1, k - 1) for k in range(2, max_k + 1))
    if candidate_count > MAX_CANDIDATES:
        raise ValueError(f"Requested exact-sum grid has {candidate_count:,} candidates; limit is {MAX_CANDIDATES:,}. Reduce total_parts or max_colors (for example, total_parts=6 and max_colors=4).")
    rows = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(candidate_names, k):
            parts_iter = [[1]] if k == 1 else compositions(total_parts, k)
            for parts in parts_iter:
                names, parts2 = canonicalize_recipe(combo, parts)
                rows.append({"pigment_names": names, "parts": parts2, "normalized_weights": normalize(parts2)})
    return pd.DataFrame(rows)


def m7_risk_penalty(meta: dict) -> float:
    if not isinstance(meta, dict) or "confidence_tier" not in meta:
        raise ValueError("Prediction metadata must contain a known coverage tier")
    tier = meta["confidence_tier"]
    if tier in ("single_anchor", "nary_exact_observed_anchor", "pair_exact_observed", "pair_endpoint_anchor", "pair_dense_interpolation"):
        return 0.0
    if tier == "pair_medium_interpolation":
        return 1.0
    if tier == "pair_sparse_interpolation":
        return 2.0
    if tier == "nary_dense_pairwise_composition":
        return 1.5
    if tier == "nary_medium_pairwise_composition":
        return 2.5
    if tier in ("nary_sparse_pairwise_composition", "pair_unmeasured_interpolation"):
        return 4.0
    raise ValueError(f"Unknown prediction coverage tier: {tier}")


def m7_unmix(model: MeasuredPairwiseModel, target_hex: str, max_colors: int = 4, total_parts: int = 6, top_n: int = 5) -> pd.DataFrame:
    target_hex = clean_hex(target_hex)
    top_n = _positive_integer(top_n, "Top N")
    candidates = generate_candidates(max_colors=max_colors, total_parts=total_parts, palette_names=list(model.palette))
    scored = []
    for _, c in candidates.iterrows():
        pred, meta = model.predict_recipe(c["pigment_names"], c["parts"])
        d = de00_hex(pred, target_hex)
        penalty = m7_risk_penalty(meta)
        scored.append({
            "score_with_risk_penalty": d + penalty,
            "predicted_dE00": d,
            "predicted_match_pct": match_pct(d),
            "predicted_hex": pred,
            "confidence_tier": meta["confidence_tier"],
            "max_pair_spacing": meta["max_spacing"],
            "min_pair_points": meta["min_pair_points"],
            "risk_penalty": penalty,
            "anchor_trial_id": meta.get("anchor_trial_id", ""),
            "anchor_trycolors_name": meta.get("anchor_trycolors_name", ""),
            "anchor_source": meta.get("anchor_source", ""),
            "prediction_kind": meta.get("prediction_kind", ""),
            "is_observed": meta.get("is_observed", False),
            "release_id": RELEASE_ID,
            "pigment_names": json.dumps(c["pigment_names"]),
            "parts": json.dumps(c["parts"]),
            "normalized_weights": json.dumps(c["normalized_weights"]),
        })
    df = pd.DataFrame(scored).sort_values("score_with_risk_penalty", kind="stable").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df.head(top_n)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def build_model(data_dir: str | Path) -> MeasuredPairwiseModel:
    return MeasuredPairwiseModel.from_data_dir(data_dir)


def cmd_m4_route(args):
    names = [expand_pigment_name(x) for x in parse_list_text(args.pigments)]
    parts = parse_number_list(args.parts)
    names, parts = _canonicalize_recipe(names, parts, measured_only=False)
    choice, reason = m4_choose_model(names, parts)
    print(json.dumps({"version": "M4", "release_id": RELEASE_ID, "pigment_names": names, "parts": parts, "chosen_model": choice, "reason": reason}, indent=2))


def cmd_m7_predict(args):
    model = build_model(args.data_dir)
    names = [expand_pigment_name(x) for x in parse_list_text(args.pigments)]
    parts = parse_number_list(args.parts)
    pred, meta = model.predict_recipe(names, parts)
    print(json.dumps({"version": "M7.1", "release_id": RELEASE_ID, "pigment_names": canonicalize_recipe(names, parts)[0], "parts": canonicalize_recipe(names, parts)[1], "predicted_hex": pred, "meta": meta}, indent=2))


def cmd_m7_unmix(args):
    model = build_model(args.data_dir)
    df = m7_unmix(model, args.target_hex, max_colors=args.max_colors, total_parts=args.total_parts, top_n=args.top_n)
    print(df.to_string(index=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


def cmd_batch_unmix(args):
    model = build_model(args.data_dir)
    outdir = Path(args.outdir)
    targets = pd.read_csv(args.targets, dtype=str, keep_default_na=False)
    if targets.empty or "target_hex" not in targets.columns:
        raise ValueError("Target CSV must contain at least one row and a target_hex column")
    targets["target_hex"] = [clean_hex(value) for value in targets["target_hex"]]
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, t in targets.iterrows():
        target_hex = clean_hex(t["target_hex"])
        top = m7_unmix(model, target_hex, max_colors=args.max_colors, total_parts=args.total_parts, top_n=args.top_n)
        for _, r in top.iterrows():
            rec = r.to_dict()
            rec.update({"target_id": t.get("target_id", ""), "target_hex": target_hex, "target_name": t.get("target_name", "")})
            rows.append(rec)
    res = pd.DataFrame(rows)
    cols = ["target_id", "target_hex", "target_name"] + [c for c in res.columns if c not in ("target_id", "target_hex", "target_name")]
    res = res[cols]
    res.to_csv(outdir / "m7_1_unmix_topn.csv", index=False)
    top1 = res[res["rank"] == 1].copy()
    top1.to_csv(outdir / "m7_1_unmix_top1_capture_sheet.csv", index=False)
    summary = pd.DataFrame([{
        "n_targets": len(top1),
        "mean_internal_match": float(top1["predicted_match_pct"].mean()),
        "median_internal_match": float(top1["predicted_match_pct"].median()),
        "worst_internal_match": float(top1["predicted_match_pct"].min()),
        "count_internal_lt_85": int((top1["predicted_match_pct"] < 85).sum()),
        "count_internal_lt_80": int((top1["predicted_match_pct"] < 80).sum()),
    }])
    summary.to_csv(outdir / "m7_1_unmix_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {outdir}")


def cmd_diagnostics(args):
    model = build_model(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.curve_table().to_csv(outdir / "measured_pair_curve_table.csv", index=False)
    loocv = model.leave_one_out_captured_pairs()
    loocv.to_csv(outdir / "pair_curve_loocv.csv", index=False)
    summary = pd.DataFrame([{
        "n": len(loocv),
        "mean_match": float(loocv["match_pct"].mean()) if len(loocv) else None,
        "median_match": float(loocv["match_pct"].median()) if len(loocv) else None,
        "worst_match": float(loocv["match_pct"].min()) if len(loocv) else None,
        "mean_dE00": float(loocv["dE00"].mean()) if len(loocv) else None,
        "max_dE00": float(loocv["dE00"].max()) if len(loocv) else None,
    }])
    summary.to_csv(outdir / "pair_curve_loocv_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote diagnostics to {outdir}")


def main():
    default_data_dir = str(Path(__file__).resolve().parent / "data")
    p = argparse.ArgumentParser(description=f"Offline M4 route policy / M7.1 measured-profile maintenance candidate ({RELEASE_ID})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("m4-route", help="M4 route decision for a recipe")
    s.add_argument("--pigments", required=True, help="Comma-separated abbreviations or names, e.g. CY,CR,BK")
    s.add_argument("--parts", required=True, help="Comma/slash separated parts, e.g. 40,9,1")
    s.set_defaults(func=cmd_m4_route)

    s = sub.add_parser("m7-predict", help="Predict one recipe with the local measured profile")
    s.add_argument("--pigments", required=True, help="Comma-separated abbreviations or names, e.g. CY,QM,UB")
    s.add_argument("--parts", required=True, help="Comma/slash separated parts, e.g. 4,1,1")
    s.add_argument("--data-dir", default=default_data_dir)
    s.set_defaults(func=cmd_m7_predict)

    s = sub.add_parser("m7-unmix", help="M7.1 unmix one target color")
    s.add_argument("--target-hex", required=True)
    s.add_argument("--max-colors", type=int, default=4)
    s.add_argument("--total-parts", type=int, default=6, help="Exact total for nonsingleton recipes; this is not the deployed UI Parts contract")
    s.add_argument("--top-n", type=int, default=5)
    s.add_argument("--data-dir", default=default_data_dir)
    s.add_argument("--output", default="")
    s.set_defaults(func=cmd_m7_unmix)

    s = sub.add_parser("batch-unmix", help="M7.1 unmix a CSV of target colors")
    s.add_argument("--targets", default=str(Path(default_data_dir) / "target_colors_H01_H24.csv"))
    s.add_argument("--outdir", default="outputs_unified")
    s.add_argument("--max-colors", type=int, default=4)
    s.add_argument("--total-parts", type=int, default=6, help="Exact total for nonsingleton recipes")
    s.add_argument("--top-n", type=int, default=10)
    s.add_argument("--data-dir", default=default_data_dir)
    s.set_defaults(func=cmd_batch_unmix)

    s = sub.add_parser("diagnostics", help="Write curve table and LOOCV diagnostics")
    s.add_argument("--outdir", default="outputs_diagnostics")
    s.add_argument("--data-dir", default=default_data_dir)
    s.set_defaults(func=cmd_diagnostics)

    args = p.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
