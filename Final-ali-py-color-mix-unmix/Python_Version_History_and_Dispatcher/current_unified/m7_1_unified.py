#!/usr/bin/env python3
"""
M7.1 unified color-mixing / unmixing file
=========================================

This single Python file consolidates the current client-facing logic:

1. M4 forward-router policy
   - conservative route selection between baseline and dual-gate
   - included here as route logic only, not as the full dual-gate runtime

2. M7.1 inverse/unmix candidate
   - measured Trycolors UI pairwise model
   - fragile pair-ratio curves
   - exact observed n-ary anchors, including the H24 rescue winner
   - CIEDE2000 scoring and confidence tiers

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
- M7.1 is the latest inverse/unmix closure candidate.
- M4 remains the safest forward-router candidate.
- The Trycolors API outputs are not used here because the tested public API mode
  did not reproduce the Trycolors UI Pro Advanced behavior.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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


def expand_pigment_name(x: str) -> str:
    """Accept either abbreviation (CY) or full pigment name."""
    s = str(x).strip()
    return ABBR.get(s.upper(), s)


def parse_list_text(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace("|", ",").split(",") if x.strip()]


def parse_number_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).replace("|", ",").replace("/", ",").split(",") if x.strip()]


def normalize(parts: Sequence[float]) -> List[float]:
    vals = [float(x) for x in parts]
    s = sum(vals)
    if s <= 0:
        raise ValueError("zero-sum parts")
    return [v / s for v in vals]


def canonicalize_recipe(names: Sequence[str], parts: Sequence[float]) -> Tuple[List[str], List[float]]:
    order = {n: i for i, n in enumerate(PALETTE_ORDER)}
    full_names = [expand_pigment_name(n) for n in names]
    items = sorted([(str(n), float(p)) for n, p in zip(full_names, parts)], key=lambda x: order.get(x[0], 999))
    names2 = [n for n, _ in items]
    parts2 = [int(round(p)) if abs(p - round(p)) < 1e-9 else float(p) for _, p in items]
    return names2, parts2


def recipe_key(names: Sequence[str], parts: Sequence[float]) -> str:
    names2, parts2 = canonicalize_recipe(names, parts)
    return json.dumps({"names": names2, "parts": parts2}, sort_keys=True)


def parse_parts_text(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).replace("|", "/").split("/") if x.strip()]


def parse_pigments_full(s: str) -> List[str]:
    return [x.strip() for x in str(s).split("|") if x.strip()]


# -----------------------------------------------------------------------------
# Color utilities: sRGB, CIELAB, OKLab, CIEDE2000
# -----------------------------------------------------------------------------

def clean_hex(h: str) -> str:
    s = str(h).strip().upper()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4:
        s = "#" + "".join(c * 2 for c in s[1:])
    if len(s) != 7:
        raise ValueError(f"Invalid hex: {h}")
    return s


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
    return _RGB_TO_XYZ @ np.asarray(rgb, dtype=float)


def xyz_to_linear_rgb(xyz: np.ndarray) -> np.ndarray:
    return _XYZ_TO_RGB @ np.asarray(xyz, dtype=float)


def _f_lab(t: float) -> float:
    d = 6 / 29
    return t ** (1 / 3) if t > d ** 3 else t / (3 * d * d) + 4 / 29


def _finv_lab(f: float) -> float:
    d = 6 / 29
    return f ** 3 if f > d else 3 * d * d * (f - 4 / 29)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(xyz, dtype=float) / _D65
    fx, fy, fz = _f_lab(float(x)), _f_lab(float(y)), _f_lab(float(z))
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dtype=float)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = [float(x) for x in lab]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    return np.array([_finv_lab(fx), _finv_lab(fy), _finv_lab(fz)], dtype=float) * _D65


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
    lms = _M1 @ np.asarray(rgb, dtype=float)
    return _M2 @ np.cbrt(np.maximum(lms, 0.0))


def oklab_to_linear_rgb(ok: np.ndarray) -> np.ndarray:
    lms_ = _M2_INV @ np.asarray(ok, dtype=float)
    lms = lms_ ** 3
    return _M1_INV @ lms


def hex_to_oklab(h: str) -> np.ndarray:
    return linear_rgb_to_oklab(hex_to_linear_rgb(h))


def oklab_to_hex(ok: np.ndarray) -> str:
    return linear_rgb_to_hex(oklab_to_linear_rgb(ok))


def ciede2000(lab1, lab2) -> float:
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


def de00_hex(a: str, b: str) -> float:
    return ciede2000(hex_to_lab(a), hex_to_lab(b))


def match_pct(de: float) -> float:
    return max(0.0, 100.0 - float(de))


# -----------------------------------------------------------------------------
# M4 forward-router policy
# -----------------------------------------------------------------------------

def m4_role_weights(pigment_names: Sequence[str], weights: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, bool]]:
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


def m4_is_yrb_only(pigment_names: Sequence[str], weights: Sequence[float]) -> bool:
    rw, _ = m4_role_weights(pigment_names, weights)
    return (
        rw["yellow"] >= 0.70 and rw["red"] > 0 and rw["black"] > 0 and
        rw["blue"] == 0 and rw["green"] == 0 and rw["magenta"] == 0 and rw["white"] == 0
    )


def m4_choose_model(pigment_names: Sequence[str], weights: Sequence[float]) -> Tuple[str, str]:
    """Return (chosen_model, reason), where chosen_model is 'baseline' or 'dualgate'."""
    rw, flags = m4_role_weights(pigment_names, weights)
    if m4_is_yrb_only(pigment_names, weights):
        b = rw["black"]
        r = rw["red"]
        if 0.015 < b <= 0.025 and r >= 0.12:
            return "baseline", "Y/R/B mid-black warm-brown -> baseline"
        return "dualgate", "Y/R/B outside mid-black band -> dual-gate"
    if flags["qm"] and flags["ub"] and rw["white"] == 0 and rw["red"] == 0 and rw["black"] == 0:
        if rw["magenta"] >= 0.30:
            return "baseline", "QM+UB no-white -> baseline"
    return "dualgate", "default -> dual-gate"


# -----------------------------------------------------------------------------
# M7.1 measured pairwise model
# -----------------------------------------------------------------------------

def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    order = {n: i for i, n in enumerate(PALETTE_ORDER)}
    x, y = sorted([str(a), str(b)], key=lambda n: order.get(n, 999))
    return x, y


@dataclass
class PairCurve:
    first: str
    second: str
    t: np.ndarray       # share of canonical second pigment, from 0 to 1
    hexes: List[str]
    oklab: np.ndarray
    source_ids: List[str]

    def predict(self, second_share: float) -> str:
        x = float(np.clip(second_share, 0.0, 1.0))
        vals = [float(np.interp(x, self.t, self.oklab[:, dim])) for dim in range(3)]
        return oklab_to_hex(np.array(vals, dtype=float))

    def nearest_spacing(self, second_share: float) -> float:
        x = float(np.clip(second_share, 0.0, 1.0))
        idx = int(np.searchsorted(self.t, x))
        left = max(0, idx - 1)
        right = min(len(self.t) - 1, idx)
        if right == left:
            return 0.0
        return float(self.t[right] - self.t[left])

    def has_exact_ratio(self, second_share: float, tol: float = 1e-9) -> bool:
        x = float(np.clip(second_share, 0.0, 1.0))
        return bool(np.any(np.abs(self.t - x) <= tol))


class MeasuredPairwiseModel:
    def __init__(self, palette: Dict[str, str], observations: pd.DataFrame):
        self.palette = {k: clean_hex(v) for k, v in palette.items()}
        self.observations = observations.copy()
        self.nary_anchors = self._build_nary_anchors()
        self.curves = self._build_curves()

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
        for p in csvs:
            if p.exists():
                frames.append(pd.read_csv(p))
        obs = pd.concat(frames, ignore_index=True)
        obs = obs[obs["trycolors_result_hex"].astype(str).str.startswith("#")].copy()
        return cls(palette, obs)

    def _build_nary_anchors(self) -> Dict[str, dict]:
        anchors: Dict[str, dict] = {}
        for _, r in self.observations.iterrows():
            names = parse_pigments_full(r.get("pigments_full", ""))
            parts = parse_parts_text(r.get("parts_text", ""))
            if len(names) < 3 or len(names) != len(parts):
                continue
            true_hex = clean_hex(r["trycolors_result_hex"])
            key = recipe_key(names, parts)
            anchors[key] = {
                "hex": true_hex,
                "trial_id": str(r.get("trial_id", "")),
                "trycolors_name": str(r.get("trycolors_name", "")),
                "source": str(r.get("source", "observed_nary_anchor")),
                "notes": str(r.get("notes", "")),
            }
        return anchors

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
                t = round(float(p["t"]), 12)
                priority = 0 if not str(p["source_id"]).startswith("endpoint") else 1
                if t not in by_t or priority < by_t[t][0]:
                    by_t[t] = (priority, p)
            pts2 = [v[1] for _, v in sorted(by_t.items(), key=lambda kv: kv[0])]
            t = np.array([p["t"] for p in pts2], dtype=float)
            hexes = [p["hex"] for p in pts2]
            oks = np.vstack([hex_to_oklab(h) for h in hexes])
            curves[key] = PairCurve(first=key[0], second=key[1], t=t, hexes=hexes, oklab=oks, source_ids=[p["source_id"] for p in pts2])
        return curves

    def predict_pair(self, a: str, b: str, weight_a: float, weight_b: float) -> Tuple[str, dict]:
        first, second = canonical_pair(a, b)
        key = (first, second)
        if key not in self.curves:
            raise KeyError(f"No curve for pair: {first} + {second}")
        curve = self.curves[key]
        weights = {a: float(weight_a), b: float(weight_b)}
        wf = weights.get(first, 0.0)
        ws = weights.get(second, 0.0)
        share_second = ws / (wf + ws) if (wf + ws) > 0 else 0.5
        hx = curve.predict(share_second)
        spacing = curve.nearest_spacing(share_second)
        exact = curve.has_exact_ratio(share_second)
        if exact:
            tier = "pair_exact_observed"
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
        }
        return hx, meta

    def predict_recipe(self, names: Sequence[str], parts: Sequence[float]) -> Tuple[str, dict]:
        names, parts = canonicalize_recipe(names, parts)
        key = recipe_key(names, parts)
        if key in self.nary_anchors:
            anchor = self.nary_anchors[key]
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
            }
        weights = normalize(parts)
        if len(names) == 1:
            return self.palette[names[0]], {
                "confidence_tier": "single_anchor",
                "n_pairs": 0,
                "max_spacing": 0.0,
                "min_pair_points": 999,
                "pair_meta": [],
            }
        pair_oklabs = []
        pair_weights = []
        pair_meta = []
        for i, j in itertools.combinations(range(len(names)), 2):
            hx, meta = self.predict_pair(names[i], names[j], weights[i], weights[j])
            pair_oklabs.append(hex_to_oklab(hx))
            pair_weights.append(max(1e-12, weights[i] * weights[j]))
            pair_meta.append(meta)
        pw = np.array(pair_weights, dtype=float)
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
        return pred, {"confidence_tier": conf, "n_pairs": len(pair_meta), "max_spacing": max_spacing, "min_pair_points": min_points, "pair_meta": pair_meta}

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
            obs2 = self.observations[self.observations.get("trial_id", "") != trial_id].copy()
            model2 = MeasuredPairwiseModel(self.palette, obs2)
            pred, meta = model2.predict_recipe(n2, p2)
            d = de00_hex(pred, true_hex)
            rows.append({"trial_id": trial_id, "recipe_short": r.get("recipe_short", ""), "parts_text": r.get("parts_text", ""), "true_hex": true_hex, "pred_hex": pred, "dE00": d, "match_pct": match_pct(d), "confidence_tier": meta["confidence_tier"], "max_spacing": meta["max_spacing"], "min_pair_points": meta["min_pair_points"]})
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# M7.1 candidate generation and unmix scoring
# -----------------------------------------------------------------------------

def compositions(total: int, k: int) -> Iterable[List[int]]:
    if k == 1:
        yield [total]
        return
    for first in range(1, total - k + 2):
        for rest in compositions(total - first, k - 1):
            yield [first] + rest


def generate_candidates(max_colors: int = 4, total_parts: int = 6) -> pd.DataFrame:
    rows = []
    for k in range(1, max_colors + 1):
        for combo in itertools.combinations(PALETTE_ORDER, k):
            parts_iter = [[1]] if k == 1 else compositions(total_parts, k)
            for parts in parts_iter:
                names, parts2 = canonicalize_recipe(combo, parts)
                rows.append({"pigment_names": names, "parts": parts2, "normalized_weights": normalize(parts2)})
    return pd.DataFrame(rows)


def m7_risk_penalty(meta: dict) -> float:
    tier = meta["confidence_tier"]
    if tier in ("single_anchor", "nary_exact_observed_anchor", "pair_exact_observed", "pair_dense_interpolation"):
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


def m7_unmix(model: MeasuredPairwiseModel, target_hex: str, max_colors: int = 4, total_parts: int = 6, top_n: int = 5) -> pd.DataFrame:
    target_hex = clean_hex(target_hex)
    candidates = generate_candidates(max_colors=max_colors, total_parts=total_parts)
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
            "pigment_names": json.dumps(c["pigment_names"]),
            "parts": json.dumps(c["parts"]),
            "normalized_weights": json.dumps(c["normalized_weights"]),
        })
    df = pd.DataFrame(scored).sort_values("score_with_risk_penalty").reset_index(drop=True)
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
    names, parts = canonicalize_recipe(names, parts)
    choice, reason = m4_choose_model(names, normalize(parts))
    print(json.dumps({"version": "M4", "pigment_names": names, "parts": parts, "chosen_model": choice, "reason": reason}, indent=2))


def cmd_m7_predict(args):
    model = build_model(args.data_dir)
    names = [expand_pigment_name(x) for x in parse_list_text(args.pigments)]
    parts = parse_number_list(args.parts)
    pred, meta = model.predict_recipe(names, parts)
    print(json.dumps({"version": "M7.1", "pigment_names": canonicalize_recipe(names, parts)[0], "parts": canonicalize_recipe(names, parts)[1], "predicted_hex": pred, "meta": meta}, indent=2))


def cmd_m7_unmix(args):
    model = build_model(args.data_dir)
    df = m7_unmix(model, args.target_hex, max_colors=args.max_colors, total_parts=args.total_parts, top_n=args.top_n)
    print(df.to_string(index=False))
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


def cmd_batch_unmix(args):
    model = build_model(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.targets)
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
    p = argparse.ArgumentParser(description="Unified M4/M7.1 color mixing and unmixing file")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("m4-route", help="M4 route decision for a recipe")
    s.add_argument("--pigments", required=True, help="Comma-separated abbreviations or names, e.g. CY,CR,BK")
    s.add_argument("--parts", required=True, help="Comma/slash separated parts, e.g. 40,9,1")
    s.set_defaults(func=cmd_m4_route)

    s = sub.add_parser("m7-predict", help="M7.1 predict Trycolors UI result for one recipe")
    s.add_argument("--pigments", required=True, help="Comma-separated abbreviations or names, e.g. CY,QM,UB")
    s.add_argument("--parts", required=True, help="Comma/slash separated parts, e.g. 4,1,1")
    s.add_argument("--data-dir", default=default_data_dir)
    s.set_defaults(func=cmd_m7_predict)

    s = sub.add_parser("m7-unmix", help="M7.1 unmix one target color")
    s.add_argument("--target-hex", required=True)
    s.add_argument("--max-colors", type=int, default=4)
    s.add_argument("--total-parts", type=int, default=6)
    s.add_argument("--top-n", type=int, default=5)
    s.add_argument("--data-dir", default=default_data_dir)
    s.add_argument("--output", default="")
    s.set_defaults(func=cmd_m7_unmix)

    s = sub.add_parser("batch-unmix", help="M7.1 unmix a CSV of target colors")
    s.add_argument("--targets", default=str(Path(default_data_dir) / "target_colors_H01_H24.csv"))
    s.add_argument("--outdir", default="outputs_unified")
    s.add_argument("--max-colors", type=int, default=4)
    s.add_argument("--total-parts", type=int, default=6)
    s.add_argument("--top-n", type=int, default=10)
    s.add_argument("--data-dir", default=default_data_dir)
    s.set_defaults(func=cmd_batch_unmix)

    s = sub.add_parser("diagnostics", help="Write curve table and LOOCV diagnostics")
    s.add_argument("--outdir", default="outputs_diagnostics")
    s.add_argument("--data-dir", default=default_data_dir)
    s.set_defaults(func=cmd_diagnostics)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
