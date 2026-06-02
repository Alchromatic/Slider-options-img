
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import itertools
import json
import math

import numpy as np
import pandas as pd

from color_metrics import (
    clean_hex, hex_to_oklab, oklab_to_hex, de00_hex, match_pct
)

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


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    order = {n: i for i, n in enumerate(PALETTE_ORDER)}
    x, y = sorted([str(a), str(b)], key=lambda n: order.get(n, 999))
    return x, y


def canonicalize_recipe(names: Sequence[str], parts: Sequence[float]) -> Tuple[List[str], List[float]]:
    order = {n: i for i, n in enumerate(PALETTE_ORDER)}
    items = sorted([(str(n), float(p)) for n, p in zip(names, parts)], key=lambda x: order.get(x[0], 999))
    names2 = [n for n, _ in items]
    parts2 = [int(round(p)) if abs(p-round(p)) < 1e-9 else float(p) for _, p in items]
    return names2, parts2


def normalize(parts: Sequence[float]) -> List[float]:
    vals = [float(x) for x in parts]
    s = sum(vals)
    if s <= 0:
        raise ValueError("zero-sum parts")
    return [v/s for v in vals]


def parse_parts_text(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).replace("|", "/").split("/") if x.strip()]


def parse_pigments_full(s: str) -> List[str]:
    return [x.strip() for x in str(s).split("|") if x.strip()]


@dataclass
class PairCurve:
    first: str
    second: str
    # t = share of `second` in the pair, from 0 to 1
    t: np.ndarray
    hexes: List[str]
    oklab: np.ndarray
    source_ids: List[str]

    def predict(self, second_share: float) -> str:
        x = float(np.clip(second_share, 0.0, 1.0))
        vals = []
        for dim in range(3):
            vals.append(float(np.interp(x, self.t, self.oklab[:, dim])))
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
        self.curves = self._build_curves()

    @classmethod
    def from_csvs(cls, palette_path: str | Path, pairwise_csv: str | Path, curves_csv: str | Path, *extra_csvs: str | Path):
        palette = json.loads(Path(palette_path).read_text())
        parts = []
        for p in [pairwise_csv, curves_csv, *extra_csvs]:
            if p and Path(p).exists():
                df = pd.read_csv(p)
                parts.append(df)
        obs = pd.concat(parts, ignore_index=True)
        obs = obs[obs["trycolors_result_hex"].astype(str).str.startswith("#")].copy()
        return cls(palette, obs)

    def _build_curves(self) -> Dict[Tuple[str, str], PairCurve]:
        points: Dict[Tuple[str, str], List[dict]] = {}

        # endpoints from palette
        for a, b in itertools.combinations(PALETTE_ORDER, 2):
            if a in self.palette and b in self.palette:
                key = (a, b)
                points.setdefault(key, [])
                points[key].append({"t": 0.0, "hex": self.palette[a], "source_id": f"endpoint:{NAME_TO_ABBR[a]}"})
                points[key].append({"t": 1.0, "hex": self.palette[b], "source_id": f"endpoint:{NAME_TO_ABBR[b]}"})

        for _, r in self.observations.iterrows():
            names = parse_pigments_full(r["pigments_full"])
            parts = parse_parts_text(r["parts_text"])
            if len(names) != 2 or len(parts) != 2:
                continue
            n2, p2 = canonicalize_recipe(names, parts)
            a, b = n2
            w = normalize(p2)
            second_share = w[1]
            key = (a, b)
            points.setdefault(key, [])
            points[key].append({
                "t": float(second_share),
                "hex": clean_hex(r["trycolors_result_hex"]),
                "source_id": str(r.get("trial_id", "")),
            })

        curves = {}
        for key, pts in points.items():
            # Deduplicate by t: prefer real observed point over endpoint for same t.
            by_t = {}
            for p in pts:
                t = round(float(p["t"]), 12)
                # Observations should override endpoints if any duplicate.
                priority = 0 if not str(p["source_id"]).startswith("endpoint") else 1
                if t not in by_t or priority < by_t[t][0]:
                    by_t[t] = (priority, p)
            pts2 = [v[1] for k, v in sorted(by_t.items(), key=lambda kv: kv[0])]
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

        # Convert weights into share of canonical second.
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
        for (i, j) in itertools.combinations(range(len(names)), 2):
            hx, meta = self.predict_pair(names[i], names[j], weights[i], weights[j])
            pair_oklabs.append(hex_to_oklab(hx))
            # Product weights emphasize pairs where both pigments are present with substantial weight.
            pair_weights.append(max(1e-12, weights[i] * weights[j]))
            pair_meta.append(meta)

        pw = np.array(pair_weights, dtype=float)
        pw /= pw.sum()
        ok = np.sum(np.vstack(pair_oklabs) * pw[:, None], axis=0)
        pred = oklab_to_hex(ok)

        max_spacing = max(float(m["nearest_spacing"]) for m in pair_meta) if pair_meta else 0.0
        min_points = min(int(m["curve_points"]) for m in pair_meta) if pair_meta else 999

        # Confidence tier for full recipe.
        if len(names) == 2:
            conf = pair_meta[0]["pair_confidence"]
        elif max_spacing <= 0.25 and min_points >= 5:
            conf = "nary_dense_pairwise_composition"
        elif max_spacing <= 0.50:
            conf = "nary_medium_pairwise_composition"
        else:
            conf = "nary_sparse_pairwise_composition"

        return pred, {
            "confidence_tier": conf,
            "n_pairs": len(pair_meta),
            "max_spacing": max_spacing,
            "min_pair_points": min_points,
            "pair_meta": pair_meta,
        }

    def curve_table(self) -> pd.DataFrame:
        rows = []
        for key, c in self.curves.items():
            for t, hx, sid in zip(c.t, c.hexes, c.source_ids):
                rows.append({
                    "pair_first": c.first,
                    "pair_second": c.second,
                    "second_share": float(t),
                    "hex": hx,
                    "source_id": sid,
                    "observed": not str(sid).startswith("endpoint"),
                })
        return pd.DataFrame(rows)

    def leave_one_out_captured_pairs(self) -> pd.DataFrame:
        """LOOCV for captured pair points only. Endpoints stay fixed.

        The held-out captured point is predicted from the same pair curve with
        that point removed. This tests interpolation power on the measured pair data.
        """
        rows = []
        # Build rows from observations only.
        for _, r in self.observations.iterrows():
            names = parse_pigments_full(r["pigments_full"])
            parts = parse_parts_text(r["parts_text"])
            if len(names) != 2 or len(parts) != 2:
                continue
            n2, p2 = canonicalize_recipe(names, parts)
            true_hex = clean_hex(r["trycolors_result_hex"])
            trial_id = str(r.get("trial_id", ""))
            # Create observations without this trial_id (only one occurrence).
            obs2 = self.observations[self.observations.get("trial_id", "") != trial_id].copy()
            model2 = MeasuredPairwiseModel(self.palette, obs2)
            pred, meta = model2.predict_recipe(n2, p2)
            d = de00_hex(pred, true_hex)
            rows.append({
                "trial_id": trial_id,
                "recipe_short": r.get("recipe_short", ""),
                "parts_text": r.get("parts_text", ""),
                "true_hex": true_hex,
                "pred_hex": pred,
                "dE00": d,
                "match_pct": match_pct(d),
                "confidence_tier": meta["confidence_tier"],
                "max_spacing": meta["max_spacing"],
                "min_pair_points": meta["min_pair_points"],
            })
        return pd.DataFrame(rows)
