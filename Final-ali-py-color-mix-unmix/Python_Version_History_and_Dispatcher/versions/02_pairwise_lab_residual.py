
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Color utilities
# ============================================================

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def srgb_to_linear(c: float) -> float:
    c = float(c)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    c = clamp01(float(c))
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055


def clean_hex(h: str) -> str:
    s = str(h).strip().upper()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4:
        s = "#" + "".join(ch * 2 for ch in s[1:])
    if len(s) != 7:
        raise ValueError(f"Invalid hex: {h}")
    return s


def hex_to_rgb255(h: str) -> Tuple[int, int, int]:
    h = clean_hex(h).lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb255_to_hex(r: int, g: int, b: int) -> str:
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def hex_to_linear_rgb(h: str) -> Tuple[float, float, float]:
    r8, g8, b8 = hex_to_rgb255(h)
    return (
        srgb_to_linear(r8 / 255.0),
        srgb_to_linear(g8 / 255.0),
        srgb_to_linear(b8 / 255.0),
    )


def linear_rgb_to_hex(rgb: Sequence[float]) -> str:
    r, g, b = rgb
    return rgb255_to_hex(
        int(round(linear_to_srgb(float(r)) * 255)),
        int(round(linear_to_srgb(float(g)) * 255)),
        int(round(linear_to_srgb(float(b)) * 255)),
    )


def normalize_weights(w: Sequence[float], tol: float = 1e-12) -> List[float]:
    s = float(sum(w))
    if s <= tol:
        raise ValueError("Weights sum to 0.")
    return [float(x) / s for x in w]


def rmse_lin(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ============================================================
# Baseline mini-app mixer (locked defaults)
# ============================================================

def mix_linear_rgb(bases_lin: List[Tuple[float, float, float]], w: Sequence[float]) -> Tuple[float, float, float]:
    w = normalize_weights(w)
    return tuple(
        clamp01(sum(b[i] * wi for b, wi in zip(bases_lin, w)))
        for i in range(3)
    )  # type: ignore[return-value]


def _ks_from_R(R: float, eps: float) -> float:
    R = max(float(R), float(eps))
    return (1.0 - R) ** 2 / (2.0 * R)


def _R_from_ks(KS: float) -> float:
    KS = float(KS)
    return max(0.0, (1.0 + KS) - math.sqrt(KS * KS + 2.0 * KS))


def mix_km(bases_lin: List[Tuple[float, float, float]], w: Sequence[float], eps: float) -> Tuple[float, float, float]:
    w = normalize_weights(w)
    out: List[float] = []
    for ch in range(3):
        KS = sum(_ks_from_R(b[ch], eps) * wi for b, wi in zip(bases_lin, w))
        out.append(clamp01(_R_from_ks(KS)))
    return (out[0], out[1], out[2])


def mix_hybrid_km_linear(
    bases_lin: List[Tuple[float, float, float]],
    w: Sequence[float],
    eps: float,
    t: float,
) -> Tuple[float, float, float]:
    t = clamp01(float(t))
    km = mix_km(bases_lin, w, eps)
    lin = mix_linear_rgb(bases_lin, w)
    return (
        clamp01((1.0 - t) * km[0] + t * lin[0]),
        clamp01((1.0 - t) * km[1] + t * lin[1]),
        clamp01((1.0 - t) * km[2] + t * lin[2]),
    )


@dataclass
class BaseMixerConfig:
    engine: str = "Hybrid (KM ⊕ Linear)"
    ks_eps: float = 0.015
    hybrid_t: float = 0.28

    def predict_linear_rgb(self, pigment_hexes: Sequence[str], weights: Sequence[float]) -> Tuple[float, float, float]:
        bases = [hex_to_linear_rgb(h) for h in pigment_hexes]
        return mix_hybrid_km_linear(bases, weights, eps=float(self.ks_eps), t=float(self.hybrid_t))


# ============================================================
# Lab / XYZ / ΔE00
# ============================================================

_RGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=float,
)
_XYZ_TO_RGB_D65 = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=float,
)
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=float)


def linear_rgb_to_xyz(rgb_lin: Sequence[float]) -> np.ndarray:
    v = np.array(rgb_lin, dtype=float).reshape(3)
    return _RGB_TO_XYZ_D65 @ v


def xyz_to_linear_rgb_arr(xyz: Sequence[float]) -> np.ndarray:
    return _XYZ_TO_RGB_D65 @ np.asarray(xyz, dtype=float).reshape(3)


def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    t = np.asarray(t, dtype=float)
    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4.0 / 29.0))


def xyz_to_lab(xyz: Sequence[float], white: np.ndarray = _D65_WHITE) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    x = xyz[..., 0] / white[0]
    y = xyz[..., 1] / white[1]
    z = xyz[..., 2] / white[2]
    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.array([L, a, b], dtype=float)


def linear_rgb_to_lab(rgb_lin: Sequence[float]) -> np.ndarray:
    return xyz_to_lab(linear_rgb_to_xyz(rgb_lin))


def hex_to_lab(h: str) -> np.ndarray:
    return linear_rgb_to_lab(hex_to_linear_rgb(h))


def lab_to_xyz(lab: Sequence[float], white: np.ndarray = _D65_WHITE) -> np.ndarray:
    L, a, b = [float(x) for x in np.asarray(lab, dtype=float).reshape(3)]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def finv(t: float) -> float:
        if t > delta:
            return t**3
        return 3.0 * (delta**2) * (t - 4.0 / 29.0)

    x = white[0] * finv(fx)
    y = white[1] * finv(fy)
    z = white[2] * finv(fz)
    return np.array([x, y, z], dtype=float)


def lab_to_linear_rgb(lab: Sequence[float]) -> Tuple[float, float, float]:
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_linear_rgb_arr(xyz)
    return tuple(clamp01(float(x)) for x in rgb)  # type: ignore[return-value]


def delta_e00(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    L1, a1, b1 = [float(x) for x in np.asarray(lab1, dtype=float).tolist()]
    L2, a2, b2 = [float(x) for x in np.asarray(lab2, dtype=float).tolist()]
    kL = 1.0
    kC = 1.0
    kH = 1.0

    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + (25.0**7)))) if C_bar > 0 else 0.0

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)

    def _hp(ap: float, b: float) -> float:
        if ap == 0.0 and b == 0.0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360.0 if h < 0.0 else h

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180.0:
            dhp -= 360.0
        elif dhp < -180.0:
            dhp += 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    if C1p * C2p == 0.0:
        hp_bar = h1p + h2p
    else:
        hsum = h1p + h2p
        hdiff = abs(h1p - h2p)
        if hdiff > 180.0:
            hp_bar = (hsum + 360.0) / 2.0 if hsum < 360.0 else (hsum - 360.0) / 2.0
        else:
            hp_bar = hsum / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hp_bar - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hp_bar))
        + 0.32 * math.cos(math.radians(3.0 * hp_bar + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hp_bar - 63.0))
    )

    d_ro = 30.0 * math.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Rc = 2.0 * math.sqrt((Cp_bar**7) / ((Cp_bar**7) + (25.0**7))) if Cp_bar > 0 else 0.0

    Sl = 1.0 + (0.015 * ((Lp_bar - 50.0) ** 2)) / math.sqrt(20.0 + ((Lp_bar - 50.0) ** 2))
    Sc = 1.0 + 0.045 * Cp_bar
    Sh = 1.0 + 0.015 * Cp_bar * T

    Rt = -math.sin(math.radians(2.0 * d_ro)) * Rc

    dE = math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return float(dE)


def match_percent_from_de00(de00: float) -> float:
    return float(max(0.0, min(100.0, 100.0 - float(de00))))


# ============================================================
# Benchmark row
# ============================================================

@dataclass
class RecipeRow:
    recipe_id: int
    pigment_names: List[str]
    pigment_hexes: List[str]
    weights: List[float]
    true_hex: str
    source_pdf: Optional[str] = None


def load_benchmark_csv(path: str | Path) -> List[RecipeRow]:
    df = pd.read_csv(path)
    rows: List[RecipeRow] = []
    required = {"recipe_id", "pigment_names", "pigment_hexes", "weights", "true_hex"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    for _, r in df.iterrows():
        names = [str(x) for x in json.loads(r["pigment_names"])]
        hexes = [clean_hex(x) for x in json.loads(r["pigment_hexes"])]
        weights = [float(x) for x in json.loads(r["weights"])]
        rows.append(
            RecipeRow(
                recipe_id=int(r["recipe_id"]),
                pigment_names=names,
                pigment_hexes=hexes,
                weights=weights,
                true_hex=clean_hex(r["true_hex"]),
                source_pdf=str(r["source_pdf"]) if "source_pdf" in df.columns and pd.notna(r["source_pdf"]) else None,
            )
        )
    return rows


# ============================================================
# Pairwise Lab residual features
# ============================================================

@dataclass
class FeatureSchema:
    known_hexes: List[str]
    known_pairs: List[Tuple[str, str]]

    def to_jsonable(self) -> Dict[str, Any]:
        return {"known_hexes": self.known_hexes, "known_pairs": [list(p) for p in self.known_pairs]}

    @staticmethod
    def from_rows(rows: Sequence[RecipeRow]) -> "FeatureSchema":
        known_hexes = sorted({clean_hex(h) for row in rows for h in row.pigment_hexes})
        known_pairs = sorted(
            {tuple(sorted((clean_hex(a), clean_hex(b)))) for row in rows for a, b in combinations(row.pigment_hexes, 2)}
        )
        return FeatureSchema(known_hexes=known_hexes, known_pairs=known_pairs)

    @staticmethod
    def from_jsonable(obj: Dict[str, Any]) -> "FeatureSchema":
        return FeatureSchema(
            known_hexes=[clean_hex(x) for x in obj["known_hexes"]],
            known_pairs=[tuple(sorted((clean_hex(a), clean_hex(b)))) for a, b in obj["known_pairs"]],
        )


def _base_pigment_stats(pigment_hexes: Sequence[str], weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labs = np.array([linear_rgb_to_lab(hex_to_linear_rgb(h)) for h in pigment_hexes], dtype=float)
    lumas = []
    for h in pigment_hexes:
        rgb = np.asarray(hex_to_linear_rgb(h), dtype=float)
        lumas.append(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    return labs, np.asarray(lumas, dtype=float), np.asarray(normalize_weights(weights), dtype=float)


def _role_features(pigment_hexes: Sequence[str], weights: Sequence[float]) -> np.ndarray:
    labs, lumas, w = _base_pigment_stats(pigment_hexes, weights)
    chromas = np.sqrt(labs[:, 1] ** 2 + labs[:, 2] ** 2)

    white_mask = np.exp(-((1.0 - lumas) / 0.20) ** 2) * np.exp(-(chromas / 20.0) ** 2)
    black_mask = np.exp(-(lumas / 0.15) ** 2)
    chromatic_mask = 1.0 - np.exp(-(chromas / 25.0) ** 2)
    cool_mask = (labs[:, 2] < 0.0).astype(float)
    warm_mask = (labs[:, 2] > 0.0).astype(float)

    return np.array(
        [
            float((w * white_mask).sum()),
            float((w * black_mask).sum()),
            float((w * chromatic_mask).sum()),
            float((w * cool_mask).sum()),
            float((w * warm_mask).sum()),
            float((w * white_mask * black_mask).sum()),
            float((w * white_mask * chromatic_mask).sum()),
            float((w * black_mask * chromatic_mask).sum()),
        ],
        dtype=float,
    )


def extract_pairwise_lab_features(
    pigment_hexes: Sequence[str],
    weights: Sequence[float],
    base_mixer: BaseMixerConfig,
    schema: FeatureSchema,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Feature design:
      - baseline prediction in Lab
      - recipe complexity stats
      - weighted mean/std of pigment Lab
      - weighted luma stats
      - complementary hue score and pair spread
      - soft role features (white/black/chromatic/cool/warm)
      - generic pair-aggregate features
      - exact pigment weights for known training hexes
      - exact pair weights sqrt(w_i*w_j) for known training pairs
    """
    hex_index = {h: i for i, h in enumerate(schema.known_hexes)}
    pair_index = {p: i for i, p in enumerate(schema.known_pairs)}

    hexes = [clean_hex(h) for h in pigment_hexes]
    w = np.asarray(normalize_weights(weights), dtype=float)

    base_lin = np.asarray(base_mixer.predict_linear_rgb(hexes, w), dtype=float)
    base_lab = linear_rgb_to_lab(base_lin)
    base_chroma = float(np.sqrt(base_lab[1] ** 2 + base_lab[2] ** 2))
    base_hue = float(math.atan2(base_lab[2], base_lab[1])) if base_chroma > 1e-8 else 0.0

    base_labs, lumas, w = _base_pigment_stats(hexes, w)
    wmean = (w[:, None] * base_labs).sum(axis=0)
    centered = base_labs - wmean[None, :]
    wstd = np.sqrt(np.maximum((w[:, None] * (centered**2)).sum(axis=0), 0.0))
    entropy = float(-(w * np.log(np.maximum(w, 1e-12))).sum())

    hues = np.array(
        [math.atan2(lb[2], lb[1]) if (lb[1] ** 2 + lb[2] ** 2) > 1e-8 else 0.0 for lb in base_labs],
        dtype=float,
    )
    comp_score = 0.0
    spread = 0.0
    pair_sums = np.zeros(5, dtype=float)  # total, |ΔL|, |Δa|, |Δb|, Lab dist
    pair_vec = np.zeros(len(schema.known_pairs), dtype=float)

    for i in range(len(hexes)):
        for j in range(i + 1, len(hexes)):
            wi = float(w[i])
            wj = float(w[j])
            pair_w = math.sqrt(max(0.0, wi * wj))
            lab_i = base_labs[i]
            lab_j = base_labs[j]
            dh = abs(hues[i] - hues[j])
            dh = min(dh, 2.0 * math.pi - dh)
            comp_score += wi * wj * (1.0 - math.cos(dh))
            dist_lab = float(np.linalg.norm(lab_i - lab_j))
            spread += wi * wj * dist_lab
            pair_sums += np.array(
                [
                    pair_w,
                    pair_w * abs(float(lab_i[0] - lab_j[0])),
                    pair_w * abs(float(lab_i[1] - lab_j[1])),
                    pair_w * abs(float(lab_i[2] - lab_j[2])),
                    pair_w * dist_lab,
                ],
                dtype=float,
            )
            pair_key = tuple(sorted((hexes[i], hexes[j])))
            idx = pair_index.get(pair_key)
            if idx is not None:
                pair_vec[idx] += pair_w

    weight_vec = np.zeros(len(schema.known_hexes), dtype=float)
    for h, wi in zip(hexes, w):
        idx = hex_index.get(h)
        if idx is not None:
            weight_vec[idx] += float(wi)

    generic = np.array(
        [
            float(base_lab[0]),
            float(base_lab[1]),
            float(base_lab[2]),
            float(base_chroma),
            float(math.cos(base_hue)),
            float(math.sin(base_hue)),
            float(len(hexes)),
            float(w.max()),
            float(w.min()),
            entropy,
            float(wmean[0]),
            float(wmean[1]),
            float(wmean[2]),
            float(wstd[0]),
            float(wstd[1]),
            float(wstd[2]),
            float(lumas.min()),
            float(lumas.max()),
            float((w * lumas).sum()),
            float(comp_score),
            float(spread),
        ],
        dtype=float,
    )
    role = _role_features(hexes, w)
    feature_vector = np.concatenate([generic, role, pair_sums, weight_vec, pair_vec]).astype(float)

    aux = {
        "base_lab_L": float(base_lab[0]),
        "base_lab_a": float(base_lab[1]),
        "base_lab_b": float(base_lab[2]),
        "base_chroma": float(base_chroma),
        "weighted_luma": float((w * lumas).sum()),
        "max_weight": float(w.max()),
        "comp_score": float(comp_score),
    }
    return feature_vector, aux


# ============================================================
# kNN Pairwise Lab residual model
# ============================================================

@dataclass
class PairwiseLabResidualModel:
    base_mixer: BaseMixerConfig
    schema: FeatureSchema
    k: int = 3
    distance_penalty: float = 0.02
    feature_mean: Optional[List[float]] = None
    feature_std: Optional[List[float]] = None
    train_features_z: Optional[List[List[float]]] = None
    train_delta_lab: Optional[List[List[float]]] = None
    train_recipe_ids: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None

    def fit(self, rows: Sequence[RecipeRow]) -> "PairwiseLabResidualModel":
        self.schema = FeatureSchema.from_rows(rows)
        features = []
        deltas = []
        ids = []
        for row in rows:
            feat, _ = extract_pairwise_lab_features(row.pigment_hexes, row.weights, self.base_mixer, self.schema)
            base_lab = linear_rgb_to_lab(self.base_mixer.predict_linear_rgb(row.pigment_hexes, row.weights))
            true_lab = hex_to_lab(row.true_hex)
            delta = true_lab - base_lab
            features.append(feat)
            deltas.append(delta)
            ids.append(int(row.recipe_id))

        X = np.vstack(features).astype(float)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        Xz = (X - mu) / sigma

        self.feature_mean = mu.tolist()
        self.feature_std = sigma.tolist()
        self.train_features_z = Xz.tolist()
        self.train_delta_lab = np.vstack(deltas).astype(float).tolist()
        self.train_recipe_ids = ids
        self.metadata = {
            "n_rows": len(rows),
            "feature_dim": int(X.shape[1]),
            "schema_hex_count": len(self.schema.known_hexes),
            "schema_pair_count": len(self.schema.known_pairs),
        }
        return self

    def _predict_delta_from_feature(self, feat: np.ndarray, exclude_recipe_id: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.feature_mean is None or self.feature_std is None or self.train_features_z is None or self.train_delta_lab is None:
            raise RuntimeError("Model is not fitted.")
        mu = np.asarray(self.feature_mean, dtype=float)
        sigma = np.asarray(self.feature_std, dtype=float)
        feat_z = (feat - mu) / sigma

        train_X = np.asarray(self.train_features_z, dtype=float)
        train_Y = np.asarray(self.train_delta_lab, dtype=float)
        train_ids = np.asarray(self.train_recipe_ids, dtype=int) if self.train_recipe_ids is not None else None

        if exclude_recipe_id is not None and train_ids is not None:
            mask = train_ids != int(exclude_recipe_id)
            train_X = train_X[mask]
            train_Y = train_Y[mask]
            kept_ids = train_ids[mask]
        else:
            kept_ids = train_ids

        d = np.sqrt(np.sum((train_X - feat_z[None, :]) ** 2, axis=1))
        order = np.argsort(d)[: max(1, int(self.k))]
        d_sel = d[order]
        bw = float(np.median(d_sel) + 1e-6)
        w = np.exp(-(d_sel**2) / (2.0 * bw * bw))
        if float(w.sum()) <= 0.0:
            w = np.ones_like(d_sel)
        w = w / float(w.sum())

        yhat = (w[:, None] * train_Y[order]).sum(axis=0)
        median_d = float(np.median(d_sel))
        # Reliability shrinkage for out-of-domain queries.
        scale = 1.0 / (1.0 + float(self.distance_penalty) * median_d)
        yhat = yhat * scale

        info = {
            "neighbor_recipe_ids": kept_ids[order].tolist() if kept_ids is not None else None,
            "neighbor_distances": d_sel.tolist(),
            "neighbor_weights": w.tolist(),
            "median_neighbor_distance": median_d,
            "distance_scale": scale,
        }
        return yhat, info

    def predict(
        self,
        pigment_hexes: Sequence[str],
        weights: Sequence[float],
        recipe_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        feat, aux = extract_pairwise_lab_features(pigment_hexes, weights, self.base_mixer, self.schema)
        base_lin = np.asarray(self.base_mixer.predict_linear_rgb(pigment_hexes, weights), dtype=float)
        base_lab = linear_rgb_to_lab(base_lin)
        delta_lab, info = self._predict_delta_from_feature(feat, exclude_recipe_id=recipe_id)
        corrected_lab = base_lab + delta_lab
        corrected_lin = np.asarray(lab_to_linear_rgb(corrected_lab), dtype=float)
        return {
            "base_linear_rgb": base_lin.tolist(),
            "base_hex": linear_rgb_to_hex(base_lin),
            "base_lab": base_lab.tolist(),
            "delta_lab": delta_lab.tolist(),
            "corrected_lab": corrected_lab.tolist(),
            "corrected_linear_rgb": corrected_lin.tolist(),
            "corrected_hex": linear_rgb_to_hex(corrected_lin),
            "aux": aux,
            "knn_info": info,
        }

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "model_type": "pairwise_lab_knn_residual",
            "base_mixer": asdict(self.base_mixer),
            "schema": self.schema.to_jsonable(),
            "k": int(self.k),
            "distance_penalty": float(self.distance_penalty),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "train_features_z": self.train_features_z,
            "train_delta_lab": self.train_delta_lab,
            "train_recipe_ids": self.train_recipe_ids,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_jsonable(obj: Dict[str, Any]) -> "PairwiseLabResidualModel":
        return PairwiseLabResidualModel(
            base_mixer=BaseMixerConfig(**obj["base_mixer"]),
            schema=FeatureSchema.from_jsonable(obj["schema"]),
            k=int(obj["k"]),
            distance_penalty=float(obj["distance_penalty"]),
            feature_mean=obj["feature_mean"],
            feature_std=obj["feature_std"],
            train_features_z=obj["train_features_z"],
            train_delta_lab=obj["train_delta_lab"],
            train_recipe_ids=obj["train_recipe_ids"],
            metadata=obj.get("metadata"),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_jsonable(), indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> "PairwiseLabResidualModel":
        return PairwiseLabResidualModel.from_jsonable(json.loads(Path(path).read_text(encoding="utf-8")))


# ============================================================
# Evaluation helpers
# ============================================================

def summarize_results(df: pd.DataFrame, match_col: str = "match_pct", de_col: str = "dE00") -> Dict[str, float]:
    m = np.asarray(df[match_col], dtype=float)
    d = np.asarray(df[de_col], dtype=float)
    return {
        "mean_match_pct": float(np.mean(m)),
        "median_match_pct": float(np.median(m)),
        "worst_match_pct": float(np.min(m)),
        "best_match_pct": float(np.max(m)),
        "mean_dE00": float(np.mean(d)),
        "median_dE00": float(np.median(d)),
        "p95_dE00": float(np.quantile(d, 0.95)),
        "max_dE00": float(np.max(d)),
    }


def evaluate_baseline(rows: Sequence[RecipeRow], base_mixer: BaseMixerConfig) -> pd.DataFrame:
    out = []
    for row in rows:
        pred_lin = np.asarray(base_mixer.predict_linear_rgb(row.pigment_hexes, row.weights), dtype=float)
        pred_hex = linear_rgb_to_hex(pred_lin)
        de = delta_e00(hex_to_lab(pred_hex), hex_to_lab(row.true_hex))
        out.append(
            {
                "recipe_id": int(row.recipe_id),
                "true_hex": row.true_hex,
                "pred_hex": pred_hex,
                "match_pct": match_percent_from_de00(de),
                "dE00": de,
                "rmse": rmse_lin(pred_lin, hex_to_linear_rgb(row.true_hex)),
                "pigment_names": json.dumps(row.pigment_names),
                "pigment_hexes": json.dumps(row.pigment_hexes),
                "weights": json.dumps([float(x) for x in normalize_weights(row.weights)]),
                "source_pdf": row.source_pdf,
            }
        )
    return pd.DataFrame(out)


def evaluate_loocv_knn(
    rows: Sequence[RecipeRow],
    base_mixer: BaseMixerConfig,
    k: int = 3,
    distance_penalty: float = 0.02,
) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    rows = list(rows)
    for i in range(len(rows)):
        train_rows = [rows[j] for j in range(len(rows)) if j != i]
        model = PairwiseLabResidualModel(
            base_mixer=base_mixer,
            schema=FeatureSchema.from_rows(train_rows),
            k=int(k),
            distance_penalty=float(distance_penalty),
        ).fit(train_rows)
        row = rows[i]
        pred = model.predict(row.pigment_hexes, row.weights, recipe_id=row.recipe_id)
        pred_hex = pred["corrected_hex"]
        de = delta_e00(hex_to_lab(pred_hex), hex_to_lab(row.true_hex))
        out.append(
            {
                "recipe_id": int(row.recipe_id),
                "true_hex": row.true_hex,
                "base_hex": pred["base_hex"],
                "pred_hex": pred_hex,
                "match_pct": match_percent_from_de00(de),
                "dE00": de,
                "rmse": rmse_lin(pred["corrected_linear_rgb"], hex_to_linear_rgb(row.true_hex)),
                "distance_scale": pred["knn_info"]["distance_scale"],
                "median_neighbor_distance": pred["knn_info"]["median_neighbor_distance"],
                "neighbor_recipe_ids": json.dumps(pred["knn_info"]["neighbor_recipe_ids"]),
                "neighbor_distances": json.dumps(pred["knn_info"]["neighbor_distances"]),
                "pigment_names": json.dumps(row.pigment_names),
                "pigment_hexes": json.dumps(row.pigment_hexes),
                "weights": json.dumps([float(x) for x in normalize_weights(row.weights)]),
                "source_pdf": row.source_pdf,
            }
        )
    return pd.DataFrame(out)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise Lab residual improver for the mini-app benchmark.")
    parser.add_argument("--benchmark-csv", required=True, help="Path to benchmark CSV.")
    parser.add_argument("--out-model-json", help="Where to save the fitted full-data model JSON.")
    parser.add_argument("--out-baseline-csv", help="Where to save baseline evaluation CSV.")
    parser.add_argument("--out-loocv-csv", help="Where to save LOOCV evaluation CSV.")
    parser.add_argument("--k", type=int, default=3, help="k for kNN residual memory.")
    parser.add_argument("--distance-penalty", type=float, default=0.02, help="Shrink residual by 1 / (1 + penalty * median_knn_distance).")
    args = parser.parse_args()

    rows = load_benchmark_csv(args.benchmark_csv)
    base_mixer = BaseMixerConfig()

    baseline_df = evaluate_baseline(rows, base_mixer)
    loocv_df = evaluate_loocv_knn(rows, base_mixer, k=args.k, distance_penalty=args.distance_penalty)
    fitted_model = PairwiseLabResidualModel(
        base_mixer=base_mixer,
        schema=FeatureSchema.from_rows(rows),
        k=int(args.k),
        distance_penalty=float(args.distance_penalty),
    ).fit(rows)

    print("Baseline summary:")
    print(json.dumps(summarize_results(baseline_df), indent=2))
    print("LOOCV improved summary:")
    print(json.dumps(summarize_results(loocv_df), indent=2))

    if args.out_model_json:
        fitted_model.save_json(args.out_model_json)
    if args.out_baseline_csv:
        baseline_df.to_csv(args.out_baseline_csv, index=False)
    if args.out_loocv_csv:
        loocv_df.to_csv(args.out_loocv_csv, index=False)


if __name__ == "__main__":
    main()
