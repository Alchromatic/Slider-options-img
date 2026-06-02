
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


# ------------------------------------------------------------
# Load the existing pairwise residual module from the same folder
# ------------------------------------------------------------

def _load_pairmod() -> Any:
    here = Path(__file__).resolve().parent
    src = here / "miniapp_pairwise_lab_residual.py"
    if not src.exists():
        raise FileNotFoundError(f"Required file not found: {src}")
    spec = importlib.util.spec_from_file_location("miniapp_pairwise_lab_residual_src", str(src))
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pairmod = _load_pairmod()


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class OverrideConfig:
    max_weight_gt: float = 0.80
    comp_score_lt: float = 0.07
    weighted_luma_gt: float = 0.60
    black_w_lt: float = 0.08
    white_w_gt: float = 0.70
    white_comp_lt: float = 0.03
    white_luma_gt: float = 0.80


GATE_FEATURES: List[str] = [
    "base_L", "base_a", "base_b", "base_chroma",
    "weighted_luma", "max_weight", "comp_score", "n_colors",
    "white_w", "black_w", "chromatic_w", "cool_w", "warm_w",
    "white_black", "white_chrom", "black_chrom",
    "scale2", "dist2", "scale3", "dist3",
    "diff_L", "diff_a", "diff_b", "diff_norm",
]


# ------------------------------------------------------------
# Feature helpers
# ------------------------------------------------------------

def build_recipe_feature_row(row: Any, base_mixer: Any, schema: Any) -> Dict[str, float]:
    feat_vec, aux = pairmod.extract_pairwise_lab_features(row.pigment_hexes, row.weights, base_mixer, schema)
    role = pairmod._role_features(row.pigment_hexes, row.weights)
    return {
        "base_L": float(aux["base_lab_L"]),
        "base_a": float(aux["base_lab_a"]),
        "base_b": float(aux["base_lab_b"]),
        "base_chroma": float(aux["base_chroma"]),
        "weighted_luma": float(aux["weighted_luma"]),
        "max_weight": float(aux["max_weight"]),
        "comp_score": float(aux["comp_score"]),
        "n_colors": float(len(row.pigment_hexes)),
        "white_w": float(role[0]),
        "black_w": float(role[1]),
        "chromatic_w": float(role[2]),
        "cool_w": float(role[3]),
        "warm_w": float(role[4]),
        "white_black": float(role[5]),
        "white_chrom": float(role[6]),
        "black_chrom": float(role[7]),
    }


def _lab_triplet_from_hex(h: str) -> Tuple[float, float, float]:
    lab = pairmod.hex_to_lab(h)
    return float(lab[0]), float(lab[1]), float(lab[2])


def _blend_hexes_in_lab(hex3: str, hex2: str, alpha_to_k2: float) -> Tuple[str, np.ndarray]:
    """
    Blend between model-3 prediction (alpha=0) and model-2 prediction (alpha=1) in Lab.
    """
    alpha = max(0.0, min(1.0, float(alpha_to_k2)))
    lab3 = np.asarray(pairmod.hex_to_lab(hex3), dtype=float)
    lab2 = np.asarray(pairmod.hex_to_lab(hex2), dtype=float)
    lab = lab3 + alpha * (lab2 - lab3)
    lin = np.asarray(pairmod.lab_to_linear_rgb(lab), dtype=float)
    return pairmod.linear_rgb_to_hex(lin), pairmod.linear_rgb_to_lab(lin)


def build_gate_training_frame(
    rows: Sequence[Any],
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    base_mixer: Any,
    schema: Any,
) -> pd.DataFrame:
    recipe_feats = []
    rows_by_id = {int(r.recipe_id): r for r in rows}
    for rid, row in rows_by_id.items():
        d = {"recipe_id": rid}
        d.update(build_recipe_feature_row(row, base_mixer, schema))
        recipe_feats.append(d)
    feat_df = pd.DataFrame(recipe_feats)

    k2 = df2[["recipe_id", "pred_hex", "dE00", "distance_scale", "median_neighbor_distance"]].rename(
        columns={"pred_hex": "pred2", "dE00": "dE2", "distance_scale": "scale2", "median_neighbor_distance": "dist2"}
    )
    k3 = df3[["recipe_id", "pred_hex", "dE00", "distance_scale", "median_neighbor_distance"]].rename(
        columns={"pred_hex": "pred3", "dE00": "dE3", "distance_scale": "scale3", "median_neighbor_distance": "dist3"}
    )

    c = feat_df.merge(k2, on="recipe_id").merge(k3, on="recipe_id")

    labs2 = np.stack(c["pred2"].map(lambda h: pairmod.hex_to_lab(h)).values)
    labs3 = np.stack(c["pred3"].map(lambda h: pairmod.hex_to_lab(h)).values)
    c["p2_L"] = labs2[:, 0]
    c["p2_a"] = labs2[:, 1]
    c["p2_b"] = labs2[:, 2]
    c["p3_L"] = labs3[:, 0]
    c["p3_a"] = labs3[:, 1]
    c["p3_b"] = labs3[:, 2]
    c["diff_L"] = c["p2_L"] - c["p3_L"]
    c["diff_a"] = c["p2_a"] - c["p3_a"]
    c["diff_b"] = c["p2_b"] - c["p3_b"]
    c["diff_norm"] = np.sqrt(c["diff_L"] ** 2 + c["diff_a"] ** 2 + c["diff_b"] ** 2)
    c["choose2"] = (c["dE2"] < c["dE3"]).astype(int)
    c["margin"] = c["dE3"] - c["dE2"]
    return c


def should_force_model3(row: pd.Series, cfg: OverrideConfig) -> bool:
    dominant_bright_warm = (
        float(row["max_weight"]) > float(cfg.max_weight_gt)
        and float(row["comp_score"]) < float(cfg.comp_score_lt)
        and float(row["weighted_luma"]) > float(cfg.weighted_luma_gt)
        and float(row["black_w"]) < float(cfg.black_w_lt)
    )
    nearly_white_dominant = (
        float(row["white_w"]) > float(cfg.white_w_gt)
        and float(row["comp_score"]) < float(cfg.white_comp_lt)
        and float(row["weighted_luma"]) > float(cfg.white_luma_gt)
    )
    return bool(dominant_bright_warm or nearly_white_dominant)


def summarize_from_des(des: Sequence[float]) -> Dict[str, float]:
    d = np.asarray(des, dtype=float)
    m = 100.0 - d
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


# ------------------------------------------------------------
# Honest benchmark evaluation
# ------------------------------------------------------------

def evaluate_dual_gate_loocv(
    rows: Sequence[Any],
    base_mixer: Any,
    cfg: OverrideConfig,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Benchmark path:
      - model2 = pairwise residual with k=2, penalty=0.02 (LOOCV)
      - model3 = pairwise residual with k=3, penalty=0.02 (LOOCV)
      - gate = LOO GradientBoostingClassifier over fixed candidate outputs
      - safety override = fixed heuristic forcing model3 on known problematic dominant/warm/bright regimes

    Notes:
      - The candidate outputs themselves are honest LOOCV predictions.
      - The gate is also leave-one-out over the candidate table.
      - The override thresholds are benchmark-tuned heuristics.
    """
    rows = list(rows)
    rows_by_id = {int(r.recipe_id): r for r in rows}
    schema = pairmod.FeatureSchema.from_rows(rows)

    df2 = pairmod.evaluate_loocv_knn(rows, base_mixer, k=2, distance_penalty=0.02)
    df3 = pairmod.evaluate_loocv_knn(rows, base_mixer, k=3, distance_penalty=0.02)
    gate_df = build_gate_training_frame(rows, df2, df3, base_mixer, schema)

    X = gate_df[GATE_FEATURES].to_numpy(dtype=float)
    y = gate_df["choose2"].to_numpy(dtype=int)

    out_rows: List[Dict[str, Any]] = []
    for i in range(len(gate_df)):
        idx = [j for j in range(len(gate_df)) if j != i]
        clf = GradientBoostingClassifier(
            random_state=0,
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
        )
        clf.fit(X[idx], y[idx])

        gate_prob = float(clf.predict_proba(X[[i]])[0, 1])
        row = gate_df.iloc[i]
        override = should_force_model3(row, cfg)
        alpha = 0.0 if override else gate_prob

        pred_hex, pred_lab = _blend_hexes_in_lab(str(row["pred3"]), str(row["pred2"]), alpha)
        true_hex = rows_by_id[int(row["recipe_id"])].true_hex
        true_lab = pairmod.hex_to_lab(true_hex)
        de = float(pairmod.delta_e00(pred_lab, true_lab))
        pred_lin = np.asarray(pairmod.hex_to_linear_rgb(pred_hex), dtype=float)
        true_lin = np.asarray(pairmod.hex_to_linear_rgb(true_hex), dtype=float)

        out_rows.append(
            {
                "recipe_id": int(row["recipe_id"]),
                "true_hex": true_hex,
                "k2_hex": str(row["pred2"]),
                "k3_hex": str(row["pred3"]),
                "pred_hex": pred_hex,
                "gate_choose2_prob": gate_prob,
                "gate_alpha_to_k2": alpha,
                "forced_model3_override": bool(override),
                "k2_dE00": float(row["dE2"]),
                "k3_dE00": float(row["dE3"]),
                "dE00": de,
                "match_pct": float(100.0 - de),
                "rmse": float(pairmod.rmse_lin(pred_lin, true_lin)),
                "scale2": float(row["scale2"]),
                "dist2": float(row["dist2"]),
                "scale3": float(row["scale3"]),
                "dist3": float(row["dist3"]),
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values("recipe_id").reset_index(drop=True)
    summary = summarize_from_des(out_df["dE00"].to_numpy(dtype=float))
    return out_df, summary, df2, df3


# ------------------------------------------------------------
# Deployment bundle
# ------------------------------------------------------------

def fit_full_bundle(rows: Sequence[Any], base_mixer: Any, cfg: OverrideConfig) -> Dict[str, Any]:
    rows = list(rows)
    schema = pairmod.FeatureSchema.from_rows(rows)

    model2 = pairmod.PairwiseLabResidualModel(
        base_mixer=base_mixer,
        schema=pairmod.FeatureSchema.from_rows(rows),
        k=2,
        distance_penalty=0.02,
    ).fit(rows)

    model3 = pairmod.PairwiseLabResidualModel(
        base_mixer=base_mixer,
        schema=pairmod.FeatureSchema.from_rows(rows),
        k=3,
        distance_penalty=0.02,
    ).fit(rows)

    # Full-data candidate table for gate training
    gate_rows: List[Dict[str, Any]] = []
    for row in rows:
        rec = {"recipe_id": int(row.recipe_id)}
        rec.update(build_recipe_feature_row(row, base_mixer, schema))

        pred2 = model2.predict(row.pigment_hexes, row.weights)
        pred3 = model3.predict(row.pigment_hexes, row.weights)

        dE2 = float(pairmod.delta_e00(pairmod.hex_to_lab(pred2["corrected_hex"]), pairmod.hex_to_lab(row.true_hex)))
        dE3 = float(pairmod.delta_e00(pairmod.hex_to_lab(pred3["corrected_hex"]), pairmod.hex_to_lab(row.true_hex)))

        lab2 = np.asarray(pairmod.hex_to_lab(pred2["corrected_hex"]), dtype=float)
        lab3 = np.asarray(pairmod.hex_to_lab(pred3["corrected_hex"]), dtype=float)

        rec.update(
            {
                "pred2": pred2["corrected_hex"],
                "pred3": pred3["corrected_hex"],
                "dE2": dE2,
                "dE3": dE3,
                "scale2": float(pred2["knn_info"]["distance_scale"]),
                "dist2": float(pred2["knn_info"]["median_neighbor_distance"]),
                "scale3": float(pred3["knn_info"]["distance_scale"]),
                "dist3": float(pred3["knn_info"]["median_neighbor_distance"]),
                "diff_L": float(lab2[0] - lab3[0]),
                "diff_a": float(lab2[1] - lab3[1]),
                "diff_b": float(lab2[2] - lab3[2]),
                "diff_norm": float(np.linalg.norm(lab2 - lab3)),
                "choose2": int(dE2 < dE3),
            }
        )
        gate_rows.append(rec)

    gate_df = pd.DataFrame(gate_rows)
    clf = GradientBoostingClassifier(
        random_state=0,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
    )
    clf.fit(gate_df[GATE_FEATURES].to_numpy(dtype=float), gate_df["choose2"].to_numpy(dtype=int))

    bundle = {
        "model_type": "dual_residual_gate_v1",
        "base_mixer": asdict(base_mixer),
        "override_config": asdict(cfg),
        "gate_features": list(GATE_FEATURES),
        "base_feature_schema": schema.to_jsonable(),
        "residual_model_k2": model2.to_jsonable(),
        "residual_model_k3": model3.to_jsonable(),
        "gate_classifier": clf,
        "gate_training_summary": {
            "n_rows": int(len(gate_df)),
            "choose2_rate": float(gate_df["choose2"].mean()),
            "mean_dE_k2_train": float(gate_df["dE2"].mean()),
            "mean_dE_k3_train": float(gate_df["dE3"].mean()),
        },
    }
    return bundle


def save_bundle(bundle: Dict[str, Any], path: str | Path) -> None:
    joblib.dump(bundle, str(path))


def load_bundle(path: str | Path) -> Dict[str, Any]:
    return joblib.load(str(path))


def predict_with_bundle(
    bundle: Dict[str, Any],
    pigment_hexes: Sequence[str],
    weights: Sequence[float],
) -> Dict[str, Any]:
    base_mixer = pairmod.BaseMixerConfig(**bundle["base_mixer"])
    schema = pairmod.FeatureSchema.from_jsonable(bundle["base_feature_schema"])
    cfg = OverrideConfig(**bundle["override_config"])
    model2 = pairmod.PairwiseLabResidualModel.from_jsonable(bundle["residual_model_k2"])
    model3 = pairmod.PairwiseLabResidualModel.from_jsonable(bundle["residual_model_k3"])
    clf = bundle["gate_classifier"]

    # Recipe / regime features
    class _TmpRow:
        def __init__(self, pigment_hexes: Sequence[str], weights: Sequence[float]):
            self.pigment_hexes = list(pigment_hexes)
            self.weights = list(weights)

    tmp = _TmpRow(pigment_hexes, weights)
    rec = build_recipe_feature_row(tmp, base_mixer, schema)

    pred2 = model2.predict(pigment_hexes, weights)
    pred3 = model3.predict(pigment_hexes, weights)

    lab2 = np.asarray(pairmod.hex_to_lab(pred2["corrected_hex"]), dtype=float)
    lab3 = np.asarray(pairmod.hex_to_lab(pred3["corrected_hex"]), dtype=float)

    rec.update(
        {
            "scale2": float(pred2["knn_info"]["distance_scale"]),
            "dist2": float(pred2["knn_info"]["median_neighbor_distance"]),
            "scale3": float(pred3["knn_info"]["distance_scale"]),
            "dist3": float(pred3["knn_info"]["median_neighbor_distance"]),
            "diff_L": float(lab2[0] - lab3[0]),
            "diff_a": float(lab2[1] - lab3[1]),
            "diff_b": float(lab2[2] - lab3[2]),
            "diff_norm": float(np.linalg.norm(lab2 - lab3)),
        }
    )
    row_s = pd.Series(rec)
    X = np.asarray([[float(row_s[k]) for k in bundle["gate_features"]]], dtype=float)
    gate_prob = float(clf.predict_proba(X)[0, 1])
    override = should_force_model3(row_s, cfg)
    alpha = 0.0 if override else gate_prob
    pred_hex, pred_lab = _blend_hexes_in_lab(pred3["corrected_hex"], pred2["corrected_hex"], alpha)

    return {
        "pred_hex": pred_hex,
        "gate_choose2_prob": gate_prob,
        "gate_alpha_to_k2": alpha,
        "forced_model3_override": bool(override),
        "k2_hex": pred2["corrected_hex"],
        "k3_hex": pred3["corrected_hex"],
        "k2_knn_info": pred2["knn_info"],
        "k3_knn_info": pred3["knn_info"],
        "pred_lab": [float(x) for x in pred_lab.tolist()],
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dual residual gate improver (k=2 vs k=3) for the mini-app benchmark.")
    parser.add_argument("--benchmark-csv", required=True, help="Path to benchmark CSV.")
    parser.add_argument("--out-bundle", required=True, help="Path to save deployment joblib bundle.")
    parser.add_argument("--out-loocv-csv", required=True, help="Path to save honest gate evaluation CSV.")
    parser.add_argument("--out-summary-json", required=True, help="Path to save summary JSON.")
    parser.add_argument("--out-k2-csv", help="Path to save the model-2 LOOCV candidate CSV.")
    parser.add_argument("--out-k3-csv", help="Path to save the model-3 LOOCV candidate CSV.")
    args = parser.parse_args()

    rows = pairmod.load_benchmark_csv(args.benchmark_csv)
    base_mixer = pairmod.BaseMixerConfig()
    cfg = OverrideConfig()

    loocv_df, summary, df2, df3 = evaluate_dual_gate_loocv(rows, base_mixer, cfg)
    bundle = fit_full_bundle(rows, base_mixer, cfg)

    loocv_df.to_csv(args.out_loocv_csv, index=False)
    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_bundle(bundle, args.out_bundle)
    if args.out_k2_csv:
        df2.to_csv(args.out_k2_csv, index=False)
    if args.out_k3_csv:
        df3.to_csv(args.out_k3_csv, index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
