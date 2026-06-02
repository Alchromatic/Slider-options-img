#!/usr/bin/env python3
"""
M6.1 Tiered Confidence Unmix

Structural fix for inverse-search exploitation:
- observed recipe override when Trycolors result is already known;
- trust-region / pair-coverage scoring;
- multi-model agreement gating;
- simple-model fallback for unanchored regions;
- confidence tier labels.

This script does not scrape Trycolors. It generates proposals and capture sheets.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import miniapp_pairwise_lab_residual as pairmod
import miniapp_dual_residual_gate as dg


# ---------------- OKLab utilities ----------------
M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6309787005],
], dtype=float)

M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
], dtype=float)


def clean_hex(h: str) -> str:
    return pairmod.clean_hex(str(h))


def hex_to_lab(h: str) -> np.ndarray:
    return np.asarray(pairmod.hex_to_lab(clean_hex(h)), dtype=float)


def lab_to_hex(lab: np.ndarray) -> str:
    rgb = pairmod.lab_to_linear_rgb(np.asarray(lab, dtype=float))
    return pairmod.linear_rgb_to_hex(tuple(float(np.clip(x, 0, 1)) for x in rgb))


def de00(a: str, b: str) -> float:
    return float(pairmod.delta_e00(pairmod.hex_to_lab(clean_hex(a)), pairmod.hex_to_lab(clean_hex(b))))


def match_pct(d: float) -> float:
    return max(0.0, 100.0 - float(d))


def hex_to_linrgb(h: str) -> np.ndarray:
    return np.asarray(pairmod.hex_to_linear_rgb(clean_hex(h)), dtype=float)


def linrgb_to_hex(rgb: np.ndarray) -> str:
    return pairmod.linear_rgb_to_hex(tuple(float(np.clip(x, 0, 1)) for x in rgb))


def linrgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = M1 @ np.asarray(rgb, dtype=float)
    return M2 @ np.cbrt(np.maximum(lms, 0.0))


def hex_to_oklab(h: str) -> np.ndarray:
    return linrgb_to_oklab(hex_to_linrgb(h))


def oklab_to_linrgb(ok: np.ndarray) -> np.ndarray:
    L, a, b = [float(x) for x in ok]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3
    return np.array([
         3.2409699419 * l - 2.2944965458 * m - 0.5056333816 * s,
        -1.5373831776 * l + 1.8759675015 * m + 0.0415550574 * s,
         0.1549195946 * l - 0.3159581810 * m + 1.8729161141 * s,
    ], dtype=float)


def oklab_to_hex(ok: np.ndarray) -> str:
    return linrgb_to_hex(oklab_to_linrgb(ok))


def parse_json(x):
    return json.loads(x) if isinstance(x, str) else x


def normalize(parts: Sequence[float]) -> List[float]:
    vals = [float(x) for x in parts]
    s = sum(vals)
    if s <= 0:
        raise ValueError("zero-sum parts")
    return [x / s for x in vals]


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


def canonicalize(names: Sequence[str], parts: Sequence[float]) -> Tuple[List[str], List[float]]:
    order = {n: i for i, n in enumerate(PALETTE_ORDER)}
    items = sorted([(str(n), float(p)) for n, p in zip(names, parts)], key=lambda x: order.get(x[0], 999))
    names2 = [n for n, _ in items]
    parts2 = [int(round(p)) if abs(p - round(p)) < 1e-9 else float(p) for _, p in items]
    return names2, parts2


def recipe_key(names: Sequence[str], parts: Sequence[float]) -> str:
    n, p = canonicalize(names, parts)
    return json.dumps({"names": n, "parts": p}, sort_keys=True)


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([str(a), str(b)]))


def pair_share(names: Sequence[str], weights: Sequence[float], a: str, b: str) -> float:
    # share of first pigment in sorted pair.
    pk = pair_key(a, b)
    share = 0.0
    total = 0.0
    for n, w in zip(names, weights):
        if str(n) in pk:
            total += float(w)
            if str(n) == pk[0]:
                share += float(w)
    return share / total if total > 0 else 0.5


def compositions(total: int, k: int) -> Iterable[List[int]]:
    if k == 1:
        yield [total]
        return
    for first in range(1, total - k + 2):
        for rest in compositions(total - first, k - 1):
            yield [first] + rest


def generate_candidate_grid(palette: Dict[str, str], max_colors: int, total_parts: int) -> pd.DataFrame:
    names = list(palette.keys())
    rows = []
    for k in range(1, max_colors + 1):
        for combo in itertools.combinations(names, k):
            if k == 1:
                parts_iter = [[1]]
            else:
                parts_iter = compositions(total_parts, k)
            for parts in parts_iter:
                n2, p2 = canonicalize(combo, parts)
                rows.append({
                    "candidate_source": "grid",
                    "source_id": "",
                    "true_hex": "",
                    "trycolors_name": "",
                    "pigment_names": n2,
                    "pigment_hexes": [palette[n] for n in n2],
                    "parts": p2,
                    "weights": normalize(p2),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["recipe_key"] = [recipe_key(n, p) for n, p in zip(df["pigment_names"], df["parts"])]
    return df.drop_duplicates("recipe_key").drop(columns=["recipe_key"]).reset_index(drop=True)


def load_observed(path: Path, palette: Dict[str, str]) -> pd.DataFrame:
    obs = pd.read_csv(path)
    rows = []
    for _, r in obs.iterrows():
        names = parse_json(r["pigment_names"])
        parts = parse_json(r["parts"])
        n2, p2 = canonicalize(names, parts)
        if any(n not in palette for n in n2):
            continue
        rows.append({
            "candidate_source": "observed_anchor",
            "source_id": str(r.get("source_id", "")),
            "true_hex": clean_hex(r["true_hex"]),
            "trycolors_name": r.get("trycolors_name", ""),
            "pigment_names": n2,
            "pigment_hexes": [palette[n] for n in n2],
            "parts": p2,
            "weights": normalize(p2),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["recipe_key"] = [recipe_key(n, p) for n, p in zip(df["pigment_names"], df["parts"])]
    return df.drop_duplicates("recipe_key").drop(columns=["recipe_key"]).reset_index(drop=True)


def add_observed_to_grid(grid: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    if obs.empty:
        return grid.copy()
    all_df = pd.concat([obs, grid], ignore_index=True)
    all_df["recipe_key"] = [recipe_key(n, p) for n, p in zip(all_df["pigment_names"], all_df["parts"])]
    all_df["priority"] = all_df["candidate_source"].map({"observed_anchor": 0, "grid": 1}).fillna(2)
    all_df = all_df.sort_values(["recipe_key", "priority"]).drop_duplicates("recipe_key", keep="first")
    return all_df.drop(columns=["recipe_key", "priority"]).reset_index(drop=True)


def build_pair_stats(obs: pd.DataFrame) -> Dict[Tuple[str, str], dict]:
    stats: Dict[Tuple[str, str], dict] = {}
    for _, r in obs.iterrows():
        names = r["pigment_names"]
        weights = r["weights"]
        if len(names) < 2:
            continue
        for a, b in itertools.combinations(names, 2):
            pk = pair_key(a, b)
            s = pair_share(names, weights, a, b)
            if pk not in stats:
                stats[pk] = {"count": 0, "shares": []}
            stats[pk]["count"] += 1
            stats[pk]["shares"].append(float(s))
    for v in stats.values():
        v["min_share"] = min(v["shares"])
        v["max_share"] = max(v["shares"])
    return stats


def pair_coverage(names: Sequence[str], weights: Sequence[float], pair_stats: Dict[Tuple[str, str], dict], min_pair_obs: int = 1, margin: float = 0.10) -> Tuple[bool, int, int, float, str]:
    if len(names) < 2:
        return True, 0, 0, 1.0, "single-pigment"
    total = 0
    ok = 0
    missing = []
    for a, b in itertools.combinations(names, 2):
        total += 1
        pk = pair_key(a, b)
        st = pair_stats.get(pk)
        if not st or st["count"] < min_pair_obs:
            missing.append(f"{pk[0]}+{pk[1]} missing")
            continue
        s = pair_share(names, weights, a, b)
        if st["min_share"] - margin <= s <= st["max_share"] + margin:
            ok += 1
        else:
            missing.append(f"{pk[0]}+{pk[1]} ratio outside hull")
    frac = ok / total if total else 1.0
    return ok == total, ok, total, frac, "; ".join(missing)


def linear_lab_prediction(hexes: Sequence[str], weights: Sequence[float]) -> str:
    w = np.asarray(normalize(weights), dtype=float)
    labs = np.vstack([hex_to_lab(h) for h in hexes])
    return lab_to_hex((w[:, None] * labs).sum(axis=0))


def linear_oklab_prediction(hexes: Sequence[str], weights: Sequence[float]) -> str:
    w = np.asarray(normalize(weights), dtype=float)
    oks = np.vstack([hex_to_oklab(h) for h in hexes])
    return oklab_to_hex((w[:, None] * oks).sum(axis=0))


def robust_median_prediction(preds: Sequence[str]) -> str:
    labs = np.vstack([hex_to_lab(p) for p in preds if isinstance(p, str) and p.startswith("#")])
    return lab_to_hex(np.median(labs, axis=0))


def model_agreement(preds: Sequence[str]) -> float:
    p = [x for x in preds if isinstance(x, str) and x.startswith("#")]
    if len(p) < 2:
        return 0.0
    return max(de00(a, b) for a, b in itertools.combinations(p, 2))


def compute_predictions(candidates: pd.DataFrame) -> pd.DataFrame:
    cfg = pairmod.BaseMixerConfig(engine="Hybrid (KM ⊕ Linear)", ks_eps=0.015, hybrid_t=0.28)
    bundle = dg.load_bundle(str(ROOT / "src" / "miniapp_dual_residual_gate_bundle.joblib"))
    rows = []
    for _, r in candidates.iterrows():
        hexes = r["pigment_hexes"]
        weights = r["weights"]
        pred_base = pairmod.linear_rgb_to_hex(cfg.predict_linear_rgb(hexes, weights))
        pred_dg = dg.predict_with_bundle(bundle, hexes, weights)["pred_hex"]
        pred_lab = linear_lab_prediction(hexes, weights)
        pred_ok = linear_oklab_prediction(hexes, weights)
        observed_pred = r["true_hex"] if isinstance(r.get("true_hex", ""), str) and str(r.get("true_hex", "")).startswith("#") else ""
        rows.append({
            **r.to_dict(),
            "pred_baseline": pred_base,
            "pred_dualgate": pred_dg,
            "pred_linear_lab": pred_lab,
            "pred_linear_oklab": pred_ok,
            "pred_observed": observed_pred,
        })
    return pd.DataFrame(rows)


def assign_tier(row, pair_stats, min_pair_obs: int, agreement_threshold: float) -> Tuple[str, str, str, float, float, str]:
    names = row["pigment_names"]
    weights = row["weights"]
    anchored, ok, total, frac, missing = pair_coverage(names, weights, pair_stats, min_pair_obs=min_pair_obs)
    preds = [row["pred_baseline"], row["pred_dualgate"], row["pred_linear_lab"], row["pred_linear_oklab"]]
    agree = model_agreement(preds)

    if row["candidate_source"] == "observed_anchor" and row["pred_observed"]:
        return "tier_1_observed_anchor", row["pred_observed"], "observed Trycolors recipe result used as forward prediction", frac, agree, missing

    if anchored and agree <= agreement_threshold:
        # Any model is safe; use robust median to avoid tiny individual model drift.
        return "tier_1_anchored_agree", robust_median_prediction(preds), f"all pairs anchored; models agree ΔE≤{agreement_threshold}", frac, agree, missing

    if anchored:
        return "tier_2_anchored_disagree", robust_median_prediction(preds), "pairs anchored but models disagree; robust median prediction", frac, agree, missing

    # Unanchored fallback: use simple model. Linear Lab was chosen because it
    # avoids trained-model extrapolation and performed well for the UB+PG failure.
    return "tier_3_simple_fallback", row["pred_linear_lab"], "unanchored pair region; using simple Lab fallback", frac, agree, missing


def search_targets(targets: pd.DataFrame, pred_df: pd.DataFrame, top_n: int, tier_max: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, t in targets.iterrows():
        target_hex = clean_hex(t["target_hex"])
        candidates = []
        for _, r in pred_df.iterrows():
            d = de00(r["pred_tiered"], target_hex)
            candidates.append((d, r))
        # Tiered dispatcher: try high tiers first with threshold; otherwise fall through.
        chosen_pool = None
        for tier_group, max_d in [
            (["tier_1_observed_anchor", "tier_1_anchored_agree"], tier_max["tier1"]),
            (["tier_2_anchored_disagree"], tier_max["tier2"]),
            (["tier_3_simple_fallback"], tier_max["tier3"]),
        ]:
            pool = [(d, r) for d, r in candidates if r["confidence_tier"] in tier_group and d <= max_d]
            if pool:
                chosen_pool = pool
                break
        if chosen_pool is None:
            # Best effort: nearest observed anchor if possible; otherwise nearest any candidate.
            pool = [(d, r) for d, r in candidates if r["confidence_tier"] == "tier_1_observed_anchor"]
            chosen_pool = pool if pool else candidates
        chosen_pool.sort(key=lambda x: x[0])

        for rank, (d, r) in enumerate(chosen_pool[:top_n], start=1):
            rows.append({
                "target_id": t["target_id"],
                "target_hex": target_hex,
                "target_name": t.get("target_name", ""),
                "rank": rank,
                "confidence_tier": r["confidence_tier"],
                "tier_reason": r["tier_reason"],
                "predicted_hex": r["pred_tiered"],
                "internal_dE00": d,
                "internal_match_pct": match_pct(d),
                "candidate_source": r["candidate_source"],
                "source_id": r.get("source_id", ""),
                "observed_true_hex": r.get("true_hex", ""),
                "known_external_dE_if_observed": de00(r.get("true_hex", ""), target_hex) if isinstance(r.get("true_hex", ""), str) and r.get("true_hex", "").startswith("#") else "",
                "known_external_match_if_observed": match_pct(de00(r.get("true_hex", ""), target_hex)) if isinstance(r.get("true_hex", ""), str) and r.get("true_hex", "").startswith("#") else "",
                "pair_coverage_fraction": r["pair_coverage_fraction"],
                "model_agreement_dE": r["model_agreement_dE"],
                "missing_pair_reason": r["missing_pair_reason"],
                "pigment_names": json.dumps(r["pigment_names"]),
                "pigment_hexes": json.dumps(r["pigment_hexes"]),
                "parts": json.dumps(r["parts"]),
                "normalized_weights": json.dumps(r["weights"]),
                "trycolors_result_hex": "",
                "trycolors_name": "",
                "stability_status": "",
                "screenshot_file": "",
                "notes": "",
            })
    return pd.DataFrame(rows)


def summarize(top1: pd.DataFrame) -> pd.DataFrame:
    vals = top1["internal_match_pct"].astype(float)
    rows = [{
        "segment": "all targets internal",
        "n": int(len(top1)),
        "mean_internal_match": float(vals.mean()),
        "median_internal_match": float(vals.median()),
        "worst_internal_match": float(vals.min()),
        "count_internal_lt_85": int((vals < 85).sum()),
        "count_internal_lt_80": int((vals < 80).sum()),
    }]
    known = pd.to_numeric(top1["known_external_match_if_observed"], errors="coerce")
    known = known.dropna()
    if len(known):
        rows.append({
            "segment": "observed-anchor rows only",
            "n": int(len(known)),
            "mean_internal_match": "",
            "median_internal_match": "",
            "worst_internal_match": "",
            "count_internal_lt_85": "",
            "count_internal_lt_80": "",
            "mean_known_external_match": float(known.mean()),
            "median_known_external_match": float(known.median()),
            "worst_known_external_match": float(known.min()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(ROOT / "data" / "target_colors_H01_H24.csv"))
    ap.add_argument("--max-colors", type=int, default=4)
    ap.add_argument("--total-parts", type=int, default=6)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--min-pair-obs", type=int, default=1)
    ap.add_argument("--agreement-threshold", type=float, default=6.0)
    ap.add_argument("--tier1-max-de", type=float, default=8.0)
    ap.add_argument("--tier2-max-de", type=float, default=10.0)
    ap.add_argument("--tier3-max-de", type=float, default=15.0)
    ap.add_argument("--outdir", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    palette = json.loads((ROOT / "data" / "palette_v1.json").read_text())
    targets = pd.read_csv(args.targets)
    obs = load_observed(ROOT / "data" / "observed_anchor_pool.csv", palette)
    grid = generate_candidate_grid(palette, max_colors=args.max_colors, total_parts=args.total_parts)
    candidates = add_observed_to_grid(grid, obs)
    candidates.to_csv(outdir / "tiered_candidate_grid.csv", index=False)

    pair_stats = build_pair_stats(obs)
    pair_rows = []
    for pk, st in pair_stats.items():
        pair_rows.append({"pair": f"{pk[0]} | {pk[1]}", "count": st["count"], "min_share": st["min_share"], "max_share": st["max_share"]})
    pd.DataFrame(pair_rows).sort_values("count", ascending=False).to_csv(outdir / "pair_trust_region_summary.csv", index=False)

    pred = compute_predictions(candidates)
    tier_rows = []
    for _, r in pred.iterrows():
        tier, pred_tier, reason, cov_frac, agree, missing = assign_tier(r, pair_stats, args.min_pair_obs, args.agreement_threshold)
        tier_rows.append({
            "confidence_tier": tier,
            "pred_tiered": pred_tier,
            "tier_reason": reason,
            "pair_coverage_fraction": cov_frac,
            "model_agreement_dE": agree,
            "missing_pair_reason": missing,
        })
    tier_df = pd.concat([pred.reset_index(drop=True), pd.DataFrame(tier_rows)], axis=1)
    tier_df_csv = tier_df.copy()
    for col in ["pigment_names", "pigment_hexes", "parts", "weights"]:
        tier_df_csv[col] = tier_df_csv[col].apply(json.dumps)
    tier_df_csv.to_csv(outdir / "tiered_candidate_predictions.csv", index=False)

    result = search_targets(
        targets,
        tier_df,
        top_n=args.top_n,
        tier_max={"tier1": args.tier1_max_de, "tier2": args.tier2_max_de, "tier3": args.tier3_max_de},
    )
    result.to_csv(outdir / "tiered_unmix_topn.csv", index=False)

    top1 = result[result["rank"] == 1].copy()
    top1.to_csv(outdir / "tiered_unmix_top1_capture_sheet.csv", index=False)
    top1.drop_duplicates(subset=["pigment_names", "parts", "predicted_hex"]).to_csv(outdir / "tiered_unmix_top1_deduplicated_capture_sheet.csv", index=False)

    s = summarize(top1)
    s.to_csv(outdir / "tiered_unmix_summary.csv", index=False)

    top1.groupby("confidence_tier", as_index=False).agg(
        n=("target_id", "size"),
        mean_internal_match=("internal_match_pct", "mean"),
        worst_internal_match=("internal_match_pct", "min"),
        mean_model_agreement_dE=("model_agreement_dE", "mean"),
    ).to_csv(outdir / "tier_distribution_summary.csv", index=False)

    print(s.to_string(index=False))
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
