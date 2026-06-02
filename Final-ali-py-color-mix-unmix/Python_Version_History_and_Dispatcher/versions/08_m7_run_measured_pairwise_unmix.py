
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from color_metrics import de00_hex, match_pct
from measured_pairwise_model import MeasuredPairwiseModel, PALETTE_ORDER, canonicalize_recipe, normalize


def compositions(total: int, k: int):
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
            if k == 1:
                parts_iter = [[1]]
            else:
                parts_iter = compositions(total_parts, k)
            for parts in parts_iter:
                names, parts2 = canonicalize_recipe(combo, parts)
                rows.append({
                    "pigment_names": names,
                    "parts": parts2,
                    "normalized_weights": normalize(parts2),
                })
    return pd.DataFrame(rows)


def risk_penalty(meta: dict) -> float:
    tier = meta["confidence_tier"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=str(ROOT / "data" / "target_colors_H01_H24.csv"))
    parser.add_argument("--max-colors", type=int, default=4)
    parser.add_argument("--total-parts", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extra_binary = ROOT / "data" / "trycolors_ui_additional_binary_captures.csv"
    model = MeasuredPairwiseModel.from_csvs(
        ROOT / "data" / "palette_v1.json",
        ROOT / "data" / "trycolors_ui_pairwise_50_50_matrix_P01_P28.csv",
        ROOT / "data" / "trycolors_ui_phase4_fragile_pair_curves_C01_C16.csv",
        extra_binary,
    )

    # Save curve table and LOOCV diagnostics.
    curves = model.curve_table()
    curves.to_csv(outdir / "measured_pair_curve_table.csv", index=False)

    loocv = model.leave_one_out_captured_pairs()
    loocv.to_csv(outdir / "pair_curve_loocv.csv", index=False)
    loocv_summary = pd.DataFrame([{
        "n": len(loocv),
        "mean_match": float(loocv["match_pct"].mean()) if len(loocv) else None,
        "median_match": float(loocv["match_pct"].median()) if len(loocv) else None,
        "worst_match": float(loocv["match_pct"].min()) if len(loocv) else None,
        "mean_dE00": float(loocv["dE00"].mean()) if len(loocv) else None,
        "max_dE00": float(loocv["dE00"].max()) if len(loocv) else None,
    }])
    loocv_summary.to_csv(outdir / "pair_curve_loocv_summary.csv", index=False)

    targets = pd.read_csv(args.targets)
    candidates = generate_candidates(max_colors=args.max_colors, total_parts=args.total_parts)
    cand_rows = []
    for _, c in candidates.iterrows():
        pred, meta = model.predict_recipe(c["pigment_names"], c["parts"])
        rec = c.to_dict()
        rec["predicted_hex"] = pred
        rec["confidence_tier"] = meta["confidence_tier"]
        rec["max_pair_spacing"] = meta["max_spacing"]
        rec["min_pair_points"] = meta["min_pair_points"]
        rec["risk_penalty"] = risk_penalty(meta)
        rec["pair_meta_json"] = json.dumps(meta["pair_meta"])
        cand_rows.append(rec)
    cand_df = pd.DataFrame(cand_rows)
    # Serialize list fields.
    cand_out = cand_df.copy()
    for col in ["pigment_names", "parts", "normalized_weights"]:
        cand_out[col] = cand_out[col].apply(json.dumps)
    cand_out.to_csv(outdir / "candidate_predictions_measured_pairwise.csv", index=False)

    result_rows = []
    for _, t in targets.iterrows():
        target_hex = t["target_hex"]
        scored = []
        for _, c in cand_df.iterrows():
            d = de00_hex(c["predicted_hex"], target_hex)
            score = d + float(c["risk_penalty"])
            scored.append((score, d, c))
        scored.sort(key=lambda x: x[0])
        for rank, (score, d, c) in enumerate(scored[:args.top_n], start=1):
            result_rows.append({
                "target_id": t["target_id"],
                "target_hex": target_hex,
                "target_name": t.get("target_name", ""),
                "rank": rank,
                "score_with_risk_penalty": score,
                "predicted_dE00": d,
                "predicted_match_pct": match_pct(d),
                "predicted_hex": c["predicted_hex"],
                "confidence_tier": c["confidence_tier"],
                "max_pair_spacing": c["max_pair_spacing"],
                "min_pair_points": c["min_pair_points"],
                "risk_penalty": c["risk_penalty"],
                "pigment_names": json.dumps(c["pigment_names"]),
                "parts": json.dumps(c["parts"]),
                "normalized_weights": json.dumps(c["normalized_weights"]),
                "trycolors_result_hex": "",
                "trycolors_name": "",
                "stability_status": "",
                "screenshot_file": "",
                "notes": "",
            })
    res = pd.DataFrame(result_rows)
    res.to_csv(outdir / "m7_measured_pairwise_unmix_topn.csv", index=False)
    top1 = res[res["rank"] == 1].copy()
    top1.to_csv(outdir / "m7_measured_pairwise_unmix_top1_capture_sheet.csv", index=False)
    top1.drop_duplicates(subset=["pigment_names", "parts"]).to_csv(outdir / "m7_measured_pairwise_unmix_top1_deduplicated_capture_sheet.csv", index=False)

    summary = pd.DataFrame([{
        "n_targets": len(top1),
        "mean_internal_match": float(top1["predicted_match_pct"].mean()),
        "median_internal_match": float(top1["predicted_match_pct"].median()),
        "worst_internal_match": float(top1["predicted_match_pct"].min()),
        "count_internal_lt_85": int((top1["predicted_match_pct"] < 85).sum()),
        "count_internal_lt_80": int((top1["predicted_match_pct"] < 80).sum()),
    }])
    summary.to_csv(outdir / "m7_measured_pairwise_unmix_summary.csv", index=False)

    print("Pair-curve LOOCV:")
    print(loocv_summary.to_string(index=False))
    print("\nM7 measured-pairwise internal target summary:")
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
