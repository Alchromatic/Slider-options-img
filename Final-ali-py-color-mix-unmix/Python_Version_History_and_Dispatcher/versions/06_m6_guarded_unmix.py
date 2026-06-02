#!/usr/bin/env python3
"""
Guarded M4/M5 unmix runner.

This is the next-step version after the first closed-loop Trycolors screenshots:
it keeps the M4/M5 forward predictors, but adds external-reliability guards to
the inverse search so it does not over-trust recipe families that looked good
internally but failed in Trycolors.

Added safeguards:
- injects stable Trycolors anchor recipes into the candidate pool;
- penalizes recipe families that failed externally in the first screenshot round;
- penalizes high baseline/dualgate disagreement;
- gives a small reliability bonus to observed anchor recipes and observed pigment sets;
- outputs an M6_GUARDED branch beside original M4 and M5 proposals.

This does not scrape Trycolors. It creates a better capture sheet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from unmix_targets_m4_m5 import (
    generate_candidate_recipes,
    compute_candidate_predictions,
    de00_hex,
    match_pct,
    clean_hex,
    normalize,
    role_weights,
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


def parse_json(x):
    if isinstance(x, str):
        return json.loads(x)
    return x


def canonicalize_recipe(names: Sequence[str], parts: Sequence[float]):
    """Canonicalize recipe to palette order."""
    items = list(zip([str(n) for n in names], [float(p) for p in parts]))
    order = {n: i for i, n in enumerate(PALETTE_ORDER)}
    items = sorted(items, key=lambda t: order.get(t[0], 999))
    names2 = [n for n, _ in items]
    parts2 = [p for _, p in items]
    # Convert integer-looking parts to int.
    parts3 = [int(round(p)) if abs(p - round(p)) < 1e-9 else float(p) for p in parts2]
    return names2, parts3


def recipe_key(names: Sequence[str], parts: Sequence[float]) -> str:
    names2, parts2 = canonicalize_recipe(names, parts)
    return json.dumps({"names": names2, "parts": parts2}, sort_keys=True)


def pigment_set_key(names: Sequence[str]) -> str:
    names2 = sorted([str(n) for n in names])
    return json.dumps(names2)


def load_observed_anchors(path: Path, palette: dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    obs = pd.read_csv(path)
    rows = []
    for _, r in obs.iterrows():
        names = parse_json(r["pigment_names"])
        parts = parse_json(r["parts"])
        names, parts = canonicalize_recipe(names, parts)
        # Keep only known palette names.
        if any(n not in palette for n in names):
            continue
        weights = normalize(parts)
        rows.append({
            "pigment_names": names,
            "pigment_hexes": [palette[n] for n in names],
            "parts": parts,
            "weights": weights,
            "candidate_source": "observed_anchor",
            "anchor_case_id": r.get("case_id", ""),
            "anchor_true_hex": r.get("true_hex", ""),
            "anchor_trycolors_name": r.get("trycolors_name", ""),
        })
    return pd.DataFrame(rows)


def add_sources_and_anchors(grid: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    g = grid.copy()
    g["candidate_source"] = "grid"
    g["anchor_case_id"] = ""
    g["anchor_true_hex"] = ""
    g["anchor_trycolors_name"] = ""
    all_df = pd.concat([g, anchors], ignore_index=True) if len(anchors) else g

    # Deduplicate, preferring observed anchors over grid if same recipe exists.
    all_df["recipe_key"] = [recipe_key(n, p) for n, p in zip(all_df["pigment_names"], all_df["parts"])]
    all_df["source_priority"] = all_df["candidate_source"].map({"observed_anchor": 0, "grid": 1}).fillna(2)
    all_df = all_df.sort_values(["recipe_key", "source_priority"]).drop_duplicates("recipe_key", keep="first")
    all_df = all_df.drop(columns=["source_priority"]).reset_index(drop=True)
    return all_df


def family_risk_penalty(row) -> tuple[float, str]:
    names = row["pigment_names"]
    weights = row["weights"]
    rw, flags = role_weights(names, weights)

    present = set(names)
    penalty = 0.0
    reasons = []

    # Exact externally failed family patterns from first round.
    if {"Cadmium Yellow Light", "India Yellow Hue", "Titanium White", "Carbon Black"}.issubset(present) and rw["blue"] == 0 and rw["green"] == 0 and rw["red"] == 0 and rw["magenta"] == 0:
        penalty += 12.0
        reasons.append("penalty: yellow+india+white+black neutral family failed externally")

    if rw["yellow"] > 0 and rw["magenta"] > 0 and rw["blue"] == 0 and rw["green"] == 0 and rw["red"] == 0 and rw["black"] == 0 and rw["white"] == 0:
        penalty += 10.0
        reasons.append("penalty: yellow+magenta only failed externally")

    if rw["red"] > 0 and rw["magenta"] > 0 and rw["white"] > 0 and rw["yellow"] == 0 and rw["blue"] == 0 and rw["green"] == 0 and rw["black"] == 0:
        penalty += 10.0
        reasons.append("penalty: red+magenta+white failed externally")

    if rw["yellow"] > 0 and rw["red"] > 0 and rw["blue"] > 0 and rw["green"] > 0 and rw["white"] == 0 and rw["black"] == 0:
        penalty += 12.0
        reasons.append("penalty: yellow+red+blue+green no neutral failed externally")

    return penalty, "; ".join(reasons)


def disagreement_penalty(row) -> tuple[float, str]:
    try:
        d = de00_hex(row["pred_baseline"], row["pred_dualgate"])
    except Exception:
        return 0.0, ""
    if d <= 18.0:
        return 0.0, ""
    # Soft penalty, not a hard rejection.
    p = min(8.0, (d - 18.0) * 0.20)
    return p, f"penalty: baseline/dualgate disagreement {d:.1f}"


def reliability_bonus(row, observed_sets: set[str]) -> tuple[float, str]:
    bonus = 0.0
    reasons = []
    if row.get("candidate_source", "") == "observed_anchor":
        bonus += 3.0
        reasons.append("bonus: observed Trycolors anchor recipe")
    set_key = pigment_set_key(row["pigment_names"])
    if set_key in observed_sets:
        bonus += 1.25
        reasons.append("bonus: observed pigment set")
    return bonus, "; ".join(reasons)


def score_candidates_for_targets(targets: pd.DataFrame, pred_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    observed_sets = set(
        pigment_set_key(n)
        for n in pred_df.loc[pred_df["candidate_source"].eq("observed_anchor"), "pigment_names"]
    )

    rows = []
    for _, target in targets.iterrows():
        t_hex = clean_hex(target["target_hex"])
        for branch in ["m4", "m5", "m6_guarded"]:
            scored = []
            for _, cand in pred_df.iterrows():
                pred_col = "pred_m5" if branch == "m6_guarded" else f"pred_{branch}"
                pred = cand[pred_col]
                internal_d = de00_hex(pred, t_hex)
                risk, risk_reason = family_risk_penalty(cand)
                disg, disg_reason = disagreement_penalty(cand)
                bonus, bonus_reason = reliability_bonus(cand, observed_sets)

                anchor_true_hex = cand.get("anchor_true_hex", "")
                known_external_de = ""
                known_external_match = ""
                target_anchor_bonus = 0.0
                target_anchor_reason = ""
                if isinstance(anchor_true_hex, str) and anchor_true_hex.startswith("#"):
                    known_external_de = de00_hex(anchor_true_hex, t_hex)
                    known_external_match = match_pct(known_external_de)
                    if known_external_de <= 5.0:
                        target_anchor_bonus = 12.0 - known_external_de
                        target_anchor_reason = f"bonus: anchor target match dE={known_external_de:.2f}"

                guarded_score = internal_d + risk + disg - bonus - target_anchor_bonus

                # For original M4/M5, rank only by internal distance.
                ranking_score = guarded_score if branch == "m6_guarded" else internal_d
                scored.append((ranking_score, internal_d, guarded_score, risk, disg, bonus, target_anchor_bonus, risk_reason, disg_reason, bonus_reason, target_anchor_reason, known_external_de, known_external_match, cand))
            scored.sort(key=lambda x: x[0])
            for rank, (ranking_score, internal_d, guarded_score, risk, disg, bonus, target_anchor_bonus, risk_reason, disg_reason, bonus_reason, target_anchor_reason, known_external_de, known_external_match, cand) in enumerate(scored[:top_n], start=1):
                pred_col = "pred_m5" if branch == "m6_guarded" else f"pred_{branch}"
                choice_col = "m5_choice" if branch == "m6_guarded" else f"{branch}_choice"
                reason_col = "m5_reason" if branch == "m6_guarded" else f"{branch}_reason"

                anchor_true_hex = cand.get("anchor_true_hex", "")

                rows.append({
                    "target_id": target["target_id"],
                    "target_hex": t_hex,
                    "target_name": target.get("target_name", ""),
                    "target_family": target.get("family", ""),
                    "branch": branch.upper(),
                    "rank": rank,
                    "ranking_score": ranking_score,
                    "internal_dE00": internal_d,
                    "internal_match_pct": match_pct(internal_d),
                    "guarded_score": guarded_score,
                    "risk_penalty": risk,
                    "disagreement_penalty": disg,
                    "anchor_bonus": bonus,
                    "target_anchor_bonus": target_anchor_bonus,
                    "guard_reasons": " | ".join([x for x in [risk_reason, disg_reason, bonus_reason, target_anchor_reason] if x]),
                    "predicted_hex": cand[pred_col],
                    "chosen_model": cand[choice_col],
                    "routing_reason": cand[reason_col],
                    "candidate_source": cand.get("candidate_source", "grid"),
                    "anchor_case_id": cand.get("anchor_case_id", ""),
                    "anchor_true_hex": anchor_true_hex,
                    "anchor_trycolors_name": cand.get("anchor_trycolors_name", ""),
                    "known_external_dE_if_anchor": known_external_de,
                    "known_external_match_if_anchor": known_external_match,
                    "pigment_names": json.dumps(cand["pigment_names"]),
                    "pigment_hexes": json.dumps(cand["pigment_hexes"]),
                    "parts": json.dumps([int(x) if float(x).is_integer() else float(x) for x in cand["parts"]]),
                    "normalized_weights": json.dumps(cand["weights"]),
                    "trycolors_result_hex": "",
                    "trycolors_name": "",
                    "screenshot_filename": "",
                    "stability_status": "",
                    "external_dE00_trycolors_vs_target": "",
                    "external_match_trycolors_vs_target": "",
                    "notes": "",
                })
    return pd.DataFrame(rows)

def summarize_top1(top1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for branch, g in top1.groupby("branch"):
        vals = g["internal_match_pct"].astype(float).to_numpy()
        rec = {
            "branch": branch,
            "n": int(len(g)),
            "mean_internal_match": float(vals.mean()),
            "median_internal_match": float(np.median(vals)),
            "worst_internal_match": float(vals.min()),
            "count_internal_lt_85": int((vals < 85).sum()),
            "count_internal_lt_80": int((vals < 80).sum()),
        }
        # If anchors give known external proxies, summarize those rows too.
        known = g[pd.to_numeric(g["known_external_match_if_anchor"], errors="coerce").notna()]
        if len(known):
            kvals = known["known_external_match_if_anchor"].astype(float).to_numpy()
            rec.update({
                "n_known_anchor_external": int(len(known)),
                "mean_known_anchor_external_match": float(kvals.mean()),
                "worst_known_anchor_external_match": float(kvals.min()),
            })
        else:
            rec.update({
                "n_known_anchor_external": 0,
                "mean_known_anchor_external_match": "",
                "worst_known_anchor_external_match": "",
            })
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=str(ROOT / "data" / "target_colors_initial.csv"))
    parser.add_argument("--max-colors", type=int, default=4)
    parser.add_argument("--total-parts", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    palette = json.loads((ROOT / "data" / "palette_v1.json").read_text())
    targets = pd.read_csv(args.targets)

    grid = generate_candidate_recipes(palette, max_colors=args.max_colors, total_parts=args.total_parts)
    anchors = load_observed_anchors(ROOT / "data" / "observed_palette_v1_anchors.csv", palette)
    candidates = add_sources_and_anchors(grid, anchors)
    candidates.to_csv(outdir / "guarded_candidate_recipe_grid.csv", index=False)

    pred_df = compute_candidate_predictions(candidates)
    # Serialize for CSV.
    pred_csv = pred_df.copy()
    for col in ["pigment_names", "pigment_hexes", "parts", "weights"]:
        pred_csv[col] = pred_csv[col].apply(json.dumps)
    pred_csv.to_csv(outdir / "guarded_candidate_forward_predictions.csv", index=False)

    results = score_candidates_for_targets(targets, pred_df, top_n=args.top_n)
    results.to_csv(outdir / "guarded_unmix_proposed_recipes_topn.csv", index=False)

    top1 = results[results["rank"] == 1].copy()
    top1.to_csv(outdir / "guarded_unmix_capture_sheet_top1.csv", index=False)

    dedup = top1.drop_duplicates(subset=["target_id", "pigment_names", "parts", "predicted_hex"]).copy()
    dedup.to_csv(outdir / "guarded_unmix_capture_sheet_top1_deduplicated.csv", index=False)

    summary = summarize_top1(top1)
    summary.to_csv(outdir / "guarded_unmix_internal_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
