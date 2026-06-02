#!/usr/bin/env python3
"""
M5 Candidate Router (Exploratory)

This branch tests five improvement ideas around M4:
1. baseline-bias-corrected baseline as an auxiliary candidate,
2. pair-spline candidate scaffold,
3. Mixbox-guided capture selection,
4. narrow QM+UB exception,
5. local Y/R/B correction.

The production candidate remains M4 unless M5 is validated on a frozen holdout.
"""
from __future__ import annotations

import json
from typing import Dict, Sequence, Tuple


def _norm(weights: Sequence[float]):
    vals = [float(x) for x in weights]
    s = sum(vals)
    if s <= 0:
        raise ValueError("weights sum to zero")
    return [x / s for x in vals]


def role_weights(pigment_names: Sequence[str], weights: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    weights = _norm(weights)
    rw = {k: 0.0 for k in ["yellow", "red", "blue", "green", "magenta", "white", "black", "other"]}
    flags = {k: False for k in ["india_yellow", "cad_yellow", "cad_red", "qm", "ub", "phg", "white", "black"]}

    for name, w in zip(pigment_names, weights):
        n = str(name).lower()
        if "india yellow" in n:
            rw["yellow"] += w
            flags["india_yellow"] = True
        elif "yellow" in n:
            rw["yellow"] += w
            flags["cad_yellow"] = True
        elif "red" in n:
            rw["red"] += w
            flags["cad_red"] = True
        elif "ultramarine" in n or "blue" in n:
            rw["blue"] += w
            flags["ub"] = True
        elif "phthalo green" in n or "green" in n or "killarney" in n:
            rw["green"] += w
            flags["phg"] = True
        elif "quinacridone" in n or "magenta" in n or "wisteria" in n or "contessa" in n:
            rw["magenta"] += w
            flags["qm"] = True
        elif "white" in n or "bon jour" in n:
            rw["white"] += w
            flags["white"] = True
        elif "black" in n:
            rw["black"] += w
            flags["black"] = True
        else:
            rw["other"] += w

    return rw, flags


def is_yrb_only(pigment_names: Sequence[str], weights: Sequence[float]) -> bool:
    rw, _ = role_weights(pigment_names, weights)
    return (
        rw["yellow"] >= 0.70
        and rw["red"] > 0
        and rw["black"] > 0
        and rw["blue"] == 0
        and rw["green"] == 0
        and rw["magenta"] == 0
        and rw["white"] == 0
    )


def choose_model(pigment_names: Sequence[str], weights: Sequence[float]) -> Tuple[str, str]:
    """Return (candidate, reason). Candidate is baseline, dualgate, or p3."""
    rw, flags = role_weights(pigment_names, weights)

    # Local Y/R/B correction/routing surface.
    if is_yrb_only(pigment_names, weights):
        b = rw["black"]
        r = rw["red"]

        if b < 0.015:
            return "dualgate", "Y/R/B low-black -> dual-gate"

        if 0.015 < b <= 0.025:
            if r >= 0.12:
                return "baseline", "Y/R/B mid-black red>=12 -> baseline"
            return "dualgate", "Y/R/B mid-black low-red -> dual-gate"

        if b > 0.025 and r >= 0.20:
            return "p3", "Y/R/B high-black high-red -> P3"

        return "dualgate", "Y/R/B high-black default -> dual-gate"

    # QM+UB no-white exception from M4.
    if flags["qm"] and flags["ub"] and rw["white"] == 0 and rw["red"] == 0 and rw["black"] == 0 and rw["yellow"] == 0 and rw["green"] == 0:
        if rw["magenta"] >= 0.30:
            return "baseline", "QM+UB no-white magenta>=30 -> baseline"

    # Narrow QM+UB+White exception.
    if flags["qm"] and flags["ub"] and rw["white"] > 0 and rw["red"] == 0 and rw["black"] == 0 and rw["yellow"] == 0 and rw["green"] == 0:
        if 0.20 <= rw["white"] <= 0.40 and rw["magenta"] >= 0.25 and rw["blue"] >= 0.10:
            return "p3", "QM+UB+white mid-white purple -> P3"

    return "dualgate", "default -> dual-gate"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pigment_names", required=True, help="JSON list of pigment names")
    parser.add_argument("--weights", required=True, help="JSON list of parts or weights")
    args = parser.parse_args()
    names = json.loads(args.pigment_names)
    weights = json.loads(args.weights)
    chosen, reason = choose_model(names, weights)
    print(json.dumps({"chosen_model": chosen, "reason": reason}, indent=2))
