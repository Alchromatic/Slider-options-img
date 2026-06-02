#!/usr/bin/env python3
"""
M4 Balanced Router

Production-candidate routing policy over two runtime candidates:

- frozen baseline
- dual-gate sidecar

This policy does not use Trycolors result hex at prediction time.

Policy:
    default -> dual-gate

    Y/R/B only:
        if black is approximately 1.5%-2.5% and red >= 12%:
            baseline
        else:
            dual-gate

    QM + Ultramarine Blue, no white/red/black:
        if magenta >= 30%:
            baseline

    otherwise:
        dual-gate

The rule is designed to improve over M3b by preserving old 75-case benchmark
behavior while keeping the fresh-case safety profile.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple


def _norm(weights: Sequence[float]) -> List[float]:
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
    """Return (chosen_model, reason), where chosen_model is 'baseline' or 'dualgate'."""
    rw, flags = role_weights(pigment_names, weights)

    if is_yrb_only(pigment_names, weights):
        b = rw["black"]
        r = rw["red"]
        if 0.015 < b <= 0.025 and r >= 0.12:
            return "baseline", "Y/R/B mid-black warm-brown -> baseline"
        return "dualgate", "Y/R/B outside mid-black band -> dual-gate"

    if flags["qm"] and flags["ub"] and rw["white"] == 0 and rw["red"] == 0 and rw["black"] == 0:
        if rw["magenta"] >= 0.30:
            return "baseline", "QM+UB no-white -> baseline"

    return "dualgate", "default -> dual-gate"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pigment_names", required=True, help='JSON list of pigment names')
    parser.add_argument("--weights", required=True, help='JSON list of parts or normalized weights')
    args = parser.parse_args()

    names = json.loads(args.pigment_names)
    weights = json.loads(args.weights)
    print(json.dumps({"chosen_model": choose_model(names, weights)[0], "reason": choose_model(names, weights)[1]}, indent=2))
