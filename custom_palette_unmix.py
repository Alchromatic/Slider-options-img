#!/usr/bin/env python3
"""
custom_palette_unmix.py
=======================

A *new, self-contained* unmix endpoint that runs the M7.1-style unmixer
**over the user's own colors** (the Color Library / "Edit Library" palette),
instead of the fixed measured 8-pigment palette.

Why this exists
---------------
The measured models (M7 / M7.1) are accurate because they interpolate *measured*
Trycolors-UI pairwise curves that only exist for their fixed 8 pigments.  When a
user adds new colors (name + RGB) in the Color Library, those measured curves do
not exist, so the measured models cannot use them (they say "the loaded palette
above does not apply").

This module keeps the *same unmixer algorithm and associated calcs* as M7.1 --
candidate-recipe generation, CIEDE2000 scoring, ranked top-N proposals, a
confidence tier and a risk-adjusted score -- but swaps the one step that needed
measured data (mix prediction) for a physical mixing model (Kubelka-Munk /
Yule-Nielsen / linear).  That lets the unmixer work with *any* user-supplied
palette.

It is mounted into the existing FastAPI app with a single line in main.py:

    from custom_palette_unmix import router as custom_unmix_router
    app.include_router(custom_unmix_router)

and exposes:

    POST /unmix/custom   -> propose recipes for a target color from the user's palette

Nothing else in the app is changed.  The color-science functions (CIEDE2000,
OKLab, hex parsing, part compositions) are reused from the shipped
``m7_1_unified.py`` so the numbers are identical to the measured models.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

import measured_profile as mp

router = APIRouter(tags=["13. Custom palette unmix"])


# ---------------------------------------------------------------------------
# Reuse M7.1's color science so "associated calcs" match the measured models.
# We load the already-shipped module by path and reuse the same sys.modules
# entry the dispatcher uses, so it is only ever loaded once.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_M71_PATH = _ROOT / "Final-ali-py-color-mix-unmix" / "M7_1_Unified_Single_Py_Package" / "m7_1_unified.py"


def _m71():
    """Return the shared m7_1_unified module (CIEDE2000 / OKLab / helpers)."""
    mod = sys.modules.get("m7_1_unified")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("m7_1_unified", str(_M71_PATH))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load m7_1_unified from {_M71_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m7_1_unified"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Physical mixing model (Kubelka-Munk / Yule-Nielsen / linear).
# Mirrors the app's existing /unmix mixer (main.py) so results are consistent.
# ---------------------------------------------------------------------------

_KS_EPS = 1e-6


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = _clamp01(c)
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def _hex_to_rgb255(h: str) -> Tuple[int, int, int]:
    h = str(h).strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"Bad hex: {h}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_linear(h: str) -> Tuple[float, float, float]:
    r, g, b = _hex_to_rgb255(h)
    return (_srgb_to_linear(r / 255.0), _srgb_to_linear(g / 255.0), _srgb_to_linear(b / 255.0))


def _linear_to_hex(rgb: Tuple[float, float, float]) -> str:
    vals = [max(0, min(255, int(round(_linear_to_srgb(c) * 255)))) for c in rgb]
    return "#{:02X}{:02X}{:02X}".format(*vals)


def _clean_validate(h, m71) -> str:
    """Normalize to '#RRGGBB' AND verify the digits are valid hex.

    m71.clean_hex only checks length, so a string like 'nothex' would pass as
    '#NOTHEX'; _hex_to_rgb255 raises on non-hex digits, giving real validation.
    """
    hx = m71.clean_hex(h)
    _hex_to_rgb255(hx)  # raises ValueError on non-hex characters
    return hx


def _ks_from_R(R: float) -> float:
    R = max(R, _KS_EPS)
    return (1 - R) ** 2 / (2 * R)


def _R_from_ks(KS: float) -> float:
    return max(0.0, (1 + KS) - math.sqrt(KS * KS + 2 * KS))


def _mix_km(bases: List[Tuple[float, float, float]], w: List[float]) -> Tuple[float, float, float]:
    return tuple(_clamp01(_R_from_ks(sum(_ks_from_R(b[i]) * ww for b, ww in zip(bases, w)))) for i in range(3))


def _mix_linear(bases: List[Tuple[float, float, float]], w: List[float]) -> Tuple[float, float, float]:
    return tuple(_clamp01(sum(b[i] * ww for b, ww in zip(bases, w))) for i in range(3))


def _mix_hex(hexes: List[str], weights: List[float], method: str = "kubelka_munk", yn_n: float = 1.5) -> str:
    total = sum(weights)
    if total <= 1e-12:
        return hexes[0] if hexes else "#000000"
    w = [wi / total for wi in weights]
    bases = [_hex_to_linear(h) for h in hexes]
    if method == "linear":
        mixed = _mix_linear(bases, w)
    elif method == "yn_km" and yn_n > 0:
        fwd = [tuple(c ** (1.0 / yn_n) for c in b) for b in bases]
        mixed_fwd = _mix_km(fwd, w)
        mixed = tuple(_clamp01(c) ** yn_n for c in mixed_fwd)
    else:  # kubelka_munk
        mixed = _mix_km(bases, w)
    return _linear_to_hex(mixed)


# ---------------------------------------------------------------------------
# Confidence tier + risk penalty (physical-model analogue of M7.1's tiers).
# There is no measured data for custom colors, so the tier reflects how direct
# the prediction is (recipe size) and how good the match is, and the penalty
# gently prefers simpler recipes -- mirroring M7.1's "most reliable" intent.
# ---------------------------------------------------------------------------

def _tier_and_penalty(n_colors: int, de: float) -> Tuple[str, float]:
    if n_colors <= 1:
        return ("custom_single_swatch", 0.0)
    base = {2: ("custom_physical_binary", 0.5),
            3: ("custom_physical_ternary", 1.0)}.get(n_colors, ("custom_physical_nary", 1.5))
    tier, penalty = base
    if de > 12.0:
        penalty += 1.0
    return tier, penalty


# ---------------------------------------------------------------------------
# Candidate generation + scoring (the M7.1 unmix loop, over the user palette).
# ---------------------------------------------------------------------------

def _load_profile_model(palette_id: Optional[str], colors: List[Tuple[str, str]]):
    """Load a measured palette profile (if one exists) for this palette.

    Returns (model, profile) or (None, None).  The lookup uses an explicit
    ``palette_id`` when given, otherwise the deterministic id derived from this
    color set -- so a palette the admin already measured is auto-upgraded to
    measured behavior, while an unmeasured palette stays on the KM model (R7).
    """
    try:
        pid = palette_id or mp.palette_id_for([{"hex": h} for _, h in colors])
        profile = mp.load_profile(pid)
        if not profile or not profile.get("comparisons"):
            return None, None
        return mp.MeasuredProfileModel(profile), profile
    except Exception:
        return None, None


def unmix_custom_palette(
    target_color: str,
    palette: List[Dict[str, Optional[str]]],
    max_colors: int = 4,
    total_parts: int = 6,
    prefilter_top_n: int = 12,
    top_n: int = 5,
    mix_method: str = "kubelka_munk",
    palette_id: Optional[str] = None,
) -> Dict:
    """
    Propose recipes for ``target_color`` using only the user-supplied ``palette``.

    palette: list of {"hex": "#RRGGBB", "name": "optional name"}.
    Returns a dict shaped like version_dispatch.unmix() so the existing
    versioned-unmix UI can render it unchanged.

    If a measured palette profile exists for this palette (generated offline via
    profile_generator.py / TryColors pro/2025), each candidate it covers is
    predicted from that measured mixing behavior (the M7.1/M7.2 algorithm); any
    candidate it does not cover falls back to the physical Kubelka-Munk model
    (R7).  With no profile present, behavior is exactly the KM model as before.
    """
    m71 = _m71()
    method = (mix_method or "kubelka_munk").lower().strip()
    if method not in ("kubelka_munk", "yn_km", "linear"):
        method = "kubelka_munk"

    # --- validate target ---
    try:
        target_hex = _clean_validate(target_color, m71)
    except Exception:
        return {"version": "custom", "mode": "unmix", "available": True,
                "target_color": target_color, "proposals": [],
                "error": f"Invalid target color: {target_color}"}

    # --- validate + dedupe palette (keep first name per unique hex) ---
    colors: List[Tuple[str, str]] = []  # (name, hex)
    seen: set = set()
    for entry in palette or []:
        try:
            hx = _clean_validate(entry.get("hex") if isinstance(entry, dict) else entry, m71)
        except Exception:
            continue
        if hx in seen:
            continue
        seen.add(hx)
        name = (entry.get("name") if isinstance(entry, dict) else None) or hx
        colors.append((str(name), hx))

    if not colors:
        return {"version": "custom", "mode": "unmix", "available": True,
                "target_color": target_hex, "proposals": [],
                "error": "No valid palette colors provided."}

    # --- load a measured palette profile if one exists for this palette (R7) ---
    profile_model, profile = _load_profile_model(palette_id, colors)

    # --- clamp controls (bound the combinatorial search) ---
    max_colors = max(1, min(5, int(max_colors)))
    total_parts = max(2, min(12, int(total_parts)))
    top_n = max(1, min(50, int(top_n)))
    prefilter_top_n = max(0, min(60, int(prefilter_top_n)))

    # --- prefilter palette to the closest colors to the target (CIEDE2000) ---
    de_single = [(i, m71.de00_hex(hx, target_hex)) for i, (_, hx) in enumerate(colors)]
    de_single.sort(key=lambda t: t[1])
    if prefilter_top_n > 0:
        keep = [i for i, _ in de_single[:prefilter_top_n]]
    else:
        keep = [i for i, _ in de_single]
    pool = [colors[i] for i in keep]
    max_colors = min(max_colors, len(pool))

    # --- generate candidate recipes (combinations x part-compositions) ---
    n_measured = 0
    n_physical = 0
    scored: List[Dict] = []
    for k in range(1, max_colors + 1):
        for combo in itertools.combinations(range(len(pool)), k):
            parts_iter = [[1]] if k == 1 else m71.compositions(total_parts, k)
            for parts in parts_iter:
                names = [pool[idx][0] for idx in combo]
                hexes = [pool[idx][1] for idx in combo]
                weights = m71.normalize(parts)

                # Measured profile where it covers this recipe, else KM (R7).
                ids = None
                if profile_model is not None:
                    mapped = [profile_model.hex_to_id.get(h) for h in hexes]
                    if all(i is not None for i in mapped):
                        ids = mapped
                anchor_name = ""
                if ids is not None and profile_model.covers(ids, list(parts)):
                    pred_hex, meta = profile_model.predict_recipe(ids, list(parts))
                    tier = meta.get("confidence_tier", "measured")
                    penalty = mp.risk_penalty(tier)
                    anchor_name = meta.get("anchor_trycolors_name", "") or ""
                    mix_source = "measured_profile"
                    n_measured += 1
                else:
                    pred_hex = _mix_hex(hexes, list(parts), method=method)
                    de_tmp = m71.de00_hex(pred_hex, target_hex)
                    tier, penalty = _tier_and_penalty(k, de_tmp)
                    mix_source = "physical_km"
                    n_physical += 1

                de = m71.de00_hex(pred_hex, target_hex)
                scored.append({
                    "pigment_names": names,
                    "pigment_abbr": names,            # no abbreviations for user colors
                    "pigment_hexes": hexes,
                    "parts": list(parts),
                    "percentages": [round(w * 100, 1) for w in weights],
                    "predicted_hex": pred_hex,
                    "delta_e": round(de, 2),
                    "match_percentage": round(m71.match_pct(de), 1),
                    "score_with_risk_penalty": round(de + penalty, 3),
                    "risk_penalty": round(penalty, 2),
                    "confidence_tier": tier,
                    "mix_source": mix_source,
                    "anchor_trycolors_name": anchor_name,
                    "_sort": de + penalty,
                })

    scored.sort(key=lambda r: r["_sort"])
    proposals = []
    for rank, row in enumerate(scored[:top_n], start=1):
        row.pop("_sort", None)
        row["rank"] = rank
        proposals.append(row)

    if profile is not None:
        comp = profile.get("completeness", {})
        note = ("Unmix over your loaded Color Library using the M7.1/M7.2 algorithm "
                "and CIEDE2000 calcs. Candidates covered by your measured palette "
                f"profile (TryColors {profile.get('mixer_mode', 'pro')}/"
                f"{profile.get('engine', '2025')}) use that measured mixing behavior; "
                f"the rest fall back to a physical ({method}) mix model.")
        return {
            "version": "custom",
            "mode": "unmix",
            "available": True,
            "target_color": target_hex,
            "palette_mode": "custom_measured_profile",
            "mix_method": method,
            "measured_profile": {
                "palette_id": profile.get("palette_id"),
                "profile_version": profile.get("profile_version"),
                "engine": profile.get("engine"),
                "mixer_mode": profile.get("mixer_mode"),
                "present_count": comp.get("present_count"),
                "expected_count": comp.get("expected_count"),
                "complete": comp.get("complete"),
            },
            "candidate_mix_sources": {"measured_profile": n_measured, "physical_km": n_physical},
            "palette": [{"name": n, "hex": h} for n, h in colors],
            "note": note,
            "proposals": proposals,
        }

    return {
        "version": "custom",
        "mode": "unmix",
        "available": True,
        "target_color": target_hex,
        "palette_mode": "custom_user_palette",
        "mix_method": method,
        "palette": [{"name": n, "hex": h} for n, h in colors],
        "note": ("Unmix over your loaded Color Library using the M7.1 unmixer "
                 "algorithm and CIEDE2000 calcs, with a physical "
                 f"({method}) mix model in place of measured curves."),
        "proposals": proposals,
    }


# ---------------------------------------------------------------------------
# API schema + route
# ---------------------------------------------------------------------------

class CustomPaletteColor(BaseModel):
    hex: str = Field(..., description="Color hex, e.g. '#884513' (with or without '#').")
    name: Optional[str] = Field(None, description="Optional color name, e.g. 'Burnt Sienna'.")


class CustomUnmixRequest(BaseModel):
    """Target color -> proposed recipes, mixed from the user's own palette."""
    target_color: str = Field(..., description="Target hex, e.g. '#706A35'.")
    palette: List[CustomPaletteColor] = Field(
        ..., description="The user's colors (name + hex) from the Color Library.")
    max_colors: int = Field(4, ge=1, le=5, description="Max colors per recipe.")
    total_parts: int = Field(6, ge=2, le=12, description="Total parts to split across colors (ratio precision).")
    prefilter_top_n: int = Field(12, ge=0, le=60, description="Pre-filter palette to the N closest colors (0 = use all).")
    top_n: int = Field(5, ge=1, le=50, description="Number of ranked proposals to return.")
    mix_method: str = Field("kubelka_munk", description="kubelka_munk | yn_km | linear (KM fallback when no measured profile).")
    palette_id: Optional[str] = Field(
        None, description="Optional palette id of a pre-generated measured profile. "
                          "If omitted, a profile is looked up by this exact color set.")


@router.post("/unmix/custom", tags=["13. Custom palette unmix"])
def unmix_custom(req: CustomUnmixRequest):
    """
    Propose mixing recipes for a target color using **only the user's own colors**
    (the Color Library / "Edit Library" palette).

    This runs the same unmixer algorithm and CIEDE2000 calcs as the M7.1 model,
    but over your supplied palette, predicting each candidate mix with a physical
    mixing model (Kubelka-Munk by default).  Unlike the measured M7/M7.1 models,
    it works with any colors you add -- new names and RGBs included.

    **Request example:**
    ```json
    {
        "target_color": "#884513",
        "palette": [
            {"hex": "#FEE100", "name": "Cadmium Yellow Light"},
            {"hex": "#DE290C", "name": "Cadmium Red Light"},
            {"hex": "#19123F", "name": "Ultramarine Blue"},
            {"hex": "#F7F5F1", "name": "Titanium White"}
        ],
        "max_colors": 3,
        "total_parts": 6,
        "top_n": 5,
        "mix_method": "kubelka_munk"
    }
    ```

    **Response:** ranked `proposals`, each with the recipe (`pigment_names`,
    `pigment_hexes`, `parts`, `percentages`), the `predicted_hex`, `delta_e`,
    `match_percentage`, a `confidence_tier` and a risk-adjusted
    `score_with_risk_penalty`.  The shape matches `/version/unmix` so the existing
    ranked-proposals UI can render it directly.
    """
    try:
        return unmix_custom_palette(
            target_color=req.target_color,
            palette=[c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in req.palette],
            max_colors=req.max_colors,
            total_parts=req.total_parts,
            prefilter_top_n=req.prefilter_top_n,
            top_n=req.top_n,
            mix_method=req.mix_method,
            palette_id=req.palette_id,
        )
    except Exception as e:  # pragma: no cover - defensive
        return {"version": "custom", "mode": "unmix", "available": True,
                "target_color": req.target_color, "proposals": [], "error": str(e)}
