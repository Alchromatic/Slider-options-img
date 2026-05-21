"""
M4 Balanced Router – Art-slider colour matching.

Mount on the existing FastAPI app with:
    from m4_balanced_router import router as m4_router
    app.include_router(m4_router)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
from itertools import combinations, product
from collections import defaultdict

from fastapi import APIRouter
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/m4",
    tags=["11. Balanced Color Router"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PaletteColor(BaseModel):
    hex: str
    name: Optional[str] = None


class SliderWeights(BaseModel):
    """
    Three sliders the artist can drag (0-100 each).
    They are normalised internally so only the *ratios* matter.

    perceptual  – favour CIE-LAB Delta-E accuracy (closest visual match)
    pigment     – favour Kubelka-Munk subtractive mixing (real-paint behaviour)
    harmony     – favour hue-wheel proximity / analogous relationships
    """
    perceptual: float = Field(default=50, ge=0, le=100)
    pigment: float = Field(default=30, ge=0, le=100)
    harmony: float = Field(default=20, ge=0, le=100)


class BalancedMatchRequest(BaseModel):
    target_color: str
    palette: Optional[List[PaletteColor]] = None
    sliders: Optional[SliderWeights] = None
    max_results: Optional[int] = 10
    threshold: Optional[float] = 0.0


class BalancedMatchResult(BaseModel):
    hex: str
    name: Optional[str] = None
    score: float
    match_percentage: int
    delta_e: float
    pigment_score: float
    harmony_score: float


class BalancedMatchResponse(BaseModel):
    target_color: str
    slider_weights: SliderWeights
    matches: List[BalancedMatchResult]
    best_match: Optional[BalancedMatchResult] = None
    has_matches: bool
    error: Optional[str] = None


class BalancedMixRequest(BaseModel):
    target_color: str
    palette: Optional[List[PaletteColor]] = None
    sliders: Optional[SliderWeights] = None
    max_colors: Optional[int] = 3
    max_parts: Optional[int] = 10
    top_k: Optional[int] = 5


class RecipeComponent(BaseModel):
    hex: str
    name: Optional[str] = None
    parts: int
    percentage: float


class BalancedMixResponse(BaseModel):
    target_color: str
    slider_weights: SliderWeights
    recipe: List[RecipeComponent]
    result_color: str
    combined_score: float
    match_percentage: float
    delta_e: float
    total_parts: int
    mix_method: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Default palette (same 20 paints as main.py)
# ---------------------------------------------------------------------------

DEFAULT_PAINT_PALETTE = [
    {"hex": "#FEE100", "name": "Cadmium Yellow Light"},
    {"hex": "#FFB800", "name": "Cadmium Yellow Medium"},
    {"hex": "#FF8C00", "name": "Cadmium Orange"},
    {"hex": "#DE290C", "name": "Cadmium Red Light"},
    {"hex": "#B01B0F", "name": "Cadmium Red Medium"},
    {"hex": "#8B0000", "name": "Alizarin Crimson"},
    {"hex": "#19123F", "name": "Ultramarine Blue"},
    {"hex": "#1E3A8A", "name": "Cobalt Blue"},
    {"hex": "#00CED1", "name": "Cerulean Blue"},
    {"hex": "#006400", "name": "Viridian Green"},
    {"hex": "#228B22", "name": "Sap Green"},
    {"hex": "#32CD32", "name": "Permanent Green Light"},
    {"hex": "#8B4513", "name": "Burnt Sienna"},
    {"hex": "#654321", "name": "Burnt Umber"},
    {"hex": "#D2691E", "name": "Raw Sienna"},
    {"hex": "#A0522D", "name": "Raw Umber"},
    {"hex": "#FFD700", "name": "Yellow Ochre"},
    {"hex": "#F7F5F1", "name": "Titanium White"},
    {"hex": "#1A1A1A", "name": "Ivory Black"},
    {"hex": "#2F4F4F", "name": "Payne's Gray"},
]

# ---------------------------------------------------------------------------
# Colour-space helpers (self-contained, no dependency on main.py)
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = _clamp01(c)
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055


def _hex_to_rgb255(h: str) -> Tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"Bad hex: {h}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb255_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
    )


def _normalize_hex(h: str) -> str:
    h = h.strip()
    if not h.startswith("#"):
        h = "#" + h
    return h.upper()


def _hex_to_linear(h: str) -> Tuple[float, float, float]:
    r, g, b = _hex_to_rgb255(h)
    return (_srgb_to_linear(r / 255.0),
            _srgb_to_linear(g / 255.0),
            _srgb_to_linear(b / 255.0))


def _linear_to_hex(rgb: Tuple[float, float, float]) -> str:
    r = int(round(_clamp01(_linear_to_srgb(rgb[0])) * 255))
    g = int(round(_clamp01(_linear_to_srgb(rgb[1])) * 255))
    b = int(round(_clamp01(_linear_to_srgb(rgb[2])) * 255))
    return _rgb255_to_hex(r, g, b)


def _rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    rl = _srgb_to_linear(r / 255.0)
    gl = _srgb_to_linear(g / 255.0)
    bl = _srgb_to_linear(b / 255.0)
    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
    xr, yr, zr = 0.95047, 1.0, 1.08883
    x /= xr; y /= yr; z /= zr

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (903.3 * t + 16) / 116

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Return (H 0-360, S 0-1, L 0-1)."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rf, gf, bf), min(rf, gf, bf)
    l = (mx + mn) / 2.0
    if mx == mn:
        return (0.0, 0.0, l)
    d = mx - mn
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == rf:
        h = (gf - bf) / d + (6.0 if gf < bf else 0.0)
    elif mx == gf:
        h = (bf - rf) / d + 2.0
    else:
        h = (rf - gf) / d + 4.0
    return (h * 60.0, s, l)


def _delta_e(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def _delta_e_to_pct(de: float) -> float:
    if de <= 0:
        return 100.0
    return max(0.0, min(100.0, 100.0 * math.exp(-0.046 * de)))


# ---------------------------------------------------------------------------
# Kubelka-Munk mixing (self-contained)
# ---------------------------------------------------------------------------

_KS_EPS = 1e-6

def _ks(R: float) -> float:
    R = max(R, _KS_EPS)
    return (1 - R) ** 2 / (2 * R)

def _R(ks: float) -> float:
    return max(0.0, (1 + ks) - math.sqrt(ks ** 2 + 2 * ks))

def _mix_km(bases: List[Tuple[float, float, float]], weights: List[float]) -> Tuple[float, float, float]:
    total = sum(weights)
    if total <= 1e-12:
        return bases[0] if bases else (0.5, 0.5, 0.5)
    w = [wi / total for wi in weights]
    def ch(ci):
        return _clamp01(_R(sum(_ks(b[ci]) * ww for b, ww in zip(bases, w))))
    return (ch(0), ch(1), ch(2))

def _mix_linear(bases: List[Tuple[float, float, float]], weights: List[float]) -> Tuple[float, float, float]:
    total = sum(weights)
    if total <= 1e-12:
        return bases[0] if bases else (0.5, 0.5, 0.5)
    w = [wi / total for wi in weights]
    return tuple(_clamp01(sum(b[i] * ww for b, ww in zip(bases, w))) for i in range(3))

def _mix_hex(hexes: List[str], weights: List[float], method: str = "kubelka_munk") -> str:
    bases = [_hex_to_linear(h) for h in hexes]
    mixed = _mix_km(bases, weights) if method == "kubelka_munk" else _mix_linear(bases, weights)
    return _linear_to_hex(mixed)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _perceptual_score(target_lab, candidate_lab) -> float:
    """0-100, higher is better (inverse of Delta-E)."""
    de = _delta_e(target_lab, candidate_lab)
    return _delta_e_to_pct(de)


def _pigment_score(target_hex: str, candidate_hex: str) -> float:
    """
    How well the candidate could *participate* in a KM mix toward the target.
    Measures the reflectance-channel alignment in linear RGB.
    Returns 0-100.
    """
    t = _hex_to_linear(target_hex)
    c = _hex_to_linear(candidate_hex)
    diffs = [abs(t[i] - c[i]) for i in range(3)]
    avg_diff = sum(diffs) / 3.0
    return max(0.0, 100.0 * (1.0 - avg_diff))


def _harmony_score(target_hsl, candidate_hsl) -> float:
    """
    Hue-wheel proximity score (analogous colours score highest).
    Returns 0-100.
    """
    dh = abs(target_hsl[0] - candidate_hsl[0])
    if dh > 180:
        dh = 360 - dh
    if dh <= 30:
        return 100.0
    if dh <= 60:
        return 80.0 + (60 - dh) * (20.0 / 30.0)
    if dh <= 120:
        return 40.0 + (120 - dh) * (40.0 / 60.0)
    return max(0.0, 40.0 * (1.0 - (dh - 120) / 60.0))


def _combined_score(
    perceptual: float,
    pigment: float,
    harmony: float,
    weights: SliderWeights,
) -> float:
    ws = weights.perceptual + weights.pigment + weights.harmony
    if ws <= 0:
        ws = 1.0
    wp = weights.perceptual / ws
    wm = weights.pigment / ws
    wh = weights.harmony / ws
    return wp * perceptual + wm * pigment + wh * harmony


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _resolve_palette(palette: Optional[List[PaletteColor]]) -> List[PaletteColor]:
    if palette:
        out = []
        for p in palette:
            try:
                n = _normalize_hex(p.hex)
                _hex_to_rgb255(n)
                out.append(PaletteColor(hex=n, name=p.name))
            except Exception:
                continue
        if out:
            return out
    return [PaletteColor(hex=p["hex"], name=p["name"]) for p in DEFAULT_PAINT_PALETTE]


@router.post("/match", response_model=BalancedMatchResponse)
def balanced_match(req: BalancedMatchRequest):
    """
    Art-slider colour match.

    Rank every palette colour against the target using three weighted criteria
    controlled by the **sliders** object:

    | slider       | what it favours |
    |--------------|-----------------|
    | `perceptual` | CIE-LAB Delta-E closeness (visual accuracy) |
    | `pigment`    | Linear-RGB reflectance alignment (paint mixing potential) |
    | `harmony`    | Hue-wheel proximity (analogous / complementary feel) |

    Each slider runs 0-100; only the *ratios* matter.
    Default: perceptual 50 / pigment 30 / harmony 20.
    """
    try:
        target_hex = _normalize_hex(req.target_color)
        target_rgb = _hex_to_rgb255(target_hex)
    except Exception:
        return BalancedMatchResponse(
            target_color=req.target_color,
            slider_weights=req.sliders or SliderWeights(),
            matches=[], best_match=None, has_matches=False,
            error=f"Invalid target color: {req.target_color}",
        )

    sliders = req.sliders or SliderWeights()
    palette = _resolve_palette(req.palette)
    target_lab = _rgb_to_lab(*target_rgb)
    target_hsl = _rgb_to_hsl(*target_rgb)

    results: List[BalancedMatchResult] = []
    for pc in palette:
        try:
            c_rgb = _hex_to_rgb255(pc.hex)
            c_lab = _rgb_to_lab(*c_rgb)
            c_hsl = _rgb_to_hsl(*c_rgb)

            ps = _perceptual_score(target_lab, c_lab)
            ms = _pigment_score(target_hex, pc.hex)
            hs = _harmony_score(target_hsl, c_hsl)
            cs = _combined_score(ps, ms, hs, sliders)
            de = _delta_e(target_lab, c_lab)

            if cs >= req.threshold:
                results.append(BalancedMatchResult(
                    hex=pc.hex,
                    name=pc.name,
                    score=round(cs, 2),
                    match_percentage=int(round(ps)),
                    delta_e=round(de, 2),
                    pigment_score=round(ms, 2),
                    harmony_score=round(hs, 2),
                ))
        except Exception:
            continue

    results.sort(key=lambda r: r.score, reverse=True)
    if req.max_results and req.max_results > 0:
        results = results[:req.max_results]

    return BalancedMatchResponse(
        target_color=target_hex,
        slider_weights=sliders,
        matches=results,
        best_match=results[0] if results else None,
        has_matches=bool(results),
    )


@router.post("/mix", response_model=BalancedMixResponse)
def balanced_mix(req: BalancedMixRequest):
    """
    Art-slider colour unmix / recipe finder.

    Works like `/unmix` but the *best recipe* is chosen by the combined
    slider score rather than raw Delta-E alone.  This lets the artist
    bias toward pigment-realistic or harmonious recipes.
    """
    try:
        target_hex = _normalize_hex(req.target_color)
        target_rgb = _hex_to_rgb255(target_hex)
    except Exception:
        return BalancedMixResponse(
            target_color=req.target_color,
            slider_weights=req.sliders or SliderWeights(),
            recipe=[], result_color=req.target_color,
            combined_score=0, match_percentage=0, delta_e=100,
            total_parts=0,
            error=f"Invalid target color: {req.target_color}",
        )

    sliders = req.sliders or SliderWeights()
    palette = _resolve_palette(req.palette)
    max_colors = max(1, min(5, req.max_colors or 3))
    max_parts = max(1, min(20, req.max_parts or 10))
    top_k = max(1, min(20, req.top_k or 5))

    target_lab = _rgb_to_lab(*target_rgb)
    target_hsl = _rgb_to_hsl(*target_rgb)
    palette_hexes = [pc.hex for pc in palette]
    n = len(palette_hexes)

    if n == 0:
        return BalancedMixResponse(
            target_color=target_hex,
            slider_weights=sliders,
            recipe=[], result_color=target_hex,
            combined_score=0, match_percentage=0, delta_e=100,
            total_parts=0, error="No valid palette colours",
        )

    palette_labs = [_rgb_to_lab(*_hex_to_rgb255(h)) for h in palette_hexes]

    # Rank palette by perceptual distance for pre-filtering
    dists = sorted(
        [(i, _delta_e(target_lab, lab)) for i, lab in enumerate(palette_labs)],
        key=lambda x: x[1],
    )
    top_indices = [i for i, _ in dists[:min(12, n)]]

    best_score = -1.0
    best_recipe = []
    best_hex = target_hex
    best_de = 100.0
    best_method = "kubelka_munk"

    def _evaluate(recipe_indices_parts, method):
        nonlocal best_score, best_recipe, best_hex, best_de, best_method
        indices, parts = zip(*recipe_indices_parts)
        colors = [palette_hexes[i] for i in indices]
        mixed = _mix_hex(colors, list(parts), method=method)
        m_rgb = _hex_to_rgb255(mixed)
        m_lab = _rgb_to_lab(*m_rgb)
        m_hsl = _rgb_to_hsl(*m_rgb)

        ps = _perceptual_score(target_lab, m_lab)
        ms = _pigment_score(target_hex, mixed)
        hs = _harmony_score(target_hsl, m_hsl)
        cs = _combined_score(ps, ms, hs, sliders)
        de = _delta_e(target_lab, m_lab)

        if cs > best_score:
            best_score = cs
            best_recipe = list(recipe_indices_parts)
            best_hex = mixed
            best_de = de
            best_method = method

    # 1-colour (exact match)
    for i in top_indices:
        for method in ("kubelka_munk", "linear"):
            try:
                _evaluate([(i, max_parts)], method)
            except Exception:
                pass

    # 2-colour combinations
    coarse = list(range(1, min(max_parts + 1, 6)))
    for method in ("kubelka_munk", "linear"):
        for i, j in combinations(top_indices, 2):
            for p1, p2 in product(coarse, repeat=2):
                try:
                    _evaluate([(i, p1), (j, p2)], method)
                except Exception:
                    pass
            if best_score >= 95:
                break
        if best_score >= 95:
            break

    # 3-colour if needed
    if max_colors >= 3 and best_score < 80:
        top8 = top_indices[:8]
        coarse3 = [1, 2, 3, 4]
        for method in ("kubelka_munk", "linear"):
            for idxs in combinations(top8, 3):
                for parts in product(coarse3, repeat=3):
                    try:
                        _evaluate(list(zip(idxs, parts)), method)
                    except Exception:
                        pass
                if best_score >= 90:
                    break
            if best_score >= 90:
                break

    # Build response
    combined = defaultdict(int)
    for idx, parts in best_recipe:
        combined[idx] += parts
    parts_list = list(combined.values())
    if parts_list:
        g = parts_list[0]
        for p in parts_list[1:]:
            g = math.gcd(g, p)
        if g > 1:
            combined = {k: v // g for k, v in combined.items()}

    total_parts = sum(combined.values())
    recipe_out = []
    for idx, parts in combined.items():
        pct = (parts / total_parts * 100) if total_parts else 0
        recipe_out.append(RecipeComponent(
            hex=palette[idx].hex,
            name=palette[idx].name,
            parts=parts,
            percentage=round(pct, 1),
        ))
    recipe_out.sort(key=lambda r: r.parts, reverse=True)

    match_pct = _delta_e_to_pct(best_de)

    return BalancedMixResponse(
        target_color=target_hex,
        slider_weights=sliders,
        recipe=recipe_out,
        result_color=best_hex,
        combined_score=round(best_score, 2),
        match_percentage=round(match_pct, 1),
        delta_e=round(best_de, 2),
        total_parts=total_parts,
        mix_method=best_method,
    )


@router.get("/sliders/presets")
def slider_presets():
    """
    Return suggested slider presets for common art workflows.
    """
    return {
        "presets": [
            {
                "name": "Accurate Match",
                "description": "Prioritise visual accuracy above all else",
                "sliders": {"perceptual": 90, "pigment": 5, "harmony": 5},
            },
            {
                "name": "Paint Realism",
                "description": "Favour recipes that behave like real paint on canvas",
                "sliders": {"perceptual": 30, "pigment": 60, "harmony": 10},
            },
            {
                "name": "Harmonious Palette",
                "description": "Keep the recipe within an analogous colour family",
                "sliders": {"perceptual": 25, "pigment": 15, "harmony": 60},
            },
            {
                "name": "Balanced (default)",
                "description": "Even blend of accuracy, realism, and harmony",
                "sliders": {"perceptual": 50, "pigment": 30, "harmony": 20},
            },
        ]
    }
