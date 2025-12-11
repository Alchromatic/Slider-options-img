from typing import List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import Counter
import math
import json

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # or your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ShapeModel(BaseModel):
    type: int
    data: List[float]
    color: List[int]
    score: float


class Logic(str, Enum):
    exterior_to_center = "exterior_to_center"
    center_to_exterior = "center_to_exterior"
    color_sequence = "color_sequence"
    custom_sequence = "custom_sequence"
    light_to_dark = "light_to_dark"
    dark_to_light = "dark_to_light"
    frequency_by_color = "frequency_by_color"
    frequency_by_color_reverse = "frequency_by_color_reverse"
    top_to_bottom = "top_to_bottom"
    bottom_to_top = "bottom_to_top"


class RegionModel(BaseModel):
    x: float
    y: float
    width: float
    height: float


class OrderRequest(BaseModel):
    shapes: List[ShapeModel]
    logic: Logic
    limit: Optional[int] = None
    color_order: Optional[List[List[int]]] = None


class SelectiveResolutionRequest(BaseModel):
    base_shapes: List[ShapeModel]  # Low res upload (example 300 shapes)
    detail_shapes: List[ShapeModel]  # High res upload (e.g., 3000 shapes)
    regions: List[RegionModel]  # Areas to enhance


class OrderResponse(BaseModel):
    shapes: List[ShapeModel]
    total_shapes: int

# ---------- Color matching ----------

class ColorMatchRequest(BaseModel):
    selected_color: str  # Hex color to compare (e.g., "#FF5733")
    palette_colors: List[str]  # List of hex colors to compare against
    threshold: Optional[float] = 0.0  # Minimum match % to include (default: show all)


class ColorMatchResult(BaseModel):
    hex: str
    match_percentage: int
    delta_e: float  # CIE Delta E value for reference


class ColorMatchResponse(BaseModel):
    selected_color: str
    matches: List[ColorMatchResult]
    best_match: Optional[ColorMatchResult]
    has_matches: bool
    error: Optional[str] = None


# ---------- Unmix / Get Recipe (like trycolors) ----------

class PaletteColor(BaseModel):
    hex: str
    name: Optional[str] = None  # Optional paint name like "Cadmium Red"


class UnmixRequest(BaseModel):
    target_color: str  # The color you want to achieve
    palette: List[PaletteColor]  # Available paints to mix from
    max_colors: Optional[int] = 3  # Max colors in the recipe (1-5)
    max_parts: Optional[int] = 10  # Max parts per color for ratio precision


class RecipeComponent(BaseModel):
    hex: str
    name: Optional[str] = None
    parts: int  # Number of parts (e.g., 3 parts of this color)
    percentage: float  # Percentage of total mix


class UnmixResponse(BaseModel):
    target_color: str
    recipe: List[RecipeComponent]
    result_color: str  # The actual color you'll get from mixing
    match_percentage: float  # How close the result is to target (0-100)
    delta_e: float  # CIE Delta E for reference
    total_parts: int
    mix_method: Optional[str] = None  # 'kubelka_munk', 'linear', or 'exact'
    error: Optional[str] = None


# --------- Color Utilities -------------

def clamp01(x: float) -> float:
    # Clamp value to 0-1 range
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def srgb_to_linear(c: float) -> float:
    # Convert sRGB component to linear RGB
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def hex_to_rgb255(h: str) -> Tuple[int, int, int]:
    # Convert hex color to RGB (0-255) - from color.py
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"Bad hex: {h}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb255_to_hex(r: int, g: int, b: int) -> str:
    # Convert RGB (0-255) to hex color - from color.py
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def hex_to_linear_rgb(h: str) -> Tuple[float, float, float]:
    # Convert hex color to linear RGB (0.0-1.0) - from color.py
    r8, g8, b8 = hex_to_rgb255(h)
    return (
        srgb_to_linear(r8 / 255.0),
        srgb_to_linear(g8 / 255.0),
        srgb_to_linear(b8 / 255.0),
    )


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB (0-255) to CIE LAB color space.
    This provides perceptually uniform color distance calculations.
    Uses the same linearization approach as color.py's srgb_to_linear.
    """

    # Normalize RGB to 0-1 and convert to linear
    r_lin = srgb_to_linear(r / 255.0)
    g_lin = srgb_to_linear(g / 255.0)
    b_lin = srgb_to_linear(b / 255.0)
    
    # Convert to XYZ (D65 illuminant)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    
    # Reference white D65
    x_ref, y_ref, z_ref = 0.95047, 1.0, 1.08883
    
    x = x / x_ref
    y = y / y_ref
    z = z / z_ref
    
    # Convert to LAB
    def f(t):
        if t > 0.008856:
            return t ** (1/3)
        return (903.3 * t + 16) / 116
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    
    return (L, a, b_val)


def delta_e_cie76(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """
    Calculate CIE76 Delta E (color difference).
    Similar to rmse_hex in color.py but using LAB for perceptual accuracy.
    Values: 0 = identical, ~2.3 = just noticeable difference, >100 = very different
    """
    return math.sqrt(
        (lab1[0] - lab2[0]) ** 2 +
        (lab1[1] - lab2[1]) ** 2 +
        (lab1[2] - lab2[2]) ** 2
    )


def delta_e_to_match_percentage(delta_e: float) -> float:
    """
    Convert Delta E to a match percentage (0-100%).
    
    Delta E interpretation:
    - 0-1: Not perceptible by human eyes
    - 1-2: Perceptible through close observation
    - 2-10: Perceptible at a glance
    - 11-49: Colors are more similar than opposite
    - 100+: Colors are exact opposites
    
    We use an exponential decay function to map Delta E to percentage:
    - Delta E = 0 → 100%
    - Delta E = 2.3 (JND) → ~90%
    - Delta E = 10 → ~60%
    - Delta E = 50 → ~10%
    - Delta E = 100+ → ~0%
    """
    if delta_e <= 0:
        return 100.0
    
    # Exponential decay: match% = 100 * e^(-k * deltaE)
    # k chosen so that delta_e=50 gives ~10% match
    k = 0.046
    match = 100.0 * math.exp(-k * delta_e)
    
    return max(0.0, min(100.0, match))


def calculate_color_match(hex1: str, hex2: str) -> Tuple[float, float]:
    """
    Calculate match percentage and Delta E between two hex colors.
    Returns (match_percentage, delta_e)
    """
    try:
        rgb1 = hex_to_rgb255(hex1)
        rgb2 = hex_to_rgb255(hex2)
        
        lab1 = rgb_to_lab(*rgb1)
        lab2 = rgb_to_lab(*rgb2)
        
        delta_e = delta_e_cie76(lab1, lab2)
        match_pct = delta_e_to_match_percentage(delta_e)
        
        return (match_pct, delta_e)
    except Exception as e:
        raise ValueError(f"Error calculating color match: {e}")


def normalize_hex(hex_color: str) -> str:
    """Normalize hex color to uppercase with # prefix"""
    hex_color = hex_color.strip()
    if not hex_color.startswith("#"):
        hex_color = "#" + hex_color
    return hex_color.upper()


# --------- Color Mixing Utilities (Kubelka-Munk) -------------

def linear_to_srgb(c: float) -> float:
    """Convert linear RGB to sRGB"""
    c = clamp01(c)
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055


def linear_rgb_to_hex(rgb_lin: Tuple[float, float, float]) -> str:
    """Convert linear RGB (0-1) to hex color"""
    r_lin, g_lin, b_lin = rgb_lin
    r = int(round(clamp01(linear_to_srgb(r_lin)) * 255))
    g = int(round(clamp01(linear_to_srgb(g_lin)) * 255))
    b = int(round(clamp01(linear_to_srgb(b_lin)) * 255))
    return rgb255_to_hex(r, g, b)


KS_EPS: float = 1e-6

def _ks_from_R(R: float, eps: float = KS_EPS) -> float:
    """Convert reflectance to K/S ratio (Kubelka-Munk)"""
    R = max(R, eps)
    return (1 - R) ** 2 / (2 * R)


def _R_from_ks(KS: float) -> float:
    """Convert K/S ratio back to reflectance"""
    return max(0.0, (1 + KS) - math.sqrt(KS**2 + 2*KS))


def mix_kubelka_munk(bases: List[Tuple[float, float, float]], weights: List[float]) -> Tuple[float, float, float]:
    """
    Mix colors using Kubelka-Munk theory (subtractive mixing like real paint).
    bases: list of linear RGB tuples (0-1 range)
    weights: list of weights (will be normalized)
    """
    # Normalize weights
    total = sum(weights)
    if total <= 1e-12:
        return bases[0] if bases else (0.5, 0.5, 0.5)
    w = [wi / total for wi in weights]
    
    def mix_channel(channel_idx: int) -> float:
        KS = 0.0
        for base, weight in zip(bases, w):
            KS += _ks_from_R(base[channel_idx]) * weight
        return clamp01(_R_from_ks(KS))
    
    return (mix_channel(0), mix_channel(1), mix_channel(2))


def mix_kubelka_munk_yn(bases: List[Tuple[float, float, float]], weights: List[float], n: float = 1.5) -> Tuple[float, float, float]:
    """
    Mix colors using Yule-Nielsen modified Kubelka-Munk.
    The n parameter (typically 1.0-2.0) adjusts for ink/paint layering effects.
    n=1.0 is equivalent to standard KM; n=1.5-2.0 is more realistic for many paints.
    """
    if n <= 0:
        return mix_kubelka_munk(bases, weights)
    
    def yn_fwd(R: float) -> float:
        return R ** (1.0 / n)
    
    def yn_inv(Rp: float) -> float:
        return clamp01(Rp) ** n
    
    # Transform bases through YN forward
    bases_yn = [(yn_fwd(r), yn_fwd(g), yn_fwd(b)) for (r, g, b) in bases]
    
    # Mix in transformed space using KM
    mix_yn = mix_kubelka_munk(bases_yn, weights)
    
    # Transform back
    return (yn_inv(mix_yn[0]), yn_inv(mix_yn[1]), yn_inv(mix_yn[2]))


def mix_linear_rgb(bases: List[Tuple[float, float, float]], weights: List[float]) -> Tuple[float, float, float]:
    """
    Mix colors using simple weighted average in linear RGB space.
    This is additive mixing (like light mixing).
    """
    total = sum(weights)
    if total <= 1e-12:
        return bases[0] if bases else (0.5, 0.5, 0.5)
    w = [wi / total for wi in weights]
    
    r = sum(base[0] * weight for base, weight in zip(bases, w))
    g = sum(base[1] * weight for base, weight in zip(bases, w))
    b = sum(base[2] * weight for base, weight in zip(bases, w))
    
    return (clamp01(r), clamp01(g), clamp01(b))


def mix_colors_hex(hex_colors: List[str], weights: List[float], method: str = "kubelka_munk", yn_n: float = 1.5) -> str:
    """
    Mix hex colors and return result as hex. 
    Methods: 'kubelka_munk', 'yn_km' (Yule-Nielsen KM), or 'linear'
    """
    bases = [hex_to_linear_rgb(h) for h in hex_colors]
    if method == "linear":
        mixed_lin = mix_linear_rgb(bases, weights)
    elif method == "yn_km":
        mixed_lin = mix_kubelka_munk_yn(bases, weights, n=yn_n)
    else:  # kubelka_munk
        mixed_lin = mix_kubelka_munk(bases, weights)
    return linear_rgb_to_hex(mixed_lin)


# --------- Unmix Algorithm (Find Recipe) -------------

def find_best_recipe(
    target_hex: str,
    palette: List[PaletteColor],
    max_colors: int = 3,
    max_parts: int = 10
) -> Tuple[List[Tuple[int, int]], str, float, float, str]:
    """
    Find the best combination of palette colors to achieve target color.
    
    Uses optimized search:
    1. Pre-filter palette to most promising colors
    2. Use coarse-to-fine search for ratios
    3. Early termination when good match found
    
    Returns: (recipe as [(palette_idx, parts), ...], result_hex, match_pct, delta_e, mix_method)
    """
    from itertools import combinations, product
    
    target_rgb = hex_to_rgb255(target_hex)
    target_lab = rgb_to_lab(*target_rgb)
    
    best_recipe = []
    best_result_hex = target_hex
    best_delta_e = float('inf')
    best_method = "kubelka_munk"
    
    palette_hexes = [normalize_hex(p.hex) for p in palette]
    n_palette = len(palette_hexes)
    
    if n_palette == 0:
        return [], target_hex, 0.0, 100.0, "none"
    
    # Pre-compute LAB values for all palette colors
    palette_labs = []
    for hex_color in palette_hexes:
        rgb = hex_to_rgb255(hex_color)
        palette_labs.append(rgb_to_lab(*rgb))
    
    # Step 1: Find single color matches and rank palette by proximity
    color_distances = []
    for i, lab in enumerate(palette_labs):
        de = delta_e_cie76(target_lab, lab)
        color_distances.append((i, de))
        if de < best_delta_e:
            best_delta_e = de
            best_recipe = [(i, max_parts)]
            best_result_hex = palette_hexes[i]
    
    # If exact match found, return early
    if best_delta_e < 1.0:
        return best_recipe, best_result_hex, delta_e_to_match_percentage(best_delta_e), best_delta_e, "exact"
    
    # Sort by distance - focus on closest colors
    color_distances.sort(key=lambda x: x[1])
    
    # Take top N most promising colors (limit search space)
    top_n = min(12, n_palette)  # Use at most 12 closest colors
    promising_indices = [idx for idx, _ in color_distances[:top_n]]
    
    # Mixing methods to try
    mixing_methods = ["kubelka_munk", "yn_km", "linear"]
    
    # Step 2: Coarse search with fewer parts first
    coarse_parts = list(range(1, min(max_parts + 1, 6)))  # 1-5 for coarse
    
    # Try 2-color combinations (most common case)
    for method in mixing_methods:
        for i, j in combinations(promising_indices, 2):
            for p1, p2 in product(coarse_parts, repeat=2):
                try:
                    colors = [palette_hexes[i], palette_hexes[j]]
                    weights = [p1, p2]
                    mixed_hex = mix_colors_hex(colors, weights, method=method)
                    mixed_rgb = hex_to_rgb255(mixed_hex)
                    mixed_lab = rgb_to_lab(*mixed_rgb)
                    de = delta_e_cie76(target_lab, mixed_lab)
                    
                    if de < best_delta_e:
                        best_delta_e = de
                        best_recipe = [(i, p1), (j, p2)]
                        best_result_hex = mixed_hex
                        best_method = method
                        
                        if de < 2.0:  # Good enough for coarse search
                            break
                except:
                    continue
            if best_delta_e < 2.0:
                break
        if best_delta_e < 2.0:
            break
    
    # Step 3: Fine-tune the best 2-color recipe with more precision
    if len(best_recipe) == 2 and best_delta_e > 1.0:
        i, p1 = best_recipe[0]
        j, p2 = best_recipe[1]
        
        # Search around the coarse solution
        fine_range = range(max(1, p1 - 2), min(max_parts + 1, p1 + 3))
        fine_range2 = range(max(1, p2 - 2), min(max_parts + 1, p2 + 3))
        
        for method in mixing_methods:
            for fp1, fp2 in product(fine_range, fine_range2):
                try:
                    colors = [palette_hexes[i], palette_hexes[j]]
                    weights = [fp1, fp2]
                    mixed_hex = mix_colors_hex(colors, weights, method=method)
                    mixed_rgb = hex_to_rgb255(mixed_hex)
                    mixed_lab = rgb_to_lab(*mixed_rgb)
                    de = delta_e_cie76(target_lab, mixed_lab)
                    
                    if de < best_delta_e:
                        best_delta_e = de
                        best_recipe = [(i, fp1), (j, fp2)]
                        best_result_hex = mixed_hex
                        best_method = method
                except:
                    continue
    
    # Step 4: Try 3-color only if 2-color match is poor
    if max_colors >= 3 and best_delta_e > 5.0:
        top_8 = promising_indices[:8]  # Even smaller set for 3-color
        coarse_3 = [1, 2, 3, 4]  # Very coarse for 3-color
        
        for method in mixing_methods:
            for indices in combinations(top_8, 3):
                for parts in product(coarse_3, repeat=3):
                    try:
                        colors = [palette_hexes[idx] for idx in indices]
                        weights = list(parts)
                        mixed_hex = mix_colors_hex(colors, weights, method=method)
                        mixed_rgb = hex_to_rgb255(mixed_hex)
                        mixed_lab = rgb_to_lab(*mixed_rgb)
                        de = delta_e_cie76(target_lab, mixed_lab)
                        
                        if de < best_delta_e:
                            best_delta_e = de
                            best_recipe = list(zip(indices, parts))
                            best_result_hex = mixed_hex
                            best_method = method
                            
                            if de < 3.0:
                                break
                    except:
                        continue
                if best_delta_e < 3.0:
                    break
            if best_delta_e < 3.0:
                break
    
    match_pct = delta_e_to_match_percentage(best_delta_e)
    return best_recipe, best_result_hex, match_pct, best_delta_e, best_method


def simplify_recipe(recipe: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Combine duplicate colors and simplify ratios using GCD"""
    from collections import defaultdict
    
    # Combine same colors
    combined = defaultdict(int)
    for idx, parts in recipe:
        combined[idx] += parts
    
    # Convert back to list and simplify with GCD
    parts_list = list(combined.values())
    if parts_list:
        gcd = parts_list[0]
        for p in parts_list[1:]:
            gcd = math.gcd(gcd, p)
        if gcd > 1:
            combined = {idx: parts // gcd for idx, parts in combined.items()}
    
    return [(idx, parts) for idx, parts in combined.items()]


@app.post("/color_match", response_model=ColorMatchResponse)
def color_match(req: ColorMatchRequest):
    """
    Compare a selected color against a palette of colors using CIE LAB color space.
    
    Returns perceptually accurate color matching with Delta E values and match percentages.
    Results are sorted by match percentage (highest first).
    
    **Delta E interpretation:**
    - 0-1: Not perceptible by human eyes
    - 1-2: Perceptible through close observation
    - 2-10: Perceptible at a glance
    - 11-49: Colors are more similar than opposite
    - 100+: Colors are exact opposites
    """
    # Validate selected color
    try:
        selected_normalized = normalize_hex(req.selected_color)
        _ = hex_to_rgb255(selected_normalized)
    except Exception as e:
        return ColorMatchResponse(
            selected_color=req.selected_color,
            matches=[],
            best_match=None,
            has_matches=False,
            error=f"Invalid selected color: {req.selected_color}"
        )
    
    # Validate palette colors
    if not req.palette_colors:
        return ColorMatchResponse(
            selected_color=selected_normalized,
            matches=[],
            best_match=None,
            has_matches=False,
            error="No palette colors provided"
        )
    
    matches = []
    for hex_color in req.palette_colors:
        try:
            color_normalized = normalize_hex(hex_color)
            
            match_pct, delta_e = calculate_color_match(selected_normalized, color_normalized)
            
            # Apply threshold filter
            if match_pct >= req.threshold:
                matches.append(ColorMatchResult(
                    hex=color_normalized,
                    match_percentage=int(round(match_pct)),
                    delta_e=round(delta_e, 2)
                ))
        except Exception:
            # Skip invalid colors
            continue
    
    # Sort by match percentage (highest first)
    matches.sort(key=lambda x: x.match_percentage, reverse=True)
    
    # Check if all matches are effectively 0
    has_meaningful_matches = any(m.match_percentage > 0.5 for m in matches)
    
    if not matches:
        return ColorMatchResponse(
            selected_color=selected_normalized,
            matches=[],
            best_match=None,
            has_matches=False,
            error="No valid palette colors could be processed"
        )
    
    if not has_meaningful_matches:
        return ColorMatchResponse(
            selected_color=selected_normalized,
            matches=matches,
            best_match=matches[0] if matches else None,
            has_matches=False,
            error="No significant matches found - all colors are too different from the selected color"
        )
    
    return ColorMatchResponse(
        selected_color=selected_normalized,
        matches=matches,
        best_match=matches[0] if matches else None,
        has_matches=True,
        error=None
    )


@app.post("/unmix", response_model=UnmixResponse)
def unmix_color(req: UnmixRequest):
    """
    Find a mixing recipe to achieve a target color from your palette.
    
    This is like trycolors.com's "Get Mix" feature - given a target color and
    available paints, it calculates the optimal mix proportions.
    
    **Request:**
    - `target_color`: The hex color you want to achieve (e.g., "#8B4513")
    - `palette`: List of available paints with hex and optional name
    - `max_colors`: Maximum colors to use in recipe (1-5, default 3)
    - `max_parts`: Maximum parts per color for ratio precision (1-20, default 10)
    
    **Response:**
    - `recipe`: List of colors with parts and percentage
    - `result_color`: The actual color from mixing the recipe
    - `match_percentage`: How close the result matches target (0-100%)
    - `delta_e`: CIE Delta E value for technical reference
    
    **Example usage:**
    ```json
    {
        "target_color": "#8B4513",
        "palette": [
            {"hex": "#FF0000", "name": "Cadmium Red"},
            {"hex": "#FFFF00", "name": "Cadmium Yellow"},
            {"hex": "#0000FF", "name": "Ultramarine Blue"},
            {"hex": "#FFFFFF", "name": "Titanium White"},
            {"hex": "#000000", "name": "Ivory Black"}
        ],
        "max_colors": 3,
        "max_parts": 10
    }
    ```
    """
    # Validate target color
    try:
        target_normalized = normalize_hex(req.target_color)
        _ = hex_to_rgb255(target_normalized)
    except Exception as e:
        return UnmixResponse(
            target_color=req.target_color,
            recipe=[],
            result_color=req.target_color,
            match_percentage=0.0,
            delta_e=100.0,
            total_parts=0,
            error=f"Invalid target color: {req.target_color}"
        )
    
    # Validate palette
    if not req.palette:
        return UnmixResponse(
            target_color=target_normalized,
            recipe=[],
            result_color=target_normalized,
            match_percentage=0.0,
            delta_e=100.0,
            total_parts=0,
            error="No palette colors provided"
        )
    
    # Validate and normalize palette colors
    valid_palette = []
    for p in req.palette:
        try:
            normalized = normalize_hex(p.hex)
            _ = hex_to_rgb255(normalized)
            valid_palette.append(PaletteColor(hex=normalized, name=p.name))
        except:
            continue
    
    if not valid_palette:
        return UnmixResponse(
            target_color=target_normalized,
            recipe=[],
            result_color=target_normalized,
            match_percentage=0.0,
            delta_e=100.0,
            total_parts=0,
            error="No valid palette colors"
        )
    
    # Clamp parameters
    max_colors = max(1, min(5, req.max_colors or 3))
    max_parts = max(1, min(20, req.max_parts or 10))
    
    # Find best recipe
    try:
        raw_recipe, result_hex, match_pct, delta_e, mix_method = find_best_recipe(
            target_normalized,
            valid_palette,
            max_colors=max_colors,
            max_parts=max_parts
        )
        
        # Simplify recipe (combine duplicates, reduce ratios)
        simplified = simplify_recipe(raw_recipe)
        
        # Build response recipe
        total_parts = sum(parts for _, parts in simplified)
        recipe_components = []
        for idx, parts in simplified:
            palette_item = valid_palette[idx]
            pct = (parts / total_parts * 100) if total_parts > 0 else 0
            recipe_components.append(RecipeComponent(
                hex=palette_item.hex,
                name=palette_item.name,
                parts=parts,
                percentage=round(pct, 1)
            ))
        
        # Sort by parts (largest first)
        recipe_components.sort(key=lambda x: x.parts, reverse=True)
        
        return UnmixResponse(
            target_color=target_normalized,
            recipe=recipe_components,
            result_color=result_hex,
            match_percentage=round(match_pct, 1),
            delta_e=round(delta_e, 2),
            total_parts=total_parts,
            mix_method=mix_method,
            error=None
        )
    
    except Exception as e:
        return UnmixResponse(
            target_color=target_normalized,
            recipe=[],
            result_color=target_normalized,
            match_percentage=0.0,
            delta_e=100.0,
            total_parts=0,
            mix_method=None,
            error=f"Error finding recipe: {str(e)}"
        )


# ------------ Original Shape Processing Code ------------

@dataclass
class CanvasSize:
    width: float
    height: float


def get_canvas_size(shapes: List[ShapeModel]) -> CanvasSize:
    bg = next((s for s in shapes if s.type == 0), None)
    if bg is not None and len(bg.data) >= 4:
        x1, y1, x2, y2 = bg.data[:4]
        w = abs(x2 - x1) or x2
        h = abs(y2 - y1) or y2
        return CanvasSize(width=w, height=h)
    
    max_x = 0.0
    max_y = 0.0
    for s in shapes:
        if s.type == 0 and len(s.data) >= 4:
            x1, y1, x2, y2 = s.data[:4]
            max_x = max(max_x, x1, x2)
            max_y = max(max_y, y1, y2)
        elif s.type == 4 and len(s.data) >= 4:
            cx, cy, rx, ry = s.data[:4]
            max_x = max(max_x, cx + rx)
            max_y = max(max_y, cy + ry)
    
    return CanvasSize(width=max_x or 1.0, height=max_y or 1.0)


def shape_center(shape: ShapeModel, canvas: CanvasSize) -> Tuple[float, float]:
    if shape.type == 0 and len(shape.data) >= 4:
        x1, y1, x2, y2 = shape.data[:4]
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
    if shape.type == 4 and len(shape.data) >= 2:
        cx, cy = shape.data[:2]
        return (cx, cy)
    return (canvas.width / 2.0, canvas.height / 2.0)


def is_shape_in_region(shape: ShapeModel, region: RegionModel, canvas: CanvasSize) -> bool:
    # Check if a shape's center is within a given region
    cx, cy = shape_center(shape, canvas)
    return (region.x <= cx <= region.x + region.width and 
            region.y <= cy <= region.y + region.height)


def distance_to_image_center(shape: ShapeModel, canvas: CanvasSize) -> float:
    x, y = shape_center(shape, canvas)
    cx = canvas.width / 2.0
    cy = canvas.height / 2.0
    dx = x - cx
    dy = y - cy
    return math.hypot(dx, dy)


def distance_to_nearest_edge(shape: ShapeModel, canvas: CanvasSize) -> float:
    x, y = shape_center(shape, canvas)
    d_left = x
    d_right = canvas.width - x
    d_top = y
    d_bottom = canvas.height - y
    return min(d_left, d_right, d_top, d_bottom)


def luminance_from_color(color: List[int]) -> float:
    if len(color) < 3:
        return 0.0
    r, g, b = color[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def color_key(color: List[int]) -> str:
    return ",".join(str(int(c)) for c in color[:4])


def order_exterior_to_center(shapes: List[ShapeModel]) -> List[ShapeModel]:
    canvas = get_canvas_size(shapes)
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    others_sorted = sorted(others, key=lambda s: distance_to_nearest_edge(s, canvas))
    return bg + others_sorted


def order_center_to_exterior(shapes: List[ShapeModel]) -> List[ShapeModel]:
    canvas = get_canvas_size(shapes)
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    others_sorted = sorted(others, key=lambda s: distance_to_image_center(s, canvas))
    return bg + others_sorted


def order_top_to_bottom(shapes: List[ShapeModel]) -> List[ShapeModel]:
    canvas = get_canvas_size(shapes)
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    return bg + sorted(others, key=lambda s: shape_center(s, canvas)[1])


def order_bottom_to_top(shapes: List[ShapeModel]) -> List[ShapeModel]:
    canvas = get_canvas_size(shapes)
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    return bg + sorted(others, key=lambda s: -shape_center(s, canvas)[1])


def order_by_color_sequence(shapes, color_order):
    if not color_order:
        return shapes
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    index_by_key = {color_key(col): idx for idx, col in enumerate(color_order)}
    fallback = len(color_order) + 1
    def sort_key(s):
        return (index_by_key.get(color_key(s.color), fallback), -luminance_from_color(s.color))
    return bg + sorted(others, key=sort_key)


def order_light_to_dark(shapes):
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    return bg + sorted(others, key=lambda s: -luminance_from_color(s.color))


def order_dark_to_light(shapes):
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    return bg + sorted(others, key=lambda s: luminance_from_color(s.color))


def order_by_color_frequency(shapes):
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    freq = Counter(color_key(s.color) for s in others)
    sorted_keys = sorted(freq.keys(), key=lambda ck: -freq[ck])
    rank = {ck: i for i, ck in enumerate(sorted_keys)}
    return bg + sorted(others, key=lambda s: rank[color_key(s.color)])


def order_by_color_frequency_reverse(shapes):
    bg = [s for s in shapes if s.type == 0]
    others = [s for s in shapes if s.type != 0]
    freq = Counter(color_key(s.color) for s in others)
    sorted_keys = sorted(freq.keys(), key=lambda ck: freq[ck])
    rank = {ck: i for i, ck in enumerate(sorted_keys)}
    return bg + sorted(others, key=lambda s: rank[color_key(s.color)])


def order_shapes(shapes: List[ShapeModel], logic: Logic, color_order=None):
    if logic == Logic.exterior_to_center:
        return order_exterior_to_center(shapes)
    if logic == Logic.center_to_exterior:
        return order_center_to_exterior(shapes)
    if logic == Logic.top_to_bottom:
        return order_top_to_bottom(shapes)
    if logic == Logic.bottom_to_top:
        return order_bottom_to_top(shapes)
    if logic == Logic.color_sequence or logic == Logic.custom_sequence:
        return order_by_color_sequence(shapes, color_order)
    if logic == Logic.light_to_dark:
        return order_light_to_dark(shapes)
    if logic == Logic.dark_to_light:
        return order_dark_to_light(shapes)
    if logic == Logic.frequency_by_color:
        return order_by_color_frequency(shapes)
    if logic == Logic.frequency_by_color_reverse:
        return order_by_color_frequency_reverse(shapes)
    return shapes


def apply_limit(shapes: List[ShapeModel], limit: Optional[int]) -> List[ShapeModel]:
    if limit is None:
        return shapes
    if limit <= 0:
        return []
    return shapes[: min(limit, len(shapes))]


@app.post("/order_shapes_from_json", response_model=OrderResponse)
def order_shapes_from_json(req: OrderRequest):
    """
    Apply various ordering strategies to a list of shapes.
    
    **Available ordering strategies:**
    - `exterior_to_center`: Order from edges toward center
    - `center_to_exterior`: Order from center toward edges
    - `top_to_bottom`: Order by vertical position (top first)
    - `bottom_to_top`: Order by vertical position (bottom first)
    - `light_to_dark`: Order by luminance (bright to dark)
    - `dark_to_light`: Order by luminance (dark to bright)
    - `frequency_by_color`: Order by color frequency (most common first)
    - `frequency_by_color_reverse`: Order by color frequency (least common first)
    - `color_sequence`: Order by custom color sequence (requires color_order)
    - `custom_sequence`: Same as color_sequence
    
    Background shapes (type=0) are always placed first.
    """

    ordered = order_shapes(req.shapes, req.logic, req.color_order)
    sliced = apply_limit(ordered, req.limit)
    return OrderResponse(shapes=sliced, total_shapes=len(ordered))


@app.post("/selective_resolution", response_model=OrderResponse)
def apply_selective_resolution(req: SelectiveResolutionRequest):

    """
    Enhance specific regions with higher resolution shapes.
    
    This endpoint allows you to:
    1. Start with low-resolution base shapes
    2. Define regions of interest
    3. Replace shapes in those regions with high-resolution detail shapes
    
    Useful for optimizing performance while maintaining quality in important areas.
    """

    if not req.base_shapes:
        return OrderResponse(shapes=[], total_shapes=0)
    
    canvas = get_canvas_size(req.base_shapes)
    
    bg_shapes = [s for s in req.base_shapes if s.type == 0]
    base_others = [s for s in req.base_shapes if s.type != 0]
    detail_others = [s for s in req.detail_shapes if s.type != 0]
    
    base_in_regions = set()
    for i, shape in enumerate(base_others):
        for region in req.regions:
            if is_shape_in_region(shape, region, canvas):
                base_in_regions.add(i)
                break
    
    result_shapes = bg_shapes.copy()
    result_shapes.extend([s for i, s in enumerate(base_others) if i not in base_in_regions])
    
    for shape in detail_others:
        for region in req.regions:
            if is_shape_in_region(shape, region, canvas):
                result_shapes.append(shape)
                break
    
    return OrderResponse(shapes=result_shapes, total_shapes=len(result_shapes))

@app.post("/bake_opaque", response_model=OrderResponse)
async def bake_opaque(
    req: str = File(...),
    file: UploadFile = File(...)
):
    """
    Converts all shapes to fully opaque by sampling colors from the final rendered PNG image.
    
    This is useful when you want to:
    - Remove transparency from shapes
    - Capture the actual rendered color (accounting for blending)
    - Simplify shape data for further processing
    
    **Requires:**
    - `req`: JSON string containing shapes array
    - `file`: PNG image file of the rendered shapes
    """

    # Converts all shapes into fully opaque shapes by sampling color
    # from the final rendered PNG image.

    # Read PNG image
    img_bytes = await file.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    arr = np.array(pil)
    h, w, _ = arr.shape

    
    req_json = json.loads(req) # Parse JSON payload
    shapes = [ShapeModel(**s) for s in req_json["shapes"]]

    baked_shapes = []

    def clamp(v, low, high):
        return max(low, min(high, int(v)))

    def sample_color_at(shape: ShapeModel):
        # Use shape center (consistent with your system)
        cx, cy = shape_center(shape, get_canvas_size(shapes))

        px = clamp(cx, 0, w - 1)
        py = clamp(cy, 0, h - 1)

        rgba = arr[py, px]
        return [int(rgba[0]), int(rgba[1]), int(rgba[2]), 255]

    
    for s in shapes: # Background remains unchanged
        if s.type == 0:
            baked_shapes.append(s)
            continue

        # Sample real visible pixel color
        new_color = sample_color_at(s)

        baked_shapes.append(
            ShapeModel(
                type=s.type,
                data=s.data,
                color=new_color,
                score=s.score
            )
        )

    return OrderResponse(shapes=baked_shapes, total_shapes=len(baked_shapes))