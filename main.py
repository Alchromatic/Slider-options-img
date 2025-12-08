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