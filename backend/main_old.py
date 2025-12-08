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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
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
    base_shapes: List[ShapeModel]  # Low resolution (e.g., 300 shapes)
    detail_shapes: List[ShapeModel]  # High resolution (e.g., 3000 shapes)
    regions: List[RegionModel]  # Areas to enhance


class OrderResponse(BaseModel):
    shapes: List[ShapeModel]
    total_shapes: int


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
    """Check if a shape's center is within a given region"""
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
    ordered = order_shapes(req.shapes, req.logic, req.color_order)
    sliced = apply_limit(ordered, req.limit)
    return OrderResponse(shapes=sliced, total_shapes=len(ordered))


@app.post("/selective_resolution", response_model=OrderResponse)
def apply_selective_resolution(req: SelectiveResolutionRequest):

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
    Converts all shapes into fully opaque shapes by sampling color
    from the final rendered PNG image.
    """

    # Read PNG image
    img_bytes = await file.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    arr = np.array(pil)
    h, w, _ = arr.shape

    # Parse JSON payload
    req_json = json.loads(req)
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

    # Background remains unchanged
    for s in shapes:
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
