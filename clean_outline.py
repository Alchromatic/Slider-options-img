"""'outline' line-drawing style: the client's outline.py, split at the slider boundary.

clean_outline_base() is the slider-independent first half (main.py serves it to the
browser so the sliders run locally; ai_lineart_routes.py traces on top of it) and
clean_outline_lines() is the rest. Together they equal outline.py exactly.
"""
import cv2
import numpy as np

OUTLINE_MAX_DIM = 1600


def clean_outline_base(gray: np.ndarray) -> np.ndarray:
    """outline.py before Canny: standardize resolution, then suppress fine texture."""
    scale = min(1.0, OUTLINE_MAX_DIM / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.bilateralFilter(gray, 9, 60, 60)


def clean_outline_lines(smooth: np.ndarray, low: int, high: int, min_size: int, thickness: int) -> np.ndarray:
    """outline.py from Canny on: edges, drop short fragments, thicken. Black lines on white."""
    edges = cv2.Canny(smooth, low, high, L2gradient=True)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_size).astype(np.uint8) * 255
    edges = keep[labels]
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        edges = cv2.dilate(edges, kernel)
    return 255 - edges
