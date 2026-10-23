"""Snapshot line comparison. RGB inputs; coordinates and distances in canvas space."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, remove_small_objects

@dataclass(frozen=True)
class Settings:
    position: bool = True
    thickness: bool = False
    tolerance_mm: float = 2.0
    width_tolerance: float = 0.20
    association_mm: float = 10.0
    threshold: int = 150
    min_ink_pixels: int = 8


def rectify(rgb, corners, width_mm, height_mm, long_side=1200):
    """corners: TL, TR, BR, BL in source pixels. Returns image and homography."""
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError('Canvas dimensions must be positive.')
    p = np.asarray(corners, np.float32)
    h, w = rgb.shape[:2]
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError('Supply four finite x,y corner coordinates.')
    if (p < 0).any() or (p[:, 0] >= w).any() or (p[:, 1] >= h).any():
        raise ValueError('Corners must be inside the camera image.')
    if not cv2.isContourConvex(p) or cv2.contourArea(p) < 100:
        raise ValueError('Corners must form a nondegenerate convex quadrilateral.')
    scale = long_side / max(width_mm, height_mm)
    out_w, out_h = round(width_mm * scale) + 1, round(height_mm * scale) + 1
    dst = np.float32([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]])
    H = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(rgb, H, (out_w, out_h), borderValue=(255,255,255)), H


def extract(rgb, width_mm, height_mm, settings):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = remove_small_objects(gray < settings.threshold, min_size=settings.min_ink_pixels)
    skel = skeletonize(mask)
    y, x = np.nonzero(skel)
    sx, sy = width_mm/(mask.shape[1]-1), height_mm/(mask.shape[0]-1)
    xy = np.column_stack((x*sx, y*sy))
    # Local width estimate: twice distance from centerline to background.
    radii = distance_transform_edt(np.pad(mask, 1), sampling=(sy, sx))[1:-1,1:-1]
    widths = 2*radii[y, x]
    # Approximate arc length: allocate half of each skeleton graph edge to each end.
    weights = np.zeros(mask.shape, float)
    for dy, dx in ((0,1),(1,0),(1,1),(1,-1)):
        ya, xa = np.nonzero(skel)
        yb, xb = ya+dy, xa+dx
        inside = (yb >= 0)&(yb < mask.shape[0])&(xb >= 0)&(xb < mask.shape[1])
        ya, xa, yb, xb = ya[inside], xa[inside], yb[inside], xb[inside]
        keep = skel[yb, xb]
        # Avoid a diagonal shortcut where an orthogonal connection already exists.
        if dy and dx:
            keep &= ~(skel[ya, xb] | skel[yb, xa])
        ya, xa, yb, xb = ya[keep], xa[keep], yb[keep], xb[keep]
        edge = np.hypot(dx*sx, dy*sy)/2
        np.add.at(weights, (ya,xa), edge)
        np.add.at(weights, (yb,xb), edge)
    weights = weights[y,x]
    weights[weights == 0] = min(sx, sy)
    return dict(mask=mask, skel=skel, x=x, y=y, xy=xy, widths=widths, weights=weights)


def directional(source, target, s, source_is_drawing):
    n = len(source['xy'])
    if not len(target['xy']):
        return dict(position=np.zeros(n,bool), thickness=np.zeros(n,bool), combined=np.zeros(n,bool))
    d, j = cKDTree(target['xy']).query(source['xy'])
    position = d <= s.tolerance_mm
    # Denominator is ALWAYS reference width, including the reverse coverage pass.
    ref_width = target['widths'][j] if source_is_drawing else source['widths']
    width_error = np.abs(source['widths']-target['widths'][j])/np.maximum(ref_width, 1e-9)
    thickness = (d <= s.association_mm) & (width_error <= s.width_tolerance)
    combined = np.ones(n, bool)
    if s.position:
        combined &= position
    if s.thickness:
        combined &= thickness
    return dict(position=position, thickness=thickness, combined=combined)


def compare(reference_rgb, drawing_rgb, width_mm, height_mm, settings=Settings()):
    if not (settings.position or settings.thickness):
        raise ValueError('Enable position, thickness, or both.')
    if reference_rgb.shape != drawing_rgb.shape:
        raise ValueError('Reference and drawing must share the rectified image shape.')
    ref = extract(reference_rgb, width_mm, height_mm, settings)
    drawing = extract(drawing_rgb, width_mm, height_mm, settings)
    if not len(ref['xy']):
        raise ValueError('Reference contains no detected lines. Check the ink threshold.')
    forward = directional(drawing, ref, settings, True)
    reverse = directional(ref, drawing, settings, False)
    metrics = {}
    for key in ('position','thickness','combined'):
        def fraction(data, passed):
            return float(np.sum(data['weights']*passed)/np.sum(data['weights'])) if len(passed) else 0.0
        p, r = fraction(drawing, forward[key]), fraction(ref, reverse[key])
        metrics[key] = dict(accuracy=p, coverage=r, score=2*p*r/(p+r) if p+r else 0.0)
    # Separate layer can be inverse-warped onto the captured camera image.
    layer = np.zeros_like(drawing_rgb)
    visible = np.zeros(drawing_rgb.shape[:2], np.uint8)
    dash = ((ref['x']+ref['y'])//6)%2 == 0
    layer[ref['y'][dash],ref['x'][dash]] = (130,140,150)
    visible[ref['y'][dash],ref['x'][dash]] = 255
    for i, (x,y) in enumerate(zip(drawing['x'],drawing['y'])):
        color = (35,190,85) if forward['combined'][i] else (240,75,65)
        cv2.circle(layer,(int(x),int(y)),1,color,-1)
        cv2.circle(visible,(int(x),int(y)),1,255,-1)
    return dict(metrics=metrics, layer=layer, visible=visible,
                reference_mask=ref['mask'], drawing_mask=drawing['mask'])


def overlay_camera(camera, result, homography):
    h,w = camera.shape[:2]
    inverse = np.linalg.inv(homography)
    layer = cv2.warpPerspective(result['layer'],inverse,(w,h),flags=cv2.INTER_NEAREST)
    mask = cv2.warpPerspective(result['visible'],inverse,(w,h),flags=cv2.INTER_NEAREST)>0
    out = camera.copy()
    out[mask] = (0.15*camera[mask] + 0.85*layer[mask]).astype(np.uint8)
    return out
