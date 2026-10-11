"""AI Line Art: an AI image model marks which outlines matter; the lines are the photo's own.

POST /line-drawing/ai-guide (image) returns, for the browser's Drawing panel:
  base_png   the Clean Outline base (grayscale, <= 1600px, bilateral) - same as /outline-base
  guide_png  the AI's line drawing, auto-aligned to the photo (black lines on white)
  band_png   where the guide is: white within a few px of a guide line
The browser runs Canny on the base and keeps only edges inside the band (clean-outline.js),
so every drawn line lies exactly on an edge of the photo. Anything the AI invents or moves
has no photo edge under it and drops out; texture the AI skipped drops out too.

Model choice comes from a test on the client's harbour photo: Gemini image models keep the
photo's framing (within ~0.5% scale / 4px), OpenAI's re-compose it. Nano Banana 2.1 traced
the full structure on every run, in ~12 s for ~$0.04.
"""
import asyncio
import base64
import hashlib
import os
import threading
import time
from collections import OrderedDict
from datetime import date

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from clean_outline import clean_outline_base

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

router = APIRouter(prefix="/line-drawing", tags=["1. Drawing"])

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [m.strip() for m in os.getenv(
    "AI_LINEART_MODELS", "google/gemini-nano-banana-2.1,google/gemini-3.1-flash-image").split(",") if m.strip()]
DAILY_PER_CLIENT = int(os.getenv("AI_LINEART_DAILY_PER_CLIENT", "20"))
DAILY_TOTAL = int(os.getenv("AI_LINEART_DAILY_TOTAL", "300"))

# Share of the guide's lines that must land on real photo edges after alignment
# (faithful outputs scored 0.66-0.89 on the test photo; cloud contours lower it).
MIN_FIT = 0.55
# The AI must keep the photo's framing: faithful outputs needed <= 1% scale and
# <= 4 px correction, re-composed ones 5%+ and ~100 px. Beyond these limits the
# drawing is rejected, however well it fits after correction.
MAX_SCALE_FIX, MAX_SHIFT_FIX = 0.03, 0.03
# Share of the photo's strong structural edges the guide should cover (complete
# drawings scored 1.0, ones that skipped structure ~0.65); below this one more
# model is tried and the better result kept.
GOOD_COVERAGE = 0.85

# Gemini image models accept these output aspect ratios.
ASPECTS = {"1:1": 1.0, "2:3": 2 / 3, "3:2": 1.5, "3:4": 0.75, "4:3": 4 / 3, "4:5": 0.8,
           "5:4": 1.25, "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9}

PROMPT = """Convert the attached photograph into a precise line drawing that will be used as a tracing guide for painting this exact photo. Accuracy to the photograph matters more than artistic style. The drawing will be laid over the photo and checked line by line for alignment.

OUTPUT
- One image only: black line art on a pure white background. Every line is solid pure black (#000000); the background is pure white (#FFFFFF). Never use grey or light lines, colour, shading, hatching, stippling, gradients, shadows or texture.
- Never fill any area with black or grey. Dark areas (rocks, shadows, dark water, windows, silhouettes) get an outline only, with white inside.
- Same aspect ratio and exactly the same framing as the photograph. Do not crop, zoom, pad, extend, rotate, mirror, straighten or recompose anything. Every object keeps its exact position, size, proportions and perspective, so each line lies exactly on top of the edge it traces, as if drawn on tracing paper placed over the photo.
- No border, frame, text, caption, label, signature or watermark.

WHAT TO DRAW
- The outer silhouette of every main subject and major object (for example buildings, towers, walls, piers, rocks, boats, trees, poles, lamp posts).
- The major internal edges that define each object's form: windows, doors, openings, ledges, balconies, rooflines, steps, and where one surface meets another. Keep each object's real shape exactly, including its top and its smaller attached parts.
- Landscape boundaries: the horizon, shorelines, mountain and hill silhouettes, and the edge of distant towns or tree lines as a simple outline.
- Clouds: outline only the few large, clearly bounded cloud masses, each as one simple, smooth closed contour. Skip thin, wispy or streaky clouds entirely. Never invent clouds.
- Water: keep it mostly empty. Use at most a few short, sparse horizontal strokes, plus the outline of clearly visible reflections.

WHAT NOT TO DRAW
- Nothing that is not in the photograph: no added objects, animals, statues, ornaments, birds, boats, sun, moon or stars. Do not replace any part of an object with something else.
- No people at all, however many or however small.
- No fine texture or noise: no ripples, grain, individual stones or bricks, foliage detail, pebbles, sky gradients or surface patterns.
- No tiny incidental details smaller than about 1% of the image width.

LINE QUALITY
- Clean, continuous, confident strokes of one uniform medium weight, all equally dark. Close an outline wherever the edge in the photo is continuous. No sketchy, doubled, broken or hairy lines.

Return only the finished drawing as an image."""


# ---------------------------------------------------------------------------
# Cost control: identical uploads reuse the last result; daily caps per client and overall.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_cache: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX, _CACHE_TTL = 32, 6 * 3600
_usage = {"day": None, "total": 0, "clients": {}}


def _client_key(request: Request) -> str:
    """Signed-in user if a valid token is sent, else the caller's IP."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            from auth_db import decode_jwt_token
            return "user:" + str(decode_jwt_token(auth[7:])["sub"])
        except Exception:
            pass
    fwd = request.headers.get("x-forwarded-for", "")
    return "ip:" + (fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown"))


def _take_quota(client: str) -> None:
    with _lock:
        today = date.today().isoformat()
        if _usage["day"] != today:
            _usage.update(day=today, total=0, clients={})
        if _usage["total"] >= DAILY_TOTAL:
            raise HTTPException(status_code=429, detail="AI Line Art has reached today's limit. Please try again tomorrow.")
        used = _usage["clients"].get(client, 0)
        if used >= DAILY_PER_CLIENT:
            raise HTTPException(status_code=429, detail=f"You've used all {DAILY_PER_CLIENT} AI Line Art drawings for today. Please try again tomorrow.")
        _usage["clients"][client] = used + 1
        _usage["total"] += 1


def _cache_get(key: str):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit["at"] < _CACHE_TTL:
            _cache.move_to_end(key)
            return hit["result"]
        return None


def _cache_put(key: str, result: dict) -> None:
    with _lock:
        _cache[key] = {"at": time.time(), "result": result}
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
def _png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _model_input(color: np.ndarray, w: int, h: int):
    """The photo at the working size, padded with white to the nearest aspect ratio the
    model supports (so its output can be mapped back without stretching)."""
    small = cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA)
    aspect_key = min(ASPECTS, key=lambda k: abs(np.log(ASPECTS[k] * h / w)))
    target = ASPECTS[aspect_key]
    pw, ph = w, h
    if abs(np.log(target * h / w)) > 0.005:
        if target > w / h:
            pw = int(round(h * target))
        else:
            ph = int(round(w / target))
    px, py = (pw - w) // 2, (ph - h) // 2
    padded = cv2.copyMakeBorder(small, py, ph - h - py, px, pw - w - px, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    ok, buf = cv2.imencode(".jpg", padded, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode(), aspect_key, (px, py, pw, ph)


async def _ask_model(client: httpx.AsyncClient, model: str, jpeg_b64: str, aspect_key: str) -> bytes:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="AI Line Art is not configured (OPENROUTER_API_KEY)")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64}},
            {"type": "text", "text": PROMPT},
        ]}],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": aspect_key},
    }
    res = await client.post(OPENROUTER_URL, json=body, headers={"Authorization": "Bearer " + key})
    data = res.json()
    if res.status_code != 200:
        raise RuntimeError(f"{model}: HTTP {res.status_code} {str(data.get('error', data))[:200]}")
    images = (data.get("choices") or [{}])[0].get("message", {}).get("images") or []
    if not images:
        raise RuntimeError(f"{model}: no image returned")
    return base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])


def _guide_from(raw: bytes, pad, w: int, h: int) -> np.ndarray:
    """AI output -> binary guide (255 = line) in working-size photo coordinates."""
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("unreadable image")
    px, py, pw, ph = pad
    img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)[py:py + h, px:px + w]
    return (img < 200).astype(np.uint8) * 255   # grey strokes count as lines too


def _align(guide: np.ndarray, smooth: np.ndarray):
    """Small scale/shift that best lays the guide on the photo's edges.
    Returns (aligned guide, fit, coverage, transform)."""
    h, w = smooth.shape
    tol = max(4.0, 0.006 * max(w, h))                      # "on the edge": ~10 px at 1600
    edges = cv2.Canny(smooth, 15, 45, L2gradient=True)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(guide)
    if len(xs) < 50:
        return None
    pick = np.random.default_rng(0).choice(len(xs), min(2500, len(xs)), replace=False)
    gx, gy = xs[pick].astype(np.float32), ys[pick].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0

    def cost(sx, sy, dx, dy):
        X = np.rint((gx - cx) * sx + cx + dx).astype(np.int32)
        Y = np.rint((gy - cy) * sy + cy + dy).astype(np.int32)
        inside = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
        d = np.full(len(gx), tol, np.float32)
        d[inside] = np.minimum(dist[Y[inside], X[inside]], tol)
        return float(d.mean())

    best = (cost(1, 1, 0, 0), 1.0, 1.0, 0.0, 0.0)
    step = max(2.0, round(0.01 * max(w, h)))
    for s in np.arange(0.94, 1.0601, 0.02):                # coarse: uniform scale, +-6% shift
        for dx in np.arange(-6, 7) * step:
            for dy in np.arange(-6, 7) * step:
                c = cost(s, s, dx, dy)
                if c < best[0]:
                    best = (c, s, s, dx, dy)
    _, s0, _, dx0, dy0 = best
    for sx in np.arange(s0 - 0.01, s0 + 0.0101, 0.005):    # fine: per-axis scale, 2 px shifts
        for sy in np.arange(s0 - 0.01, s0 + 0.0101, 0.005):
            for dx in np.arange(dx0 - step, dx0 + step + 1, 2):
                for dy in np.arange(dy0 - step, dy0 + step + 1, 2):
                    c = cost(sx, sy, dx, dy)
                    if c < best[0]:
                        best = (c, sx, sy, dx, dy)
    _, sx, sy, dx, dy = best
    M = np.float32([[sx, 0, (1 - sx) * cx + dx], [0, sy, (1 - sy) * cy + dy]])
    aligned = cv2.warpAffine(guide, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    ys, xs = np.nonzero(aligned)
    fit = float((dist[ys, xs] <= tol).mean()) if len(xs) else 0.0
    strong = cv2.Canny(smooth, 70, 150, L2gradient=True)
    n, lab, st, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = (st[1:, cv2.CC_STAT_AREA] >= 0.075 * max(w, h)).astype(np.uint8)
    strong = keep[lab]
    near_guide = cv2.distanceTransform(255 - aligned, cv2.DIST_L2, 5) <= tol
    coverage = float(near_guide[strong > 0].mean()) if strong.any() else 1.0
    return aligned, fit, coverage, {"scale_x": round(float(sx), 4), "scale_y": round(float(sy), 4),
                                     "dx": int(dx), "dy": int(dy), "tolerance_px": round(tol, 1)}


@router.post("/ai-guide")
async def line_drawing_ai_guide(request: Request, image: UploadFile = File(...), fresh: bool = Form(False)):
    """
    AI Line Art: the AI decides which outlines matter, the photo decides where they are.
    Returns the Clean Outline base, the aligned AI guide and the band around it (PNGs, base64).
    `fresh=true` asks the AI again instead of reusing the result for an identical upload.
    """
    data = await image.read()
    key = hashlib.sha256(data).hexdigest()
    if not fresh:
        hit = _cache_get(key)
        if hit:
            return {**hit, "cached": True}

    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    color = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if gray is None or color is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    _take_quota(_client_key(request))

    smooth = await asyncio.to_thread(clean_outline_base, gray)
    h, w = smooth.shape
    jpeg_b64, aspect_key, pad = _model_input(color, w, h)

    attempts, best = [], None
    async with httpx.AsyncClient(timeout=150) as client:
        for model in MODELS:
            t = time.time()
            try:
                raw = await _ask_model(client, model, jpeg_b64, aspect_key)
                guide = _guide_from(raw, pad, w, h)
                res = await asyncio.to_thread(_align, guide, smooth)
                if res is None:
                    raise RuntimeError(f"{model}: drawing was empty")
                aligned, fit, coverage, transform = res
            except HTTPException:
                raise
            except Exception as e:
                attempts.append({"model": model, "error": str(e)[:200], "secs": round(time.time() - t, 1)})
                continue
            reframed = (max(abs(transform["scale_x"] - 1), abs(transform["scale_y"] - 1)) > MAX_SCALE_FIX
                        or abs(transform["dx"]) > MAX_SHIFT_FIX * w or abs(transform["dy"]) > MAX_SHIFT_FIX * h)
            attempt = {"model": model, "fit": round(fit, 3), "coverage": round(coverage, 3),
                       "reframed": reframed, "secs": round(time.time() - t, 1)}
            attempts.append(attempt)
            if fit < MIN_FIT or reframed:
                continue                        # re-composed or misplaced: never use it
            if best is None or coverage > best[2]:
                best = (aligned, fit, coverage, transform, model)
            if coverage >= GOOD_COVERAGE:
                break

    if best is None:
        raise HTTPException(status_code=502, detail="The AI couldn't produce a drawing that lines up with this photo. "
                                                    "Please try again, or use Clean Outline.")
    aligned, fit, coverage, transform, model = best
    tol = transform["tolerance_px"]
    band = (cv2.distanceTransform(255 - aligned, cv2.DIST_L2, 5) <= tol).astype(np.uint8) * 255
    result = {
        "width": w, "height": h, "model": model,
        "fit": round(fit, 3), "coverage": round(coverage, 3), "transform": transform, "attempts": attempts,
        "base_png": _png_b64(smooth), "guide_png": _png_b64(255 - aligned), "band_png": _png_b64(band),
    }
    _cache_put(key, result)
    return {**result, "cached": False}
