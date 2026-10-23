"""Standalone HTTP wrapper around the client's unchanged comparison algorithm."""
import io
import os
import threading
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from comparison import Settings, compare, rectify, overlay_camera

app = FastAPI(title="Alchroma Drawing Comparison", version="1.0.0")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000
slot = threading.BoundedSemaphore(1)


def decode(upload):
    data = upload.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "Each image must be at most 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ("PNG", "JPEG"):
                raise HTTPException(415, "Use PNG or JPEG images.")
            if source.width * source.height > MAX_PIXELS:
                raise HTTPException(413, "Each image must contain at most 12 million pixels.")
            if min(source.size) < 16:
                raise HTTPException(422, "Image dimensions must be at least 16 pixels.")
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return np.array(Image.alpha_composite(background, rgba).convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(415, "The uploaded file is not a valid supported image.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).with_name("index.html").read_text()


@app.post("/compare", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
def compare_images(
    original_image: Annotated[UploadFile, File()],
    drawing_image: Annotated[UploadFile, File()],
    canvas_width_mm: Annotated[float, Form(ge=1, le=10000)] = 300,
    canvas_height_mm: Annotated[float, Form(ge=1, le=10000)] = 400,
    tolerance_mm: Annotated[float, Form(ge=0, le=100)] = 2,
    threshold: Annotated[int, Form(ge=1, le=254)] = 150,
    check_thickness: Annotated[bool, Form()] = False,
    x_api_key: Annotated[str | None, Header()] = None,
):
    """Return a PNG at the EXIF-oriented drawing dimensions. Inputs must show the same cropped canvas."""
    import secrets
    expected = os.getenv("ALCHROMA_API_KEY")
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(401, "Invalid API key.")
    if min(canvas_width_mm, canvas_height_mm) / max(canvas_width_mm, canvas_height_mm) < 0.05:
        raise HTTPException(422, "Canvas aspect ratio must be between 1:20 and 20:1.")
    if not slot.acquire(blocking=False):
        raise HTTPException(503, "Comparison is busy. Please retry shortly.", headers={"Retry-After": "2"})
    try:
        reference, drawing = decode(original_image), decode(drawing_image)
        h, w = drawing.shape[:2]
        corrected, transform = rectify(drawing, [[0,0],[w-1,0],[w-1,h-1],[0,h-1]], canvas_width_mm, canvas_height_mm)
        reference = cv2.resize(reference, (corrected.shape[1], corrected.shape[0]), interpolation=cv2.INTER_AREA)
        settings = Settings(tolerance_mm=tolerance_mm, threshold=threshold, thickness=check_thickness)
        result = compare(reference, corrected, canvas_width_mm, canvas_height_mm, settings)
        output = overlay_camera(drawing, result, transform)
        success, png = cv2.imencode(".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("PNG encoding failed")
        return Response(png.tobytes(), media_type="image/png", headers={
            "Cache-Control": "no-store", "X-Image-Width": str(w), "X-Image-Height": str(h),
            "X-Comparison-Score": str(result["metrics"]["combined"]["score"]),
        })
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        original_image.file.close()
        drawing_image.file.close()
        slot.release()
