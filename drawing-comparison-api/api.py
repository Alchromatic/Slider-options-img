"""Standalone HTTP wrapper around the client's unchanged comparison algorithm."""
import io
import logging
import os
import threading
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from comparison import Settings, compare, rectify, overlay_camera

app = FastAPI(title="Alchroma Drawing Comparison", version="1.0.0")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000
# Fixed legacy comparison coordinates preserve the previously tested default output.
# These are internal units, not measurements of a physical canvas.
COMPARISON_WIDTH = 300
COMPARISON_HEIGHT = 400
slot = threading.BoundedSemaphore(1)



ERROR_CODES = {400: "BAD_REQUEST", 401: "UNAUTHORIZED", 404: "NOT_FOUND",
               405: "METHOD_NOT_ALLOWED", 413: "IMAGE_TOO_LARGE",
               415: "INVALID_IMAGE", 422: "INVALID_INPUT", 503: "SERVER_BUSY"}


def error_response(status, message, *, code=None, fields=None, headers=None):
    error = {"code": code or ERROR_CODES.get(status, "REQUEST_ERROR"), "message": message}
    if fields:
        error["fields"] = fields
    # detail stays available for existing clients and the upload page.
    return JSONResponse(status_code=status, content={"success": False, "error": error, "detail": message},
                        headers={**(headers or {}), "Cache-Control": "no-store"})


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    return error_response(exc.status_code, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    fields = [{"field": ".".join(str(part) for part in item["loc"]),
               "message": item["msg"]} for item in exc.errors()]
    return error_response(422, "Check the required image fields and comparison settings.", fields=fields)


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    logging.getLogger("uvicorn.error").error("Unexpected API failure", exc_info=exc)
    return error_response(500, "The server could not process the request. Please try again later.",
                          code="INTERNAL_ERROR")


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
    if not slot.acquire(blocking=False):
        raise HTTPException(503, "Comparison is busy. Please retry shortly.", headers={"Retry-After": "2"})
    try:
        reference, drawing = decode(original_image), decode(drawing_image)
        h, w = drawing.shape[:2]
        corrected, transform = rectify(drawing, [[0,0],[w-1,0],[w-1,h-1],[0,h-1]], COMPARISON_WIDTH, COMPARISON_HEIGHT)
        reference = cv2.resize(reference, (corrected.shape[1], corrected.shape[0]), interpolation=cv2.INTER_AREA)
        settings = Settings(tolerance_mm=tolerance_mm, threshold=threshold, thickness=check_thickness)
        result = compare(reference, corrected, COMPARISON_WIDTH, COMPARISON_HEIGHT, settings)
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
