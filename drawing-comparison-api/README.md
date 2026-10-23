# Alchroma drawing comparison API

Standalone service for Android, iOS, Google Cardboard and Meta Quest clients. It accepts two still images and returns a PNG. No comparison history or database is included. `comparison.py` is an unchanged copy of the client-supplied algorithm; the API follows its Streamlit full-frame processing pipeline.

## Local setup (Python 3.12)

Run from this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 for the upload page, or http://127.0.0.1:8000/docs for interactive API documentation. The `samples` folder includes an original and matching, missing-line and shifted drawings.

```sh
curl --fail-with-body http://127.0.0.1:8000/compare \
  -F original_image=@samples/reference.png \
  -F drawing_image=@samples/drawing_shifted_8mm.png \
  --output result.png
python -m pytest tests -q
```

## API contract

- `GET /health`: returns `{"status":"ok"}`.
- `POST /compare`: multipart form fields `original_image` and `drawing_image` (PNG/JPEG).
- Success: raw PNG bytes, `Content-Type: image/png`. Output dimensions equal the drawing dimensions **after applying JPEG EXIF orientation**, even when the reference resolution differs. Transparent input pixels are composited onto white.
- Optional fields: `canvas_width_mm=300`, `canvas_height_mm=400`, `tolerance_mm=2`, `threshold=150`, `check_thickness=false`. Use actual canvas dimensions for meaningful millimetre tolerances. Position checking is always enabled; thickness uses the original algorithm's default 20% width tolerance.
- Headers: `X-Image-Width`, `X-Image-Height`, `X-Comparison-Score` (0–1), `Cache-Control: no-store`.
- Errors return JSON: 401 invalid configured key, 413 image limit exceeded, 415 unsupported/invalid file, 422 missing fields or unsuitable inputs, 503 busy (retry after 2 seconds).
- Limit: 10 MB and 12 million pixels per image; minimum 16 pixels per side. One comparison at a time per worker. Upload parsing can use temporary spool files, cleaned up by the framework. No results are intentionally stored.

## Input preparation and visual behavior

Both images must show the same canvas, cropped to its edges and viewed straight on. This endpoint uses full-frame corners; it does not detect a photographed canvas or automatically correct its perspective. Camera images need suitable cropping/rectification before submission. The algorithm resamples internally to a 1200-pixel long-side canvas, then maps highlights back to the drawing's original resolution. It marks matching drawn lines green and deviations red; dashed gray reference lines show expected strokes, including missing areas. Returning full-size output does not imply processing every source pixel at native resolution.

Unity sends a multipart request and loads the returned PNG into a Texture2D. Display it as a 2D result on mobile or a 2D panel in VR. A tracked AR overlay is not part of this API.

## Railway — after local approval

All runtime files are inside this folder. Set Railway Root Directory to `/drawing-comparison-api` and use its Dockerfile. The container starts `uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}` and supports Railway's injected PORT. Healthcheck path: `/health`. Do not replace another application's service configuration without confirming that the service is dedicated to this API.

Set `ALCHROMA_API_KEY` as a Railway environment variable to require an `X-API-Key` request header. Do not commit secrets. A key embedded in Unity can be extracted; production user authentication, gateway request limits and rate limiting should be decided before a public launch. File limits here are checked after multipart parsing, so they are not a proxy-level total request limit. HTTPS is required for remote use. Hosting and deployment are not performed by this local implementation.

For phone/headset testing on the same Wi-Fi, bind to `0.0.0.0` and use the computer's LAN IP instead of localhost. Local HTTP may require development-only platform network allowances; use HTTPS in release builds.
