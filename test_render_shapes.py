#!/usr/bin/env python3
"""
End-to-end test for the backend rendering flow.

Two steps, both on the backend:
  1. POST an image to  /geometrize     -> get a shape list (JSON)
  2. POST that list to /render_shapes  -> get the rendered image (PNG/JPEG)

This mirrors what the app does, but with rendering moved server-side.

Usage:
    # Make sure the API is running, e.g.:
    #   uvicorn main:app --reload --port 8000
    python test_render_shapes.py
    python test_render_shapes.py --image test.png --shape-type circle \
        --shape-count 2400 --width 300 --height 300 --out out.png

    # Skip step 1 and render a shape list you already have:
    python test_render_shapes.py --shapes my_shapes.json
"""

import argparse
import base64
import io
import json
import sys
import time

import requests


def geometrize(url, image_path, shape_type, shape_count, opacity, mode):
    """Step 1: send an image to /geometrize and return the shape list."""
    endpoint = url.rstrip("/") + "/geometrize"
    print(f"[1/2] POST {endpoint}")
    print(f"      image={image_path}, type={shape_type}, count={shape_count}, mode={mode}")
    t = time.time()
    with open(image_path, "rb") as f:
        resp = requests.post(
            endpoint,
            files={"image": (image_path, f, "application/octet-stream")},
            data={"shape_count": shape_count, "shape_type": shape_type,
                  "opacity": opacity, "mode": mode},
            timeout=600,
        )
    if resp.status_code != 200:
        print(f"ERROR (geometrize): HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    dt = (time.time() - t) * 1000
    print(f"      -> {data['total_shapes']} shapes, "
          f"source canvas {data['canvas_width']}x{data['canvas_height']}  ({dt:.0f} ms)")
    return data["shapes"]


def render(url, shapes, width, height, supersample, fmt, return_base64):
    """Step 2: send the shape list to /render_shapes and return image bytes."""
    endpoint = url.rstrip("/") + "/render_shapes"
    payload = {"shapes": shapes, "supersample": supersample,
               "fmt": fmt, "return_base64": return_base64}
    if width:
        payload["canvas_width"] = width
    if height:
        payload["canvas_height"] = height
    print(f"[2/2] POST {endpoint}")
    print(f"      {len(shapes)} shapes, size={width}x{height}, supersample={supersample}, fmt={fmt}")
    t = time.time()
    resp = requests.post(endpoint, json=payload, timeout=120)
    if resp.status_code != 200:
        print(f"ERROR (render): HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    dt = (time.time() - t) * 1000
    if return_base64:
        data = resp.json()
        print(f"      -> base64 image {data['canvas_width']}x{data['canvas_height']}  ({dt:.0f} ms)")
        return base64.b64decode(data["image"].split(",", 1)[1])
    print(f"      -> {resp.headers.get('content-type')}, {len(resp.content)} bytes  ({dt:.0f} ms)")
    return resp.content


def main():
    p = argparse.ArgumentParser(description="Geometrize an image then render it on the backend")
    p.add_argument("--url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--image", default="test.png", help="Image to geometrize (step 1)")
    p.add_argument("--shapes", help="Skip step 1: render this JSON shape list instead "
                                    "(a list, or an object with a 'shapes' key)")
    p.add_argument("--shape-type", default="circle",
                   help="Shape type for geometrize (circle, triangle, ellipse, rectangle, ...)")
    p.add_argument("--shape-count", type=int, default=2400, help="Number of shapes to generate")
    p.add_argument("--opacity", type=int, default=128, help="Shape opacity 0-255")
    p.add_argument("--mode", default="fast", choices=["fast", "quality"],
                   help="Geometrize mode. 'fast' downscales to 256px like the browser (much "
                        "faster); 'quality' runs primitive on the full-res image (slow).")
    p.add_argument("--width", type=int, default=300, help="Render canvas width (px)")
    p.add_argument("--height", type=int, default=300, help="Render canvas height (px)")
    p.add_argument("--supersample", type=int, default=2, help="Antialiasing factor 1-4")
    p.add_argument("--fmt", default="png", choices=["png", "jpeg"], help="Output format")
    p.add_argument("--base64", action="store_true", help="Use the base64 JSON response path")
    p.add_argument("--out", default="render_output.png", help="Output image path")
    args = p.parse_args()

    try:
        # Step 1: get shapes (from /geometrize, or from a local file if --shapes given).
        if args.shapes:
            with open(args.shapes, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            shapes = loaded["shapes"] if isinstance(loaded, dict) and "shapes" in loaded else loaded
            print(f"[1/2] Loaded {len(shapes)} shapes from {args.shapes} (skipping /geometrize)")
        else:
            shapes = geometrize(args.url, args.image, args.shape_type,
                                args.shape_count, args.opacity, args.mode)

        # Step 2: render the shapes.
        img_bytes = render(args.url, shapes, args.width, args.height,
                           args.supersample, args.fmt, args.base64)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: could not connect to {args.url}. Is the API running?", file=sys.stderr)
        print("Start it with:  uvicorn main:app --reload --port 8000", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "wb") as f:
        f.write(img_bytes)
    print(f"Saved rendered image to: {args.out}")

    # Sanity check: confirm it's a valid image.
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(img_bytes))
        print(f"Verified image: {im.format} {im.size} {im.mode}")
    except Exception as e:
        print(f"WARNING: could not verify image: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
