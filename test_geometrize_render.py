#!/usr/bin/env python3
"""
Test for /geometrize_render — the single-call endpoint that uses the real
browser geometrize engine (frontend/geometrize.js via embedded V8) to turn an
image into shapes, then renders them server-side.

Send an image, get an image back. Every knob is a parameter.

Usage:
    # API running:  uvicorn main:app --reload --port 8000
    python test_geometrize_render.py
    python test_geometrize_render.py --image test.png --shape-types circle \
        --shape-count 2400 --width 300 --height 300 --out out.png
    python test_geometrize_render.py --shape-types triangle,circle --shape-count 500 \
        --candidates 30 --mutations 60
"""

import argparse
import base64
import io
import sys
import time

import requests


def main():
    p = argparse.ArgumentParser(description="Test /geometrize_render (JS engine + backend render)")
    p.add_argument("--url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--image", default="test.png", help="Input image")
    p.add_argument("--shape-types", default="circle",
                   help="Shape type(s): names or codes 0-7, comma-separated "
                        "(e.g. 'circle' or 'triangle,ellipse' or '2,5')")
    p.add_argument("--shape-count", type=int, default=2400, help="Number of shapes")
    p.add_argument("--opacity", type=int, default=128, help="Shape alpha 0-255")
    p.add_argument("--candidates", type=int, default=50, help="candidates_per_step (browser=50)")
    p.add_argument("--mutations", type=int, default=100, help="mutations_per_step (browser=100)")
    p.add_argument("--max-resolution", type=int, default=256, help="Working resolution (browser=256)")
    p.add_argument("--width", type=int, default=None, help="Output width (default: source size)")
    p.add_argument("--height", type=int, default=None, help="Output height (default: source size)")
    p.add_argument("--supersample", type=int, default=2, help="Render antialiasing 1-4")
    p.add_argument("--fmt", default="png", choices=["png", "jpeg"], help="Output format")
    p.add_argument("--base64", action="store_true", help="Use base64 JSON response path")
    p.add_argument("--out", default="geometrize_render_output.png", help="Output image path")
    args = p.parse_args()

    endpoint = args.url.rstrip("/") + "/geometrize_render"
    data = {
        "shape_count": args.shape_count,
        "shape_types": args.shape_types,
        "opacity": args.opacity,
        "candidates_per_step": args.candidates,
        "mutations_per_step": args.mutations,
        "max_resolution": args.max_resolution,
        "supersample": args.supersample,
        "fmt": args.fmt,
        "return_base64": str(args.base64).lower(),
    }
    if args.width:
        data["output_width"] = args.width
    if args.height:
        data["output_height"] = args.height

    print(f"POST {endpoint}")
    print(f"  image={args.image}, types={args.shape_types}, count={args.shape_count}, "
          f"candidates={args.candidates}, mutations={args.mutations}, "
          f"max_res={args.max_resolution}, out={args.width}x{args.height}")

    t = time.time()
    try:
        with open(args.image, "rb") as f:
            resp = requests.post(
                endpoint,
                files={"image": (args.image, f, "application/octet-stream")},
                data=data,
                timeout=600,
            )
    except requests.exceptions.ConnectionError:
        print(f"ERROR: could not connect to {args.url}. Is the API running?", file=sys.stderr)
        print("Start it with:  uvicorn main:app --reload --port 8000", file=sys.stderr)
        sys.exit(1)
    dt = (time.time() - t) * 1000

    if resp.status_code != 200:
        print(f"ERROR: HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    if args.base64:
        j = resp.json()
        print(f"  -> {j['canvas_width']}x{j['canvas_height']}, {j['total_shapes']} shapes, "
              f"bg={j.get('background')}  ({dt:.0f} ms total)")
        img_bytes = base64.b64decode(j["image"].split(",", 1)[1])
    else:
        print(f"  -> {resp.headers.get('content-type')}, {len(resp.content)} bytes  ({dt:.0f} ms total)")
        img_bytes = resp.content

    with open(args.out, "wb") as f:
        f.write(img_bytes)
    print(f"Saved: {args.out}")

    try:
        from PIL import Image
        im = Image.open(io.BytesIO(img_bytes))
        print(f"Verified image: {im.format} {im.size} {im.mode}")
    except Exception as e:
        print(f"WARNING: could not verify image: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
