#!/usr/bin/env python3
"""
Minimal working example for the /geometrize endpoint.

The key point: send everything as FORM-DATA (files= + data=), NOT json=.
That's the whole reason it was always returning triangles.
"""
import sys
import requests

API_URL = "https://alchromaticdemo.up.railway.app/geometrize"

def geometrize(image_path, shape_type="rectangle", shape_count=200,
               opacity=128, mode="fast"):
    with open(image_path, "rb") as f:
        resp = requests.post(
            API_URL,
            files={"image": f},          # <-- the image upload
            data={                       # <-- settings as FORM fields (not json!)
                "shape_type": shape_type,
                "shape_count": shape_count,
                "opacity": opacity,
                "mode": mode,
            },
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "pic.png"
    shape = sys.argv[2] if len(sys.argv) > 2 else "rectangle"

    result = geometrize(image, shape_type=shape)

    print(f"shape_type requested : {shape}")
    print(f"total_shapes         : {result['total_shapes']}")
    print(f"canvas               : {result['canvas_width']}x{result['canvas_height']}")
    if result["shapes"]:
        print(f"first shape          : {result['shapes'][0]}")