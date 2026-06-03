"""
js_geometrize.py — run the REAL browser geometrize engine (frontend/geometrize.js)
inside Python via an embedded V8 (mini_racer).

This is the same code path the website uses, executed on the same engine (V8),
so generation speed matches the browser. No Go `primitive` binary, no Node server.

Public API:
    generate_shapes(rgba_bytes, w, h, shape_types, count, alpha,
                    candidates_per_step, mutations_per_step) -> dict
        returns {"shapes": [...], "width": w, "height": h, "background": [r,g,b,a]}
"""

import os
import threading

from py_mini_racer import MiniRacer

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEOMETRIZE_JS = os.path.join(_HERE, "frontend", "geometrize.js")

# JS shim appended after the library: a single entry point we can call from Python.
# Uses only the engine's exported globals plus geometrize.Util, which we expose
# below by replacing the library's worker-only `onmessage = ...` line.
_SHIM = r"""
globalThis.__generateShapes = function (pixels, w, h, shapeTypes, count, alpha,
                                        candidates, mutations) {
    var G = globalThis.geometrize;
    var bmp = G.bitmap.Bitmap.createFromByteArray(w, h, pixels);
    var bg = G.Util.getAverageImageColor(bmp);            // packed int rgba
    var runner = new G.runner.ImageRunner(bmp, bg);
    var opts = {
        shapeTypes: shapeTypes,
        alpha: alpha,
        candidateShapesPerStep: candidates,
        shapeMutationsPerStep: mutations
    };
    var out = [];
    for (var i = 0; i < count; i++) {
        var res = runner.step(opts);
        for (var j = 0; j < res.length; j++) {
            var r = res[j];
            out.push({
                type: r.shape.getType(),
                data: r.shape.getRawShapeData(),
                color: [(r.color >> 24) & 255, (r.color >> 16) & 255,
                        (r.color >> 8) & 255, r.color & 255],
                score: r.score
            });
        }
    }
    return {
        shapes: out,
        width: w,
        height: h,
        background: [(bg >> 24) & 255, (bg >> 16) & 255, (bg >> 8) & 255, 255]
    };
};
"""

_ctx = None
_lock = threading.Lock()  # one V8 isolate, shared; serialize generation calls


def _get_ctx() -> MiniRacer:
    """Lazily create and cache the V8 context with the engine + shim loaded."""
    global _ctx
    if _ctx is not None:
        return _ctx
    with _lock:
        if _ctx is not None:
            return _ctx
        with open(_GEOMETRIZE_JS, "r", encoding="utf-8") as f:
            code = f.read()
        # The library's last in-IIFE statement assigns to `onmessage` (a Web Worker
        # global) which throws under V8's strict mode. Replace it with an export of
        # the otherwise-private Util (we need getAverageImageColor).
        code = code.replace(
            "onmessage = GeometrizeWorker.prototype.messageHandler;",
            "$hx_exports.geometrize.Util = geometrize_Util;",
        )
        ctx = MiniRacer()
        ctx.eval(code)
        ctx.eval(_SHIM)
        _ctx = ctx
        return _ctx


def generate_shapes(rgba_bytes, w: int, h: int, shape_types, count: int,
                    alpha: int = 128, candidates_per_step: int = 50,
                    mutations_per_step: int = 100) -> dict:
    """
    Generate shapes from raw RGBA pixel bytes using the browser engine.

    Args:
        rgba_bytes: bytes/bytearray of length w*h*4 (RGBA, row-major).
        w, h: pixel dimensions of rgba_bytes (the working resolution).
        shape_types: list[int] of geometrize shape type codes (0..7).
        count: number of shapes (steps) to generate.
        alpha: shape opacity 0..255.
        candidates_per_step / mutations_per_step: engine quality knobs
            (browser defaults are 50 / 100).

    Returns dict with keys: shapes, width, height, background.
    """
    pixels = list(rgba_bytes)  # MiniRacer JSON-encodes the argument
    ctx = _get_ctx()
    with _lock:
        return ctx.call(
            "__generateShapes",
            pixels, int(w), int(h),
            [int(t) for t in shape_types],
            int(count), int(alpha),
            int(candidates_per_step), int(mutations_per_step),
        )
