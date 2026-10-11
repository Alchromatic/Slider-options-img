/* ============================================================
   Clean Outline — tracing line art from a photo
   The client's OpenCV recipe (outline.py):
     read as grayscale -> resize to <= 1600px (INTER_AREA)
     -> bilateralFilter(9, 60, 60) -> Canny(low, high, L2gradient)
     -> drop 8-connected edge pieces smaller than min_size px
     -> dilate with an elliptical thickness x thickness kernel
     -> black lines on white.
   prepareFromServer() gets the slider-independent first half from the
   backend (/line-drawing/outline-base), computed by OpenCV itself: a
   browser's JPEG decode and colour handling differ from OpenCV's, so
   only that keeps the result identical to outline.py. render() then
   applies the sliders here (bit-exact with OpenCV) in a few ms.
   prepareImage() is an in-browser fallback for when the server is
   unreachable; it is close to outline.py but not identical.
   ============================================================ */
(function (root) {
   "use strict";

   const MAX_DIM = 1600;
   const DEFAULTS = { low: 40, high: 100, minSize: 25, thickness: 1 };

   // RGBA pixels -> gray, with OpenCV's BGR2GRAY fixed-point weights.
   function grayFromRGBA(d, n) {
      const g = new Uint8Array(n);
      for (let i = 0, p = 0; i < n; i++, p += 4) {
         g[i] = (d[p] * 4899 + d[p + 1] * 9617 + d[p + 2] * 1868 + 8192) >> 14;
      }
      return g;
   }

   // OpenCV INTER_AREA weights along one axis (computeResizeAreaTab):
   // each output cell averages the source pixels it covers, by overlap.
   function areaTab(ssize, dsize, scale) {
      const tab = [];
      for (let dx = 0; dx < dsize; dx++) {
         const fsx1 = dx * scale, fsx2 = fsx1 + scale;
         const cell = Math.min(scale, ssize - fsx1);
         let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
         sx2 = Math.min(sx2, ssize - 1);
         sx1 = Math.min(sx1, sx2);
         if (sx1 - fsx1 > 1e-3) tab.push([dx, sx1 - 1, (sx1 - fsx1) / cell]);
         for (let sx = sx1; sx < sx2; sx++) tab.push([dx, sx, 1 / cell]);
         if (fsx2 - sx2 > 1e-3) tab.push([dx, sx2, Math.min(Math.min(fsx2 - sx2, 1), cell) / cell]);
      }
      return tab;
   }

   // Downscale gray (w x h) to (dw x dh) like cv2.resize(..., INTER_AREA).
   function resizeArea(src, w, h, dw, dh, scale) {
      const xt = areaTab(w, dw, scale), yt = areaTab(h, dh, scale);
      const rows = new Float32Array(h * dw);           // horizontal pass
      for (let y = 0; y < h; y++) {
         const so = y * w, ro = y * dw;
         for (const [dx, sx, a] of xt) rows[ro + dx] += src[so + sx] * a;
      }
      const acc = new Float32Array(dw * dh);           // vertical pass
      for (const [dy, sy, a] of yt) {
         const ro = sy * dw, ao = dy * dw;
         for (let x = 0; x < dw; x++) acc[ao + x] += rows[ro + x] * a;
      }
      const out = new Uint8Array(dw * dh);
      for (let i = 0; i < out.length; i++) out[i] = Math.min(255, Math.max(0, Math.round(acc[i])));
      return out;
   }

   // cv2.bilateralFilter(gray, 9, 60, 60): smooths texture (water ripples,
   // grain) while keeping strong boundaries. Border: reflect-101.
   function bilateral(src, w, h) {
      const radius = 4, sigma = 60;
      const cc = -0.5 / (sigma * sigma), sc = -0.5 / (sigma * sigma);
      const colorW = new Float32Array(256);
      for (let i = 0; i < 256; i++) colorW[i] = Math.exp(i * i * cc);
      const offX = [], offY = [], spaceW = [];
      for (let i = -radius; i <= radius; i++) {
         for (let j = -radius; j <= radius; j++) {
            const r = Math.sqrt(i * i + j * j);
            if (r > radius) continue;
            offY.push(i); offX.push(j); spaceW.push(Math.exp(r * r * sc));
         }
      }
      const K = spaceW.length;
      // Padded copy so the inner loop needs no bounds checks.
      const pw = w + 2 * radius, ph = h + 2 * radius;
      const pad = new Uint8Array(pw * ph);
      const refl = (i, n) => (n === 1 ? 0 : i < 0 ? -i : i >= n ? 2 * n - 2 - i : i);
      for (let y = 0; y < ph; y++) {
         const sy = refl(y - radius, h) * w;
         for (let x = 0; x < pw; x++) pad[y * pw + x] = src[sy + refl(x - radius, w)];
      }
      const ofs = new Int32Array(K);
      for (let k = 0; k < K; k++) ofs[k] = offY[k] * pw + offX[k];
      const sw = Float32Array.from(spaceW);
      const out = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
         let p = (y + radius) * pw + radius, o = y * w;
         for (let x = 0; x < w; x++, p++, o++) {
            const v0 = pad[p];
            let sum = 0, wsum = 0;
            for (let k = 0; k < K; k++) {
               const v = pad[p + ofs[k]];
               const wt = sw[k] * colorW[v > v0 ? v - v0 : v0 - v];
               sum += v * wt; wsum += wt;
            }
            out[o] = Math.round(sum / wsum);
         }
      }
      return out;
   }

   // Canny, threshold-independent half: Sobel gradients (replicated border),
   // L2 magnitude (dx²+dy²) and OpenCV's non-maximum suppression, so the
   // slider-driven half only has to threshold and link.
   function gradientPeaks(g, w, h) {
      const n = w * h;
      const mag = new Int32Array(n), dxs = new Int16Array(n), dys = new Int16Array(n);
      for (let y = 0; y < h; y++) {
         const yu = (y > 0 ? y - 1 : 0) * w, yc = y * w, yd = (y < h - 1 ? y + 1 : h - 1) * w;
         for (let x = 0; x < w; x++) {
            const xl = x > 0 ? x - 1 : 0, xr = x < w - 1 ? x + 1 : w - 1;
            const dx = (g[yu + xr] + 2 * g[yc + xr] + g[yd + xr]) - (g[yu + xl] + 2 * g[yc + xl] + g[yd + xl]);
            const dy = (g[yd + xl] + 2 * g[yd + x] + g[yd + xr]) - (g[yu + xl] + 2 * g[yu + x] + g[yu + xr]);
            dxs[yc + x] = dx; dys[yc + x] = dy;
            mag[yc + x] = dx * dx + dy * dy;
         }
      }
      // peak[i] = 1 where the pixel is a ridge along its gradient direction.
      const TG22 = 13573; // tan(22.5°) << 15, as in OpenCV
      const M = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : mag[y * w + x]);
      const peak = new Uint8Array(n);
      for (let y = 0; y < h; y++) {
         for (let x = 0; x < w; x++) {
            const i = y * w + x, m = mag[i];
            if (m === 0) continue;
            const xs = dxs[i], ys = dys[i];
            const ax = Math.abs(xs), ay = Math.abs(ys) * 32768;
            const tg22x = ax * TG22;
            let ok;
            if (ay < tg22x) ok = m > M(x - 1, y) && m >= M(x + 1, y);
            else if (ay > tg22x + ax * 65536) ok = m > M(x, y - 1) && m >= M(x, y + 1);
            else {
               const s = (xs ^ ys) < 0 ? -1 : 1;
               ok = m > M(x - s, y - 1) && m > M(x + s, y + 1);
            }
            if (ok) peak[i] = 1;
         }
      }
      return { mag, peak };
   }

   // OpenCV's elliptical structuring element (cv2.MORPH_ELLIPSE, k x k),
   // returned as [dx, dy] offsets from its anchor (k/2, k/2).
   function ellipseOffsets(k) {
      if (k <= 1) return [[0, 0]];
      const r = k >> 1, c = k >> 1, inv = 1 / (r * r), pts = [];
      for (let i = 0; i < k; i++) {
         const dy = i - r;
         if (Math.abs(dy) > r) continue;
         const dx = Math.round(c * Math.sqrt((r * r - dy * dy) * inv));
         for (let j = Math.max(c - dx, 0); j < Math.min(c + dx + 1, k); j++) pts.push([j - c, i - r]);
      }
      return pts;
   }

   // Binary dilation by a (2r+1)-square, via running sums (rows, then columns).
   function boxDilate(src, w, h, r) {
      const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
         const o = y * w;
         let run = 0;
         for (let x = 0; x < Math.min(r, w); x++) run += src[o + x];
         for (let x = 0; x < w; x++) {
            if (x + r < w) run += src[o + x + r];
            if (x - r - 1 >= 0) run -= src[o + x - r - 1];
            tmp[o + x] = run > 0 ? 1 : 0;
         }
      }
      for (let x = 0; x < w; x++) {
         let run = 0;
         for (let y = 0; y < Math.min(r, h); y++) run += tmp[y * w + x];
         for (let y = 0; y < h; y++) {
            if (y + r < h) run += tmp[(y + r) * w + x];
            if (y - r - 1 >= 0) run -= tmp[(y - r - 1) * w + x];
            out[y * w + x] = run > 0 ? 1 : 0;
         }
      }
      return out;
   }

   // Zhang-Suen thinning: strokes of any width -> 1 px centre lines (1 = ink).
   function thinLines(src, w, h) {
      const img = Uint8Array.from(src);
      const del = [];
      let changed = true;
      while (changed) {
         changed = false;
         for (let step = 0; step < 2; step++) {
            del.length = 0;
            for (let y = 1; y < h - 1; y++) {
               for (let x = 1; x < w - 1; x++) {
                  const i = y * w + x;
                  if (!img[i]) continue;
                  const p2 = img[i - w], p3 = img[i - w + 1], p4 = img[i + 1], p5 = img[i + w + 1];
                  const p6 = img[i + w], p7 = img[i + w - 1], p8 = img[i - 1], p9 = img[i - w - 1];
                  const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                  if (B < 2 || B > 6) continue;
                  let A = 0;
                  if (!p2 && p3) A++; if (!p3 && p4) A++; if (!p4 && p5) A++; if (!p5 && p6) A++;
                  if (!p6 && p7) A++; if (!p7 && p8) A++; if (!p8 && p9) A++; if (!p9 && p2) A++;
                  if (A !== 1) continue;
                  if (step === 0 ? (p2 * p4 * p6 || p4 * p6 * p8) : (p2 * p4 * p8 || p2 * p6 * p8)) continue;
                  del.push(i);
               }
            }
            if (del.length) { changed = true; for (const i of del) img[i] = 0; }
         }
      }
      return img;
   }

   // Slider-driven half: hysteresis thresholds, speck removal, thickness.
   // With an AI guide, opts.aiMode picks how much the AI decides:
   //   "photo"  only the photo's own edges inside the guide band (pixel-exact)
   //   "snap"   the AI's strokes, each replaced by the photo's edge where the
   //            photo clearly has one under it, kept as drawn where it doesn't
   //   "ai"     the AI's drawing as drawn (aligned to the photo, thinned to 1 px)
   // Returns gray pixels: 0 = line, 255 = paper.
   function render(state, opts) {
      const o = Object.assign({}, DEFAULTS, opts || {});
      const { w, h, mag, peak } = state;
      const n = w * h;
      let low = Math.max(0, o.low), high = Math.max(0, o.high);
      if (low > high) [low, high] = [high, low];     // cv2.Canny swaps too
      const lo = Math.floor(Math.min(32767, low) ** 2), hi = Math.floor(Math.min(32767, high) ** 2);

      // 2 = edge, 1 = weak candidate, 0 = nothing; then grow edges into weak.
      const map = new Uint8Array(n);
      const stack = new Int32Array(n);
      let sp = 0;
      for (let i = 0; i < n; i++) {
         if (!peak[i] || mag[i] <= lo) continue;
         if (mag[i] > hi) { map[i] = 2; stack[sp++] = i; } else map[i] = 1;
      }
      while (sp) {
         const i = stack[--sp], x = i % w, y = (i - x) / w;
         for (let ny = y - 1; ny <= y + 1; ny++) {
            if (ny < 0 || ny >= h) continue;
            for (let nx = x - 1; nx <= x + 1; nx++) {
               if (nx < 0 || nx >= w) continue;
               const j = ny * w + nx;
               if (map[j] === 1) { map[j] = 2; stack[sp++] = j; }
            }
         }
      }

      const aiMode = state.guide ? (o.aiMode || "ai") : null;
      const tol = Math.max(1, Math.round(state.tol || 10));
      if (aiMode) state.thinGuide = state.thinGuide || thinLines(state.guide, w, h);

      // AI Line Art: keep only the photo's edges that the AI guide marks.
      if (state.band && aiMode !== "ai") for (let i = 0; i < n; i++) if (!state.band[i]) map[i] = 0;
      // "ai": the drawing itself is the candidate set (sliders Faint/Strong don't apply).
      if (aiMode === "ai") for (let i = 0; i < n; i++) map[i] = state.thinGuide[i] ? 2 : 0;

      // 8-connected pieces of `src` (value 2) -> edges, keeping those with at
      // least minSize pixels; `test(piece, count)` can veto or replace a piece.
      const edges = new Uint8Array(n);
      const seen = new Uint8Array(n), piece = new Int32Array(n);
      function keepPieces(src, test) {
         seen.fill(0);
         for (let s = 0; s < n; s++) {
            if (src[s] !== 2 || seen[s]) continue;
            let count = 0; sp = 0;
            stack[sp++] = s; seen[s] = 1;
            while (sp) {
               const i = stack[--sp], x = i % w, y = (i - x) / w;
               piece[count++] = i;
               for (let ny = y - 1; ny <= y + 1; ny++) {
                  if (ny < 0 || ny >= h) continue;
                  for (let nx = x - 1; nx <= x + 1; nx++) {
                     if (nx < 0 || nx >= w) continue;
                     const j = ny * w + nx;
                     if (src[j] === 2 && !seen[j]) { seen[j] = 1; stack[sp++] = j; }
                  }
               }
            }
            if (count >= o.minSize && (!test || test(piece, count))) for (let k = 0; k < count; k++) edges[piece[k]] = 1;
         }
      }
      keepPieces(map);

      // "snap": each AI stroke that the photo doesn't back up (under 60% of it
      // within tol of a kept photo edge) stays in as drawn, thinned to 1 px.
      if (aiMode === "snap") {
         const covered = boxDilate(edges, w, h, tol);
         const guide2 = new Uint8Array(n);
         for (let i = 0; i < n; i++) guide2[i] = state.guide[i] ? 2 : 0;
         const photoEdges = edges.slice();
         keepPieces(guide2, (pc, count) => {
            let hit = 0;
            for (let k = 0; k < count; k++) hit += covered[pc[k]];
            return hit < 0.6 * count;
         });
         // the vetoed test only chose pieces; thin the ones added so they match
         for (let i = 0; i < n; i++) if (edges[i] && !photoEdges[i] && !state.thinGuide[i]) edges[i] = 0;
      }

      // Thicken: dst(p) = max over kernel of src(p + offset).
      const out = new Uint8Array(n).fill(255);
      const offs = ellipseOffsets(Math.max(1, o.thickness | 0));
      for (let i = 0; i < n; i++) {
         if (!edges[i]) continue;
         const x = i % w, y = (i - x) / w;
         for (const [ox, oy] of offs) {
            const tx = x - ox, ty = y - oy;
            if (tx >= 0 && ty >= 0 && tx < w && ty < h) out[ty * w + tx] = 0;
         }
      }
      return out;
   }

   // State for render() from an already shrunk + smoothed grayscale image.
   function fromSmoothed(smooth, w, h, exact) {
      const { mag, peak } = gradientPeaks(smooth, w, h);
      return { w, h, mag, peak, exact: !!exact };
   }

   // In-browser first half (fallback): shrink + smooth, then fromSmoothed.
   function prepare(gray, w, h) {
      let W = w, H = h, g = gray;
      const scale = Math.min(1, MAX_DIM / Math.max(w, h));
      if (scale < 1) {
         W = Math.round(w * scale); H = Math.round(h * scale);
         g = resizeArea(gray, w, h, W, H, 1 / scale);
      }
      return fromSmoothed(bilateral(g, W, H), W, H, false);
   }

   // Fallback for the pages: an <img> (or canvas) -> prepared state.
   function prepareImage(img) {
      const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const cx = c.getContext("2d");
      cx.drawImage(img, 0, 0);
      return prepare(grayFromRGBA(cx.getImageData(0, 0, w, h).data, w * h), w, h);
   }

   // A grayscale PNG (Blob) read back pixel for pixel: { w, h, gray }.
   async function readGrayPng(blob) {
      const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
      const w = bmp.width, h = bmp.height;
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const cx = c.getContext("2d");
      cx.drawImage(bmp, 0, 0);
      if (bmp.close) bmp.close();
      const d = cx.getImageData(0, 0, w, h).data, gray = new Uint8Array(w * h);
      for (let i = 0; i < gray.length; i++) gray[i] = d[i * 4]; // gray PNG: R = G = B
      return { w, h, gray };
   }
   const b64Blob = (b64) => new Blob([Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0))], { type: "image/png" });

   // Exact path: upload the original file; the backend returns outline.py's
   // smoothed image as a grayscale PNG (lossless).
   async function prepareFromServer(apiBase, file) {
      const fd = new FormData();
      fd.append("image", file);
      const res = await fetch(String(apiBase).replace(/\/$/, "") + "/line-drawing/outline-base", { method: "POST", body: fd });
      if (!res.ok) throw new Error("outline-base: HTTP " + res.status);
      const { w, h, gray } = await readGrayPng(await res.blob());
      return fromSmoothed(gray, w, h, true);
   }

   // AI Line Art: the backend asks an AI model which outlines matter and returns
   // the same smoothed base plus the AI's aligned guide and the band around it.
   // `fresh` asks the AI again instead of reusing the result for this file.
   async function prepareAiGuide(apiBase, file, fresh, token) {
      const fd = new FormData();
      fd.append("image", file);
      if (fresh) fd.append("fresh", "true");
      let res;
      try {
         res = await fetch(String(apiBase).replace(/\/$/, "") + "/line-drawing/ai-guide", {
            method: "POST", body: fd, headers: token ? { Authorization: "Bearer " + token } : {},
         });
      } catch (e) {
         throw new Error("Couldn't reach the server for AI Line Art. Check your connection and try again.");
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
         const err = new Error(res.status === 404
            ? "AI Line Art isn't available on this server yet (it needs the latest backend deployed)."
            : data.detail || "AI Line Art failed (HTTP " + res.status + ")");
         err.status = res.status;
         throw err;
      }
      const base = await readGrayPng(b64Blob(data.base_png));
      const band = await readGrayPng(b64Blob(data.band_png));
      const guide = await readGrayPng(b64Blob(data.guide_png));
      const state = fromSmoothed(base.gray, base.w, base.h, true);
      state.band = band.gray.map((v) => (v > 127 ? 1 : 0));
      state.guide = guide.gray.map((v) => (v < 128 ? 1 : 0));
      state.tol = data.transform && data.transform.tolerance_px;
      state.ai = { model: data.model, fit: data.fit, coverage: data.coverage, cached: data.cached };
      return state;
   }

   const api = { MAX_DIM, DEFAULTS, grayFromRGBA, prepare, prepareImage, prepareFromServer, prepareAiGuide, fromSmoothed, render };
   if (typeof module !== "undefined" && module.exports) module.exports = api;
   else root.CleanOutline = api;
})(typeof window !== "undefined" ? window : this);
