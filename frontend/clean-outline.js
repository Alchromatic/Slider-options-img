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

   // Slider-driven half: hysteresis thresholds, speck removal, thickness.
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

      // Drop 8-connected edge pieces smaller than minSize pixels.
      const edges = new Uint8Array(n);
      const seen = new Uint8Array(n), piece = new Int32Array(n);
      for (let s = 0; s < n; s++) {
         if (map[s] !== 2 || seen[s]) continue;
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
                  if (map[j] === 2 && !seen[j]) { seen[j] = 1; stack[sp++] = j; }
               }
            }
         }
         if (count >= o.minSize) for (let k = 0; k < count; k++) edges[piece[k]] = 1;
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

   // Exact path: upload the original file; the backend returns outline.py's
   // smoothed image as a grayscale PNG (lossless), read back pixel for pixel.
   async function prepareFromServer(apiBase, file) {
      const fd = new FormData();
      fd.append("image", file);
      const res = await fetch(String(apiBase).replace(/\/$/, "") + "/line-drawing/outline-base", { method: "POST", body: fd });
      if (!res.ok) throw new Error("outline-base: HTTP " + res.status);
      const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
      const w = bmp.width, h = bmp.height;
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const cx = c.getContext("2d");
      cx.drawImage(bmp, 0, 0);
      if (bmp.close) bmp.close();
      const d = cx.getImageData(0, 0, w, h).data, smooth = new Uint8Array(w * h);
      for (let i = 0; i < smooth.length; i++) smooth[i] = d[i * 4]; // gray PNG: R = G = B
      return fromSmoothed(smooth, w, h, true);
   }

   const api = { MAX_DIM, DEFAULTS, grayFromRGBA, prepare, prepareImage, prepareFromServer, fromSmoothed, render };
   if (typeof module !== "undefined" && module.exports) module.exports = api;
   else root.CleanOutline = api;
})(typeof window !== "undefined" ? window : this);
