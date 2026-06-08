/* ============================================================
   Dashboard advanced controls
   - Line Drawing: client-side edge / sketch / crosshatch / architectural
     styles (ported from frontend/index.html) rendered to #dash-canvas.
   - Canvas Size + Aspect Ratio + Auto Calibrate.
   - Slider value displays (Detail / Line Weight).
   - Download JSON of the most recently generated Strokes shapes.
   - Engine select + Load Result As radio are read by the engine wiring.
   ============================================================ */
(function () {
   "use strict";

   const $ = (sel) => document.querySelector(sel);

   // ---------- 1. Line Drawing ----------------------------------------------
   const styleSel    = $("#lineDrawingStyle");
   const detailEl    = $("#lineDrawingDetail");
   const detailVal   = $("#ldDetailVal");
   const weightEl    = $("#lineDrawingWeight");
   const weightVal   = $("#ldWeightVal");
   const gridEl      = $("#lineDrawingGrid");
   const outlineEl   = $("#lineDrawingOutlineOnly");
   const generateBtn = $("#generateLineDrawingBtn");
   const downloadBtn = $("#downloadLineDrawingBtn");
   const statusEl    = $("#lineDrawingStatus");
   const sourceBadge = $("#lineDrawingSource");

   const mainCanvas  = $("#dash-canvas");
   const dropZone    = $("#dash-drop-zone");
   const mainImageInput = $("#dash-image-input"); // the single shared uploader above

   let fullSizeCanvas = null;

   function setStatus(msg, kind) {
      if (!statusEl) return;
      statusEl.textContent = msg || "";
      statusEl.classList.remove("text-theme-primary", "text-secondary-3", "text-neutral-4");
      if (kind === "error")      statusEl.classList.add("text-secondary-3");
      else if (kind === "info")  statusEl.classList.add("text-theme-primary");
      else                       statusEl.classList.add("text-neutral-4");
   }

   // Reflect whether an image has been uploaded to the shared canvas above.
   function currentSourceFile() {
      if (window.DashGeo && typeof window.DashGeo.getUploadedFile === "function") {
         return window.DashGeo.getUploadedFile();
      }
      return (mainImageInput && mainImageInput.files && mainImageInput.files[0]) || null;
   }
   function updateSourceBadge() {
      if (!sourceBadge) return;
      const f = currentSourceFile();
      const dot = sourceBadge.querySelector("span");
      sourceBadge.childNodes[sourceBadge.childNodes.length - 1].nodeValue =
         f ? ` ${f.name.length > 22 ? f.name.slice(0, 21) + "…" : f.name}` : " No image yet";
      if (dot) { dot.classList.toggle("bg-theme-primary", !!f); dot.classList.toggle("bg-neutral-4", !f); }
   }
   if (mainImageInput) mainImageInput.addEventListener("change", () => setTimeout(updateSourceBadge, 50));
   if (dropZone) dropZone.addEventListener("drop", () => setTimeout(updateSourceBadge, 100));
   updateSourceBadge();

   if (detailEl && detailVal) {
      detailEl.addEventListener("input", () => { detailVal.textContent = detailEl.value; });
   }
   if (weightEl && weightVal) {
      weightEl.addEventListener("input", () => { weightVal.textContent = weightEl.value; });
   }

   if (generateBtn) {
      generateBtn.addEventListener("click", async () => {
         const file = currentSourceFile();
         if (!file) { setStatus("Upload an image in the preview above first.", "error"); return; }
         if (!mainCanvas) { setStatus("Preview canvas missing.", "error"); return; }

         setStatus("Generating line drawing…", "info");
         generateBtn.disabled = true;

         try {
            const img = await loadImage(file);
            const result = processImage(
               img,
               styleSel ? styleSel.value : "detailed",
               parseInt(detailEl && detailEl.value, 10) || 80,
               parseInt(weightEl && weightEl.value, 10) || 2,
               !!(gridEl && gridEl.checked),
               !!(outlineEl && outlineEl.checked)
            );
            fullSizeCanvas = result.canvas;

            mainCanvas.width  = result.width;
            mainCanvas.height = result.height;
            mainCanvas.getContext("2d").drawImage(result.canvas, 0, 0);
            mainCanvas.classList.remove("hidden");
            if (dropZone) dropZone.style.display = "none";

            if (downloadBtn) downloadBtn.classList.remove("hidden");
            setStatus(`Line drawing generated (${result.width}×${result.height})`);
         } catch (err) {
            console.error(err);
            setStatus("Error: " + (err && err.message || err), "error");
         }
         generateBtn.disabled = false;
      });
   }

   if (downloadBtn) {
      downloadBtn.addEventListener("click", () => {
         if (!fullSizeCanvas) return;
         const a = document.createElement("a");
         a.download = "line-drawing-" + Date.now() + ".png";
         a.href = fullSizeCanvas.toDataURL("image/png");
         document.body.appendChild(a);
         a.click();
         a.remove();
      });
   }

   function loadImage(file) {
      return new Promise((resolve, reject) => {
         const reader = new FileReader();
         reader.onload = (e) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = e.target.result;
         };
         reader.onerror = reject;
         reader.readAsDataURL(file);
      });
   }

   // --------- Image-processing kernels (ported 1:1 from the original frontend)
   function processImage(img, style, detail, lineWeight, showGrid, outlineOnly) {
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;

      const src = document.createElement("canvas");
      src.width = w; src.height = h;
      src.getContext("2d").drawImage(img, 0, 0);
      const gray = toGrayscale(src.getContext("2d").getImageData(0, 0, w, h));

      let result;
      if (outlineOnly) {
         result = extractOutlineOnly(gray, w, h, detail, lineWeight);
      } else {
         switch (style) {
            case "stencil":      result = stencilOutline(gray, w, h, detail, lineWeight); break;
            case "sketch":       result = pencilSketch(gray, w, h, detail, lineWeight); break;
            case "crosshatch":   result = crosshatchFast(gray, w, h, detail, lineWeight); break;
            case "architectural":result = architecturalFast(gray, w, h, detail, lineWeight); break;
            default:             result = detailedSketch(gray, w, h, detail, lineWeight);
         }
      }
      if (showGrid) addGridLines(result, w, h);

      const out = document.createElement("canvas");
      out.width = w; out.height = h;
      const octx = out.getContext("2d");
      const od = octx.createImageData(w, h);
      for (let i = 0; i < result.length; i++) {
         const v = result[i] | 0;
         od.data[i * 4]     = v;
         od.data[i * 4 + 1] = v;
         od.data[i * 4 + 2] = v;
         od.data[i * 4 + 3] = 255;
      }
      octx.putImageData(od, 0, 0);
      return { canvas: out, width: w, height: h };
   }

   function toGrayscale(imageData) {
      const d = imageData.data, len = imageData.width * imageData.height;
      const gray = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
         gray[i] = (d[i * 4] * 77 + d[i * 4 + 1] * 150 + d[i * 4 + 2] * 29) >> 8;
      }
      return gray;
   }

   function stencilOutline(gray, w, h, detail, lineWeight) {
      const len = w * h;
      const blurR = Math.max(1, ((100 - detail) / 20) | 0);
      const blurred = boxBlurFast(gray, w, h, blurR);
      const gx = new Int16Array(len), gy = new Int16Array(len), mag = new Uint8Array(len);
      for (let y = 1; y < h - 1; y++) {
         for (let x = 1; x < w - 1; x++) {
            const i = y * w + x;
            const px = -blurred[i-w-1] + blurred[i-w+1] - 2*blurred[i-1] + 2*blurred[i+1] - blurred[i+w-1] + blurred[i+w+1];
            const py = -blurred[i-w-1] - 2*blurred[i-w] - blurred[i-w+1] + blurred[i+w-1] + 2*blurred[i+w] + blurred[i+w+1];
            gx[i] = px; gy[i] = py;
            mag[i] = Math.min(255, (Math.sqrt(px*px + py*py) * 0.5) | 0);
         }
      }
      const thin = new Uint8Array(len);
      for (let y = 1; y < h - 1; y++) {
         for (let x = 1; x < w - 1; x++) {
            const i = y * w + x, m = mag[i];
            if (m < 10) continue;
            const angle = Math.atan2(gy[i], gx[i]) * 180 / Math.PI;
            let n1, n2;
            if ((angle >= -22.5 && angle < 22.5) || angle >= 157.5 || angle < -157.5) { n1 = mag[i-1]; n2 = mag[i+1]; }
            else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)) { n1 = mag[i-w+1]; n2 = mag[i+w-1]; }
            else if ((angle >= 67.5 && angle < 112.5) || (angle >= -112.5 && angle < -67.5)) { n1 = mag[i-w]; n2 = mag[i+w]; }
            else { n1 = mag[i-w-1]; n2 = mag[i+w+1]; }
            if (m >= n1 && m >= n2) thin[i] = m;
         }
      }
      const lowT = Math.max(10, 50 - (detail / 3) | 0);
      const highT = lowT * 2;
      const edges = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
         if (thin[i] >= highT) edges[i] = 255;
         else if (thin[i] >= lowT) edges[i] = 128;
      }
      let changed = true;
      while (changed) {
         changed = false;
         for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
               const i = y * w + x;
               if (edges[i] === 128) {
                  if (edges[i-w-1] === 255 || edges[i-w] === 255 || edges[i-w+1] === 255 ||
                      edges[i-1] === 255 || edges[i+1] === 255 ||
                      edges[i+w-1] === 255 || edges[i+w] === 255 || edges[i+w+1] === 255) {
                     edges[i] = 255; changed = true;
                  }
               }
            }
         }
      }
      for (let i = 0; i < len; i++) if (edges[i] === 128) edges[i] = 0;

      let finalEdges = edges;
      if (lineWeight > 1) finalEdges = dilateFast(edges, w, h, lineWeight);

      const result = new Uint8Array(len);
      for (let i = 0; i < len; i++) result[i] = finalEdges[i] > 0 ? 0 : 255;
      return result;
   }

   function extractOutlineOnly(gray, w, h, detail, lineWeight) {
      const len = w * h;
      const blurR = Math.max(1, ((110 - detail) / 15) | 0);
      const blurred = boxBlurFast(gray, w, h, blurR);
      const edges = sobelFast(blurred, w, h);
      const threshold = Math.max(15, 60 - (detail * 0.4)) | 0;
      const result = new Uint8Array(len);
      for (let i = 0; i < len; i++) result[i] = edges[i] > threshold ? 0 : 255;
      if (lineWeight > 1) {
         const inv = new Uint8Array(len);
         for (let i = 0; i < len; i++) inv[i] = 255 - result[i];
         const dil = dilateFast(inv, w, h, lineWeight);
         for (let i = 0; i < len; i++) result[i] = 255 - dil[i];
      }
      return result;
   }

   function pencilSketch(gray, w, h, detail, lineWeight) {
      const len = w * h;
      const inv = new Uint8Array(len);
      for (let i = 0; i < len; i++) inv[i] = 255 - gray[i];
      const blurR = Math.max(1, ((110 - detail) / 10) | 0);
      const blurred = boxBlurFast(inv, w, h, blurR);
      const result = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
         const b = blurred[i];
         result[i] = b >= 255 ? 255 : Math.min(255, (gray[i] << 8) / (256 - b)) | 0;
      }
      const edges = sobelFast(gray, w, h);
      for (let i = 0; i < len; i++) result[i] = Math.max(0, result[i] - (edges[i] >> 2)) | 0;
      return result;
   }

   function detailedSketch(gray, w, h, detail, lineWeight) {
      const len = w * h;
      const enhanced = localContrastFast(gray, w, h);
      const edges1 = sobelFast(enhanced, w, h);
      const edges2 = laplacianFast(enhanced, w, h);
      const inv = new Uint8Array(len);
      for (let i = 0; i < len; i++) inv[i] = 255 - gray[i];
      const blurR = Math.max(1, ((100 - detail) / 15) | 0);
      const blurred = boxBlurFast(inv, w, h, blurR);
      const shade = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
         const b = blurred[i];
         shade[i] = b >= 255 ? 255 : Math.min(255, (gray[i] << 8) / (256 - b)) | 0;
      }
      const result = new Uint8Array(len);
      const ef = detail / 80;
      for (let i = 0; i < len; i++) {
         const edgeVal = Math.min((edges1[i] + (edges2[i] >> 1)) * ef, 200) | 0;
         result[i] = Math.max(0, shade[i] - edgeVal) | 0;
      }
      if (lineWeight > 1) {
         const ri = new Uint8Array(len);
         for (let i = 0; i < len; i++) ri[i] = 255 - result[i];
         const dil = dilateFast(ri, w, h, lineWeight - 1);
         for (let i = 0; i < len; i++) result[i] = 255 - dil[i];
      }
      return result;
   }

   function crosshatchFast(gray, w, h, detail) {
      const len = w * h;
      const edges = sobelFast(gray, w, h);
      const result = new Uint8Array(len); result.fill(255);
      const spacing = Math.max(2, 6 - (detail / 25) | 0);
      const sp2 = Math.max(1, spacing >> 1);
      for (let y = 0; y < h; y++) {
         const yOff = y * w;
         for (let x = 0; x < w; x++) {
            const i = yOff + x;
            const lum = gray[i];
            let val = 255;
            if (lum < 200 && (x + y) % spacing === 0) val = 180;
            if (lum < 150 && Math.abs(x - y) % spacing === 0) val = Math.min(val, 140);
            if (lum < 100 && (x + y) % sp2 === 0) val = Math.min(val, 100);
            if (lum < 50) val = Math.min(val, 60);
            if (edges[i] > 50) val = Math.min(val, 255 - Math.min(edges[i], 200));
            result[i] = val;
         }
      }
      return result;
   }

   function architecturalFast(gray, w, h, detail, lineWeight) {
      const len = w * h;
      const blurred = boxBlurFast(gray, w, h, 2);
      const edges = sobelFast(blurred, w, h);
      const blockSize = Math.max(11, 31 - (detail / 4) | 0) | 1;
      const adaptive = adaptiveThresholdFast(blurred, w, h, blockSize, 3);
      const result = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
         const edgeVal = edges[i] > 40 ? 0 : 255;
         result[i] = Math.min(adaptive[i], edgeVal);
      }
      if (lineWeight > 1) {
         const inv = new Uint8Array(len);
         for (let i = 0; i < len; i++) inv[i] = 255 - result[i];
         const dil = dilateFast(inv, w, h, lineWeight);
         for (let i = 0; i < len; i++) result[i] = 255 - dil[i];
      }
      return result;
   }

   function boxBlurFast(src, w, h, r) {
      if (r < 1) return src.slice();
      let result = src;
      for (let pass = 0; pass < 2; pass++) {
         result = boxBlurH(result, w, h, r);
         result = boxBlurV(result, w, h, r);
      }
      return result;
   }
   function boxBlurH(src, w, h, r) {
      const dst = new Uint8Array(w * h);
      const iarr = 1 / (r + r + 1);
      for (let y = 0; y < h; y++) {
         let ti = y * w, li = ti, ri = ti + r;
         const fv = src[ti], lv = src[ti + w - 1];
         let val = (r + 1) * fv;
         for (let j = 0; j < r; j++) val += src[ti + j];
         for (let j = 0; j <= r; j++) { val += src[ri++] - fv; dst[ti++] = (val * iarr + 0.5) | 0; }
         for (let j = r + 1; j < w - r; j++) { val += src[ri++] - src[li++]; dst[ti++] = (val * iarr + 0.5) | 0; }
         for (let j = w - r; j < w; j++) { val += lv - src[li++]; dst[ti++] = (val * iarr + 0.5) | 0; }
      }
      return dst;
   }
   function boxBlurV(src, w, h, r) {
      const dst = new Uint8Array(w * h);
      const iarr = 1 / (r + r + 1);
      for (let x = 0; x < w; x++) {
         let ti = x, li = ti, ri = ti + r * w;
         const fv = src[ti], lv = src[ti + w * (h - 1)];
         let val = (r + 1) * fv;
         for (let j = 0; j < r; j++) val += src[ti + j * w];
         for (let j = 0; j <= r; j++) { val += src[ri] - fv; dst[ti] = (val * iarr + 0.5) | 0; ri += w; ti += w; }
         for (let j = r + 1; j < h - r; j++) { val += src[ri] - src[li]; dst[ti] = (val * iarr + 0.5) | 0; li += w; ri += w; ti += w; }
         for (let j = h - r; j < h; j++) { val += lv - src[li]; dst[ti] = (val * iarr + 0.5) | 0; li += w; ti += w; }
      }
      return dst;
   }
   function sobelFast(g, w, h) {
      const r = new Uint8Array(w * h);
      for (let y = 1; y < h - 1; y++) {
         for (let x = 1; x < w - 1; x++) {
            const i = y * w + x;
            const gx = -g[i-w-1] + g[i-w+1] - 2*g[i-1] + 2*g[i+1] - g[i+w-1] + g[i+w+1];
            const gy = -g[i-w-1] - 2*g[i-w] - g[i-w+1] + g[i+w-1] + 2*g[i+w] + g[i+w+1];
            r[i] = Math.min(255, Math.sqrt(gx*gx + gy*gy) | 0);
         }
      }
      return r;
   }
   function laplacianFast(g, w, h) {
      const r = new Uint8Array(w * h);
      for (let y = 1; y < h - 1; y++) {
         for (let x = 1; x < w - 1; x++) {
            const i = y * w + x;
            const v = Math.abs(-g[i-w] - g[i-1] + 4*g[i] - g[i+1] - g[i+w]);
            r[i] = Math.min(255, v);
         }
      }
      return r;
   }
   function localContrastFast(g, w, h) {
      const r = new Uint8Array(w * h);
      const blurL = boxBlurFast(g, w, h, 8);
      for (let i = 0; i < g.length; i++) {
         const diff = g[i] - blurL[i] + 128;
         r[i] = Math.max(0, Math.min(255, diff));
      }
      return r;
   }
   function adaptiveThresholdFast(g, w, h, blockSize, c) {
      const r = new Uint8Array(w * h);
      const radius = blockSize >> 1;
      const blurred = boxBlurFast(g, w, h, radius);
      for (let i = 0; i < g.length; i++) r[i] = g[i] > blurred[i] - c ? 255 : 0;
      return r;
   }
   function dilateFast(e, w, h, iter) {
      let cur = e;
      for (let it = 0; it < iter; it++) {
         const r = new Uint8Array(w * h);
         for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
               const i = y * w + x;
               r[i] = Math.max(cur[i-w-1], cur[i-w], cur[i-w+1], cur[i-1], cur[i], cur[i+1], cur[i+w-1], cur[i+w], cur[i+w+1]);
            }
         }
         cur = r;
      }
      return cur;
   }
   function addGridLines(r, w, h) {
      const gridSize = Math.max(50, (Math.min(w, h) / 8) | 0);
      for (let x = gridSize; x < w; x += gridSize) {
         for (let y = 0; y < h; y++) r[y * w + x] = Math.min(r[y * w + x], 200);
      }
      for (let y = gridSize; y < h; y += gridSize) {
         for (let x = 0; x < w; x++) r[y * w + x] = Math.min(r[y * w + x], 200);
      }
   }

   // ---------- 2. Canvas Size + Aspect Ratio + Auto Calibrate ----------------
   const widthInput   = $("#canvasWidthInput");
   const heightInput  = $("#canvasHeightInput");
   const ratioSelect  = $("#aspectRatioSelect");
   const calibrateBtn = $("#autoCalibrateBtn");
   const calibrateInfo = $("#calibrateInfo");

   const RATIOS = {
      "1:1":  [1, 1],   "4:3":  [4, 3],   "3:4":  [3, 4],
      "16:9": [16, 9],  "9:16": [9, 16],
      "3:2":  [3, 2],   "2:3":  [2, 3],
   };

   function clampSize(v) { return Math.max(100, Math.min(4000, parseInt(v, 10) || 100)); }

   function autoCalibrate() {
      if (!ratioSelect || !widthInput || !heightInput) return;
      const choice = ratioSelect.value;
      if (choice === "free") {
         if (calibrateInfo) calibrateInfo.textContent = "Free mode — width and height are used exactly as entered.";
         return;
      }
      const r = RATIOS[choice]; if (!r) return;
      const w = clampSize(widthInput.value);
      heightInput.value = clampSize(Math.round(w * r[1] / r[0]));
      widthInput.value = w;
      if (calibrateInfo) calibrateInfo.textContent = `Output: ${widthInput.value} × ${heightInput.value} px (${choice})`;
   }
   // Calibration is manual: only the Calibrate button applies the ratio to height.
   if (calibrateBtn) calibrateBtn.addEventListener("click", autoCalibrate);
   if (ratioSelect) ratioSelect.addEventListener("change", () => {
      if (ratioSelect.value === "free" && calibrateInfo) {
         calibrateInfo.textContent = "Free mode — width and height are used exactly as entered.";
      } else if (calibrateInfo) {
         calibrateInfo.textContent = "Press Calibrate to set the height for this ratio.";
      }
   });

   // Read by external callers (engine wiring).
   window.getChosenCanvasSize = function () {
      return {
         width:  clampSize(widthInput && widthInput.value),
         height: clampSize(heightInput && heightInput.value),
      };
   };

   // ---------- 3. Download JSON of current Strokes shapes --------------------
   const downloadJsonBtn = $("#downloadJsonBtn");
   const STORE_KEY = "geomagic:state";

   function getStoredState() {
      try { return JSON.parse(sessionStorage.getItem(STORE_KEY) || "null"); } catch (_) { return null; }
   }
   function setStoredState(s) {
      try { sessionStorage.setItem(STORE_KEY, JSON.stringify(s)); } catch (_) {}
   }

   if (downloadJsonBtn) {
      downloadJsonBtn.addEventListener("click", () => {
         const saved = getStoredState();
         if (!saved || !Array.isArray(saved.baseShapes) || !saved.baseShapes.length) {
            alert("Generate shapes first — then Download JSON will export them.");
            return;
         }
         const payload = {
            width: saved.w || (mainCanvas && mainCanvas.width) || null,
            height: saved.h || (mainCanvas && mainCanvas.height) || null,
            logic: saved.logic || "original",
            shapes: saved.baseShapes,
         };
         const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
         const url = URL.createObjectURL(blob);
         const a = document.createElement("a");
         a.href = url;
         a.download = "geomagic-shapes-" + Date.now() + ".json";
         document.body.appendChild(a);
         a.click();
         a.remove();
         URL.revokeObjectURL(url);
      });
   }

   // ---------- 4. Order Shapes From Json (base/detail loaders) ---------------
   const baseFileInput   = $("#baseFileInput");
   const detailFileInput = $("#detailFileInput");
   const baseFileName    = $("#baseFileName");
   const detailFileName  = $("#detailFileName");
   const baseFileStatus  = $("#baseFileStatus");
   const detailFileStatus = $("#detailFileStatus");

   let detailShapes = []; // loaded detail JSON, used by Enhance
   let detailW = 0, detailH = 0; // detail JSON's own coordinate space (may differ from base)

   // Same shape-trace + render as dashboard-geometrize.js — duplicated locally
   // so this module can repaint #dash-canvas without reaching into the other IIFE.
   function tracePath(ctx, s) {
      const d = s.data;
      ctx.beginPath();
      switch (s.type) {
         case 0: { const [x1,y1,x2,y2]=d; ctx.rect(Math.min(x1,x2),Math.min(y1,y2),Math.abs(x2-x1),Math.abs(y2-y1)); return "fill"; }
         case 1: { const [x1,y1,x2,y2,ang=0]=d; const cx=(x1+x2)/2,cy=(y1+y2)/2,w=Math.abs(x2-x1),h=Math.abs(y2-y1); const r=(ang||0)*Math.PI/180,co=Math.cos(r),si=Math.sin(r); const pts=[[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]].map(([px,py])=>[cx+px*co-py*si,cy+px*si+py*co]); ctx.moveTo(pts[0][0],pts[0][1]); for(let i=1;i<4;i++) ctx.lineTo(pts[i][0],pts[i][1]); ctx.closePath(); return "fill"; }
         case 2: { const [a,b,c,dd,e,f]=d; ctx.moveTo(a,b); ctx.lineTo(c,dd); ctx.lineTo(e,f); ctx.closePath(); return "fill"; }
         case 3: { const [cx,cy,rx,ry]=d; ctx.ellipse(cx,cy,Math.abs(rx),Math.abs(ry),0,0,Math.PI*2); return "fill"; }
         case 4: { const [cx,cy,rx,ry,ang=0]=d; ctx.ellipse(cx,cy,Math.abs(rx),Math.abs(ry),(ang||0)*Math.PI/180,0,Math.PI*2); return "fill"; }
         case 5: { const [cx,cy,rad]=d; ctx.arc(cx,cy,Math.abs(rad),0,Math.PI*2); return "fill"; }
         case 6: { const [x1,y1,x2,y2]=d; ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); return "stroke"; }
         case 7: { const [x1,y1,cx,cy,x2,y2]=d; ctx.moveTo(x1,y1); ctx.quadraticCurveTo(cx,cy,x2,y2); return "stroke"; }
      }
      return "fill";
   }

   function renderShapes(shapes) {
      if (!mainCanvas) return;
      const ctx = mainCanvas.getContext("2d");
      ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
      for (const s of shapes) {
         const [r, g, b, a = 255] = s.color || [0, 0, 0, 255];
         const col = `rgba(${r},${g},${b},${a / 255})`;
         const mode = tracePath(ctx, s);
         if (mode === "fill") { ctx.fillStyle = col; ctx.fill(); }
         else { ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.stroke(); }
      }
   }

   function shapesFromPayload(raw) {
      if (Array.isArray(raw)) return raw;
      if (raw && Array.isArray(raw.shapes)) return raw.shapes;
      throw new Error("JSON has no shape array");
   }

   function canvasFromShapes(shapes) {
      let w = 0, h = 0;
      const bg = shapes.find((s) => s.type === 0);
      if (bg && bg.data.length >= 4) {
         w = Math.max(Math.abs(bg.data[2] - bg.data[0]) || bg.data[2] || 0, 0);
         h = Math.max(Math.abs(bg.data[3] - bg.data[1]) || bg.data[3] || 0, 0);
      }
      if (!w || !h) {
         for (const s of shapes) {
            for (let i = 0; i < s.data.length; i += 2) {
               w = Math.max(w, s.data[i] || 0);
               h = Math.max(h, s.data[i + 1] || 0);
            }
         }
      }
      return { width: Math.max(1, Math.round(w)), height: Math.max(1, Math.round(h)) };
   }

   // Repaint the canvas from the single source of truth (geometrize), falling
   // back to a local render only if the engine module isn't present.
   function repaintBase() {
      if (window.DashGeo) { window.DashGeo.render(); return; }
      const saved = getStoredState();
      if (saved && saved.baseShapes && saved.baseShapes.length) renderShapes(saved.baseShapes);
   }

   if (baseFileInput) {
      baseFileInput.addEventListener("change", async (e) => {
         const file = e.target.files && e.target.files[0];
         if (!file) return;
         if (baseFileName) baseFileName.textContent = file.name;
         try {
            const text = await file.text();
            const shapes = shapesFromPayload(JSON.parse(text));
            const size = canvasFromShapes(shapes);
            if (window.DashGeo) {
               // Hand shapes to geometrize so it owns the canvas + selection.
               window.DashGeo.loadShapes(shapes, size.width, size.height);
            } else {
               if (mainCanvas) {
                  mainCanvas.width = size.width;
                  mainCanvas.height = size.height;
                  mainCanvas.classList.remove("hidden");
                  if (dropZone) dropZone.style.display = "none";
               }
               renderShapes(shapes);
               setStoredState({
                  baseShapes: shapes, logic: "original", colorOrder: null,
                  limit: shapes.length, w: size.width, h: size.height,
               });
            }
            if (baseFileStatus) baseFileStatus.textContent = `Loaded ${shapes.length} shapes`;
         } catch (err) {
            if (baseFileStatus) baseFileStatus.textContent = "Error: " + (err && err.message || err);
         }
      });
   }

   if (detailFileInput) {
      detailFileInput.addEventListener("change", async (e) => {
         const file = e.target.files && e.target.files[0];
         if (!file) return;
         if (detailFileName) detailFileName.textContent = file.name;
         try {
            const text = await file.text();
            const raw = JSON.parse(text);
            detailShapes = shapesFromPayload(raw);
            // Capture the detail's own coordinate space so Enhance can map regions
            // correctly even when base and detail were generated at different sizes.
            const size = canvasFromShapes(detailShapes);
            detailW = (raw && raw.width) || size.width;
            detailH = (raw && raw.height) || size.height;
            if (detailFileStatus) detailFileStatus.textContent = `Loaded ${detailShapes.length} shapes (${detailW}×${detailH}) — ready to enhance`;
         } catch (err) {
            detailShapes = []; detailW = 0; detailH = 0;
            if (detailFileStatus) detailFileStatus.textContent = "Error: " + (err && err.message || err);
         }
      });
   }

   // ---------- 5. Refining (region selection + enhance) ---------------------
   const selectRegionsBtn = $("#selectRegionsBtn");
   const clearRegionsBtn  = $("#clearRegionsBtn");
   const applyEnhanceBtn  = $("#applyEnhanceBtn");
   const enhanceProgress  = $("#enhanceProgress");
   const enhanceFill      = $("#enhanceProgressFill");
   const enhanceText      = $("#enhanceProgressText");
   const selectLabel      = document.querySelector(".select-regions-label");
   const regionCountValue = $("#regionCountValue");
   const regionCountDot   = $("#regionCountDot");
   const regionCountBadge = $("#regionCountBadge");

   // Regions are simple AABB rectangles in canvas-pixel space: {x1,y1,x2,y2}
   let regions = [];
   let isSelecting = false;
   let drawing = false;
   let dragStart = null;
   let dragCur = null;

   function canvasPt(e) {
      const rect = mainCanvas.getBoundingClientRect();
      return {
         x: (e.clientX - rect.left) * (mainCanvas.width / rect.width),
         y: (e.clientY - rect.top) * (mainCanvas.height / rect.height),
      };
   }

   function refreshBadge() {
      if (regionCountValue) regionCountValue.textContent = String(regions.length);
      if (regionCountDot) {
         regionCountDot.classList.toggle("bg-theme-primary", regions.length > 0);
         regionCountDot.classList.toggle("bg-neutral-4", regions.length === 0);
      }
      if (regionCountBadge) {
         regionCountBadge.classList.toggle("text-theme-primary", regions.length > 0);
         regionCountBadge.classList.toggle("text-neutral-4", regions.length === 0);
      }
   }

   function drawRect(ctx, r, opts) {
      const x = Math.min(r.x1, r.x2), y = Math.min(r.y1, r.y2);
      const w = Math.abs(r.x2 - r.x1), h = Math.abs(r.y2 - r.y1);
      ctx.save();
      ctx.fillStyle = opts.fill;
      ctx.strokeStyle = opts.stroke;
      ctx.lineWidth = opts.lineWidth || Math.max(2, mainCanvas.width / 300);
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      if (opts.label) {
         ctx.fillStyle = opts.stroke;
         const pad = Math.max(4, mainCanvas.width / 200);
         const fs = Math.max(11, mainCanvas.width / 60);
         ctx.font = `bold ${fs}px Inter, system-ui, sans-serif`;
         const tw = ctx.measureText(opts.label).width + pad * 2;
         const th = fs + pad;
         ctx.fillRect(x, y - th, tw, th);
         ctx.fillStyle = "#ffffff";
         ctx.fillText(opts.label, x + pad, y - pad / 2 - 2);
      }
      ctx.restore();
   }

   function paintRegions() {
      if (!mainCanvas) return;
      const ctx = mainCanvas.getContext("2d");
      repaintBase();
      regions.forEach((r, i) => {
         drawRect(ctx, r, {
            fill: "rgba(43, 133, 255, 0.18)",
            stroke: "rgba(43, 133, 255, 1)",
            label: `#${i + 1}`,
         });
      });
      if (drawing && dragStart && dragCur) {
         ctx.setLineDash([10, 6]);
         drawRect(ctx, { x1: dragStart.x, y1: dragStart.y, x2: dragCur.x, y2: dragCur.y }, {
            fill: "rgba(34, 197, 94, 0.18)",
            stroke: "rgba(34, 197, 94, 1)",
         });
         ctx.setLineDash([]);
      }
      refreshBadge();
   }

   function setSelecting(on) {
      isSelecting = !!on;
      // Lock geometrize's own click-to-select while we're drawing regions.
      if (window.DashGeo) window.DashGeo.setInteractionLocked(isSelecting);
      if (selectRegionsBtn) selectRegionsBtn.classList.toggle("active", isSelecting);
      if (selectLabel) selectLabel.textContent = isSelecting ? "Done Selecting" : "Select Regions";
      if (mainCanvas) mainCanvas.style.cursor = isSelecting ? "crosshair" : "";
      if (!isSelecting) { drawing = false; dragStart = null; dragCur = null; }
      paintRegions();
   }

   if (selectRegionsBtn) selectRegionsBtn.addEventListener("click", () => setSelecting(!isSelecting));
   if (clearRegionsBtn) clearRegionsBtn.addEventListener("click", () => { regions = []; paintRegions(); });

   function rectContains(r, pt) {
      const x1 = Math.min(r.x1, r.x2), x2 = Math.max(r.x1, r.x2);
      const y1 = Math.min(r.y1, r.y2), y2 = Math.max(r.y1, r.y2);
      return pt.x >= x1 && pt.x <= x2 && pt.y >= y1 && pt.y <= y2;
   }

   if (mainCanvas) {
      mainCanvas.addEventListener("mousedown", (e) => {
         if (!isSelecting) return;
         e.preventDefault();
         const p = canvasPt(e);
         // Click on an existing region to delete it.
         for (let i = regions.length - 1; i >= 0; i--) {
            if (rectContains(regions[i], p)) { regions.splice(i, 1); paintRegions(); return; }
         }
         drawing = true; dragStart = p; dragCur = p;
      });
      mainCanvas.addEventListener("mousemove", (e) => {
         if (!isSelecting || !drawing) return;
         dragCur = canvasPt(e);
         paintRegions();
      });
      const finishDrag = () => {
         if (!isSelecting || !drawing) return;
         drawing = false;
         if (dragStart && dragCur) {
            const w = Math.abs(dragCur.x - dragStart.x);
            const h = Math.abs(dragCur.y - dragStart.y);
            if (w > 6 && h > 6) regions.push({ x1: dragStart.x, y1: dragStart.y, x2: dragCur.x, y2: dragCur.y });
         }
         dragStart = null; dragCur = null;
         paintRegions();
      };
      mainCanvas.addEventListener("mouseup", finishDrag);
      mainCanvas.addEventListener("mouseleave", finishDrag);
   }

   // Treat a region as a 4-corner polygon for point-in-region tests.
   function pointInRegion(pt, r) {
      const x1 = Math.min(r.x1, r.x2), x2 = Math.max(r.x1, r.x2);
      const y1 = Math.min(r.y1, r.y2), y2 = Math.max(r.y1, r.y2);
      return pt.x >= x1 && pt.x <= x2 && pt.y >= y1 && pt.y <= y2;
   }

   function shapeCenter(s) {
      const d = s.data;
      switch (s.type) {
         case 0: return { x: (d[0]+d[2])/2, y: (d[1]+d[3])/2 };
         case 1: return { x: (d[0]+d[2])/2, y: (d[1]+d[3])/2 };
         case 2: return { x: (d[0]+d[2]+d[4])/3, y: (d[1]+d[3]+d[5])/3 };
         case 3: case 4: case 5: return { x: d[0], y: d[1] };
         case 6: return { x: (d[0]+d[2])/2, y: (d[1]+d[3])/2 };
         case 7: return { x: (d[0]+d[2]+d[4])/3, y: (d[1]+d[3]+d[5])/3 };
      }
      return { x: d[0] || 0, y: d[1] || 0 };
   }

   // Scale a shape from the detail coordinate space into the base/canvas space.
   // (Base and detail JSONs can be generated at different resolutions.)
   function scaleShape(s, sx, sy) {
      const d = s.data, r = (v) => Math.round(v);
      const avg = (sx + sy) / 2;
      let nd;
      switch (s.type) {
         case 0: case 2: case 6: case 7: // all-point shapes: scale x by sx, y by sy
            nd = d.map((v, i) => r(v * (i % 2 === 0 ? sx : sy))); break;
         case 1: // rotated rect [x1,y1,x2,y2,angle] — keep angle
            nd = [r(d[0]*sx), r(d[1]*sy), r(d[2]*sx), r(d[3]*sy), d[4]]; break;
         case 3: // ellipse [cx,cy,rx,ry]
            nd = [r(d[0]*sx), r(d[1]*sy), r(d[2]*sx), r(d[3]*sy)]; break;
         case 4: // rotated ellipse [cx,cy,rx,ry,angle]
            nd = [r(d[0]*sx), r(d[1]*sy), r(d[2]*sx), r(d[3]*sy), d[4]]; break;
         case 5: // circle [cx,cy,radius]
            nd = [r(d[0]*sx), r(d[1]*sy), r(d[2]*avg)]; break;
         default:
            nd = d.map((v, i) => r(v * (i % 2 === 0 ? sx : sy)));
      }
      return { type: s.type, data: nd, color: s.color, score: s.score };
   }

   function setProgress(pct, msg) {
      if (!enhanceProgress) return;
      enhanceProgress.classList.remove("hidden");
      if (enhanceFill) enhanceFill.style.width = Math.max(0, Math.min(100, pct)) + "%";
      if (enhanceText && msg) enhanceText.textContent = msg;
   }
   function hideProgress(delay) {
      if (!enhanceProgress) return;
      setTimeout(() => enhanceProgress.classList.add("hidden"), delay || 0);
   }

   if (applyEnhanceBtn) {
      applyEnhanceBtn.addEventListener("click", async () => {
         const hasBase = window.DashGeo
            ? window.DashGeo.hasShapes()
            : !!(getStoredState() && getStoredState().baseShapes && getStoredState().baseShapes.length);
         if (!hasBase) { alert("Generate shapes or import a Base JSON first."); return; }
         if (!regions.length) { alert("Select at least one region first — click \"Select Regions\" and drag a box on the canvas."); return; }

         // Mode A — no Detail JSON: re-geometrize the selected regions directly from
         // the working image (like the original's selective-shapes icon). No extra file.
         if (!detailShapes.length) {
            if (!window.DashGeo || typeof window.DashGeo.enhanceRegions !== "function") {
               alert("Import a Detail (High-res) JSON, or generate shapes from an image, to enhance regions.");
               return;
            }
            const rects = regions.slice();
            regions = [];
            setSelecting(false);
            setProgress(5, "Re-generating detail in selected regions…");
            const total = await window.DashGeo.enhanceRegions(rects, (frac, msg) => setProgress(5 + frac * 90, msg));
            setProgress(100, total ? `Done — added ${total} detail shapes in ${rects.length} region(s).` : "No detail could be added there.");
            hideProgress(2500);
            return;
         }

         // Mode B — Detail JSON loaded: pull its shapes that fall inside the regions.
         setProgress(5, "Picking detail shapes inside selected regions…");

         // Map between the detail JSON's coordinate space and the base/canvas space.
         // Regions are drawn in canvas (base) pixels; detail shapes live in detail
         // pixels. We scale each detail shape into base space, then test its center
         // against the regions, so it works even when the two files differ in size.
         const baseW = (mainCanvas && mainCanvas.width)  || detailW || 1;
         const baseH = (mainCanvas && mainCanvas.height) || detailH || 1;
         const sx = (detailW ? baseW / detailW : 1);
         const sy = (detailH ? baseH / detailH : 1);

         // Detail shapes whose (scaled) center falls in ANY selected region get appended.
         const added = [];
         for (let i = 0; i < detailShapes.length; i++) {
            const s = detailShapes[i];
            if (!s || s.type === 0 || !Array.isArray(s.data)) continue; // skip bg / malformed
            const scaled = scaleShape(s, sx, sy);
            const c = shapeCenter(scaled);
            for (const r of regions) {
               if (pointInRegion(c, r)) { added.push(scaled); break; }
            }
            if (i % 250 === 0) setProgress(5 + (i / detailShapes.length) * 80, `Scanning ${i}/${detailShapes.length}`);
         }

         if (!added.length) {
            setProgress(100, "No detail shapes fell inside the selected regions.");
            hideProgress(2500);
            return;
         }

         // Clear the overlay state, unlock geometrize, then hand it the new shapes.
         regions = [];
         setSelecting(false);
         if (window.DashGeo) {
            window.DashGeo.appendShapes(added);
         } else {
            const saved = getStoredState() || {};
            const merged = (saved.baseShapes || []).concat(added);
            setStoredState({
               baseShapes: merged, logic: saved.logic || "original",
               colorOrder: saved.colorOrder || null, limit: merged.length,
               w: saved.w || mainCanvas.width, h: saved.h || mainCanvas.height,
            });
            renderShapes(merged);
         }
         setProgress(100, `Done — added ${added.length} detail shapes.`);
         hideProgress(2500);
      });
   }
})();
