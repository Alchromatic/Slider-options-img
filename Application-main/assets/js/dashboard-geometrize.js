/* ============================================================
   Dashboard geometrize — runs the engine IN THE BROWSER
   - The geometrize engine (assets/js/geometrize.js, same file as
     the original app) runs in a Web Worker. Shapes are generated
     step-by-step client-side (no backend, no network delay) and
     drawn on a <canvas> as they build up.
   - Click a shape on the canvas to select it (hit-testing),
     then delete it. Save exports the canvas to PNG.
   ============================================================ */
(function () {
   "use strict";

   // The engine (assets/js/geometrize.js) is loaded via a <script> tag in the
   // page and exposed as window.geometrize — so generation runs entirely
   // in-browser with NO network, fetch, or worker.
   function engine() {
      const g = window.geometrize;
      if (!g || !g.runner || !g.runner.ImageRunner || !g.exporter) {
         throw new Error("geometrize engine not loaded (assets/js/geometrize.js)");
      }
      return g;
   }

   const $ = (sel) => document.querySelector(sel);

   // ---- DOM ----
   const fileInput   = $("#dash-image-input");
   const dropZone    = $("#dash-drop-zone");
   const previewWrap = $("#dash-preview-wrap");
   const canvas      = $("#dash-canvas");
   const statusEl    = $("#dash-status");

   const selBar     = $("#dash-selection");
   const selType    = $("#dash-sel-type");
   const selColor   = $("#dash-sel-color");
   const selRgba    = $("#dash-sel-rgba");
   const selDelete  = $("#dash-sel-delete");
   const selClear   = $("#dash-sel-clear");

   const btnReset    = $("#btn-reset");
   const btnRandom   = $("#btn-random");
   const btnStep     = $("#btn-step");
   const btnGenerate = $("#btn-generate");
   const btnSave     = $("#btn-save");

   const shapeAddedEl = $("#shape-added");
   const maxShapeEl   = $("#max-shape-limit");

   const renderingSection = $("#rendering-logic-section");

   const opacityEl   = document.querySelector('input[title="shape-opacity"]');
   const randomEl    = document.querySelector('input[title="random-shapes"]');
   const mutationsEl = document.querySelector('input[title="mutations"]');

   if (!canvas) { console.warn("dashboard-geometrize: canvas not found"); return; }
   const ctx = canvas.getContext("2d");

   // ---- shape type checkboxes (read by label text) -> engine codes 0-7 ----
   const SHAPE_CODE = {
      "rectangles": 0,
      "rotated": 1,
      "triangles": 2,
      "ellipses": 3,
      "rotates ellipses": 4,
      "rotated ellipses": 4,
      "circles": 5,
      "lines": 6,
      "bezier curves": 7,
   };
   const SHAPE_NAME = {
      0: "Rectangle", 1: "Rotated Rectangle", 2: "Triangle", 3: "Ellipse",
      4: "Rotated Ellipse", 5: "Circle", 6: "Line", 7: "Quadratic Bezier",
   };

   function shapeBoxes() {
      const grid = document.getElementById("shape-types-grid");
      if (!grid) return [];
      return Array.from(grid.querySelectorAll("label")).map((label) => {
         const cb = label.querySelector('input[type="checkbox"]');
         const txt = (label.querySelector("span:last-child")?.textContent || "").trim().toLowerCase();
         return { cb, code: SHAPE_CODE[txt] };
      }).filter((x) => x.cb && x.code != null);
   }
   // Array of engine shape-type codes for the checked boxes (default: triangle).
   function selectedShapeCodes() {
      const codes = shapeBoxes().filter((b) => b.cb.checked).map((b) => b.code);
      return codes.length ? codes : [2];
   }

   // ---- state ----
   let uploadedFile = null;
   let shapes = [];            // shapes currently displayed (possibly reordered)
   let baseShapes = [];        // shapes in original generated order
   let currentLogic = "original";
   let customColorOrder = null; // color sequence from the Drag & Drop modal
   let shapeLimit = null;      // how many shapes to draw (null = all)
   let selectedIndex = null;
   let currentCount = 0;       // how many shapes were last requested
   let selectiveMode = false;  // Selective Resolution: draw a region to add detail
   let srDrawing = false;      // true while dragging out the region
   let srPath = [];            // freehand region path (canvas px)

   // carries the generated result across pages (dashboard -> templates)
   const STORE_KEY = "geomagic:state";
   function persistState() {
      try {
         sessionStorage.setItem(STORE_KEY, JSON.stringify({
            baseShapes, logic: currentLogic, colorOrder: customColorOrder,
            limit: shapeLimit, w: canvas.width, h: canvas.height,
         }));
      } catch (_) {}
   }
   function clearState() {
      try { sessionStorage.removeItem(STORE_KEY); } catch (_) {}
   }

   // ---- helpers ----
   const clampInt = (v, lo, hi, dflt) => {
      v = parseInt(v, 10);
      if (isNaN(v)) v = dflt;
      return Math.max(lo, Math.min(hi, v));
   };
   const opacity   = () => clampInt(opacityEl   && opacityEl.value,   0, 255, 180);
   const candidates= () => clampInt(randomEl    && randomEl.value,    1, 300, 50);
   const mutations = () => clampInt(mutationsEl && mutationsEl.value, 1, 300, 100);

   function setStatus(msg) {
      if (!statusEl) return;
      if (msg) { statusEl.textContent = msg; statusEl.classList.remove("hidden"); }
      else statusEl.classList.add("hidden");
   }
   function showCanvas() {
      canvas.classList.remove("hidden");
      if (dropZone) dropZone.style.display = "none";
   }
   function showDropZone() {
      canvas.classList.add("hidden");
      if (dropZone) dropZone.style.display = "";
   }

   // ---- per-shape path tracing (mirrors the original drawPreview) ----
   // Returns "fill" or "stroke" so the same routine drives render + hit-test.
   function tracePath(s) {
      const d = s.data;
      ctx.beginPath();
      switch (s.type) {
         case 0: { // Rectangle [x1,y1,x2,y2]
            const [x1, y1, x2, y2] = d;
            ctx.rect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
            return "fill";
         }
         case 1: { // Rotated Rectangle [x1,y1,x2,y2,angle]
            const [x1, y1, x2, y2, ang = 0] = d;
            const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
            const w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
            const r = (ang || 0) * Math.PI / 180, cos = Math.cos(r), sin = Math.sin(r);
            const pts = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
               .map(([px, py]) => [cx + px * cos - py * sin, cy + px * sin + py * cos]);
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
            ctx.closePath();
            return "fill";
         }
         case 2: { // Triangle [x1,y1,x2,y2,x3,y3]
            const [a, b, c, dd, e, f] = d;
            ctx.moveTo(a, b); ctx.lineTo(c, dd); ctx.lineTo(e, f); ctx.closePath();
            return "fill";
         }
         case 3: { // Ellipse [cx,cy,rx,ry]
            const [cx, cy, rx, ry] = d;
            ctx.ellipse(cx, cy, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2);
            return "fill";
         }
         case 4: { // Rotated Ellipse [cx,cy,rx,ry,angle]
            const [cx, cy, rx, ry, ang = 0] = d;
            ctx.ellipse(cx, cy, Math.abs(rx), Math.abs(ry), (ang || 0) * Math.PI / 180, 0, Math.PI * 2);
            return "fill";
         }
         case 5: { // Circle [cx,cy,rad]
            const [cx, cy, rad] = d;
            ctx.arc(cx, cy, Math.abs(rad), 0, Math.PI * 2);
            return "fill";
         }
         case 6: { // Line [x1,y1,x2,y2]
            const [x1, y1, x2, y2] = d;
            ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
            return "stroke";
         }
         case 7: { // Quadratic Bezier [x1,y1,cx,cy,x2,y2]
            const [x1, y1, cx, cy, x2, y2] = d;
            ctx.moveTo(x1, y1); ctx.quadraticCurveTo(cx, cy, x2, y2);
            return "stroke";
         }
      }
      return "fill";
   }

   // Draw shapes[0 .. upTo). Omit upTo to use the slider limit (or all).
   function render(upTo) {
      const total = shapes.length;
      const n = (upTo != null) ? Math.min(upTo, total)
              : (shapeLimit != null ? Math.min(shapeLimit, total) : total);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < n; i++) {
         const s = shapes[i];
         const [r, g, b, a] = s.color;
         const col = `rgba(${r},${g},${b},${(a == null ? 255 : a) / 255})`;
         const mode = tracePath(s);
         if (mode === "fill") { ctx.fillStyle = col; ctx.fill(); }
         else { ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.stroke(); }
      }

      // selection highlight (only when the full image is shown)
      if (n === total && selectedIndex != null && shapes[selectedIndex]) {
         tracePath(shapes[selectedIndex]);
         ctx.save();
         ctx.lineWidth = Math.max(2, canvas.width / 200);
         ctx.setLineDash([8, 5]);
         ctx.strokeStyle = "#00e5ff";
         ctx.stroke();
         ctx.restore();
      }
   }

   // true while the engine is generating (blocks selection clicks)
   let animating = false;

   // ---- engine helpers (ported from the original frontend) ----

   // Decode an image file into the packed-RGBA buffer the worker expects.
   function imageToBitmapData(file) {
      return new Promise((resolve, reject) => {
         const img = new Image();
         img.onload = () => {
            let width = img.width, height = img.height;
            const maxDim = 256; // working resolution (same as the original app)
            if (width > maxDim || height > maxDim) {
               const ratio = Math.min(maxDim / width, maxDim / height);
               width = Math.floor(width * ratio);
               height = Math.floor(height * ratio);
            }
            const c = document.createElement("canvas");
            c.width = width; c.height = height;
            const cx = c.getContext("2d");
            cx.drawImage(img, 0, 0, width, height);
            const px = cx.getImageData(0, 0, width, height).data;

            const data = new Array(width * height);
            // Weight by alpha so transparent pixels of a PNG don't drag the
            // average toward the canvas-default white (or black), which would
            // otherwise make the engine's starting bg-rect nearly invisible
            // against a light page and the build look "blank".
            let rSum = 0, gSum = 0, bSum = 0, aSum = 0;
            for (let i = 0; i < width * height; i++) {
               const r = px[i * 4], g = px[i * 4 + 1], b = px[i * 4 + 2], a = px[i * 4 + 3];
               rSum += r * a; gSum += g * a; bSum += b * a; aSum += a;
               data[i] = ((r & 255) << 24) | ((g & 255) << 16) | ((b & 255) << 8) | (a & 255);
            }
            const denom = aSum || (width * height * 255); // fully-transparent → safe fallback
            URL.revokeObjectURL(img.src);
            resolve({
               width, height, data,
               originalWidth: img.width, originalHeight: img.height,
               avgColor: [Math.round(rSum / denom), Math.round(gSum / denom), Math.round(bSum / denom), 255],
            });
         };
         img.onerror = reject;
         img.src = URL.createObjectURL(file);
      });
   }

   // Parse the worker's JSON-fragment output and scale shapes to source size.
   function parseAndScaleShapes(jsonString, scaleX, scaleY) {
      try {
         const shapesIn = JSON.parse("[\n" + jsonString + "\n]");
         return shapesIn.map((shape) => {
            const t = shape.type, d = shape.data;
            let sd;
            switch (t) {
               case 0: case 6:
                  sd = [Math.round(d[0] * scaleX), Math.round(d[1] * scaleY), Math.round(d[2] * scaleX), Math.round(d[3] * scaleY)]; break;
               case 1:
                  sd = [Math.round(d[0] * scaleX), Math.round(d[1] * scaleY), Math.round(d[2] * scaleX), Math.round(d[3] * scaleY), d[4]]; break;
               case 2: case 7:
                  sd = [Math.round(d[0] * scaleX), Math.round(d[1] * scaleY), Math.round(d[2] * scaleX), Math.round(d[3] * scaleY), Math.round(d[4] * scaleX), Math.round(d[5] * scaleY)]; break;
               case 3:
                  sd = [Math.round(d[0] * scaleX), Math.round(d[1] * scaleY), Math.round(d[2] * scaleX), Math.round(d[3] * scaleY)]; break;
               case 4:
                  sd = [Math.round(d[0] * scaleX), Math.round(d[1] * scaleY), Math.round(d[2] * scaleX), Math.round(d[3] * scaleY), d[4]]; break;
               case 5: {
                  const avg = (scaleX + scaleY) / 2;
                  sd = [Math.round(d[0] * scaleX), Math.round(d[1] * scaleY), Math.round(d[2] * avg)]; break;
               }
               default: sd = d;
            }
            return { type: t, data: sd, color: shape.color, score: shape.score };
         });
      } catch (e) {
         console.error("parseAndScaleShapes failed:", e);
         return [];
      }
   }

   // Yield to the browser so the canvas repaints. Races rAF against a short
   // timer so generation still progresses if the tab is backgrounded (rAF paused).
   const yieldFrame = () => new Promise((resolve) => {
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(); } };
      requestAnimationFrame(finish);
      setTimeout(finish, 32);
   });

   // topmost shape under (x,y) in canvas pixel space; skip background (index 0)
   function hitTest(x, y) {
      for (let i = shapes.length - 1; i >= 1; i--) {
         const mode = tracePath(shapes[i]);
         if (mode === "fill") {
            if (ctx.isPointInPath(x, y)) return i;
         } else {
            ctx.lineWidth = Math.max(6, canvas.width / 80);
            if (ctx.isPointInStroke(x, y)) return i;
         }
      }
      return null;
   }

   function updateSelectionInfo() {
      if (selectedIndex == null || !shapes[selectedIndex]) {
         if (selBar) selBar.classList.add("hidden");
         return;
      }
      const s = shapes[selectedIndex];
      const [r, g, b, a = 255] = s.color;
      if (selType) selType.textContent = `#${selectedIndex} · ${SHAPE_NAME[s.type] || s.type}`;
      if (selColor) selColor.style.background = `rgba(${r},${g},${b},${a / 255})`;
      if (selRgba) selRgba.textContent = `rgba(${r}, ${g}, ${b}, ${a})`;
      if (selBar) selBar.classList.remove("hidden");

      const tcInput = document.getElementById("trycolorsInput");
      if (tcInput) {
         const hex = ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
         tcInput.value = hex;
         const tcPreview = document.getElementById("trycolorsPreview");
         if (tcPreview) tcPreview.style.background = "#" + hex;
         const tcBtn = document.getElementById("trycolorsBtn");
         if (tcBtn) tcBtn.disabled = false;
      }
   }

   function deleteSelected() {
      if (selectedIndex == null) return;
      const removed = shapes[selectedIndex];
      shapes.splice(selectedIndex, 1);
      const bi = baseShapes.indexOf(removed); // keep deletion across reorders
      if (bi >= 0) baseShapes.splice(bi, 1);
      selectedIndex = null;
      render();
      updateSelectionInfo();
      persistState();
   }

   // ---- canvas selection events ----
   canvas.addEventListener("click", (e) => {
      if (!shapes.length || animating || selectiveMode) return;
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);
      selectedIndex = hitTest(x, y);
      render();
      updateSelectionInfo();
   });
   document.addEventListener("keydown", (e) => {
      if ((e.key === "Delete" || e.key === "Backspace") && selectedIndex != null) {
         e.preventDefault();
         deleteSelected();
      }
   });
   if (selDelete) selDelete.addEventListener("click", deleteSelected);
   if (selClear) selClear.addEventListener("click", () => {
      selectedIndex = null; render(); updateSelectionInfo();
   });

   // ---- image upload ----
   function loadFile(file) {
      if (!file) return;
      uploadedFile = file;
      shapes = [];
      selectedIndex = null;
      currentCount = 0;
      updateSelectionInfo();
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
         const maxd = 1000;
         const ratio = Math.min(maxd / img.width, maxd / img.height, 1);
         canvas.width = Math.max(1, Math.round(img.width * ratio));
         canvas.height = Math.max(1, Math.round(img.height * ratio));
         ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
         URL.revokeObjectURL(url);
         showCanvas();
         setStatus("Image loaded — press Generate");
      };
      img.src = url;
   }
   if (fileInput) fileInput.addEventListener("change", (e) => loadFile(e.target.files[0]));
   if (dropZone) {
      ["dragenter", "dragover"].forEach((ev) =>
         dropZone.addEventListener(ev, (e) => { e.preventDefault(); }));
      dropZone.addEventListener("drop", (e) => {
         e.preventDefault();
         const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
         if (f) loadFile(f);
      });
   }

   // ============================================================
   //  RENDERING LOGIC — shape ordering, ported 1:1 from the backend
   //  (main.py /order_shapes_from_json). Runs in-browser, instant.
   // ============================================================
   function rlCanvasSize(list) {
      let maxX = 0, maxY = 0;
      for (const s of list) {
         if (s.type === 0 && s.data.length >= 4) {
            maxX = Math.max(maxX, s.data[0], s.data[2]);
            maxY = Math.max(maxY, s.data[1], s.data[3]);
         } else if (s.type === 4 && s.data.length >= 4) {
            maxX = Math.max(maxX, s.data[0] + s.data[2]);
            maxY = Math.max(maxY, s.data[1] + s.data[3]);
         }
      }
      return { width: maxX || 1, height: maxY || 1 };
   }
   function rlCenter(s, cv) {
      const d = s.data;
      if (s.type === 0 && d.length >= 4) return [0.5 * (d[0] + d[2]), 0.5 * (d[1] + d[3])];
      if (s.type === 2 && d.length >= 6) return [(d[0] + d[2] + d[4]) / 3, (d[1] + d[3] + d[5]) / 3];
      if (s.type === 6 && d.length >= 4) return [0.5 * (d[0] + d[2]), 0.5 * (d[1] + d[3])];
      if (s.type === 7 && d.length >= 6) return [(d[0] + d[2] + d[4]) / 3, (d[1] + d[3] + d[5]) / 3];
      if (d.length >= 2) return [d[0], d[1]]; // types 1,3,4,5 (and fallbacks)
      return [cv.width / 2, cv.height / 2];
   }
   const rlLum = (c) => (c.length < 3 ? 0 : 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]);
   const rlColorKey = (c) => c.slice(0, 4).map((x) => String(Math.trunc(x))).join(",");
   const rlDistCenter = (s, cv) => Math.hypot(rlCenter(s, cv)[0] - cv.width / 2, rlCenter(s, cv)[1] - cv.height / 2);
   const rlDistEdge = (s, cv) => {
      const [x, y] = rlCenter(s, cv);
      return Math.min(x, cv.width - x, y, cv.height - y);
   };

   // Returns a NEW array (same shape objects) reordered per `logic`.
   function rlOrder(list, logic, colorOrder) {
      const cv = rlCanvasSize(list);
      const bg = list.filter((s) => s.type === 0);
      const others = list.filter((s) => s.type !== 0);
      const by = (keyFn) => others.slice().sort((a, b) => keyFn(a) - keyFn(b));
      switch (logic) {
         case "exterior_to_center": return bg.concat(by((s) => rlDistEdge(s, cv)));
         case "center_to_exterior": return bg.concat(by((s) => rlDistCenter(s, cv)));
         case "top_to_bottom":      return bg.concat(by((s) => rlCenter(s, cv)[1]));
         case "bottom_to_top":      return bg.concat(by((s) => -rlCenter(s, cv)[1]));
         case "light_to_dark":      return bg.concat(by((s) => -rlLum(s.color)));
         case "dark_to_light":      return bg.concat(by((s) => rlLum(s.color)));
         case "frequency_by_color":
         case "frequency_by_color_reverse": {
            const freq = {};
            for (const s of others) { const k = rlColorKey(s.color); freq[k] = (freq[k] || 0) + 1; }
            const asc = logic === "frequency_by_color_reverse";
            const keys = Object.keys(freq).sort((a, b) => asc ? freq[a] - freq[b] : freq[b] - freq[a]);
            const rank = {}; keys.forEach((k, i) => (rank[k] = i));
            return bg.concat(others.slice().sort((a, b) => rank[rlColorKey(a.color)] - rank[rlColorKey(b.color)]));
         }
         case "color_sequence":
         case "custom_sequence": {
            if (!colorOrder || !colorOrder.length) return list.slice(); // needs a palette order
            const idx = {}; colorOrder.forEach((c, i) => (idx[rlColorKey(c)] = i));
            const fb = colorOrder.length + 1;
            return bg.concat(others.slice().sort((a, b) => {
               const ka = idx[rlColorKey(a.color)] ?? fb, kb = idx[rlColorKey(b.color)] ?? fb;
               return ka !== kb ? ka - kb : (-rlLum(a.color)) - (-rlLum(b.color));
            }));
         }
         default: return list.slice(); // "original"
      }
   }

   // ---- rendering logic buttons ----
   const renderingButtons = renderingSection
      ? Array.from(renderingSection.querySelectorAll(".rendering-icon-btn")) : [];

   function setActiveLogic(btn) {
      renderingButtons.forEach((b) => {
         if (b.id !== "selectiveResolutionIcon") b.classList.remove("active");
      });
      if (btn) btn.classList.add("active");
   }

   function applyLogic(logic, btn) {
      // Selective Resolution is a draw-on-canvas tool, handled separately below.
      if (btn && btn.id === "selectiveResolutionIcon") return;
      if (selectiveMode) setSelectiveMode(false);
      if (!baseShapes.length) return;
      setActiveLogic(btn);
      currentLogic = logic || "original";
      shapes = rlOrder(baseShapes, currentLogic, customColorOrder);
      selectedIndex = null;
      render();
      updateSelectionInfo();
      persistState();
   }

   renderingButtons.forEach((btn) => {
      btn.addEventListener("click", () => applyLogic(btn.dataset.logic, btn));
   });

   // ---- Selective Resolution: draw a region, re-geometrize it for more detail ----
   const selectiveIcon = $("#selectiveResolutionIcon");

   function setSelectiveMode(on) {
      selectiveMode = on;
      if (selectiveIcon) selectiveIcon.classList.toggle("active", on);
      canvas.style.cursor = on ? "crosshair" : "";
      if (!on) { srDrawing = false; srPath = []; render(); }
      setStatus(on ? "Draw a region on the image to add detail"
                   : `${shapes.length} shapes — click one to select`);
   }

   if (selectiveIcon) {
      selectiveIcon.addEventListener("click", () => {
         if (animating) return;
         if (!shapes.length) { alert("Generate or load an image first."); return; }
         setSelectiveMode(!selectiveMode);
      });
   }

   function canvasPt(e) {
      const rect = canvas.getBoundingClientRect();
      return {
         x: (e.clientX - rect.left) * (canvas.width / rect.width),
         y: (e.clientY - rect.top) * (canvas.height / rect.height),
      };
   }
   function drawSrPath() {
      if (srPath.length < 2) return;
      ctx.save();
      ctx.strokeStyle = "#2a85ff";
      ctx.lineWidth = Math.max(2, canvas.width / 300);
      ctx.setLineDash([8, 6]);
      ctx.lineJoin = ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(srPath[0].x, srPath[0].y);
      for (let i = 1; i < srPath.length; i++) ctx.lineTo(srPath[i].x, srPath[i].y);
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = "rgba(42,133,255,0.12)";
      ctx.fill();
      ctx.restore();
   }
   canvas.addEventListener("mousedown", (e) => {
      if (!selectiveMode || animating) return;
      srDrawing = true;
      srPath = [canvasPt(e)];
   });
   canvas.addEventListener("mousemove", (e) => {
      if (!selectiveMode || !srDrawing) return;
      srPath.push(canvasPt(e));
      render();
      drawSrPath();
   });
   canvas.addEventListener("mouseup", async () => {
      if (!selectiveMode || !srDrawing) return;
      srDrawing = false;
      if (srPath.length < 3) { srPath = []; render(); return; }
      await processSelectiveResolution();
      srPath = [];
   });

   // Scale engine-space shapes to crop size, then offset into the full canvas.
   function parseScaleOffsetShapes(jsonString, scaleX, scaleY, offX, offY) {
      try {
         const shapesIn = JSON.parse("[\n" + jsonString + "\n]");
         const SX = (v) => Math.round(v * scaleX + offX);
         const SY = (v) => Math.round(v * scaleY + offY);
         const RX = (v) => Math.round(v * scaleX);
         const RY = (v) => Math.round(v * scaleY);
         return shapesIn.map((shape) => {
            const t = shape.type, d = shape.data; let sd;
            switch (t) {
               case 0: case 6:  sd = [SX(d[0]), SY(d[1]), SX(d[2]), SY(d[3])]; break;
               case 1:          sd = [SX(d[0]), SY(d[1]), SX(d[2]), SY(d[3]), d[4]]; break;
               case 2: case 7:  sd = [SX(d[0]), SY(d[1]), SX(d[2]), SY(d[3]), SX(d[4]), SY(d[5])]; break;
               case 3:          sd = [SX(d[0]), SY(d[1]), RX(d[2]), RY(d[3])]; break;
               case 4:          sd = [SX(d[0]), SY(d[1]), RX(d[2]), RY(d[3]), d[4]]; break;
               case 5: { const avg = (scaleX + scaleY) / 2; sd = [SX(d[0]), SY(d[1]), Math.round(d[2] * avg)]; break; }
               default: sd = d;
            }
            return { type: t, data: sd, color: shape.color, score: shape.score };
         });
      } catch (e) { console.error("parseScaleOffsetShapes failed:", e); return []; }
   }

   async function processSelectiveResolution() {
      let G;
      try { G = engine(); } catch (err) { setStatus("Error: " + err.message); return; }

      // bounding box of the drawn path, in canvas (image) pixels
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of srPath) {
         minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
         maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y);
      }
      const pad = 4;
      const bx = Math.max(0, Math.floor(minX - pad));
      const by = Math.max(0, Math.floor(minY - pad));
      const bw = Math.min(canvas.width,  Math.ceil(maxX + pad)) - bx;
      const bh = Math.min(canvas.height, Math.ceil(maxY + pad)) - by;
      if (bw < 8 || bh < 8) { render(); return; }

      // clear the dashed overlay so it isn't baked into the sampled crop
      const prevSel = selectedIndex; selectedIndex = null; render();

      // crop the current render and downscale to a working size for the engine
      const crop = document.createElement("canvas");
      crop.width = bw; crop.height = bh;
      crop.getContext("2d").drawImage(canvas, bx, by, bw, bh, 0, 0, bw, bh);

      const maxDim = 256;
      let ww = bw, wh = bh;
      if (ww > maxDim || wh > maxDim) {
         const r = Math.min(maxDim / ww, maxDim / wh);
         ww = Math.max(1, Math.floor(ww * r));
         wh = Math.max(1, Math.floor(wh * r));
      }
      const work = document.createElement("canvas");
      work.width = ww; work.height = wh;
      const wctx = work.getContext("2d");
      wctx.drawImage(crop, 0, 0, ww, wh);
      const px = wctx.getImageData(0, 0, ww, wh).data;

      const data = new Array(ww * wh);
      let rS = 0, gS = 0, bS = 0;
      for (let i = 0; i < ww * wh; i++) {
         const r = px[i * 4], g = px[i * 4 + 1], b = px[i * 4 + 2], a = px[i * 4 + 3];
         rS += r; gS += g; bS += b;
         data[i] = ((r & 255) << 24) | ((g & 255) << 16) | ((b & 255) << 8) | (a & 255);
      }
      const n = ww * wh || 1;
      const bgPacked = ((Math.round(rS / n) & 255) << 24) | ((Math.round(gS / n) & 255) << 16)
                     | ((Math.round(bS / n) & 255) << 8) | 255;

      const options = {
         shapeTypes: selectedShapeCodes(),
         alpha: opacity(),
         candidateShapesPerStep: 30,
         shapeMutationsPerStep: 50,
      };
      const scaleX = bw / ww, scaleY = bh / wh;

      animating = true; setBusy(true);
      try {
         const runner = new G.runner.ImageRunner({ width: ww, height: wh, data }, bgPacked);
         const STEPS = 40;
         for (let step = 1; step <= STEPS; step++) {
            const results = runner.step(options);
            const json = G.exporter.ShapeJsonExporter.exportShapes(results);
            if (json && json.length > 0) {
               const added = parseScaleOffsetShapes(json, scaleX, scaleY, bx, by);
               shapes.push(...added);
               baseShapes.push(...added);
            }
            if (step % 5 === 0 || step === STEPS) {
               render();
               setStatus(`Adding detail… ${step} / ${STEPS}`);
               await yieldFrame();
            }
         }
      } catch (err) {
         console.error("Selective resolution error:", err);
         setStatus("Selective resolution failed: " + (err.message || err));
      }

      animating = false; setBusy(false);
      setSelectiveMode(false);
      selectedIndex = (prevSel != null && prevSel < shapes.length) ? prevSel : null;
      shapeLimit = shapes.length;
      syncSlider();
      render();
      updateSelectionInfo();
      persistState();
      setStatus(`${shapes.length} shapes — click one to select`);
   }

   // ---- shape-count slider (limits how many shapes are drawn) ----
   const shapeSlider = $("#shapeSlider");
   const shapeSliderLabel = $("#shapeSliderLabel");
   function syncSlider() {
      const total = shapes.length;
      if (shapeLimit == null || shapeLimit > total) shapeLimit = total;
      if (!shapeSlider) return;
      shapeSlider.max = String(total);
      shapeSlider.value = String(shapeLimit);
      if (shapeSliderLabel) shapeSliderLabel.textContent = `Shapes: ${shapeLimit} / ${total}`;
   }
   if (shapeSlider) {
      shapeSlider.addEventListener("input", () => {
         shapeLimit = parseInt(shapeSlider.value, 10);
         if (shapeSliderLabel) shapeSliderLabel.textContent = `Shapes: ${shapeLimit} / ${shapes.length}`;
         render();
         persistState();
      });
   }

   // ---- Drag & Drop Color Order modal (sets the custom_sequence order) ----
   const dragColorBtn = $("#dragColorBtn");
   const dragModal = $("#dragModal");
   const colorPalette = $("#colorPalette");
   const frequencyInfo = $("#frequencyInfo");

   function renumberChips() {
      if (!colorPalette) return;
      [...colorPalette.children].forEach((chip, i) => {
         const n = chip.querySelector(".chip-number");
         if (n) n.textContent = `#${i + 1}`;
      });
   }
   function populateColorPalette() {
      if (!colorPalette) return;
      colorPalette.innerHTML = "";
      const freq = new Map();
      for (const s of baseShapes) {
         if (s.type === 0) continue;
         const k = JSON.stringify(s.color);
         freq.set(k, (freq.get(k) || 0) + 1);
      }
      const list = [...freq.entries()].map(([k, count]) => ({ color: JSON.parse(k), count }));
      list.sort((a, b) => b.count - a.count);
      if (frequencyInfo) frequencyInfo.textContent = `${list.length} unique colors — drag to set the order`;

      list.forEach((item, i) => {
         const chip = document.createElement("div");
         chip.className = "color-chip";
         chip.draggable = true;
         chip.dataset.color = JSON.stringify(item.color);
         const [r, g, b, a = 255] = item.color;
         chip.style.backgroundColor = `rgba(${r},${g},${b},${a / 255})`;
         chip.innerHTML =
            `<div class="chip-number">#${i + 1}</div>` +
            `<div class="rgb-value">${item.color.slice(0, 3).join(", ")}</div>` +
            `<div class="count">${item.count}×</div>`;
         chip.addEventListener("dragstart", () => chip.classList.add("dragging"));
         chip.addEventListener("dragend", () => chip.classList.remove("dragging"));
         chip.addEventListener("dragover", (e) => e.preventDefault());
         chip.addEventListener("drop", (e) => {
            e.preventDefault();
            const dragging = colorPalette.querySelector(".dragging");
            if (dragging && dragging !== chip) {
               const all = [...colorPalette.children];
               if (all.indexOf(dragging) < all.indexOf(chip)) chip.after(dragging);
               else chip.before(dragging);
               renumberChips();
            }
         });
         colorPalette.appendChild(chip);
      });
   }
   function closeDragModal() { if (dragModal) dragModal.style.display = "none"; }
   if (dragColorBtn) dragColorBtn.addEventListener("click", () => {
      if (!baseShapes.length) { alert("Generate an image first."); return; }
      populateColorPalette();
      if (dragModal) dragModal.style.display = "flex";
   });
   { const c = $("#cancelDragBtn"); if (c) c.addEventListener("click", closeDragModal); }
   if (dragModal) dragModal.addEventListener("click", (e) => { if (e.target === dragModal) closeDragModal(); });
   { const sv = $("#saveDragBtn"); if (sv) sv.addEventListener("click", () => {
      if (!colorPalette) return;
      customColorOrder = [...colorPalette.children].map((c) => JSON.parse(c.dataset.color));
      closeDragModal();
      currentLogic = "custom_sequence";
      setActiveLogic(renderingButtons.find((b) => b.dataset.logic === "custom_sequence"));
      shapes = rlOrder(baseShapes, "custom_sequence", customColorOrder);
      selectedIndex = null;
      render();
      updateSelectionInfo();
      persistState();
   }); }

   // Restore a carried-forward result (used on the Templates page).
   function restoreState() {
      let saved = null;
      try { saved = JSON.parse(sessionStorage.getItem(STORE_KEY) || "null"); } catch (_) {}
      if (!saved || !Array.isArray(saved.baseShapes) || !saved.baseShapes.length) return false;
      baseShapes = saved.baseShapes;
      currentLogic = saved.logic || "original";
      customColorOrder = saved.colorOrder || null;
      shapeLimit = (typeof saved.limit === "number") ? saved.limit : null;
      canvas.width = saved.w || canvas.width;
      canvas.height = saved.h || canvas.height;
      shapes = rlOrder(baseShapes, currentLogic, customColorOrder);
      selectedIndex = null;
      showCanvas();
      syncSlider();
      render();
      updateSelectionInfo();
      if (renderingSection) renderingSection.classList.remove("hidden");
      setActiveLogic(
         renderingButtons.find((b) => b.dataset.logic === currentLogic) ||
         renderingButtons.find((b) => b.dataset.logic === "original")
      );
      setStatus(`${shapes.length} shapes — click one to select`);
      return true;
   }

   // ---- generate shapes IN-BROWSER, in-page (no network / no worker) ----
   let runToken = 0; // bumped to cancel an in-flight build (e.g. on Reset)

   function setBusy(b) {
      [btnGenerate, btnStep, btnRandom].forEach((el) => el && (el.disabled = b));
   }

   async function generate(count) {
      if (!uploadedFile) { alert("Upload an image first."); return; }
      if (animating) return; // a build is already running

      let G;
      try { G = engine(); }
      catch (err) { setStatus("Error: " + err.message); console.error(err); return; }

      // ---- plan enforcement: each generation consumes one image ----
      if (window.Billing && typeof window.Billing.consumeImage === "function") {
         setStatus("Checking your plan…");
         const quota = await window.Billing.consumeImage();
         if (quota && quota.allowed === false) {
            setStatus("");
            window.Billing.showUpgrade
               ? window.Billing.showUpgrade(quota.reason || "You're out of images.")
               : alert(quota.reason || "You're out of images.");
            return;
         }
      }

      const maxLimit = clampInt(maxShapeEl && maxShapeEl.value, 1, 10000, 4000);
      count = clampInt(count, 1, maxLimit, 255);

      const options = {
         shapeTypes: selectedShapeCodes(),
         alpha: opacity(),
         candidateShapesPerStep: candidates(),
         shapeMutationsPerStep: mutations(),
      };

      setStatus("Loading image…");
      setBusy(true);
      animating = true;
      selectedIndex = null;
      updateSelectionInfo();

      let bitmap;
      try {
         bitmap = await imageToBitmapData(uploadedFile);
      } catch (err) {
         console.error(err);
         setStatus("Error: could not read image");
         setBusy(false); animating = false;
         return;
      }

      const { width, height, data, originalWidth, originalHeight, avgColor } = bitmap;
      const scaleX = originalWidth / width, scaleY = originalHeight / height;
      const [aR, aG, aB] = avgColor;
      const bgPacked = ((aR & 255) << 24) | ((aG & 255) << 16) | ((aB & 255) << 8) | 255;

      // Reset canvas to source size; start with the background rectangle.
      canvas.width = originalWidth;
      canvas.height = originalHeight;
      shapes = [{ type: 0, data: [0, 0, originalWidth, originalHeight], color: avgColor, score: 0 }];
      showCanvas();
      render();
      await yieldFrame();

      const myToken = ++runToken;
      try {
         const runner = new G.runner.ImageRunner({ width, height, data }, bgPacked);
         for (let step = 1; step <= count; step++) {
            if (myToken !== runToken) return; // cancelled
            const results = runner.step(options);
            const json = G.exporter.ShapeJsonExporter.exportShapes(results);
            if (json && json.length > 0) {
               shapes.push(...parseAndScaleShapes(json, scaleX, scaleY));
            }
            // redraw periodically so the image visibly builds up, and yield
            // back to the browser so the canvas actually repaints.
            if (step % 4 === 0 || step === count) {
               render();
               setStatus(`Building… ${step} / ${count} shapes`);
               await yieldFrame();
            }
         }
      } catch (err) {
         console.error("Engine error:", err);
         setStatus("Error: engine failed (" + (err.message || "unknown") + ")");
      }

      if (myToken !== runToken) return; // a newer run / reset took over
      animating = false;
      currentCount = count;
      baseShapes = shapes.slice();   // remember the original generated order
      currentLogic = "original";
      customColorOrder = null;
      shapeLimit = shapes.length;
      syncSlider();
      setBusy(false);
      render();
      updateSelectionInfo();
      setStatus(`${shapes.length} shapes — click one to select`);

      // image has loaded -> reveal the rendering logic controls, reset to Original
      if (renderingSection) renderingSection.classList.remove("hidden");
      setActiveLogic(renderingButtons.find((b) => b.dataset.logic === "original"));
      persistState();   // carry the result forward to the Templates page
   }

   if (btnGenerate) btnGenerate.addEventListener("click", () =>
      generate(clampInt(shapeAddedEl && shapeAddedEl.value, 1, 10000, 255)));

   if (btnStep) btnStep.addEventListener("click", () => {
      const step = clampInt(shapeAddedEl && shapeAddedEl.value, 1, 10000, 255);
      const maxLimit = clampInt(maxShapeEl && maxShapeEl.value, 1, 10000, 4000);
      generate(Math.min((currentCount || 0) + step, maxLimit));
   });

   if (btnReset) btnReset.addEventListener("click", () => {
      runToken++;           // cancel any in-flight build
      animating = false;
      setBusy(false);
      shapes = [];
      baseShapes = [];
      currentLogic = "original";
      customColorOrder = null;
      shapeLimit = null;
      syncSlider();
      selectedIndex = null;
      currentCount = 0;
      uploadedFile = null;
      if (fileInput) fileInput.value = "";
      updateSelectionInfo();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      showDropZone();
      setStatus("");
      if (renderingSection) renderingSection.classList.add("hidden");
      clearState();
   });

   if (btnRandom) btnRandom.addEventListener("click", () => {
      const rnd = (lo, hi) => Math.floor(lo + Math.random() * (hi - lo + 1));
      if (opacityEl)   { opacityEl.value   = rnd(90, 200); opacityEl.dispatchEvent(new Event("input")); }
      if (mutationsEl) { mutationsEl.value = rnd(40, 200); mutationsEl.dispatchEvent(new Event("input")); }
      if (randomEl)    { randomEl.value    = rnd(40, 200); randomEl.dispatchEvent(new Event("input")); }
      const boxes = shapeBoxes();
      if (boxes.length) {
         boxes.forEach((b) => (b.cb.checked = false));
         boxes[rnd(0, boxes.length - 1)].cb.checked = true;
      }
      generate(clampInt(shapeAddedEl && shapeAddedEl.value, 1, 10000, 255));
   });

   if (btnSave) btnSave.addEventListener("click", () => {
      if (!shapes.length) { alert("Generate an image first."); return; }
      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = "geomagic.png";
      document.body.appendChild(a);
      a.click();
      a.remove();
   });

   // Pages that opt in (e.g. Templates) carry forward the result generated
   // elsewhere by restoring the saved shapes on load.
   if (window.GEOMAGIC_RESTORE) restoreState();
})();
