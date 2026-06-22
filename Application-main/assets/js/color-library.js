/* ============================================================
   Color Library — store & edit named colors (name + RGB) that
   feed the Trycolors unmixer.
   - Persisted in localStorage (per device), survives reloads.
   - Surfaces as "★ My Colors" in the unmixer palette dropdown
     (see palette-mixing.js hooks: ColorLibrary / applyUnmixerPalette).
   - Edited via a modal opened from the "Edit Library" button.
   ============================================================ */
(function () {
   "use strict";

   const KEY = "geomagic:colorLibrary";

   function load() {
      try {
         const a = JSON.parse(localStorage.getItem(KEY) || "[]");
         return Array.isArray(a) ? a.filter((x) => x && x.hex) : [];
      } catch (_) { return []; }
   }
   let lib = load();
   function persist() { try { localStorage.setItem(KEY, JSON.stringify(lib)); } catch (_) {} }

   // ---- server sync: per-user palettes saved in the DB (/api/palettes) ----
   // Lets a signed-in user's colors follow them across devices instead of
   // living only in this browser's localStorage. Best-effort: if logged out or
   // offline, everything still works locally.
   const MY = "My Colors";
   let serverPalettes = [];               // [{id, name, colors:[{hex,name}]}]
   function apiBase() {
      try {
         const q = (new URLSearchParams(location.search).get("api") || "").replace(/\/$/, "");
         if (q) return q;
      } catch (_) {}
      return (location.protocol.indexOf("http") === 0) ? location.origin : "";
   }
   function authToken() { try { return localStorage.getItem("gm_access_token"); } catch (_) { return null; } }
   function authHeaders() {
      const h = { "Content-Type": "application/json" };
      const t = authToken();
      if (t) h["Authorization"] = "Bearer " + t;
      return h;
   }
   async function serverSave(name, colors) {
      if (!authToken()) return null;
      try {
         const r = await fetch(apiBase() + "/api/palettes", {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ name: name, colors: colors }),
         });
         return r.ok ? await r.json() : null;
      } catch (_) { return null; }
   }
   function rememberServer(saved) {
      if (!saved) return;
      serverPalettes = serverPalettes.filter((p) => p.name !== saved.name);
      serverPalettes.unshift(saved);
   }
   function syncMyColors() { serverSave(MY, lib).then(rememberServer); }
   async function serverLoad() {
      if (!authToken()) return;
      try {
         const r = await fetch(apiBase() + "/api/palettes", { headers: authHeaders() });
         if (!r.ok) return;
         const data = await r.json();
         serverPalettes = Array.isArray(data.palettes) ? data.palettes : [];
         const mine = serverPalettes.find((p) => p.name === MY);
         if (mine && Array.isArray(mine.colors) && mine.colors.length) {
            lib = mine.colors.filter((c) => c && c.hex).map((c) => ({ name: c.name || "", hex: c.hex }));
            persist();
         } else if (lib.length) {
            rememberServer(await serverSave(MY, lib));   // first-time migration up
         }
      } catch (_) {}
      refreshSelector();
      const sel = document.getElementById("trycolorsPaletteSelect");
      if (sel && sel.value === "__mycolors__" && typeof window.applyUnmixerPalette === "function") {
         window.applyUnmixerPalette(ColorLibrary.asPalette(), "__mycolors__");
      }
   }

   function normHex(h) {
      if (!h) return null;
      h = String(h).trim();
      if (!h.startsWith("#")) h = "#" + h;
      if (/^#[0-9a-fA-F]{3}$/.test(h)) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3];
      return /^#[0-9a-fA-F]{6}$/.test(h) ? h.toUpperCase() : null;
   }
   function hexToRgb(h) {
      h = normHex(h); if (!h) return null;
      return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
   }

   // ---- public API consumed by palette-mixing.js ----
   window.ColorLibrary = {
      list: () => lib.slice(),
      size: () => lib.length,
      // [{hex, name}] for the unmixer (names fall back to a label if blank).
      asPalette: () => lib.map((c, i) => ({ hex: c.hex, name: (c.name && c.name.trim()) || ("Color " + (i + 1)) })),
      add(name, hex) { const H = normHex(hex); if (!H) return false; lib.push({ name: (name || "").trim() || H, hex: H }); persist(); return true; },
      update(i, name, hex) { if (!lib[i]) return; const H = normHex(hex); lib[i] = { name: (name || "").trim() || lib[i].name, hex: H || lib[i].hex }; persist(); },
      remove(i) { if (lib[i]) { lib.splice(i, 1); persist(); } },
      // server-saved palettes (DB) for the unmixer dropdown
      serverPalettes: () => serverPalettes.slice(),
      getServerPalette: (id) => serverPalettes.find((p) => String(p.id) === String(id)) || null,
      appendServerOptions(select) {
         if (!select) return;
         serverPalettes.forEach((p) => {
            if (p.name === MY) return;   // "My Colors" is already shown via the editable library
            const v = "__srvpal__:" + p.id;
            if (Array.from(select.options).some((o) => o.value === v)) return;
            const o = document.createElement("option");
            o.value = v;
            o.textContent = "★ " + p.name + " (" + ((p.colors || []).length) + " colors)";
            select.appendChild(o);
         });
      },
      reloadFromServer: () => serverLoad(),
   };

   // ---- one-time CSS (scoped, theme-aware) ----
   (function injectCss() {
      if (document.getElementById("color-library-css")) return;
      const s = document.createElement("style");
      s.id = "color-library-css";
      s.textContent = `
      /* self-contained modal + buttons so this works in any host page */
      .cl-modal { position:fixed; inset:0; z-index:10000; display:none; align-items:center; justify-content:center; background:rgba(0,0,0,0.55); padding:20px; }
      .cl-modal.open { display:flex; }
      .cl-modal-card { width:100%; max-width:580px; max-height:88vh; overflow:auto; background:#fff; color:#1a1a1a; border-radius:14px; padding:22px; box-shadow:0 20px 60px rgba(0,0,0,0.35); font-family:inherit; }
      .dark .cl-modal-card, html[data-theme=dark] .cl-modal-card { background:#1a1d1f; color:#e9eef2; }
      .cl-title { font-size:18px; font-weight:700; margin:0 0 6px; }
      .cl-info { font-size:13px; opacity:0.75; margin:0 0 12px; line-height:1.5; }
      .cl-btn { height:38px; padding:0 16px; border-radius:10px; border:1px solid #d6d6d9; background:#f4f4f4; color:#1a1a1a; font-size:14px; font-weight:600; cursor:pointer; transition:all .15s; }
      .cl-btn:hover { background:#ececec; }
      .dark .cl-btn, html[data-theme=dark] .cl-btn { background:#272b30; border-color:#33383f; color:#e9eef2; }
      .cl-btn-primary { background:#2a85ff; border-color:#2a85ff; color:#fff; }
      .cl-btn-primary:hover { background:#1e6fe0; }
      .cl-actions { display:flex; align-items:center; gap:10px; margin-top:16px; flex-wrap:wrap; }
      .cl-list { display:flex; flex-direction:column; gap:8px; max-height:46vh; overflow:auto; margin:4px 0 14px; }
      .cl-row { display:flex; align-items:center; gap:10px; padding:8px; border-radius:10px; background:#f4f4f4; }
      .dark .cl-row { background:#202225; }
      .cl-row input[type=color] { width:34px; height:34px; padding:0; border:none; background:none; cursor:pointer; border-radius:8px; flex:0 0 auto; }
      .cl-row .cl-name { flex:1 1 auto; min-width:0; height:36px; border-radius:8px; border:1px solid #e0e0e0; background:#fff; color:#1a1a1a; padding:0 10px; font-size:14px; font-weight:600; }
      .dark .cl-row .cl-name { background:#16181a; border-color:#33383f; color:#e9eef2; }
      .cl-row .cl-hex { width:92px; flex:0 0 auto; height:36px; text-align:center; border-radius:8px; border:1px solid #e0e0e0; background:#fff; color:#1a1a1a; font-family:monospace; font-size:13px; }
      .dark .cl-row .cl-hex { background:#16181a; border-color:#33383f; color:#e9eef2; }
      .cl-row .cl-del { flex:0 0 auto; width:34px; height:34px; border-radius:8px; border:none; background:#ffe2e2; color:#c0392b; cursor:pointer; font-size:14px; }
      .dark .cl-row .cl-del { background:#3a2326; color:#ff8e8e; }
      .cl-addrow { display:flex; align-items:center; gap:10px; padding-top:6px; border-top:1px solid #ececec; }
      .dark .cl-addrow { border-top-color:#2a2d31; }
      .cl-addrow input[type=color] { width:38px; height:38px; padding:0; border:none; background:none; cursor:pointer; border-radius:8px; }
      .cl-addrow .cl-name, .cl-addrow .cl-hex { height:38px; border-radius:8px; border:1px solid #e0e0e0; background:#fff; color:#1a1a1a; padding:0 10px; font-size:14px; }
      .dark .cl-addrow .cl-name, .dark .cl-addrow .cl-hex { background:#16181a; border-color:#33383f; color:#e9eef2; }
      .cl-addrow .cl-name { flex:1 1 auto; min-width:0; font-weight:600; }
      .cl-addrow .cl-hex { width:96px; text-align:center; font-family:monospace; }
      .cl-saverow { display:flex; align-items:center; gap:10px; margin-top:10px; }
      .cl-saverow input { flex:1 1 auto; min-width:0; height:38px; border-radius:8px; border:1px solid #e0e0e0; background:#fff; color:#1a1a1a; padding:0 10px; font-size:14px; font-weight:600; }
      .dark .cl-saverow input, html[data-theme=dark] .cl-saverow input { background:#16181a; border-color:#33383f; color:#e9eef2; }
      .cl-empty { text-align:center; color:#8a8f94; font-size:13px; padding:18px 0; }
      #colorLibraryBtn { margin-left:8px; }
      `;
      document.head.appendChild(s);
   })();

   // ---- modal (built lazily) ----
   let modal = null, listEl = null;
   function buildModal() {
      if (modal) return;
      modal = document.createElement("div");
      modal.id = "colorLibraryModal";
      modal.className = "cl-modal";
      modal.innerHTML = `
         <div class="cl-modal-card">
            <h3 class="cl-title">Color Library</h3>
            <div class="cl-info">Add colors with their RGB, give the palette a name, and click <strong>Save palette</strong> &mdash; it appears in the unmixer's palette dropdown and is saved to your account. Or use <strong>Save &amp; Use as My Colors</strong> for your quick everyday set.</div>
            <div id="colorLibraryList" class="cl-list"></div>
            <div class="cl-addrow">
               <input type="color" id="clNewColor" value="#cc4444" title="Pick color">
               <input type="text" class="cl-name" id="clNewName" placeholder="Color name (e.g. Cadmium Red)">
               <input type="text" class="cl-hex" id="clNewHex" placeholder="#CC4444" maxlength="7">
               <button id="clAddBtn" type="button" class="cl-btn cl-btn-primary">Add</button>
            </div>
            <div class="cl-saverow">
               <input type="text" id="clPaletteName" placeholder="Name this palette (e.g. Studio Set)" maxlength="80">
               <button id="clSaveAsBtn" type="button" class="cl-btn cl-btn-primary">Save palette</button>
            </div>
            <div class="cl-actions">
               <button id="clImportBtn" type="button" class="cl-btn">Import current palette</button>
               <span style="flex:1 1 auto;"></span>
               <button id="clCloseBtn" type="button" class="cl-btn">Close</button>
               <button id="clSaveUseBtn" type="button" class="cl-btn">Save &amp; Use as My Colors</button>
            </div>
         </div>`;
      document.body.appendChild(modal);
      listEl = modal.querySelector("#colorLibraryList");

      modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
      modal.querySelector("#clCloseBtn").addEventListener("click", close);
      modal.querySelector("#clSaveUseBtn").addEventListener("click", saveAndUse);
      modal.querySelector("#clImportBtn").addEventListener("click", importCurrent);
      modal.querySelector("#clSaveAsBtn").addEventListener("click", saveAsNew);
      modal.querySelector("#clAddBtn").addEventListener("click", addFromInputs);

      // keep the color picker and hex field in sync in the add row
      const nc = modal.querySelector("#clNewColor"), nh = modal.querySelector("#clNewHex");
      nc.addEventListener("input", () => { nh.value = nc.value.toUpperCase(); });
      nh.addEventListener("input", () => { const h = normHex(nh.value); if (h) nc.value = h; });
      nh.addEventListener("keydown", (e) => { if (e.key === "Enter") addFromInputs(); });
      modal.querySelector("#clNewName").addEventListener("keydown", (e) => { if (e.key === "Enter") addFromInputs(); });
      modal.querySelector("#clPaletteName").addEventListener("keydown", (e) => { if (e.key === "Enter") saveAsNew(); });
   }

   function rgbLabel(hex) { const r = hexToRgb(hex); return r ? `rgb(${r[0]}, ${r[1]}, ${r[2]})` : ""; }

   function renderList() {
      if (!listEl) return;
      if (!lib.length) { listEl.innerHTML = `<div class="cl-empty">No colors yet. Add one below, or import the current palette.</div>`; return; }
      listEl.innerHTML = lib.map((c, i) => `
         <div class="cl-row" data-i="${i}">
            <input type="color" value="${c.hex}" data-role="color" title="${rgbLabel(c.hex)}">
            <input type="text" class="cl-name" value="${(c.name || "").replace(/"/g, "&quot;")}" data-role="name" placeholder="Name">
            <input type="text" class="cl-hex" value="${c.hex}" data-role="hex" maxlength="7">
            <button type="button" class="cl-del" data-role="del" title="Remove">✕</button>
         </div>`).join("");

      listEl.querySelectorAll(".cl-row").forEach((row) => {
         const i = parseInt(row.dataset.i, 10);
         const color = row.querySelector('[data-role=color]');
         const name = row.querySelector('[data-role=name]');
         const hex = row.querySelector('[data-role=hex]');
         color.addEventListener("input", () => { hex.value = color.value.toUpperCase(); ColorLibrary.update(i, name.value, color.value); });
         hex.addEventListener("input", () => { const h = normHex(hex.value); if (h) { color.value = h; ColorLibrary.update(i, name.value, h); } });
         name.addEventListener("input", () => ColorLibrary.update(i, name.value, hex.value));
         row.querySelector('[data-role=del]').addEventListener("click", () => { ColorLibrary.remove(i); renderList(); afterLibraryMutation(); });
      });
   }

   function addFromInputs() {
      const name = modal.querySelector("#clNewName").value;
      const hex = modal.querySelector("#clNewHex").value || modal.querySelector("#clNewColor").value;
      if (!ColorLibrary.add(name, hex)) { alert("Enter a valid color (e.g. #CC4444)."); return; }
      modal.querySelector("#clNewName").value = "";
      modal.querySelector("#clNewHex").value = "";
      renderList();
   }

   function importCurrent() {
      const cur = (typeof window.getUnmixerPalette === "function") ? window.getUnmixerPalette() : [];
      if (!cur.length) { alert("No palette is currently loaded in the unmixer."); return; }
      let added = 0;
      const have = new Set(lib.map((c) => c.hex));
      cur.forEach((c) => { const H = normHex(c.hex); if (H && !have.has(H)) { lib.push({ name: c.name || H, hex: H }); have.add(H); added++; } });
      persist();
      renderList();
      if (!added) alert("Those colors are already in your library.");
   }

   function refreshSelector() {
      if (typeof window.populateTrycolorsPaletteSelector === "function") {
         window.populateTrycolorsPaletteSelector();
      }
      ensureOption();
      // Ensure the user's saved palettes are listed even if the preset list
      // hasn't loaded yet (populate bails early without palettePresetsData).
      const sel = document.getElementById("trycolorsPaletteSelect");
      if (sel) ColorLibrary.appendServerOptions(sel);
   }
   // Make sure the "★ My Colors" option exists/updates even if the preset list
   // hasn't loaded yet (populate bails early without palettePresetsData).
   function ensureOption() {
      const sel = document.getElementById("trycolorsPaletteSelect");
      if (!sel) return;
      let opt = Array.from(sel.options).find((o) => o.value === "__mycolors__");
      if (lib.length === 0) { if (opt) opt.remove(); return; }
      if (!opt) { opt = document.createElement("option"); opt.value = "__mycolors__"; sel.appendChild(opt); }
      opt.textContent = `★ My Colors (${lib.length} colors)`;
   }

   // After a structural change to the library (e.g. removing a color), keep
   // everything that mirrors it in sync: the dropdown count/option AND — if the
   // unmixer is currently showing "★ My Colors" — its loaded palette grid, so a
   // removed color disappears immediately instead of lingering until the next
   // "Save & Use in Unmixer".
   function afterLibraryMutation() {
      syncMyColors();          // keep the DB copy in sync after add/remove
      refreshSelector();
      const sel = document.getElementById("trycolorsPaletteSelect");
      if (sel && sel.value === "__mycolors__" && typeof window.applyUnmixerPalette === "function") {
         window.applyUnmixerPalette(ColorLibrary.asPalette(), "__mycolors__");
      }
   }

   async function saveAsNew() {
      if (!authToken()) { alert("Sign in to save palettes to your account."); return; }
      if (!lib.length) { alert("Add some colors first."); return; }
      const input = modal && modal.querySelector("#clPaletteName");
      const name = (input ? input.value : "").trim();
      if (!name) { alert("Type a name for the palette first."); if (input) input.focus(); return; }
      const saved = await serverSave(name, lib);
      if (!saved) { alert("Could not save the palette. Please try again."); return; }
      rememberServer(saved);
      refreshSelector();
      // Select + load the saved palette in the unmixer so it's usable right away.
      const sel = document.getElementById("trycolorsPaletteSelect");
      if (sel) {
         sel.value = "__srvpal__:" + saved.id;
         if (typeof window.applyUnmixerPalette === "function") {
            window.applyUnmixerPalette((saved.colors || []).map((c) => ({ hex: c.hex, name: c.name || null })), "__srvpal__");
         }
      }
      if (input) input.value = "";
      alert('Saved "' + name + '". It now appears in the palette dropdown.');
      close();
   }

   function saveAndUse() {
      persist();
      syncMyColors();          // remember this palette for the user in the DB
      refreshSelector();
      const sel = document.getElementById("trycolorsPaletteSelect");
      if (lib.length && typeof window.applyUnmixerPalette === "function") {
         if (sel) sel.value = "__mycolors__";
         window.applyUnmixerPalette(ColorLibrary.asPalette(), "__mycolors__");
      } else {
         refreshSelector();
      }
      close();
   }

   function open() { buildModal(); renderList(); modal.classList.add("open"); modal.style.display = "flex"; }
   function close() { if (modal) { modal.classList.remove("open"); modal.style.display = "none"; } }

   // ---- wire the "Edit Library" trigger button (added in templates.html) ----
   function wireButton() {
      const btn = document.getElementById("colorLibraryBtn");
      if (btn && !btn.dataset.wired) { btn.dataset.wired = "1"; btn.addEventListener("click", open); }
   }
   if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => { wireButton(); refreshSelector(); serverLoad(); });
   } else {
      wireButton(); refreshSelector(); serverLoad();
   }

   window.ColorLibrary.open = open; // allow programmatic open
})();
