/* ============================================================
   Color Library — store & edit named colors (name + RGB) that
   feed the Trycolors unmixer.
   - "My Colors" lives on the server (/api/palettes/my-colors) and is
     changed one color at a time, so edits here, in the apps and colors
     captured on devices don't overwrite each other. The server list is
     the truth, even when it is empty.
   - A per-user copy in localStorage ("geomagic:colorLibrary:<user id>")
     shows the list instantly; nobody signed in = empty library.
   - Surfaces as "★ My Colors" in the unmixer palette dropdown
     (see palette-mixing.js hooks: ColorLibrary / applyUnmixerPalette).
   - Edited via a modal opened from the "Edit Library" button.
   ============================================================ */
(function () {
   "use strict";

   const KEY = "geomagic:colorLibrary";   // + ":" + user id; the bare key is the old shared list

   function authToken() { try { return localStorage.getItem("gm_access_token"); } catch (_) { return null; } }
   // the signed-in user's id, from the token (same id the server keys palettes by)
   function userId() {
      const t = authToken();
      if (!t) return null;
      try {
         const p = JSON.parse(atob(t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
         if (p && p.sub) return String(p.sub);
      } catch (_) {}
      try { const u = JSON.parse(localStorage.getItem("gm_user") || "null"); return u && u.id ? String(u.id) : null; } catch (_) { return null; }
   }
   function libKey() { const u = userId(); return u ? KEY + ":" + u : null; }
   function load() {
      const k = libKey();
      if (!k) return [];
      try {
         const a = JSON.parse(localStorage.getItem(k) || "[]");
         return Array.isArray(a) ? a.filter((x) => x && x.hex).map((x) => ({ name: x.name || "", hex: String(x.hex).toUpperCase() })) : [];
      } catch (_) { return []; }
   }
   let lib = load();
   function persist() {
      const k = libKey();
      if (!k) return;
      try { localStorage.setItem(k, JSON.stringify(lib.map((c) => ({ name: c.name, hex: c.hex })))); } catch (_) {}
   }

   // ---- server sync ----
   const MY = "My Colors";
   const MYC = "/api/palettes/my-colors";
   let serverPalettes = [];               // named palettes [{id, name, colors:[{hex,name}]}]
   const srv = new WeakMap();             // library entry -> {hex, name} as the server has it
   const unsent = new Set();              // entries added here that never reached the server (offline)
   const adding = new Set();              // entries whose add / import is on its way
   const dirty = new Set();               // entries edited here, PATCH pending
   let flushTimer = null, flushing = null;

   function apiBase() {
      try {
         const q = (new URLSearchParams(location.search).get("api") || "").replace(/\/$/, "");
         if (q) return q;
      } catch (_) {}
      return (location.protocol.indexOf("http") === 0) ? location.origin : "";
   }
   function authHeaders() {
      const h = { "Content-Type": "application/json" };
      const t = authToken();
      if (t) h["Authorization"] = "Bearer " + t;
      return h;
   }
   // -> {ok, status, data}; status 0 = network error
   async function call(method, path, body) {
      try {
         // keepalive: a rename flushed as the tab closes still reaches the server
         const r = await fetch(apiBase() + path, { method: method, headers: authHeaders(), body: body ? JSON.stringify(body) : undefined, keepalive: method !== "GET" });
         let data = null;
         try { data = await r.json(); } catch (_) {}
         return { ok: r.ok, status: r.status, data: data };
      } catch (_) { return { ok: false, status: 0, data: null }; }
   }
   function colorPath(hex) { return MYC + "/colors/" + String(hex).replace("#", ""); }
   function detail(r, fallback) { return (r && r.data && typeof r.data.detail === "string") ? r.data.detail : fallback; }

   // Replace the library with the server's list (the truth, even when empty).
   // Skipped while edits typed here are still on their way, so they aren't lost;
   // the next load catches up.
   function applyServer(p) {
      if (!p || !Array.isArray(p.colors)) return;
      // entries added here (import, offline) that the server now has: track them
      // even if the swap below is skipped, so later edits / deletes reach it
      p.colors.forEach((c) => {
         const H = String(c.hex).toUpperCase();
         const e = lib.find((x) => x.hex === H && !srv.has(x));
         if (e) { srv.set(e, { hex: H, name: c.name || "" }); unsent.delete(e); }
      });
      if (dirty.size || flushing) return;
      // Entries still being added (or waiting to be sent) keep their identity,
      // so their pending request still recognises them after the swap.
      const keep = new Map();
      adding.forEach((e) => keep.set(e.hex, e));
      unsent.forEach((e) => keep.set(e.hex, e));
      lib = p.colors.filter((c) => c && c.hex).map((c) => {
         const H = String(c.hex).toUpperCase();
         let e = keep.get(H);
         if (e) keep.delete(H);
         else { e = { name: c.name || "", hex: H }; srv.set(e, { hex: H, name: e.name }); }
         return e;
      });
      keep.forEach((e) => lib.push(e));                                  // not on the server yet
      persist();
      changed();
   }

   async function sendAdd(e) {
      if (!authToken()) return;
      const sent = { hex: e.hex, name: e.name };
      adding.add(e);
      const r = await call("POST", MYC + "/colors", sent);
      adding.delete(e);
      if (r.ok) {
         unsent.delete(e);
         const onServer = (r.data.colors || []).find((c) => String(c.hex).toUpperCase() === sent.hex);
         srv.set(e, { hex: sent.hex, name: onServer ? onServer.name : sent.name });
         if (lib.indexOf(e) < 0) { sendDelete(e); return; }            // removed while the add was on its way
         if (e.hex !== sent.hex || e.name !== sent.name) { markDirty(e); return; }
         applyServer(r.data);
      } else if (r.status === 0) {
         unsent.add(e);                                                   // retried on the next load
      } else if (r.status === 409 || r.status === 400) {
         const i = lib.indexOf(e);
         if (i >= 0) { lib.splice(i, 1); persist(); changed(); }
         alert(detail(r, "Could not add that color."));
      }
   }

   async function sendDelete(e) {
      if (flushing) await flushing;                                       // let a pending hex change land first
      const s = srv.get(e);
      if (!s || !authToken()) return;
      const r = await call("DELETE", colorPath(s.hex));
      if (r.ok) applyServer(r.data);
      else if (r.status === 404) serverLoad();
   }

   function markDirty(e) {
      dirty.add(e);
      clearTimeout(flushTimer);
      flushTimer = setTimeout(flushEdits, 600);   // typing / dragging the picker -> one PATCH
   }
   // Send pending renames / hex changes, one PATCH per color (from the hex the server knows).
   function flushEdits() {
      clearTimeout(flushTimer); flushTimer = null;
      if (flushing) return flushing.then(() => (dirty.size ? flushEdits() : null));
      if (!dirty.size) return Promise.resolve();
      // `flushing` is cleared in .then (always after this assignment), never
      // inside sendEdits: a run that finishes without awaiting would otherwise
      // leave it set for good and block every later applyServer().
      flushing = sendEdits().then((next) => {
         flushing = null;
         if (next === "retry") { clearTimeout(flushTimer); flushTimer = setTimeout(flushEdits, 5000); }
         else if (next === "reload") return serverLoad();
      });
      return flushing;
   }
   async function sendEdits() {
      let reload = false;
      while (dirty.size) {
         const e = dirty.values().next().value;
         dirty.delete(e);
         const s = srv.get(e);
         if (!s || lib.indexOf(e) < 0 || !authToken()) continue;          // not on the server yet / removed
         const body = {};
         if (e.hex !== s.hex) body.hex = e.hex;
         if (e.name !== s.name) body.name = e.name;
         if (!Object.keys(body).length) continue;
         const sent = { hex: e.hex, name: e.name };
         const r = await call("PATCH", colorPath(s.hex), body);
         if (r.ok) {
            const c = (r.data.colors || []).find((x) => String(x.hex).toUpperCase() === sent.hex);
            if (c) {
               srv.set(e, { hex: sent.hex, name: c.name });
               if (e.hex === sent.hex && e.name === sent.name) e.name = c.name;   // e.g. blank name -> the hex
               else dirty.add(e);                                               // changed again meanwhile
            }
            persist();
         } else if (r.status === 409) {
            alert(detail(r, "That color is already in your library."));
            reload = true;
         } else if (r.status === 404) {
            reload = true;                                                  // removed on another device
         } else if (r.status === 0) {
            dirty.add(e);
            return "retry";
         } else {
            return null;
         }
      }
      return reload ? "reload" : null;
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
   async function serverLoad() {
      if (!authToken() || !libKey()) {                  // nobody signed in: empty library
         if (lib.length) { lib = []; changed(); }
         return;
      }
      // One time: move the old list that was shared by everyone on this browser
      // into the account (merge only adds), then drop it.
      let legacy = null, hadLegacy = false;
      try { const raw = localStorage.getItem(KEY); hadLegacy = raw != null; legacy = JSON.parse(raw || "null"); } catch (_) {}
      if (hadLegacy) {
         const cols = (Array.isArray(legacy) ? legacy : [])
            .filter((c) => c && normHex(c.hex)).map((c) => ({ hex: normHex(c.hex), name: (c.name || "").trim() }));
         const r = cols.length ? await call("POST", MYC + "/merge", { colors: cols }) : { ok: true };
         if (r.ok) { try { localStorage.removeItem(KEY); } catch (_) {} }
      }
      // colors added here while offline
      const pending = Array.from(unsent).filter((e) => lib.indexOf(e) >= 0);
      if (pending.length) {
         const r = await call("POST", MYC + "/merge", { colors: pending.map((e) => ({ hex: e.hex, name: e.name })) });
         if (r.ok) applyServer(r.data);
      }
      const res = await Promise.all([call("GET", MYC), call("GET", "/api/palettes")]);
      if (res[1].ok && res[1].data && Array.isArray(res[1].data.palettes)) serverPalettes = res[1].data.palettes;
      if (res[0].ok) applyServer(res[0].data);
      refreshSelector();
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
      has: (hex) => { const H = normHex(hex); return !!H && lib.some((c) => c.hex === H); },
      // Each change is applied here at once and sent to the server on its own.
      add(name, hex) {
         const H = normHex(hex); if (!H) return false;
         if (lib.some((c) => c.hex === H)) return true;                  // already in the library
         const e = { name: (name || "").trim() || H, hex: H };
         lib.push(e); persist(); changed();
         sendAdd(e);
         return true;
      },
      update(i, name, hex) {
         const e = lib[i]; if (!e) return;
         const H = normHex(hex);
         e.name = (name || "").trim() || e.name;
         if (H) e.hex = H;
         persist(); markDirty(e);
      },
      remove(i) {
         const e = lib[i]; if (!e) return;
         lib.splice(i, 1); dirty.delete(e); unsent.delete(e); adding.delete(e); persist();
         sendDelete(e);
      },
      flush: () => flushEdits(),
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
         // only a full #RRGGBB counts (a 3-digit prefix typed on the way is not a color change)
         hex.addEventListener("input", () => { const v = hex.value.trim(); const h = /^#?[0-9a-f]{6}$/i.test(v) ? normHex(v) : null; if (h) { color.value = h; ColorLibrary.update(i, name.value, h); } });
         name.addEventListener("input", () => ColorLibrary.update(i, name.value, hex.value));
         row.querySelector('[data-role=del]').addEventListener("click", () => { ColorLibrary.remove(i); renderList(); changed(); });
      });
   }

   function addFromInputs() {
      const name = modal.querySelector("#clNewName").value;
      const hex = modal.querySelector("#clNewHex").value || modal.querySelector("#clNewColor").value;
      if (!normHex(hex)) { alert("Enter a valid color (e.g. #CC4444)."); return; }
      if (ColorLibrary.has(hex)) { alert("That color is already in your library."); return; }
      ColorLibrary.add(name, hex);
      modal.querySelector("#clNewName").value = "";
      modal.querySelector("#clNewHex").value = "";
      renderList();
   }

   function importCurrent() {
      const cur = (typeof window.getUnmixerPalette === "function") ? window.getUnmixerPalette() : [];
      if (!cur.length) { alert("No palette is currently loaded in the unmixer."); return; }
      const added = [];
      const have = new Set(lib.map((c) => c.hex));
      cur.forEach((c) => { const H = normHex(c.hex); if (H && !have.has(H)) { const e = { name: c.name || H, hex: H }; lib.push(e); added.push(e); have.add(H); } });
      persist();
      renderList();
      if (!added.length) { alert("Those colors are already in your library."); return; }
      changed();
      if (!authToken()) return;
      // one call for the whole import; merge only adds, so nothing saved elsewhere is lost
      added.forEach((e) => adding.add(e));
      call("POST", MYC + "/merge", { colors: added.map((e) => ({ hex: e.hex, name: e.name })) }).then((r) => {
         added.forEach((e) => adding.delete(e));
         if (r.ok) applyServer(r.data);
         else if (r.status === 0) added.forEach((e) => unsent.add(e));
      });
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

   // After the list changes (added / removed here, or reloaded from the server),
   // keep everything that mirrors it in sync: the open editor, the dropdown
   // count/option AND — if the unmixer is currently showing "★ My Colors" — its
   // loaded palette grid, so a removed color disappears immediately.
   function changed() {
      if (modal && modal.classList.contains("open")) renderList();
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
      if (name === MY) { alert('"My Colors" is this library itself. Pick another name for the palette.'); if (input) input.focus(); return; }
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
      flushEdits();            // send any rename / hex change still waiting
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
   // coming back to the tab picks up colors added on other devices (captures, the apps)
   document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") serverLoad();
      else flushEdits();
   });

   window.ColorLibrary.open = open; // allow programmatic open
})();
