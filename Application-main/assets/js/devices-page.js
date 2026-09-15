/* Devices page — pairs the Alchroma mobile / tablet / Quest apps to the signed-in
 * account and shows what those devices captured. Talks to /api/devices/* on the
 * same backend as auth.js (see devices_routes.py).
 *
 *   phone / tablet : this page shows a code + QR  -> app redeems it   (POST /pair/web-code)
 *   headset        : the headset shows a code     -> typed in here    (POST /pair/claim)
 *   captures       : GET /captures, refreshed every 15 s while the tab is visible
 */
(function () {
   var API = (window.GM && GM.apiBase) || (window.Auth && Auth.apiBase) || '';
   var toast = function (m, o) { if (window.GM && GM.toast) GM.toast(m, o); else alert(m); };
   var $ = function (id) { return document.getElementById(id); };
   var esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };

   var config = { deep_link_scheme: 'alchroma', pair_code_ttl: 600, apps: {} };
   var captureOffset = 0, PAGE = 24, captureTotal = 0;

   // ---------------------------------------------------------------- API ----
   async function api(method, path, body) {
      var headers = { 'Content-Type': 'application/json' };
      var tok = window.Auth ? Auth.getToken() : null;
      if (tok) headers.Authorization = 'Bearer ' + tok;
      var res = await fetch(API + path, { method: method, headers: headers, body: body ? JSON.stringify(body) : undefined });
      var data = null; try { data = await res.json(); } catch (e) { data = {}; }
      if (res.status === 401 && path.indexOf('/pair/') === -1) {
         // session gone — same handling as the rest of the app
         toast('Your session expired — please sign in again.', { kind: 'warn', icon: 'fa-regular fa-triangle-exclamation' });
         setTimeout(function () { Auth.logout('signin.html'); }, 1200);
      }
      if (!res.ok) { var err = new Error((data && data.detail) || ('Request failed (' + res.status + ')')); err.status = res.status; throw err; }
      return data;
   }

   // ------------------------------------------------------------ helpers ----
   function ago(iso) {
      if (!iso) return 'never';
      var s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
      if (s < 45) return 'just now';
      if (s < 3600) return Math.round(s / 60) + ' min ago';
      if (s < 86400) return Math.round(s / 3600) + ' h ago';
      if (s < 7 * 86400) return Math.round(s / 86400) + ' d ago';
      return new Date(iso).toLocaleDateString();
   }
   var TYPE_ICON = { phone: 'fa-mobile-screen-button', tablet: 'fa-tablet-screen-button', headset: 'fa-vr-cardboard', projector: 'fa-projector', desktop: 'fa-desktop', other: 'fa-signal' };
   var TYPE_LABEL = { phone: 'Phone', tablet: 'Tablet', headset: 'Headset', projector: 'Projector', desktop: 'Desktop', other: 'Device' };
   var PLATFORM_LABEL = { android: 'Android', ios: 'iOS', quest: 'Meta Quest', web: 'Web', windows: 'Windows', macos: 'macOS' };
   var SOURCE_LABEL = { camera: 'Camera', unmix: 'Unmix', picker: 'Picker', projector: 'Projector', web_camera: 'Web camera', manual: 'Manual', other: 'Other' };
   function isMobileBrowser() { return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent); }
   function pairPageUrl(code) {
      var origin = (location.protocol.indexOf('http') === 0) ? location.origin : API;
      return origin + '/pair.html?code=' + encodeURIComponent(code);
   }

   // ------------------------------------------------------- config / links ----
   async function loadConfig() {
      try { config = await api('GET', '/api/devices/config'); } catch (e) { /* defaults */ }
      var links = [];
      if (config.apps && config.apps.android) links.push('<a href="' + esc(config.apps.android) + '" target="_blank" rel="noopener">Android APK</a>');
      if (config.apps && config.apps.ios) links.push('<a href="' + esc(config.apps.ios) + '" target="_blank" rel="noopener">iOS (TestFlight)</a>');
      if (config.apps && config.apps.quest) links.push('<a href="' + esc(config.apps.quest) + '" target="_blank" rel="noopener">Quest APK</a>');
      $('dvAppLinks').innerHTML = links.length ? '— ' + links.join(' · ') : '<small style="color:#8c9097">(download links will appear here once published)</small>';
   }

   // ------------------------------------------------ flow B: web code + QR ----
   var codeTimer = null;
   function stopCodeTimer() { if (codeTimer) { clearInterval(codeTimer); codeTimer = null; } }
   function showWebCode(data) {
      var code = data.code, grouped = code.slice(0, 4) + ' ' + code.slice(4);
      $('dvWebCodeText').textContent = grouped;
      $('dvWebCode').hidden = false;
      try { QR.draw($('dvQr'), pairPageUrl(code), { scale: 5, margin: 3, dark: '#111315', light: '#ffffff' }); }
      catch (e) { $('dvQr').hidden = true; }
      var open = $('dvOpenApp');
      open.href = config.deep_link_scheme + '://pair?code=' + code;
      open.hidden = !isMobileBrowser();
      var end = Date.now() + (data.expires_in || config.pair_code_ttl || 600) * 1000;
      stopCodeTimer();
      function tick() {
         var left = Math.max(0, Math.round((end - Date.now()) / 1000));
         var m = Math.floor(left / 60), s = left % 60;
         $('dvWebCodeExpiry').textContent = left ? ('expires in ' + m + ':' + (s < 10 ? '0' : '') + s) : 'expired — get a new code';
         if (!left) { stopCodeTimer(); $('dvWebCodeText').classList.add('expired'); }
      }
      $('dvWebCodeText').classList.remove('expired');
      tick(); codeTimer = setInterval(tick, 1000);
      $('dvCopyCode').onclick = function () {
         var done = function () { toast('Code copied', { kind: 'ok', icon: 'fa-regular fa-check' }); };
         if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(code).then(done, function () { window.prompt('Pairing code', code); });
         else window.prompt('Pairing code', code);
      };
   }
   $('dvWebCodeBtn').addEventListener('click', async function () {
      var b = this; b.disabled = true;
      try { showWebCode(await api('POST', '/api/devices/pair/web-code')); }
      catch (e) { toast(e.message || 'Could not create a pairing code', { kind: 'warn', icon: 'fa-regular fa-triangle-exclamation' }); }
      b.disabled = false;
   });

   // -------------------------------------------- flow A: claim headset code ----
   async function claimCode() {
      var input = $('dvPairCode'), code = (input.value || '').replace(/\D/g, '');
      if (code.length !== 6) { toast('Enter the 6-digit code shown in the headset app.', { kind: 'warn', icon: 'fa-regular fa-triangle-exclamation' }); input.focus(); return; }
      try {
         var r = await api('POST', '/api/devices/pair/claim', { code: code });
         var d = r.device || {};
         toast((d.name || TYPE_LABEL[d.device_type] || 'Device') + ' paired — the headset will finish on its own in a moment.', { kind: 'ok', icon: 'fa-regular fa-check' });
         input.value = '';
         refreshAll();
      } catch (e) {
         toast(e.status === 404 ? 'Code not found or expired — ask the headset for a new one.' : (e.message || 'Pairing failed'), { kind: 'warn', icon: 'fa-regular fa-triangle-exclamation' });
      }
   }
   $('dvPairCode').addEventListener('keydown', function (e) { if (e.key === 'Enter') claimCode(); });
   $('dvPairCode').addEventListener('input', function () {
      var v = this.value.replace(/\D/g, '').slice(0, 6);
      this.value = v.length > 3 ? v.slice(0, 3) + ' ' + v.slice(3) : v;
   });

   // ------------------------------------------------------- devices list ----
   async function loadDevices() {
      var host = $('dvDevices');
      try {
         var r = await api('GET', '/api/devices');
         var devs = r.devices || [];
         var online = devs.filter(function (d) { return d.online; }).length;
         var pill = $('dvPill');
         pill.classList.toggle('on', online > 0);
         $('dvPillText').textContent = devs.length ? (devs.length + ' device' + (devs.length > 1 ? 's' : '') + ' paired · ' + online + ' online') : 'No device paired yet';
         if (!devs.length) { host.innerHTML = '<div class="dv-empty">No devices yet. Pair a phone, tablet or headset with one of the cards above.</div>'; return; }
         host.innerHTML = devs.map(function (d) {
            var icon = TYPE_ICON[d.device_type] || TYPE_ICON.other;
            var name = d.name || (TYPE_LABEL[d.device_type] || 'Device');
            var plat = PLATFORM_LABEL[d.platform] || '';
            return '<div class="dv-row" data-id="' + d.id + '">'
               + '<span class="dv-row-ico"><i class="fa-regular ' + icon + '"></i></span>'
               + '<div class="dv-row-main"><b>' + esc(name) + '</b>'
               + '<small>' + esc(TYPE_LABEL[d.device_type] || 'Device') + (plat ? ' · ' + esc(plat) : '') + (d.app_version ? ' · v' + esc(d.app_version) : '') + '</small></div>'
               + '<span class="dv-status' + (d.online ? ' on' : '') + '"><i></i>' + (d.online ? 'Online' : 'Seen ' + ago(d.last_seen_at)) + '</span>'
               + '<div class="dv-row-actions">'
               + '<button type="button" class="dv-iconbtn" data-act="rename" title="Rename"><i class="fa-regular fa-pen"></i></button>'
               + '<button type="button" class="dv-iconbtn danger" data-act="unpair" title="Unpair"><i class="fa-regular fa-link-slash"></i></button>'
               + '</div></div>';
         }).join('');
      } catch (e) {
         host.innerHTML = '<div class="dv-empty">Could not load devices: ' + esc(e.message) + '</div>';
      }
   }
   $('dvDevices').addEventListener('click', async function (e) {
      var btn = e.target.closest('button[data-act]'); if (!btn) return;
      var row = btn.closest('.dv-row'), id = row.getAttribute('data-id'), name = row.querySelector('b').textContent;
      if (btn.getAttribute('data-act') === 'rename') {
         var n = window.prompt('Name this device', name); if (n == null) return;
         n = n.trim(); if (!n) return;
         try { await api('PATCH', '/api/devices/' + id, { name: n }); toast('Renamed', { kind: 'ok', icon: 'fa-regular fa-check' }); loadDevices(); }
         catch (err) { toast(err.message, { kind: 'warn' }); }
      } else {
         if (!window.confirm('Unpair “' + name + '”? The app will be signed out of this account on its next request; its captures stay in your history.')) return;
         try { await api('DELETE', '/api/devices/' + id); toast('Device unpaired', { kind: 'ok', icon: 'fa-regular fa-check' }); refreshAll(); }
         catch (err) { toast(err.message, { kind: 'warn' }); }
      }
   });

   // ----------------------------------------------------------- captures ----
   function captureRow(c) {
      var name = c.name && c.name !== c.hex ? c.name : '';
      var recipe = (c.recipe || []).filter(function (p) { return p && (p.name || p.hex); });
      var recipeHtml = recipe.length ? '<div class="dv-recipe">' + recipe.map(function (p) {
         return '<span title="' + esc(p.name || '') + '"><i style="background:' + esc(p.hex || '#999') + '"></i>' + esc(p.name || p.hex) + (p.percentage != null ? ' <small>' + (Math.round(p.percentage * 10) / 10) + '%</small>' : '') + '</span>';
      }).join('') + '</div>' : '';
      return '<div class="dv-row dv-cap" data-id="' + c.id + '">'
         + '<span class="dv-sw" style="background:' + esc(c.hex) + '"></span>'
         + '<div class="dv-row-main"><b>' + esc(name || c.hex) + '</b>'
         + '<small><code>' + esc(c.hex) + '</code> · RGB ' + esc(c.rgb) + ' · ' + esc(SOURCE_LABEL[c.source] || c.source || '') + (c.device_name || c.device_type ? ' · ' + esc(c.device_name || TYPE_LABEL[c.device_type] || '') : '') + ' · ' + ago(c.created_at) + '</small>'
         + recipeHtml + '</div>'
         + '<div class="dv-row-actions">'
         + '<a class="dv-iconbtn" href="templates.html?target=' + encodeURIComponent(c.hex) + '" title="Find a paint recipe for this color"><i class="fa-regular fa-eye-dropper"></i></a>'
         + '<button type="button" class="dv-iconbtn danger" data-act="delete" title="Delete"><i class="fa-regular fa-trash-can"></i></button>'
         + '</div></div>';
   }
   async function loadCaptures(reset) {
      var host = $('dvCaptures');
      if (reset) captureOffset = 0;
      try {
         var r = await api('GET', '/api/devices/captures?limit=' + PAGE + '&offset=' + captureOffset);
         captureTotal = r.total || 0;
         var rows = (r.captures || []).map(captureRow).join('');
         if (reset) host.innerHTML = rows || '<div class="dv-empty">Nothing captured yet. Pick a color in the app (or with the camera card above) and it will show up here.</div>';
         else host.insertAdjacentHTML('beforeend', rows);
         captureOffset += (r.captures || []).length;
         $('dvCapCount').textContent = captureTotal ? captureTotal : '';
         $('dvMore').hidden = captureOffset >= captureTotal;
      } catch (e) {
         if (reset) host.innerHTML = '<div class="dv-empty">Could not load captures: ' + esc(e.message) + '</div>';
      }
   }
   $('dvMore').addEventListener('click', function () { loadCaptures(false); });
   $('dvCaptures').addEventListener('click', async function (e) {
      var btn = e.target.closest('button[data-act="delete"]'); if (!btn) return;
      var row = btn.closest('.dv-row'), id = row.getAttribute('data-id');
      try { await api('DELETE', '/api/devices/captures/' + id); row.remove(); captureTotal--; $('dvCapCount').textContent = captureTotal || ''; }
      catch (err) { toast(err.message, { kind: 'warn' }); }
   });

   function refreshAll() { loadDevices(); loadCaptures(true); }
   $('dvRefresh').addEventListener('click', refreshAll);
   setInterval(function () { if (document.visibilityState === 'visible') { loadDevices(); if (captureOffset <= PAGE) loadCaptures(true); } }, 15000);

   // ------------------------------------------ camera picker (web capture) ----
   (function () {
      var video = $('dvVideo'), wrap = $('dvCamWrap'), idle = $('dvCamIdle'), cross = $('dvCross');
      var start = $('dvCamStart'), stop = $('dvCamStop'), picked = $('dvPicked'), sw = $('dvPickedSw');
      var hexEl = $('dvPickedHex'), nameEl = $('dvPickedName'), saveBtn = $('dvPickedSave'), savedEl = $('dvSaved');
      var sample = $('dvSample'), stream = null, SAVED_KEY = 'gm_camera_colors';
      function saved() { try { return JSON.parse(localStorage.getItem(SAVED_KEY) || '[]'); } catch (e) { return []; } }
      function renderSaved() {
         savedEl.innerHTML = saved().slice(0, 12).map(function (c) { return '<span><i style="background:' + esc(c.hex) + '"></i>' + esc(c.name) + ' <small style="color:#8c9097">' + esc(c.hex) + '</small></span>'; }).join('');
      }
      start.addEventListener('click', function () {
         if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { toast('Camera access is not available in this browser.', { kind: 'warn', icon: 'fa-regular fa-triangle-exclamation' }); return; }
         navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false }).then(function (s) {
            stream = s; video.srcObject = s; video.hidden = false; idle.hidden = true; start.hidden = true; stop.hidden = false; video.play();
         }).catch(function () { toast('Camera permission was denied.', { kind: 'warn', icon: 'fa-regular fa-triangle-exclamation' }); });
      });
      stop.addEventListener('click', function () {
         if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
         stream = null; video.hidden = true; idle.hidden = false; start.hidden = false; stop.hidden = true; cross.hidden = true;
      });
      wrap.addEventListener('click', function (e) {
         if (!stream || !video.videoWidth) return;
         var r = wrap.getBoundingClientRect();
         var fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
         sample.width = video.videoWidth; sample.height = video.videoHeight;
         var ctx = sample.getContext('2d'); ctx.drawImage(video, 0, 0);
         var x = Math.round(fx * sample.width), y = Math.round(fy * sample.height);
         var d = ctx.getImageData(Math.max(0, x - 3), Math.max(0, y - 3), 7, 7).data;
         var R = 0, G = 0, B = 0, n = d.length / 4;
         for (var i = 0; i < d.length; i += 4) { R += d[i]; G += d[i + 1]; B += d[i + 2]; }
         var hex = '#' + [R / n, G / n, B / n].map(function (v) { return ('0' + Math.round(v).toString(16)).slice(-2); }).join('').toUpperCase();
         hexEl.value = hex; sw.style.background = hex; picked.hidden = false;
         cross.hidden = false; cross.style.left = (fx * 100) + '%'; cross.style.top = (fy * 100) + '%';
      });
      saveBtn.addEventListener('click', async function () {
         var hex = hexEl.value, name = (nameEl.value || '').trim() || hex;
         var list = saved(); list.unshift({ hex: hex, name: name, at: Date.now() });
         try { localStorage.setItem(SAVED_KEY, JSON.stringify(list.slice(0, 200))); } catch (e) {}
         if (window.ColorLibrary && ColorLibrary.add) ColorLibrary.add(name, hex);
         renderSaved(); nameEl.value = '';
         // also record it as a capture so it lines up with the device history
         try { await api('POST', '/api/devices/captures', { hex: hex, name: name, source: 'web_camera', add_to_library: false }); loadCaptures(true); } catch (e) {}
         toast('“' + esc(name) + '” saved to your Color Library', { kind: 'ok', icon: 'fa-regular fa-check', href: 'paint-collection.html?open=library', linkText: 'Open' });
      });
      renderSaved();
   })();

   // --------------------------------------------- other card buttons ----
   document.querySelectorAll('[data-dv]').forEach(function (b) {
      b.addEventListener('click', function () {
         var what = b.getAttribute('data-dv');
         if (what === 'pair') { claimCode(); return; }
         if (what === 'project') {
            var st = null; try { st = sessionStorage.getItem('geomagic:state'); } catch (e) {}
            var items = (window.GM && GM.portfolio) ? GM.portfolio.list() : [];
            var src = items.length ? items[0].thumb : null;
            if (!src && !st) { toast('Save a composition to your Portfolio first, then project it.', { kind: 'warn', icon: 'fa-regular fa-triangle-exclamation', href: 'dashboard.html', linkText: 'Dashboard' }); return; }
            var w = window.open('', 'gm-projection', 'width=1280,height=800');
            if (!w) { toast('Pop-up blocked — allow pop-ups to open the projection window.', { kind: 'warn' }); return; }
            w.document.write('<title>GeoMagic projection</title><body style="margin:0;background:#000;display:flex;align-items:center;justify-content:center;height:100vh"><img src="' + (src || '') + '" style="max-width:100%;max-height:100%;object-fit:contain"></body>');
            toast('Projection window opened — drag it to the projector screen and press F11.', { kind: 'ok', icon: 'fa-regular fa-check' });
         }
      });
   });

   // ------------------------------------------------------------- boot ----
   loadConfig();
   refreshAll();
   // arriving from a headset link (?pair=123456) pre-fills the code
   var pre = new URLSearchParams(location.search).get('pair');
   if (pre && /^\d{6}$/.test(pre)) { $('dvPairCode').value = pre.slice(0, 3) + ' ' + pre.slice(3); $('dvHeadsetCard').scrollIntoView({ behavior: 'smooth' }); }
})();
