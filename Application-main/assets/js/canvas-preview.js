/* =========================================================================
   Shared canvas preview controller
   - Fits #dash-preview-wrap to #dash-canvas (no dead space around image)
   - Four corner resize handles, aspect-locked, bounded by MIN/MAX
   - Reformats #dash-status into a polished badge with icon + count + label
   ========================================================================= */
(function () {
   var MIN_W = 220;                 // canvas can't shrink below this width
   var MIN_H = 160;                 // or this height
   var MAX_W_VW_PCT = 0.92;         // can't exceed 92% of viewport width
   var MAX_H_VH_PCT = 0.78;         // or 78% of viewport height
   var ABS_MAX = 1800;              // absolute cap

   function maxW() { return Math.min(ABS_MAX, Math.round(window.innerWidth * MAX_W_VW_PCT)); }
   function maxH() { return Math.min(ABS_MAX, Math.round(window.innerHeight * MAX_H_VH_PCT)); }

   function init() {
      var wrap = document.getElementById('dash-preview-wrap');
      var canvas = document.getElementById('dash-canvas');
      if (!wrap || !canvas) return false;

      // Ensure the wrap has a host parent that centers it.
      var host = document.getElementById('dash-preview-host');
      if (!host) {
         host = document.createElement('div');
         host.id = 'dash-preview-host';
         wrap.parentNode.insertBefore(host, wrap);
         host.appendChild(wrap);
      }

      // Ensure resize handles exist.
      if (!wrap.querySelector('.canvas-resize-handle')) {
         ['nw', 'ne', 'sw', 'se'].forEach(function (c) {
            var d = document.createElement('div');
            d.className = 'canvas-resize-handle ' + c;
            d.setAttribute('data-corner', c);
            d.title = 'Drag to resize';
            wrap.appendChild(d);
         });
      }

      function isLoaded() { return !canvas.classList.contains('hidden'); }

      function clamp(w, h, aspect) {
         var mw = maxW(), mh = maxH();
         if (w < MIN_W) { w = MIN_W; h = w / aspect; }
         if (h < MIN_H) { h = MIN_H; w = h * aspect; }
         if (w > mw)    { w = mw;    h = w / aspect; }
         if (h > mh)    { h = mh;    w = h * aspect; }
         return { w: Math.round(w), h: Math.round(h) };
      }

      function fitToCanvas() {
         if (!isLoaded()) {
            wrap.classList.remove('tpl-canvas-loaded');
            wrap.style.width = '';
            wrap.style.height = '';
            canvas.style.width = '';
            canvas.style.height = '';
            return;
         }
         wrap.classList.add('tpl-canvas-loaded');
         var iw = canvas.width || 1, ih = canvas.height || 1;
         var aspect = iw / ih;
         var availW = Math.max(MIN_W, (host.getBoundingClientRect().width || window.innerWidth) - 24);
         var availH = Math.max(MIN_H, window.innerHeight - 280);
         var w = availW, h = w / aspect;
         if (h > availH) { h = availH; w = h * aspect; }
         var cl = clamp(w, h, aspect);
         wrap.style.width = cl.w + 'px';
         wrap.style.height = cl.h + 'px';
         canvas.style.width = cl.w + 'px';
         canvas.style.height = cl.h + 'px';
      }

      // Coalesce many rapid attr changes (canvas.width + canvas.height during
      // generation) into a single fitToCanvas() per animation frame, so we don't
      // thrash layout while the engine is rendering.
      var rafPending = false;
      function scheduleFit() {
         if (rafPending) return;
         rafPending = true;
         requestAnimationFrame(function () { rafPending = false; fitToCanvas(); });
      }
      var mo = new MutationObserver(function (recs) {
         var visibilityChange = false, dimChange = false;
         for (var i = 0; i < recs.length; i++) {
            if (recs[i].attributeName === 'class')                                         visibilityChange = true;
            else if (recs[i].attributeName === 'width' || recs[i].attributeName === 'height') dimChange = true;
         }
         if (!visibilityChange && !dimChange) return;
         if (visibilityChange && canvas.classList.contains('hidden')) wrap.__userSized = false;
         if (dimChange) wrap.__userSized = false; // a new image resets the user's resize
         // Skip if the user is mid-drag — their resize is the source of truth.
         if (wrap.__userSized && !visibilityChange && !dimChange) return;
         scheduleFit();
      });
      mo.observe(canvas, { attributes: true, attributeFilter: ['class', 'width', 'height'] });
      window.addEventListener('resize', function () {
         if (isLoaded() && !wrap.__userSized) fitToCanvas();
         // Still clamp after user-resize if viewport shrinks
         if (isLoaded() && wrap.__userSized) {
            var aspect = (canvas.width || 1) / (canvas.height || 1);
            var cl = clamp(wrap.offsetWidth, wrap.offsetHeight, aspect);
            wrap.style.width = cl.w + 'px';
            wrap.style.height = cl.h + 'px';
            canvas.style.width = cl.w + 'px';
            canvas.style.height = cl.h + 'px';
         }
      });
      setTimeout(fitToCanvas, 250);

      wrap.querySelectorAll('.canvas-resize-handle').forEach(function (h) {
         h.addEventListener('mousedown', function (e) {
            if (!isLoaded()) return;
            e.preventDefault();
            e.stopPropagation();
            wrap.__userSized = true;
            wrap.classList.add('tpl-canvas-resizing');
            var corner = h.dataset.corner;
            var startX = e.clientX, startY = e.clientY;
            var startW = wrap.offsetWidth, startH = wrap.offsetHeight;
            var aspect = startW / startH || ((canvas.width || 1) / (canvas.height || 1));
            var dirX = (corner === 'ne' || corner === 'se') ? 1 : -1;
            var dirY = (corner === 'sw' || corner === 'se') ? 1 : -1;
            function onMove(ev) {
               var dx = (ev.clientX - startX) * dirX;
               var dy = (ev.clientY - startY) * dirY;
               var newW, newH;
               if (Math.abs(dx) >= Math.abs(dy)) {
                  newW = startW + dx;
                  newH = newW / aspect;
               } else {
                  newH = startH + dy;
                  newW = newH * aspect;
               }
               var cl = clamp(newW, newH, aspect);
               wrap.style.width = cl.w + 'px';
               wrap.style.height = cl.h + 'px';
               canvas.style.width = cl.w + 'px';
               canvas.style.height = cl.h + 'px';
            }
            function onUp() {
               document.removeEventListener('mousemove', onMove);
               document.removeEventListener('mouseup', onUp);
               wrap.classList.remove('tpl-canvas-resizing');
            }
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
         });
      });

      // ---- Status badge enrichment ----
      var statusEl = document.getElementById('dash-status');
      if (statusEl && !statusEl.__enhanced) {
         statusEl.__enhanced = true;
         // Strip the legacy utility classes that pin it to top-left and override layout.
         ['top-2', 'left-2', 'bg-black/60', 'text-white', 'text-xs', 'font-semibold',
          'px-3', 'py-1', 'rounded-lg'].forEach(function (cl) { statusEl.classList.remove(cl); });

         // Watch text changes and re-render as { dot · count · label }.
         function render() {
            var raw = (statusEl.textContent || '').trim();
            if (!raw) return;
            // If we've already rendered, skip (the inner spans contain raw text too).
            if (statusEl.querySelector('.ds-dot')) return;

            statusEl.classList.remove('is-working', 'is-error');
            var lc = raw.toLowerCase();
            if (lc.indexOf('error') === 0 || lc.indexOf('failed') >= 0)            statusEl.classList.add('is-error');
            else if (/^(loading|building|adding|finding|generating|draw)/.test(lc)) statusEl.classList.add('is-working');

            // Detect leading number ("1234 shapes — click to select")
            var m = raw.match(/^(\d[\d,]*)\s+(.*)$/);
            var html;
            if (m) {
               var count = m[1];
               var rest = m[2].replace(/^[—–-]\s*/, '');
               html = '<span class="ds-dot"></span><span class="ds-count">' + count + '</span><span class="ds-label">' + rest + '</span>';
            } else {
               html = '<span class="ds-dot"></span><span class="ds-label">' + raw + '</span>';
            }
            statusEl.innerHTML = html;
         }

         // The status element gets its text via setStatus(); watch it.
         var smo = new MutationObserver(function () { render(); });
         smo.observe(statusEl, { childList: true, characterData: true, subtree: true });
         render();
      }

      return true;
   }

   if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
   } else {
      init();
   }
   // Retry once more after other scripts settle (handles late DOM rewires).
   setTimeout(init, 600);
})();
