/* Billing / entitlements client + per-plan gating.
 * Talks to /api/billing/*. Token comes from auth.js (localStorage gm_access_token).
 * Exposes window.Billing. Auto-applies gating (rendering icons + palette dropdown)
 * and meters image generation via Billing.consumeImage(). */
(function () {
   const API_BASE =
      (new URLSearchParams(location.search).get('api') || '').replace(/\/$/, '') ||
      'https://alchromaticdemo.up.railway.app';

   const BASIC_PALETTE = 'Common Color Names & Values';

   function token() {
      try { return localStorage.getItem('gm_access_token'); } catch (e) { return null; }
   }
   function authHeaders(extra) {
      const h = Object.assign({}, extra || {});
      const t = token();
      if (t) h['Authorization'] = 'Bearer ' + t;
      return h;
   }

   const Billing = {
      apiBase: API_BASE,
      ent: null,          // resolved entitlements for the current user

      async load(force) {
         if (this.ent && !force) return this.ent;
         try {
            const res = await fetch(`${API_BASE}/api/billing/me`, { headers: authHeaders() });
            if (res.ok) this.ent = await res.json();
         } catch (e) { /* offline → leave ent null (fail open) */ }
         return this.ent;
      },

      async plans() {
         const res = await fetch(`${API_BASE}/api/billing/plans`);
         return (await res.json()).plans || [];
      },

      /* Check + consume one image. Returns {allowed, remaining, reason}. */
      async consumeImage() {
         try {
            const res = await fetch(`${API_BASE}/api/billing/consume-image`, {
               method: 'POST',
               headers: authHeaders({ 'Content-Type': 'application/json' }),
            });
            if (res.status === 401) return { allowed: false, reason: 'Please sign in.' };
            const data = await res.json().catch(() => ({}));
            // Refresh cached counts + the live badge so "images left" updates in real time.
            if (this.ent && !this.ent.unlimited_images && typeof data.remaining === 'number') {
               this.ent.images_remaining = data.remaining;
               // When a PAYG credit was spent, keep the credits figure in sync too.
               if (data.source === 'credit' && typeof this.ent.credits === 'number') {
                  this.ent.credits = Math.max(0, this.ent.credits - 1);
               }
               if (typeof Billing.renderBadge === 'function') Billing.renderBadge();
            }
            return data;
         } catch (e) {
            // Network failure: fail open so a backend hiccup doesn't block users.
            return { allowed: true, remaining: null, offline: true };
         }
      },

      async startCheckout(planId) {
         if (!token()) { window.location.href = 'signin.html'; return; }
         // Stripe can only redirect to an http(s) URL. When the app is served from
         // the backend (the supported setup) location.origin is that http origin and
         // the token survives the round-trip. If opened via file://, redirect Stripe
         // back to the backend-hosted copy of the page instead.
         const success = location.protocol === 'file:'
            ? 'https://alchromaticdemo.up.railway.app/' + (location.pathname.split('/').pop() || 'pricing.html')
            : location.origin + location.pathname;
         const res = await fetch(`${API_BASE}/api/billing/create-checkout-session`, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ plan_id: planId, success_url: success, cancel_url: success + '?status=cancelled' }),
         });
         const data = await res.json().catch(() => ({}));
         if (!res.ok || !data.checkout_url) throw new Error(data.detail || 'Could not start checkout');
         window.location.href = data.checkout_url;
      },

      async verifySession(sessionId) {
         const res = await fetch(`${API_BASE}/api/billing/verify-session`, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ session_id: sessionId }),
         });
         return res.json().catch(() => ({}));
      },

      // ---- gating helpers ----
      allowedLogics() {
         return (this.ent && this.ent.rendering_logics) || null; // null → no restriction (fail open)
      },
      paletteMode() {
         return (this.ent && this.ent.palettes) || 'all';
      },
      isLogicAllowed(logic) {
         const allowed = this.allowedLogics();
         if (!allowed) return true;
         return allowed.indexOf(logic) !== -1;
      },
   };

   window.Billing = Billing;

   // =========================================================================
   // Gating: rendering-logic icons
   // =========================================================================
   function gateRenderingIcons() {
      const allowed = Billing.allowedLogics();
      if (!allowed) return;
      const btns = document.querySelectorAll('.rendering-icon-btn');
      btns.forEach((btn) => {
         let logic = btn.dataset.logic;
         if (!logic && btn.id === 'selectiveResolutionIcon') logic = 'selective_resolution';
         if (!logic) return;
         if (allowed.indexOf(logic) === -1) {
            if (btn.dataset.locked === '1') return;
            btn.dataset.locked = '1';
            btn.classList.add('locked');
            btn.style.opacity = '0.4';
            btn.style.cursor = 'not-allowed';
            const base = btn.getAttribute('title') || '';
            btn.setAttribute('title', base + ' - Upgrade to unlock');
            // Block activation (capture phase, before the page's own handler).
            btn.addEventListener('click', function (e) {
               e.preventDefault();
               e.stopImmediatePropagation();
               showUpgrade('This rendering style is available on a higher plan.');
            }, true);
         }
      });
   }

   // =========================================================================
   // Gating: paint-palette dropdown
   // =========================================================================
   function allowedPaletteName(name, brandKeys, mode) {
      if (mode === 'all') return true;
      if (name === 'default' || name === BASIC_PALETTE) return true; // basic always allowed
      if (mode === 'single') return brandKeys.length && name === brandKeys[0];
      return false; // basic
   }

   function gatePaletteSelect(sel) {
      const mode = Billing.paletteMode();
      if (mode === 'all') return;
      const presets = window.PALETTE_PRESETS || {};
      const brandKeys = Object.keys(presets).filter((k) => k !== BASIC_PALETTE);
      let changed = false;
      Array.from(sel.options).forEach((opt) => {
         const keep = allowedPaletteName(opt.value, brandKeys, mode);
         if (!keep && opt.parentNode) { opt.remove(); changed = true; }
      });
      // Make sure the current selection is still valid.
      if (changed && sel.selectedIndex < 0 && sel.options.length) sel.selectedIndex = 0;
   }

   function watchPaletteSelect() {
      if (Billing.paletteMode() === 'all') return;
      const sel = document.getElementById('trycolorsPaletteSelect');
      if (!sel) return;
      gatePaletteSelect(sel);
      // palette-mixing.js repopulates the <select> asynchronously → re-gate on change.
      const obs = new MutationObserver(() => gatePaletteSelect(sel));
      obs.observe(sel, { childList: true });
   }

   // =========================================================================
   // Upgrade prompt + usage badge
   // =========================================================================
   function showUpgrade(msg) {
      if (confirm((msg || 'Upgrade to unlock this feature.') + '\n\nView plans now?')) {
         window.location.href = 'pricing.html';
      }
   }
   Billing.showUpgrade = showUpgrade;

   function renderBadge() {
      if (!Billing.ent) return;
      const e = Billing.ent;
      let badge = document.getElementById('planBadge');
      if (!badge) {
         badge = document.createElement('a');
         badge.id = 'planBadge';
         badge.href = 'pricing.html';
         badge.style.cssText =
            'position:fixed;left:16px;bottom:16px;z-index:9999;display:inline-flex;align-items:center;gap:8px;' +
            'padding:8px 14px;border-radius:9999px;color:#fff;font:600 13px/1 Inter,sans-serif;' +
            'text-decoration:none;box-shadow:0 6px 18px -6px rgba(0,0,0,.4);transition:background .2s;';
         document.body.appendChild(badge);
      }
      // images_remaining is the COMBINED total (monthly allowance + PAYG credits).
      const remaining = e.unlimited_images ? Infinity : (e.images_remaining || 0);
      const remainingLabel = e.unlimited_images ? '∞' : remaining;
      // Out of images → red; running low (≤2) → amber; otherwise brand blue.
      badge.style.background = remaining <= 0 ? '#FF5252' : (remaining <= 2 ? '#FF9F2D' : '#2A85FF');
      badge.textContent = remaining <= 0
         ? `${e.plan_name} · Out of images — upgrade`
         : `${e.plan_name} · ${remainingLabel} images left`;
   }
   Billing.renderBadge = renderBadge;

   // =========================================================================
   // Post-checkout activation (Stripe returns ?status=success&session_id=...)
   // =========================================================================
   async function handleReturn() {
      const qs = new URLSearchParams(location.search);
      if (qs.get('status') === 'success' && qs.get('session_id')) {
         try {
            await Billing.verifySession(qs.get('session_id'));
            await Billing.load(true);
         } catch (e) { /* ignore */ }
         // clean the URL
         const clean = location.origin + location.pathname;
         window.history.replaceState({}, '', clean);
      }
   }

   // =========================================================================
   // Boot
   // =========================================================================
   async function boot() {
      await handleReturn();
      await Billing.load();
      gateRenderingIcons();
      watchPaletteSelect();
      renderBadge();
      // Re-apply once more after the page's own scripts settle (icons added late).
      setTimeout(() => { gateRenderingIcons(); }, 800);
   }

   if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', boot);
   } else {
      boot();
   }
})();
