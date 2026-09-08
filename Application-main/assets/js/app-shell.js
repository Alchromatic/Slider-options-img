/* =========================================================================
   GeoMagic app shell — shared across every dashboard-style page.

   - Rebuilds the left sidebar from ONE menu definition (so pages never drift:
     no missing "Dashboard" entry, no misspelled "Templates" label), marks the
     current page active, and adds the Art Library / Devices / Videos /
     Lessons / Community / Privacy entries the client asked for.
   - Desktop collapse toggle for the left panel (client asked whether the
     left panel should go — it can now be tucked away to an icon rail).
   - Header search: searches the indexed image library inline and shows
     recent searches, matching artworks, their colors and categories.
   - Small shared helpers: GM.toast(), GM.tooltip(), GM.portfolio.
   ========================================================================= */
(function () {
   'use strict';

   var page = (location.pathname.split('/').pop() || 'index.html').toLowerCase();

   var API_BASE =
      (new URLSearchParams(location.search).get('api') || '').replace(/\/$/, '') ||
      (location.protocol === 'http:' || location.protocol === 'https:'
         ? location.origin
         : 'https://alchromaticdemo.up.railway.app');

   // ---------------------------------------------------------------------
   // Menu definition (single source of truth)
   // ---------------------------------------------------------------------
   var ICONS = {
      dashboard: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M10.1082 2.83511C11.2104 1.93957 12.7896 1.93957 13.8918 2.83511L19.8918 7.71011C20.5929 8.27978 21 9.13507 21 10.0385V18.8365C21 20.4934 19.6569 21.8365 18 21.8365H16V15.8363C16 14.1795 14.6569 12.8363 13 12.8363H11C9.34315 12.8363 8 14.1794 8 15.8363V21.8363L6 21.8365C4.34315 21.8365 3 20.4934 3 18.8365V10.0385C3 9.13507 3.40709 8.27978 4.10822 7.71011L10.1082 2.83511Z" fill="currentColor"/><path d="M10 21.8365H14V15.8363C14 15.284 13.5523 14.8363 13 14.8363H11C10.4477 14.8363 10 15.284 10 15.8363V21.8365Z" fill="currentColor"/></svg>',
      portfolio: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M16 20L9.41421 13.4142C8.63316 12.6332 7.36684 12.6332 6.58579 13.4142L4 16M6 20H18C19.1046 20 20 19.1046 20 18V6C20 4.89543 19.1046 4 18 4H6C4.89543 4 4 4.89543 4 6V18C4 19.1046 4.89543 20 6 20ZM16.5 9.5C16.5 10.6046 15.6046 11.5 14.5 11.5C13.3954 11.5 12.5 10.6046 12.5 9.5C12.5 8.39543 13.3954 7.5 14.5 7.5C15.6046 7.5 16.5 8.39543 16.5 9.5Z" stroke="currentColor" stroke-width="2"/></svg>',
      templates: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path fill-rule="evenodd" clip-rule="evenodd" d="M11.3161 3.1845C11.7576 3.02395 12.2416 3.02395 12.6831 3.1845L20.7569 6.12044C22.5123 6.75874 22.5123 9.2413 20.7569 9.87961L12.6831 12.8155C12.2416 12.9761 11.7576 12.9761 11.3161 12.8155L3.2423 9.87961C1.48695 9.2413 1.48693 6.75875 3.24229 6.12044L11.3161 3.1845ZM20.0734 8.00002L11.9996 5.06409L3.92578 8.00002L11.9996 10.936L20.0734 8.00002Z" fill="currentColor"/><path d="M20.7569 13.8796L12.6831 16.8155C12.2416 16.9761 11.7576 16.9761 11.3161 16.8155L3.2423 13.8796C2.10327 13.4654 1.70334 12.2747 2.04252 11.3152L11.9996 14.936L21.9567 11.3152C22.2959 12.2747 21.896 13.4654 20.7569 13.8796Z" fill="currentColor"/><path d="M20.7569 17.8796L12.6831 20.8155C12.2416 20.9761 11.7576 20.9761 11.3161 20.8155L3.2423 17.8796C2.10327 17.4654 1.70334 16.2747 2.04252 15.3152L11.9996 18.936L21.9567 15.3152C22.2959 16.2747 21.896 17.4654 20.7569 17.8796Z" fill="currentColor"/></svg>',
      paint: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M4 20l4.5-1 9.8-9.8a2.1 2.1 0 0 0-3-3L5.5 16 4 20Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M13.5 7.5l3 3" stroke="currentColor" stroke-width="2"/></svg>',
      proposals: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="4" y="3" width="16" height="18" rx="2" stroke="currentColor" stroke-width="2"/><path d="M8 8h8M8 12h8M8 16h5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
      pricing: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M9.49962 7.91418L7.49962 9.91418L9.49962 11.9142M13.4138 19.5L21.5277 11.3861C22.3309 10.5829 22.3047 9.27275 21.4701 8.50228L17.0742 4.44458C16.7048 4.10355 16.2204 3.91418 15.7177 3.91418H8.28159C7.77881 3.91418 7.29447 4.10355 6.92503 4.44458L2.52918 8.50228C1.6945 9.27275 1.66831 10.5829 2.47153 11.3861L10.5854 19.5C11.3665 20.281 12.6328 20.281 13.4138 19.5Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
      settings: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M7.98941 5.39755L7.30515 5.23965C6.72652 5.10612 6.11991 5.28009 5.7 5.7C5.28009 6.11991 5.10612 6.72651 5.23965 7.30515L5.39755 7.9894C5.57956 8.7781 5.25434 9.59711 4.58086 10.0461L3.74885 10.6008C3.28101 10.9127 3 11.4377 3 12C3 12.5623 3.28101 13.0873 3.74885 13.3992L4.58086 13.9539C5.25434 14.4029 5.57956 15.2219 5.39756 16.0106L5.23965 16.6949C5.10612 17.2735 5.28009 17.8801 5.7 18.3C6.11991 18.7199 6.72651 18.8939 7.30515 18.7604L7.9894 18.6024C8.7781 18.4204 9.59711 18.7457 10.0461 19.4191L10.6008 20.2512C10.9127 20.719 11.4377 21 12 21C12.5623 21 13.0873 20.719 13.3992 20.2512L13.9539 19.4191C14.4029 18.7457 15.2219 18.4204 16.0106 18.6024L16.6949 18.7604C17.2735 18.8939 17.8801 18.7199 18.3 18.3C18.7199 17.8801 18.8939 17.2735 18.7604 16.6949L18.6024 16.0106C18.4204 15.2219 18.7457 14.4029 19.4191 13.9539L20.2512 13.3992C20.719 13.0873 21 12.5623 21 12C21 11.4377 20.719 10.9127 20.2512 10.6008L19.4191 10.0461C18.7457 9.59711 18.4204 8.7781 18.6024 7.98941L18.7604 7.30515C18.8939 6.72652 18.7199 6.11991 18.3 5.7C17.8801 5.28009 17.2735 5.10612 16.6949 5.23965L16.0106 5.39755C15.2219 5.57956 14.4029 5.25434 13.9539 4.58086L13.3992 3.74884C13.0873 3.28101 12.5623 3 12 3C11.4377 3 10.9127 3.28101 10.6008 3.74885L10.0461 4.58086C9.59711 5.25434 8.7781 5.57956 7.98941 5.39755Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
      devices: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M3 9.5A2.5 2.5 0 0 1 5.5 7h13A2.5 2.5 0 0 1 21 9.5v4.2c0 1.6-1.2 2.8-2.8 2.8h-1.4c-.9 0-1.7-.5-2.2-1.2l-.9-1.3a2 2 0 0 0-3.4 0l-.9 1.3c-.5.7-1.3 1.2-2.2 1.2H5.8A2.8 2.8 0 0 1 3 13.7V9.5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
      artlib: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="3" y="4" width="18" height="16" rx="2" stroke="currentColor" stroke-width="2"/><path d="M3 16l5-5 4 4 3-3 6 6" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><circle cx="15.5" cy="8.5" r="1.5" fill="currentColor"/></svg>',
      library: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 3a9 9 0 1 0 0 18c1.3 0 2-.9 2-2 0-.6-.3-1-.6-1.4-.3-.4-.5-.8-.5-1.3 0-1 .8-1.8 1.8-1.8H16a5 5 0 0 0 5-5c0-3.9-4-6.5-9-6.5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><circle cx="7.5" cy="12" r="1.3" fill="currentColor"/><circle cx="9.5" cy="7.8" r="1.3" fill="currentColor"/><circle cx="14.5" cy="7.8" r="1.3" fill="currentColor"/></svg>',
      videos: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="3" y="6" width="13" height="12" rx="2" stroke="currentColor" stroke-width="2"/><path d="m16 10 5-2.5v9L16 14" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
      lessons: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="4" y="3" width="16" height="18" rx="2" stroke="currentColor" stroke-width="2"/><path d="M8 8h8M8 12h8M8 16h5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
      community: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M4 19v-1a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v1" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="10" cy="7" r="3" stroke="currentColor" stroke-width="2"/><path d="M16 11a3 3 0 1 0 0-6M20 19v-1a4 4 0 0 0-3-3.9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
      privacy: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 3 5 6v5c0 4.4 3 8.3 7 9.5 4-1.2 7-5.1 7-9.5V6l-7-3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="m9.5 12 1.8 1.8L15 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
      guide: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 6.5C9.5 4.7 6.3 4.6 3 5.8v12.7c3.3-1.2 6.5-1.1 9 .7 2.5-1.8 5.7-1.9 9-.7V5.8c-3.3-1.2-6.5-1.1-9 .7Zm0 0v12.7" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>'
   };

   var MENU = [
      { group: 'main', items: [
         { label: 'Dashboard',        href: 'dashboard.html',        icon: 'dashboard' },
         { label: 'Portfolio',        href: 'portfolio.html',        icon: 'portfolio' },
         { label: 'Templates',        href: 'templates.html',        icon: 'templates', match: ['palette-selection.html'] },
         { label: 'Art Library',      href: 'image-gallery.html',    icon: 'artlib', match: ['art-map.html'] },
         { label: 'Paint Collection', href: 'paint-collection.html', icon: 'paint' },
         { label: 'Proposals',        href: 'proposals.html',        icon: 'proposals' },
         { label: 'Pricing',          href: 'pricing.html',          icon: 'pricing' },
         { label: 'Settings',         href: 'profile.html',          icon: 'settings' }
      ]},
      { group: 'more', items: [
         { label: 'Devices',          href: 'devices.html',          icon: 'devices', badge: 'VR/AR' },
         { label: 'Videos',           href: 'videos.html',           icon: 'videos' },
         { label: 'Art Lessons',      href: 'art-lesson.html',       icon: 'lessons' },
         { label: 'Color Guide',      href: 'color-guide.html',      icon: 'guide' },
         { label: 'Community',        href: 'invite-teams.html',     icon: 'community' },
         { label: 'Privacy Policy',   href: 'privacy-policy.html',   icon: 'privacy' }
      ]}
   ];

   var LINK_CLS = 'w-full flex items-center gap-3 p-3 text-neutral-4 text-[15px] leading-6 font-semibold -tracking-[0.15px] rounded-xl hover:text-neutral-6 dark:hover:text-neutral-1';

   function isActive(item) {
      var base = item.href.split('?')[0].toLowerCase();
      if (page === base) return true;
      return (item.match || []).indexOf(page) > -1;
   }

   function buildSidebar() {
      var ul = document.querySelector('aside ul.sidebar-menu');
      if (!ul) return;
      var html = '';
      MENU.forEach(function (g, gi) {
         if (gi > 0) html += '<li class="gm-menu-sep" aria-hidden="true"></li>';
         g.items.forEach(function (it) {
            var on = isActive(it);
            html += '<li><a href="' + it.href + '" class="' + LINK_CLS + (on ? ' active' : '') + '" data-gm-label="' + it.label + '">'
               + (ICONS[it.icon] || '') + '<span class="gm-menu-label">' + it.label + '</span>'
               + (it.badge ? '<span class="gm-menu-badge">' + it.badge + '</span>' : '')
               + '</a></li>';
         });
      });
      ul.innerHTML = html;
      ul.classList.add('gm-sidebar-menu');
   }

   // ---------------------------------------------------------------------
   // Collapse toggle (desktop only). State persists in localStorage.
   // ---------------------------------------------------------------------
   var COLLAPSE_KEY = 'gm_sidebar_collapsed';
   function applyCollapsed(on) {
      document.documentElement.classList.toggle('gm-sidebar-collapsed', !!on);
      var b = document.getElementById('gmSidebarCollapse');
      if (b) {
         b.setAttribute('aria-expanded', on ? 'false' : 'true');
         b.title = on ? 'Expand left panel' : 'Collapse left panel';
      }
      try { localStorage.setItem(COLLAPSE_KEY, on ? '1' : '0'); } catch (_) {}
      window.dispatchEvent(new Event('resize'));
   }
   function buildCollapseToggle() {
      var aside = document.querySelector('aside');
      if (!aside || document.getElementById('gmSidebarCollapse')) return;
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.id = 'gmSidebarCollapse';
      btn.className = 'gm-collapse-btn';
      btn.innerHTML = '<i class="fa-regular fa-angles-left"></i><span>Hide panel</span>';
      btn.addEventListener('click', function () {
         applyCollapsed(!document.documentElement.classList.contains('gm-sidebar-collapsed'));
      });
      aside.appendChild(btn);
      var saved = '0';
      try { saved = localStorage.getItem(COLLAPSE_KEY) || '0'; } catch (_) {}
      if (saved === '1') applyCollapsed(true);
   }

   // ---------------------------------------------------------------------
   // Toast
   // ---------------------------------------------------------------------
   function toast(msg, opts) {
      opts = opts || {};
      var host = document.getElementById('gmToastHost');
      if (!host) {
         host = document.createElement('div');
         host.id = 'gmToastHost';
         document.body.appendChild(host);
      }
      var t = document.createElement('div');
      t.className = 'gm-toast' + (opts.kind ? ' ' + opts.kind : '');
      t.innerHTML = (opts.icon ? '<i class="' + opts.icon + '"></i>' : '') + '<span>' + msg + '</span>'
         + (opts.href ? '<a href="' + opts.href + '">' + (opts.linkText || 'Open') + '</a>' : '');
      host.appendChild(t);
      requestAnimationFrame(function () { t.classList.add('show'); });
      setTimeout(function () { t.classList.remove('show'); setTimeout(function () { t.remove(); }, 300); }, opts.ms || 3200);
      return t;
   }

   // ---------------------------------------------------------------------
   // Tooltip helper — one consistent style (title + description card),
   // matching the HTML/Figma mockup. Replaces native `title` tooltips so the
   // user never sees two tooltips for one icon.
   // ---------------------------------------------------------------------
   function tooltip(el, title, desc, width) {
      if (!el || !el.parentNode) return;
      var wrap = el.parentNode;
      if (!wrap.classList.contains('group')) {
         var w = document.createElement('div');
         w.className = 'shrink-0 relative group';
         el.parentNode.insertBefore(w, el);
         w.appendChild(el);
         wrap = w;
      }
      var old = wrap.querySelector(':scope > .tpl-tooltip');
      if (old) old.remove();
      var tip = document.createElement('div');
      tip.className = 'tpl-tooltip ' + (width || 'w-38.5');
      tip.innerHTML = '<div class="bg-white dark:bg-neutral-8 p-2.5 rounded-lg text-center">'
         + '<h6 class="text-neutral-7 dark:text-white text-sm leading-[17px] -tracking-[0.02em] mb-1">' + title + '</h6>'
         + (desc ? '<p class="text-[#5D6C76] text-xs leading-[15px] -tracking-[0.02em]">' + desc + '</p>' : '')
         + '</div>';
      wrap.appendChild(tip);
      if (el.hasAttribute('title')) {
         el.setAttribute('data-tip', el.getAttribute('title'));
         el.removeAttribute('title');
      }
   }

   // ---------------------------------------------------------------------
   // Portfolio store (per user, in this browser)
   // ---------------------------------------------------------------------
   function userKey() {
      var u = null;
      try { u = window.Auth && Auth.getUser && Auth.getUser(); } catch (_) {}
      var id = (u && (u.email || u.id)) ? String(u.email || u.id).toLowerCase() : 'guest';
      return 'gm_portfolio:' + id;
   }
   var portfolio = {
      list: function () {
         try { return JSON.parse(localStorage.getItem(userKey()) || '[]'); } catch (_) { return []; }
      },
      _write: function (items) {
         try { localStorage.setItem(userKey(), JSON.stringify(items)); return true; }
         catch (e) { return false; }
      },
      add: function (item) {
         var items = portfolio.list();
         item.id = item.id || ('p' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6));
         item.createdAt = item.createdAt || Date.now();
         items.unshift(item);
         // localStorage is ~5MB: drop shape data from the oldest entries first,
         // then drop the oldest entries, until the save fits.
         while (!portfolio._write(items) && items.length) {
            var shrunk = false;
            for (var i = items.length - 1; i >= 0; i--) {
               if (items[i].shapes) { delete items[i].shapes; shrunk = true; break; }
            }
            if (!shrunk) items.pop();
         }
         return item;
      },
      remove: function (id) {
         portfolio._write(portfolio.list().filter(function (p) { return p.id !== id; }));
      },
      get: function (id) {
         return portfolio.list().filter(function (p) { return p.id === id; })[0] || null;
      }
   };

   // ---------------------------------------------------------------------
   // Header search → image library
   // ---------------------------------------------------------------------
   var RECENT_KEY = 'gm_recent_searches';
   function recent() { try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); } catch (_) { return []; } }
   function remember(q) {
      q = (q || '').trim(); if (!q) return;
      var r = recent().filter(function (x) { return x.toLowerCase() !== q.toLowerCase(); });
      r.unshift(q); r = r.slice(0, 5);
      try { localStorage.setItem(RECENT_KEY, JSON.stringify(r)); } catch (_) {}
   }
   function forget(q) {
      var r = recent().filter(function (x) { return x !== q; });
      try { localStorage.setItem(RECENT_KEY, JSON.stringify(r)); } catch (_) {}
   }
   function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
   function thumbOf(a) {
      if (a.source === 'bucket') return API_BASE + '/api/library/img/' + a.id + '?size=thumb';
      return 'assets/library/' + a.id + '.jpg';
   }
   function useArtwork(a) {
      try {
         localStorage.setItem('gm_pending_library_image', JSON.stringify({
            id: a.id, title: a.title, url: API_BASE + '/api/library/img/' + a.id + '?size=full'
         }));
      } catch (_) {}
      location.href = 'dashboard.html';
   }
   var facetCache = null;
   function facets() {
      if (facetCache) return Promise.resolve(facetCache);
      return fetch(API_BASE + '/api/library/facets').then(function (r) { return r.ok ? r.json() : {}; })
         .then(function (f) { facetCache = f || {}; return facetCache; })
         .catch(function () { return {}; });
   }
   function colorsOf(list) {
      // Pull any color info the library exposes (palette / dominant_colors / colors);
      // dedupe to a handful of chips.
      var out = [], seen = {};
      list.forEach(function (a) {
         var pal = a.palette || a.dominant_colors || a.colors || [];
         (Array.isArray(pal) ? pal : []).forEach(function (c) {
            var hex = typeof c === 'string' ? c : (c && (c.hex || c.color));
            if (!hex || !/^#?[0-9a-f]{6}$/i.test(hex)) return;
            hex = ('#' + hex.replace('#', '')).toUpperCase();
            if (!seen[hex] && out.length < 6) { seen[hex] = 1; out.push(hex); }
         });
      });
      return out;
   }

   function buildSearch() {
      var inputs = document.querySelectorAll('header input[type="search"], aside input[type="search"]');
      inputs.forEach(function (input) {
         var box = input.parentNode;
         if (!box || box.querySelector('.gm-search-dd')) return;
         box.classList.add('gm-search-box');
         var dd = document.createElement('div');
         dd.className = 'gm-search-dd';
         dd.hidden = true;
         box.appendChild(dd);

         var timer = null, seq = 0;

         function close() { dd.hidden = true; }
         function open() { dd.hidden = false; }

         function renderIdle() {
            var r = recent();
            var html = '';
            if (r.length) {
               html += '<div class="gm-sd-title">Recent search</div>';
               r.forEach(function (q) {
                  html += '<div class="gm-sd-row gm-sd-recent" data-q="' + esc(q) + '">'
                     + '<span class="gm-sd-ico"><i class="fa-regular fa-clock-rotate-left"></i></span>'
                     + '<span class="gm-sd-text"><small>Search again</small><b>' + esc(q) + '</b></span>'
                     + '<button type="button" class="gm-sd-x" data-forget="' + esc(q) + '" title="Remove"><i class="fa-regular fa-xmark"></i></button></div>';
               });
            } else {
               html += '<div class="gm-sd-empty">Search the image library — titles, artists, colors and categories.</div>';
            }
            html += '<a class="gm-sd-link" href="image-gallery.html"><i class="fa-regular fa-images"></i> Browse the whole library</a>';
            dd.innerHTML = html;
            open();
         }

         function renderResults(q, data, f) {
            var list = (data && (data.items || data.artworks || data.results)) || [];
            var html = '';
            if (!list.length) {
               html += '<div class="gm-sd-empty">No artworks match “' + esc(q) + '”.</div>';
            } else {
               html += '<div class="gm-sd-title">Artworks</div>';
               list.slice(0, 6).forEach(function (a) {
                  html += '<div class="gm-sd-row gm-sd-art" data-id="' + esc(a.id) + '">'
                     + '<img class="gm-sd-thumb" src="' + esc(thumbOf(a)) + '" alt="" loading="lazy" onerror="this.style.visibility=\'hidden\'">'
                     + '<span class="gm-sd-text"><small>' + esc(a.artist || a.source || 'Artwork') + '</small><b>' + esc(a.title || 'Untitled') + '</b></span>'
                     + '<span class="gm-sd-use">Use</span></div>';
               });
               var cols = colorsOf(list);
               if (cols.length) {
                  html += '<div class="gm-sd-title">Colors</div><div class="gm-sd-chips">';
                  cols.forEach(function (h) {
                     html += '<a class="gm-sd-chip" href="templates.html?target=' + encodeURIComponent(h) + '"><i style="background:' + h + '"></i>' + h + '</a>';
                  });
                  html += '</div>';
               }
            }
            var cats = ((f && f.genre) || []).slice(0, 4);
            if (cats.length) {
               html += '<div class="gm-sd-title">Categories</div><div class="gm-sd-chips">';
               cats.forEach(function (c) {
                  html += '<a class="gm-sd-chip gm-sd-cat" href="image-gallery.html?search=' + encodeURIComponent(q) + '&genre=' + encodeURIComponent(c.value) + '">' + esc(c.value) + ' <small>' + c.count + '</small></a>';
               });
               html += '</div>';
            }
            html += '<a class="gm-sd-link" href="image-gallery.html?search=' + encodeURIComponent(q) + '"><i class="fa-regular fa-magnifying-glass"></i> See all results for “' + esc(q) + '”</a>';
            dd.innerHTML = html;
            open();
         }

         function run(q) {
            var my = ++seq;
            dd.innerHTML = '<div class="gm-sd-empty">Searching…</div>'; open();
            var params = new URLSearchParams({ page: 1, page_size: 6, search: q });
            Promise.all([
               fetch(API_BASE + '/api/library/artworks?' + params.toString()).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
               facets()
            ]).then(function (res) {
               if (my !== seq) return;
               if (!res[0]) { dd.innerHTML = '<div class="gm-sd-empty">The image library is not reachable right now.</div>'; return; }
               renderResults(q, res[0], res[1]);
            });
         }

         input.setAttribute('autocomplete', 'off');
         input.addEventListener('focus', function () { if (!input.value.trim()) renderIdle(); else run(input.value.trim()); });
         input.addEventListener('input', function () {
            clearTimeout(timer);
            var q = input.value.trim();
            if (!q) { renderIdle(); return; }
            timer = setTimeout(function () { run(q); }, 260);
         });
         input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
               var q = input.value.trim();
               if (q) { remember(q); location.href = 'image-gallery.html?search=' + encodeURIComponent(q); }
            } else if (e.key === 'Escape') { close(); input.blur(); }
         });
         dd.addEventListener('mousedown', function (e) { e.preventDefault(); }); // keep focus
         dd.addEventListener('click', function (e) {
            var x = e.target.closest('[data-forget]');
            if (x) { forget(x.getAttribute('data-forget')); renderIdle(); return; }
            var rc = e.target.closest('.gm-sd-recent');
            if (rc) { input.value = rc.getAttribute('data-q'); run(input.value); return; }
            var art = e.target.closest('.gm-sd-art');
            if (art) {
               remember(input.value.trim());
               useArtwork({ id: art.getAttribute('data-id'), title: art.querySelector('b').textContent });
               return;
            }
            var a = e.target.closest('a');
            if (a) remember(input.value.trim());
         });
         document.addEventListener('click', function (e) { if (!box.contains(e.target)) close(); });
      });
   }

   // ---------------------------------------------------------------------
   // Color guide (educational guide modal) — available on every app page.
   // The modal's CSS/JS/catalog are only fetched the first time it is opened
   // (color-guide.html loads them directly). GM.openColorGuide(id) is the
   // programmatic entry point; any element with data-open-color-guide works too.
   // ---------------------------------------------------------------------
   var guideLoading = null;
   function loadColorGuide() {
      if (window.GM && window.GM.colorGuide) return Promise.resolve(window.GM.colorGuide);
      if (guideLoading) return guideLoading;
      guideLoading = new Promise(function (resolve, reject) {
         if (!document.querySelector('link[href^="assets/css/color-guide.css"]')) {
            var l = document.createElement('link');
            l.rel = 'stylesheet'; l.href = 'assets/css/color-guide.css';
            document.head.appendChild(l);
         }
         var sc = document.createElement('script');
         sc.src = 'assets/js/color-guide.js';
         sc.onload = function () { window.GM.colorGuide ? resolve(window.GM.colorGuide) : reject(new Error('Color guide failed to initialise')); };
         sc.onerror = function () { reject(new Error('Could not load the color guide')); };
         document.head.appendChild(sc);
      });
      return guideLoading;
   }
   function openColorGuide(id) {
      return loadColorGuide().then(function (cg) { return cg.open(id); }).catch(function (e) {
         toast(e.message || 'Could not open the color guide.', { kind: 'warn', icon: 'fa-regular fa-triangle-exclamation' });
      });
   }
   function buildGuideLauncher() {
      // Header "Color guide" button, placed right after the "+ New" action
      // (desktop header and the mobile copy inside the sidebar).
      var anchors = document.querySelectorAll('a.theme-btn-primary');
      anchors.forEach(function (a) {
         if (!/\bNew\b/.test(a.textContent) || a.parentNode.querySelector('.gm-guide-btn')) return;
         var b = document.createElement('button');
         b.type = 'button';
         b.className = 'gm-guide-btn';
         b.setAttribute('data-open-color-guide', '');
         b.title = 'Color guide — definitions and visual examples';
         b.innerHTML = ICONS.guide.replace('width="24" height="24"', 'width="20" height="20"') + '<span>Color guide</span>';
         a.parentNode.insertBefore(b, a.nextSibling);
      });
      // Inject the launcher styles without loading the whole guide stylesheet up front.
      if (!document.getElementById('gmGuideBtnCss')) {
         var st = document.createElement('style');
         st.id = 'gmGuideBtnCss';
         st.textContent = '.gm-guide-btn{display:inline-flex;align-items:center;gap:8px;height:48px;padding:0 18px;border-radius:12px;font-size:15px;font-weight:700;letter-spacing:-.15px;color:#1a1d1f;background:#f4f4f4;border:2px solid #f4f4f4;cursor:pointer;white-space:nowrap;transition:background .15s,color .15s,border-color .15s}.gm-guide-btn:hover{background:#fff;border-color:#2a85ff;color:#2a85ff}.gm-guide-btn svg{flex:0 0 20px}.dark .gm-guide-btn{color:#fcfcfc;background:transparent;border-color:#272b30}.dark .gm-guide-btn:hover{border-color:#2a85ff;color:#2a85ff}@media (max-width:1279px){.gm-guide-btn span{display:none}.gm-guide-btn{padding:0 12px}}';
         document.head.appendChild(st);
      }
      if (!document.body.hasAttribute('data-gm-guide-delegate')) {
         document.body.setAttribute('data-gm-guide-delegate', '1');
         document.addEventListener('click', function (e) {
            var t = e.target.closest('[data-open-color-guide]');
            if (!t || (window.GM && window.GM.colorGuide)) return;   // color-guide.js handles it once loaded
            e.preventDefault();
            openColorGuide(t.getAttribute('data-open-color-guide') || undefined);
         });
      }
   }

   // ---------------------------------------------------------------------
   // Boot
   // ---------------------------------------------------------------------
   function boot() {
      buildSidebar();
      buildCollapseToggle();
      buildSearch();
      buildGuideLauncher();
      // ?guide=<id> deep link works on every app page, not only color-guide.html
      var wanted = new URLSearchParams(location.search).get('guide');
      if (wanted && !(window.GM && window.GM.colorGuide)) openColorGuide(wanted);
   }
   if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
   else boot();

   window.GM = window.GM || {};
   window.GM.toast = toast;
   window.GM.tooltip = tooltip;
   window.GM.portfolio = portfolio;
   window.GM.apiBase = API_BASE;
   window.GM.setSidebarCollapsed = applyCollapsed;
   window.GM.openColorGuide = openColorGuide;
})();
