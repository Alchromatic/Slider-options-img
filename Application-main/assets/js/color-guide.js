/* =========================================================================
   GeoMagic Color guide — educational guide modal + guide library.

   Content comes from assets/js/color-guide-data.js (window.GM_COLOR_GUIDE_CATALOG),
   generated from the client's color-study-catalog.json. This file:

     - builds ONE <dialog class="cg-dialog"> on demand and renders any of the
       20 guides into it (search, tabs for the four "original" guides,
       breadcrumb + "Browse topics", entry cards, tip, previous/next footer)
     - renders the guide library grid on color-guide.html (GM.colorGuide.renderLibrary)
     - deep links: color-guide.html#guide=<id>  or  any page ?guide=<id>

   Public API (window.GM.colorGuide):
     open(id?)          open the modal on a guide (default: first guide)
     close()
     renderLibrary(el)  build the searchable library into `el`
     catalog            the loaded catalog
   ========================================================================= */
(function () {
   'use strict';

   var EXPLORER_PAGE = 'templates.html';       // GeoMagic "Colors" page = the colour explorer
   var page = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
   var isLibraryPage = page === 'color-guide.html' || page === 'color-guide';   // dev servers may drop .html

   var catalog = null;
   var dialog = null;
   var current = null;
   var els = {};

   function esc(s) {
      return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
         return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
      });
   }
   function $(id) { return document.getElementById(id); }
   function guideById(id) { for (var i = 0; i < catalog.guides.length; i++) if (catalog.guides[i].id === id) return catalog.guides[i]; return null; }
   function categoryTitle(id) { for (var i = 0; i < catalog.categories.length; i++) if (catalog.categories[i].id === id) return catalog.categories[i].title; return id; }
   function collectionById(id) { for (var i = 0; i < catalog.collections.length; i++) if (catalog.collections[i].id === id) return catalog.collections[i]; return null; }
   function assetImg(id, cls) {
      var a = catalog.assets[id];
      if (!a) return '';
      return '<img class="' + (cls || '') + '" src="' + esc(a.path) + '" alt="' + esc(a.alt) + '" width="' + a.width + '" height="' + a.height + '" loading="lazy">';
   }
   function searchable(g) {
      var parts = [g.title, g.description, g.tip, g.header_description, g.tip_description];
      g.entries.forEach(function (e) { parts.push(e.title, e.description, e.additional_description); });
      return parts.join(' ').toLowerCase();
   }
   function highlight(text, q) {
      if (!q) return esc(text);
      var i = String(text).toLowerCase().indexOf(q.toLowerCase());
      if (i < 0) return esc(text);
      return esc(text.slice(0, i)) + '<mark>' + esc(text.slice(i, i + q.length)) + '</mark>' + esc(text.slice(i + q.length));
   }

   var ICON_BOOK = '<svg width="34" height="34" viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="M16 7C12 4 7 4 3 6v21c4-2 9-2 13 1 4-3 9-3 13-1V6c-4-2-9-2-13 1Zm0 0v21" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>';
   var ICON_SEARCH = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="2"/><path d="m20 20-3.5-3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
   var ICON_BULB = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
   var ICON_INFO = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M12 11v5M12 8h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
   var ARROW_R = '<span aria-hidden="true">&rarr;</span>';
   var ARROW_L = '<span aria-hidden="true">&larr;</span>';
   var CHEV_R = '<span aria-hidden="true">&rsaquo;</span>';

   // ---------------------------------------------------------------------
   // Modal DOM (built once)
   // ---------------------------------------------------------------------
   function build() {
      if (dialog) return;
      dialog = document.createElement('dialog');
      dialog.className = 'cg-dialog cg-theme-light';
      dialog.setAttribute('aria-labelledby', 'cgTitle');
      dialog.innerHTML =
         '<div class="cg-shell">' +
            '<header class="cg-header">' +
               '<div class="cg-brand">' + ICON_BOOK + '<span>' + esc(catalog.ui.guide_title) + '</span></div>' +
               '<button type="button" class="cg-close" id="cgClose" aria-label="' + esc(catalog.ui.close) + '">&times;</button>' +
            '</header>' +
            '<p id="cgHeaderDesc" class="cg-header-desc" hidden></p>' +
            '<label class="cg-search">' + ICON_SEARCH + '<input id="cgSearch" type="search" placeholder="' + esc(catalog.ui.search_placeholder) + '" aria-label="Search all guide terms" autocomplete="off"></label>' +
            '<div id="cgSearchResults" class="cg-search-results" hidden aria-live="polite"></div>' +
            '<div class="cg-nav" id="cgNav">' +
               '<div class="cg-crumb" id="cgCrumb"></div>' +
               '<label class="cg-topics"><span class="sr-only">' + esc(catalog.ui.browse_topics) + '</span><select id="cgTopics" aria-label="' + esc(catalog.ui.browse_topics) + '"></select></label>' +
            '</div>' +
            '<nav id="cgTabs" class="cg-tabs" aria-label="Guide categories" hidden></nav>' +
            '<div class="cg-title-row"><h1 id="cgTitle" tabindex="-1"></h1><span class="cg-counter" id="cgCounter"></span></div>' +
            '<p id="cgSubtitle" class="cg-subtitle" hidden></p>' +
            '<div id="cgEntries" class="cg-entries"></div>' +
            '<aside class="cg-tip" id="cgTip"></aside>' +
            '<footer class="cg-footer">' +
               '<div class="cg-footer-left" id="cgFooterLeft"></div>' +
               '<div class="cg-dots" id="cgDots" hidden></div>' +
               '<div class="cg-footer-right" id="cgFooterRight"></div>' +
            '</footer>' +
         '</div>';
      document.body.appendChild(dialog);

      ['cgClose', 'cgHeaderDesc', 'cgSearch', 'cgSearchResults', 'cgNav', 'cgCrumb', 'cgTopics', 'cgTabs', 'cgTitle', 'cgCounter', 'cgSubtitle', 'cgEntries', 'cgTip', 'cgFooterLeft', 'cgDots', 'cgFooterRight']
         .forEach(function (id) { els[id] = $(id); });

      // Browse topics <select>: grouped by collection
      els.cgTopics.innerHTML = '<option value="">' + esc(catalog.ui.browse_topics) + '</option>' + catalog.collections.map(function (c) {
         return '<optgroup label="' + esc(c.title) + '">' + catalog.guides.filter(function (g) { return g.collection_id === c.id; }).map(function (g) {
            return '<option value="' + g.id + '">' + g.order + '. ' + esc(g.title) + '</option>';
         }).join('') + '</optgroup>';
      }).join('');

      els.cgClose.addEventListener('click', close);
      els.cgTopics.addEventListener('change', function (e) { if (e.target.value) show(e.target.value, true); });
      els.cgSearch.addEventListener('input', onSearch);
      dialog.addEventListener('click', function (e) {
         if (e.target === dialog) { close(); return; }              // backdrop click
         var g = e.target.closest('[data-guide]');
         if (g) { show(g.getAttribute('data-guide'), true); return; }
         if (e.target.closest('[data-explorer]')) { goToExplorer(); return; }
         if (e.target.closest('[data-close]')) { close(); return; }
      });
      dialog.addEventListener('cancel', function () { syncHash(null); });
      dialog.addEventListener('close', function () { document.body.classList.remove('cg-open'); syncHash(null); });
      dialog.addEventListener('keydown', function (e) {
         if (e.key === 'Escape') { e.preventDefault(); close(); }
      });
   }

   // ---------------------------------------------------------------------
   // Render one guide
   // ---------------------------------------------------------------------
   function gridClass(g) {
      if (g.id === 'understand-your-color') return 'cg-rows';
      if (g.id === 'build-a-color-harmony') return 'cg-harmony';
      if (g.id === 'see-color-in-context') return 'cg-context';
      return '';
   }
   function entryHTML(e, g, i) {
      var action = g.entry_action_labels && g.entry_action_labels[i];
      return '<article class="cg-entry" id="' + esc(e.id) + '">' +
         assetImg(e.asset_id, 'cg-ill') +
         '<div class="cg-entry-text"><h2>' + esc(e.title) + '</h2><p>' + esc(e.description) + '</p>' +
         (e.additional_description ? '<p class="cg-additional">' + esc(e.additional_description) + '</p>' : '') +
         (e.image_labels ? '<div class="cg-image-labels">' + e.image_labels.map(function (s) { return '<span>' + esc(s) + '</span>'; }).join('') + '</div>' : '') +
         '</div>' +
         (action ? '<button type="button" class="cg-entry-action" data-explorer>' + esc(action) + ' ' + CHEV_R + '</button>' : '') +
         '</article>';
   }
   function show(id, focus) {
      var g = guideById(id) || catalog.guides[0];   // unknown id (typo in a link) → first guide
      build();
      current = g;
      var isOriginal = g.collection_id === 'original';
      var col = collectionById(g.collection_id);
      var colIdx = col ? col.guide_ids.indexOf(g.id) : -1;

      dialog.className = 'cg-dialog cg-theme-' + (g.theme || 'light');
      els.cgTitle.textContent = g.title;
      els.cgSubtitle.textContent = g.description || '';
      els.cgSubtitle.hidden = !g.description;
      els.cgHeaderDesc.textContent = g.header_description || '';
      els.cgHeaderDesc.hidden = !g.header_description;
      els.cgCounter.textContent = (!isOriginal && col) ? (colIdx + 1) + ' / ' + col.guide_ids.length : '';

      // breadcrumb + topics
      els.cgCrumb.innerHTML = '<span>' + esc(g.collection_id === 'practical' ? catalog.ui.practical_guides : catalog.ui.guide_collection) + '</span><span aria-hidden="true">/</span><span>' + esc(isOriginal ? categoryTitle(g.category_id) : g.title) + '</span>';
      els.cgTopics.value = g.id;

      // tabs (the four original guides only — mirrors the PNG designs)
      els.cgTabs.hidden = !isOriginal;
      if (isOriginal) {
         els.cgTabs.innerHTML = collectionById('original').guide_ids.map(function (gid, i) {
            var t = guideById(gid);
            return '<button type="button" data-guide="' + gid + '" class="' + (gid === g.id ? 'active' : '') + '" aria-current="' + (gid === g.id ? 'page' : 'false') + '">' + esc(catalog.ui.original_tabs[i] || categoryTitle(t.category_id)) + '</button>';
         }).join('');
      }

      // entries
      els.cgEntries.className = 'cg-entries ' + gridClass(g);
      els.cgEntries.innerHTML = g.entries.map(function (e, i) { return entryHTML(e, g, i); }).join('');

      // tip (the "Perception" design puts its CTA inside the tip box)
      var tipCta = !!g.tip_description;
      els.cgTip.innerHTML = (g.theme === 'dark' ? ICON_INFO : ICON_BULB) +
         '<div><p>' + tipText(g.tip) + '</p>' + (g.tip_description ? '<p>' + esc(g.tip_description) + '</p>' : '') + '</div>' +
         (tipCta ? '<button type="button" class="cg-btn cg-primary" data-explorer>' + esc(g.footer.primary_label) + '</button>' : '');

      // footer
      var left = '', right = '';
      els.cgDots.hidden = true;
      if (isOriginal) {
         left = '<button type="button" class="cg-btn" data-close>' + esc(g.footer.previous_label || catalog.ui.back_to_exploring) + '</button>';
         if (!tipCta) right = '<button type="button" class="cg-btn cg-primary" data-explorer>' + esc(g.footer.primary_label) + ' ' + ARROW_R + '</button>';
      } else {
         left = '<button type="button" class="cg-btn" ' + (g.previous_guide_id ? 'data-guide="' + g.previous_guide_id + '"' : 'disabled') + '>' + ARROW_L + ' ' + esc(catalog.ui.previous) + '</button>';
         right = g.next_guide_id
            ? '<button type="button" class="cg-btn cg-primary" data-guide="' + g.next_guide_id + '">' + esc(catalog.ui.next) + ' ' + ARROW_R + '</button>'
            : '<button type="button" class="cg-btn cg-primary" data-close>' + esc(catalog.ui.back_to_exploring) + '</button>';
         if (col) {
            els.cgDots.hidden = false;
            els.cgDots.innerHTML = col.guide_ids.map(function (gid, i) {
               var t = guideById(gid);
               return '<button type="button" data-guide="' + gid + '" class="' + (gid === g.id ? 'active' : '') + '" title="' + esc(t.title) + '" aria-label="' + esc(catalog.ui.guide) + ' ' + (i + 1) + ' ' + esc(catalog.ui.of) + ' ' + col.guide_ids.length + ': ' + esc(t.title) + '"></button>';
            }).join('');
         }
      }
      els.cgFooterLeft.innerHTML = left;
      els.cgFooterRight.innerHTML = right;

      els.cgSearch.value = '';
      els.cgSearchResults.hidden = true;

      if (!dialog.open) { try { dialog.showModal(); } catch (_) { dialog.setAttribute('open', ''); } }
      document.body.classList.add('cg-open');
      dialog.scrollTop = 0;
      if (focus) els.cgTitle.focus({ preventScroll: true });
      syncHash(g.id);
   }
   function tipText(tip) {
      // "Try it: ..." / "Ask: ..." → bold lead-in, like the PNG designs.
      var m = /^([A-Z][^:]{1,24}):\s+(.*)$/.exec(tip || '');
      return m ? '<b>' + esc(m[1]) + ':</b> ' + esc(m[2]) : esc(tip);
   }

   function onSearch(e) {
      var q = e.target.value.trim();
      var box = els.cgSearchResults;
      box.hidden = !q;
      if (!q) return;
      var ql = q.toLowerCase();
      var found = catalog.guides.filter(function (g) { return searchable(g).indexOf(ql) > -1; });
      box.innerHTML = found.length ? found.map(function (g) {
         var hits = g.entries.filter(function (en) { return (en.title + ' ' + en.description + ' ' + (en.additional_description || '')).toLowerCase().indexOf(ql) > -1; });
         var sub = hits.length ? hits.map(function (en) { return highlight(en.title, q); }).join(' &middot; ') : esc(categoryTitle(g.category_id));
         return '<button type="button" data-guide="' + g.id + '"><b>' + highlight(g.title, q) + '</b><small>' + sub + '</small></button>';
      }).join('') : '<p>' + esc(catalog.ui.no_results) + '</p>';
   }

   function open(id) {
      build();
      show(id || (location.hash.indexOf('#guide=') === 0 ? decodeURIComponent(location.hash.slice(7)) : '') || catalog.guides[0].id, true);
   }
   function close() {
      if (dialog && dialog.open) dialog.close();
      document.body.classList.remove('cg-open');
      syncHash(null);
   }
   function goToExplorer() {
      // The four original guides' CTAs ("Explore dimensions", "Try complementary", ...)
      // and the "See hue on explorer" links open GeoMagic's Colors page.
      close();
      if (page !== EXPLORER_PAGE && page !== EXPLORER_PAGE.replace('.html', '')) location.href = EXPLORER_PAGE;
   }
   // Deep links only on the library page so we never clobber other pages' hashes.
   function syncHash(id) {
      if (!isLibraryPage) return;
      var want = id ? '#guide=' + encodeURIComponent(id) : '';
      if (location.hash === want) return;
      if (want) history.replaceState(null, '', location.pathname + location.search + want);
      else history.replaceState(null, '', location.pathname + location.search);
   }

   // ---------------------------------------------------------------------
   // Library page
   // ---------------------------------------------------------------------
   function renderLibrary(root) {
      if (!root) return;
      var filter = { collection: null, category: null, query: '' };
      root.innerHTML =
         '<div class="cgl-layout">' +
            '<aside class="cgl-side">' +
               '<p class="cgl-eyebrow">' + esc(catalog.ui.guide_collection) + '</p><nav id="cglCollections" aria-label="Guide collections"></nav>' +
               '<p class="cgl-eyebrow">Topics</p><nav id="cglCategories" aria-label="Guide topics"></nav>' +
               '<div class="cgl-note"><b id="cglNoteCount"></b>Color language, painting techniques and practical decisions.</div>' +
            '</aside>' +
            '<div>' +
               '<div class="cgl-chips" id="cglChips"></div>' +
               '<div class="cgl-filterbar">' +
                  '<label class="cgl-search">' + ICON_SEARCH + '<input id="cglSearch" type="search" placeholder="' + esc(catalog.ui.search_placeholder) + '" aria-label="Search guides and definitions" autocomplete="off"></label>' +
                  '<button type="button" id="cglReset" class="theme-btn-regular">' + esc(catalog.ui.reset) + '</button>' +
               '</div>' +
               '<div class="cgl-results"><h2 id="cglTitle"></h2><span id="cglCount" aria-live="polite"></span></div>' +
               '<div id="cglGrid" class="cgl-grid"></div>' +
            '</div>' +
         '</div>';

      var total = 0; catalog.guides.forEach(function (g) { total += g.entries.length; });
      $('cglNoteCount').textContent = catalog.guides.length + ' guides · ' + total + ' ' + catalog.ui.entry_count;

      function navHTML() {
         var all = '<button type="button" data-collection="" class="' + (!filter.collection && !filter.category ? 'active' : '') + '">' + esc(catalog.ui.all_guides) + ' <span>' + catalog.guides.length + '</span></button>';
         var cols = catalog.collections.map(function (c) { return '<button type="button" data-collection="' + c.id + '" class="' + (filter.collection === c.id ? 'active' : '') + '">' + esc(c.title) + ' <span>' + c.guide_ids.length + '</span></button>'; }).join('');
         var cats = catalog.categories.map(function (c) { return '<button type="button" data-category="' + c.id + '" class="' + (filter.category === c.id ? 'active' : '') + '">' + esc(c.title) + ' <span>' + c.guide_ids.length + '</span></button>'; }).join('');
         return { all: all, cols: cols, cats: cats };
      }
      function render() {
         var n = navHTML();
         $('cglCollections').innerHTML = n.all + n.cols;
         $('cglCategories').innerHTML = n.cats;
         $('cglChips').innerHTML = n.all + n.cols + n.cats;
         var ql = filter.query.toLowerCase();
         var list = catalog.guides.filter(function (g) {
            return (!filter.collection || g.collection_id === filter.collection) &&
                   (!filter.category || g.category_id === filter.category) &&
                   (!ql || searchable(g).indexOf(ql) > -1);
         });
         $('cglTitle').textContent = filter.category ? categoryTitle(filter.category) : (filter.collection ? collectionById(filter.collection).title : (ql ? catalog.ui.search_results : catalog.ui.all_guides));
         $('cglCount').textContent = list.length + ' guides';
         $('cglGrid').innerHTML = list.length ? list.map(function (g) {
            return '<button type="button" class="cgl-card" data-guide="' + g.id + '">' +
               '<div class="cgl-preview">' + g.entries.slice(0, 2).map(function (e) { return assetImg(e.asset_id); }).join('') + '</div>' +
               '<div class="cgl-card-body"><span class="cgl-card-cat">' + esc(categoryTitle(g.category_id)) + '</span><h3>' + esc(g.title) + '</h3>' +
               '<p>' + esc(g.entries.map(function (e) { return e.title; }).join(' · ')) + '</p>' +
               '<small>' + g.entries.length + ' ' + esc(catalog.ui.entry_count) + '</small></div></button>';
         }).join('') : '<p class="cgl-empty">' + esc(catalog.ui.no_results) + '</p>';
      }
      root.addEventListener('click', function (e) {
         var card = e.target.closest('[data-guide]');
         if (card) { open(card.getAttribute('data-guide')); return; }
         var col = e.target.closest('[data-collection]');
         if (col) { filter.collection = col.getAttribute('data-collection') || null; filter.category = null; render(); return; }
         var cat = e.target.closest('[data-category]');
         if (cat) { filter.category = cat.getAttribute('data-category'); filter.collection = null; render(); return; }
      });
      $('cglSearch').addEventListener('input', function (e) { filter.query = e.target.value.trim(); render(); });
      $('cglReset').addEventListener('click', function () { filter = { collection: null, category: null, query: '' }; $('cglSearch').value = ''; render(); });
      render();

      // #guide=<id> deep link
      function fromHash() {
         if (location.hash.indexOf('#guide=') !== 0) return;
         var id = decodeURIComponent(location.hash.slice(7));
         if (guideById(id)) show(id, true);
      }
      window.addEventListener('hashchange', fromHash);
      fromHash();
   }

   // ---------------------------------------------------------------------
   // Boot: make sure the catalog is present (load it if the page didn't)
   // ---------------------------------------------------------------------
   var ready = null;
   function loadCatalog() {
      if (ready) return ready;
      ready = new Promise(function (resolve, reject) {
         if (window.GM_COLOR_GUIDE_CATALOG) { catalog = window.GM_COLOR_GUIDE_CATALOG; resolve(catalog); return; }
         var s = document.createElement('script');
         s.src = 'assets/js/color-guide-data.js';
         s.onload = function () { catalog = window.GM_COLOR_GUIDE_CATALOG; catalog ? resolve(catalog) : reject(new Error('Color guide catalog missing')); };
         s.onerror = function () { reject(new Error('Could not load the color guide catalog')); };
         document.head.appendChild(s);
      });
      return ready;
   }

   var api = {
      open: function (id) { return loadCatalog().then(function () { open(id); }); },
      close: close,
      renderLibrary: function (el) { return loadCatalog().then(function () { renderLibrary(el); }); },
      get catalog() { return catalog; }
   };
   window.GM = window.GM || {};
   window.GM.colorGuide = api;

   function boot() {
      var lib = document.getElementById('cgLibrary');
      if (lib) api.renderLibrary(lib);
      // ?guide=<id> opens the modal on any page that loaded this script
      var q = new URLSearchParams(location.search).get('guide');
      if (q) api.open(q);
      // Event delegation for any launcher: <button data-open-color-guide="[guide-id]">
      document.addEventListener('click', function (e) {
         var b = e.target.closest('[data-open-color-guide]');
         if (!b) return;
         e.preventDefault();
         api.open(b.getAttribute('data-open-color-guide') || undefined);
      });
   }
   if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
   else boot();
})();
