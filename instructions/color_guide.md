# Color guide — educational guide pages and modals

Client deliverable (Sep 2026): the "Color Study" guide package — 20 illustrated
guides (definitions + visual examples) shown in a **Color guide** popup, plus a
searchable **content library** page. Source package: `color-study-catalog.json`
and `color-study-html-assets.zip` from the client's Dropbox folder.

## What was added

| File | Purpose |
| --- | --- |
| `Application-main/color-guide.html` | Guide library page (collections, topics, search, cards). Sidebar entry **Color Guide**. |
| `Application-main/assets/js/color-guide.js` | Builds the `<dialog>` modal, renders any guide, library grid, deep links. |
| `Application-main/assets/js/color-guide-data.js` | Generated catalog (`window.GM_COLOR_GUIDE_CATALOG`): UI strings, collections, categories, 20 guides, 113 image assets. |
| `Application-main/assets/css/color-guide.css` | Modal + library styles (`cg-` / `cgl-` prefixes), 4 guide themes, GeoMagic dark mode. |
| `Application-main/assets/color-guide/illustrations/*.png` | 113 illustration crops (the client's extracted artwork). |
| `Application-main/assets/color-guide/catalog.source.json` | The client's full catalog (kept for regeneration). |
| `Application-main/assets/js/app-shell.js` | `Color Guide` menu entry, header **Color guide** button on every page, lazy loader `GM.openColorGuide(id)`. |
| `Application-main/templates.html` | **Color guide** button in the Colors-page toolbar (the explorer the designs show behind the popup). |

## How the popup opens

* Any element with `data-open-color-guide="<guide-id>"` (empty = first guide).
  `app-shell.js` lazy-loads the CSS, catalog and script the first time.
* Programmatically: `GM.openColorGuide('harmony-recipes')`.
* Deep links: `color-guide.html#guide=<id>` or `<any page>?guide=<id>`
  (the `?guide=` form needs `color-guide.js` on that page; the library page has it).

## Behaviour (mirrors the client's PNG designs)

* **Original guides** (`understand-your-color`, `build-a-color-harmony`,
  `see-color-in-context`, `understand-paint-behavior`): tabs
  Dimensions / Relationships / Perception / Paint, each with its own theme
  (light, dark, blue, warm). Footer: *Back to exploring* + the guide's CTA
  (`footer.primary_label`, e.g. *Explore dimensions*), which opens the Colors
  page (`templates.html`). "See hue on explorer" entry links do the same.
* **Extended + practical guides**: breadcrumb, *Browse topics* dropdown,
  `n / N` counter, dot pager, *Previous* / *Next guide* (last guide → *Back to
  exploring*).
* Search box inside the modal searches every guide's terms and definitions.
* Escape / backdrop click / × close it.

## Editing text or adding a guide

1. Edit `Application-main/assets/color-guide/catalog.source.json`
   (guide `title`, `description`, `entries[].title/description`, `tip`,
   `footer.primary_label`, `previous_guide_id` / `next_guide_id`, and add the
   guide id to a `collections[].guide_ids` and `categories[].guide_ids`).
2. Drop any new illustration PNG into `assets/color-guide/illustrations/` and
   add an `assets` record (`path`, `width`, `height`, `alt`).
3. Regenerate the JS catalog (run from the repo root):

```bash
python - <<'EOF'
import json, io
src = json.load(open('Application-main/assets/color-guide/catalog.source.json', encoding='utf-8'))
out = {'schema_version': src['schema_version'], 'title': src['title'], 'ui': src['ui'],
       'collections': src['collections'], 'categories': src['categories'], 'guides': [], 'assets': {}}
for g in src['guides']:
    gg = {k: g.get(k) for k in ['id','title','description','header_description','category_id','collection_id','theme','tip','tip_description','order','previous_guide_id','next_guide_id','footer','entry_action_labels']}
    gg['entries'] = [{k: e.get(k) for k in ['id','title','description','additional_description','order','asset_id','image_labels']} for e in g['entries']]
    out['guides'].append(gg)
    for e in g['entries']:
        a = src['assets'][e['asset_id']]
        out['assets'][e['asset_id']] = {'path': a['path'].replace('assets/illustrations/', 'assets/color-guide/illustrations/'),
                                        'width': a['width'], 'height': a['height'], 'alt': a['alt']}
js = "/* GeoMagic Color guide catalog — generated from the client's color-study-catalog.json.\n   Text is the client's approved copy; edit assets/color-guide/catalog.source.json and re-run the\n   build snippet in instructions/color_guide.md rather than editing this file by hand. */\nwindow.GM_COLOR_GUIDE_CATALOG = " + json.dumps(out, ensure_ascii=False, separators=(',',':')) + ";\n"
io.open('Application-main/assets/js/color-guide-data.js','w',encoding='utf-8').write(js)
print('ok', len(out['guides']), 'guides')
EOF
```

Guide-specific card layouts live in `gridClass()` in `color-guide.js`
(`understand-your-color` = single rows, `build-a-color-harmony` = image under
text, `see-color-in-context` = image right). New guides default to the
two-column card grid.

## Not included on purpose

* The client's `assets/screens/*.png` (34 MB of full-screen mockups of a
  separate "Color Study" explorer UI) and the four "workspace variants" — they
  are reference images, not product features.
* Developer-only controls from the reference HTML (Download JSON, View original
  PNG, Asset details).
