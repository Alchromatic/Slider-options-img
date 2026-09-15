# Netlify redirect site

Drag **this folder** onto https://app.netlify.com/drop (or set it as the publish
directory for the `geomagic-app` site). It contains no app code on purpose — the
whole site is one forwarding rule to the Alchromatic app on Railway.

- `_redirects` — catch-all: every path -> `alchromaticdemo.up.railway.app/signin.html`

Do not add HTML files here. Any file present at a requested path competes with
the redirect rule (the `!` flag is what keeps the rule winning).
