# Coupon code generator (standalone page)

One self-contained page (`index.html`, no build, no other files) for creating
GeoMagic coupon codes that give free images. It can be hosted anywhere, apart
from the app: drag this folder onto https://app.netlify.com/drop, or any other
static host.

It talks to the GeoMagic backend's admin API (`/api/admin/promo-codes`, see
`promo_routes.py`) with the server's `ADMIN_TOKEN`, so:

- `ADMIN_TOKEN` must be set on the backend (Railway → Variables), or the page
  says coupon admin is switched off.
- Whoever uses the page needs that token. It is kept only in that browser, and
  only when "Remember on this computer" is ticked.

Users redeem the codes under "Have a code?" on the app's Pricing page, or by
opening the link in the CSV (`/pricing.html?code=…`).
