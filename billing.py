"""
Billing & Plans — Stripe (TEST MODE) + per-plan entitlements.

Plans (single source of truth: PLAN_CONFIG below)
  • payg       — $0.99 one-time  → +5 image credits (no recurring features)
  • apprentice — $4.99 / month   → 5 images/mo,  basic colors,  few rendering logics
  • atelier    — $9.99 / month   → 15 images/mo, one paint palette, more rendering logics
  • maestro    — $19.99 / month  → unlimited images, all palettes, all rendering logics
  • free       — default for signed-in users with no subscription (small trial)

Database: the SAME Supabase Postgres / auth_users used by auth_db.py and by the
multi_model_cmprxn_role project. Stripe runs in test mode (sk_test_/pk_test_).
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel

import stripe

from auth_db import get_db, RealDictCursor, decode_jwt_token

# =============================================================================
# Configuration
# =============================================================================
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
stripe.api_key = STRIPE_SECRET_KEY

print(f"[Stripe] Secret key loaded: {'YES (' + STRIPE_SECRET_KEY[:10] + '...)' if STRIPE_SECRET_KEY else 'NO - NOT SET'}")
print(f"[Stripe] Publishable key loaded: {'YES' if STRIPE_PUBLISHABLE_KEY else 'NO - NOT SET'}")

UNLIMITED = -1

# Max PAYG packs per single checkout (safety bound against fat-finger amounts).
# 20000 packs × 5 = 100,000 images per purchase.
MAX_PAYG_PACKS = 20000

# All rendering-logic icons (data-logic values used by dashboard-geometrize.js).
ALL_RENDERING_LOGICS = [
    "original", "exterior_to_center", "center_to_exterior", "top_to_bottom",
    "bottom_to_top", "color_sequence", "light_to_dark", "dark_to_light",
    "frequency_by_color", "frequency_by_color_reverse", "custom_sequence",
    "selective_resolution",
]
# "A few" rendering logics for the lower tiers.
FEW_RENDERING_LOGICS = ["original", "top_to_bottom", "light_to_dark", "dark_to_light"]
MORE_RENDERING_LOGICS = [
    "original", "exterior_to_center", "center_to_exterior", "top_to_bottom",
    "bottom_to_top", "light_to_dark", "dark_to_light", "color_sequence",
]

# Palette access mode:
#   "basic"  → only the basic "Common Color Names & Values" preset (no brand dropdowns)
#   "single" → basic + ONE brand palette
#   "all"    → every brand palette in the dropdown
# The frontend resolves these names against window.PALETTE_PRESETS.

# =============================================================================
# PLAN CONFIG — edit here to change pricing / quotas / entitlements
# =============================================================================
PLAN_CONFIG = {
    "free": {
        "name": "Free",
        "price": 0.00,
        "billing_type": "free",          # no checkout
        "images_per_month": 3,           # small trial so the app isn't dead on arrival
        "rendering_logics": FEW_RENDERING_LOGICS,
        "palettes": "basic",
        "description": "Try GeoMagic — a few images each month.",
        "order": 0,
    },
    "payg": {
        "name": "Pay as you go",
        "price": 0.99,
        "billing_type": "one_time",      # one-time payment, adds credits
        "image_credits": 5,              # +5 images per $0.99 pack
        "images_per_month": 0,           # not a subscription; entitlements stay at current tier
        "rendering_logics": FEW_RENDERING_LOGICS,
        "palettes": "basic",
        "description": "5 images for $0.99 — buy as many as you like, credits stack on your plan.",
        "order": 1,
    },
    "apprentice": {
        "name": "Apprentice",
        "price": 4.99,
        "billing_type": "subscription",
        "images_per_month": 5,
        "rendering_logics": FEW_RENDERING_LOGICS,
        "palettes": "basic",             # basic colors only, no brand-specific dropdowns
        "description": "5 images per month with the essentials.",
        "order": 2,
    },
    "atelier": {
        "name": "Atelier",
        "price": 9.99,
        "billing_type": "subscription",
        "images_per_month": 15,
        "rendering_logics": MORE_RENDERING_LOGICS,
        "palettes": "single",            # one paint palette
        "description": "15 images per month, one paint palette, more rendering styles.",
        "order": 3,
    },
    "maestro": {
        "name": "Maestro",
        "price": 19.99,
        "billing_type": "subscription",
        "images_per_month": UNLIMITED,
        "rendering_logics": ALL_RENDERING_LOGICS,
        "palettes": "all",               # multiple paint palettes
        "description": "Unlimited images, every paint palette, all rendering styles.",
        "order": 4,
    },
}

DEFAULT_PLAN = "free"


def _plan_public(plan_id: str) -> dict:
    """Public, frontend-safe view of a plan + its entitlements."""
    p = PLAN_CONFIG[plan_id]
    return {
        "plan_id": plan_id,
        "name": p["name"],
        "price": p["price"],
        "billing_type": p["billing_type"],
        "description": p.get("description", ""),
        "order": p.get("order", 99),
        "entitlements": {
            "images_per_month": p.get("images_per_month", 0),
            "image_credits": p.get("image_credits", 0),
            "rendering_logics": p.get("rendering_logics", []),
            "palettes": p.get("palettes", "basic"),
            "unlimited_images": p.get("images_per_month") == UNLIMITED,
        },
    }


# =============================================================================
# Database init
# =============================================================================
def init_billing_tables():
    """Create billing tables and seed plans into Supabase."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subscription_plans (
                    id SERIAL PRIMARY KEY,
                    plan_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    billing_cycle TEXT DEFAULT 'monthly',
                    description TEXT,
                    stripe_price_id TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_subscriptions (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT UNIQUE NOT NULL,
                    plan_id TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    current_period_start TIMESTAMPTZ DEFAULT NOW(),
                    current_period_end TIMESTAMPTZ,
                    stripe_subscription_id TEXT,
                    stripe_customer_id TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Monthly image usage counter (one row per user per YYYY-MM).
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_usage (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    period TEXT NOT NULL,           -- 'YYYY-MM'
                    used INTEGER DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, period)
                )
            """)
            # Pay-as-you-go credit balance.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_credits (
                    user_id TEXT PRIMARY KEY,
                    credits INTEGER DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Idempotency ledger: every Stripe checkout session is fulfilled ONCE,
            # even if verify-session (frontend) and the webhook both fire for it.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fulfilled_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    plan_id TEXT,
                    fulfilled_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            conn.commit()

            # Seed/refresh plan rows from PLAN_CONFIG.
            for plan_id, p in PLAN_CONFIG.items():
                cursor.execute("""
                    INSERT INTO subscription_plans (plan_id, name, price, description, is_active)
                    VALUES (%s, %s, %s, %s, TRUE)
                    ON CONFLICT (plan_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        price = EXCLUDED.price,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                """, (plan_id, p["name"], p["price"], p.get("description", "")))
            conn.commit()
            print("[OK] Billing tables ready")
    except Exception as e:
        print(f"[WARN] Billing tables init error: {e}")


# =============================================================================
# Helpers
# =============================================================================
def _current_period() -> str:
    now = datetime.utcnow()
    return f"{now.year:04d}-{now.month:02d}"


def _stripe_meta(obj) -> dict:
    """Extract a Stripe object's `metadata` as a plain dict.

    The Stripe SDK (v15) StripeObject does NOT support `.get()` and is not
    `dict()`-convertible, so `session.get("metadata")` / `dict(session.metadata)`
    raise. Use attribute access + `.to_dict()` instead.
    """
    md = getattr(obj, "metadata", None)
    if md is None:
        return {}
    if hasattr(md, "to_dict"):
        try:
            return md.to_dict()
        except Exception:
            pass
    try:
        return dict(md)
    except Exception:
        return {}


def _user_from_request(request: Request) -> dict:
    """Decode the Bearer JWT and return its payload, or raise 401."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return decode_jwt_token(auth_header[7:])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_user_plan_id(user_id: str) -> str:
    """Return the user's active subscription plan_id, or DEFAULT_PLAN."""
    try:
        with get_db() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT plan_id, status, current_period_end
                FROM user_subscriptions WHERE user_id = %s
            """, (user_id,))
            row = cursor.fetchone()
            if not row:
                return DEFAULT_PLAN
            if row["status"] != "active":
                return DEFAULT_PLAN
            # Subscriptions only — payg is never stored as the active plan.
            plan_id = row["plan_id"]
            if plan_id not in PLAN_CONFIG or PLAN_CONFIG[plan_id]["billing_type"] != "subscription":
                return DEFAULT_PLAN
            end = row.get("current_period_end")
            if end and end < datetime.now(end.tzinfo):
                return DEFAULT_PLAN
            return plan_id
    except Exception:
        return DEFAULT_PLAN


def get_credits(user_id: str) -> int:
    try:
        with get_db() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT credits FROM image_credits WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
            return int(row["credits"]) if row else 0
    except Exception:
        return 0


def add_credits(user_id: str, amount: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO image_credits (user_id, credits, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                credits = image_credits.credits + EXCLUDED.credits,
                updated_at = NOW()
        """, (user_id, amount))
        conn.commit()


def get_usage(user_id: str) -> int:
    period = _current_period()
    try:
        with get_db() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT used FROM image_usage WHERE user_id = %s AND period = %s",
                (user_id, period),
            )
            row = cursor.fetchone()
            return int(row["used"]) if row else 0
    except Exception:
        return 0


def get_entitlements(user_id: str) -> dict:
    """Resolved entitlements + live usage for a user."""
    plan_id = get_user_plan_id(user_id)
    p = PLAN_CONFIG[plan_id]
    limit = p.get("images_per_month", 0)
    used = get_usage(user_id)
    credits = get_credits(user_id)
    unlimited = (limit == UNLIMITED)
    if unlimited:
        remaining = UNLIMITED
    else:
        remaining = max(0, limit - used) + credits
    return {
        "plan_id": plan_id,
        "plan_name": p["name"],
        "images_per_month": limit,
        "images_used": used,
        "images_remaining": remaining,
        "credits": credits,
        "unlimited_images": unlimited,
        "rendering_logics": p.get("rendering_logics", []),
        "palettes": p.get("palettes", "basic"),
    }


# =============================================================================
# Stripe sync helpers
# =============================================================================
def _get_or_create_product(plan_id: str, name: str, description: str) -> str:
    products = stripe.Product.search(query=f"metadata['plan_id']:'{plan_id}'", limit=1)
    if products.data:
        return products.data[0].id
    product = stripe.Product.create(
        name=name, description=description or f"{name} plan", metadata={"plan_id": plan_id}
    )
    return product.id


def _get_or_create_price(product_id: str, amount_cents: int, plan_id: str, recurring: bool) -> str:
    prices = stripe.Price.list(product=product_id, active=True, limit=20)
    for price in prices.data:
        if price.unit_amount != amount_cents:
            continue
        if recurring and price.recurring and price.recurring.interval == "month":
            return price.id
        if not recurring and not price.recurring:
            return price.id
    kwargs = dict(product=product_id, unit_amount=amount_cents, currency="usd",
                  metadata={"plan_id": plan_id})
    if recurring:
        kwargs["recurring"] = {"interval": "month"}
    return stripe.Price.create(**kwargs).id


def sync_plan_to_stripe(plan_id: str) -> str:
    p = PLAN_CONFIG[plan_id]
    product_id = _get_or_create_product(plan_id, p["name"], p.get("description", ""))
    amount_cents = int(round(p["price"] * 100))
    recurring = p["billing_type"] == "subscription"
    price_id = _get_or_create_price(product_id, amount_cents, plan_id, recurring)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE subscription_plans SET stripe_price_id = %s, updated_at = NOW() WHERE plan_id = %s",
            (price_id, plan_id),
        )
        conn.commit()
    return price_id


def get_or_create_customer(user_id: str) -> str:
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT stripe_customer_id FROM user_subscriptions WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and row.get("stripe_customer_id"):
            return row["stripe_customer_id"]

    email = name = None
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT email, name FROM auth_users WHERE id = %s", (user_id,))
        u = cursor.fetchone()
        if u:
            email, name = u.get("email"), u.get("name")

    customer = stripe.Customer.create(email=email, name=name, metadata={"user_id": user_id})
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_subscriptions (user_id, plan_id, status, stripe_customer_id)
            VALUES (%s, %s, 'pending', %s)
            ON CONFLICT (user_id) DO UPDATE SET
                stripe_customer_id = EXCLUDED.stripe_customer_id, updated_at = NOW()
        """, (user_id, DEFAULT_PLAN, customer.id))
        conn.commit()
    return customer.id


def activate_subscription(user_id: str, plan_id: str, sub_id: Optional[str],
                          customer_id: Optional[str], period_end: Optional[datetime] = None):
    now = datetime.utcnow()
    if not period_end:
        period_end = now + timedelta(days=30)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_subscriptions (user_id, plan_id, status, current_period_start,
                                            current_period_end, stripe_subscription_id, stripe_customer_id)
            VALUES (%s, %s, 'active', %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                plan_id = EXCLUDED.plan_id,
                status = 'active',
                current_period_start = EXCLUDED.current_period_start,
                current_period_end = EXCLUDED.current_period_end,
                stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                stripe_customer_id = COALESCE(EXCLUDED.stripe_customer_id, user_subscriptions.stripe_customer_id),
                updated_at = NOW()
        """, (user_id, plan_id, now, period_end, sub_id, customer_id))
        # Reset usage so upgraded users get their full new allocation.
        period = _current_period()
        cursor.execute(
            "DELETE FROM image_usage WHERE user_id = %s AND period = %s",
            (user_id, period),
        )
        conn.commit()


# =============================================================================
# Router
# =============================================================================
router = APIRouter(prefix="/api/billing", tags=["Billing"])


class CheckoutRequest(BaseModel):
    plan_id: str
    quantity: Optional[int] = 1      # number of one-time packs to buy (ignored for subscriptions)
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


@router.get("/config")
async def billing_config():
    return {"publishable_key": STRIPE_PUBLISHABLE_KEY, "test_mode": STRIPE_SECRET_KEY.startswith("sk_test_")}


@router.get("/plans")
async def list_plans():
    plans = [_plan_public(pid) for pid in PLAN_CONFIG]
    plans.sort(key=lambda x: x["order"])
    return {"plans": plans}


@router.get("/me")
async def billing_me(request: Request):
    payload = _user_from_request(request)
    return get_entitlements(payload["sub"])


@router.get("/entitlements")
async def entitlements(request: Request):
    payload = _user_from_request(request)
    return get_entitlements(payload["sub"])


@router.post("/consume-image")
async def consume_image(request: Request):
    """Atomically check the user's image quota and consume one unit.

    Order: monthly allowance first, then pay-as-you-go credits. Unlimited plans
    always pass. Returns {allowed, remaining, reason}.
    """
    payload = _user_from_request(request)
    user_id = payload["sub"]
    plan_id = get_user_plan_id(user_id)
    limit = PLAN_CONFIG[plan_id].get("images_per_month", 0)

    if limit == UNLIMITED:
        return {"allowed": True, "remaining": UNLIMITED, "unlimited": True}

    period = _current_period()
    try:
        with get_db() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                INSERT INTO image_usage (user_id, period, used)
                VALUES (%s, %s, 0)
                ON CONFLICT (user_id, period) DO NOTHING
            """, (user_id, period))
            cursor.execute(
                "SELECT used FROM image_usage WHERE user_id = %s AND period = %s FOR UPDATE",
                (user_id, period),
            )
            used = int((cursor.fetchone() or {"used": 0})["used"])

            if used < limit:
                cursor.execute("""
                    UPDATE image_usage SET used = used + 1, updated_at = NOW()
                    WHERE user_id = %s AND period = %s
                """, (user_id, period))
                conn.commit()
                monthly_remaining = limit - (used + 1)
                return {
                    "allowed": True,
                    "remaining": monthly_remaining + get_credits(user_id),
                    "source": "plan",
                }

            # Monthly allowance exhausted → try a PAYG credit.
            cursor.execute("SELECT credits FROM image_credits WHERE user_id = %s FOR UPDATE", (user_id,))
            crow = cursor.fetchone()
            credits = int(crow["credits"]) if crow else 0
            if credits > 0:
                cursor.execute("""
                    UPDATE image_credits SET credits = credits - 1, updated_at = NOW()
                    WHERE user_id = %s
                """, (user_id,))
                conn.commit()
                return {"allowed": True, "remaining": credits - 1, "source": "credit"}

            conn.commit()
            return {
                "allowed": False,
                "remaining": 0,
                "reason": "You've used all your images for this month. Upgrade your plan or buy more.",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest, request: Request):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe is not configured")
    payload = _user_from_request(request)
    user_id = payload["sub"]

    if req.plan_id not in PLAN_CONFIG:
        raise HTTPException(status_code=404, detail="Plan not found")
    plan = PLAN_CONFIG[req.plan_id]
    if plan["billing_type"] == "free":
        raise HTTPException(status_code=400, detail="The free plan does not require checkout")

    try:
        price_id = sync_plan_to_stripe(req.plan_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync plan to Stripe: {e}")

    customer_id = get_or_create_customer(user_id)

    base = req.success_url or "https://alchromaticdemo.up.railway.app/pricing.html"
    success_url = base + ("&" if "?" in base else "?") + "session_id={CHECKOUT_SESSION_ID}&status=success"
    # Default the cancel URL from the (absolute) success base so Stripe always
    # receives a valid URL even when the caller omits cancel_url.
    cancel_url = req.cancel_url or (base + ("&" if "?" in base else "?") + "status=cancelled")
    mode = "subscription" if plan["billing_type"] == "subscription" else "payment"
    meta = {"user_id": user_id, "plan_id": req.plan_id, "kind": plan["billing_type"]}

    # One-time (PAYG) purchases are quantity-driven: the user buys N $0.99 packs and
    # receives N × image_credits credits. Subscriptions always use quantity 1.
    qty = 1
    if plan["billing_type"] == "one_time":
        qty = int(req.quantity or 1)
        if qty < 1 or qty > MAX_PAYG_PACKS:
            raise HTTPException(
                status_code=400,
                detail=f"Quantity must be between 1 and {MAX_PAYG_PACKS} packs ({MAX_PAYG_PACKS * plan.get('image_credits', 5)} images)",
            )
        meta["credits"] = str(qty * plan.get("image_credits", 0))

    try:
        kwargs = dict(
            customer=customer_id,
            payment_method_types=["card"],
            mode=mode,
            line_items=[{"price": price_id, "quantity": qty}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata=meta,
        )
        if mode == "subscription":
            kwargs["subscription_data"] = {"metadata": meta}
        session = stripe.checkout.Session.create(**kwargs)
        return {"checkout_url": session.url, "session_id": session.id}
    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/verify-session")
async def verify_session(request: Request):
    """Fallback activation when webhooks aren't configured (test mode)."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=f"Stripe error: {e}")

    if getattr(session, "payment_status", None) not in ("paid", "no_payment_required"):
        return {"activated": False, "reason": f"Payment not completed ({getattr(session, 'payment_status', None)})"}

    meta = _stripe_meta(session)
    user_id = meta.get("user_id")
    plan_id = meta.get("plan_id")
    kind = meta.get("kind")
    if not user_id or not plan_id:
        return {"activated": False, "reason": "Missing metadata"}

    return _fulfill(user_id, plan_id, kind, session)


def _claim_session(session_id: Optional[str], user_id: str, plan_id: str) -> bool:
    """Atomically record a checkout session as fulfilled.

    Returns True if THIS call claimed it (caller should fulfill), False if it was
    already fulfilled (caller should skip). Guarantees credits/activation apply
    exactly once even when verify-session and the webhook both fire.
    """
    if not session_id:
        return True  # no id to dedupe on → let it through
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO fulfilled_sessions (session_id, user_id, plan_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, user_id, plan_id),
            )
            claimed = cursor.rowcount > 0
            conn.commit()
            return claimed
    except Exception:
        return True  # on ledger failure, don't block fulfillment


def _fulfill(user_id: str, plan_id: str, kind: str, session) -> dict:
    """Apply the purchase: subscription → activate; one_time → add credits.

    Idempotent: each Stripe session is applied at most once (see _claim_session).
    """
    session_id = getattr(session, "id", None)
    if not _claim_session(session_id, user_id, plan_id):
        # Already fulfilled — return current state without re-applying.
        return {"activated": True, "type": "duplicate", "plan_id": plan_id}

    if kind == "one_time" or PLAN_CONFIG.get(plan_id, {}).get("billing_type") == "one_time":
        # Credits purchased are carried in the session metadata (quantity × image_credits).
        # Fall back to the plan default for older sessions that predate variable quantity.
        meta = _stripe_meta(session)
        amount = int(meta.get("credits") or PLAN_CONFIG.get(plan_id, {}).get("image_credits", 0))
        add_credits(user_id, amount)
        return {"activated": True, "type": "credits", "added": amount, "plan_id": plan_id}

    sub_id = getattr(session, "subscription", None)
    customer_id = getattr(session, "customer", None)
    period_end = None
    if sub_id:
        try:
            sub = stripe.Subscription.retrieve(sub_id)
            if getattr(sub, "current_period_end", None):
                period_end = datetime.utcfromtimestamp(int(sub.current_period_end))
        except Exception:
            pass
    activate_subscription(user_id, plan_id, sub_id, customer_id, period_end)
    return {"activated": True, "type": "subscription", "plan_id": plan_id}


@router.post("/create-portal-session")
async def create_portal_session(request: Request):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe is not configured")
    payload = _user_from_request(request)
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT stripe_customer_id FROM user_subscriptions WHERE user_id = %s", (payload["sub"],))
        row = cursor.fetchone()
    if not row or not row.get("stripe_customer_id"):
        raise HTTPException(status_code=404, detail="No Stripe customer. Subscribe to a plan first.")
    try:
        session = stripe.billing_portal.Session.create(
            customer=row["stripe_customer_id"], return_url="/pricing.html"
        )
        return {"portal_url": session.url}
    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
    else:
        try:
            event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")

    try:
        if event.type == "checkout.session.completed":
            session = event.data.object
            meta = _stripe_meta(session)
            user_id, plan_id, kind = meta.get("user_id"), meta.get("plan_id"), meta.get("kind")
            if user_id and plan_id:
                _fulfill(user_id, plan_id, kind, session)
        elif event.type == "customer.subscription.deleted":
            sub = event.data.object
            customer_id = getattr(sub, "customer", None)
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE user_subscriptions SET status = 'cancelled', updated_at = NOW() WHERE stripe_customer_id = %s",
                    (customer_id,),
                )
                conn.commit()
        else:
            print(f"[Stripe Webhook] Unhandled: {event.type}")
    except Exception as e:
        print(f"[Stripe Webhook] Error: {e}")

    return {"received": True}
