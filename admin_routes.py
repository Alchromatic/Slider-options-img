#!/usr/bin/env python3
"""
admin_routes.py
===============

Independent **Super-Admin / CEO dashboard** API.

Powers ``Application-main/admin-dashboard.html`` (the client's mockup from
Alchromatic-admin.docx). Gives a read-only view of every user with their total
image count, plan, lifetime revenue and payment history.

Auth: gated behind ``ADMIN_TOKEN`` (sent as the ``X-Admin-Token`` header) — the
SAME convention already used by ``profile_admin_routes`` and
``image_library_routes`` / ``library-admin.html``. This keeps the panel fully
independent of normal user login.

Data sources (all live Supabase tables — no fabricated metrics):
    auth_users          identity
    user_subscriptions  current plan + status
    image_usage         per-month image counter  → total images = SUM(used)
    image_credits       pay-as-you-go balance
    fulfilled_sessions  completed Stripe checkouts → payment history
    subscription_plans  plan price (for revenue)

Fields the mockup shows but the schema does not yet track (country, platform,
health score, MAU, churn, payment fees, promo codes) are returned as null and
rendered as "—" in the UI rather than faked. Building the client's proposed
data-model tables would let those become real later.

Endpoints (all require X-Admin-Token):
    GET /api/admin/verify
    GET /api/admin/overview
    GET /api/admin/customers
    GET /api/admin/transactions
    GET /api/admin/customer/{user_id}
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from auth_db import get_db, RealDictCursor
from billing import PLAN_CONFIG, DEFAULT_PLAN, UNLIMITED

router = APIRouter(prefix="/api/admin", tags=["14. Super Admin"])

# Per-image "value" baseline used for Implied Value Consumed (client doc: $2.99).
IMAGE_VALUE = 2.99


# =============================================================================
# Auth gate — identical pattern to image_library_routes._require_admin
# =============================================================================
def _require_admin(token: Optional[str]) -> None:
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin dashboard is disabled (ADMIN_TOKEN not configured).",
        )
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token.")


# =============================================================================
# Helpers
# =============================================================================
def _current_period() -> str:
    now = datetime.utcnow()
    return f"{now.year:04d}-{now.month:02d}"


def _plan_price(plan_id: Optional[str]) -> float:
    return float(PLAN_CONFIG.get(plan_id or "", {}).get("price", 0.0) or 0.0)


def _plan_name(plan_id: Optional[str]) -> str:
    return PLAN_CONFIG.get(plan_id or "", {}).get("name", plan_id or "—")


def _plan_billing_type(plan_id: Optional[str]) -> str:
    return PLAN_CONFIG.get(plan_id or "", {}).get("billing_type", "")


def _resolve_effective_plan(sub_plan: Optional[str], sub_status: Optional[str],
                            period_end) -> str:
    """Mirror billing.get_user_plan_id: only an active, unexpired *subscription*
    counts as the user's plan — everything else falls back to the free tier."""
    if not sub_plan or sub_status != "active":
        return DEFAULT_PLAN
    if _plan_billing_type(sub_plan) != "subscription":
        return DEFAULT_PLAN
    if period_end is not None:
        try:
            if period_end < datetime.now(period_end.tzinfo):
                return DEFAULT_PLAN
        except Exception:
            pass
    return sub_plan


def _image_limit(plan_id: str):
    """Monthly image allowance for a plan; UNLIMITED (-1) for maestro."""
    return PLAN_CONFIG.get(plan_id, {}).get("images_per_month", 0)


# Subscription states worth surfacing on a free-tier row (churn / billing signal).
_CHURN_STATUSES = {"past_due", "cancelled", "canceled", "paused", "trialing"}


def _display_status(eff_plan: str, sub_status: Optional[str]) -> str:
    """Status shown in the table, consistent with the effective plan.

    A live paid subscription → 'active'. A free-tier user whose subscription is
    past_due/cancelled/paused keeps that (informative) status; everyone else
    (no sub, expired, pending) shows 'free'.
    """
    if eff_plan != DEFAULT_PLAN:
        return "active"
    if sub_status and sub_status.lower() in _CHURN_STATUSES:
        return sub_status
    return "free"


def _fmt_dt(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _last_months(n: int = 12) -> list:
    """['YYYY-MM', …] for the last n months ending with the current month."""
    now = datetime.utcnow()
    y, m = now.year, now.month
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return list(reversed(out))


def _last_days(n: int = 30) -> list:
    """['YYYY-MM-DD', …] for the last n days ending today (UTC)."""
    today = datetime.utcnow().date()
    return [(today - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


# =============================================================================
# GET /api/admin/verify — cheap token check used by the dashboard login gate
# =============================================================================
@router.get("/verify")
def admin_verify(x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return {"ok": True}


# =============================================================================
# GET /api/admin/overview — executive KPI cards
# =============================================================================
@router.get("/overview")
def admin_overview(x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    period = _current_period()
    try:
        with get_db() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("SELECT COUNT(*) AS n FROM auth_users")
            total_users = int(cur.fetchone()["n"])

            cur.execute(
                "SELECT COUNT(*) AS n FROM auth_users WHERE created_at >= NOW() - INTERVAL '30 days'"
            )
            new_users_30d = int(cur.fetchone()["n"])

            # Plan distribution among active subscriptions (+ effective MRR).
            cur.execute(
                """
                SELECT plan_id, status, current_period_end
                FROM user_subscriptions
                """
            )
            sub_rows = cur.fetchall()

            # Total images ever generated (sum of the monthly counters).
            cur.execute("SELECT COALESCE(SUM(used), 0) AS n FROM image_usage")
            total_images = int(cur.fetchone()["n"])

            cur.execute(
                "SELECT COALESCE(SUM(used), 0) AS n FROM image_usage WHERE period = %s",
                (period,),
            )
            images_this_cycle = int(cur.fetchone()["n"])

            cur.execute("SELECT COALESCE(SUM(credits), 0) AS n FROM image_credits")
            credits_outstanding = int(cur.fetchone()["n"])

            # Payment history → revenue. Price comes from subscription_plans.
            cur.execute(
                """
                SELECT fs.plan_id, fs.fulfilled_at, sp.price
                FROM fulfilled_sessions fs
                LEFT JOIN subscription_plans sp ON sp.plan_id = fs.plan_id
                """
            )
            txn_rows = cur.fetchall()

            cur.execute("SELECT COUNT(*) AS n FROM fulfilled_sessions")
            transactions_count = int(cur.fetchone()["n"])

        # --- derive metrics in Python ---
        active_subscribers = 0
        mrr = 0.0
        plan_distribution: dict[str, int] = {}
        for r in sub_rows:
            eff = _resolve_effective_plan(r["plan_id"], r["status"], r.get("current_period_end"))
            if eff != DEFAULT_PLAN:
                active_subscribers += 1
                mrr += _plan_price(eff)
                plan_distribution[eff] = plan_distribution.get(eff, 0) + 1

        now = datetime.utcnow()
        gross_total = 0.0
        gross_mtd = 0.0
        for r in txn_rows:
            price = float(r["price"] or 0.0)
            gross_total += price
            ts = r.get("fulfilled_at")
            if ts is not None and ts.year == now.year and ts.month == now.month:
                gross_mtd += price

        return {
            "generated_at": _fmt_dt(now),
            "period": period,
            "total_users": total_users,
            "new_users_30d": new_users_30d,
            "active_subscribers": active_subscribers,
            "mrr": round(mrr, 2),
            "gross_revenue_mtd": round(gross_mtd, 2),
            "gross_revenue_total": round(gross_total, 2),
            "total_images": total_images,
            "images_this_cycle": images_this_cycle,
            "credits_outstanding": credits_outstanding,
            "transactions_count": transactions_count,
            "plan_distribution": plan_distribution,
            # Metrics the schema cannot yet back — surfaced as null, shown as "—".
            "churn_rate": None,
            "trial_conversion": None,
            "payment_failure_rate": None,
            "mau": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GET /api/admin/customers — the subscribers table
# =============================================================================
@router.get("/customers")
def admin_customers(
    x_admin_token: Optional[str] = Header(None),
    search: str = Query("", description="Match email / name / user id"),
    plan: str = Query("all", description="Effective plan filter or 'all'"),
    status: str = Query("all", description="Subscription status filter or 'all'"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    _require_admin(x_admin_token)
    period = _current_period()
    try:
        with get_db() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # One pass: each user joined to its plan, credit balance, image totals
            # (lifetime + current cycle) and lifetime revenue / txn count.
            base = """
                FROM auth_users u
                LEFT JOIN user_subscriptions s ON s.user_id = u.id
                LEFT JOIN image_credits c ON c.user_id = u.id
                LEFT JOIN (
                    SELECT user_id, SUM(used) AS total_used, MAX(updated_at) AS last_used
                    FROM image_usage GROUP BY user_id
                ) iu ON iu.user_id = u.id
                LEFT JOIN (
                    SELECT user_id, used AS cycle_used, updated_at AS cycle_updated
                    FROM image_usage WHERE period = %(period)s
                ) cyc ON cyc.user_id = u.id
                LEFT JOIN (
                    SELECT fs.user_id,
                           COUNT(*) AS txn_count,
                           COALESCE(SUM(sp.price), 0) AS lifetime
                    FROM fulfilled_sessions fs
                    LEFT JOIN subscription_plans sp ON sp.plan_id = fs.plan_id
                    GROUP BY fs.user_id
                ) pay ON pay.user_id = u.id
            """
            params: dict = {"period": period}
            where = []
            if search.strip():
                where.append("(u.email ILIKE %(q)s OR u.name ILIKE %(q)s OR u.id ILIKE %(q)s)")
                params["q"] = f"%{search.strip()}%"
            if status and status != "all":
                if status == "free":
                    where.append("(s.status IS NULL OR s.status <> 'active')")
                else:
                    where.append("s.status = %(status)s")
                    params["status"] = status
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""

            cur.execute(f"SELECT COUNT(*) AS n {base}{where_sql}", params)
            total = int(cur.fetchone()["n"])

            params["limit"] = page_size
            params["offset"] = (page - 1) * page_size
            cur.execute(
                f"""
                SELECT u.id, u.email, u.name, u.created_at, u.status AS account_status,
                       u.is_admin,
                       s.plan_id AS sub_plan, s.status AS sub_status,
                       s.current_period_start, s.current_period_end,
                       COALESCE(c.credits, 0) AS credits,
                       COALESCE(iu.total_used, 0) AS total_images,
                       iu.last_used,
                       COALESCE(cyc.cycle_used, 0) AS cycle_used,
                       COALESCE(pay.txn_count, 0) AS txn_count,
                       COALESCE(pay.lifetime, 0) AS lifetime_revenue
                {base}{where_sql}
                ORDER BY u.created_at DESC NULLS LAST
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                params,
            )
            rows = cur.fetchall()

        customers = []
        for r in rows:
            eff_plan = _resolve_effective_plan(
                r["sub_plan"], r["sub_status"], r.get("current_period_end")
            )
            limit = _image_limit(eff_plan)
            cycle_used = int(r["cycle_used"])
            unlimited = (limit == UNLIMITED)
            if unlimited:
                used_pct = None
                granted = None
            else:
                granted = int(limit)
                used_pct = round((cycle_used / granted) * 100) if granted else 0
            disp_status = _display_status(eff_plan, r["sub_status"])
            last_active = r.get("last_used")
            customers.append({
                "id": r["id"],
                "email": r["email"],
                "name": r["name"],
                "created_at": _fmt_dt(r["created_at"]),
                "is_admin": bool(r["is_admin"]),
                "account_status": r["account_status"],
                "plan_id": eff_plan,
                "plan_name": _plan_name(eff_plan),
                "status": disp_status,
                "total_images": int(r["total_images"]),
                "credits": int(r["credits"]),
                "cycle_used": cycle_used,
                "cycle_granted": granted,
                "unlimited": unlimited,
                "used_pct": used_pct,
                "lifetime_revenue": round(float(r["lifetime_revenue"]), 2),
                "txn_count": int(r["txn_count"]),
                "last_active": _fmt_dt(last_active),
                # Not tracked in the schema (yet).
                "country": None,
                "platform": None,
                "health": None,
            })

        # Filter by effective plan in Python (it's a derived value, not a column).
        if plan and plan != "all":
            customers = [c for c in customers if c["plan_id"] == plan]

        return {
            "customers": customers,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GET /api/admin/transactions — payment history
# =============================================================================
@router.get("/transactions")
def admin_transactions(
    x_admin_token: Optional[str] = Header(None),
    search: str = Query("", description="Match user email / session id"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    _require_admin(x_admin_token)
    try:
        with get_db() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            base = """
                FROM fulfilled_sessions fs
                LEFT JOIN auth_users u ON u.id = fs.user_id
                LEFT JOIN subscription_plans sp ON sp.plan_id = fs.plan_id
            """
            params: dict = {}
            where = []
            if search.strip():
                where.append("(u.email ILIKE %(q)s OR fs.session_id ILIKE %(q)s)")
                params["q"] = f"%{search.strip()}%"
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""

            cur.execute(f"SELECT COUNT(*) AS n {base}{where_sql}", params)
            total = int(cur.fetchone()["n"])

            params["limit"] = page_size
            params["offset"] = (page - 1) * page_size
            cur.execute(
                f"""
                SELECT fs.session_id, fs.user_id, fs.plan_id, fs.fulfilled_at,
                       u.email, COALESCE(sp.price, 0) AS price
                {base}{where_sql}
                ORDER BY fs.fulfilled_at DESC NULLS LAST
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                params,
            )
            rows = cur.fetchall()

        txns = []
        gross_total = 0.0
        for r in rows:
            price = float(r["price"] or 0.0)
            gross_total += price
            billing_type = _plan_billing_type(r["plan_id"])
            txn_type = "purchase" if billing_type == "one_time" else (
                "subscription" if billing_type == "subscription" else (billing_type or "—")
            )
            txns.append({
                "session_id": r["session_id"],
                "user_id": r["user_id"],
                "email": r["email"],
                "plan_id": r["plan_id"],
                "plan_name": _plan_name(r["plan_id"]),
                "type": txn_type,
                "gross": round(price, 2),
                "net": round(price, 2),     # no fee data → net == gross
                "status": "paid",           # fulfilled_sessions only records successes
                "occurred_at": _fmt_dt(r["fulfilled_at"]),
                # Not tracked in the schema.
                "fees": None,
                "promo": None,
            })

        return {
            "transactions": txns,
            "total": total,
            "page": page,
            "page_size": page_size,
            "gross_on_page": round(gross_total, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GET /api/admin/customer/{user_id} — Customer 360 (drawer)
# =============================================================================
@router.get("/customer/{user_id}")
def admin_customer_detail(user_id: str, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    period = _current_period()
    try:
        with get_db() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(
                """
                SELECT id, email, name, organization_name, created_at,
                       status AS account_status, is_admin
                FROM auth_users WHERE id = %s
                """,
                (user_id,),
            )
            u = cur.fetchone()
            if not u:
                raise HTTPException(status_code=404, detail="User not found")

            cur.execute(
                """
                SELECT plan_id, status, current_period_start, current_period_end,
                       stripe_customer_id, stripe_subscription_id
                FROM user_subscriptions WHERE user_id = %s
                """,
                (user_id,),
            )
            sub = cur.fetchone()

            cur.execute("SELECT COALESCE(credits, 0) AS credits FROM image_credits WHERE user_id = %s", (user_id,))
            crow = cur.fetchone()
            credits = int(crow["credits"]) if crow else 0

            cur.execute(
                "SELECT COALESCE(SUM(used), 0) AS total, MAX(updated_at) AS last_used FROM image_usage WHERE user_id = %s",
                (user_id,),
            )
            iu = cur.fetchone()
            total_images = int(iu["total"])
            last_active = iu["last_used"]

            cur.execute(
                "SELECT COALESCE(used, 0) AS used FROM image_usage WHERE user_id = %s AND period = %s",
                (user_id, period),
            )
            cyc = cur.fetchone()
            cycle_used = int(cyc["used"]) if cyc else 0

            # Per-month usage history (last 12 months) for the activity view.
            cur.execute(
                """
                SELECT period, used, updated_at FROM image_usage
                WHERE user_id = %s ORDER BY period DESC LIMIT 12
                """,
                (user_id,),
            )
            usage_history = [
                {"period": r["period"], "used": int(r["used"]), "updated_at": _fmt_dt(r["updated_at"])}
                for r in cur.fetchall()
            ]

            # Payment history for this user.
            cur.execute(
                """
                SELECT fs.session_id, fs.plan_id, fs.fulfilled_at, COALESCE(sp.price, 0) AS price
                FROM fulfilled_sessions fs
                LEFT JOIN subscription_plans sp ON sp.plan_id = fs.plan_id
                WHERE fs.user_id = %s
                ORDER BY fs.fulfilled_at DESC
                """,
                (user_id,),
            )
            payments = []
            lifetime_revenue = 0.0
            for r in cur.fetchall():
                price = float(r["price"] or 0.0)
                lifetime_revenue += price
                payments.append({
                    "session_id": r["session_id"],
                    "plan_id": r["plan_id"],
                    "plan_name": _plan_name(r["plan_id"]),
                    "type": "purchase" if _plan_billing_type(r["plan_id"]) == "one_time" else "subscription",
                    "gross": round(price, 2),
                    "status": "paid",
                    "occurred_at": _fmt_dt(r["fulfilled_at"]),
                })

        sub_plan = sub["plan_id"] if sub else None
        sub_status = sub["status"] if sub else None
        period_end = sub.get("current_period_end") if sub else None
        eff_plan = _resolve_effective_plan(sub_plan, sub_status, period_end)
        limit = _image_limit(eff_plan)
        unlimited = (limit == UNLIMITED)
        granted = None if unlimited else int(limit)
        used_pct = None if unlimited else (round((cycle_used / granted) * 100) if granted else 0)

        return {
            "id": u["id"],
            "email": u["email"],
            "name": u["name"],
            "organization_name": u["organization_name"],
            "created_at": _fmt_dt(u["created_at"]),
            "account_status": u["account_status"],
            "is_admin": bool(u["is_admin"]),
            "plan_id": eff_plan,
            "plan_name": _plan_name(eff_plan),
            "status": _display_status(eff_plan, sub_status),
            "current_period_start": _fmt_dt(period_end and sub.get("current_period_start")),
            "current_period_end": _fmt_dt(period_end),
            "stripe_customer_id": sub.get("stripe_customer_id") if sub else None,
            "stripe_subscription_id": sub.get("stripe_subscription_id") if sub else None,
            "credits": credits,
            "total_images": total_images,
            "cycle_used": cycle_used,
            "cycle_granted": granted,
            "unlimited": unlimited,
            "used_pct": used_pct,
            "last_active": _fmt_dt(last_active),
            "lifetime_revenue": round(lifetime_revenue, 2),
            "usage_history": usage_history,
            "payments": payments,
            # Not tracked in the schema (yet).
            "country": None,
            "platform": None,
            "health": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GET /api/admin/analytics — real series that power the dashboard charts
# =============================================================================
@router.get("/analytics")
def admin_analytics(x_admin_token: Optional[str] = Header(None)):
    """Aggregations for every chart the live schema can back.

    Charts that need data the schema does not collect yet (churn surveys, health
    rollups, cohort tracking, generation latency, fraud signals) are NOT faked
    here — the UI renders an explicit empty state for those instead.
    """
    _require_admin(x_admin_token)
    period = _current_period()
    days = _last_days(30)
    months = _last_months(12)
    try:
        with get_db() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # --- Revenue by day (last 30d), split by plan for stacking ---
            cur.execute(
                """
                SELECT to_char(fs.fulfilled_at::date, 'YYYY-MM-DD') AS d,
                       fs.plan_id, COALESCE(SUM(sp.price), 0) AS rev
                FROM fulfilled_sessions fs
                LEFT JOIN subscription_plans sp ON sp.plan_id = fs.plan_id
                WHERE fs.fulfilled_at >= (NOW() - INTERVAL '30 days')
                GROUP BY 1, 2
                """
            )
            rev_by_plan: dict = {}
            for r in cur.fetchall():
                plan = r["plan_id"] or "unknown"
                rev_by_plan.setdefault(plan, {d: 0.0 for d in days})
                if r["d"] in rev_by_plan[plan]:
                    rev_by_plan[plan][r["d"]] = float(r["rev"] or 0.0)
            revenue_by_day = {
                "days": days,
                "by_plan": {p: [round(v[d], 2) for d in days] for p, v in rev_by_plan.items()},
                "plan_names": {p: _plan_name(p) for p in rev_by_plan},
            }

            # --- New signups by day (last 30d) ---
            cur.execute(
                """
                SELECT to_char(created_at::date, 'YYYY-MM-DD') AS d, COUNT(*) AS n
                FROM auth_users
                WHERE created_at >= (NOW() - INTERVAL '30 days')
                GROUP BY 1
                """
            )
            nu = {r["d"]: int(r["n"]) for r in cur.fetchall()}
            new_users_by_day = {"days": days, "counts": [nu.get(d, 0) for d in days]}

            # --- Images used by month (last 12) ---
            cur.execute("SELECT period, COALESCE(SUM(used), 0) AS used FROM image_usage GROUP BY period")
            im = {r["period"]: int(r["used"]) for r in cur.fetchall()}
            images_by_month = {"months": months, "used": [im.get(m, 0) for m in months]}

            # --- Subscription status counts (Payments card) ---
            cur.execute("SELECT COALESCE(status, 'none') AS status, COUNT(*) AS n FROM user_subscriptions GROUP BY status")
            payment_status = {r["status"]: int(r["n"]) for r in cur.fetchall()}

            cur.execute("SELECT COALESCE(SUM(credits), 0) AS n FROM image_credits")
            credits_outstanding = int(cur.fetchone()["n"])

            # --- One per-user pass → plan mix, funnel, value-by-plan, usage hist ---
            cur.execute(
                """
                SELECT u.id,
                       s.plan_id AS sub_plan, s.status AS sub_status, s.current_period_end,
                       COALESCE(iu.total, 0) AS total_images,
                       COALESCE(cyc.used, 0) AS cycle_used,
                       COALESCE(pay.rev, 0) AS rev
                FROM auth_users u
                LEFT JOIN user_subscriptions s ON s.user_id = u.id
                LEFT JOIN (SELECT user_id, SUM(used) AS total FROM image_usage GROUP BY user_id) iu ON iu.user_id = u.id
                LEFT JOIN (SELECT user_id, used FROM image_usage WHERE period = %(period)s) cyc ON cyc.user_id = u.id
                LEFT JOIN (
                    SELECT fs.user_id, SUM(sp.price) AS rev
                    FROM fulfilled_sessions fs
                    LEFT JOIN subscription_plans sp ON sp.plan_id = fs.plan_id
                    GROUP BY fs.user_id
                ) pay ON pay.user_id = u.id
                """,
                {"period": period},
            )
            user_rows = cur.fetchall()

        # --- derive per-user aggregates in Python ---
        plan_mix: dict = {}
        value_by_plan: dict = {}
        signed_up = len(user_rows)
        activated = 0
        subscribed = 0
        granted_capacity = 0
        used_this_cycle = 0
        # used% buckets across users on a finite plan (Wasted Credits histogram)
        buckets = ["0%", "1-25%", "26-50%", "51-75%", "76-99%", "100%"]
        usage_hist = {b: 0 for b in buckets}

        for r in user_rows:
            eff = _resolve_effective_plan(r["sub_plan"], r["sub_status"], r.get("current_period_end"))
            limit = _image_limit(eff)
            images = int(r["total_images"])
            cyc_used = int(r["cycle_used"])
            rev = float(r["rev"] or 0.0)

            plan_mix[eff] = plan_mix.get(eff, 0) + 1
            if images > 0:
                activated += 1
            if eff != DEFAULT_PLAN:
                subscribed += 1

            vp = value_by_plan.setdefault(eff, {"images": 0, "revenue": 0.0})
            vp["images"] += images
            vp["revenue"] += rev

            used_this_cycle += cyc_used
            if limit not in (UNLIMITED, 0):
                granted_capacity += int(limit)
                pct = (cyc_used / limit) * 100 if limit else 0
                if pct <= 0:
                    usage_hist["0%"] += 1
                elif pct <= 25:
                    usage_hist["1-25%"] += 1
                elif pct <= 50:
                    usage_hist["26-50%"] += 1
                elif pct <= 75:
                    usage_hist["51-75%"] += 1
                elif pct < 100:
                    usage_hist["76-99%"] += 1
                else:
                    usage_hist["100%"] += 1

        plan_mix_out = [
            {"plan_id": p, "plan_name": _plan_name(p), "count": c}
            for p, c in sorted(plan_mix.items(), key=lambda kv: -kv[1])
        ]
        value_out = [
            {
                "plan_id": p,
                "plan_name": _plan_name(p),
                "images": v["images"],
                "implied_value": round(v["images"] * IMAGE_VALUE, 2),
                "revenue": round(v["revenue"], 2),
            }
            for p, v in sorted(value_by_plan.items(), key=lambda kv: -kv[1]["images"])
            if v["images"] > 0 or v["revenue"] > 0
        ]

        return {
            "image_value": IMAGE_VALUE,
            "revenue_by_day": revenue_by_day,
            "new_users_by_day": new_users_by_day,
            "images_by_month": images_by_month,
            "plan_mix": plan_mix_out,
            "funnel": {"signed_up": signed_up, "activated": activated, "subscribed": subscribed},
            "value_by_plan": value_out,
            "payment_status": payment_status,
            "credits": {
                "granted_capacity": granted_capacity,
                "used_this_cycle": used_this_cycle,
                "outstanding": credits_outstanding,
            },
            "usage_hist": {"buckets": buckets, "counts": [usage_hist[b] for b in buckets]},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


__all__ = ["router"]
