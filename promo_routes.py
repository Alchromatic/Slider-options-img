#!/usr/bin/env python3
"""
promo_routes.py
===============

Promo codes that give free images: an admin generates N codes (each worth X
images, usable U times), a signed-in user types one in the "Have a code?" box
and the images are added to their image credits (the same balance a PAYG
purchase fills, spent after the monthly allowance). Every use is recorded
against the user id, so it's clear who used which code, and a user can use a
given code only once. Meant for testing and giveaways without Stripe test
cards.

User (Bearer user JWT or device token; an unpaired device gets 401):
    POST /api/billing/redeem-code        {code}  -> {images, credits, entitlements}
    GET  /api/billing/redemptions                -> the codes this user has used

Admin (X-Admin-Token, same as the rest of /api/admin):
    POST  /api/admin/promo-codes                  generate N codes (or one custom code)
    GET   /api/admin/promo-codes                  list, filter by batch / status / text
    GET   /api/admin/promo-codes/batches          one row per batch with totals
    PATCH /api/admin/promo-codes/{code}           {disabled} turn a code off / on
    POST  /api/admin/promo-codes/batches/disable  {batch, disabled} a whole batch

Codes are stored uppercase without separators ("GM7KQ2XP9WRD"); generated ones
are shown grouped ("GM-7KQ2X-P9WRD"), custom ones as entered ("LAUNCH50"). Users
can type them in any case, with or without dashes and spaces.
"""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from admin_routes import _require_admin
from auth_db import RealDictCursor, get_db
from billing import get_entitlements
from devices_routes import _auth, _record_failure, _throttle

user_router = APIRouter(prefix="/api/billing", tags=["Billing"])
admin_router = APIRouter(prefix="/api/admin/promo-codes", tags=["14. Super Admin"])

# No 0/O/1/I so a code read off a screen or paper is unambiguous.
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_RANDOM_LEN = 10                       # 32^10 ≈ 10^15 per prefix
_CODE_RE = re.compile(r"^[A-Z0-9]{4,32}$")
MAX_GENERATE = 1000


# ---------------------------------------------------------------------------
# Tables (called at app startup from main.py)
# ---------------------------------------------------------------------------

def init_promo_tables() -> None:
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS promo_codes (
                    code             TEXT PRIMARY KEY,
                    display          TEXT NOT NULL,
                    images           INTEGER NOT NULL CHECK (images > 0),
                    max_redemptions  INTEGER NOT NULL DEFAULT 1 CHECK (max_redemptions > 0),
                    redeemed_count   INTEGER NOT NULL DEFAULT 0,
                    batch            TEXT,
                    note             TEXT,
                    expires_at       TIMESTAMPTZ,
                    disabled         BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_promo_codes_batch ON promo_codes(batch)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS promo_redemptions (
                    id           SERIAL PRIMARY KEY,
                    code         TEXT NOT NULL REFERENCES promo_codes(code),
                    user_id      TEXT NOT NULL,
                    images       INTEGER NOT NULL,
                    redeemed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (code, user_id)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_promo_redemptions_user ON promo_redemptions(user_id)")
            conn.commit()
            print("[OK] promo code tables ready")
    except Exception as e:  # pragma: no cover - depends on live DB
        print(f"[WARN] promo code tables init error: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_code(raw: Optional[str]) -> str:
    """'gm-7kq2x p9wrd' -> 'GM7KQ2XP9WRD' (letters and digits only, uppercase)."""
    return re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()


def _new_code(prefix: str):
    """-> (stored code, how it's shown): ('GM7KQ2XP9WRD', 'GM-7KQ2X-P9WRD')."""
    body = "".join(secrets.choice(_ALPHABET) for _ in range(_RANDOM_LEN))
    shown = "-".join(([prefix] if prefix else []) + [body[:5], body[5:]])
    return prefix + body, shown


def _iso(ts) -> Optional[str]:
    return ts.isoformat() if ts else None


def _status(row: Dict[str, Any]) -> str:
    if row["disabled"]:
        return "disabled"
    exp = row.get("expires_at")
    if exp and exp < datetime.now(timezone.utc):
        return "expired"
    if row["redeemed_count"] >= row["max_redemptions"]:
        return "used"
    return "active"


def _code_out(row: Dict[str, Any], redemptions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    out = {
        "code": row["display"],
        "images": row["images"],
        "max_redemptions": row["max_redemptions"],
        "redeemed_count": row["redeemed_count"],
        "batch": row.get("batch"),
        "note": row.get("note"),
        "expires_at": _iso(row.get("expires_at")),
        "disabled": row["disabled"],
        "created_at": _iso(row.get("created_at")),
        "status": _status(row),
    }
    if redemptions is not None:
        out["redemptions"] = redemptions
    return out


# ---------------------------------------------------------------------------
# User: redeem
# ---------------------------------------------------------------------------

class RedeemIn(BaseModel):
    code: str = Field(..., max_length=64)


@user_router.post("/redeem-code")
def redeem_code(req: RedeemIn, request: Request):
    """Add the code's images to the signed-in user's image credits (once per user per code)."""
    ctx = _auth(request)
    uid = ctx["user_id"]
    code = normalize_code(req.code)
    # 10 wrong codes per user per 10 min -> 429. Per user, not per IP, so one
    # tester's typos don't lock out a whole office; guessing is hopeless anyway
    # (10 random characters from 32).
    key = "promo:user:" + uid
    _throttle(key)

    def fail(status: int, detail: str):
        _record_failure(key)
        raise HTTPException(status_code=status, detail=detail)

    if not _CODE_RE.match(code):
        fail(400, "Enter the code exactly as you received it.")

    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT 1 FROM auth_users WHERE id = %s", (uid,))
        if not cur.fetchone():
            raise HTTPException(status_code=401, detail="Please sign in again.")
        # lock the code row: parallel redeems of the same code queue up here
        cur.execute("SELECT * FROM promo_codes WHERE code = %s FOR UPDATE", (code,))
        row = cur.fetchone()
        if not row:
            fail(404, "That code isn't valid.")
        cur.execute("SELECT 1 FROM promo_redemptions WHERE code = %s AND user_id = %s", (code, uid))
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="You've already used this code.")
        status = _status(row)
        if status == "disabled":
            raise HTTPException(status_code=410, detail="This code is no longer active.")
        if status == "expired":
            raise HTTPException(status_code=410, detail="This code has expired.")
        if status == "used":
            raise HTTPException(status_code=409, detail="This code has already been used.")

        images = int(row["images"])
        cur.execute(
            "INSERT INTO promo_redemptions (code, user_id, images) VALUES (%s, %s, %s)",
            (code, uid, images),
        )
        cur.execute("UPDATE promo_codes SET redeemed_count = redeemed_count + 1 WHERE code = %s", (code,))
        cur.execute(
            """
            INSERT INTO image_credits (user_id, credits, updated_at) VALUES (%s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                credits = image_credits.credits + EXCLUDED.credits, updated_at = NOW()
            RETURNING credits
            """,
            (uid, images),
        )
        credits = int(cur.fetchone()["credits"])
        conn.commit()

    return {
        "ok": True,
        "code": row["display"],
        "images": images,
        "credits": credits,
        "entitlements": get_entitlements(uid),
    }


@user_router.get("/redemptions")
def my_redemptions(request: Request):
    """The codes the signed-in user has used, newest first."""
    uid = _auth(request)["user_id"]
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT c.display, r.images, r.redeemed_at FROM promo_redemptions r "
            "JOIN promo_codes c ON c.code = r.code WHERE r.user_id = %s ORDER BY r.redeemed_at DESC",
            (uid,),
        )
        rows = cur.fetchall()
    return {"redemptions": [
        {"code": r["display"], "images": r["images"], "redeemed_at": _iso(r["redeemed_at"])}
        for r in rows
    ]}


# ---------------------------------------------------------------------------
# Admin: generate / list / disable
# ---------------------------------------------------------------------------

class GenerateIn(BaseModel):
    count: int = Field(1, ge=1, le=MAX_GENERATE, description="How many codes.")
    images: int = Field(..., ge=1, le=100000, description="Free images each code gives.")
    max_redemptions: int = Field(1, ge=1, le=1000000, description="How many different users can use each code.")
    batch: Optional[str] = Field(None, max_length=80, description="Label to find the batch later.")
    note: Optional[str] = Field(None, max_length=300)
    expires_at: Optional[datetime] = Field(None, description="ISO date/time; empty = never expires.")
    prefix: Optional[str] = Field("GM", max_length=6, description="Letters in front of the random part.")
    code: Optional[str] = Field(None, max_length=40, description="One specific code instead of random ones (count must be 1).")


class DisableIn(BaseModel):
    disabled: bool = True


class BatchDisableIn(BaseModel):
    batch: str = Field(..., min_length=1, max_length=80)
    disabled: bool = True


@admin_router.post("")
def generate_codes(req: GenerateIn, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    batch = (req.batch or "").strip() or datetime.utcnow().strftime("batch-%Y%m%d-%H%M%S")
    note = (req.note or "").strip() or None
    expires = req.expires_at
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires is not None and expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="The expiry date is in the past.")

    if req.code:
        custom = normalize_code(req.code)
        if req.count != 1:
            raise HTTPException(status_code=400, detail="A custom code is a single code: set count to 1.")
        if not _CODE_RE.match(custom):
            raise HTTPException(status_code=400, detail="A custom code needs 4-32 letters or digits.")
        wanted = (custom, custom)
    else:
        prefix = re.sub(r"[^A-Z]", "", (req.prefix or "").upper())[:6]
        wanted = None

    made: List[str] = []
    with get_db() as conn:
        cur = conn.cursor()
        attempts = 0
        while len(made) < req.count:
            attempts += 1
            if attempts > req.count * 3 + 10:
                raise HTTPException(status_code=500, detail="Could not generate unique codes, try again.")
            code, shown = wanted or _new_code(prefix)
            cur.execute(
                """
                INSERT INTO promo_codes (code, display, images, max_redemptions, batch, note, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (code) DO NOTHING
                """,
                (code, shown, req.images, req.max_redemptions, batch, note, expires),
            )
            if cur.rowcount:
                made.append(shown)
            elif wanted:
                raise HTTPException(status_code=409, detail="That code already exists.")
        conn.commit()
    return {
        "batch": batch,
        "images": req.images,
        "max_redemptions": req.max_redemptions,
        "expires_at": _iso(expires),
        "codes": made,
    }


@admin_router.get("")
def list_codes(
    x_admin_token: Optional[str] = Header(None),
    batch: Optional[str] = None,
    status: Optional[str] = Query(None, description="active | used | expired | disabled"),
    q: Optional[str] = Query(None, description="Part of a code, or a user's email"),
    limit: int = 200,
    offset: int = 0,
):
    """Codes with who used them (user id + email), newest first."""
    _require_admin(x_admin_token)
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    where, params = ["TRUE"], []
    if batch:
        where.append("c.batch = %s")
        params.append(batch)
    if q:
        text = q.strip()
        where.append(
            "(c.code LIKE %s OR EXISTS (SELECT 1 FROM promo_redemptions r JOIN auth_users u ON u.id = r.user_id "
            "WHERE r.code = c.code AND (u.email ILIKE %s OR r.user_id = %s)))"
        )
        params += ["%" + normalize_code(text) + "%", "%" + text + "%", text]
    if status == "disabled":
        where.append("c.disabled")
    elif status == "expired":
        where.append("NOT c.disabled AND c.expires_at IS NOT NULL AND c.expires_at < NOW()")
    elif status == "used":
        where.append("NOT c.disabled AND (c.expires_at IS NULL OR c.expires_at >= NOW()) AND c.redeemed_count >= c.max_redemptions")
    elif status == "active":
        where.append("NOT c.disabled AND (c.expires_at IS NULL OR c.expires_at >= NOW()) AND c.redeemed_count < c.max_redemptions")
    sql_where = " AND ".join(where)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SELECT COUNT(*) AS n FROM promo_codes c WHERE {sql_where}", params)
        total = cur.fetchone()["n"]
        cur.execute(
            f"SELECT c.* FROM promo_codes c WHERE {sql_where} ORDER BY c.created_at DESC, c.code LIMIT %s OFFSET %s",
            params + [limit, offset],
        )
        rows = [dict(r) for r in cur.fetchall()]
        used: Dict[str, List[Dict[str, Any]]] = {}
        if rows:
            cur.execute(
                """
                SELECT r.code, r.user_id, u.email, u.name, r.images, r.redeemed_at
                FROM promo_redemptions r LEFT JOIN auth_users u ON u.id = r.user_id
                WHERE r.code = ANY(%s) ORDER BY r.redeemed_at
                """,
                ([r["code"] for r in rows],),
            )
            for r in cur.fetchall():
                used.setdefault(r["code"], []).append({
                    "user_id": r["user_id"], "email": r["email"], "name": r["name"],
                    "images": r["images"], "redeemed_at": _iso(r["redeemed_at"]),
                })
    return {
        "codes": [_code_out(r, used.get(r["code"], [])) for r in rows],
        "total": total, "limit": limit, "offset": offset,
    }


@admin_router.get("/batches")
def list_batches(x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT batch,
                   COUNT(*)                                        AS codes,
                   MIN(images)                                     AS images_min,
                   MAX(images)                                     AS images_max,
                   SUM(redeemed_count)                             AS redemptions,
                   SUM(max_redemptions)                            AS capacity,
                   SUM(redeemed_count * images)                    AS images_given,
                   COUNT(*) FILTER (WHERE disabled)                AS disabled,
                   MIN(created_at)                                 AS created_at,
                   MAX(expires_at)                                 AS expires_at,
                   MAX(note)                                       AS note
            FROM promo_codes GROUP BY batch ORDER BY MIN(created_at) DESC
            """
        )
        rows = cur.fetchall()
    return {"batches": [
        {
            "batch": r["batch"], "codes": r["codes"],
            "images": r["images_min"] if r["images_min"] == r["images_max"] else f'{r["images_min"]}-{r["images_max"]}',
            "redemptions": int(r["redemptions"] or 0), "capacity": int(r["capacity"] or 0),
            "images_given": int(r["images_given"] or 0), "disabled": r["disabled"],
            "created_at": _iso(r["created_at"]), "expires_at": _iso(r["expires_at"]), "note": r["note"],
        }
        for r in rows
    ]}


@admin_router.patch("/{code}")
def set_code_disabled(code: str, req: DisableIn, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "UPDATE promo_codes SET disabled = %s WHERE code = %s RETURNING *",
            (req.disabled, normalize_code(code)),
        )
        row = cur.fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Code not found.")
    return {"code": _code_out(dict(row))}


@admin_router.post("/batches/disable")
def set_batch_disabled(req: BatchDisableIn, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE promo_codes SET disabled = %s WHERE batch = %s", (req.disabled, req.batch))
        n = cur.rowcount
        conn.commit()
    if not n:
        raise HTTPException(status_code=404, detail="Batch not found.")
    return {"batch": req.batch, "disabled": req.disabled, "codes": n}


__all__ = ["user_router", "admin_router", "init_promo_tables", "normalize_code"]
