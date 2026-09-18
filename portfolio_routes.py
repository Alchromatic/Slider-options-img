#!/usr/bin/env python3
"""
portfolio_routes.py
===================

Per-user Portfolio (saved compositions), stored in the shared Supabase
PostgreSQL database so the webapp and the Alchroma apps see the same items.
Before this the web Portfolio lived only in each browser's localStorage and the
apps kept their own copy on the device.

Endpoints (Bearer user JWT or device token, same scheme as devices_routes.py;
an unpaired device gets ``401 device_revoked``):

    GET    /api/portfolio              -> list (newest first, without shapes)
    GET    /api/portfolio/{id}         -> one item, with shapes
    POST   /api/portfolio              -> create or update one item (by id)
    POST   /api/portfolio/batch        -> create or update several (sync / offline queue)
    PATCH  /api/portfolio/{id}         -> rename
    DELETE /api/portfolio/{id}

Items use the web Portfolio's own shape (camelCase):

    {id, title, thumb, w, h, shapeCount, logic, colorOrder, limit, meta,
     source, deviceId, createdAt, updatedAt, hasShapes, shapes?}

``id`` is chosen by the client (so an item saved offline keeps its id when it is
uploaded later); ``createdAt`` / ``updatedAt`` are epoch milliseconds.  Posting
an existing id updates it; fields left out (e.g. ``shapes``) are kept.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from auth_db import RealDictCursor, get_db
from devices_routes import _auth

router = APIRouter(prefix="/api/portfolio", tags=["16. Portfolio"])

MAX_THUMB_CHARS = 3_000_000        # ~2 MB image as a data URL
MAX_SHAPES_CHARS = 10_000_000      # serialized shapes JSON
MAX_BATCH = 50
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
SOURCES = {"web", "app", "quest", "other"}


# ---------------------------------------------------------------------------
# Table init (called at app startup from main.py)
# ---------------------------------------------------------------------------

def init_portfolio_tables() -> None:
    """Create the user_portfolio table if it doesn't exist."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_portfolio (
                    user_id      TEXT NOT NULL,
                    id           TEXT NOT NULL,
                    title        TEXT NOT NULL DEFAULT 'Untitled',
                    thumb        TEXT,
                    w            INTEGER,
                    h            INTEGER,
                    shape_count  INTEGER,
                    logic        TEXT,
                    shapes       JSONB,
                    meta         JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source       TEXT NOT NULL DEFAULT 'web',
                    device_id    TEXT,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, id)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_portfolio_user_created "
                "ON user_portfolio(user_id, created_at DESC)"
            )
            conn.commit()
            print("[OK] user_portfolio table ready")
    except Exception as e:  # pragma: no cover - depends on live DB
        print(f"[WARN] user_portfolio table init error: {e}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PortfolioIn(BaseModel):
    id: Optional[str] = Field(None, description="Client id; generated when omitted.")
    title: Optional[str] = Field(None, max_length=200)
    thumb: Optional[str] = Field(None, description="data:image/... URL or https URL.")
    w: Optional[int] = None
    h: Optional[int] = None
    shapeCount: Optional[int] = None
    logic: Optional[str] = Field(None, max_length=40)
    shapes: Optional[List[Any]] = Field(None, description="Geometrize shapes; kept when omitted.")
    colorOrder: Optional[Any] = None
    limit: Optional[int] = None
    meta: Optional[Dict[str, Any]] = Field(None, description="Free-form extra fields (app-specific).")
    source: Optional[str] = Field(None, description="web | app | quest | other")
    createdAt: Optional[float] = Field(None, description="Epoch ms; defaults to now.")


class PortfolioBatch(BaseModel):
    items: List[PortfolioIn] = Field(default_factory=list)


class PortfolioRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(ts) -> Optional[int]:
    return int(ts.timestamp() * 1000) if ts else None


def _item_out(row: Dict[str, Any], with_shapes: bool = False) -> Dict[str, Any]:
    meta = dict(row.get("meta") or {})
    out = {
        "id": row["id"],
        "title": row["title"],
        "thumb": row.get("thumb"),
        "w": row.get("w"),
        "h": row.get("h"),
        "shapeCount": row.get("shape_count"),
        "logic": row.get("logic"),
        "colorOrder": meta.pop("colorOrder", None),
        "limit": meta.pop("limit", None),
        "meta": meta,
        "source": row.get("source"),
        "deviceId": row.get("device_id"),
        "createdAt": _ms(row.get("created_at")),
        "updatedAt": _ms(row.get("updated_at")),
        "hasShapes": bool(row.get("has_shapes")),
    }
    if with_shapes:
        out["shapes"] = row.get("shapes") or []
    return out


def _check_id(item_id: str) -> str:
    if not _ID_RE.match(item_id or ""):
        raise HTTPException(status_code=400, detail="Invalid portfolio id")
    return item_id


def _upsert(cur, ctx: Dict[str, Any], item: PortfolioIn) -> Dict[str, Any]:
    item_id = _check_id(item.id) if item.id else "p" + secrets.token_hex(8)
    thumb = item.thumb
    if thumb is not None:
        if not (thumb.startswith("data:image/") or thumb.startswith("https://") or thumb.startswith("http://")):
            raise HTTPException(status_code=400, detail="thumb must be a data:image URL or an http(s) URL")
        if len(thumb) > MAX_THUMB_CHARS:
            raise HTTPException(status_code=413, detail="thumb is too large")
    shapes_json = None
    if item.shapes is not None:
        shapes_json = json.dumps(item.shapes, separators=(",", ":"))
        if len(shapes_json) > MAX_SHAPES_CHARS:
            raise HTTPException(status_code=413, detail="shapes are too large")
    meta = dict(item.meta or {})
    if item.colorOrder is not None:
        meta["colorOrder"] = item.colorOrder
    if item.limit is not None:
        meta["limit"] = item.limit
    source = (item.source or ("app" if ctx.get("device_id") else "web")).lower()
    if source not in SOURCES:
        source = "other"
    created_ms = item.createdAt if item.createdAt and item.createdAt > 0 else time.time() * 1000
    title = (item.title or "").strip()[:200] or None      # None keeps the stored title

    cur.execute(
        """
        INSERT INTO user_portfolio
            (user_id, id, title, thumb, w, h, shape_count, logic, shapes, meta,
             source, device_id, created_at, updated_at)
        VALUES (%s, %s, COALESCE(%s, 'Untitled'), %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                %s, %s, to_timestamp(%s / 1000.0), NOW())
        ON CONFLICT (user_id, id) DO UPDATE SET
            title       = CASE WHEN %s::text IS NULL THEN user_portfolio.title ELSE EXCLUDED.title END,
            thumb       = COALESCE(EXCLUDED.thumb, user_portfolio.thumb),
            w           = COALESCE(EXCLUDED.w, user_portfolio.w),
            h           = COALESCE(EXCLUDED.h, user_portfolio.h),
            shape_count = COALESCE(EXCLUDED.shape_count, user_portfolio.shape_count),
            logic       = COALESCE(EXCLUDED.logic, user_portfolio.logic),
            shapes      = COALESCE(EXCLUDED.shapes, user_portfolio.shapes),
            meta        = user_portfolio.meta || EXCLUDED.meta,
            updated_at  = NOW()
        RETURNING *, (shapes IS NOT NULL AND jsonb_array_length(shapes) > 0) AS has_shapes
        """,
        (
            ctx["user_id"], item_id, title, thumb, item.w, item.h, item.shapeCount,
            item.logic, shapes_json, json.dumps(meta), source, ctx.get("device_id"),
            created_ms, title,
        ),
    )
    return dict(cur.fetchone())


_LIST_COLS = (
    "user_id, id, title, thumb, w, h, shape_count, logic, meta, source, device_id, "
    "created_at, updated_at, (shapes IS NOT NULL AND jsonb_array_length(shapes) > 0) AS has_shapes"
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("")
def list_portfolio(request: Request, limit: int = 100, offset: int = 0):
    """The user's Portfolio, newest first. Shapes are left out; fetch one item for them."""
    ctx = _auth(request)
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) AS n FROM user_portfolio WHERE user_id = %s", (ctx["user_id"],))
        total = cur.fetchone()["n"]
        cur.execute(
            f"SELECT {_LIST_COLS} FROM user_portfolio WHERE user_id = %s "
            "ORDER BY created_at DESC, id LIMIT %s OFFSET %s",
            (ctx["user_id"], limit, offset),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"items": [_item_out(r) for r in rows], "total": total, "limit": limit, "offset": offset}


@router.get("/{item_id}")
def get_portfolio_item(item_id: str, request: Request):
    """One item including its shapes (to reopen it in the Colors page)."""
    ctx = _auth(request)
    _check_id(item_id)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            f"SELECT {_LIST_COLS}, shapes FROM user_portfolio WHERE user_id = %s AND id = %s",
            (ctx["user_id"], item_id),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Portfolio item not found")
    return {"item": _item_out(dict(row), with_shapes=True)}


@router.post("")
def save_portfolio_item(item: PortfolioIn, request: Request):
    """Create an item, or update it when the id already exists."""
    ctx = _auth(request)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _upsert(cur, ctx, item)
        conn.commit()
    return {"item": _item_out(row)}


@router.post("/batch")
def save_portfolio_batch(batch: PortfolioBatch, request: Request):
    """Create/update several items at once (first sync of a browser, app offline queue)."""
    ctx = _auth(request)
    if len(batch.items) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"At most {MAX_BATCH} items per batch")
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        rows = [_upsert(cur, ctx, it) for it in batch.items]
        conn.commit()
    return {"items": [_item_out(r) for r in rows], "count": len(rows)}


@router.patch("/{item_id}")
def rename_portfolio_item(item_id: str, req: PortfolioRename, request: Request):
    ctx = _auth(request)
    _check_id(item_id)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            f"UPDATE user_portfolio SET title = %s, updated_at = NOW() "
            f"WHERE user_id = %s AND id = %s RETURNING {_LIST_COLS}",
            (req.title.strip(), ctx["user_id"], item_id),
        )
        row = cur.fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Portfolio item not found")
    return {"item": _item_out(dict(row))}


@router.delete("/{item_id}")
def delete_portfolio_item(item_id: str, request: Request):
    ctx = _auth(request)
    _check_id(item_id)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM user_portfolio WHERE user_id = %s AND id = %s",
            (ctx["user_id"], item_id),
        )
        n = cur.rowcount
        conn.commit()
    if not n:
        raise HTTPException(status_code=404, detail="Portfolio item not found")
    return {"deleted": True, "id": item_id}


__all__ = ["router", "init_portfolio_tables"]
