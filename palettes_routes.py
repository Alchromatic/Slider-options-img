#!/usr/bin/env python3
"""
palettes_routes.py
==================

Per-user saved color palettes, stored in the shared Supabase PostgreSQL database
(same connection/auth as auth_db.py / billing.py).  This lets a user's Color
Library ("My Colors") and any palettes they save follow them across devices
instead of living only in browser localStorage.

Endpoints (all require a Bearer JWT, same scheme as billing.py):

    GET    /api/palettes            -> list the current user's saved palettes
    POST   /api/palettes            -> create or update a palette (by name)
    DELETE /api/palettes/{id}       -> delete one of the user's palettes

A palette is ``{id, name, colors:[{hex, name}]}``.  Palettes are unique per
(user_id, name), so saving "My Colors" again just updates it.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from auth_db import RealDictCursor, decode_jwt_token, get_db

router = APIRouter(prefix="/api/palettes", tags=["14. User palettes"])

_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$")


# ---------------------------------------------------------------------------
# Table init (called at app startup from main.py)
# ---------------------------------------------------------------------------

def init_palettes_tables() -> None:
    """Create the user_palettes table if it doesn't exist."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_palettes (
                    id          SERIAL PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    colors      JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at  TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (user_id, name)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_palettes_user ON user_palettes(user_id)"
            )
            conn.commit()
            print("[OK] user_palettes table ready")
    except Exception as e:  # pragma: no cover - depends on live DB
        print(f"[WARN] user_palettes table init error: {e}")


# ---------------------------------------------------------------------------
# Auth + helpers
# ---------------------------------------------------------------------------

def _user_id(request: Request) -> str:
    """Return the user id from the Bearer JWT, or raise 401."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_jwt_token(auth[7:])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token")
    return str(uid)


def _norm_hex(h: str) -> Optional[str]:
    s = str(h or "").strip()
    if not _HEX_RE.match(s):
        return None
    s = s.lstrip("#").upper()
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return "#" + s


def _clean_colors(colors: List["Color"]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for c in colors:
        hx = _norm_hex(c.hex)
        if not hx or hx in seen:
            continue
        seen.add(hx)
        out.append({"hex": hx, "name": (c.name or "").strip() or hx})
    return out


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Color(BaseModel):
    hex: str = Field(..., description="Color hex, e.g. '#FEE100'.")
    name: Optional[str] = Field(None, description="Optional color name.")


class PaletteIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80, description="Palette name, e.g. 'My Colors'.")
    colors: List[Color] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", tags=["14. User palettes"])
def list_palettes(request: Request):
    """List the current user's saved palettes (most recently updated first)."""
    uid = _user_id(request)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT id, name, colors, updated_at FROM user_palettes "
            "WHERE user_id = %s ORDER BY updated_at DESC",
            (uid,),
        )
        rows = cur.fetchall()
    return {"palettes": [
        {"id": r["id"], "name": r["name"], "colors": r["colors"] or []}
        for r in rows
    ]}


@router.post("", tags=["14. User palettes"])
def save_palette(req: PaletteIn, request: Request):
    """Create or update (by name) one of the current user's palettes."""
    uid = _user_id(request)
    colors = _clean_colors(req.colors)
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Palette name is required.")
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            INSERT INTO user_palettes (user_id, name, colors, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (user_id, name)
            DO UPDATE SET colors = EXCLUDED.colors, updated_at = NOW()
            RETURNING id, name, colors
            """,
            (uid, name, json.dumps(colors)),
        )
        row = cur.fetchone()
        conn.commit()
    return {"id": row["id"], "name": row["name"], "colors": row["colors"] or []}


@router.delete("/{palette_id}", tags=["14. User palettes"])
def delete_palette(palette_id: int, request: Request):
    """Delete one of the current user's palettes by id."""
    uid = _user_id(request)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM user_palettes WHERE id = %s AND user_id = %s",
            (palette_id, uid),
        )
        deleted = cur.rowcount
        conn.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="Palette not found.")
    return {"deleted": True, "id": palette_id}


__all__ = ["router", "init_palettes_tables"]
