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
    POST   /api/palettes            -> create or update a named palette (by name)
    DELETE /api/palettes/{id}       -> delete one of the user's palettes

A palette is ``{id, name, colors:[{hex, name}]}``.  Palettes are unique per
(user_id, name), so saving a named palette again just updates it.

"My Colors" is edited one color at a time (Bearer user JWT or device token; an
unpaired device gets ``401 device_revoked``), because the web Color Library,
the apps and device captures all change the same list:

    GET    /api/palettes/my-colors                -> {id, name, colors, updated_at}
    POST   /api/palettes/my-colors/colors         {hex, name?}   add one (no-op if present)
    PATCH  /api/palettes/my-colors/colors/{hex}   {name?, hex?}  rename / change hex
    DELETE /api/palettes/my-colors/colors/{hex}                  remove one
    POST   /api/palettes/my-colors/merge          {colors:[...]} add missing, never remove

Groups ("Studio Set", "Oils"...) are the user's other palettes; each one is its
own entry in the recipe palette dropdown. Same auth and rules as My Colors:

    GET    /api/palettes/groups                       -> {groups: [{id, name, colors, updated_at}]}
    POST   /api/palettes/groups                       {name, colors?}  create
    PATCH  /api/palettes/groups/{id}                  {name}           rename
    DELETE /api/palettes/groups/{id}                                   delete (colors stay in My Colors)
    POST   /api/palettes/groups/{id}/colors           {hex, name?}     add one (no-op if present)
    DELETE /api/palettes/groups/{id}/colors/{hex}                      remove one

``{hex}`` is the color without "#" (e.g. ``FEE100``).  Each change runs in one
transaction holding the row lock, so concurrent edits and captures don't
overwrite each other.  POST /api/palettes refuses the name "My Colors" (a whole
list replace would drop colors added elsewhere in the meantime).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from auth_db import RealDictCursor, decode_jwt_token, get_db
from devices_routes import MY_COLORS, MY_COLORS_CAP, _auth

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
    if name == MY_COLORS:
        raise HTTPException(
            status_code=409,
            detail='"My Colors" is edited one color at a time: use /api/palettes/my-colors.',
        )
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


# ---------------------------------------------------------------------------
# "My Colors": single-color edits
# ---------------------------------------------------------------------------

class ColorEdit(BaseModel):
    name: Optional[str] = Field(None, description="New name (blank = the hex).")
    hex: Optional[str] = Field(None, description="New hex, e.g. '#FEE100'.")


class ColorMerge(BaseModel):
    colors: List[Color] = Field(default_factory=list)


def _my_colors_out(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {"id": None, "name": MY_COLORS, "colors": [], "updated_at": None}
    ts = row.get("updated_at")
    return {
        "id": row["id"],
        "name": row["name"],
        "colors": row["colors"] or [],
        "updated_at": ts.isoformat() if ts else None,
    }


def _lock_my_colors(cur, uid: str) -> Dict[str, Any]:
    """Create the user's "My Colors" row if needed, then lock it until commit."""
    cur.execute(
        "INSERT INTO user_palettes (user_id, name, colors, updated_at) "
        "VALUES (%s, %s, '[]'::jsonb, NOW()) ON CONFLICT (user_id, name) DO NOTHING",
        (uid, MY_COLORS),
    )
    cur.execute(
        "SELECT id, name, colors, updated_at FROM user_palettes "
        "WHERE user_id = %s AND name = %s FOR UPDATE",
        (uid, MY_COLORS),
    )
    return dict(cur.fetchone())


def _store_my_colors(cur, row_id: int, colors: List[dict]) -> Dict[str, Any]:
    cur.execute(
        "UPDATE user_palettes SET colors = %s, updated_at = NOW() WHERE id = %s "
        "RETURNING id, name, colors, updated_at",
        (json.dumps(colors), row_id),
    )
    return dict(cur.fetchone())


def _hex_or_400(value: Optional[str]) -> str:
    hx = _norm_hex(value or "")
    if not hx:
        raise HTTPException(status_code=400, detail=f"Invalid hex color: {value!r}")
    return hx


def _index_of(colors: List[dict], hx: str) -> Optional[int]:
    for i, c in enumerate(colors):
        if str(c.get("hex", "")).upper() == hx:
            return i
    return None


@router.get("/my-colors")
def get_my_colors(request: Request):
    """The user's "My Colors" (``id: null`` and no colors until the first one is added)."""
    uid = _auth(request)["user_id"]
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT id, name, colors, updated_at FROM user_palettes WHERE user_id = %s AND name = %s",
            (uid, MY_COLORS),
        )
        row = cur.fetchone()
    return _my_colors_out(dict(row) if row else None)


@router.post("/my-colors/colors")
def add_my_color(req: Color, request: Request):
    """Add one color to the end of "My Colors". Nothing changes if the hex is already there."""
    uid = _auth(request)["user_id"]
    hx = _hex_or_400(req.hex)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _lock_my_colors(cur, uid)
        colors = list(row["colors"] or [])
        if _index_of(colors, hx) is None:
            if len(colors) >= MY_COLORS_CAP:
                raise HTTPException(status_code=409, detail=f"My Colors is full ({MY_COLORS_CAP} colors).")
            colors.append({"hex": hx, "name": (req.name or "").strip() or hx})
            row = _store_my_colors(cur, row["id"], colors)
        conn.commit()
    return _my_colors_out(row)


@router.patch("/my-colors/colors/{hex}")
def edit_my_color(hex: str, req: ColorEdit, request: Request):
    """Rename a color and/or change its hex. 404 if it isn't there, 409 if the new hex is."""
    uid = _auth(request)["user_id"]
    old = _hex_or_400(hex)
    new = _hex_or_400(req.hex) if req.hex is not None else old
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _lock_my_colors(cur, uid)
        colors = list(row["colors"] or [])
        i = _index_of(colors, old)
        if i is None:
            raise HTTPException(status_code=404, detail="That color is not in My Colors.")
        if new != old and _index_of(colors, new) is not None:
            raise HTTPException(status_code=409, detail="That color is already in My Colors.")
        name = str(colors[i].get("name") or "").strip() or old
        if req.name is not None:
            name = req.name.strip() or new
        elif new != old and name.upper() == old:
            name = new                     # a name that was just the hex follows the hex
        if colors[i] != {"hex": new, "name": name}:
            colors[i] = {"hex": new, "name": name}
            row = _store_my_colors(cur, row["id"], colors)
        conn.commit()
    return _my_colors_out(row)


@router.delete("/my-colors/colors/{hex}")
def delete_my_color(hex: str, request: Request):
    """Remove one color. 404 if it isn't there."""
    uid = _auth(request)["user_id"]
    hx = _hex_or_400(hex)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _lock_my_colors(cur, uid)
        colors = list(row["colors"] or [])
        i = _index_of(colors, hx)
        if i is None:
            raise HTTPException(status_code=404, detail="That color is not in My Colors.")
        del colors[i]
        row = _store_my_colors(cur, row["id"], colors)
        conn.commit()
    return _my_colors_out(row)


@router.post("/my-colors/merge")
def merge_my_colors(req: ColorMerge, request: Request):
    """Append the colors that aren't in "My Colors" yet (up to the cap); never removes any.

    Used once to move an old browser list into the account, and by the apps to
    send colors added while offline. Invalid hexes and duplicates are skipped.
    """
    uid = _auth(request)["user_id"]
    incoming = _clean_colors(req.colors)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _lock_my_colors(cur, uid)
        colors = list(row["colors"] or [])
        have = {str(c.get("hex", "")).upper() for c in colors}
        added = [c for c in incoming if c["hex"] not in have][: max(0, MY_COLORS_CAP - len(colors))]
        if added:
            row = _store_my_colors(cur, row["id"], colors + added)
        conn.commit()
    return _my_colors_out(row)


# ---------------------------------------------------------------------------
# Groups: named palettes other than "My Colors"
# ---------------------------------------------------------------------------

class GroupIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    colors: List[Color] = Field(default_factory=list)


class GroupRename(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


def _group_name(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="A group needs a name.")
    if name.lower() == MY_COLORS.lower():
        raise HTTPException(status_code=409, detail='"My Colors" is the library itself; pick another name.')
    return name


def _lock_group(cur, uid: str, group_id: int) -> Dict[str, Any]:
    cur.execute(
        "SELECT id, name, colors, updated_at FROM user_palettes "
        "WHERE id = %s AND user_id = %s AND name <> %s FOR UPDATE",
        (group_id, uid, MY_COLORS),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Group not found.")
    return dict(row)


def _name_taken(cur, uid: str, name: str, except_id: Optional[int] = None) -> bool:
    cur.execute(
        "SELECT 1 FROM user_palettes WHERE user_id = %s AND lower(name) = lower(%s) AND id <> %s",
        (uid, name, except_id or -1),
    )
    return cur.fetchone() is not None


@router.get("/groups")
def list_groups(request: Request):
    """Every palette of the user except "My Colors", newest first."""
    uid = _auth(request)["user_id"]
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT id, name, colors, updated_at FROM user_palettes "
            "WHERE user_id = %s AND name <> %s ORDER BY id DESC",
            (uid, MY_COLORS),
        )
        rows = cur.fetchall()
    return {"groups": [_my_colors_out(dict(r)) for r in rows]}


@router.post("/groups")
def create_group(req: GroupIn, request: Request):
    uid = _auth(request)["user_id"]
    name = _group_name(req.name)
    colors = _clean_colors(req.colors)[:MY_COLORS_CAP]
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if _name_taken(cur, uid, name):
            raise HTTPException(status_code=409, detail="You already have a group with that name.")
        cur.execute(
            "INSERT INTO user_palettes (user_id, name, colors, updated_at) VALUES (%s, %s, %s, NOW()) "
            "ON CONFLICT (user_id, name) DO NOTHING RETURNING id, name, colors, updated_at",
            (uid, name, json.dumps(colors)),
        )
        row = cur.fetchone()
        if not row:                        # created by a parallel request a moment ago
            raise HTTPException(status_code=409, detail="You already have a group with that name.")
        conn.commit()
    return _my_colors_out(dict(row))


@router.patch("/groups/{group_id}")
def rename_group(group_id: int, req: GroupRename, request: Request):
    uid = _auth(request)["user_id"]
    name = _group_name(req.name)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _lock_group(cur, uid, group_id)
        if _name_taken(cur, uid, name, except_id=group_id):
            raise HTTPException(status_code=409, detail="You already have a group with that name.")
        if row["name"] != name:
            cur.execute(
                "UPDATE user_palettes SET name = %s, updated_at = NOW() WHERE id = %s "
                "RETURNING id, name, colors, updated_at",
                (name, group_id),
            )
            row = dict(cur.fetchone())
        conn.commit()
    return _my_colors_out(row)


@router.delete("/groups/{group_id}")
def delete_group(group_id: int, request: Request):
    """Delete a group. Its colors stay in My Colors."""
    uid = _auth(request)["user_id"]
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        _lock_group(cur, uid, group_id)
        cur.execute("DELETE FROM user_palettes WHERE id = %s", (group_id,))
        conn.commit()
    return {"deleted": True, "id": group_id}


@router.post("/groups/{group_id}/colors")
def add_group_color(group_id: int, req: Color, request: Request):
    uid = _auth(request)["user_id"]
    hx = _hex_or_400(req.hex)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _lock_group(cur, uid, group_id)
        colors = list(row["colors"] or [])
        if _index_of(colors, hx) is None:
            if len(colors) >= MY_COLORS_CAP:
                raise HTTPException(status_code=409, detail=f"This group is full ({MY_COLORS_CAP} colors).")
            colors.append({"hex": hx, "name": (req.name or "").strip() or hx})
            row = _store_my_colors(cur, group_id, colors)
        conn.commit()
    return _my_colors_out(row)


@router.delete("/groups/{group_id}/colors/{hex}")
def delete_group_color(group_id: int, hex: str, request: Request):
    uid = _auth(request)["user_id"]
    hx = _hex_or_400(hex)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _lock_group(cur, uid, group_id)
        colors = list(row["colors"] or [])
        i = _index_of(colors, hx)
        if i is None:
            raise HTTPException(status_code=404, detail="That color is not in this group.")
        del colors[i]
        row = _store_my_colors(cur, group_id, colors)
        conn.commit()
    return _my_colors_out(row)


__all__ = ["router", "init_palettes_tables"]
