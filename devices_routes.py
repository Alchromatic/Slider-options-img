#!/usr/bin/env python3
"""
devices_routes.py
=================

Device <-> user linkage for the Alchroma mobile / tablet / Meta Quest apps
(same idea as the modiqom device linkage): every device that talks to this
backend is paired to an ``auth_users`` row, and everything the device captures
(picked colors, unmix recipes) is stored under that user so the webapp can show
it.  Same Supabase database / JWT scheme as auth_db.py, billing.py and
palettes_routes.py.

Tables
------
    user_devices        one row per (device_id, user_id) pairing
    device_pair_codes   short-lived pairing codes (both flows below)
    device_captures     colors captured on a device, newest first

Two pairing flows (both end with the device holding a *device token*: a JWT
with the user's ``sub`` plus a ``did`` claim, valid 90 days, revocable from the
webapp Devices page):

  A. Device shows a code (headset / projector, no typing on the device)
       app  POST /api/devices/pair/start      -> {code, poll_secret}
       web  POST /api/devices/pair/claim      {code}          (Bearer user JWT)
       app  GET  /api/devices/pair/status     ?code&poll_secret -> access_token

  B. Web shows a code / QR / deep link (phone or tablet)
       web  POST /api/devices/pair/web-code   -> {code, deep_link}  (Bearer user JWT)
       app  POST /api/devices/pair/redeem     {code, device...} -> access_token

  C. App login with email + password (same /api/auth/login as the webapp)
       app  POST /api/devices/register        (Bearer user JWT) -> device token

Captures
--------
    POST   /api/devices/captures         save one color (device or user token)
    POST   /api/devices/captures/batch   save several (offline queue flush)
    GET    /api/devices/captures         list the user's captures
    DELETE /api/devices/captures/{id}

Every capture is also merged into the user's "My Colors" palette
(``user_palettes``) so it shows up in the Paint Collection / unmixer dropdown
without any extra step.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from typing import Any, Dict, List, Optional

import jwt
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from auth_db import RealDictCursor, decode_jwt_token, get_db

router = APIRouter(prefix="/api/devices", tags=["15. Devices"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAIR_CODE_TTL_SECONDS = int(os.getenv("DEVICE_PAIR_CODE_TTL", "600"))       # 10 min
DEVICE_TOKEN_TTL_SECONDS = int(os.getenv("DEVICE_TOKEN_TTL", str(86400 * 90)))  # 90 days
PAIR_POLL_INTERVAL_SECONDS = 3
ONLINE_WINDOW_SECONDS = 120          # last_seen within this => "online"
MY_COLORS = "My Colors"              # must match color-library.js / palettes_routes.py
MY_COLORS_CAP = 500
DEEP_LINK_SCHEME = os.getenv("APP_DEEP_LINK_SCHEME", "alchroma")

DEVICE_TYPES = {"phone", "tablet", "headset", "projector", "desktop", "other"}
PLATFORMS = {"android", "ios", "quest", "web", "windows", "macos", "other"}
SOURCES = {"camera", "unmix", "picker", "projector", "web_camera", "manual", "other"}

_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$")
# No 0/O/1/I so the code is unambiguous when read off a screen.
_LINK_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


# ---------------------------------------------------------------------------
# Table init (called at app startup from main.py)
# ---------------------------------------------------------------------------

def init_devices_tables() -> None:
    """Create the device tables if they don't exist."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_devices (
                    id            SERIAL PRIMARY KEY,
                    device_id     TEXT NOT NULL,
                    user_id       TEXT NOT NULL,
                    workspace_id  TEXT,
                    device_type   TEXT NOT NULL DEFAULT 'phone',
                    platform      TEXT,
                    name          TEXT,
                    app_version   TEXT,
                    paired_at     TIMESTAMPTZ DEFAULT NOW(),
                    last_seen_at  TIMESTAMPTZ DEFAULT NOW(),
                    revoked_at    TIMESTAMPTZ,
                    UNIQUE (device_id, user_id)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_devices_user ON user_devices(user_id)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS device_pair_codes (
                    id            SERIAL PRIMARY KEY,
                    code          TEXT NOT NULL,
                    flow          TEXT NOT NULL,
                    device_id     TEXT,
                    device_type   TEXT,
                    platform      TEXT,
                    name          TEXT,
                    app_version   TEXT,
                    user_id       TEXT,
                    poll_secret   TEXT,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    created_at    TIMESTAMPTZ DEFAULT NOW(),
                    expires_at    TIMESTAMPTZ NOT NULL,
                    consumed_at   TIMESTAMPTZ
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_device_pair_codes_code ON device_pair_codes(code)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS device_captures (
                    id            SERIAL PRIMARY KEY,
                    user_id       TEXT NOT NULL,
                    workspace_id  TEXT,
                    device_id     TEXT,
                    device_type   TEXT,
                    hex           TEXT NOT NULL,
                    name          TEXT,
                    source        TEXT,
                    recipe        JSONB,
                    meta          JSONB,
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_device_captures_user "
                "ON device_captures(user_id, created_at DESC)"
            )
            conn.commit()
            print("[OK] device tables ready")
    except Exception as e:  # pragma: no cover - depends on live DB
        print(f"[WARN] device tables init error: {e}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _norm_hex(h: Optional[str]) -> Optional[str]:
    s = str(h or "").strip()
    if not _HEX_RE.match(s):
        return None
    s = s.lstrip("#").upper()
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return "#" + s


def _hex_to_rgb(hx: str) -> str:
    h = hx.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


def _clean_enum(value: Optional[str], allowed: set, default: str) -> str:
    v = (value or "").strip().lower()
    return v if v in allowed else default


def _iso(ts) -> Optional[str]:
    return ts.isoformat() if ts else None


def _numeric_code() -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(6))


def _link_code() -> str:
    return "".join(secrets.choice(_LINK_ALPHABET) for _ in range(8))


def _user_summary(cur, user_id: str) -> Dict[str, Any]:
    cur.execute(
        "SELECT id, email, name, organization_name, workspace_id FROM auth_users WHERE id = %s",
        (user_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(row)


def _device_out(row: Dict[str, Any]) -> Dict[str, Any]:
    last = row.get("last_seen_at")
    online = False
    if last is not None:
        try:
            online = (time.time() - last.timestamp()) < ONLINE_WINDOW_SECONDS
        except Exception:
            online = False
    return {
        "id": row["id"],
        "device_id": row["device_id"],
        "device_type": row.get("device_type"),
        "platform": row.get("platform"),
        "name": row.get("name"),
        "app_version": row.get("app_version"),
        "paired_at": _iso(row.get("paired_at")),
        "last_seen_at": _iso(last),
        "online": online,
    }


def _capture_out(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "hex": row["hex"],
        "rgb": _hex_to_rgb(row["hex"]),
        "name": row.get("name") or row["hex"],
        "source": row.get("source"),
        "device_id": row.get("device_id"),
        "device_type": row.get("device_type"),
        "device_name": row.get("device_name"),
        "recipe": row.get("recipe") or [],
        "meta": row.get("meta") or {},
        "created_at": _iso(row.get("created_at")),
    }


# ---------------------------------------------------------------------------
# Auth: user tokens (webapp) and device tokens (apps)
# ---------------------------------------------------------------------------

def _device_token(user_id: str, email: str, device_id: str) -> str:
    """Long-lived JWT bound to one device. Same secret/algorithm as auth_db."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email,
        "did": device_id,
        "typ": "device",
        "iat": now,
        "exp": now + DEVICE_TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, os.getenv("JWT_SECRET", ""), algorithm="HS256")


def _auth(request: Request, allow_device: bool = True) -> Dict[str, Any]:
    """Return {"user_id", "email", "device_id"} from the Bearer token.

    A device token (``did`` claim) is rejected if the pairing was revoked from
    the webapp, so "Unpair" takes effect on the device's next request.
    """
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_jwt_token(header[7:])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token")
    ctx = {"user_id": str(uid), "email": payload.get("email"), "device_id": None}
    did = payload.get("did")
    if did:
        if not allow_device:
            raise HTTPException(status_code=403, detail="A user login is required for this action")
        with get_db() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT revoked_at FROM user_devices WHERE device_id = %s AND user_id = %s",
                (did, ctx["user_id"]),
            )
            row = cur.fetchone()
        if not row or row["revoked_at"] is not None:
            raise HTTPException(status_code=401, detail="device_revoked")
        ctx["device_id"] = str(did)
    return ctx


# ---------------------------------------------------------------------------
# Brute-force throttle for the code endpoints (per process, best effort)
# ---------------------------------------------------------------------------

_ATTEMPTS: Dict[str, List[float]] = {}
_ATTEMPT_LIMIT = 10
_ATTEMPT_WINDOW = 600.0


def _throttle(key: str) -> None:
    now = time.time()
    hits = [t for t in _ATTEMPTS.get(key, []) if now - t < _ATTEMPT_WINDOW]
    if len(hits) >= _ATTEMPT_LIMIT:
        _ATTEMPTS[key] = hits
        raise HTTPException(status_code=429, detail="Too many attempts, try again in a few minutes")
    _ATTEMPTS[key] = hits


def _record_failure(key: str) -> None:
    _ATTEMPTS.setdefault(key, []).append(time.time())


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DeviceInfo(BaseModel):
    device_id: str = Field(..., min_length=4, max_length=128, description="Stable per-install id.")
    device_type: Optional[str] = Field("phone", description="phone | tablet | headset | projector | desktop | other")
    platform: Optional[str] = Field(None, description="android | ios | quest | web | windows | macos | other")
    name: Optional[str] = Field(None, max_length=80)
    app_version: Optional[str] = Field(None, max_length=40)


class PairClaim(BaseModel):
    code: str = Field(..., min_length=4, max_length=16)


class PairRedeem(DeviceInfo):
    code: str = Field(..., min_length=4, max_length=16)


class DeviceRename(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class RecipeItem(BaseModel):
    name: Optional[str] = None
    hex: Optional[str] = None
    percentage: Optional[float] = None
    parts: Optional[float] = None


class CaptureIn(BaseModel):
    hex: str = Field(..., description="Captured color, e.g. '#1E2448'.")
    name: Optional[str] = Field(None, max_length=120)
    source: Optional[str] = Field("camera", description="camera | unmix | picker | projector | web_camera | manual | other")
    recipe: Optional[List[RecipeItem]] = None
    meta: Optional[Dict[str, Any]] = None
    device_id: Optional[str] = Field(None, description="Only needed with a user token; device tokens carry it.")
    add_to_library: bool = Field(True, description="Also merge into the user's 'My Colors' palette.")


class CaptureBatch(BaseModel):
    captures: List[CaptureIn] = Field(..., min_length=1, max_length=200)


# ---------------------------------------------------------------------------
# Device registry internals
# ---------------------------------------------------------------------------

def _upsert_device(cur, user_id: str, workspace_id: Optional[str], info: DeviceInfo) -> Dict[str, Any]:
    """Create or refresh the pairing row and return it (un-revokes on re-pair)."""
    cur.execute(
        """
        INSERT INTO user_devices
            (device_id, user_id, workspace_id, device_type, platform, name, app_version, paired_at, last_seen_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (device_id, user_id) DO UPDATE SET
            workspace_id = COALESCE(EXCLUDED.workspace_id, user_devices.workspace_id),
            device_type  = EXCLUDED.device_type,
            platform     = COALESCE(EXCLUDED.platform, user_devices.platform),
            name         = COALESCE(EXCLUDED.name, user_devices.name),
            app_version  = COALESCE(EXCLUDED.app_version, user_devices.app_version),
            last_seen_at = NOW(),
            paired_at    = CASE WHEN user_devices.revoked_at IS NULL THEN user_devices.paired_at ELSE NOW() END,
            revoked_at   = NULL
        RETURNING *
        """,
        (
            info.device_id.strip(),
            user_id,
            workspace_id,
            _clean_enum(info.device_type, DEVICE_TYPES, "phone"),
            _clean_enum(info.platform, PLATFORMS, "other") if info.platform else None,
            (info.name or "").strip() or None,
            (info.app_version or "").strip() or None,
        ),
    )
    return dict(cur.fetchone())


def _session_payload(cur, user_id: str, device_row: Dict[str, Any]) -> Dict[str, Any]:
    user = _user_summary(cur, user_id)
    return {
        "access_token": _device_token(user["id"], user["email"], device_row["device_id"]),
        "token_type": "device",
        "expires_in": DEVICE_TOKEN_TTL_SECONDS,
        "user": user,
        "device": _device_out(device_row),
    }


def _touch_device(cur, user_id: str, device_id: Optional[str]) -> None:
    if device_id:
        cur.execute(
            "UPDATE user_devices SET last_seen_at = NOW() WHERE device_id = %s AND user_id = %s",
            (device_id, user_id),
        )


def _pending_code(cur, code: str, flow: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        SELECT * FROM device_pair_codes
        WHERE code = %s AND flow = %s AND status = 'pending' AND expires_at > NOW()
        ORDER BY created_at DESC LIMIT 1
        """,
        (code, flow),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _new_code(cur, flow: str, generator) -> str:
    """Generate a code that is not currently pending for this flow."""
    cur.execute("DELETE FROM device_pair_codes WHERE expires_at < NOW() - INTERVAL '1 day'")
    for _ in range(20):
        code = generator()
        if not _pending_code(cur, code, flow):
            return code
    raise HTTPException(status_code=500, detail="Could not allocate a pairing code")


# ---------------------------------------------------------------------------
# Public config (no auth) — app download links etc. for the Devices page
# ---------------------------------------------------------------------------

@router.get("/config")
def devices_config():
    """Public: deep-link scheme and app download links (set via env vars)."""
    return {
        "deep_link_scheme": DEEP_LINK_SCHEME,
        "pair_code_ttl": PAIR_CODE_TTL_SECONDS,
        "poll_interval": PAIR_POLL_INTERVAL_SECONDS,
        "apps": {
            "android": os.getenv("APP_ANDROID_URL", ""),
            "ios": os.getenv("APP_IOS_URL", ""),
            "quest": os.getenv("APP_QUEST_URL", ""),
        },
    }


# ---------------------------------------------------------------------------
# Flow A — device shows a code, user types it in the webapp
# ---------------------------------------------------------------------------

@router.post("/pair/start")
def pair_start(info: DeviceInfo, request: Request):
    """App: begin code pairing. Show ``code`` on the device and poll ``/pair/status``."""
    _throttle("start:" + _client_ip(request))
    poll_secret = secrets.token_hex(24)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        code = _new_code(cur, "device", _numeric_code)
        cur.execute(
            """
            INSERT INTO device_pair_codes
                (code, flow, device_id, device_type, platform, name, app_version, poll_secret, expires_at)
            VALUES (%s, 'device', %s, %s, %s, %s, %s, %s, NOW() + (%s || ' seconds')::interval)
            """,
            (
                code,
                info.device_id.strip(),
                _clean_enum(info.device_type, DEVICE_TYPES, "headset"),
                _clean_enum(info.platform, PLATFORMS, "other") if info.platform else None,
                (info.name or "").strip() or None,
                (info.app_version or "").strip() or None,
                poll_secret,
                str(PAIR_CODE_TTL_SECONDS),
            ),
        )
        conn.commit()
    return {
        "code": code,
        "poll_secret": poll_secret,
        "expires_in": PAIR_CODE_TTL_SECONDS,
        "interval": PAIR_POLL_INTERVAL_SECONDS,
    }


@router.get("/pair/status")
def pair_status(code: str, poll_secret: str):
    """App: poll until the webapp claims the code; returns the device session once."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT * FROM device_pair_codes
            WHERE code = %s AND flow = 'device' AND poll_secret = %s
            ORDER BY created_at DESC LIMIT 1
            """,
            (code.strip(), poll_secret.strip()),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown pairing code")
        row = dict(row)
        if row["status"] == "consumed":
            return {"status": "consumed"}
        if row["status"] == "pending":
            cur.execute("SELECT NOW() > %s AS expired", (row["expires_at"],))
            if cur.fetchone()["expired"]:
                return {"status": "expired"}
            return {"status": "pending", "interval": PAIR_POLL_INTERVAL_SECONDS}
        # claimed -> hand over the device session exactly once
        cur.execute(
            "UPDATE device_pair_codes SET status = 'consumed', consumed_at = NOW() WHERE id = %s AND status = 'claimed'",
            (row["id"],),
        )
        if cur.rowcount != 1:
            conn.rollback()
            return {"status": "consumed"}
        cur.execute(
            "SELECT * FROM user_devices WHERE device_id = %s AND user_id = %s",
            (row["device_id"], row["user_id"]),
        )
        dev = cur.fetchone()
        if not dev:
            conn.rollback()
            raise HTTPException(status_code=409, detail="Pairing was revoked before the device collected it")
        payload = _session_payload(cur, row["user_id"], dict(dev))
        conn.commit()
    payload["status"] = "claimed"
    return payload


@router.post("/pair/claim")
def pair_claim(req: PairClaim, request: Request):
    """Web (logged in): enter the code shown on the device to link it to this account."""
    ctx = _auth(request, allow_device=False)
    key = "claim:" + ctx["user_id"]
    _throttle(key)
    code = re.sub(r"\D", "", req.code)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _pending_code(cur, code, "device")
        if not row:
            _record_failure(key)
            raise HTTPException(status_code=404, detail="Code not found or expired — ask the device for a new one")
        user = _user_summary(cur, ctx["user_id"])
        info = DeviceInfo(
            device_id=row["device_id"],
            device_type=row.get("device_type") or "headset",
            platform=row.get("platform"),
            name=row.get("name"),
            app_version=row.get("app_version"),
        )
        dev = _upsert_device(cur, user["id"], user.get("workspace_id"), info)
        cur.execute(
            "UPDATE device_pair_codes SET status = 'claimed', user_id = %s WHERE id = %s",
            (user["id"], row["id"]),
        )
        conn.commit()
    return {"paired": True, "device": _device_out(dev)}


# ---------------------------------------------------------------------------
# Flow B — webapp shows a code / QR / deep link, the app redeems it
# ---------------------------------------------------------------------------

@router.post("/pair/web-code")
def pair_web_code(request: Request):
    """Web (logged in): get a code + deep link for a phone/tablet to redeem."""
    ctx = _auth(request, allow_device=False)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # One live web code per user at a time keeps the page simple.
        cur.execute(
            "UPDATE device_pair_codes SET status = 'expired' WHERE flow = 'web' AND user_id = %s AND status = 'pending'",
            (ctx["user_id"],),
        )
        code = _new_code(cur, "web", _link_code)
        cur.execute(
            """
            INSERT INTO device_pair_codes (code, flow, user_id, expires_at)
            VALUES (%s, 'web', %s, NOW() + (%s || ' seconds')::interval)
            """,
            (code, ctx["user_id"], str(PAIR_CODE_TTL_SECONDS)),
        )
        conn.commit()
    return {
        "code": code,
        "deep_link": f"{DEEP_LINK_SCHEME}://pair?code={code}",
        "expires_in": PAIR_CODE_TTL_SECONDS,
    }


@router.post("/pair/redeem")
def pair_redeem(req: PairRedeem, request: Request):
    """App: redeem a code from the webapp (typed, scanned or deep-linked)."""
    key = "redeem:" + _client_ip(request)
    _throttle(key)
    code = re.sub(r"[^A-Za-z0-9]", "", req.code).upper()
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        row = _pending_code(cur, code, "web")
        if not row:
            _record_failure(key)
            raise HTTPException(status_code=404, detail="Code not found or expired — get a new one from the Devices page")
        user = _user_summary(cur, row["user_id"])
        dev = _upsert_device(cur, user["id"], user.get("workspace_id"), req)
        cur.execute(
            "UPDATE device_pair_codes SET status = 'consumed', consumed_at = NOW(), device_id = %s WHERE id = %s",
            (req.device_id.strip(), row["id"]),
        )
        payload = _session_payload(cur, user["id"], dev)
        conn.commit()
    return payload


# ---------------------------------------------------------------------------
# Flow C / registry — logged-in app registers itself; webapp manages the list
# ---------------------------------------------------------------------------

@router.post("/register")
def register_device(info: DeviceInfo, request: Request):
    """App (Bearer user *or* device token): register/refresh this device and get a device token."""
    ctx = _auth(request)
    if ctx["device_id"] and ctx["device_id"] != info.device_id.strip():
        raise HTTPException(status_code=403, detail="Device token does not match device_id")
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        user = _user_summary(cur, ctx["user_id"])
        dev = _upsert_device(cur, user["id"], user.get("workspace_id"), info)
        payload = _session_payload(cur, user["id"], dev)
        conn.commit()
    return payload


@router.post("/heartbeat")
def heartbeat(request: Request):
    """App: mark the device as seen (call every minute or so while open)."""
    ctx = _auth(request)
    if not ctx["device_id"]:
        raise HTTPException(status_code=400, detail="Heartbeat needs a device token")
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "UPDATE user_devices SET last_seen_at = NOW() WHERE device_id = %s AND user_id = %s RETURNING *",
            (ctx["device_id"], ctx["user_id"]),
        )
        row = cur.fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=401, detail="device_revoked")
    return {"ok": True, "device": _device_out(dict(row))}


@router.get("")
def list_devices(request: Request):
    """List the current user's paired (not revoked) devices."""
    ctx = _auth(request)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM user_devices WHERE user_id = %s AND revoked_at IS NULL ORDER BY last_seen_at DESC",
            (ctx["user_id"],),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"devices": [_device_out(r) for r in rows]}


@router.patch("/{device_row_id}")
def rename_device(device_row_id: int, req: DeviceRename, request: Request):
    ctx = _auth(request, allow_device=False)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "UPDATE user_devices SET name = %s WHERE id = %s AND user_id = %s RETURNING *",
            (req.name.strip(), device_row_id, ctx["user_id"]),
        )
        row = cur.fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"device": _device_out(dict(row))}


@router.delete("/{device_row_id}")
def revoke_device(device_row_id: int, request: Request):
    """Unpair: the device's token stops working on its next request."""
    ctx = _auth(request, allow_device=False)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE user_devices SET revoked_at = NOW() WHERE id = %s AND user_id = %s AND revoked_at IS NULL",
            (device_row_id, ctx["user_id"]),
        )
        n = cur.rowcount
        conn.commit()
    if not n:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"revoked": True, "id": device_row_id}


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------

def _merge_into_my_colors(cur, user_id: str, colors: List[Dict[str, str]]) -> None:
    """Prepend new colors to the user's 'My Colors' palette (dedupe by hex)."""
    if not colors:
        return
    cur.execute(
        "SELECT colors FROM user_palettes WHERE user_id = %s AND name = %s FOR UPDATE",
        (user_id, MY_COLORS),
    )
    row = cur.fetchone()
    existing = list(row["colors"] or []) if row else []
    seen = {str(c.get("hex", "")).upper() for c in existing}
    added = []
    for c in colors:
        if c["hex"] in seen:
            continue
        seen.add(c["hex"])
        added.append({"hex": c["hex"], "name": c["name"]})
    if not added:
        return
    merged = (added + existing)[:MY_COLORS_CAP]
    cur.execute(
        """
        INSERT INTO user_palettes (user_id, name, colors, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (user_id, name)
        DO UPDATE SET colors = EXCLUDED.colors, updated_at = NOW()
        """,
        (user_id, MY_COLORS, json.dumps(merged)),
    )


def _insert_captures(cur, ctx: Dict[str, Any], items: List[CaptureIn]) -> List[Dict[str, Any]]:
    user = _user_summary(cur, ctx["user_id"])
    out: List[Dict[str, Any]] = []
    library: List[Dict[str, str]] = []
    for item in items:
        hx = _norm_hex(item.hex)
        if not hx:
            raise HTTPException(status_code=400, detail=f"Invalid hex color: {item.hex!r}")
        device_id = ctx["device_id"] or (item.device_id or "").strip() or None
        device_type = None
        device_name = None
        if device_id:
            cur.execute(
                "SELECT device_type, name FROM user_devices WHERE device_id = %s AND user_id = %s",
                (device_id, user["id"]),
            )
            d = cur.fetchone()
            if d:
                device_type, device_name = d["device_type"], d["name"]
        recipe = None
        if item.recipe:
            recipe = [
                {
                    "name": (r.name or "").strip() or None,
                    "hex": _norm_hex(r.hex) if r.hex else None,
                    "percentage": r.percentage,
                    "parts": r.parts,
                }
                for r in item.recipe
            ]
        name = (item.name or "").strip() or hx
        cur.execute(
            """
            INSERT INTO device_captures
                (user_id, workspace_id, device_id, device_type, hex, name, source, recipe, meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                user["id"],
                user.get("workspace_id"),
                device_id,
                device_type,
                hx,
                name,
                _clean_enum(item.source, SOURCES, "other"),
                json.dumps(recipe) if recipe is not None else None,
                json.dumps(item.meta) if item.meta is not None else None,
            ),
        )
        row = dict(cur.fetchone())
        row["device_name"] = device_name
        out.append(row)
        if item.add_to_library:
            library.append({"hex": hx, "name": name})
    _merge_into_my_colors(cur, user["id"], library)
    _touch_device(cur, user["id"], ctx["device_id"])
    return out


@router.post("/captures")
def save_capture(item: CaptureIn, request: Request):
    """Save one captured color under the current user (device or user token)."""
    ctx = _auth(request)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        rows = _insert_captures(cur, ctx, [item])
        conn.commit()
    return {"capture": _capture_out(rows[0])}


@router.post("/captures/batch")
def save_captures_batch(batch: CaptureBatch, request: Request):
    """Save several captures at once (used by the apps to flush an offline queue)."""
    ctx = _auth(request)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        rows = _insert_captures(cur, ctx, batch.captures)
        conn.commit()
    return {"captures": [_capture_out(r) for r in rows], "count": len(rows)}


@router.get("/captures")
def list_captures(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    device_id: Optional[str] = None,
    source: Optional[str] = None,
):
    """List the user's captures, newest first (device or user token)."""
    ctx = _auth(request)
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    where = ["c.user_id = %s"]
    params: List[Any] = [ctx["user_id"]]
    if device_id:
        where.append("c.device_id = %s")
        params.append(device_id)
    if source:
        where.append("c.source = %s")
        params.append(source)
    sql_where = " AND ".join(where)
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SELECT COUNT(*) AS n FROM device_captures c WHERE {sql_where}", params)
        total = cur.fetchone()["n"]
        cur.execute(
            f"""
            SELECT c.*, d.name AS device_name
            FROM device_captures c
            LEFT JOIN user_devices d ON d.device_id = c.device_id AND d.user_id = c.user_id
            WHERE {sql_where}
            ORDER BY c.created_at DESC, c.id DESC
            LIMIT %s OFFSET %s
            """,
            params + [limit, offset],
        )
        rows = [dict(r) for r in cur.fetchall()]
        _touch_device(cur, ctx["user_id"], ctx["device_id"])
        conn.commit()
    return {"captures": [_capture_out(r) for r in rows], "total": total, "limit": limit, "offset": offset}


@router.delete("/captures/{capture_id}")
def delete_capture(capture_id: int, request: Request):
    ctx = _auth(request)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM device_captures WHERE id = %s AND user_id = %s",
            (capture_id, ctx["user_id"]),
        )
        n = cur.rowcount
        conn.commit()
    if not n:
        raise HTTPException(status_code=404, detail="Capture not found")
    return {"deleted": True, "id": capture_id}


__all__ = ["router", "init_devices_tables"]
