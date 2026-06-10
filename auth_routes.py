"""
Auth routes — login / sign up / current-user / OAuth (Google, Facebook).

Mirrors the /api/auth/* endpoints from the
sunnysanwar_integrated_multi_model_cmprxn_role project and writes to the same
`auth_users` table, so accounts are shared between the two apps.
"""

import os
import uuid
import secrets
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from auth_db import (
    get_db,
    RealDictCursor,
    hash_password,
    verify_password,
    generate_jwt_token,
    decode_jwt_token,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])

# ─── OAuth configuration ────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID", "")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET", "")
OAUTH_REDIRECT_BASE = os.getenv("OAUTH_REDIRECT_BASE", "https://alchromaticdemo.up.railway.app")
FRONTEND_URL = os.getenv("FRONTEND_URL", "")


class AuthRegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    organization_name: Optional[str] = None


class AuthLoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
async def auth_register(req: AuthRegisterRequest):
    """Register a new user."""
    try:
        with get_db() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id FROM auth_users WHERE email = %s", (req.email,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")

            user_id = str(uuid.uuid4())
            workspace_id = str(uuid.uuid4())
            pw_hash, _ = hash_password(req.password)

            cursor.execute(
                """
                INSERT INTO auth_users (id, email, password_hash, name, organization_name, workspace_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, req.email, pw_hash, req.name, req.organization_name, workspace_id),
            )
            conn.commit()

            token = generate_jwt_token(user_id, req.email)
            user_data = {
                "id": user_id,
                "email": req.email,
                "name": req.name,
                "organization_name": req.organization_name,
                "workspace_id": workspace_id,
            }
            return {
                "access_token": token,
                "refresh_token": token,
                "user": user_data,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
async def auth_login(req: AuthLoginRequest):
    """Login with email/password."""
    try:
        with get_db() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM auth_users WHERE email = %s", (req.email,))
            user = cursor.fetchone()

            if not user or not verify_password(req.password, user["password_hash"]):
                raise HTTPException(status_code=401, detail="Invalid email or password")

            token = generate_jwt_token(user["id"], user["email"])
            user_data = {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "organization_name": user["organization_name"],
                "workspace_id": user["workspace_id"],
            }
            return {
                "access_token": token,
                "refresh_token": token,
                "user": user_data,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def auth_me(request: Request):
    """Get the current user from a Bearer JWT."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = auth_header[7:]
    try:
        payload = decode_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        with get_db() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT id, email, name, organization_name, workspace_id FROM auth_users WHERE id = %s",
                (payload["sub"],),
            )
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return {
                "user": {
                    "id": user["id"],
                    "email": user["email"],
                    "name": user["name"],
                    "organization_name": user["organization_name"],
                    "workspace_id": user["workspace_id"],
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── OAuth helpers ───────────────────────────────────────────────────────────

def _oauth_upsert_user(email: str, name: Optional[str] = None) -> dict:
    """Find or create a user by email (for OAuth logins). Returns token + user."""
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM auth_users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            token = generate_jwt_token(user["id"], user["email"])
            return {
                "access_token": token,
                "refresh_token": token,
                "user": {
                    "id": user["id"],
                    "email": user["email"],
                    "name": user["name"],
                    "organization_name": user["organization_name"],
                    "workspace_id": user["workspace_id"],
                },
            }

        user_id = str(uuid.uuid4())
        workspace_id = str(uuid.uuid4())
        pw_hash, _ = hash_password(secrets.token_hex(32))

        cursor.execute(
            """
            INSERT INTO auth_users (id, email, password_hash, name, workspace_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, email, pw_hash, name, workspace_id),
        )
        conn.commit()

        token = generate_jwt_token(user_id, email)
        return {
            "access_token": token,
            "refresh_token": token,
            "user": {
                "id": user_id,
                "email": email,
                "name": name,
                "organization_name": None,
                "workspace_id": workspace_id,
            },
        }


# ─── Google OAuth ────────────────────────────────────────────────────────────

@router.get("/oauth/google")
async def oauth_google_redirect():
    """Redirect user to Google's OAuth consent screen."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": f"{OAUTH_REDIRECT_BASE}/api/auth/oauth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}")


@router.get("/oauth/google/callback")
async def oauth_google_callback(code: str = "", error: str = ""):
    """Handle the callback from Google after user consent."""
    if error or not code:
        return RedirectResponse(f"{FRONTEND_URL}/signin.html?error=oauth_cancelled")

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": f"{OAUTH_REDIRECT_BASE}/api/auth/oauth/google/callback",
                "grant_type": "authorization_code",
            },
        )
        if token_res.status_code != 200:
            return RedirectResponse(f"{FRONTEND_URL}/signin.html?error=oauth_failed")
        tokens = token_res.json()

        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        if userinfo_res.status_code != 200:
            return RedirectResponse(f"{FRONTEND_URL}/signin.html?error=oauth_failed")
        userinfo = userinfo_res.json()

    email = userinfo.get("email")
    name = userinfo.get("name")
    if not email:
        return RedirectResponse(f"{FRONTEND_URL}/signin.html?error=no_email")

    import json as _json
    session = _oauth_upsert_user(email, name)
    qs = urlencode({"token": session["access_token"], "user": _json.dumps(session["user"])})
    return RedirectResponse(f"{FRONTEND_URL}/oauth-callback.html?{qs}")


# ─── Facebook OAuth ──────────────────────────────────────────────────────────

@router.get("/oauth/debug")
async def oauth_debug():
    """Debug: check which OAuth env vars are loaded."""
    return {
        "google_client_id_set": bool(GOOGLE_CLIENT_ID),
        "google_client_id_preview": GOOGLE_CLIENT_ID[:10] + "..." if GOOGLE_CLIENT_ID else "",
        "facebook_app_id_set": bool(FACEBOOK_APP_ID),
        "facebook_app_id": FACEBOOK_APP_ID,
        "facebook_secret_set": bool(FACEBOOK_APP_SECRET),
        "oauth_redirect_base": OAUTH_REDIRECT_BASE,
        "frontend_url": FRONTEND_URL,
    }


@router.get("/oauth/facebook")
async def oauth_facebook_redirect():
    """Redirect user to Facebook's OAuth consent screen."""
    if not FACEBOOK_APP_ID:
        raise HTTPException(status_code=500, detail="Facebook OAuth not configured")
    params = {
        "client_id": FACEBOOK_APP_ID,
        "redirect_uri": f"{OAUTH_REDIRECT_BASE}/api/auth/oauth/facebook/callback",
        "response_type": "code",
        "scope": "email,public_profile",
    }
    return RedirectResponse(f"https://www.facebook.com/v22.0/dialog/oauth?{urlencode(params)}")


@router.get("/oauth/facebook/callback")
async def oauth_facebook_callback(code: str = "", error: str = ""):
    """Handle the callback from Facebook after user consent."""
    if error or not code:
        return RedirectResponse(f"{FRONTEND_URL}/signin.html?error=oauth_cancelled")

    async with httpx.AsyncClient() as client:
        token_res = await client.get(
            "https://graph.facebook.com/v22.0/oauth/access_token",
            params={
                "client_id": FACEBOOK_APP_ID,
                "client_secret": FACEBOOK_APP_SECRET,
                "redirect_uri": f"{OAUTH_REDIRECT_BASE}/api/auth/oauth/facebook/callback",
                "code": code,
            },
        )
        if token_res.status_code != 200:
            return RedirectResponse(f"{FRONTEND_URL}/signin.html?error=oauth_failed")
        tokens = token_res.json()

        userinfo_res = await client.get(
            "https://graph.facebook.com/me",
            params={"fields": "id,name,email", "access_token": tokens["access_token"]},
        )
        if userinfo_res.status_code != 200:
            return RedirectResponse(f"{FRONTEND_URL}/signin.html?error=oauth_failed")
        userinfo = userinfo_res.json()

    email = userinfo.get("email")
    name = userinfo.get("name")
    if not email:
        return RedirectResponse(f"{FRONTEND_URL}/signin.html?error=no_email")

    import json as _json
    session = _oauth_upsert_user(email, name)
    qs = urlencode({"token": session["access_token"], "user": _json.dumps(session["user"])})
    return RedirectResponse(f"{FRONTEND_URL}/oauth-callback.html?{qs}")
