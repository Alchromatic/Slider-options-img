"""
Auth routes — login / sign up / current-user.

Mirrors the /api/auth/* endpoints from the
sunnysanwar_integrated_multi_model_cmprxn_role project and writes to the same
`auth_users` table, so accounts are shared between the two apps.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
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
