"""
Auth database layer — Supabase PostgreSQL.

Uses the SAME `auth_users` table (and connection) as the
sunnysanwar_integrated_multi_model_cmprxn_role project so the two apps share a
single user directory. Point this app at the same database by setting the
SUPABASE_CONNECTION_STRING environment variable.
"""

import os
import time
import hashlib
import secrets
from contextlib import contextmanager

import jwt
import psycopg2
from psycopg2.extras import RealDictCursor  # re-exported for routes

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# Database configuration — Supabase PostgreSQL (same as the reference project)
# =============================================================================
SUPABASE_CONNECTION_STRING = os.getenv("SUPABASE_CONNECTION_STRING", "")


@contextmanager
def get_db():
    """Get a database connection (same pattern as role_profile_routes.get_db)."""
    conn = psycopg2.connect(SUPABASE_CONNECTION_STRING)
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# Table initialisation — identical schema to the reference `auth_users` table
# =============================================================================
def init_auth_tables():
    """Create the auth_users table in Supabase if it doesn't exist."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT,
                    organization_name TEXT,
                    workspace_id TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
            # Admin/status columns — added separately so existing tables upgrade.
            for col, col_type, default in [
                ("status", "TEXT", "'active'"),
                ("workflow_access", "BOOLEAN", "TRUE"),
                ("is_admin", "BOOLEAN", "FALSE"),
            ]:
                try:
                    cursor.execute(
                        f"ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS {col} {col_type} DEFAULT {default}"
                    )
                except Exception:
                    pass
            conn.commit()
            print("[OK] Auth users table ready")
    except Exception as e:
        print(f"[WARN] Auth tables init error: {e}")


# =============================================================================
# Password hashing — pbkdf2_hmac sha256 (matches the reference exactly)
# =============================================================================
def hash_password(password: str, salt: str = None) -> tuple:
    """Hash a password with a salt. Returns (stored_hash, salt)."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return f"{salt}:{hashed}", salt


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored `salt:hash` value."""
    try:
        salt, _ = stored_hash.split(":")
    except ValueError:
        return False
    computed, _ = hash_password(password, salt)
    return computed == stored_hash


# =============================================================================
# JWT helpers — HS256 signed with JWT_SECRET (matches the reference exactly)
# =============================================================================
def generate_jwt_token(user_id: str, email: str) -> str:
    """Generate a 7-day JWT for the user."""
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400 * 7,  # 7 days
    }
    return jwt.encode(payload, os.getenv("JWT_SECRET", ""), algorithm="HS256")


def decode_jwt_token(token: str) -> dict:
    """Decode and verify a JWT token."""
    return jwt.decode(token, os.getenv("JWT_SECRET", ""), algorithms=["HS256"])
