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
import threading
from contextlib import contextmanager

import jwt
import psycopg2
import psycopg2.extensions
from fastapi import HTTPException
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

# -----------------------------------------------------------------------------
# Connection pool
# -----------------------------------------------------------------------------
# Supabase's session-mode pooler (port 5432) accepts only `pool_size` (15)
# client connections in total, shared with the multi-model project, and refuses
# the next one (EMAXCONNSESSION). Opening a connection per request made ~15
# overlapping requests fail with 500. So instead:
#   * at most DB_POOL_MAX connections per process; a request that finds them
#     all busy waits up to DB_POOL_WAIT s for one instead of failing,
#   * connections are reused, and closed after DB_POOL_IDLE s unused, so a
#     quiet app doesn't keep session slots the other project needs,
#   * a refused connect is retried briefly; if the database still can't be
#     reached the request gets 503 (try again), not a 500.
DB_POOL_MAX = max(1, int(os.getenv("DB_POOL_MAX", "8")))
DB_POOL_WAIT = float(os.getenv("DB_POOL_WAIT", "20"))
DB_POOL_IDLE = float(os.getenv("DB_POOL_IDLE", "60"))
_PING_AFTER = 20.0                      # re-check a connection unused this long before reuse

_slots = threading.BoundedSemaphore(DB_POOL_MAX)
_idle_lock = threading.Lock()
_idle = []                              # [(conn, last_used_monotonic)], most recent last
_reaper = None


class DatabaseBusy(HTTPException):
    """No database connection could be had in time (503, safe to retry)."""

    def __init__(self, detail="The database is busy right now, please try again."):
        super().__init__(status_code=503, detail=detail, headers={"Retry-After": "2"})


def _close_quietly(conn):
    try:
        conn.close()
    except Exception:
        pass


def _connect():
    last = None
    for delay in (0, 0.25, 0.75, 1.5):
        if delay:
            time.sleep(delay)
        try:
            return psycopg2.connect(
                SUPABASE_CONNECTION_STRING,
                connect_timeout=10,
                keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=3,
                application_name="geomagic-api",
            )
        except psycopg2.OperationalError as e:   # e.g. EMAXCONNSESSION while the other app is busy
            last = e
    print(f"[WARN] database connect failed: {last}")
    raise DatabaseBusy() from last


def _reap_loop():
    while True:
        time.sleep(10)
        now = time.monotonic()
        with _idle_lock:
            stale = [c for c, used in _idle if now - used > DB_POOL_IDLE or c.closed]
            _idle[:] = [(c, used) for c, used in _idle if not (now - used > DB_POOL_IDLE or c.closed)]
        for c in stale:
            _close_quietly(c)


def _take():
    """A live connection: a recent idle one if there is one, else a new one."""
    while True:
        with _idle_lock:
            item = _idle.pop() if _idle else None
        if item is None:
            return _connect()
        conn, used = item
        idle_for = time.monotonic() - used
        if conn.closed or idle_for > DB_POOL_IDLE:
            _close_quietly(conn)
            continue
        if idle_for > _PING_AFTER:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                conn.rollback()
            except Exception:
                _close_quietly(conn)
                continue
        return conn


def _give(conn):
    """Back to the pool, with any uncommitted work rolled back (as closing used to do)."""
    global _reaper
    try:
        if conn.closed:
            return
        if conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
            conn.rollback()
        if conn.autocommit:
            conn.autocommit = False
    except Exception:
        _close_quietly(conn)
        return
    with _idle_lock:
        _idle.append((conn, time.monotonic()))
        if _reaper is None:
            _reaper = threading.Thread(target=_reap_loop, name="db-pool-reaper", daemon=True)
            _reaper.start()


@contextmanager
def get_db():
    """A pooled database connection; uncommitted changes are rolled back on exit."""
    if not _slots.acquire(timeout=DB_POOL_WAIT):
        raise DatabaseBusy()
    conn = None
    try:
        conn = _take()
        yield conn
    finally:
        if conn is not None:
            _give(conn)
        _slots.release()


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
