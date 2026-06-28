#!/usr/bin/env python3
"""
image_library_db.py
===================

Database layer for the **Reference Image Library** — a bulk-ingested collection
of public-domain artwork (paintings, etc.) pulled from open sources (NGA Open
Access, Artvee, publicdomainpictures.net), stored with rich metadata so it can
be searched and filtered.

Uses the SAME shared Supabase PostgreSQL connection as ``auth_db`` /
``palettes_routes`` (``get_db``).  Two tables:

    image_library       -- one row per ingested artwork (+ metadata)
    image_library_jobs  -- one row per bulk-ingestion run (progress tracking)

The actual image *files* are downloaded to disk (see ``image_sources.py`` /
``IMAGE_LIBRARY_DIR``); this table stores their metadata + relative path.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from auth_db import RealDictCursor, get_db


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_image_library_tables() -> None:
    """Create the image_library + image_library_jobs tables if they don't exist."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS image_library (
                    id             SERIAL PRIMARY KEY,
                    source         TEXT NOT NULL,
                    source_id      TEXT NOT NULL,
                    title          TEXT,
                    artist         TEXT,
                    date_text      TEXT,
                    year_start     INTEGER,
                    year_end       INTEGER,
                    medium         TEXT,
                    classification TEXT,
                    description    TEXT,
                    tags           JSONB DEFAULT '[]'::jsonb,
                    source_url     TEXT,
                    image_url      TEXT,
                    thumb_url      TEXT,
                    local_path     TEXT,
                    width          INTEGER,
                    height         INTEGER,
                    file_size      INTEGER,
                    created_at     TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (source, source_id)
                )
                """
            )
            # Normalized cross-source filter columns + object-storage keys
            # (added separately so existing tables upgrade in place).
            for col in ("nationality", "genre", "era", "storage_key", "thumb_key"):
                cur.execute(f"ALTER TABLE image_library ADD COLUMN IF NOT EXISTS {col} TEXT")
            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_imglib_source ON image_library(source)",
                "CREATE INDEX IF NOT EXISTS idx_imglib_classification ON image_library(classification)",
                "CREATE INDEX IF NOT EXISTS idx_imglib_year ON image_library(year_start)",
                "CREATE INDEX IF NOT EXISTS idx_imglib_genre ON image_library(genre)",
                "CREATE INDEX IF NOT EXISTS idx_imglib_era ON image_library(era)",
                "CREATE INDEX IF NOT EXISTS idx_imglib_nationality ON image_library(nationality)",
            ):
                cur.execute(stmt)

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS image_library_jobs (
                    id          SERIAL PRIMARY KEY,
                    source      TEXT NOT NULL,
                    params      JSONB DEFAULT '{}'::jsonb,
                    status      TEXT DEFAULT 'pending',
                    requested   INTEGER DEFAULT 0,
                    fetched     INTEGER DEFAULT 0,
                    saved       INTEGER DEFAULT 0,
                    skipped     INTEGER DEFAULT 0,
                    failed      INTEGER DEFAULT 0,
                    message     TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            conn.commit()
            print("[OK] Image library tables ready")
    except Exception as e:  # pragma: no cover - startup best-effort
        print(f"[WARN] Image library tables init error: {e}")


# ---------------------------------------------------------------------------
# Artwork rows
# ---------------------------------------------------------------------------

# Columns an ingest adapter is allowed to set on an artwork row.
_ARTWORK_FIELDS = (
    "source", "source_id", "title", "artist", "date_text", "year_start",
    "year_end", "medium", "classification", "description", "tags",
    "source_url", "image_url", "thumb_url", "local_path", "width", "height",
    "file_size", "nationality", "genre", "era", "storage_key", "thumb_key",
)


def insert_artwork(meta: Dict[str, Any]) -> Optional[int]:
    """Insert one artwork row. Returns the new row's id, or None if it already
    existed (same source + source_id) and was skipped. (Truthy on insert, so
    existing ``if insert_artwork(...)`` callers keep working.)"""
    row = {k: meta.get(k) for k in _ARTWORK_FIELDS}
    row["tags"] = json.dumps(row.get("tags") or [])
    cols = ", ".join(_ARTWORK_FIELDS)
    placeholders = ", ".join(["%s"] * len(_ARTWORK_FIELDS))
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            INSERT INTO image_library ({cols})
            VALUES ({placeholders})
            ON CONFLICT (source, source_id) DO NOTHING
            RETURNING id
            """,
            [row[k] for k in _ARTWORK_FIELDS],
        )
        result = cur.fetchone()
        conn.commit()
        return result[0] if result else None


def artwork_exists(source: str, source_id: str) -> bool:
    """True if this (source, source_id) is already in the library — lets the
    worker skip the download for known duplicates."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM image_library WHERE source = %s AND source_id = %s LIMIT 1",
            [source, source_id],
        )
        return cur.fetchone() is not None


def query_artworks(
    *,
    source: Optional[str] = None,
    search: Optional[str] = None,
    classification: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    era: Optional[str] = None,
    genre: Optional[str] = None,
    artist: Optional[str] = None,
    nationality: Optional[str] = None,
    page: int = 1,
    page_size: int = 40,
) -> Tuple[List[Dict[str, Any]], int]:
    """Filtered, paginated listing. Returns (rows, total_count)."""
    where: List[str] = []
    args: List[Any] = []
    if source:
        where.append("source = %s")
        args.append(source)
    if classification:
        where.append("classification ILIKE %s")
        args.append(f"%{classification}%")
    if search:
        where.append("(title ILIKE %s OR artist ILIKE %s OR description ILIKE %s)")
        like = f"%{search}%"
        args += [like, like, like]
    if year_from is not None:
        where.append("year_end >= %s")
        args.append(year_from)
    if year_to is not None:
        where.append("year_start <= %s")
        args.append(year_to)
    if era:
        where.append("era = %s")
        args.append(era)
    if genre:
        where.append("genre = %s")
        args.append(genre)
    if nationality:
        where.append("nationality = %s")
        args.append(nationality)
    if artist:
        where.append("artist ILIKE %s")
        args.append(f"%{artist}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    page = max(1, int(page))
    page_size = max(1, min(200, int(page_size)))
    offset = (page - 1) * page_size

    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SELECT COUNT(*) AS n FROM image_library {clause}", args)
        total = cur.fetchone()["n"]
        cur.execute(
            f"""
            SELECT * FROM image_library {clause}
            ORDER BY id DESC
            LIMIT %s OFFSET %s
            """,
            args + [page_size, offset],
        )
        rows = cur.fetchall()
    return rows, total


def get_artwork(artwork_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM image_library WHERE id = %s", [artwork_id])
        return cur.fetchone()


def delete_artwork(artwork_id: int) -> Optional[str]:
    """Delete a row. Returns its local_path (so the caller can remove the file),
    or None if no such row existed."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM image_library WHERE id = %s RETURNING local_path",
            [artwork_id],
        )
        row = cur.fetchone()
        conn.commit()
        return row[0] if row else None


def library_facets() -> Dict[str, Any]:
    """Distinct values (with counts) for the normalized filter dropdowns."""
    out: Dict[str, Any] = {}
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        for field in ("era", "genre", "nationality"):
            cur.execute(
                f"""
                SELECT {field} AS value, COUNT(*) AS n
                FROM image_library
                WHERE {field} IS NOT NULL AND TRIM({field}) <> ''
                GROUP BY {field} ORDER BY n DESC
                """
            )
            out[field] = [{"value": r["value"], "count": r["n"]} for r in cur.fetchall()]
    return out


def library_stats() -> Dict[str, Any]:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) AS total FROM image_library")
        total = cur.fetchone()["total"]
        cur.execute(
            "SELECT source, COUNT(*) AS n FROM image_library GROUP BY source ORDER BY n DESC"
        )
        by_source = {r["source"]: r["n"] for r in cur.fetchall()}
    return {"total": total, "by_source": by_source}


# ---------------------------------------------------------------------------
# Ingestion jobs
# ---------------------------------------------------------------------------

def create_job(source: str, params: Dict[str, Any], requested: int) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO image_library_jobs (source, params, requested, status)
            VALUES (%s, %s, %s, 'pending')
            RETURNING id
            """,
            [source, json.dumps(params), requested],
        )
        job_id = cur.fetchone()[0]
        conn.commit()
        return job_id


def update_job(job_id: int, **fields: Any) -> None:
    """Update mutable job columns (status / fetched / saved / skipped / failed /
    message). Always bumps updated_at."""
    allowed = {"status", "fetched", "saved", "skipped", "failed", "message", "requested"}
    sets = [f"{k} = %s" for k in fields if k in allowed]
    args = [v for k, v in fields.items() if k in allowed]
    if not sets:
        return
    sets.append("updated_at = NOW()")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE image_library_jobs SET {', '.join(sets)} WHERE id = %s",
            args + [job_id],
        )
        conn.commit()


def get_job(job_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM image_library_jobs WHERE id = %s", [job_id])
        return cur.fetchone()


def list_jobs(limit: int = 25) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM image_library_jobs ORDER BY id DESC LIMIT %s",
            [max(1, min(100, int(limit)))],
        )
        return cur.fetchall()


__all__ = [
    "init_image_library_tables",
    "insert_artwork",
    "artwork_exists",
    "query_artworks",
    "get_artwork",
    "delete_artwork",
    "library_stats",
    "library_facets",
    "create_job",
    "update_job",
    "get_job",
    "list_jobs",
]
