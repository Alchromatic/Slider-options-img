#!/usr/bin/env python3
"""
image_library_routes.py
=======================

Admin-only endpoints for the **Reference Image Library** bulk-ingestion tool.

A "job" pulls public-domain artwork from one source (NGA / Artvee /
publicdomainpictures), downloads the image files, and stores rich metadata in
Postgres so the collection can be searched and filtered.  Jobs run in a
background thread; the UI polls ``GET /api/library/jobs/{id}`` for progress.

All routes are gated behind ``ADMIN_TOKEN`` (sent as the ``X-Admin-Token``
header), exactly like ``profile_admin_routes`` — this is a backend/operator
tool, not a user-facing action.

Endpoints
---------
    GET    /api/library/sources          -> available sources (for the UI)
    GET    /api/library/stats            -> counts per source
    POST   /api/library/ingest           -> start an ingestion job
    GET    /api/library/jobs             -> recent jobs
    GET    /api/library/jobs/{id}        -> one job (poll for progress)
    POST   /api/library/jobs/{id}/cancel -> request cancellation
    GET    /api/library/artworks         -> filtered, paginated listing
    DELETE /api/library/artworks/{id}    -> delete a row + its file
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import image_library_db as db
from image_sources import (
    ADAPTERS, IMAGE_LIBRARY_DIR, PROXY_CACHE_DIR, SOURCE_LABELS,
    bake_thumbnail, render_cached_image,
)

router = APIRouter(prefix="/api/library", tags=["15. Reference image library"])

# Job id -> threading.Event used to request cancellation of a running job.
_CANCEL: dict[int, threading.Event] = {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _require_admin(token: Optional[str]) -> None:
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Image library endpoints are disabled (ADMIN_TOKEN not configured).",
        )
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token.")


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------

def _run_job(job_id: int, source: str, params: dict, limit: int) -> None:
    cancel = _CANCEL.setdefault(job_id, threading.Event())
    adapter = ADAPTERS[source]
    saved = skipped = failed = fetched = 0

    def status(msg: str) -> None:
        db.update_job(job_id, message=msg)

    db.update_job(job_id, status="running", message="Starting…")
    try:
        for meta in adapter(params, limit, status):
            if cancel.is_set():
                db.update_job(job_id, status="cancelled", message="Cancelled by operator.")
                return
            if saved >= limit:
                break
            fetched += 1
            # Skip items we already have.
            if db.artwork_exists(source, meta["source_id"]):
                skipped += 1
                continue
            # Store metadata first (no slow full-res download — the proxy serves
            # full size on demand), then bake the static gallery thumbnail so the
            # new image shows instantly in the gallery grid.
            meta["local_path"] = None
            art_id = db.insert_artwork(meta)
            if not art_id:
                skipped += 1
                continue
            saved += 1
            baked = bake_thumbnail(art_id, [meta.get("thumb_url"), meta.get("image_url")])
            if not baked:
                failed += 1  # metadata saved, but no thumbnail (gallery proxy will retry)
            if fetched % 3 == 0 or saved >= limit:
                db.update_job(
                    job_id, fetched=fetched, saved=saved, skipped=skipped, failed=failed,
                    message=f"Imported {saved}/{limit}…",
                )
        db.update_job(
            job_id, status="done", fetched=fetched, saved=saved, skipped=skipped,
            failed=failed,
            message=f"Done. {saved} imported, {skipped} duplicates, {failed} without thumbnail.",
        )
    except Exception as e:  # noqa: BLE001
        db.update_job(
            job_id, status="error", fetched=fetched, saved=saved, skipped=skipped,
            failed=failed, message=f"Error: {e}",
        )
    finally:
        _CANCEL.pop(job_id, None)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    source: str = Field(..., description="One of: nga, artvee, publicdomainpictures.")
    search: str = Field("", description="Keyword filter (title/artist/medium/description).")
    classification: str = Field("painting", description="Artwork type filter, e.g. 'painting'.")
    year_from: Optional[int] = Field(None, description="Earliest year (inclusive).")
    year_to: Optional[int] = Field(None, description="Latest year (inclusive).")
    limit: int = Field(50, ge=1, le=1000, description="How many new images to ingest.")


# ---------------------------------------------------------------------------
# Routes
#
# Read endpoints (sources / stats / artworks) are PUBLIC so the user-facing
# gallery can browse the collection without a token. Mutating endpoints
# (ingest / jobs / cancel / delete) remain gated behind ADMIN_TOKEN.
# ---------------------------------------------------------------------------

@router.get("/sources")
def list_sources():
    return {"sources": [{"id": k, "label": v} for k, v in SOURCE_LABELS.items()]}


@router.get("/stats")
def stats():
    return db.library_stats()


@router.post("/ingest")
def start_ingest(req: IngestRequest, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    if req.source not in ADAPTERS:
        raise HTTPException(status_code=400, detail=f"Unknown source '{req.source}'.")
    params = {
        "search": req.search,
        "classification": req.classification,
        "year_from": req.year_from,
        "year_to": req.year_to,
    }
    job_id = db.create_job(req.source, params, req.limit)
    t = threading.Thread(
        target=_run_job, args=(job_id, req.source, params, req.limit), daemon=True
    )
    t.start()
    return {"job_id": job_id, "status": "started"}


@router.get("/jobs")
def jobs(limit: int = 25, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return {"jobs": db.list_jobs(limit)}


@router.get("/jobs/{job_id}")
def job(job_id: int, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="No such job.")
    return row


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    ev = _CANCEL.get(job_id)
    if ev:
        ev.set()
        return {"status": "cancelling"}
    raise HTTPException(status_code=404, detail="Job is not running.")


@router.get("/img/{artwork_id}")
def image_proxy(artwork_id: int, size: str = "thumb"):
    """Public, cached image proxy. Fetches the artwork's remote image server-side
    (where hotlinks succeed), downscales it, caches to disk, and serves it from
    this origin — so the gallery loads fast and never shows broken hotlinks.

    ``size`` = 'thumb' (grid, ~420px) or 'full' (lightbox, ~1600px)."""
    size = "full" if size == "full" else "thumb"
    art = db.get_artwork(artwork_id)
    if not art:
        raise HTTPException(status_code=404, detail="No such artwork.")

    cache_path = os.path.join(PROXY_CACHE_DIR, f"{artwork_id}_{size}.jpg")
    if size == "thumb":
        candidates = [art.get("thumb_url"), art.get("image_url")]
    else:
        candidates = [art.get("image_url"), art.get("thumb_url")]
    local_abs = (
        os.path.join(IMAGE_LIBRARY_DIR, art["local_path"]) if art.get("local_path") else None
    )

    path = render_cached_image(cache_path, [c for c in candidates if c], size, local_abs)
    if not path:
        raise HTTPException(status_code=502, detail="Image unavailable from source.")
    return FileResponse(
        path, media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800"},  # 7 days
    )


@router.get("/artworks")
def artworks(
    source: Optional[str] = None,
    search: Optional[str] = None,
    classification: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    page: int = 1,
    page_size: int = Query(40, ge=1, le=200),
):
    rows, total = db.query_artworks(
        source=source, search=search, classification=classification,
        year_from=year_from, year_to=year_to, page=page, page_size=page_size,
    )
    return {"total": total, "page": page, "page_size": page_size, "artworks": rows}


@router.delete("/artworks/{artwork_id}")
def delete_artwork(artwork_id: int, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    local_path = db.delete_artwork(artwork_id)
    if local_path is None:
        raise HTTPException(status_code=404, detail="No such artwork.")
    # Best-effort file cleanup.
    try:
        abs_path = os.path.join(IMAGE_LIBRARY_DIR, local_path)
        if os.path.isfile(abs_path):
            os.remove(abs_path)
    except Exception:
        pass
    return {"deleted": artwork_id}


__all__ = ["router", "db"]
