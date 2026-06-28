#!/usr/bin/env python3
"""
bucket_sync.py
==============

Sync the object-storage bucket into the image library: for every original image
in the bucket, generate a small grid thumbnail + a medium lightbox image (stored
back in the bucket under ``_thumbs/`` / ``_medium/``) and insert a DB row so it
shows in the gallery. Idempotent — re-run it as more files are uploaded.

Run as a script, or call ``sync_bucket()`` from the admin endpoint.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional

import bucket_storage as bs
import image_library_db as db
import library_normalize as norm

_HASH_SUFFIX = re.compile(r"_[0-9a-fA-F]{6,}\.[A-Za-z0-9]+$")
_ORIG_EXT = re.compile(r"\.(jpe?g|png|tiff?|webp|gif|bmp)$", re.I)


def _title_from_key(key: str) -> str:
    base = key.rsplit("/", 1)[-1]
    base = _HASH_SUFFIX.sub("", base)   # strip trailing _<hash>.<ext>
    base = _ORIG_EXT.sub("", base)      # strip the original extension
    name = re.sub(r"\s+", " ", base.replace("_", " ")).strip(" -")
    return name[:300] or "Untitled"


def _process_key(key: str) -> str:
    """Process one bucket object → 'saved' | 'skip' | 'fail'."""
    if db.artwork_exists("bucket", key):
        return "skip"
    tkey, mkey = bs.thumb_key_for(key), bs.medium_key_for(key)
    try:
        data = bs.get_bytes(key)
    except Exception:
        return "fail"
    thumb = bs.make_thumbnail(data)
    if not thumb:
        return "fail"
    medium = bs.make_medium(data)
    try:
        if not bs.object_exists(tkey):
            bs.put_bytes(tkey, thumb)
        if medium and not bs.object_exists(mkey):
            bs.put_bytes(mkey, medium)
    except Exception:
        return "fail"

    title = _title_from_key(key)
    db.insert_artwork({
        "source": "bucket",
        "source_id": key,
        "title": title,
        "classification": "painting",
        "genre": norm.infer_genre(title),
        "era": "Unknown",
        "tags": ["bucket"],
        "storage_key": key,
        "thumb_key": tkey,
        "source_url": None,
    })
    return "saved"


def sync_bucket(on_status: Optional[Callable[[str], None]] = None,
                workers: int = 10, max_items: Optional[int] = None) -> Dict[str, Any]:
    """Generate thumbnails + rows for every bucket image. Returns counts."""
    if not bs.is_configured():
        return {"error": "object storage not configured (S3_* env vars missing)"}
    db.init_image_library_tables()

    keys = list(bs.list_image_keys())
    if max_items:
        keys = keys[:max_items]
    total = len(keys)

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)
        else:
            print(msg, flush=True)

    status(f"Bucket has {total} images; generating thumbnails…")
    saved = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_key, k): k for k in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            saved += r == "saved"
            skipped += r == "skip"
            failed += r == "fail"
            if i % 25 == 0 or i == total:
                status(f"  {i}/{total} — {saved} new, {skipped} existing, {failed} failed")
    return {"total": total, "saved": saved, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print("RESULT:", sync_bucket())
