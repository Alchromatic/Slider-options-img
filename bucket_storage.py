#!/usr/bin/env python3
"""
bucket_storage.py
=================

Thin wrapper around the image-library object storage bucket (Railway / Tigris,
S3-compatible). Holds the user's own high-resolution art collection.

The bucket is PRIVATE, so images are served to the browser via short-lived
**presigned URLs** (the browser then loads directly from the storage edge — fast,
no proxy hop). Small thumbnails are pre-generated and stored back in the bucket
under the ``THUMB_PREFIX`` so the gallery grid loads with ~0 latency.

Config comes from env (see .env.example):
    S3_ENDPOINT, S3_BUCKET, S3_REGION, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import io
import os
import threading
from typing import Dict, Iterator, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

THUMB_PREFIX = "_thumbs/"           # generated grid thumbnails (~420px)
MEDIUM_PREFIX = "_medium/"          # generated lightbox/use size (~1600px)
_THUMB_MAX = 420                    # px
_MEDIUM_MAX = 1600                  # px
_PRESIGN_TTL = 6 * 3600            # seconds (presigned URL lifetime)
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".gif", ".bmp")

_client = None
_client_lock = threading.Lock()


def is_configured() -> bool:
    return bool(os.getenv("S3_ENDPOINT") and os.getenv("S3_BUCKET")
                and os.getenv("S3_ACCESS_KEY_ID") and os.getenv("S3_SECRET_ACCESS_KEY"))


def bucket_name() -> str:
    return os.getenv("S3_BUCKET", "")


def client():
    """Lazily build a shared boto3 S3 client (thread-safe for requests)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import boto3
                from botocore.config import Config
                _client = boto3.client(
                    "s3",
                    endpoint_url=os.getenv("S3_ENDPOINT"),
                    aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
                    region_name=os.getenv("S3_REGION", "auto"),
                    config=Config(signature_version="s3v4",
                                  s3={"addressing_style": "path"},
                                  retries={"max_attempts": 3, "mode": "standard"}),
                )
    return _client


def list_image_keys() -> Iterator[str]:
    """Yield every original-image object key in the bucket (skips thumbnails)."""
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name()):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.startswith(THUMB_PREFIX):
                continue
            if key.lower().endswith(_IMAGE_EXTS):
                yield key


def _hash(key: str) -> str:
    import hashlib
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def thumb_key_for(key: str) -> str:
    """Stable thumbnail key (grid size) for an original object key."""
    return f"{THUMB_PREFIX}{_hash(key)}.jpg"


def medium_key_for(key: str) -> str:
    """Stable medium key (lightbox/use size) for an original object key."""
    return f"{MEDIUM_PREFIX}{_hash(key)}.jpg"


def get_bytes(key: str) -> bytes:
    return client().get_object(Bucket=bucket_name(), Key=key)["Body"].read()


def object_exists(key: str) -> bool:
    try:
        client().head_object(Bucket=bucket_name(), Key=key)
        return True
    except Exception:
        return False


def put_bytes(key: str, data: bytes, content_type: str = "image/jpeg") -> None:
    client().put_object(Bucket=bucket_name(), Key=key, Body=data, ContentType=content_type)


def presigned_url(key: str, expires: int = _PRESIGN_TTL) -> str:
    """A short-lived GET URL the browser can load directly from storage."""
    return client().generate_presigned_url(
        "get_object", Params={"Bucket": bucket_name(), "Key": key}, ExpiresIn=expires
    )


def make_resized(data: bytes, max_px: int, quality: int = 82) -> Optional[bytes]:
    """Downscale image bytes to a JPEG no larger than ``max_px`` (handles TIFF/PNG)."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            if max(im.size) > max_px:
                im.thumbnail((max_px, max_px))
            out = io.BytesIO()
            im.save(out, "JPEG", quality=quality, optimize=True)
            return out.getvalue()
    except Exception as e:  # noqa: BLE001
        print(f"[bucket] resize failed: {e}")
        return None


def make_thumbnail(data: bytes) -> Optional[bytes]:
    return make_resized(data, _THUMB_MAX, quality=80)


def make_medium(data: bytes) -> Optional[bytes]:
    return make_resized(data, _MEDIUM_MAX, quality=82)


__all__ = [
    "is_configured", "bucket_name", "client", "list_image_keys",
    "thumb_key_for", "medium_key_for", "get_bytes", "object_exists", "put_bytes",
    "presigned_url", "make_thumbnail", "make_medium", "make_resized",
    "THUMB_PREFIX", "MEDIUM_PREFIX",
]
