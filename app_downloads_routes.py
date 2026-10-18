"""App downloads: the Alchroma APKs, served from the storage bucket.

The builds live in the bucket (apps/<platform>/...), not in the repo. The bucket
only hands out time-limited presigned links, so GET /downloads/<platform> is the
stable public address: it redirects to a fresh presigned link each time, with a
Content-Disposition that makes the browser save the file under its real name.

An APP_<PLATFORM>_URL env var (e.g. a Play Store page) overrides the bucket copy.
"""
import json
import os
import threading
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

import bucket_storage

router = APIRouter(tags=["Devices"])

APPS = {
    "android": {"key": "apps/android/Alchroma-android.apk", "filename": "Alchroma-android.apk",
                "type": "application/vnd.android.package-archive", "env": "APP_ANDROID_URL"},
    "quest":   {"key": "apps/quest/Alchroma-quest.apk", "filename": "Alchroma-quest.apk",
                "type": "application/vnd.android.package-archive", "env": "APP_QUEST_URL"},
    "ios":     {"key": None, "filename": None, "type": None, "env": "APP_IOS_URL"},  # TestFlight link only
}
LINK_TTL = 3600           # presigned link lifetime (seconds)
_CACHE_TTL = 600          # how long bucket lookups are remembered
_cache = {}
_lock = threading.Lock()


def _cached(name, compute):
    with _lock:
        hit = _cache.get(name)
        if hit and time.time() - hit[0] < _CACHE_TTL:
            return hit[1]
    value = compute()
    with _lock:
        _cache[name] = (time.time(), value)
    return value


def _in_bucket(platform: str) -> bool:
    key = APPS[platform]["key"]
    if not key or not bucket_storage.is_configured():
        return False
    return _cached("exists:" + platform, lambda: bucket_storage.object_exists(key))


def _info(platform: str) -> dict:
    key = APPS[platform]["key"]
    if not key or not bucket_storage.is_configured():
        return {}

    def load():
        try:
            return json.loads(bucket_storage.get_bytes(key.rsplit("/", 1)[0] + "/info.json"))
        except Exception:
            return {}
    return _cached("info:" + platform, load)


def download_links() -> dict:
    """{platform: {"url": ..., "info": {...}}} for every app that can be downloaded.
    URLs starting with "/" are relative to this backend."""
    out = {}
    for platform, app in APPS.items():
        override = os.getenv(app["env"], "")
        if override:
            out[platform] = {"url": override, "info": {}}
        elif _in_bucket(platform):
            out[platform] = {"url": f"/downloads/{platform}", "info": _info(platform)}
    return out


@router.get("/downloads/{platform}")
def download_app(platform: str):
    """Download the app for `platform` (android, quest): redirects to a fresh storage link."""
    app = APPS.get(platform)
    if not app:
        raise HTTPException(status_code=404, detail="Unknown platform")
    override = os.getenv(app["env"], "")
    if override:
        return RedirectResponse(override, status_code=302)
    if not _in_bucket(platform):
        raise HTTPException(status_code=404, detail="No build published for this platform yet")
    url = bucket_storage.client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_storage.bucket_name(), "Key": app["key"],
                "ResponseContentType": app["type"],
                "ResponseContentDisposition": f'attachment; filename="{app["filename"]}"'},
        ExpiresIn=LINK_TTL,
    )
    return RedirectResponse(url, status_code=302, headers={"Cache-Control": "no-store"})
