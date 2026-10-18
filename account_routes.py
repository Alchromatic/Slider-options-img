"""Account profile photo.

    POST   /api/account/avatar                     multipart `file` (jpeg/png/webp, <= 5 MB) -> {"avatar_url"}
    DELETE /api/account/avatar                     clears it -> {"avatar_url": null}
    GET    /api/account/avatar/{user_id}/{name}    the stored photo (public, cached forever)

The upload is re-encoded to a 512px square WebP (which also drops EXIF/GPS data)
and kept in the storage bucket under avatars/<user_id>/. The bucket only hands
out short-lived presigned links, so the photo is served through this backend at
a stable address; every upload gets a new random name, so that address can be
cached as immutable. auth_users.avatar_url holds the full address and is
returned with the user object by /api/auth/login, /register and /me.
"""
import io
import re
import secrets

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

import bucket_storage
from auth_db import decode_jwt_token, get_db

router = APIRouter(prefix="/api/account", tags=["Account"])

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 40_000_000                 # refuse decompression bombs before decoding
SIZE = 512
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_NAME_RE = re.compile(r"^[0-9a-f]{16}\.webp$")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _user_id(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return str(decode_jwt_token(header[7:])["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _public_base(request: Request) -> str:
    """This backend's public origin (honours the proxy's forwarded headers on Railway)."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host.split(',')[0].strip()}"


def _key_from_url(url):
    """Bucket key of a photo this module stored, or None for anything else."""
    m = re.search(r"/api/account/avatar/([A-Za-z0-9_-]{1,64})/([0-9a-f]{16}\.webp)$", url or "")
    return f"{bucket_storage.AVATAR_PREFIX}{m.group(1)}/{m.group(2)}" if m else None


def _square_webp(data: bytes) -> bytes:
    from PIL import Image, ImageOps
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=415, detail="Use a JPEG, PNG or WebP image")
            if im.width * im.height > MAX_PIXELS:
                raise HTTPException(status_code=413, detail="That image is too large")
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
            im = ImageOps.fit(im, (SIZE, SIZE), Image.LANCZOS)
            out = io.BytesIO()
            im.save(out, "WEBP", quality=85, method=4)
            return out.getvalue()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="That file isn't a readable image")


def _set_avatar(user_id: str, url):
    """Store `url` (or NULL) on the account; returns the previous value."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT avatar_url FROM auth_users WHERE id = %s FOR UPDATE", (user_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        cur.execute("UPDATE auth_users SET avatar_url = %s WHERE id = %s", (url, user_id))
        conn.commit()
        return row[0]


def _drop_old(url):
    key = _key_from_url(url)
    if key:
        try:
            bucket_storage.delete_object(key)
        except Exception as e:  # noqa: BLE001 — a leftover file is harmless
            print(f"[avatar] could not delete {key}: {e}")


@router.post("/avatar")
async def upload_avatar(request: Request, file: UploadFile = File(...)):
    user_id = _user_id(request)
    if not bucket_storage.is_configured():
        raise HTTPException(status_code=503, detail="Photo storage isn't configured")
    if int(request.headers.get("content-length") or 0) > MAX_BYTES + 64 * 1024:
        raise HTTPException(status_code=413, detail="Photos can be at most 5 MB")
    if (file.content_type or "").lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Use a JPEG, PNG or WebP image")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Photos can be at most 5 MB")
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty")

    webp = _square_webp(data)
    name = secrets.token_hex(8) + ".webp"
    bucket_storage.put_bytes(f"{bucket_storage.AVATAR_PREFIX}{user_id}/{name}", webp, "image/webp")
    url = f"{_public_base(request)}/api/account/avatar/{user_id}/{name}"
    _drop_old(_set_avatar(user_id, url))
    return {"avatar_url": url}


@router.delete("/avatar")
def remove_avatar(request: Request):
    user_id = _user_id(request)
    _drop_old(_set_avatar(user_id, None))
    return {"avatar_url": None}


@router.get("/avatar/{user_id}/{name}")
def get_avatar(user_id: str, name: str):
    if not (_ID_RE.match(user_id) and _NAME_RE.match(name)) or not bucket_storage.is_configured():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        data = bucket_storage.get_bytes(f"{bucket_storage.AVATAR_PREFIX}{user_id}/{name}")
    except Exception:
        raise HTTPException(status_code=404, detail="Not found")
    return Response(data, media_type="image/webp",
                    headers={"Cache-Control": "public, max-age=31536000, immutable"})
