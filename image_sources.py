#!/usr/bin/env python3
"""
image_sources.py
================

Source adapters + image downloader for the **Reference Image Library**.

Each adapter is a generator that yields *metadata dicts* (one per candidate
artwork) WITHOUT downloading the file yet — the ingestion worker
(``image_library_routes``) downloads ``image_url`` for each candidate, fills in
``local_path``/``width``/``height``/``file_size``, and writes the row.

Adapters
--------
    nga                  -- National Gallery of Art Open Access (official CC0
                            CSV dataset + IIIF image API).  Reliable, legal.
    artvee               -- artvee.com listing scraper (public-domain art).
    publicdomainpictures -- publicdomainpictures.net search scraper.

NOTE on the scrapers: Artvee and publicdomainpictures.net front their pages with
anti-bot protection and return HTTP 403 to non-browser clients.  We send
realistic browser headers and retry, but these sites may still rate-limit or
block server IPs — when that happens the job records the failure cleanly.  The
NGA adapter does not depend on scraping and is the dependable workhorse.

Storage: images are written under ``IMAGE_LIBRARY_DIR`` (env, default
``<app>/image_library``) as ``<source>/<source_id>.<ext>``; that relative path
is stored in the DB and served at ``/image-library/...`` (see main.py).
"""

from __future__ import annotations

import csv
import io
import os
import re
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
from PIL import Image, ImageFile

# Salvage partially-downloaded JPEGs instead of failing outright.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# csv has small default field-size limits; NGA rows have long provenance text.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_LIBRARY_DIR = os.getenv(
    "IMAGE_LIBRARY_DIR", os.path.join(_APP_DIR, "image_library")
)
_NGA_CACHE_DIR = os.path.join(IMAGE_LIBRARY_DIR, "_nga_cache")
_NGA_BASE = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data"
# Re-download the NGA CSVs if the local cache is older than this.
_NGA_CACHE_TTL = 7 * 24 * 3600

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

StatusCb = Optional[Callable[[str], None]]


def _status(cb: StatusCb, msg: str) -> None:
    if cb:
        try:
            cb(msg)
        except Exception:
            pass


def make_client() -> httpx.Client:
    return httpx.Client(headers=_BROWSER_HEADERS, follow_redirects=True, timeout=60.0)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120] or "item"


# ===========================================================================
# Image downloader
# ===========================================================================

def download_image(
    client: httpx.Client, url: str, source: str, source_id: str, *, retries: int = 2
) -> Optional[Dict[str, Any]]:
    """Download an image to ``IMAGE_LIBRARY_DIR/<source>/<source_id>.<ext>``.

    Returns ``{local_path, width, height, file_size}`` (local_path is relative to
    IMAGE_LIBRARY_DIR), or ``None`` if the download / decode failed.  Uses a tight
    per-request timeout so a hung host fails fast instead of stalling the run."""
    if not url:
        return None
    dest_dir = os.path.join(IMAGE_LIBRARY_DIR, source)
    os.makedirs(dest_dir, exist_ok=True)

    img_timeout = httpx.Timeout(15.0, connect=10.0)
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.get(url, timeout=img_timeout)
            if resp.status_code != 200 or not resp.content:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.content
            # Validate + measure via PIL (also rejects HTML error pages).
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
                fmt = (im.format or "JPEG").lower()
            ext = {"jpeg": "jpg", "mpo": "jpg"}.get(fmt, fmt)
            rel_path = os.path.join(source, f"{_slug(source_id)}.{ext}")
            abs_path = os.path.join(IMAGE_LIBRARY_DIR, rel_path)
            with open(abs_path, "wb") as fh:
                fh.write(data)
            return {
                "local_path": rel_path.replace(os.sep, "/"),
                "width": width,
                "height": height,
                "file_size": len(data),
            }
        except Exception as e:  # noqa: BLE001 - we want to retry on anything
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"[imglib] download failed {url}: {last_err}")
    return None


# ===========================================================================
# Image proxy cache — fetch remote art server-side (browser headers, where
# hotlinks succeed), resize to a thumbnail, cache to disk, and serve same-origin.
# This is what makes the gallery load fast and reliably (no client-side hotlink
# blocking, rate-limiting, or full-res payloads).
# ===========================================================================

PROXY_CACHE_DIR = os.path.join(IMAGE_LIBRARY_DIR, "_cache")
_THUMB_MAX = 420       # px — grid thumbnails
_FULL_MAX = 1600       # px — lightbox / detail view
_proxy_client_singleton: Optional[httpx.Client] = None


def _proxy_client() -> httpx.Client:
    # httpx.Client is thread-safe for sending requests; reuse one (keep-alive).
    global _proxy_client_singleton
    if _proxy_client_singleton is None:
        _proxy_client_singleton = make_client()
    return _proxy_client_singleton


def render_cached_image(
    cache_path: str, candidate_urls: List[str], size: str, local_abs: Optional[str] = None
) -> Optional[str]:
    """Return a path to a cached JPEG for this artwork at ``size`` ('thumb'|'full').

    On a cache miss: read the local file if present, else fetch the first
    reachable candidate URL (server-side, with browser headers), downscale, and
    write the cache. Returns None if nothing could be produced."""
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    data: Optional[bytes] = None
    if local_abs and os.path.isfile(local_abs):
        try:
            with open(local_abs, "rb") as fh:
                data = fh.read()
        except Exception:
            data = None
    if data is None:
        client = _proxy_client()
        for url in candidate_urls:
            if not url:
                continue
            try:
                r = client.get(url, timeout=httpx.Timeout(15.0, connect=10.0))
                if r.status_code == 200 and r.content:
                    data = r.content
                    break
            except Exception:
                continue
    if not data:
        return None

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            cap = _THUMB_MAX if size == "thumb" else _FULL_MAX
            if max(im.size) > cap:
                im.thumbnail((cap, cap))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp = cache_path + ".tmp"
            im.save(tmp, "JPEG", quality=82, optimize=True)
            os.replace(tmp, cache_path)
        return cache_path
    except Exception as e:
        print(f"[imglib] proxy render failed: {e}")
        return None


# ===========================================================================
# NGA — National Gallery of Art Open Access (official CC0 dataset)
# ===========================================================================

def _ensure_nga_csv(name: str, client: httpx.Client, on_status: StatusCb) -> str:
    """Download (and cache) one NGA opendata CSV; return its local path."""
    os.makedirs(_NGA_CACHE_DIR, exist_ok=True)
    path = os.path.join(_NGA_CACHE_DIR, f"{name}.csv")
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < _NGA_CACHE_TTL
    if fresh:
        return path
    _status(on_status, f"Downloading NGA dataset ({name}.csv, first run only)…")
    url = f"{_NGA_BASE}/{name}.csv"
    tmp = path + ".tmp"
    # These CSVs are large (80MB+); give the streamed read a generous timeout.
    long_timeout = httpx.Timeout(60.0, read=600.0)
    with client.stream("GET", url, timeout=long_timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    os.replace(tmp, path)
    return path


def _nga_image_index(client: httpx.Client, on_status: StatusCb) -> Dict[str, Dict[str, Any]]:
    """Map NGA objectid -> primary open-access image info."""
    path = _ensure_nga_csv("published_images", client, on_status)
    index: Dict[str, Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("viewtype") != "primary":
                continue
            if str(row.get("openaccess", "1")) not in ("1", "true", "True"):
                continue
            objid = row.get("depictstmsobjectid")
            if not objid or objid in index:
                continue
            index[objid] = {
                "iiifurl": row.get("iiifurl"),
                "thumb_url": row.get("iiifthumburl"),
                "width": _to_int(row.get("width")),
                "height": _to_int(row.get("height")),
                "description": (row.get("assistivetext") or "").strip(),
            }
    return index


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def fetch_nga(params: Dict[str, Any], limit: int, on_status: StatusCb = None) -> Iterator[Dict[str, Any]]:
    """Yield NGA Open Access artworks matching the filters.

    params: search, classification (default "painting"), year_from, year_to.
    """
    search = (params.get("search") or "").strip().lower()
    classification = (params.get("classification") or "painting").strip().lower()
    year_from = params.get("year_from")
    year_to = params.get("year_to")

    client = make_client()
    try:
        images = _nga_image_index(client, on_status)
        _status(on_status, f"NGA: {len(images)} open-access images indexed; scanning objects…")
        objects_path = _ensure_nga_csv("objects", client, on_status)

        yielded = 0
        with open(objects_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                objid = row.get("objectid")
                img = images.get(objid)
                if not img or not img.get("iiifurl"):
                    continue

                cls = (row.get("classification") or "")
                if classification and classification not in cls.lower():
                    continue

                ys = _to_int(row.get("beginyear"))
                ye = _to_int(row.get("endyear"))
                if year_from is not None and ye is not None and ye < year_from:
                    continue
                if year_to is not None and ys is not None and ys > year_to:
                    continue

                title = (row.get("title") or "").strip()
                artist = (row.get("attribution") or "").strip()
                medium = (row.get("medium") or "").strip()
                if search:
                    haystack = " ".join([title, artist, medium, img["description"]]).lower()
                    if search not in haystack:
                        continue

                tags = [t for t in [cls, row.get("subclassification"), medium] if t]
                yield {
                    "source": "nga",
                    "source_id": objid,
                    "title": title or "Untitled",
                    "artist": artist,
                    "date_text": (row.get("displaydate") or "").strip(),
                    "year_start": ys,
                    "year_end": ye,
                    "medium": medium,
                    "classification": cls,
                    "description": img["description"],
                    "tags": tags,
                    "source_url": f"https://www.nga.gov/artworks/{objid}",
                    "image_url": f"{img['iiifurl']}/full/!2000,2000/0/default.jpg",
                    "thumb_url": img["thumb_url"],
                }
                yielded += 1
                # Yield a comfortable surplus so the worker can skip dupes/failures.
                if yielded >= limit * 4:
                    break
    finally:
        client.close()


# ===========================================================================
# Artvee — public-domain art (HTML scrape)
# ===========================================================================

_ARTVEE_BASE = "https://artvee.com"


def fetch_artvee(params: Dict[str, Any], limit: int, on_status: StatusCb = None) -> Iterator[Dict[str, Any]]:
    from bs4 import BeautifulSoup  # lazy import

    search = (params.get("search") or "oil painting").strip()
    client = make_client()
    seen = 0
    try:
        for page in range(1, 26):  # safety cap on pagination
            if seen >= limit * 2:
                break
            url = f"{_ARTVEE_BASE}/main/"
            try:
                resp = client.get(url, params={"s": search, "paged": page})
            except Exception as e:  # noqa: BLE001
                _status(on_status, f"Artvee request error: {e}")
                break
            if resp.status_code == 403:
                _status(on_status, "Artvee returned 403 (anti-bot block) — stopping.")
                break
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select("div.product-grid-item, li.product")
            if not items:
                break
            for it in items:
                link = it.select_one("a[href]")
                detail = link.get("href") if link else None
                if not detail:
                    continue
                source_id = _slug(detail.rstrip("/").split("/")[-1])
                img = it.select_one("img")
                thumb = None
                if img:
                    thumb = img.get("data-src") or img.get("src")
                title_el = it.select_one(".product-title, h3, .title")
                title = title_el.get_text(strip=True) if title_el else "Untitled"
                artist_el = it.select_one(".product-artist, .artist")
                artist = artist_el.get_text(strip=True) if artist_el else ""

                image_url = _artvee_full_image(client, detail) or thumb
                if not image_url:
                    continue
                yield {
                    "source": "artvee",
                    "source_id": source_id,
                    "title": title,
                    "artist": artist,
                    "date_text": "",
                    "year_start": None,
                    "year_end": None,
                    "medium": "",
                    "classification": "painting",
                    "description": "",
                    "tags": ["artvee", search],
                    "source_url": detail,
                    "image_url": image_url,
                    "thumb_url": thumb,
                }
                seen += 1
                if seen >= limit * 2:
                    break
            time.sleep(1.0)  # politeness
    finally:
        client.close()


def _artvee_full_image(client: httpx.Client, detail_url: str) -> Optional[str]:
    """Fetch an Artvee detail page and pull the full-size image (og:image)."""
    try:
        from bs4 import BeautifulSoup
        resp = client.get(detail_url)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        og = soup.select_one('meta[property="og:image"]')
        if og and og.get("content"):
            return og["content"]
    except Exception:
        return None
    return None


# ===========================================================================
# publicdomainpictures.net — search scrape
# ===========================================================================

_PDP_BASE = "https://www.publicdomainpictures.net"


def fetch_publicdomainpictures(
    params: Dict[str, Any], limit: int, on_status: StatusCb = None
) -> Iterator[Dict[str, Any]]:
    from bs4 import BeautifulSoup  # lazy import
    from urllib.parse import urljoin

    search = (params.get("search") or "oil painting").strip()
    client = make_client()
    seen = 0
    try:
        for page in range(1, 26):
            if seen >= limit * 2:
                break
            try:
                resp = client.get(
                    f"{_PDP_BASE}/en/hledej.php",
                    params={"hleda": search, "page": page},
                )
            except Exception as e:  # noqa: BLE001
                _status(on_status, f"publicdomainpictures request error: {e}")
                break
            if resp.status_code == 403:
                _status(on_status, "publicdomainpictures returned 403 (anti-bot block) — stopping.")
                break
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.select('a[href*="view-image.php"]')
            if not links:
                break
            for a in links:
                href = a.get("href")
                if not href:
                    continue
                detail = urljoin(_PDP_BASE + "/en/", href)
                m = re.search(r"image=(\d+)", href)
                source_id = m.group(1) if m else _slug(href)
                img = a.select_one("img")
                thumb = urljoin(_PDP_BASE, img.get("src")) if (img and img.get("src")) else None
                title = (img.get("alt").strip() if (img and img.get("alt")) else "Untitled")

                image_url = _pdp_full_image(client, detail) or thumb
                if not image_url:
                    continue
                yield {
                    "source": "publicdomainpictures",
                    "source_id": source_id,
                    "title": title or "Untitled",
                    "artist": "",
                    "date_text": "",
                    "year_start": None,
                    "year_end": None,
                    "medium": "",
                    "classification": "painting",
                    "description": "",
                    "tags": ["publicdomainpictures", search],
                    "source_url": detail,
                    "image_url": image_url,
                    "thumb_url": thumb,
                }
                seen += 1
                if seen >= limit * 2:
                    break
            time.sleep(1.0)
    finally:
        client.close()


def _pdp_full_image(client: httpx.Client, detail_url: str) -> Optional[str]:
    try:
        from bs4 import BeautifulSoup
        resp = client.get(detail_url)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        og = soup.select_one('meta[property="og:image"]')
        if og and og.get("content"):
            return og["content"]
        big = soup.select_one('img#main-image, img.bigImage, a[href*="/pictures/"][href*="velka"]')
        if big:
            return big.get("href") or big.get("src")
    except Exception:
        return None
    return None


# ===========================================================================
# Registry
# ===========================================================================

ADAPTERS: Dict[str, Callable[..., Iterator[Dict[str, Any]]]] = {
    "nga": fetch_nga,
    "artvee": fetch_artvee,
    "publicdomainpictures": fetch_publicdomainpictures,
}

SOURCE_LABELS = {
    "nga": "National Gallery of Art (Open Access)",
    "artvee": "Artvee",
    "publicdomainpictures": "PublicDomainPictures.net",
}


__all__ = [
    "IMAGE_LIBRARY_DIR",
    "PROXY_CACHE_DIR",
    "ADAPTERS",
    "SOURCE_LABELS",
    "download_image",
    "render_cached_image",
    "make_client",
]
