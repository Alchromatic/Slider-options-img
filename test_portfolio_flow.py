#!/usr/bin/env python3
"""
test_portfolio_flow.py
======================

Live integration test for portfolio_routes.py (the Portfolio shared by the
webapp and the Alchroma apps).  Same setup as test_devices_flow.py: it talks to
a RUNNING backend (default http://127.0.0.1:8011, "backend-verify" launch
config) and the real Supabase DB, creates throwaway users and deletes
everything it made at the end.

Run:  python -m pytest test_portfolio_flow.py -q
"""

from __future__ import annotations

import os
import secrets

import pytest
import requests

BASE = os.getenv("DEVICES_TEST_BASE", "http://127.0.0.1:8011").rstrip("/")
PW = "Portfolio-Test-123!"
THUMB = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="
SHAPES = [{"type": 1, "data": [1, 2, 3, 4], "color": [10, 20, 30, 255]}]


def _reachable() -> bool:
    try:
        return requests.get(BASE + "/api/devices/config", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"backend not running at {BASE}")


def _req(method, path, token=None, json=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.request(method, BASE + path, json=json, headers=h, timeout=30)


@pytest.fixture(scope="module")
def users():
    """Two throwaway accounts (+ a device token for the first); removed after the module."""
    made = []
    for _ in range(2):
        email = f"portfolio-test-{secrets.token_hex(4)}@example.com"
        r = _req("POST", "/api/auth/register", json={"email": email, "password": PW, "name": "Portfolio Test"})
        assert r.status_code == 200, r.text
        d = r.json()
        made.append({"email": email, "token": d["access_token"], "id": d["user"]["id"]})
    r = _req("POST", "/api/devices/register", made[0]["token"], {
        "device_id": "pf-test-" + secrets.token_hex(4), "device_type": "phone",
        "platform": "android", "name": "Portfolio test phone",
    })
    assert r.status_code == 200, r.text
    made[0]["device_token"] = r.json()["access_token"]
    made[0]["device_row"] = r.json()["device"]["id"]
    yield made
    from auth_db import get_db

    ids = tuple(u["id"] for u in made)
    with get_db() as conn:
        cur = conn.cursor()
        for table in ("user_portfolio", "device_captures", "user_devices", "device_pair_codes", "user_palettes"):
            cur.execute(f"DELETE FROM {table} WHERE user_id IN %s", (ids,))
        cur.execute("DELETE FROM auth_users WHERE id IN %s", (ids,))
        conn.commit()


def test_requires_auth():
    assert _req("GET", "/api/portfolio").status_code == 401


def test_web_save_is_seen_by_the_app(users):
    u = users[0]
    item = {"id": "pweb1", "title": "Sunset", "thumb": THUMB, "w": 640, "h": 480,
            "shapeCount": 1, "logic": "original", "shapes": SHAPES,
            "colorOrder": ["#FF0000"], "limit": 50, "createdAt": 1_750_000_000_000}
    r = _req("POST", "/api/portfolio", u["token"], item)
    assert r.status_code == 200, r.text
    saved = r.json()["item"]
    assert saved["source"] == "web" and saved["hasShapes"] and saved["createdAt"] == 1_750_000_000_000

    # the app (device token) lists it — without shapes — then fetches them
    r = _req("GET", "/api/portfolio", u["device_token"])
    assert r.status_code == 200, r.text
    listed = [i for i in r.json()["items"] if i["id"] == "pweb1"]
    assert listed and "shapes" not in listed[0] and listed[0]["colorOrder"] == ["#FF0000"] and listed[0]["limit"] == 50
    full = _req("GET", "/api/portfolio/pweb1", u["device_token"]).json()["item"]
    assert full["shapes"] == SHAPES


def test_app_save_is_seen_by_the_web(users):
    u = users[0]
    r = _req("POST", "/api/portfolio", u["device_token"],
             {"id": "papp1", "title": "Wall scan", "thumb": THUMB, "meta": {"paints": ["Ultramarine"]}})
    assert r.status_code == 200, r.text
    it = r.json()["item"]
    assert it["source"] == "app" and it["deviceId"] and it["meta"] == {"paints": ["Ultramarine"]}
    ids = [i["id"] for i in _req("GET", "/api/portfolio", u["token"]).json()["items"]]
    assert "papp1" in ids and "pweb1" in ids


def test_upsert_keeps_omitted_fields(users):
    u = users[0]
    r = _req("POST", "/api/portfolio", u["token"], {"id": "pweb1", "w": 800})
    assert r.status_code == 200, r.text
    it = r.json()["item"]
    assert it["title"] == "Sunset" and it["w"] == 800 and it["thumb"] == THUMB and it["hasShapes"]
    assert it["colorOrder"] == ["#FF0000"]


def test_batch_rename_delete(users):
    u = users[0]
    r = _req("POST", "/api/portfolio/batch", u["token"],
             {"items": [{"id": "pb1", "title": "One", "thumb": THUMB}, {"id": "pb2", "title": "Two"}]})
    assert r.status_code == 200 and r.json()["count"] == 2, r.text
    r = _req("PATCH", "/api/portfolio/pb1", u["token"], {"title": "Renamed"})
    assert r.status_code == 200 and r.json()["item"]["title"] == "Renamed"
    assert _req("DELETE", "/api/portfolio/pb2", u["device_token"]).status_code == 200
    assert _req("GET", "/api/portfolio/pb2", u["token"]).status_code == 404
    assert _req("DELETE", "/api/portfolio/pb2", u["token"]).status_code == 404


def test_validation(users):
    t = users[0]["token"]
    assert _req("POST", "/api/portfolio", t, {"id": "bad id!", "title": "x"}).status_code == 400
    assert _req("POST", "/api/portfolio", t, {"title": "x", "thumb": "javascript:alert(1)"}).status_code == 400
    r = _req("POST", "/api/portfolio", t, {"title": "no id"})
    assert r.status_code == 200 and r.json()["item"]["id"].startswith("p")


def test_isolation_between_users(users):
    other = users[1]["token"]
    assert _req("GET", "/api/portfolio", other).json()["items"] == []
    assert _req("GET", "/api/portfolio/pweb1", other).status_code == 404
    assert _req("DELETE", "/api/portfolio/pweb1", other).status_code == 404
    assert _req("PATCH", "/api/portfolio/pweb1", other, {"title": "hijack"}).status_code == 404


def test_unpaired_device_is_rejected(users):
    u = users[0]
    assert _req("DELETE", f"/api/devices/{u['device_row']}", u["token"]).status_code == 200
    r = _req("GET", "/api/portfolio", u["device_token"])
    assert r.status_code == 401 and r.json()["detail"] == "device_revoked"
