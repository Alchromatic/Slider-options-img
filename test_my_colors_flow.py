#!/usr/bin/env python3
"""
test_my_colors_flow.py
======================

Live integration test for the "My Colors" single-color endpoints in
palettes_routes.py (/api/palettes/my-colors/*).  Same setup as
test_devices_flow.py: it talks to a RUNNING backend (default
http://127.0.0.1:8011, "backend-verify" launch config) and the real Supabase
DB, creates throwaway users and deletes everything it made at the end.

Run:  python -m pytest test_my_colors_flow.py -q
"""

from __future__ import annotations

import os
import secrets
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

BASE = os.getenv("DEVICES_TEST_BASE", "http://127.0.0.1:8011").rstrip("/")
PW = "MyColors-Test-123!"
MC = "/api/palettes/my-colors"


def _reachable() -> bool:
    try:
        return requests.get(BASE + "/api/devices/config", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"backend not running at {BASE}")


def _req(method, path, token=None, json=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.request(method, BASE + path, json=json, headers=h, timeout=60)


def _hexes(palette):
    return [c["hex"] for c in palette["colors"]]


def _new_user():
    email = f"mycolors-test-{secrets.token_hex(4)}@example.com"
    r = _req("POST", "/api/auth/register", json={"email": email, "password": PW, "name": "My Colors Test"})
    assert r.status_code == 200, r.text
    d = r.json()
    return {"email": email, "token": d["access_token"], "id": d["user"]["id"]}


@pytest.fixture(scope="module")
def made():
    users = []
    yield users
    from auth_db import get_db

    ids = tuple(u["id"] for u in users)
    if not ids:
        return
    with get_db() as conn:
        cur = conn.cursor()
        for table in ("device_captures", "user_devices", "device_pair_codes", "user_palettes", "user_portfolio"):
            cur.execute(f"DELETE FROM {table} WHERE user_id IN %s", (ids,))
        cur.execute("DELETE FROM auth_users WHERE id IN %s", (ids,))
        conn.commit()


@pytest.fixture()
def user(made):
    u = _new_user()
    made.append(u)
    return u


def _device(u, name="Test phone"):
    r = _req("POST", "/api/devices/register", u["token"], {
        "device_id": "mc-test-" + secrets.token_hex(4), "device_type": "phone",
        "platform": "android", "name": name,
    })
    assert r.status_code == 200, r.text
    return r.json()["access_token"], r.json()["device"]["id"]


# ---------------------------------------------------------------------------

def test_requires_auth():
    assert _req("GET", MC).status_code == 401
    assert _req("POST", MC + "/colors", json={"hex": "#FEE100"}).status_code == 401


def test_empty_account(user):
    r = _req("GET", MC, user["token"])
    assert r.status_code == 200, r.text
    assert r.json() == {"id": None, "name": "My Colors", "colors": [], "updated_at": None}
    # a failed edit on an empty account must not leave a row behind
    assert _req("DELETE", MC + "/colors/FEE100", user["token"]).status_code == 404
    assert _req("GET", MC, user["token"]).json()["id"] is None


def test_add_rename_rehex_delete(user):
    t = user["token"]
    r = _req("POST", MC + "/colors", t, {"hex": "fee100", "name": "  Lemon  "})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["id"] and p["updated_at"] and p["colors"] == [{"hex": "#FEE100", "name": "Lemon"}]

    # same hex again: nothing changes (name kept)
    p2 = _req("POST", MC + "/colors", t, {"hex": "#FEE100", "name": "Other"}).json()
    assert p2["colors"] == p["colors"] and p2["updated_at"] == p["updated_at"]

    # no name -> the hex; 3-digit hex expands; appended at the end
    p = _req("POST", MC + "/colors", t, {"hex": "#abc"}).json()
    assert p["colors"][-1] == {"hex": "#AABBCC", "name": "#AABBCC"}

    # rename only
    p = _req("PATCH", MC + "/colors/FEE100", t, {"name": "Cadmium Yellow"}).json()
    assert p["colors"][0] == {"hex": "#FEE100", "name": "Cadmium Yellow"}
    # change hex only: the name stays
    p = _req("PATCH", MC + "/colors/fee100", t, {"hex": "#FFE000"}).json()
    assert p["colors"][0] == {"hex": "#FFE000", "name": "Cadmium Yellow"}
    # a name that was just the hex follows the new hex
    p = _req("PATCH", MC + "/colors/AABBCC", t, {"hex": "#112233"}).json()
    assert p["colors"][1] == {"hex": "#112233", "name": "#112233"}
    # blank name -> the hex
    p = _req("PATCH", MC + "/colors/112233", t, {"name": "   "}).json()
    assert p["colors"][1]["name"] == "#112233"

    # 404 / 409 / 400
    assert _req("PATCH", MC + "/colors/000000", t, {"name": "x"}).status_code == 404
    assert _req("PATCH", MC + "/colors/FFE000", t, {"hex": "#112233"}).status_code == 409
    assert _req("PATCH", MC + "/colors/FFE000", t, {"hex": "nothex"}).status_code == 400
    assert _req("PATCH", MC + "/colors/zzzzzz", t, {"name": "x"}).status_code == 400
    assert _req("POST", MC + "/colors", t, {"hex": "#12"}).status_code == 400

    # delete one, then 404 on the second try
    r = _req("DELETE", MC + "/colors/FFE000", t)
    assert r.status_code == 200 and _hexes(r.json()) == ["#112233"]
    assert _req("DELETE", MC + "/colors/FFE000", t).status_code == 404
    # removing the last color leaves an empty list (not a missing palette)
    p = _req("DELETE", MC + "/colors/112233", t).json()
    assert p["colors"] == [] and p["id"]
    assert _req("GET", MC, t).json()["colors"] == []


def test_merge_only_adds(user):
    t = user["token"]
    _req("POST", MC + "/colors", t, {"hex": "#111111", "name": "One"})
    r = _req("POST", MC + "/merge", t, {"colors": [
        {"hex": "#111111", "name": "renamed?"},   # already there: kept as is
        {"hex": "#222222", "name": "Two"},
        {"hex": "#222222", "name": "dup"},        # duplicate in the request
        {"hex": "#zz9", "name": "skipped"},       # invalid: skipped
        {"hex": "#333", "name": ""},              # no name -> hex
    ]})
    assert r.status_code == 200, r.text
    assert r.json()["colors"] == [
        {"hex": "#111111", "name": "One"},
        {"hex": "#222222", "name": "Two"},
        {"hex": "#333333", "name": "#333333"},
    ]
    # an empty merge changes nothing
    assert _hexes(_req("POST", MC + "/merge", t, {"colors": []}).json()) == ["#111111", "#222222", "#333333"]


def test_named_palettes_still_work_but_not_my_colors(user):
    t = user["token"]
    r = _req("POST", "/api/palettes", t, {"name": "Studio Set", "colors": [{"hex": "#010203", "name": "a"}]})
    assert r.status_code == 200 and r.json()["name"] == "Studio Set"
    r = _req("POST", "/api/palettes", t, {"name": "My Colors", "colors": [{"hex": "#010203"}]})
    assert r.status_code == 409
    assert _req("GET", MC, t).json()["colors"] == []
    names = [p["name"] for p in _req("GET", "/api/palettes", t).json()["palettes"]]
    assert names == ["Studio Set"]


def test_device_token_captures_and_web_edits_share_one_list(user):
    t = user["token"]
    dev, dev_row = _device(user)
    # web adds, the phone adds through the new endpoint and through a capture
    _req("POST", MC + "/colors", t, {"hex": "#AA0000", "name": "Web red"})
    r = _req("POST", MC + "/colors", dev, {"hex": "#00AA00", "name": "Phone green"})
    assert r.status_code == 200, r.text
    r = _req("POST", "/api/devices/captures", dev, {"hex": "#0000AA", "name": "Captured blue", "source": "camera"})
    assert r.status_code == 200, r.text
    # the web renames its color: the phone's colors survive (no whole-list replace)
    _req("PATCH", MC + "/colors/AA0000", t, {"name": "Web red 2"})
    p = _req("GET", MC, dev).json()
    assert sorted(_hexes(p)) == ["#0000AA", "#00AA00", "#AA0000"]
    assert {"hex": "#AA0000", "name": "Web red 2"} in p["colors"]
    # captures go to the front (newest first), web/app adds to the end
    assert p["colors"][0]["hex"] == "#0000AA"
    # the phone deletes the web color
    assert _req("DELETE", MC + "/colors/AA0000", dev).status_code == 200
    assert "#AA0000" not in _hexes(_req("GET", MC, t).json())
    # unpair the phone: its token is refused on these endpoints
    assert _req("DELETE", f"/api/devices/{dev_row}", t).status_code == 200
    r = _req("POST", MC + "/colors", dev, {"hex": "#123123"})
    assert r.status_code == 401 and r.json()["detail"] == "device_revoked"
    assert _req("GET", MC, dev).status_code == 401


def test_isolation_between_users(user, made):
    other = _new_user()
    made.append(other)
    _req("POST", MC + "/colors", user["token"], {"hex": "#ABCDEF", "name": "Mine"})
    assert _req("GET", MC, other["token"]).json()["colors"] == []
    assert _req("PATCH", MC + "/colors/ABCDEF", other["token"], {"name": "x"}).status_code == 404
    assert _req("DELETE", MC + "/colors/ABCDEF", other["token"]).status_code == 404
    assert _req("GET", MC, user["token"]).json()["colors"] == [{"hex": "#ABCDEF", "name": "Mine"}]


def test_concurrent_adds_on_a_fresh_account_all_land(user):
    """20 parallel adds (half as captures) on an account with no My Colors row yet.

    More requests than the Supabase session pooler allows connections (15): the
    app's connection pool (auth_db.get_db) makes them wait instead of failing."""
    t = user["token"]
    dev, _ = _device(user)
    hexes = ["#%02X%02X%02X" % (i * 11, 200 - i * 7, 40 + i * 5) for i in range(20)]

    def go(i):
        if i % 2:
            return _req("POST", "/api/devices/captures", dev, {"hex": hexes[i], "name": f"cap {i}"}).status_code
        return _req("POST", MC + "/colors", t, {"hex": hexes[i], "name": f"web {i}"}).status_code

    with ThreadPoolExecutor(max_workers=20) as ex:
        codes = list(ex.map(go, range(20)))
    assert codes == [200] * 20
    got = _hexes(_req("GET", MC, t).json())
    assert sorted(got) == sorted(hexes), f"lost {set(hexes) - set(got)}"


def test_concurrent_edits_do_not_overwrite(user):
    """Parallel renames of different colors + deletes + merges: every change survives."""
    t = user["token"]
    base = [{"hex": "#%06X" % (0x101010 * (i + 1)), "name": f"c{i}"} for i in range(10)]
    assert _req("POST", MC + "/merge", t, {"colors": base}).status_code == 200
    extra = [{"hex": "#%06X" % (0xABC000 + i), "name": f"x{i}"} for i in range(5)]

    jobs = [("PATCH", MC + "/colors/" + c["hex"][1:], {"name": c["name"].upper()}) for c in base[:6]]
    jobs += [("DELETE", MC + "/colors/" + c["hex"][1:], None) for c in base[6:]]
    jobs += [("POST", MC + "/merge", {"colors": [e]}) for e in extra]

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        codes = list(ex.map(lambda j: _req(j[0], j[1], t, j[2]).status_code, jobs))
    assert codes == [200] * len(jobs)
    p = _req("GET", MC, t).json()
    want = [{"hex": c["hex"], "name": c["name"].upper()} for c in base[:6]]
    assert p["colors"][:6] == want
    assert sorted(_hexes(p)[6:]) == sorted(e["hex"] for e in extra)


def test_cap(user):
    t = user["token"]
    many = [{"hex": "#%06X" % i, "name": ""} for i in range(1, 520)]
    p = _req("POST", MC + "/merge", t, {"colors": many}).json()
    assert len(p["colors"]) == 500                      # merge stops at the cap
    r = _req("POST", MC + "/colors", t, {"hex": "#FEFEFE"})
    assert r.status_code == 409 and "full" in r.json()["detail"]
    assert _req("POST", MC + "/colors", t, {"hex": "#000001"}).status_code == 200   # existing: no-op, not full
