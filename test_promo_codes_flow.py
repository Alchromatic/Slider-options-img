#!/usr/bin/env python3
"""
test_promo_codes_flow.py
========================

Live integration test for promo_routes.py (admin-generated codes that give free
images). Same setup as test_devices_flow.py: it talks to a RUNNING backend
(default http://127.0.0.1:8011, "backend-verify" launch config; override with
DEVICES_TEST_BASE) and the real Supabase DB. It needs ADMIN_TOKEN (read from
.env) and cleans up the users, codes and redemptions it made.

Run:  python -m pytest test_promo_codes_flow.py -q
"""

from __future__ import annotations

import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

BASE = os.getenv("DEVICES_TEST_BASE", "http://127.0.0.1:8011").rstrip("/")
ADMIN = os.getenv("ADMIN_TOKEN", "")
PW = "Promo-Test-123!"
RUN = "pytest-promo-" + secrets.token_hex(3)       # batch label prefix for this run


def _reachable() -> bool:
    try:
        return requests.get(BASE + "/api/devices/config", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _reachable(), reason=f"backend not running at {BASE}"),
    pytest.mark.skipif(not ADMIN, reason="ADMIN_TOKEN not set"),
]


def _req(method, path, token=None, json=None, admin=False):
    h = {}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if admin:
        h["X-Admin-Token"] = ADMIN
    return requests.request(method, BASE + path, json=json, headers=h, timeout=60)


def _gen(batch_suffix, **body):
    body.setdefault("images", 5)
    body["batch"] = f"{RUN}-{batch_suffix}"
    r = _req("POST", "/api/admin/promo-codes", json=body, admin=True)
    assert r.status_code == 200, r.text
    return r.json()


def _credits(token):
    return _req("GET", "/api/billing/me", token).json()["credits"]


@pytest.fixture(scope="module")
def made():
    users = []
    yield users
    from auth_db import get_db

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM promo_redemptions WHERE code IN (SELECT code FROM promo_codes WHERE batch LIKE %s)", (RUN + "%",))
        cur.execute("DELETE FROM promo_codes WHERE batch LIKE %s", (RUN + "%",))
        ids = tuple(u["id"] for u in users)
        if ids:
            for t in ("promo_redemptions", "image_credits", "image_usage", "user_devices", "device_pair_codes", "user_palettes"):
                cur.execute(f"DELETE FROM {t} WHERE user_id IN %s", (ids,))
            cur.execute("DELETE FROM auth_users WHERE id IN %s", (ids,))
        conn.commit()


def _new_user(made):
    email = f"promo-test-{secrets.token_hex(4)}@example.com"
    r = _req("POST", "/api/auth/register", json={"email": email, "password": PW, "name": "Promo Test"})
    assert r.status_code == 200, r.text
    d = r.json()
    u = {"email": email, "token": d["access_token"], "id": d["user"]["id"]}
    made.append(u)
    return u


@pytest.fixture()
def user(made):
    return _new_user(made)


# ---------------------------------------------------------------------------

def test_admin_endpoints_need_the_admin_token():
    assert _req("POST", "/api/admin/promo-codes", json={"images": 1}).status_code == 401
    assert _req("GET", "/api/admin/promo-codes").status_code == 401
    r = requests.get(BASE + "/api/admin/promo-codes", headers={"X-Admin-Token": "wrong"}, timeout=30)
    assert r.status_code == 401


def test_redeem_needs_a_signed_in_user():
    assert _req("POST", "/api/billing/redeem-code", json={"code": "GM-AAAAA-AAAAA"}).status_code == 401
    assert _req("POST", "/api/billing/redeem-code", "not-a-jwt", json={"code": "GM-AAAAA-AAAAA"}).status_code == 401


def test_generate_and_redeem_adds_images(user):
    g = _gen("basic", count=3, images=7, note="pytest")
    assert len(g["codes"]) == 3 and len(set(g["codes"])) == 3
    for c in g["codes"]:
        assert c.startswith("GM-") and len(c) == len("GM-XXXXX-XXXXX")
        assert not set(c.replace("GM-", "").replace("-", "")) & set("01IO")
    before = _req("GET", "/api/billing/me", user["token"]).json()
    # lower case, spaces instead of dashes: still the same code
    typed = g["codes"][0].lower().replace("-", " ")
    r = _req("POST", "/api/billing/redeem-code", user["token"], {"code": typed})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] and d["images"] == 7 and d["code"] == g["codes"][0]
    assert d["credits"] == before["credits"] + 7
    assert d["entitlements"]["images_remaining"] == before["images_remaining"] + 7
    # the use is recorded against the user
    mine = _req("GET", "/api/billing/redemptions", user["token"]).json()["redemptions"]
    assert [m["code"] for m in mine] == [g["codes"][0]] and mine[0]["images"] == 7
    listed = _req("GET", f"/api/admin/promo-codes?batch={g['batch']}", admin=True).json()
    used = [c for c in listed["codes"] if c["code"] == g["codes"][0]][0]
    assert used["status"] == "used" and used["redeemed_count"] == 1
    assert used["redemptions"][0]["user_id"] == user["id"] and used["redemptions"][0]["email"] == user["email"]
    # search by the user's email finds the code
    found = _req("GET", "/api/admin/promo-codes?q=" + user["email"], admin=True).json()["codes"]
    assert [c["code"] for c in found] == [g["codes"][0]]


def test_single_use_and_once_per_user(made):
    a, b = _new_user(made), _new_user(made)
    code = _gen("single", images=3)["codes"][0]
    assert _req("POST", "/api/billing/redeem-code", a["token"], {"code": code}).status_code == 200
    r = _req("POST", "/api/billing/redeem-code", a["token"], {"code": code})
    assert r.status_code == 409 and "already used this code" in r.json()["detail"]
    r = _req("POST", "/api/billing/redeem-code", b["token"], {"code": code})
    assert r.status_code == 409 and "already been used" in r.json()["detail"]
    assert _credits(b["token"]) == 0


def test_multi_use_code_once_per_user(made):
    users = [_new_user(made) for _ in range(3)]
    code = _gen("multi", images=2, max_redemptions=2, prefix="")["codes"][0]
    assert len(code) == len("XXXXX-XXXXX")                  # no prefix
    codes = [_req("POST", "/api/billing/redeem-code", u["token"], {"code": code}).status_code for u in users]
    assert codes == [200, 200, 409]


def test_custom_code(made):
    u = _new_user(made)
    word = "PYT" + secrets.token_hex(3).upper()
    g = _gen("custom", code=word.lower(), images=4, max_redemptions=50)
    assert g["codes"] == [word]
    assert _req("POST", "/api/admin/promo-codes", json={"code": word, "images": 1, "batch": RUN + "-dup"}, admin=True).status_code == 409
    assert _req("POST", "/api/admin/promo-codes", json={"code": "AB", "images": 1, "batch": RUN + "-x"}, admin=True).status_code == 400
    assert _req("POST", "/api/admin/promo-codes", json={"code": word + "Z", "count": 2, "images": 1, "batch": RUN + "-x"}, admin=True).status_code == 400
    r = _req("POST", "/api/billing/redeem-code", u["token"], {"code": word})
    assert r.status_code == 200 and r.json()["images"] == 4


def test_expired_and_disabled_codes(made):
    u = _new_user(made)
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    assert _req("POST", "/api/admin/promo-codes", json={"images": 1, "expires_at": past, "batch": RUN + "-past"}, admin=True).status_code == 400
    soon = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    code = _gen("expiring", images=1, expires_at=soon)["codes"][0]
    # make it expire now, straight in the DB
    from auth_db import get_db
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE promo_codes SET expires_at = NOW() - interval '1 minute' WHERE batch = %s", (RUN + "-expiring",))
        conn.commit()
    r = _req("POST", "/api/billing/redeem-code", u["token"], {"code": code})
    assert r.status_code == 410 and "expired" in r.json()["detail"]

    g = _gen("disable", count=2, images=1)
    r = _req("PATCH", "/api/admin/promo-codes/" + g["codes"][0], json={"disabled": True}, admin=True)
    assert r.status_code == 200 and r.json()["code"]["status"] == "disabled"
    r = _req("POST", "/api/billing/redeem-code", u["token"], {"code": g["codes"][0]})
    assert r.status_code == 410 and "no longer active" in r.json()["detail"]
    # a whole batch off, then on again
    r = _req("POST", "/api/admin/promo-codes/batches/disable", json={"batch": g["batch"], "disabled": True}, admin=True)
    assert r.status_code == 200 and r.json()["codes"] == 2
    assert _req("POST", "/api/billing/redeem-code", u["token"], {"code": g["codes"][1]}).status_code == 410
    _req("POST", "/api/admin/promo-codes/batches/disable", json={"batch": g["batch"], "disabled": False}, admin=True)
    assert _req("POST", "/api/billing/redeem-code", u["token"], {"code": g["codes"][1]}).status_code == 200
    assert _credits(u["token"]) == 1


def test_bad_codes_and_throttle(user):
    r = _req("POST", "/api/billing/redeem-code", user["token"], {"code": "!!"})
    assert r.status_code == 400
    for _ in range(8):
        r = _req("POST", "/api/billing/redeem-code", user["token"], {"code": "GM-ZZZZZ-" + secrets.token_hex(3).upper()})
        assert r.status_code == 404 and r.json()["detail"] == "That code isn't valid."
    # 1 + 8 failures so far; the 10th failure is still answered, the 11th try is refused
    assert _req("POST", "/api/billing/redeem-code", user["token"], {"code": "GM-ZZZZZ-ZZZZZ"}).status_code == 404
    assert _req("POST", "/api/billing/redeem-code", user["token"], {"code": "GM-ZZZZZ-ZZZZZ"}).status_code == 429


def test_parallel_redeems_cannot_overuse_a_code(made):
    users = [_new_user(made) for _ in range(10)]
    code = _gen("race", images=3, max_redemptions=5)["codes"][0]
    with ThreadPoolExecutor(max_workers=10) as ex:
        codes = list(ex.map(lambda u: _req("POST", "/api/billing/redeem-code", u["token"], {"code": code}).status_code, users))
    assert sorted(codes) == [200] * 5 + [409] * 5
    listed = _req("GET", f"/api/admin/promo-codes?batch={RUN}-race", admin=True).json()["codes"][0]
    assert listed["redeemed_count"] == 5 and len(listed["redemptions"]) == 5


def test_double_submit_by_one_user_counts_once(user):
    code = _gen("double", images=6, max_redemptions=10)["codes"][0]
    with ThreadPoolExecutor(max_workers=5) as ex:
        codes = list(ex.map(lambda _: _req("POST", "/api/billing/redeem-code", user["token"], {"code": code}).status_code, range(5)))
    assert sorted(codes) == [200] + [409] * 4
    assert _credits(user["token"]) == 6


def test_code_images_are_spent_after_the_monthly_allowance(user):
    me = _req("GET", "/api/billing/me", user["token"]).json()
    assert me["plan_id"] == "free"
    monthly = me["images_remaining"] - me["credits"]
    code = _gen("spend", images=2)["codes"][0]
    assert _req("POST", "/api/billing/redeem-code", user["token"], {"code": code}).status_code == 200
    sources = []
    for _ in range(monthly + 3):
        d = _req("POST", "/api/billing/consume-image", user["token"]).json()
        sources.append(d.get("source") if d.get("allowed") else "denied")
    assert sources == ["plan"] * monthly + ["credit", "credit", "denied"]


def test_admin_list_filters_and_batches():
    g = _gen("filters", count=4, images=9)
    r = _req("GET", f"/api/admin/promo-codes?batch={g['batch']}&status=active&limit=2", admin=True).json()
    assert r["total"] == 4 and len(r["codes"]) == 2
    one = g["codes"][0]
    r = _req("GET", "/api/admin/promo-codes?q=" + one.lower(), admin=True).json()
    assert [c["code"] for c in r["codes"]] == [one]
    batches = _req("GET", "/api/admin/promo-codes/batches", admin=True).json()["batches"]
    b = [x for x in batches if x["batch"] == g["batch"]][0]
    assert b["codes"] == 4 and b["images"] == 9 and b["redemptions"] == 0 and b["capacity"] == 4
