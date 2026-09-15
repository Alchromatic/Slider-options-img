#!/usr/bin/env python3
"""
test_devices_flow.py
====================

Live integration test for devices_routes.py (device <-> user pairing and
captures).  It talks to a RUNNING backend (default http://127.0.0.1:8011, see
the "backend-verify" launch config) and therefore to the real Supabase DB,
so it creates its own throwaway users and deletes everything it made at the
end.

Skipped automatically when the server is not reachable.

Run:  python -m pytest test_devices_flow.py -q
"""

from __future__ import annotations

import os
import secrets
import time

import pytest
import requests

BASE = os.getenv("DEVICES_TEST_BASE", "http://127.0.0.1:8011").rstrip("/")
PW = "Devices-Test-123!"


def _reachable() -> bool:
    try:
        return requests.get(BASE + "/api/devices/config", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"backend not running at {BASE}")


def _post(path, json=None, token=None, **kw):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.post(BASE + path, json=json, headers=h, timeout=30, **kw)


def _get(path, token=None, **kw):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.get(BASE + path, headers=h, timeout=30, **kw)


def _delete(path, token=None):
    return requests.delete(BASE + path, headers={"Authorization": f"Bearer {token}"}, timeout=30)


def _patch(path, json, token):
    return requests.patch(BASE + path, json=json, headers={"Authorization": f"Bearer {token}"}, timeout=30)


@pytest.fixture(scope="module")
def users():
    """Two throwaway accounts; rows are removed after the module."""
    made = []
    for _ in range(2):
        email = f"devices-test-{secrets.token_hex(4)}@example.com"
        r = _post("/api/auth/register", {"email": email, "password": PW, "name": "Devices Test"})
        assert r.status_code == 200, r.text
        d = r.json()
        made.append({"email": email, "token": d["access_token"], "id": d["user"]["id"]})
    yield made
    # ---- cleanup straight in the DB (there is no user-delete endpoint) ----
    from auth_db import get_db

    ids = tuple(u["id"] for u in made)
    with get_db() as conn:
        cur = conn.cursor()
        for table in ("device_captures", "user_devices", "device_pair_codes", "user_palettes"):
            cur.execute(f"DELETE FROM {table} WHERE user_id IN %s", (ids,))
        cur.execute("DELETE FROM auth_users WHERE id IN %s", (ids,))
        conn.commit()


def test_config_is_public():
    r = requests.get(BASE + "/api/devices/config", timeout=10)
    assert r.status_code == 200
    d = r.json()
    assert d["deep_link_scheme"] and "apps" in d and d["pair_code_ttl"] > 0


def test_flow_a_device_shows_code(users):
    a = users[0]
    dev = {"device_id": "test-quest-" + secrets.token_hex(3), "device_type": "headset",
           "platform": "quest", "name": "Test Quest", "app_version": "0.1-test"}

    r = _post("/api/devices/pair/start", dev)
    assert r.status_code == 200, r.text
    start = r.json()
    assert len(start["code"]) == 6 and start["code"].isdigit()
    assert start["poll_secret"] and start["expires_in"] > 0

    # nothing claimed yet
    r = _get("/api/devices/pair/status", params={"code": start["code"], "poll_secret": start["poll_secret"]})
    assert r.status_code == 200 and r.json()["status"] == "pending"

    # wrong secret is not accepted
    r = _get("/api/devices/pair/status", params={"code": start["code"], "poll_secret": "nope"})
    assert r.status_code == 404

    # wrong code from the web is a clean 404
    r = _post("/api/devices/pair/claim", {"code": "000000"}, token=a["token"])
    assert r.status_code == 404

    # a device token cannot claim (must be a real user login)
    r = _post("/api/devices/pair/claim", {"code": start["code"]}, token="not-a-jwt")
    assert r.status_code == 401

    # claim from the webapp with spaces in the code (users type "123 456")
    spaced = start["code"][:3] + " " + start["code"][3:]
    r = _post("/api/devices/pair/claim", {"code": spaced}, token=a["token"])
    assert r.status_code == 200, r.text
    assert r.json()["paired"] is True
    assert r.json()["device"]["device_id"] == dev["device_id"]
    assert r.json()["device"]["device_type"] == "headset"

    # device polls and gets its session exactly once
    r = _get("/api/devices/pair/status", params={"code": start["code"], "poll_secret": start["poll_secret"]})
    assert r.status_code == 200, r.text
    sess = r.json()
    assert sess["status"] == "claimed"
    assert sess["access_token"] and sess["token_type"] == "device"
    assert sess["user"]["id"] == a["id"]
    assert sess["device"]["device_id"] == dev["device_id"]

    r = _get("/api/devices/pair/status", params={"code": start["code"], "poll_secret": start["poll_secret"]})
    assert r.json()["status"] == "consumed"

    # the code can't be claimed twice
    r = _post("/api/devices/pair/claim", {"code": start["code"]}, token=a["token"])
    assert r.status_code == 404

    a["quest_token"] = sess["access_token"]
    a["quest_device_id"] = dev["device_id"]
    a["quest_row_id"] = sess["device"]["id"]


def test_capture_from_device_lands_in_my_colors(users):
    a = users[0]
    r = _post(
        "/api/devices/captures",
        {
            "hex": "1e2448", "name": "Night Sky", "source": "unmix",
            "recipe": [{"name": "Ultramarine Blue", "hex": "#19123F", "percentage": 94.6, "parts": 19},
                       {"name": "Titanium White", "hex": "F7F5F1", "percentage": 5.4, "parts": 1}],
            "meta": {"scene": "ColorUnmix"},
        },
        token=a["quest_token"],
    )
    assert r.status_code == 200, r.text
    cap = r.json()["capture"]
    assert cap["hex"] == "#1E2448" and cap["rgb"] == "30,36,72"
    assert cap["name"] == "Night Sky" and cap["source"] == "unmix"
    assert cap["device_id"] == a["quest_device_id"] and cap["device_type"] == "headset"
    assert cap["device_name"] is None or cap["device_name"] == "Test Quest"
    assert cap["recipe"][0]["hex"] == "#19123F" and cap["recipe"][1]["hex"] == "#F7F5F1"
    a["capture_id"] = cap["id"]

    # invalid hex rejected
    r = _post("/api/devices/captures", {"hex": "zzz"}, token=a["quest_token"])
    assert r.status_code == 400

    # ... and it is now the first color in "My Colors"
    r = _get("/api/palettes", token=a["token"])
    assert r.status_code == 200, r.text
    mine = [p for p in r.json()["palettes"] if p["name"] == "My Colors"]
    assert mine, "capture did not create the My Colors palette"
    assert mine[0]["colors"][0] == {"hex": "#1E2448", "name": "Night Sky"}

    # saving the same hex again does not duplicate it in the library
    r = _post("/api/devices/captures", {"hex": "#1E2448", "name": "Night Sky again"}, token=a["quest_token"])
    assert r.status_code == 200
    r = _get("/api/palettes", token=a["token"])
    mine = [p for p in r.json()["palettes"] if p["name"] == "My Colors"][0]
    assert sum(1 for c in mine["colors"] if c["hex"] == "#1E2448") == 1

    # add_to_library=false keeps it out of the palette
    r = _post("/api/devices/captures", {"hex": "#ABCDEF", "add_to_library": False}, token=a["quest_token"])
    assert r.status_code == 200
    r = _get("/api/palettes", token=a["token"])
    mine = [p for p in r.json()["palettes"] if p["name"] == "My Colors"][0]
    assert not any(c["hex"] == "#ABCDEF" for c in mine["colors"])


def test_heartbeat_and_device_list(users):
    a = users[0]
    r = _post("/api/devices/heartbeat", token=a["quest_token"])
    assert r.status_code == 200 and r.json()["ok"] is True

    # a plain user token has no device to heartbeat
    r = _post("/api/devices/heartbeat", token=a["token"])
    assert r.status_code == 400

    r = _get("/api/devices", token=a["token"])
    assert r.status_code == 200
    devs = r.json()["devices"]
    assert len(devs) == 1
    assert devs[0]["device_id"] == a["quest_device_id"]
    assert devs[0]["online"] is True and devs[0]["name"] == "Test Quest"


def test_flow_b_web_code_redeemed_by_phone(users):
    a = users[0]
    r = _post("/api/devices/pair/web-code", token=a["token"])
    assert r.status_code == 200, r.text
    wc = r.json()
    assert len(wc["code"]) == 8 and wc["deep_link"].endswith("code=" + wc["code"])
    assert not set(wc["code"]) & set("01OI")

    phone = {"device_id": "test-phone-" + secrets.token_hex(3), "device_type": "phone",
             "platform": "android", "name": "Test Pixel"}

    # wrong code
    r = _post("/api/devices/pair/redeem", dict(phone, code="ZZZZZZZZ"))
    assert r.status_code == 404

    # lowercase / dashed input is normalised
    messy = wc["code"][:4].lower() + "-" + wc["code"][4:]
    r = _post("/api/devices/pair/redeem", dict(phone, code=messy))
    assert r.status_code == 200, r.text
    sess = r.json()
    assert sess["token_type"] == "device" and sess["user"]["id"] == a["id"]
    assert sess["device"]["platform"] == "android"

    # single use
    r = _post("/api/devices/pair/redeem", dict(phone, code=wc["code"]))
    assert r.status_code == 404

    # a new web code invalidates the previous pending one
    r1 = _post("/api/devices/pair/web-code", token=a["token"]).json()
    r2 = _post("/api/devices/pair/web-code", token=a["token"]).json()
    assert _post("/api/devices/pair/redeem", dict(phone, code=r1["code"])).status_code == 404
    assert _post("/api/devices/pair/redeem", dict(phone, code=r2["code"])).status_code == 200

    a["phone_token"] = sess["access_token"]
    a["phone_device_id"] = phone["device_id"]


def test_flow_c_register_with_user_login(users):
    a = users[0]
    tablet = {"device_id": "test-tablet-" + secrets.token_hex(3), "device_type": "tablet", "platform": "ios"}
    r = _post("/api/devices/register", tablet, token=a["token"])
    assert r.status_code == 200, r.text
    sess = r.json()
    assert sess["token_type"] == "device" and sess["device"]["device_type"] == "tablet"

    # registering again with the device token refreshes (idempotent)
    r = _post("/api/devices/register", dict(tablet, name="iPad"), token=sess["access_token"])
    assert r.status_code == 200 and r.json()["device"]["name"] == "iPad"

    # a device token cannot register a *different* device id
    r = _post("/api/devices/register", dict(tablet, device_id="someone-else"), token=sess["access_token"])
    assert r.status_code == 403

    r = _get("/api/devices", token=a["token"])
    assert len(r.json()["devices"]) == 3
    a["tablet_row_id"] = sess["device"]["id"]


def test_batch_list_filter_and_isolation(users):
    a, b = users
    r = _post(
        "/api/devices/captures/batch",
        {"captures": [{"hex": "#FF0000", "name": "Red", "source": "camera"},
                      {"hex": "#00FF00", "name": "Green", "source": "picker"}]},
        token=a["phone_token"],
    )
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2

    r = _get("/api/devices/captures", token=a["token"])
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 5            # 3 from the quest + 2 from the phone
    assert d["captures"][0]["hex"] in ("#FF0000", "#00FF00")   # newest first
    assert d["captures"][0]["device_name"] == "Test Pixel"

    r = _get("/api/devices/captures", token=a["token"], params={"device_id": a["quest_device_id"]})
    assert r.json()["total"] == 3
    r = _get("/api/devices/captures", token=a["token"], params={"source": "picker"})
    assert r.json()["total"] == 1 and r.json()["captures"][0]["name"] == "Green"
    r = _get("/api/devices/captures", token=a["token"], params={"limit": 2, "offset": 4})
    assert len(r.json()["captures"]) == 1

    # the device itself can read the user's captures (for the in-app list)
    r = _get("/api/devices/captures", token=a["quest_token"])
    assert r.json()["total"] == 5

    # user B sees none of it and cannot delete it
    r = _get("/api/devices/captures", token=b["token"])
    assert r.json()["total"] == 0
    assert _get("/api/devices", token=b["token"]).json()["devices"] == []
    assert _delete(f"/api/devices/captures/{a['capture_id']}", token=b["token"]).status_code == 404
    assert _delete(f"/api/devices/{a['quest_row_id']}", token=b["token"]).status_code == 404

    # no token at all
    assert requests.get(BASE + "/api/devices/captures", timeout=10).status_code == 401


def test_rename_revoke_and_repair(users):
    a = users[0]
    r = _patch(f"/api/devices/{a['quest_row_id']}", {"name": "Studio Quest"}, token=a["token"])
    assert r.status_code == 200 and r.json()["device"]["name"] == "Studio Quest"

    # revoke from the webapp -> device token stops working immediately
    r = _delete(f"/api/devices/{a['quest_row_id']}", token=a["token"])
    assert r.status_code == 200 and r.json()["revoked"] is True
    assert _delete(f"/api/devices/{a['quest_row_id']}", token=a["token"]).status_code == 404  # already revoked

    r = _post("/api/devices/captures", {"hex": "#123456"}, token=a["quest_token"])
    assert r.status_code == 401 and r.json()["detail"] == "device_revoked"
    assert _post("/api/devices/heartbeat", token=a["quest_token"]).status_code == 401
    assert _get("/api/devices/captures", token=a["quest_token"]).status_code == 401

    r = _get("/api/devices", token=a["token"])
    assert {d["device_id"] for d in r.json()["devices"]} == {a["phone_device_id"]} | {
        d["device_id"] for d in r.json()["devices"] if d["device_type"] == "tablet"}
    assert len(r.json()["devices"]) == 2

    # captures made by the revoked device are kept (history) ...
    assert _get("/api/devices/captures", token=a["token"]).json()["total"] == 5

    # ... and re-pairing the same device (flow C) un-revokes it
    r = _post("/api/devices/register", {"device_id": a["quest_device_id"], "device_type": "headset"}, token=a["token"])
    assert r.status_code == 200
    assert r.json()["device"]["id"] == a["quest_row_id"]      # same row, not a duplicate
    assert _post("/api/devices/heartbeat", token=r.json()["access_token"]).status_code == 200
    assert len(_get("/api/devices", token=a["token"]).json()["devices"]) == 3


def test_delete_capture(users):
    a = users[0]
    r = _delete(f"/api/devices/captures/{a['capture_id']}", token=a["token"])
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert _delete(f"/api/devices/captures/{a['capture_id']}", token=a["token"]).status_code == 404
    assert _get("/api/devices/captures", token=a["token"]).json()["total"] == 4


def test_claim_is_throttled(users):
    """10 bad codes in a row -> the 11th attempt is refused with 429 (per user)."""
    b = users[1]
    statuses = [_post("/api/devices/pair/claim", {"code": "999999"}, token=b["token"]).status_code for _ in range(11)]
    assert statuses[:10] == [404] * 10
    assert statuses[10] == 429
