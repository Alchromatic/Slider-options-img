#!/usr/bin/env python3
"""
test_color_engine.py
====================

Unit tests for the measured palette profile feature (Color-engine.docx).  These
verify the doc-review rules treated as mandatory:

* R3 -- profile is versioned and carries the required fields.
* R4 -- adding a color generates only the missing comparisons (incremental).
* R6 -- the TryColors client throttles, budgets, and retries (mocked; no live calls).
* R7 -- /unmix/custom uses measured behavior only when a profile covers a recipe,
        and otherwise falls back to the KM physical model.
* R8 -- the client refuses to run without a server-side key.

Run:  python -m pytest test_color_engine.py -q
"""

from __future__ import annotations

import json

import pytest

import measured_profile as mp
import trycolors_client as tc


# ---------------------------------------------------------------------------
# Fakes for the TryColors HTTP layer (no real network calls)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _queue_poster(responses):
    """Return a fake httpx.post that yields queued responses in order."""
    calls = {"n": 0}

    def _post(url, json=None, headers=None, timeout=None):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        r = responses[i]
        if isinstance(r, Exception):
            raise r
        return r

    _post.calls = calls
    return _post


# ---------------------------------------------------------------------------
# trycolors_client (R6, R8)
# ---------------------------------------------------------------------------

def _client(tmp_path, **kw):
    kw.setdefault("api_key", "test-key")
    kw.setdefault("rate_per_sec", 10_000)  # effectively no sleep in tests
    kw.setdefault("usage_path", tmp_path / "usage.json")
    return tc.TryColorsClient(**kw)


def test_missing_api_key_raises(tmp_path):
    client = _client(tmp_path, api_key="")
    with pytest.raises(tc.MissingApiKey):
        client.mix([{"hex": "#FFFFFF", "count": 1}, {"hex": "#000000", "count": 1}])


def test_mix_success_and_budget_decrement(tmp_path, monkeypatch):
    monkeypatch.setattr(tc.httpx, "post",
                        _queue_poster([_FakeResp(200, {"mixedColor": "#226A34"})]))
    client = _client(tmp_path, daily_budget=5)
    res = client.mix([{"hex": "#FEE100", "count": 1}, {"hex": "#19123F", "count": 1}])
    assert res.mixed_hex == "#226A34"
    assert res.source == "trycolors_api"
    assert res.mixer_mode == "pro" and res.engine == "2025"
    assert client.remaining_budget() == 4


def test_budget_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(tc.httpx, "post",
                        _queue_poster([_FakeResp(200, {"mixedColor": "#123456"})]))
    client = _client(tmp_path, daily_budget=0)
    with pytest.raises(tc.BudgetExhausted):
        client.mix([{"hex": "#FEE100", "count": 1}, {"hex": "#19123F", "count": 1}])


def test_retry_then_success(tmp_path, monkeypatch):
    poster = _queue_poster([
        _FakeResp(429, text="rate limited", headers={"Retry-After": "0"}),
        _FakeResp(200, {"mixedColor": "#ABCDEF"}),
    ])
    monkeypatch.setattr(tc.httpx, "post", poster)
    client = _client(tmp_path, daily_budget=5)
    res = client.mix([{"hex": "#FEE100", "count": 1}, {"hex": "#19123F", "count": 1}])
    assert res.mixed_hex == "#ABCDEF"
    assert poster.calls["n"] == 2          # retried once
    assert client.remaining_budget() == 4  # only the success counts


def test_non_retryable_4xx_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(tc.httpx, "post",
                        _queue_poster([_FakeResp(400, text="bad request")]))
    client = _client(tmp_path, daily_budget=5)
    with pytest.raises(tc.TryColorsError):
        client.mix([{"hex": "#FEE100", "count": 1}, {"hex": "#19123F", "count": 1}])


# ---------------------------------------------------------------------------
# measured_profile schema + incremental (R3, R4)
# ---------------------------------------------------------------------------

THREE = [
    {"name": "Cad Yellow", "hex": "#FEE100"},
    {"name": "Ultramarine", "hex": "#19123F"},
    {"name": "Cad Red", "hex": "#DE290C"},
]


def test_profile_schema_has_required_fields(tmp_path):
    prof = mp.new_profile(THREE, palette_id="p1", palette_name="P1")
    for field in ("schema_version", "profile_version", "palette_id", "colors",
                  "engine", "mixer_mode", "generated_at", "comparisons", "completeness"):
        assert field in prof
    assert prof["engine"] == "2025" and prof["mixer_mode"] == "pro"
    mp.recompute_completeness(prof)
    # 3 colors -> 3 pairs x 3 ratios = 9 expected comparisons
    assert prof["completeness"]["expected_count"] == 9


def test_save_load_roundtrip_and_version_bump(tmp_path):
    prof = mp.new_profile(THREE, palette_id="p1")
    miss = mp.diff_missing(prof)
    mp.add_results(prof, [{"pigments": m["pigments"], "parts": m["parts"],
                           "trycolors_result_hex": "#888888", "source": "trycolors_api"}
                          for m in miss])
    mp.recompute_completeness(prof)
    mp.bump_version(prof)
    mp.save_profile(prof, profiles_dir=tmp_path)
    loaded = mp.load_profile("p1", profiles_dir=tmp_path)
    assert loaded["profile_version"] == 2
    assert loaded["completeness"]["complete"] is True
    assert loaded["source"] == "trycolors_api"


def test_incremental_add_color_only_new_pairs(tmp_path):
    # start with the 3-color palette, fully measured
    prof = mp.new_profile(THREE, palette_id="p1")
    miss = mp.diff_missing(prof)
    mp.add_results(prof, [{"pigments": m["pigments"], "parts": m["parts"],
                           "trycolors_result_hex": "#888888"} for m in miss])
    mp.recompute_completeness(prof)
    assert prof["completeness"]["complete"]

    # add a 4th color -> ids preserved, only its 3 pairs (x3 ratios) are missing
    prof, added = mp.update_profile_colors(prof, THREE + [{"name": "White", "hex": "#F7F5F1"}])
    assert added == ["c4"]
    assert [c["id"] for c in prof["colors"]] == ["c1", "c2", "c3", "c4"]
    new_missing = mp.diff_missing(prof)
    assert len(new_missing) == 9  # 3 new pairs x 3 ratios
    for m in new_missing:
        assert "c4" in m["pigments"]  # every missing comparison involves the new color


# ---------------------------------------------------------------------------
# MeasuredProfileModel coverage + prediction (R1, R7 gate)
# ---------------------------------------------------------------------------

def test_model_covers_only_measured_pairs(tmp_path):
    prof = mp.new_profile(THREE, palette_id="p1")
    # measure ONLY the c1-c2 pair at 1:1
    mp.add_results(prof, [{"pigments": ["c1", "c2"], "parts": [1, 1],
                           "trycolors_result_hex": "#445566", "source": "trycolors_api"}])
    model = mp.MeasuredProfileModel(prof)
    # single color always covered; predicts its own hex
    assert model.covers(["c1"], [1]) is True
    assert model.predict_recipe(["c1"], [1])[0] == prof["colors"][0]["hex"]
    # measured pair covered; predicts the stored measured hex at the exact ratio
    assert model.covers(["c1", "c2"], [1, 1]) is True
    assert model.predict_recipe(["c1", "c2"], [1, 1])[0] == "#445566"
    # an unmeasured pair is NOT covered (endpoint-only) -> KM fallback territory
    assert model.covers(["c2", "c3"], [1, 1]) is False


# ---------------------------------------------------------------------------
# /unmix/custom profile-aware behavior (R7)
# ---------------------------------------------------------------------------

def test_unmix_falls_back_to_km_without_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_PROFILES_DIR", tmp_path)  # empty -> no profile
    import custom_palette_unmix as cpu
    r = cpu.unmix_custom_palette("#8C7A20", THREE, top_n=3)
    assert r["palette_mode"] == "custom_user_palette"
    assert all(p["mix_source"] == "physical_km" for p in r["proposals"])


def test_unmix_uses_measured_profile_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_PROFILES_DIR", tmp_path)
    # build + save a fully-measured profile for this exact color set
    prof = mp.new_profile(THREE, palette_id="studio")
    miss = mp.diff_missing(prof)
    mp.add_results(prof, [{"pigments": m["pigments"], "parts": m["parts"],
                           "trycolors_result_hex": "#777777", "source": "trycolors_api"}
                          for m in miss])
    mp.recompute_completeness(prof)
    mp.save_profile(prof, profiles_dir=tmp_path)

    import custom_palette_unmix as cpu
    r = cpu.unmix_custom_palette("#777777", THREE, palette_id="studio", top_n=3)
    assert r["palette_mode"] == "custom_measured_profile"
    assert r["measured_profile"]["palette_id"] == "studio"
    assert r["candidate_mix_sources"]["measured_profile"] > 0
    assert r["candidate_mix_sources"]["physical_km"] == 0
