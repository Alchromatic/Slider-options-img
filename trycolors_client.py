#!/usr/bin/env python3
"""
trycolors_client.py
===================

Server-side client for the official TryColors REST API, used to *measure* how
colors mix when generating a measured palette profile (see Color-engine.docx).

This is the ONLY module that reads the TryColors API key.  It enforces the rules
the doc review made mandatory:

* R8 -- the API key is read from the ``TRYCOLORS_API_KEY`` environment variable
  (server-side only).  It is never accepted from a request, logged, or returned.
* R6 -- the client throttles to the published rate limit (<=2 requests/second),
  enforces the published daily budget (<=50 mixes/day), and retries transient
  failures (HTTP 429 / 5xx) with exponential backoff.  When the daily budget is
  exhausted it raises :class:`BudgetExhausted` so the generator can fall back to
  the local mixing engine (the "hybrid" strategy) instead of stalling.
* R2 -- verified live: ``POST /v1/mix-colors`` accepts and honors
  ``mixerMode="pro"`` and ``engine="2025"`` (auth via the ``X-API-KEY`` header).

The daily-usage counter is persisted to ``profiles/.usage.json`` so the budget is
respected across separate process runs (generation is meant to be resumable).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx

try:  # load .env if python-dotenv is available (it is a project dependency)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


API_BASE = "https://api.trycolors.com/v1"
MIX_ENDPOINT = f"{API_BASE}/mix-colors"

# Published free-tier limits for the mix endpoint (see api.trycolors.com).
DEFAULT_DAILY_BUDGET = 50
DEFAULT_RATE_PER_SEC = 2.0

_USAGE_PATH = Path(__file__).resolve().parent / "profiles" / ".usage.json"


class TryColorsError(RuntimeError):
    """Base error for TryColors client failures."""


class BudgetExhausted(TryColorsError):
    """Raised when the daily mix budget has been spent.

    The profile generator catches this and falls back to the local engine.
    """


class MissingApiKey(TryColorsError):
    """Raised when no TRYCOLORS_API_KEY is configured."""


@dataclass
class MixResult:
    mixed_hex: str
    source: str  # always "trycolors_api" for this client
    mixer_mode: str
    engine: str


def _today_str() -> str:
    # Local date is fine; the budget is a coarse daily guard, not billing.
    return time.strftime("%Y-%m-%d", time.localtime())


class TryColorsClient:
    """Throttled, budgeted client for the TryColors mix endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        mixer_mode: str = "pro",
        engine: str = "2025",
        daily_budget: int = DEFAULT_DAILY_BUDGET,
        rate_per_sec: float = DEFAULT_RATE_PER_SEC,
        max_retries: int = 4,
        timeout: float = 20.0,
        usage_path: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("TRYCOLORS_API_KEY", "")
        self.mixer_mode = mixer_mode
        self.engine = engine
        self.daily_budget = int(daily_budget)
        self._min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self._usage_path = Path(usage_path) if usage_path else _USAGE_PATH
        self._lock = threading.Lock()
        self._last_call_ts = 0.0

    # ------------------------------------------------------------------ usage
    def _read_usage(self) -> Dict[str, int]:
        try:
            data = json.loads(self._usage_path.read_text())
        except Exception:
            data = {}
        if data.get("date") != _today_str():
            return {"date": _today_str(), "count": 0}
        return {"date": data["date"], "count": int(data.get("count", 0))}

    def _write_usage(self, usage: Dict[str, int]) -> None:
        self._usage_path.parent.mkdir(parents=True, exist_ok=True)
        self._usage_path.write_text(json.dumps(usage))

    def remaining_budget(self) -> int:
        return max(0, self.daily_budget - self._read_usage()["count"])

    # ------------------------------------------------------------- throttling
    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    # -------------------------------------------------------------------- mix
    def mix(self, colors: List[Dict[str, object]]) -> MixResult:
        """Mix ``colors`` ([{"hex","count"}, ...]) with pro/2025 and return the hex.

        Raises :class:`MissingApiKey`, :class:`BudgetExhausted`, or
        :class:`TryColorsError` on persistent transport/API failure.
        """
        if not self.api_key:
            raise MissingApiKey(
                "TRYCOLORS_API_KEY is not set (server-side env var). "
                "The key must never be hard-coded or sent from the frontend."
            )
        if len(colors) < 2:
            raise TryColorsError("mix() needs at least 2 colors")

        with self._lock:
            usage = self._read_usage()
            if usage["count"] >= self.daily_budget:
                raise BudgetExhausted(
                    f"TryColors daily budget reached ({self.daily_budget}/day). "
                    "Fall back to the local engine and resume tomorrow."
                )

            payload = {
                "colors": [{"hex": c["hex"], "count": int(c.get("count", 1))} for c in colors],
                "mixerMode": self.mixer_mode,
                "engine": self.engine,
            }
            headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

            last_exc: Optional[Exception] = None
            for attempt in range(self.max_retries + 1):
                self._throttle()
                try:
                    resp = httpx.post(MIX_ENDPOINT, json=payload, headers=headers, timeout=self.timeout)
                except httpx.HTTPError as exc:  # network / timeout
                    last_exc = exc
                    self._sleep_backoff(attempt)
                    continue

                if resp.status_code == 200:
                    mixed = self._extract_hex(resp)
                    usage["count"] += 1
                    self._write_usage(usage)
                    return MixResult(mixed, "trycolors_api", self.mixer_mode, self.engine)

                if resp.status_code == 429 or resp.status_code >= 500:
                    # Transient: respect Retry-After if present, else backoff.
                    last_exc = TryColorsError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    self._sleep_backoff(attempt, resp.headers.get("Retry-After"))
                    continue

                # 4xx other than 429 -> not retryable.
                raise TryColorsError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            raise TryColorsError(f"TryColors mix failed after retries: {last_exc}")

    @staticmethod
    def _extract_hex(resp: httpx.Response) -> str:
        try:
            data = resp.json()
        except Exception as exc:
            raise TryColorsError(f"Non-JSON response: {resp.text[:200]}") from exc
        mixed = data.get("mixedColor") or data.get("mixed_color") or data.get("hex")
        if not mixed:
            raise TryColorsError(f"Response missing mixedColor: {data}")
        return str(mixed).upper()

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str] = None) -> None:
        if retry_after:
            try:
                time.sleep(min(30.0, float(retry_after)))
                return
            except (TypeError, ValueError):
                pass
        # 0.5, 1, 2, 4 ... capped.
        time.sleep(min(8.0, 0.5 * (2 ** attempt)))


__all__ = [
    "TryColorsClient",
    "MixResult",
    "TryColorsError",
    "BudgetExhausted",
    "MissingApiKey",
    "API_BASE",
    "MIX_ENDPOINT",
]
