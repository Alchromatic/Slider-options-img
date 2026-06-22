#!/usr/bin/env python3
"""
profile_admin_routes.py
=======================

Admin-only endpoints for generating and inspecting **measured palette profiles**
(see Color-engine.docx).  Profile generation is the offline/backend path the doc
review mandated (rule R5) -- it is NOT a user-facing action, because it can call
the TryColors API.  These routes are therefore gated behind an ``ADMIN_TOKEN``
bearer (sent as the ``X-Admin-Token`` header) and the TryColors API key is only
ever read server-side by ``trycolors_client`` (rule R8).

Endpoints:

    POST /unmix/custom/profile/generate   -> build/update a profile (incremental)
    GET  /unmix/custom/profile/{palette_id} -> inspect a saved profile

The user-facing ``POST /unmix/custom`` (in custom_palette_unmix.py) only *loads*
a cached profile; it never generates one.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

import measured_profile as mp
from custom_palette_unmix import CustomPaletteColor
from profile_generator import generate

router = APIRouter(tags=["13. Custom palette unmix"])


def _require_admin(token: Optional[str]) -> None:
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503,
                            detail="Profile admin endpoints are disabled (ADMIN_TOKEN not configured).")
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token.")


def _parse_ratios(ratios: Optional[List[str]]) -> Optional[List[Tuple[int, int]]]:
    if not ratios:
        return None
    out: List[Tuple[int, int]] = []
    for r in ratios:
        a, b = str(r).replace("/", ":").split(":")
        out.append((int(a), int(b)))
    return out


def _summary(profile: dict, *, remaining_budget: Optional[int] = None) -> dict:
    comp = profile.get("completeness", {})
    out = {
        "palette_id": profile.get("palette_id"),
        "palette_name": profile.get("palette_name"),
        "profile_version": profile.get("profile_version"),
        "engine": profile.get("engine"),
        "mixer_mode": profile.get("mixer_mode"),
        "source": profile.get("source"),
        "updated_at": profile.get("updated_at"),
        "colors": profile.get("colors"),
        "completeness": comp,
    }
    if remaining_budget is not None:
        out["api_budget_remaining_today"] = remaining_budget
    return out


class ProfileGenerateRequest(BaseModel):
    palette_id: str = Field(..., description="Stable palette id; reuse it to add colors incrementally.")
    name: str = Field("", description="Human-readable palette name.")
    colors: List[CustomPaletteColor] = Field(..., description="Palette colors (name + hex).")
    ratios: Optional[List[str]] = Field(None, description='Pairwise ratios, e.g. ["1:1","1:3","3:1"].')
    max: Optional[int] = Field(None, description="Cap comparisons generated this run (budget control).")
    local_only: bool = Field(False, description="Never call TryColors; use the local KM engine only.")


@router.post("/unmix/custom/profile/generate", tags=["13. Custom palette unmix"])
def generate_profile(req: ProfileGenerateRequest, x_admin_token: Optional[str] = Header(None)):
    """
    Build or incrementally update a measured palette profile (admin only).

    Hybrid strategy: measures each missing comparison via TryColors pro/2025, and
    falls back to the local KM engine when the daily API budget is exhausted.
    Adding a color to an existing ``palette_id`` only generates the new pair
    comparisons (incremental). Returns a summary (not the full comparison list).
    """
    _require_admin(x_admin_token)
    try:
        profile = generate(
            palette_id=req.palette_id,
            colors=[c.model_dump() for c in req.colors],
            palette_name=req.name,
            ratios=_parse_ratios(req.ratios),
            use_api=not req.local_only,
            local_only=req.local_only,
            max_this_run=req.max,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile generation failed: {e}")

    remaining = None
    if not req.local_only:
        try:
            from trycolors_client import TryColorsClient
            remaining = TryColorsClient().remaining_budget()
        except Exception:
            remaining = None
    return _summary(profile, remaining_budget=remaining)


@router.get("/unmix/custom/profile/{palette_id}", tags=["13. Custom palette unmix"])
def get_profile(palette_id: str, full: bool = False, x_admin_token: Optional[str] = Header(None)):
    """Inspect a saved measured palette profile (admin only).

    By default returns a summary; pass ``?full=true`` for the full profile
    including every measured comparison.
    """
    _require_admin(x_admin_token)
    profile = mp.load_profile(palette_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile for palette_id '{palette_id}'.")
    return profile if full else _summary(profile)


__all__ = ["router"]
