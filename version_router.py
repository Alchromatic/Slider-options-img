"""
version_router.py
=================

FastAPI router exposing the versioned color-mixing / unmixing dispatcher so the
single front-end UI can switch models from a dropdown.

Mount on the existing app with:
    from version_router import router as version_router
    app.include_router(version_router)

Endpoints
---------
GET  /versions               -> list of forward + unmix versions and availability
POST /version/forward-mix    -> predict a mixed color for a recipe with a chosen version
POST /version/unmix          -> propose recipes for a target color with a chosen version
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

import version_dispatch as vd

router = APIRouter(tags=["12. Versioned models"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ForwardMixVersionRequest(BaseModel):
    """Recipe -> predicted mixed color, for a chosen forward version."""
    pigments: List[str] = Field(..., description="Pigment abbreviations or full names, e.g. ['CY','QM','UB']")
    parts: List[float] = Field(..., description="Parts per pigment, e.g. [4,1,1]")
    version: str = Field("m4", description="baseline | dualgate | m4 | m5 | m7_1")


class UnmixVersionRequest(BaseModel):
    """Target color -> proposed recipes, for a chosen unmix version."""
    target_color: str = Field(..., description="Target hex, e.g. '#706A35'")
    version: str = Field("m7_1", description="m7_1 | m7 | km_baseline | m6 | m6_1")
    max_colors: int = Field(4, ge=1, le=4)
    total_parts: int = Field(6, ge=2, le=12)
    top_n: int = Field(5, ge=1, le=20)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/versions")
def get_versions():
    """List every forward and unmix version, with availability and notes."""
    return vd.list_versions()


@router.post("/version/forward-mix")
def version_forward_mix(req: ForwardMixVersionRequest):
    """
    Predict the mixed color for a recipe using the selected forward version.

    Forward versions:
    - **baseline** - Kubelka-Munk subtractive mix of the measured palette
    - **dualgate** - linear-light mix of the measured palette
    - **m4** - routes baseline vs dual-gate, then mixes with the chosen mixer (current)
    - **m5** - experimental router (adds a P3 branch)
    - **m7_1** - measured-pairwise forward prediction
    """
    if not req.pigments or len(req.pigments) != len(req.parts):
        return {"version": req.version, "mode": "forward", "available": True,
                "error": "pigments and parts must be non-empty and the same length."}
    try:
        return vd.forward_mix(req.pigments, req.parts, version=req.version)
    except KeyError as e:
        return {"version": req.version, "mode": "forward", "available": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        return {"version": req.version, "mode": "forward", "available": True, "error": str(e)}


@router.post("/version/unmix")
def version_unmix(req: UnmixVersionRequest):
    """
    Propose recipes for a target color using the selected unmix version.

    Unmix versions:
    - **m7_1** - measured-pairwise unmix (current), includes observed n-ary anchors
    - **m7** - measured-pairwise unmix research candidate (no H24 n-ary rescue)
    - **km_baseline** - the existing /unmix Kubelka-Munk search (uses your palette)
    - **m6 / m6.1** - historical branches; reported as unavailable in this package

    Measured models use the fixed measured 8-pigment palette (returned in
    `palette`), so the loaded UI palette does not apply to them.
    """
    try:
        return vd.unmix(req.target_color, version=req.version,
                        max_colors=req.max_colors, total_parts=req.total_parts,
                        top_n=req.top_n)
    except KeyError as e:
        return {"version": req.version, "mode": "unmix", "available": False,
                "target_color": req.target_color, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        return {"version": req.version, "mode": "unmix", "available": True,
                "target_color": req.target_color, "error": str(e)}
