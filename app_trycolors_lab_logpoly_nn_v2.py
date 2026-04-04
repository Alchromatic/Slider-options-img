# app_trycolors_lab_logpoly_nn_recipe.py
from __future__ import annotations

import io
import json
import math
import re
import hashlib
import os
import datetime
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import streamlit as st

# Optional pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

# Optional torch (for NN residual)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(page_title="Forward Mix Engine", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# Custom CSS: match the Art-AI Renderer Adjust tab look
# ============================================================
st.markdown("""
<style>
/* ---- Hide ALL Streamlit chrome ---- */
#MainMenu, header, footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
.stDeployButton,
div[data-testid="stSidebarNav"],
section[data-testid="stSidebar"],
div[data-testid="stSidebarCollapsedControl"],
.reportview-container .main footer,
div[data-testid="stSidebarCollapsedControl"],
button[kind="header"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* ---- Remove Streamlit padding, warm white background ---- */
.stApp > header { display: none !important; }
.main .block-container {
    padding: 12px 16px 12px 16px !important;
    max-width: 100% !important;
}
.stApp, .main,
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
div[data-testid="stForm"] {
    background: #fffaf5 !important;
}

/* ---- Typography ---- */
html, body, .stApp, .stMarkdown, .stText,
[data-testid="stMarkdownContainer"],
label, .stSelectbox, .stSlider, .stNumberInput {
    font-family: Arial, sans-serif !important;
    color: #333 !important;
}

h1 { font-size: 15px !important; font-weight: bold !important; margin: 6px 0 !important; color: #e65100 !important; }
h2 { font-size: 14px !important; font-weight: bold !important; margin: 6px 0 !important; color: #e65100 !important; }
h3 { font-size: 13px !important; font-weight: 600 !important; margin: 4px 0 !important; color: #bf360c !important; }
.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 11px !important; color: #888 !important;
}

/* ---- Labels ---- */
label, [data-testid="stWidgetLabel"] {
    font-size: 11px !important;
    color: #666 !important;
    font-weight: 600 !important;
}

/* ---- Compact spacing: crush all vertical gaps ---- */
.element-container { margin-bottom: 2px !important; }
.stMarkdown { margin-bottom: 0 !important; }
.stMarkdown p { margin-bottom: 2px !important; font-size: 12px !important; color: #666 !important; }
[data-testid="stVerticalBlock"] > div { gap: 0.15rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
[data-testid="column"] { padding: 0 4px !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }

/* ---- Sections / Expanders ---- */
[data-testid="stExpander"] {
    border: 1px solid #ffe0b2 !important;
    border-radius: 6px !important;
    background: #fff8f0 !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #e65100 !important;
}
hr {
    border: none !important;
    border-top: 1px solid #ffe0b2 !important;
    margin: 8px 0 !important;
}

/* ---- Input fields: light warm style ---- */
input[type="text"], input[type="number"],
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
textarea {
    background: #fff !important;
    border: 1px solid #ccc !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    font-family: Arial, sans-serif !important;
    padding: 6px 8px !important;
    color: #333 !important;
}
input:focus, textarea:focus {
    border-color: #ff9800 !important;
    box-shadow: 0 0 0 1px rgba(255,152,0,0.25) !important;
}

/* Number input +/- buttons */
[data-testid="stNumberInput"] button {
    background: #f5f5f5 !important;
    border: 1px solid #ccc !important;
    color: #333 !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #ffe0b2 !important;
}

/* Select boxes */
[data-testid="stSelectbox"] > div > div {
    background: #fff !important;
    border: 1px solid #ccc !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    color: #333 !important;
}

/* ---- Buttons ---- */
.stButton > button {
    font-family: Arial, sans-serif !important;
    font-size: 13px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    border-radius: 4px !important;
    border: none !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
/* Primary buttons - orange gradient like Drawing */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
.stButton > button {
    background: linear-gradient(135deg, #ff9800, #f57c00) !important;
    color: white !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ffa726, #ff9800) !important;
    box-shadow: 0 2px 8px rgba(255,152,0,0.3) !important;
}
/* Secondary buttons */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
    background: #607D8B !important;
    color: white !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: #546e7a !important;
}

/* ---- Sliders: orange accent ---- */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    font-size: 10px !important;
}
.stSlider > div > div > div > div {
    background: #ff9800 !important;
}
[data-testid="stSlider"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* ---- Color picker: compact ---- */
[data-testid="stColorPicker"] {
    max-width: 80px !important;
}
[data-testid="stColorPicker"] > div {
    padding: 0 !important;
}
[data-testid="stColorPicker"] label {
    margin-bottom: 2px !important;
}

/* ---- Alerts: warm tones ---- */
[data-testid="stAlert"] {
    font-size: 11px !important;
    padding: 6px 10px !important;
    border-radius: 4px !important;
}
.stSuccess > div { background: #e8f5e9 !important; border-color: #a5d6a7 !important; }
.stInfo > div { background: #fff3e0 !important; border-color: #ffe0b2 !important; }
.stWarning > div { background: #fff8e1 !important; border-color: #ffecb3 !important; }

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] {
    font-size: 11px !important;
}

/* ---- Download buttons ---- */
.stDownloadButton > button {
    font-size: 12px !important;
    padding: 8px 16px !important;
    background: #607D8B !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
}
.stDownloadButton > button:hover {
    background: #546e7a !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 2px !important; }
.stTabs [data-baseweb="tab"] {
    font-size: 12px !important;
    font-family: Arial, sans-serif !important;
    padding: 6px 14px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: #ff9800 !important;
    color: #e65100 !important;
}

/* ---- Color swatches from Streamlit: make smaller ---- */
[data-testid="stImage"] {
    max-width: 50px !important;
}

/* ---- Checkboxes: orange accent ---- */
[data-testid="stCheckbox"] label span {
    font-size: 12px !important;
    color: #333 !important;
}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {
    border: 1px dashed #ffcc80 !important;
    border-radius: 4px !important;
    background: #fff8f0 !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] label {
    font-size: 11px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1) Color utilities
# ============================================================

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

def srgb_to_linear(c: float) -> float:
    c = float(c)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def linear_to_srgb(c: float) -> float:
    c = clamp01(float(c))
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055

def clean_hex(h: str) -> Optional[str]:
    """Return a normalized #RRGGBB uppercase string, or None if invalid."""
    if h is None:
        return None
    s = str(h).strip().upper()
    if not s:
        return None
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4:  # #RGB
        s = "#" + "".join(ch * 2 for ch in s[1:])
    if len(s) != 7:
        return None
    if any(ch not in "0123456789ABCDEF" for ch in s[1:]):
        return None
    return s



# ---- Optional URL query params (for UI integration / verification) ----
# Supported:
#   ?target=#RRGGBB  (or target=RRGGBB)
#   ?palette=#RRGGBB,#RRGGBB,...  (comma/semicolon/whitespace separated)
# This is useful if you have an external UI (e.g., a JS/React frontend) that can
# "select a region" and then open this Streamlit app with the target/palette preloaded.

def calibrator_allows_hexes(hexes: List[str]) -> bool:
    """Return True if the loaded calibrator is applicable to the given hex list.

    The bundled calibrator is palette-specific (trained on a fixed set of base paints).
    When the current palette or recipe uses colors outside that set, applying the calibrator
    can severely degrade results. This helper uses metadata (if available) to gate usage.
    """
    meta = st.session_state.get("calibrator_meta")
    if not isinstance(meta, dict):
        return True
    trained = meta.get("trained_on")
    if not isinstance(trained, dict):
        return True
    pal = trained.get("palette_unique_base_hexes")
    if not isinstance(pal, list) or not pal:
        return True

    pal_set = set()
    for h in pal:
        hh = clean_hex(str(h))
        if hh:
            pal_set.add(hh)
    if not pal_set:
        return True

    hx_set = set()
    for h in hexes:
        hh = clean_hex(str(h))
        if hh:
            hx_set.add(hh)
    if not hx_set:
        return True

    return hx_set.issubset(pal_set)

def _apply_query_params_once() -> None:
    if st.session_state.get("_qp_applied", False):
        return
    st.session_state["_qp_applied"] = True

    def _get(qp: Any, key: str) -> Optional[str]:
        try:
            v = qp.get(key, None)
        except Exception:
            v = None
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if v is None:
            return None
        return str(v)

    target_raw: Optional[str] = None
    palette_raw: Optional[str] = None
    try:
        qp = st.query_params  # Streamlit ≥1.32
        target_raw = _get(qp, "target")
        palette_raw = _get(qp, "palette")
    except Exception:
        try:
            qp = st.experimental_get_query_params()  # legacy
            target_raw = _get(qp, "target")
            palette_raw = _get(qp, "palette")
        except Exception:
            target_raw = None
            palette_raw = None

    if target_raw:
        h = clean_hex(target_raw)
        if h:
            st.session_state["target_hex"] = h
            st.session_state["fwd_target_hex"] = h  # keep forward diagnostics in sync

    if palette_raw:
        # Accept comma / semicolon separated; each token may be "Name|#RRGGBB" or just "#RRGGBB"
        tokens = re.split(r"[,;]+", str(palette_raw).strip())
        rows: List[Dict[str, str]] = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            if "|" in t:
                parts = t.split("|", 1)
                name_part = parts[0].strip()
                hex_part = clean_hex(parts[1].strip())
                if hex_part:
                    rows.append({"name": name_part, "hex": hex_part})
            else:
                hh = clean_hex(t)
                if hh:
                    rows.append({"name": "", "hex": hh})
        if rows:
            st.session_state["palette_rows"] = rows

_apply_query_params_once()

# ---- iframe postMessage bridge ----
# When this Streamlit app is embedded in an iframe (e.g., the Art-AI Renderer Netlify app),
# the parent window can send messages to set the target hex and/or palette.
# Usage from parent JS:
#   document.querySelector('iframe').contentWindow.postMessage({type:'setTarget', hex:'#D9AE57'}, '*')
#   document.querySelector('iframe').contentWindow.postMessage({type:'setPalette', colors:[{name:'Titanium White', hex:'#F7F5F1'}, ...]}, '*')
#   document.querySelector('iframe').contentWindow.postMessage({type:'setTargetAndPalette', hex:'#D9AE57', colors:[...]}, '*')

try:
    import streamlit.components.v1 as components
    _IFRAME_BRIDGE_JS = """
    <script>
    window.addEventListener('message', function(event) {
        var data = event.data;
        if (!data || !data.type) return;
        var url = new URL(window.location.href);
        if (data.type === 'setTarget' || data.type === 'setTargetAndPalette') {
            if (data.hex) url.searchParams.set('target', data.hex);
        }
        if (data.type === 'setPalette' || data.type === 'setTargetAndPalette') {
            if (data.colors && Array.isArray(data.colors)) {
                var enc = data.colors.map(function(c) {
                    return (c.name ? c.name + '|' : '') + (c.hex || '');
                }).join(',');
                url.searchParams.set('palette', enc);
            }
        }
        if (window.parent && window.parent !== window) {
            window.parent.postMessage({type:'trycolorsLabAck', received:data.type}, '*');
        }
        window.history.replaceState({}, '', url.toString());
        window.location.reload();
    });
    if (window.parent && window.parent !== window) {
        window.parent.postMessage({type:'trycolorsLabReady'}, '*');
    }
    </script>
    """
    components.html(_IFRAME_BRIDGE_JS, height=0)
except Exception:
    pass  # components unavailable — no bridge, query params still work

def hex_to_rgb255(h: str) -> Tuple[int, int, int]:
    hh = clean_hex(h)
    if hh is None:
        raise ValueError(f"Invalid hex: {h}")
    hh = hh.lstrip("#")
    return int(hh[0:2], 16), int(hh[2:4], 16), int(hh[4:6], 16)

def rgb255_to_hex(r: int, g: int, b: int) -> str:
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def hex_to_linear_rgb(h: str) -> Tuple[float, float, float]:
    r8, g8, b8 = hex_to_rgb255(h)
    return (
        srgb_to_linear(r8 / 255.0),
        srgb_to_linear(g8 / 255.0),
        srgb_to_linear(b8 / 255.0),
    )

def linear_rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    return rgb255_to_hex(
        int(round(linear_to_srgb(r) * 255)),
        int(round(linear_to_srgb(g) * 255)),
        int(round(linear_to_srgb(b) * 255)),
    )

def normalize_weights(w: Sequence[float], tol: float = 1e-12) -> List[float]:
    s = float(sum(w))
    if s <= tol:
        raise ValueError("Weights sum to 0")
    return [float(x) / s for x in w]




def merge_duplicate_bases(bases_hex: List[str], weights: List[float]) -> Tuple[List[str], List[float]]:
    """Merge duplicate base_hex entries by summing weights (preserving first-seen order)."""
    if len(bases_hex) != len(weights):
        raise ValueError("bases_hex / weights length mismatch")
    acc: Dict[str, float] = {}
    order: List[str] = []
    for hx, wi in zip(bases_hex, weights):
        hx = str(hx).strip().upper()
        if not hx.startswith("#"):
            hx = "#" + hx
        if hx not in acc:
            acc[hx] = 0.0
            order.append(hx)
        acc[hx] += float(wi)
    merged_bases = order
    merged_weights = [float(acc[hx]) for hx in order]
    return merged_bases, merged_weights
def softplus(z: float) -> float:
    """Numerically-stable softplus."""
    z = float(z)
    # softplus(z)=log(1+exp(z)) with stability tricks
    if z > 40.0:
        return z
    if z < -40.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


def apply_strength_weights(
    bases_lin: List[Tuple[float, float, float]],
    w: List[float],
    a: Tuple[float, float, float],
    b: float,
    gamma: float = 1.0,
) -> List[float]:
    """
    Recipe-aware weight re-scaling:
        s_i = softplus(a · base_linRGB_i + b) + 1e-6
        w_i' = normalize( w_i * s_i )

    This is a low-parameter way to make "some paints act stronger than others",
    which Trycolors-like engines often do implicitly.
    """
    if len(bases_lin) != len(w):
        raise ValueError("bases_lin and w length mismatch")
    a0, a1, a2 = (float(a[0]), float(a[1]), float(a[2]))
    b = float(b)

    s_list: List[float] = []
    for (r, g, bb) in bases_lin:
        z = a0 * float(r) + a1 * float(g) + a2 * float(bb) + b
        s_list.append(float(softplus(z) + 1e-6))

    gamma = float(gamma)
    if gamma <= 0.0:
        return normalize_weights(w)
    w2 = [float(wi) * (si ** gamma) for wi, si in zip(w, s_list)]
    return normalize_weights(w2)

def rmse_lin(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) / 3.0)

def rmse_hex(pred_hex: str, true_hex: str) -> float:
    """RMSE in *linear RGB* (0..1)."""
    return rmse_lin(hex_to_linear_rgb(pred_hex), hex_to_linear_rgb(true_hex))

def make_swatch(hex_color: str, size: int = 180) -> Image.Image:
    r, g, b = hex_to_rgb255(hex_color)
    return Image.new("RGB", (size, size), (r, g, b))

def show_swatch(label: str, hex_color: str, size: int = 180) -> None:
    st.markdown(f"**{label}: {hex_color}**")
    st.image(make_swatch(hex_color, size=size), use_column_width=False)


# ============================================================
# 2) Mixing engines (Linear / KM / YN-KM)
# ============================================================

def mix_linear_rgb(bases_lin: List[Tuple[float, float, float]], w: List[float]) -> Tuple[float, float, float]:
    w = normalize_weights(w)
    return tuple(
        clamp01(sum(b[i] * wi for b, wi in zip(bases_lin, w)))
        for i in range(3)
    )  # type: ignore[return-value]

def _ks_from_R(R: float, eps: float) -> float:
    R = max(float(R), float(eps))
    return (1.0 - R) ** 2 / (2.0 * R)

def _R_from_ks(KS: float) -> float:
    KS = float(KS)
    return max(0.0, (1.0 + KS) - math.sqrt(KS * KS + 2.0 * KS))

def mix_km(bases_lin: List[Tuple[float, float, float]], w: List[float], eps: float) -> Tuple[float, float, float]:
    w = normalize_weights(w)
    out: List[float] = []
    for ch in range(3):
        KS = sum(_ks_from_R(b[ch], eps) * wi for b, wi in zip(bases_lin, w))
        out.append(clamp01(_R_from_ks(KS)))
    return (out[0], out[1], out[2])

def mix_ynkm(bases_lin: List[Tuple[float, float, float]], w: List[float], n: float, eps: float) -> Tuple[float, float, float]:
    n = float(n)
    if n <= 0:
        return mix_km(bases_lin, w, eps)

    def fwd(R: float) -> float:
        return clamp01(R) ** (1.0 / n)

    def inv(Rp: float) -> float:
        return clamp01(Rp) ** n

    bases_yn = [(fwd(r), fwd(g), fwd(b)) for (r, g, b) in bases_lin]
    mix_lin = mix_km(bases_yn, w, eps)
    return (inv(mix_lin[0]), inv(mix_lin[1]), inv(mix_lin[2]))



def mix_hybrid_km_linear(bases_lin: List[Tuple[float, float, float]], w: List[float], eps: float, t: float) -> Tuple[float, float, float]:
    """Blend between KM (t=0) and Linear (t=1) in *linear RGB* space."""
    t = clamp01(float(t))
    km = mix_km(bases_lin, w, eps)
    lin = mix_linear_rgb(bases_lin, w)
    return (
        clamp01((1.0 - t) * km[0] + t * lin[0]),
        clamp01((1.0 - t) * km[1] + t * lin[1]),
        clamp01((1.0 - t) * km[2] + t * lin[2]),
    )

def mix_hybrid_ynkm_linear(
    bases_lin: List[Tuple[float, float, float]],
    w: List[float],
    n: float,
    eps: float,
    t: float,
) -> Tuple[float, float, float]:
    """Blend between YN-KM (t=0) and Linear (t=1) in *linear RGB* space."""
    t = clamp01(float(t))
    y = mix_ynkm(bases_lin, w, n=n, eps=eps)
    lin = mix_linear_rgb(bases_lin, w)
    return (
        clamp01((1.0 - t) * y[0] + t * lin[0]),
        clamp01((1.0 - t) * y[1] + t * lin[1]),
        clamp01((1.0 - t) * y[2] + t * lin[2]),
    )
# ============================================================
# 2b) GeoLogMix engines (experimental, license-clean)
# ============================================================

# Standard sRGB (D65) linear RGB <-> XYZ matrices (Y normalized to 1 for white).
_RGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=float,
)
_XYZ_TO_RGB_D65 = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=float,
)

# Messick (Paper 1) regression matrix in XYZ for GSLC; provided as a *research preset*.
# NOTE: In GeoLogMix-3 we use this as an M preset (mixing-space map), not as a post-hoc correction.
_MESSICK_THETA_XYZ = np.array(
    [
        [0.882, 0.095, 0.044],
        [-0.058, 1.021, 0.098],
        [0.211, -0.307, 1.037],
    ],
    dtype=float,
)

def linear_rgb_to_xyz(rgb_lin: Tuple[float, float, float]) -> np.ndarray:
    v = np.array([float(rgb_lin[0]), float(rgb_lin[1]), float(rgb_lin[2])], dtype=float)
    return _RGB_TO_XYZ_D65 @ v

def xyz_to_linear_rgb_arr(xyz: np.ndarray) -> np.ndarray:
    return _XYZ_TO_RGB_D65 @ np.asarray(xyz, dtype=float)


# ============================================================
# CIELAB + CIEDE2000 (ΔE00) + Trycolors-like Match%
# ============================================================
# Note: Trycolors' UI "Match %" is consistent with: Match ≈ max(0, 100 - ΔE00).
# We expose both ΔE00 and Match% for parity with the Trycolors UX.

_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=float)

def _lab_f(t: np.ndarray) -> np.ndarray:
    # CIE Lab f(t) with delta = 6/29.
    delta = 6.0 / 29.0
    t = np.asarray(t, dtype=float)
    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4.0 / 29.0))

def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = _D65_WHITE) -> np.ndarray:
    """Convert XYZ (D65, relative 0..1) to CIELAB."""
    xyz = np.asarray(xyz, dtype=float)
    white = np.asarray(white, dtype=float)
    x = xyz[..., 0] / white[0]
    y = xyz[..., 1] / white[1]
    z = xyz[..., 2] / white[2]
    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.array([L, a, b], dtype=float)

def linear_rgb_to_lab(rgb_lin: Tuple[float, float, float]) -> np.ndarray:
    xyz = linear_rgb_to_xyz(rgb_lin)
    return xyz_to_lab(xyz)

def hex_to_lab(h: str) -> np.ndarray:
    return linear_rgb_to_lab(hex_to_linear_rgb(h))

def delta_e00(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIEDE2000 color difference (ΔE00). Reference: Sharma et al. 2005."""
    L1, a1, b1 = [float(x) for x in np.asarray(lab1, dtype=float).tolist()]
    L2, a2, b2 = [float(x) for x in np.asarray(lab2, dtype=float).tolist()]

    kL = 1.0
    kC = 1.0
    kH = 1.0

    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + (25.0 ** 7))) ) if C_bar > 0 else 0.0

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)

    def _hp(ap: float, b: float) -> float:
        if ap == 0.0 and b == 0.0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360.0 if h < 0.0 else h

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    # Hue difference
    if C1p * C2p == 0.0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180.0:
            dhp -= 360.0
        elif dhp < -180.0:
            dhp += 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    # Mean hue
    if C1p * C2p == 0.0:
        hp_bar = h1p + h2p
    else:
        hsum = h1p + h2p
        hdiff = abs(h1p - h2p)
        if hdiff > 180.0:
            hp_bar = (hsum + 360.0) / 2.0 if hsum < 360.0 else (hsum - 360.0) / 2.0
        else:
            hp_bar = hsum / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hp_bar - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hp_bar))
        + 0.32 * math.cos(math.radians(3.0 * hp_bar + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hp_bar - 63.0))
    )

    d_ro = 30.0 * math.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Rc = 2.0 * math.sqrt((Cp_bar ** 7) / ((Cp_bar ** 7) + (25.0 ** 7))) if Cp_bar > 0 else 0.0

    Sl = 1.0 + (0.015 * ((Lp_bar - 50.0) ** 2)) / math.sqrt(20.0 + ((Lp_bar - 50.0) ** 2))
    Sc = 1.0 + 0.045 * Cp_bar
    Sh = 1.0 + 0.015 * Cp_bar * T

    Rt = -math.sin(math.radians(2.0 * d_ro)) * Rc

    dE = math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return float(dE)

def match_percent_from_de00(de00: float) -> float:
    # Clamp to [0,100]. This is consistent with Trycolors' displayed "Match %".
    return float(max(0.0, min(100.0, 100.0 - float(de00))))


def _clip_eps(x: np.ndarray, eps: float) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), float(eps))

def mix_geologmix3(
    bases_lin: List[Tuple[float, float, float]],
    w: List[float],
    eps: float,
    M: Optional[np.ndarray] = None,
    b_xyz: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    GeoLogMix-3 forward mixing in *affine-shifted* XYZ:

        u_i = clip_eps(M^{-1} (XYZ_i + b))
        u_mix = exp( Σ w_i log u_i )
        XYZ_mix = M u_mix - b

    The bias b (XYZ) avoids log(0) issues and improves black/shade behavior.
    """
    w = normalize_weights(w)

    if M is None:
        M = np.eye(3, dtype=float)
    M = np.asarray(M, dtype=float)
    invM = np.linalg.inv(M)

    bias = np.zeros(3, dtype=float) if b_xyz is None else np.asarray(b_xyz, dtype=float).reshape(3)

    xyz_bases = np.stack([linear_rgb_to_xyz(base) for base in bases_lin], axis=0)  # Nx3
    xyz_bases = xyz_bases + bias[None, :]
    U = (invM @ xyz_bases.T).T  # Nx3
    U = _clip_eps(U, eps)

    logU = np.log(U)
    logU_mix = (np.asarray(w, dtype=float)[:, None] * logU).sum(axis=0)
    U_mix = np.exp(logU_mix)

    xyz_mix = (M @ U_mix) - bias
    lin = xyz_to_linear_rgb_arr(xyz_mix)
    return (clamp01(float(lin[0])), clamp01(float(lin[1])), clamp01(float(lin[2])))
# ----- GeoLogMix-K (latent 8) -----
# A is a fixed positive decoder from an 8D latent to XYZ, built from anchor linear-sRGB colors.
_GEOLOGMIXK_ANCHORS_LIN = np.array(
    [
        [1.0, 0.0, 0.0],   # red
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.0, 1.0],   # magenta
        [0.0, 1.0, 1.0],   # cyan
        [1.0, 1.0, 1.0],   # white
        [1.0, 0.5, 0.0],   # orange-ish
    ],
    dtype=float,
)

_GEOLOGMIXK_A = np.stack([(_RGB_TO_XYZ_D65 @ _GEOLOGMIXK_ANCHORS_LIN[i]) for i in range(8)], axis=1)  # 3x8
_GEOLOGMIXK_A = np.maximum(_GEOLOGMIXK_A, 1e-9)

def _nnls_projected_gd(A: np.ndarray, c: np.ndarray, eps: float, ridge: float, iters: int) -> np.ndarray:
    """
    Solve min_u ||A u - c||^2 + ridge||u||^2  s.t. u >= eps via projected GD.
    """
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    K = A.shape[1]

    # Init by least squares, then clip.
    try:
        u0, *_ = np.linalg.lstsq(A, c, rcond=None)
    except Exception:
        u0 = np.ones(K, dtype=float) / max(K, 1)
    u = np.maximum(np.asarray(u0, dtype=float), float(eps))

    H = A.T @ A + float(ridge) * np.eye(K)
    L = 2.0 * float(np.linalg.eigvalsh(H).max())
    lr = 1.0 / max(L, 1e-9)

    for _ in range(int(iters)):
        grad = 2.0 * (A.T @ (A @ u - c) + float(ridge) * u)
        u = np.maximum(u - lr * grad, float(eps))
    return u

def mix_geologmixk_from_logU(
    logU_bases: np.ndarray,  # NxK
    w: List[float],
    A: np.ndarray,
) -> np.ndarray:
    """Given cached logU for bases, produce XYZ_mix = A exp( Σ w_i logU_i )."""
    w = normalize_weights(w)
    logU_mix = (np.asarray(w, dtype=float)[:, None] * logU_bases).sum(axis=0)
    U_mix = np.exp(logU_mix)
    return A @ U_mix

def precompute_geologmixk_logU_bases(
    bases_lin: List[Tuple[float, float, float]],
    eps: float,
    ridge: float,
    iters: int,
    A: np.ndarray = _GEOLOGMIXK_A,
) -> np.ndarray:
    xyz_bases = np.stack([linear_rgb_to_xyz(b) for b in bases_lin], axis=0)  # Nx3
    U = np.stack([_nnls_projected_gd(A, xyz_bases[i], eps=eps, ridge=ridge, iters=iters) for i in range(xyz_bases.shape[0])], axis=0)  # NxK
    U = _clip_eps(U, eps)
    return np.log(U)

def mix_geologmixk(
    bases_lin: List[Tuple[float, float, float]],
    w: List[float],
    eps: float,
    ridge: float,
    iters: int,
    logU_cache: Optional[np.ndarray] = None,
    A: np.ndarray = _GEOLOGMIXK_A,
) -> Tuple[float, float, float]:
    """
    GeoLogMix-K forward mixing:

        Encode: u_i = NNLS(A u ≈ XYZ_i), u_i >= eps
        Mix:    u_mix = exp( Σ w_i log u_i )
        Decode: XYZ_mix = A u_mix

    Returns linear RGB.
    """
    if logU_cache is None:
        logU_cache = precompute_geologmixk_logU_bases(bases_lin, eps=eps, ridge=ridge, iters=iters, A=A)

    xyz_mix = mix_geologmixk_from_logU(logU_cache, w=w, A=A)
    lin = xyz_to_linear_rgb_arr(xyz_mix)
    return (clamp01(float(lin[0])), clamp01(float(lin[1])), clamp01(float(lin[2])))



# ============================================================
# 3) Log-poly calibrator
# ============================================================

@dataclass
class Calibrator:
    """
    Log-polynomial calibrator (per-channel) operating in **linear RGB**.

    We fit a polynomial in log space:
        log(T) ≈ poly( log(P) )
    where P is the engine prediction (linear RGB) and T is the reference (Trycolors) linear RGB.

    IMPORTANT SAFETY GUARD:
    - During inverse search the engine may produce values outside the training range.
      High-degree polynomials can behave badly under extrapolation.
    - We therefore clamp the log-input per channel to the training range [x_min, x_max] recorded at fit time.
    """
    engine: str
    ks_eps: float
    yn_n: float
    degree: int
    coeffs: List[List[float]]
    x_mins: List[float]
    x_maxs: List[float]
    hybrid_t: float = 0.4
    palette_hash: Optional[str] = None   # NEW: store training palette hash if available
    def apply(self, rgb_lin: Tuple[float, float, float]) -> Tuple[float, float, float]:
        eps = 1e-6
        out: List[float] = []
        for ch in range(3):
            x = math.log(max(float(rgb_lin[ch]), eps))
            # Clamp to training range to avoid unstable extrapolation.
            if self.x_mins and self.x_maxs and len(self.x_mins) == 3 and len(self.x_maxs) == 3:
                x = max(float(self.x_mins[ch]), min(float(self.x_maxs[ch]), x))
            y = float(np.polyval(self.coeffs[ch], x))
            out.append(clamp01(math.exp(y)))
        return (out[0], out[1], out[2])

    def to_json(self) -> str:
        d = {
            "kind": "log_poly",
            "engine": str(self.engine),
            "ks_eps": float(self.ks_eps),
            "yn_n": float(self.yn_n),
            "hybrid_t": float(self.hybrid_t),
            "degree": int(self.degree),
            "coeffs": self.coeffs,
            "x_mins": self.x_mins,
            "x_maxs": self.x_maxs,
        }
        if self.palette_hash:
            d["palette_hash"] = self.palette_hash
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(text: str) -> "Calibrator":
        d = json.loads(text)
        if d.get("kind") not in (None, "log_poly"):
            raise ValueError(f"Unsupported calibrator kind: {d.get('kind')}")
        engine = str(d.get("engine", "YN-KM"))
        x_mins = d.get("x_mins", [-20.0, -20.0, -20.0])
        x_maxs = d.get("x_maxs", [0.0, 0.0, 0.0])
        return Calibrator(
            engine=engine,
            ks_eps=float(d.get("ks_eps", 1e-6)),
            yn_n=float(d.get("yn_n", 1.5)),
            hybrid_t=float(d.get("hybrid_t", 0.4)),
            degree=int(d["degree"]),
            coeffs=d["coeffs"],
            x_mins=[float(x) for x in x_mins],
            x_maxs=[float(x) for x in x_maxs],
            palette_hash=d.get("palette_hash"),
        )



# ============================================================
# 3a-2) 3D Log-poly calibrator (cross-channel)
# ============================================================

@dataclass
class Calibrator3D:
    """
    Multivariate log-polynomial calibrator operating in **linear RGB**.

    Instead of fitting 3 independent 1D polynomials, we fit a *cross-channel* model:

        y = Phi(x) @ W

    where:
      x = log(P) with P = engine prediction (linear RGB)
      y = log(T) with T = Trycolors reference (linear RGB)
      Phi(x) includes cross terms (e.g., xR*xG), enabling channel interaction correction.

    This tends to be much more effective when the forward engine is biased in hue space
    (e.g., complements / dark mixtures) because the correction can "move" color across channels.

    NOTE: Like the 1D calibrator, we clamp x per-channel to the training log-range
    to avoid unstable extrapolation during inverse search.
    """
    engine: str
    ks_eps: float
    yn_n: float
    hybrid_t: float
    degree: int
    ridge: float
    W: List[List[float]]   # shape (F,3)
    x_mins: List[float]
    x_maxs: List[float]
    palette_hash: Optional[str] = None   # NEW

    @staticmethod
    def _phi(x: np.ndarray, degree: int) -> np.ndarray:
        """
        Build feature vector Phi(x) for x = [xR,xG,xB] (log-linear RGB).

        degree=1: [1, R, G, B]
        degree=2: + [R^2, G^2, B^2, R*G, R*B, G*B]
        degree=3: + [R^3, G^3, B^3,
                     R^2*G, R^2*B, G^2*R, G^2*B, B^2*R, B^2*G,
                     R*G*B]
        """
        x = np.asarray(x, dtype=float).reshape(3)
        r, g, b = float(x[0]), float(x[1]), float(x[2])

        feats = [1.0, r, g, b]
        if int(degree) >= 2:
            feats += [r*r, g*g, b*b, r*g, r*b, g*b]
        if int(degree) >= 3:
            feats += [
                r*r*r, g*g*g, b*b*b,
                (r*r)*g, (r*r)*b,
                (g*g)*r, (g*g)*b,
                (b*b)*r, (b*b)*g,
                r*g*b,
            ]
        return np.asarray(feats, dtype=float)

    @staticmethod
    def _phi_matrix(X: np.ndarray, degree: int) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.stack([Calibrator3D._phi(X[i], degree) for i in range(X.shape[0])], axis=0)

    def apply(self, rgb_lin: Tuple[float, float, float]) -> Tuple[float, float, float]:
        eps = 1e-6
        x = np.log(np.maximum(np.asarray(rgb_lin, dtype=float), eps))
        # Clamp to training range (per channel).
        if self.x_mins and self.x_maxs and len(self.x_mins) == 3 and len(self.x_maxs) == 3:
            for ch in range(3):
                x[ch] = max(float(self.x_mins[ch]), min(float(self.x_maxs[ch]), float(x[ch])))
        Phi = self._phi(x, int(self.degree))  # (F,)
        W = np.asarray(self.W, dtype=float)   # (F,3)
        y = Phi @ W  # (3,)
        out = np.exp(y)
        return (clamp01(float(out[0])), clamp01(float(out[1])), clamp01(float(out[2])))

    def to_json(self) -> str:
        d = {
            "kind": "log_poly_3d",
            "engine": str(self.engine),
            "ks_eps": float(self.ks_eps),
            "yn_n": float(self.yn_n),
            "hybrid_t": float(self.hybrid_t),
            "degree": int(self.degree),
            "ridge": float(self.ridge),
            "W": self.W,
            "x_mins": self.x_mins,
            "x_maxs": self.x_maxs,
        }
        if self.palette_hash:
            d["palette_hash"] = self.palette_hash
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(text: str) -> "Calibrator3D":
        d = json.loads(text)
        if d.get("kind") != "log_poly_3d":
            raise ValueError("Not a log_poly_3d calibrator JSON.")
        x_mins = d.get("x_mins", [-20.0, -20.0, -20.0])
        x_maxs = d.get("x_maxs", [0.0, 0.0, 0.0])
        return Calibrator3D(
            engine=str(d.get("engine", "")),
            ks_eps=float(d.get("ks_eps", 1e-6)),
            yn_n=float(d.get("yn_n", 1.5)),
            hybrid_t=float(d.get("hybrid_t", 0.425)),
            degree=int(d.get("degree", 2)),
            ridge=float(d.get("ridge", 1e-3)),
            W=d.get("W", []),
            x_mins=[float(x) for x in x_mins],
            x_maxs=[float(x) for x in x_maxs],
            palette_hash=d.get("palette_hash"),
        )


def load_calibrator_any(text: str) -> Any:
    """
    Backward-compatible calibrator loader.
    Supports:
      - kind=log_poly    (1D per-channel, legacy)
      - kind=log_poly_3d (cross-channel)
    """
    try:
        d = json.loads(text)
    except Exception:
        # Legacy: assume 1D Calibrator JSON
        return Calibrator.from_json(text)
    kind = d.get("kind", "log_poly")
    if kind == "log_poly_3d":
        return Calibrator3D.from_json(text)
    return Calibrator.from_json(text)


# ============================================================
# 3b) Residual NN (optional)
# ============================================================
# ============================================================

if TORCH_AVAILABLE:
    class ResidualNN(nn.Module):
        """
        Learns a small RGB residual on top of the calibrated output.
        Input:  calibrated RGB (3) in linear space
        Output: delta RGB (3) in linear space
        """
        def __init__(self, hidden: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ============================================================
# 4) CSV parsing
# ============================================================

def _split_semicolon_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(";") if str(x).strip()]

def parse_trycolors_csv(text: str) -> List[Dict[str, Any]]:
    """
    Expects columns:
      - group_id
      - base_hexes  (#AABBCC;#DDEEFF;...)
      - weights     (0.8;0.2;...)
      - api_hex     (#RRGGBB)
    """
    import csv
    rdr = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []
    for r in rdr:
        api = clean_hex((r.get("api_hex") or "").strip())
        if not api:
            continue

        bases = [clean_hex(x) for x in _split_semicolon_list(r.get("base_hexes", ""))]
        bases = [b for b in bases if b is not None]  # type: ignore[assignment]
        if not bases:
            continue

        try:
            weights = [float(x) for x in _split_semicolon_list(r.get("weights", ""))]
        except Exception:
            continue
        if len(weights) != len(bases):
            continue

        rows.append({
            "group_id": (r.get("group_id") or "").strip(),
            "bases": bases,
            "weights": weights,
            "api_hex": api,
        })
    return rows

def extract_palette_from_trycolors_csv(text: str) -> List[str]:
    rows = parse_trycolors_csv(text)
    s: set[str] = set()
    for r in rows:
        for h in r["bases"]:
            hh = clean_hex(h)
            if hh:
                s.add(hh)
    return sorted(s)

def parse_palette_csv(text: str) -> List[str]:
    """
    Accepts either:
      - one hex per line (no header), or
      - a CSV with a column name like: hex / color / candidate_hex
    """
    # Try CSV first
    import csv
    try:
        rdr = csv.DictReader(io.StringIO(text))
        if rdr.fieldnames:
            fields = [f.strip().lower() for f in rdr.fieldnames if f]
            cand_cols = ["hex", "color", "candidate_hex", "candidate", "palette_hex"]
            col = None
            for c in cand_cols:
                if c in fields:
                    col = rdr.fieldnames[fields.index(c)]
                    break
            if col is not None:
                out: List[str] = []
                for r in rdr:
                    h = clean_hex(r.get(col, ""))
                    if h:
                        out.append(h)
                return out
    except Exception:
        pass

    # Fallback: newline-separated hexes
    out2: List[str] = []
    for line in text.splitlines():
        h = clean_hex(line.strip())
        if h:
            out2.append(h)
    return out2


# ============================================================
# 5) Forward prediction pipeline
# ============================================================

def predict_lin_from_bases_lin(
    bases_lin: List[Tuple[float, float, float]],
    weights: List[float],
    engine: str,
    ks_eps: float,
    yn_n: float,
    calibrator: Optional[Calibrator] = None,
    nn_model: Optional["ResidualNN"] = None,
    nn_scale: float = 0.25,
) -> Tuple[float, float, float]:
    """
    Forward prediction in **linear RGB**, given already-linearized bases.
    This helper exists mainly to speed up inverse search (avoid repeated HEX→linear conversions).
    """
    # Single-base identity shortcut: if the recipe contains one color at full weight,
    # return that color exactly before any KM/hybrid perturbation. This preserves
    # user-facing expectations and fixes exact-hit / one-color sanity cases.
    if len(bases_lin) == 1:
        lin = tuple(float(x) for x in bases_lin[0])
        if calibrator is not None:
            lin = calibrator.apply(lin)
        if nn_model is not None:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is not available; cannot apply NN residual.")
            nn_model.eval()
            with torch.no_grad():
                x = torch.tensor(lin, dtype=torch.float32)
                delta = nn_model(x).cpu().numpy()
            lin = tuple(clamp01(float(lin[i] + nn_scale * float(delta[i]))) for i in range(3))
        return lin  # type: ignore[return-value]

    w = normalize_weights(weights)

    # Optional: strength weighting (recipe-aware). If enabled, we rescale weights
    # based on the base colors themselves: w' = normalize(w * softplus(a·base + b)).
    if bool(st.session_state.get("strength_enable", False)):
        try:
            a = (
                float(st.session_state.get("strength_a0", -1.01798)),
                float(st.session_state.get("strength_a1", -3.69844)),
                float(st.session_state.get("strength_a2",  2.37642)),
            )
            bb = float(st.session_state.get("strength_b", 3.71610))
            gamma = float(st.session_state.get("strength_gamma", 1.0))
            w = apply_strength_weights(bases_lin, w, a=a, b=bb, gamma=gamma)
        except Exception:
            # Fail-safe: fall back to the original normalized weights.
            pass

    engine = str(engine).strip()
    if engine == "Linear":
        lin = mix_linear_rgb(bases_lin, w)
    elif engine == "KM":
        lin = mix_km(bases_lin, w, float(ks_eps))
    elif engine == "YN-KM":
        lin = mix_ynkm(bases_lin, w, n=float(yn_n), eps=float(ks_eps))
    elif engine == "Hybrid (KM ⊕ Linear)":
        t = float(st.session_state.get("hybrid_t", 0.425))
        lin = mix_hybrid_km_linear(bases_lin, w, eps=float(ks_eps), t=t)
    elif engine == "Hybrid (YN-KM ⊕ Linear)":
        t = float(st.session_state.get("hybrid_t", 0.425))
        lin = mix_hybrid_ynkm_linear(bases_lin, w, n=float(yn_n), eps=float(ks_eps), t=t)
    elif engine == "GeoLogMix-3 (XYZ)":
        eps_g = float(st.session_state.get("geolog_eps", 1e-6))
        preset = str(st.session_state.get("geolog_M_preset", "Identity")).strip().lower()

        if preset.startswith("messick"):
            M = _MESSICK_THETA_XYZ
        elif preset.startswith("fitted"):
            M_list = st.session_state.get("geolog_M_fitted")
            if isinstance(M_list, list) and len(M_list) == 3:
                try:
                    M = np.asarray(M_list, dtype=float)
                except Exception:
                    M = np.eye(3, dtype=float)
            else:
                # Fallback if user selects "Fitted" without having fitted yet.
                M = np.eye(3, dtype=float)
        else:
            M = np.eye(3, dtype=float)

        # Bias b: use fitted b if available; otherwise use scalar b_scale.
        b_scale = float(st.session_state.get("geolog_b_scale", 0.01))
        b_xyz = np.array([b_scale, b_scale, b_scale], dtype=float)
        if preset.startswith("fitted"):
            b_list = st.session_state.get("geolog_b_fitted")
            if isinstance(b_list, list) and len(b_list) == 3:
                try:
                    b_xyz = np.asarray(b_list, dtype=float)
                except Exception:
                    pass

        lin = mix_geologmix3(bases_lin, w, eps=float(eps_g), M=M, b_xyz=b_xyz)
    elif engine == "GeoLogMix-K (latent 8)":
        eps_g = float(st.session_state.get("geolog_eps", 1e-6))
        ridge = float(st.session_state.get("geolog_encode_ridge", 1e-3))
        iters = int(st.session_state.get("geolog_encode_iters", 120))

        # Cache per (bases, eps, ridge, iters) because inverse search calls this many times.
        cache = st.session_state.setdefault("_geologmixk_cache", {})
        key = (
            tuple(round(float(c), 9) for b in bases_lin for c in b),
            float(eps_g),
            float(ridge),
            int(iters),
        )
        logU = cache.get(key)
        if logU is None:
            logU = precompute_geologmixk_logU_bases(
                bases_lin=bases_lin,
                eps=float(eps_g),
                ridge=float(ridge),
                iters=int(iters),
                A=_GEOLOGMIXK_A,
            )
            cache[key] = logU

        lin = mix_geologmixk(
            bases_lin=bases_lin,
            w=w,
            eps=float(eps_g),
            ridge=float(ridge),
            iters=int(iters),
            logU_cache=logU,
            A=_GEOLOGMIXK_A,
        )
    else:
        raise ValueError(f"Unknown engine: {engine}")
    if calibrator is not None:
        lin = calibrator.apply(lin)

    if nn_model is not None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available; cannot apply NN residual.")
        nn_model.eval()
        with torch.no_grad():
            x = torch.tensor(lin, dtype=torch.float32)
            delta = nn_model(x).cpu().numpy()
        lin = tuple(clamp01(float(lin[i] + nn_scale * float(delta[i]))) for i in range(3))  # type: ignore[misc]

    return lin  # type: ignore[return-value]


def predict_lin(
    bases_hex: List[str],
    weights: List[float],
    engine: str,
    ks_eps: float,
    yn_n: float,
    calibrator: Optional[Calibrator] = None,
    nn_model: Optional["ResidualNN"] = None,
    nn_scale: float = 0.25,
) -> Tuple[float, float, float]:
    bases_lin = [hex_to_linear_rgb(h) for h in bases_hex]
    return predict_lin_from_bases_lin(
        bases_lin=bases_lin,
        weights=weights,
        engine=engine,
        ks_eps=ks_eps,
        yn_n=yn_n,
        calibrator=calibrator,
        nn_model=nn_model,
        nn_scale=nn_scale,
    )

def predict_hex(
    bases_hex: List[str],
    weights: List[float],
    engine: str,
    ks_eps: float,
    yn_n: float,
    calibrator: Optional[Calibrator] = None,
    nn_model: Optional["ResidualNN"] = None,
    nn_scale: float = 0.25,
) -> str:
    return linear_rgb_to_hex(predict_lin(bases_hex, weights, engine, ks_eps, yn_n, calibrator, nn_model, nn_scale))


def batch_evaluate(
    rows: List[Dict[str, Any]],
    engine: str,
    ks_eps: float,
    yn_n: float,
    calibrator: Optional[Calibrator],
    nn_model: Optional["ResidualNN"],
    nn_scale: float,
) -> Tuple[Any, Dict[str, float], Any]:
    """Batch-evaluate against a Trycolors-style reference CSV.

    Reports both:
    - RMSE in linear RGB (what the inverse search optimizes)
    - ΔE00 / Match% (what users typically care about perceptually, and what Trycolors surfaces)
    """
    results: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        pred = predict_hex(
            r["bases"], r["weights"],
            engine=engine,
            ks_eps=float(ks_eps),
            yn_n=float(yn_n),
            calibrator=calibrator,
            nn_model=nn_model,
            nn_scale=float(nn_scale),
        )
        de = float(delta_e00(hex_to_lab(pred), hex_to_lab(r["api_hex"])))
        results.append({
            "row": i,
            "group_id": r.get("group_id", ""),
            "base_hexes": ";".join(r["bases"]),
            "weights": ";".join([f"{x:g}" for x in r["weights"]]),
            "api_hex": r["api_hex"],
            "pred_hex": pred,
            "rmse": rmse_hex(pred, r["api_hex"]),
            "dE00": de,
            "match_pct": match_percent_from_de00(de),
        })

    if PANDAS_AVAILABLE:
        df = pd.DataFrame(results)
        summary = {
            "mean_rmse": float(df["rmse"].mean()) if len(df) else float("nan"),
            "mean_dE00": float(df["dE00"].mean()) if len(df) else float("nan"),
            "mean_match_pct": float(df["match_pct"].mean()) if len(df) else float("nan"),
            "median_dE00": float(df["dE00"].median()) if len(df) else float("nan"),
            "median_match_pct": float(df["match_pct"].median()) if len(df) else float("nan"),
        }
        group = (
            df.groupby("group_id")[["rmse", "dE00", "match_pct"]].mean().reset_index()
            if len(df) else pd.DataFrame(columns=["group_id", "rmse", "dE00", "match_pct"])
        )
        return df, summary, group

    # fallback
    mean_rmse = float(sum(r["rmse"] for r in results) / len(results)) if results else float("nan")
    mean_de = float(sum(r["dE00"] for r in results) / len(results)) if results else float("nan")
    mean_match = float(sum(r["match_pct"] for r in results) / len(results)) if results else float("nan")
    return results, {"mean_rmse": mean_rmse, "mean_dE00": mean_de, "mean_match_pct": mean_match}, {}


# ============================================================
# 6) Inverse search: "Get Mix Recipe"
# ============================================================

def compositions(total: int, k: int) -> Iterable[Tuple[int, ...]]:
    """All k-tuples of positive ints that sum to total."""
    if k <= 0 or total <= 0:
        return
    if k == 1:
        yield (total,)
        return
    # first part at least 1, leaving total-1 for remaining k-1
    for first in range(1, total - (k - 1) + 1):
        for rest in compositions(total - first, k - 1):
            yield (first,) + rest

def _compute_palette_hash(hexes: List[str]) -> str:
    """Compute a stable hash from a sorted list of hex colors."""
    sorted_hex = sorted(set(clean_hex(h) for h in hexes if h))
    joined = ",".join(sorted_hex)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

def _log_recipe_run(
    target_hex: str,
    palette_hexes: List[str],
    palette_hash: str,
    engine: str,
    ks_eps: float,
    yn_n: float,
    hybrid_t: float,
    max_colors: int,
    max_parts: int,
    prefilter_top_n: int,
    top_k: int,
    search_mode: str,
    tint_cap: int,
    loss_metric: str,
    calibrator_hash: Optional[str],
    nn_hash: Optional[str],
    best_solution: Dict[str, Any],
):
    """Append a JSON record to recipe_log.json with full run provenance."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "target_hex": target_hex,
        "palette_hexes": palette_hexes,
        "palette_hash": palette_hash,
        "engine": engine,
        "ks_eps": ks_eps,
        "yn_n": yn_n,
        "hybrid_t": hybrid_t,
        "max_colors": max_colors,
        "max_parts": max_parts,
        "prefilter_top_n": prefilter_top_n,
        "top_k": top_k,
        "search_mode": search_mode,
        "tint_cap": tint_cap,
        "loss_metric": loss_metric,
        "calibrator_hash": calibrator_hash,
        "nn_hash": nn_hash,
        "best_solution": {
            "loss": best_solution["loss"],
            "rmse": best_solution["rmse"],
            "pred_hex": best_solution["pred_hex"],
            "bases": best_solution["bases"],
            "parts": best_solution["parts"],
            "weights": best_solution["weights"],
            "dE00": best_solution.get("dE00"),
            "match_pct": best_solution.get("match_pct"),
        }
    }
    log_file = "recipe_log.json"
    try:
        # Read existing logs
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"Could not write recipe log: {e}")

def inverse_mix_recipe(
    target_hex: str,
    palette_hexes: List[str],
    engine: str,
    ks_eps: float,
    yn_n: float,
    calibrator: Optional[Calibrator],
    nn_model: Optional["ResidualNN"],
    nn_scale: float,
    max_colors: int,
    max_parts: int,
    prefilter_top_n: int = 12,
    top_k: int = 5,
    search_mode: str = "exhaustive",
    tint_cap: int = 10,
    loss_metric: str = "de00",  # forced to "de00" in client build
) -> List[Dict[str, Any]]:
    """
    Inverse search ("Get Mix Recipe") under discrete constraints.

    Two modes:

    1) search_mode="exhaustive"
       Exhaustive enumeration over integer compositions:
         - subsets of palette colors up to max_colors
         - k-tuples of positive integer parts summing to max_parts
       This matches the earlier UI semantics: weights = parts / max_parts.

    2) search_mode="dominant_micro"
       A Trycolors-like, high-precision mode that targets "dominant pigment + micro-tints".
       It keeps the exact-sum parts representation (total = max_parts), but constrains all
       non-dominant pigments to <= tint_cap parts (and >= 1 part).
       This allows high max_parts (e.g., 200 for 0.5% steps) without enumerating all compositions.

    Notes:
    - Loss is either RMSE in linear RGB or ΔE00 (perceptual).
    - Prefilter is a speed knob. To avoid throwing away useful "tint pigments", we always
      keep a few palette extremes (white/black + hue corners) in addition to the top-N nearest.
    """
    tgt = clean_hex(target_hex)
    if tgt is None:
        raise ValueError("Invalid target hex.")
    target_lin = hex_to_linear_rgb(tgt)
    target_lab = linear_rgb_to_lab(target_lin) if loss_metric == "de00" else None

    pal = [clean_hex(h) for h in palette_hexes]
    pal = [h for h in pal if h is not None]  # type: ignore[assignment]
    if not pal:
        raise ValueError("Palette is empty.")

    # Exact-hit shortcut: if the target color is already present in the palette,
    # return the one-color recipe immediately.
    if tgt in pal:
        exact_solution = {
            "loss": 0.0,
            "rmse": 0.0,
            "pred_hex": tgt,
            "target_hex": tgt,
            "bases": [tgt],
            "parts": [max_parts],
            "total_parts": int(max_parts),
            "weights": [1.0],
            "dE00": 0.0,
            "match_pct": 100.0,
        }
        return [exact_solution]

    max_colors = int(max(1, max_colors))
    max_parts = int(max(2, max_parts))
    top_k = int(max(1, top_k))

    # Precompute palette in linear RGB and optionally Lab (if using perceptual loss)
    pal_lin_full = [hex_to_linear_rgb(h) for h in pal]
    pal_lab_full = [linear_rgb_to_lab(p) for p in pal_lin_full] if loss_metric == "de00" else None

    def _luma(x: Tuple[float, float, float]) -> float:
        # sRGB luma coefficients (in linear domain).
        return 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]

    def _extreme_indices(pal_lin: List[Tuple[float, float, float]]) -> List[int]:
        if not pal_lin:
            return []
        R = [p[0] for p in pal_lin]
        G = [p[1] for p in pal_lin]
        B = [p[2] for p in pal_lin]
        luma = [_luma(p) for p in pal_lin]

        idxs: set[int] = set()
        idxs.add(int(np.argmin(luma)))  # darkest
        idxs.add(int(np.argmax(luma)))  # lightest
        idxs.add(int(np.argmax(R)))     # red-ish corner
        idxs.add(int(np.argmax(G)))     # green-ish corner
        idxs.add(int(np.argmax(B)))     # blue-ish corner

        # "Subtractive primaries" corners (rough heuristics in RGB):
        idxs.add(int(np.argmax([R[i] + G[i] - B[i] for i in range(len(pal_lin))])))  # yellow-ish
        idxs.add(int(np.argmax([R[i] + B[i] - G[i] for i in range(len(pal_lin))])))  # magenta-ish
        idxs.add(int(np.argmax([G[i] + B[i] - R[i] for i in range(len(pal_lin))])))  # cyan-ish

        return sorted(idxs)

    # Optional prefilter
    if prefilter_top_n and prefilter_top_n > 0 and prefilter_top_n < len(pal):
        # Use RGB distance for prefilter (fast)
        dists = [rmse_lin(x, target_lin) for x in pal_lin_full]
        order = list(range(len(pal)))
        order.sort(key=lambda i: dists[i])

        selected: set[int] = set(order[: int(prefilter_top_n)])
        # Always keep a few extremes to avoid excluding tint pigments.
        selected.update(_extreme_indices(pal_lin_full))

        order2 = [i for i in order if i in selected]
        pal = [pal[i] for i in order2]
        pal_lin = [pal_lin_full[i] for i in order2]
        if pal_lab_full is not None:
            pal_lab = [pal_lab_full[i] for i in order2]
        else:
            pal_lab = None
    else:
        pal_lin = pal_lin_full
        pal_lab = pal_lab_full

    best: List[Dict[str, Any]] = []

    def consider(sol: Dict[str, Any]) -> None:
        nonlocal best
        best.append(sol)
        best.sort(key=lambda x: x["loss"])  # sort by active loss
        if len(best) > top_k:
            best = best[:top_k]

    mode = str(search_mode).lower().strip()
    if mode not in ("exhaustive", "dominant_micro"):
        mode = "exhaustive"

    cap = int(max(0, tint_cap))

    # Search
    for k in range(1, max_colors + 1):
        for idxs in combinations(range(len(pal)), k):
            bases = [pal[i] for i in idxs]
            bases_lin = [pal_lin[i] for i in idxs]
            bases_lab = [pal_lab[i] for i in idxs] if pal_lab is not None else None

            if mode == "dominant_micro" and k > 1:
                if cap <= 0:
                    continue

                # One dominant base, remaining are micro-tints (each in [1..cap]).
                for dom_pos in range(k):
                    num_tints = k - 1
                    # Enumerate tint parts; dominant part is what's left.
                    for tint_parts in product(range(1, cap + 1), repeat=num_tints):
                        sum_t = int(sum(tint_parts))
                        if sum_t >= max_parts:
                            continue
                        dom_part = int(max_parts - sum_t)
                        if dom_part < 1:
                            continue

                        parts = [0] * k
                        parts[dom_pos] = dom_part
                        ti = 0
                        for j in range(k):
                            if j == dom_pos:
                                continue
                            parts[j] = int(tint_parts[ti])
                            ti += 1

                        pred_lin = predict_lin_from_bases_lin(
                            bases_lin=bases_lin,
                            weights=list(parts),
                            engine=engine,
                            ks_eps=float(ks_eps),
                            yn_n=float(yn_n),
                            calibrator=calibrator,
                            nn_model=nn_model,
                            nn_scale=float(nn_scale),
                        )
                        # Compute loss
                        if loss_metric == "rmse":
                            e = rmse_lin(pred_lin, target_lin)
                        else:  # de00
                            pred_lab = linear_rgb_to_lab(pred_lin)
                            e = delta_e00(pred_lab, target_lab)  # type: ignore[arg-type]
                        if best and len(best) >= top_k and e >= best[-1]["loss"]:
                            continue
                        pred_hex = linear_rgb_to_hex(pred_lin)
                        consider({
                            "loss": float(e),
                            "rmse": rmse_lin(pred_lin, target_lin),
                            "pred_hex": pred_hex,
                            "target_hex": tgt,
                            "bases": bases,
                            "parts": list(parts),
                            "total_parts": int(max_parts),
                            "weights": [p / max_parts for p in parts],
                        })

            else:
                # Exhaustive exact-sum parts grid (positive integers).
                for parts in compositions(max_parts, k):
                    pred_lin = predict_lin_from_bases_lin(
                        bases_lin=bases_lin,
                        weights=list(parts),
                        engine=engine,
                        ks_eps=float(ks_eps),
                        yn_n=float(yn_n),
                        calibrator=calibrator,
                        nn_model=nn_model,
                        nn_scale=float(nn_scale),
                    )
                    if loss_metric == "rmse":
                        e = rmse_lin(pred_lin, target_lin)
                    else:  # de00
                        pred_lab = linear_rgb_to_lab(pred_lin)
                        e = delta_e00(pred_lab, target_lab)  # type: ignore[arg-type]
                    if best and len(best) >= top_k and e >= best[-1]["loss"]:
                        continue
                    pred_hex = linear_rgb_to_hex(pred_lin)
                    consider({
                        "loss": float(e),
                        "rmse": rmse_lin(pred_lin, target_lin),
                        "pred_hex": pred_hex,
                        "target_hex": tgt,
                        "bases": bases,
                        "parts": list(parts),
                        "total_parts": int(max_parts),
                        "weights": [p / max_parts for p in parts],
                    })

    return best

# ============================================================
# 7) UI
# ============================================================

st.markdown("<h3 style='margin:0 0 4px 0;font-size:15px;color:#e65100;'>Forward Mix Engine</h3>", unsafe_allow_html=True)



# ============================================================
# Auto-load bundled calibrator (if present)
# ============================================================
BUNDLED_CALIBRATOR_FILENAME_V2 = "calibrator_trycolors_palette_extended_deg3_ridge1e3_v2.json"
BUNDLED_CALIBRATOR_FILENAME_V1 = "calibrator_trycolors_palette_tuned_deg3_ridge1e3.json"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Prefer the expanded (v2) calibrator if present; fall back to the original (v1).
BUNDLED_CALIBRATOR_JSON = os.path.join(APP_DIR, BUNDLED_CALIBRATOR_FILENAME_V2)
if not os.path.exists(BUNDLED_CALIBRATOR_JSON):
    BUNDLED_CALIBRATOR_JSON = os.path.join(APP_DIR, BUNDLED_CALIBRATOR_FILENAME_V1)

if "auto_loaded_bundled_calibrator" not in st.session_state:
    st.session_state["bundled_calibrator_status"] = {
        "expected_path": BUNDLED_CALIBRATOR_JSON,
        "found": False,
        "loaded": False,
        "error": None,
    }
    try:
        if os.path.exists(BUNDLED_CALIBRATOR_JSON):
            st.session_state["bundled_calibrator_status"]["found"] = True
            _cal_text = open(BUNDLED_CALIBRATOR_JSON, "r", encoding="utf-8").read()

            # Parse JSON once so we can keep metadata (palette scope, etc.)
            _d: Dict[str, Any] = {}
            try:
                _tmp = json.loads(_cal_text)
                if isinstance(_tmp, dict):
                    _d = _tmp
            except Exception:
                _d = {}

            # Store metadata separately (Calibrator objects are intentionally minimal).
            st.session_state["calibrator_meta"] = _d

            # Load calibrator object
            st.session_state.calibrator = load_calibrator_any(_cal_text)
            st.session_state["bundled_calibrator_status"]["loaded"] = True

            # If JSON contains recommended strength parameters, adopt them ONCE.
            # NOTE: we deliberately do NOT adopt engine / ks_eps / yn_n / hybrid_t
            # from the bundled calibrator. The locked client profile remains authoritative.
            if isinstance(_d, dict):
                _sp = _d.get("strength_params")
                if isinstance(_sp, dict):
                    if "a0" in _sp and "strength_a0" not in st.session_state:
                        st.session_state["strength_a0"] = float(_sp.get("a0"))
                    if "a1" in _sp and "strength_a1" not in st.session_state:
                        st.session_state["strength_a1"] = float(_sp.get("a1"))
                    if "a2" in _sp and "strength_a2" not in st.session_state:
                        st.session_state["strength_a2"] = float(_sp.get("a2"))
                    if "b" in _sp and "strength_b" not in st.session_state:
                        st.session_state["strength_b"] = float(_sp.get("b"))
                    if "gamma" in _sp and "strength_gamma" not in st.session_state:
                        st.session_state["strength_gamma"] = float(_sp.get("gamma"))

    except Exception as e:
        st.session_state["bundled_calibrator_status"]["error"] = str(e)

    st.session_state["auto_loaded_bundled_calibrator"] = True

# Locked client profile – force these settings regardless of any previous values.
# This prevents bundled calibrator metadata from silently shifting the startup profile.
if "client_profile_locked_applied" not in st.session_state:
    st.session_state["engine"] = "Hybrid (KM ⊕ Linear)"
    st.session_state["ks_eps"] = 0.015
    st.session_state["hybrid_t"] = 0.28
    st.session_state["yn_n"] = 1.5   # not used for KM but kept for consistency
    st.session_state["client_profile_locked_applied"] = True

# Hard default: keep strength weighting OFF unless the user explicitly enables it.
if "strength_enable" not in st.session_state:
    st.session_state["strength_enable"] = False

# ---------------- Sidebar: global params ----------------
with st.sidebar:
    st.markdown("<h3 style='margin:0 0 4px 0;font-size:13px;color:#e65100;'>Engine + params</h3>", unsafe_allow_html=True)
    # Bundled calibrator status (autoload)
    _bc = st.session_state.get("bundled_calibrator_status", {})
    if isinstance(_bc, dict):
        if bool(_bc.get("loaded", False)):
            st.success(f"Auto-loaded calibrator: {os.path.basename(str(_bc.get('expected_path','')))}")
        elif bool(_bc.get("found", False)) and _bc.get("error"):
            st.warning(f"Bundled calibrator found but failed to load: {_bc.get('error')}")
        else:
            st.info(f"No bundled calibrator found. Expected: {os.path.basename(str(_bc.get('expected_path','')))}")

    # For client build: default engine is Hybrid (KM ⊕ Linear), and other engines are hidden behind an expander.
    with st.expander("Advanced / Experimental (Engine selection)", expanded=False):
        engine = st.selectbox(
            "Engine",
            ["Hybrid (KM ⊕ Linear)", "Hybrid (YN-KM ⊕ Linear)", "YN-KM", "KM", "Linear", "GeoLogMix-3 (XYZ)", "GeoLogMix-K (latent 8)"],
            index=0,
            key="engine"
        )
    ks_eps = st.number_input("KS_EPS (KM / YN-KM)", value=float(st.session_state.get("ks_eps", 0.015)), min_value=1e-9, max_value=0.2, step=1e-4, format="%.6f", help="Regularization floor used in the KM K/S computation. If any base has a channel near 0 (e.g., B=0 for yellows/reds), very small KS_EPS can collapse that channel in the mix. Try 1e-3 to 2e-2 for Trycolors-like micro-tint recipes.", key="ks_eps")
    yn_n = st.number_input("Yule–Nielsen n (YN-KM)", value=float(st.session_state.get("yn_n", 1.5)), min_value=0.0, max_value=10.0, step=0.1, key="yn_n")
    if engine.startswith("Hybrid"):
        hybrid_t = st.slider("Hybrid blend t (0=KM/YN-KM, 1=Linear)", min_value=0.0, max_value=1.0, value=float(st.session_state.get("hybrid_t", 0.28)), step=0.01, key="hybrid_t", help="Blend happens in linear RGB: pred = (1-t)*KM + t*Linear. This is a cheap, robust fix for KM over-darkening on complementary pairs.")
    else:
        hybrid_t = float(st.session_state.get("hybrid_t", 0.28))



    # ---------------- GeoLogMix params (experimental) ----------------
    # We keep these variables defined even when GeoLogMix isn't selected to avoid NameError
    geolog_eps_default = float(st.session_state.get("geolog_eps", 1e-6))
    geolog_M_preset_default = str(st.session_state.get("geolog_M_preset", "Identity"))
    geolog_encode_iters_default = int(st.session_state.get("geolog_encode_iters", 120))
    geolog_encode_ridge_default = float(st.session_state.get("geolog_encode_ridge", 1e-3))
    geolog_b_scale_default = float(st.session_state.get("geolog_b_scale", 0.01))

    if str(engine).startswith("GeoLogMix"):
        st.markdown("<hr style='margin:4px 0;border-top:1px solid #ffe0b2;'>", unsafe_allow_html=True)
        st.header("GeoLogMix (experimental)")
        geolog_eps = st.number_input(
            "GeoLogMix ε (positivity floor)",
            value=float(geolog_eps_default),
            min_value=1e-12,
            max_value=1e-2,
            step=1e-6,
            format="%.8f",
            key="geolog_eps",
            help="GeoLogMix mixes in log space, so all internal coordinates must stay > 0. "
                 "Increase ε if you see numerical instability for very dark colors.",
        )

        if str(engine) == "GeoLogMix-3 (XYZ)":
            preset_opts = ["Identity", "Messick Θ_XYZ", "Fitted (from CSV)"]
            p0 = geolog_M_preset_default.strip().lower()
            if p0.startswith("messick"):
                idx_preset = 1
            elif p0.startswith("fitted"):
                idx_preset = 2
            else:
                idx_preset = 0

            geolog_M_preset = st.selectbox(
                "GeoLogMix-3 M preset",
                preset_opts,
                index=int(idx_preset),
                key="geolog_M_preset",
                help="Identity is the safest default. 'Messick Θ_XYZ' is a research preset from Paper 1. "
                     "'Fitted (from CSV)' learns a palette-specific M from labeled Trycolors samples and is usually the best choice.",
            )


            # Bias b (XYZ): an additive offset before log-mixing (critical for dark colors like black).
            geolog_b_scale = st.slider(
                "GeoLogMix bias b (XYZ) scale",
                min_value=0.0,
                max_value=0.05,
                value=float(geolog_b_scale_default),
                step=0.001,
                key="geolog_b_scale",
                help="Adds a small XYZ offset b before taking logs: u = invM(XYZ + b). This avoids log(0) issues and improves black/shade behavior. If you fit (M,b) from CSV, the fitted b overrides this slider.",
            )

            # If we have a fitted b in session_state, show it for transparency.
            try:
                b_fit = st.session_state.get("geolog_b_fitted")
                if isinstance(b_fit, list) and len(b_fit) == 3:
                    bf = [float(x) for x in b_fit]
                    st.caption(f"Fitted b (XYZ): [{bf[0]:.6f}, {bf[1]:.6f}, {bf[2]:.6f}]")
            except Exception:
                pass

            with st.expander("Load fitted GeoLogMix-3 (M + b) JSON", expanded=False):
                jfile = st.file_uploader("GeoLogMix (M,b) JSON", type=["json"], key="geolog_load_json")
                if jfile is not None:
                    try:
                        payload = json.loads(jfile.getvalue().decode("utf-8", errors="ignore"))
                        Mj = np.asarray(payload.get("M", payload.get("m")), dtype=float)
                        bj = np.asarray(payload.get("b", payload.get("bias")), dtype=float).reshape(3)
                        if Mj.shape != (3, 3):
                            raise ValueError("M must be 3×3.")
                        st.session_state["geolog_M_fitted"] = Mj.tolist()
                        st.session_state["geolog_b_fitted"] = bj.tolist()
                        st.session_state["geolog_M_preset"] = "Fitted (from CSV)"
                        st.success("Loaded fitted (M,b). Select 'Fitted (from CSV)' in the preset dropdown.")
                    except Exception as e:
                        st.error(f"Could not load JSON: {e}")
            # --- Fit M (optional) ---
            with st.expander("Fit GeoLogMix-3 (M + b) from CSV", expanded=False):
                st.caption(
                    "GeoLogMix-3 is a *parametric* model. For Trycolors-like behavior (especially complements and black shading), "
                    "you typically need to fit a palette-specific 3×3 matrix M **and** a small positive XYZ bias b. "
                    "This fitter learns (M,b) from labeled rows: base_hexes + weights → api_hex."
                )
                geolog_fit_csvs = st.file_uploader("Fit (M,b) CSV (Trycolors format)", type=["csv"], accept_multiple_files=True, key="geolog_fit_csvs")
                colFM1, colFM2 = st.columns(2)
                with colFM1:
                    geolog_fit_metric = st.selectbox("Loss", ["RMSE (XYZ)", "RMSE (linear RGB)"], index=0, key="geolog_fit_metric")
                    geolog_fit_reg = st.slider("Regularize M → Identity", min_value=0.0, max_value=0.5, value=float(st.session_state.get("geolog_fit_reg", 0.05)), step=0.01, key="geolog_fit_reg")
                    geolog_fit_regb = st.slider("Regularize b → 0.01", min_value=0.0, max_value=0.5, value=float(st.session_state.get("geolog_fit_regb", 0.05)), step=0.01, key="geolog_fit_regb")
                with colFM2:
                    geolog_fit_maxiter = st.slider("Max iterations", min_value=50, max_value=800, value=int(st.session_state.get("geolog_fit_maxiter", 250)), step=50, key="geolog_fit_maxiter")
                    geolog_fit_seed = st.number_input("Seed", min_value=0, max_value=9999, value=int(st.session_state.get("geolog_fit_seed", 0)), step=1, key="geolog_fit_seed")
            
                fitMb_clicked = st.button("Fit (M,b) now", key="geolog_fitMb_btn")
                if fitMb_clicked:
                    if not geolog_fit_csvs:
                        st.error("Upload a CSV first.")
                    else:
                        try:
                            import scipy.optimize  # type: ignore
                        except Exception:
                            st.error("SciPy is required to fit (M,b). Install: pip install scipy")
                            st.stop()
            
                        try:
                            rows = []
                            for _f in geolog_fit_csvs:
                                _text = _f.getvalue().decode("utf-8", errors="ignore")
                                rows.extend(parse_trycolors_csv(_text))
                            if not rows:
                                raise ValueError("No usable labeled rows found (api_hex + base_hexes + weights).")
            
                            eps_fit = float(st.session_state.get("geolog_eps", 1e-6))
                            reg_id = float(geolog_fit_reg)
                            reg_b = float(geolog_fit_regb)
                            metric = str(geolog_fit_metric).strip().lower()
            
                            # --- Precompute XYZ (and optionally linear RGB targets) ---
                            xyz_bases_list: List[np.ndarray] = []
                            w_list: List[np.ndarray] = []
                            xyz_t_list: List[np.ndarray] = []
                            lin_t_list: List[np.ndarray] = []
                            all_two = True
                            for r in rows:
                                bases_lin = [hex_to_linear_rgb(h) for h in r["bases"]]
                                xyz_bases = np.stack([linear_rgb_to_xyz(b) for b in bases_lin], axis=0)  # Nx3
                                w0 = normalize_weights(r["weights"])
                                xyz_bases_list.append(np.asarray(xyz_bases, dtype=float))
                                w_list.append(np.asarray(w0, dtype=float))
                                lin_t = np.asarray(hex_to_linear_rgb(r["api_hex"]), dtype=float)
                                lin_t_list.append(lin_t)
                                xyz_t_list.append(np.asarray(linear_rgb_to_xyz((float(lin_t[0]), float(lin_t[1]), float(lin_t[2]))), dtype=float))
                                if xyz_bases.shape[0] != 2:
                                    all_two = False
            
                            # Parametrization: 9 params for invertible M (LU-like), + 3 params for positive b via exp().
                            b0 = 0.01  # scale factor for b; keeps magnitudes sane
                            def M_from_params_lu(p: np.ndarray) -> np.ndarray:
                                p = np.asarray(p, float).ravel()
                                # Lower triangular L with exp diag
                                L = np.array(
                                    [
                                        [np.exp(p[0]), 0.0, 0.0],
                                        [p[1], np.exp(p[2]), 0.0],
                                        [p[3], p[4], np.exp(p[5])],
                                    ],
                                    dtype=float,
                                )
                                # Upper triangular U with ones diag
                                U = np.array(
                                    [
                                        [1.0, p[6], p[7]],
                                        [0.0, 1.0, p[8]],
                                        [0.0, 0.0, 1.0],
                                    ],
                                    dtype=float,
                                )
                                return L @ U
            
                            def params_to_Mb(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                                theta = np.asarray(theta, float).ravel()
                                M = M_from_params_lu(theta[:9])
                                q = theta[9:12]
                                b = np.exp(q) * b0
                                return M, b
            
                            # Fast vectorized objective when all mixes are 2-color.
                            if all_two:
                                X1 = np.stack([x[0] for x in xyz_bases_list], axis=0)  # Sx3
                                X2 = np.stack([x[1] for x in xyz_bases_list], axis=0)  # Sx3
                                W1 = np.array([w[0] for w in w_list], dtype=float).reshape(-1, 1)
                                W2 = np.array([w[1] for w in w_list], dtype=float).reshape(-1, 1)
                                Xt = np.stack(xyz_t_list, axis=0)  # Sx3
                                Lt = np.stack(lin_t_list, axis=0)  # Sx3
            
                                def obj(theta: np.ndarray) -> float:
                                    M, b = params_to_Mb(theta)
                                    try:
                                        invM = np.linalg.inv(M)
                                    except Exception:
                                        return 1e9
            
                                    U1 = np.maximum((X1 + b[None, :]) @ invM.T, eps_fit)
                                    U2 = np.maximum((X2 + b[None, :]) @ invM.T, eps_fit)
                                    logU = W1 * np.log(U1) + W2 * np.log(U2)
                                    Umix = np.exp(logU)
                                    Xpred = Umix @ M.T - b[None, :]
            
                                    if metric.startswith("rmse (linear"):
                                        Lpred = Xpred @ _XYZ_TO_RGB_D65.T
                                        Lpred = np.clip(Lpred, 0.0, 1.0)
                                        err = float(np.sqrt(np.mean((Lpred - Lt) ** 2)))
                                    else:
                                        err = float(np.sqrt(np.mean((Xpred - Xt) ** 2)))
            
                                    reg = reg_id * float(np.mean((M - np.eye(3)) ** 2)) + reg_b * float(np.mean((b - b0) ** 2))
                                    return err + reg
            
                            else:
                                # Generic (slower) objective for N-color rows.
                                def obj(theta: np.ndarray) -> float:
                                    M, b = params_to_Mb(theta)
                                    try:
                                        invM = np.linalg.inv(M)
                                    except Exception:
                                        return 1e9
            
                                    err_sum = 0.0
                                    for xyz_bases, ww, xyz_t, lin_t in zip(xyz_bases_list, w_list, xyz_t_list, lin_t_list):
                                        U = (invM @ (xyz_bases + b[None, :]).T).T
                                        U = np.maximum(U, eps_fit)
                                        logU = np.log(U)
                                        logU_mix = (ww[:, None] * logU).sum(axis=0)
                                        Umix = np.exp(logU_mix)
                                        xyz_pred = (M @ Umix) - b
                                        if metric.startswith("rmse (linear"):
                                            lin_pred = xyz_to_linear_rgb_arr(xyz_pred)
                                            lin_pred = np.clip(lin_pred, 0.0, 1.0)
                                            err_sum += float(np.sqrt(np.mean((lin_pred - lin_t) ** 2)))
                                        else:
                                            err_sum += float(np.sqrt(np.mean((xyz_pred - xyz_t) ** 2)))
                                    err = err_sum / max(len(xyz_bases_list), 1)
                                    reg = reg_id * float(np.mean((M - np.eye(3)) ** 2)) + reg_b * float(np.mean((b - b0) ** 2))
                                    return err + reg
            
                            # Optimize (Powell is robust and avoids expensive numerical gradients).
                            rng = np.random.default_rng(int(geolog_fit_seed))
                            theta0 = np.zeros(12, dtype=float)
                            theta0[9:] = np.log(1.0)  # b starts at b0
                            theta0 = theta0 + 0.05 * rng.standard_normal(theta0.shape)
            
                            res = scipy.optimize.minimize(obj, theta0, method="Powell", options={"maxiter": int(geolog_fit_maxiter), "xtol": 1e-4, "ftol": 1e-4})
                            M_fit, b_fit = params_to_Mb(res.x)
            
                            st.session_state["geolog_M_fitted"] = M_fit.tolist()
                            st.session_state["geolog_b_fitted"] = b_fit.tolist()
                            st.session_state["geolog_M_preset"] = "Fitted (from CSV)"
            
                            st.success(f"Fitted (M,b) on {len(xyz_bases_list)} rows. Final objective: {float(res.fun):.6f}")
            
                            payload = {
                                "type": "geologmix3_Mb",
                                "engine": "GeoLogMix-3 (XYZ, affine)",
                                "eps": float(eps_fit),
                                "reg_identity": float(reg_id),
                                "reg_bias": float(reg_b),
                                "maxiter": int(geolog_fit_maxiter),
                                "M": M_fit.tolist(),
                                "b": b_fit.tolist(),
                            }
                            st.download_button(
                                "Download fitted (M,b) JSON",
                                data=json.dumps(payload, indent=2).encode("utf-8"),
                                file_name="geologmix3_Mb_fitted.json",
                                mime="application/json",

                            )
            
                        except Exception as e:
                            st.error(f"Failed to fit (M,b): {e}")
        else:
            geolog_M_preset = str(geolog_M_preset_default)

        if str(engine) == "GeoLogMix-K (latent 8)":
            geolog_encode_iters = st.slider(
                "GeoLogMix-K encoder iterations",
                min_value=20,
                max_value=300,
                value=int(geolog_encode_iters_default),
                step=10,
                key="geolog_encode_iters",
                help="GeoLogMix-K encodes XYZ into an 8D positive latent via projected gradient NNLS.",
            )
            geolog_encode_ridge = st.number_input(
                "GeoLogMix-K ridge (encoder)",
                value=float(geolog_encode_ridge_default),
                min_value=0.0,
                max_value=1.0,
                step=1e-4,
                format="%.6f",
                key="geolog_encode_ridge",
                help="Ridge stabilizes the NNLS encoding (prevents wildly large latent coefficients).",
            )
        else:
            geolog_encode_iters = int(geolog_encode_iters_default)
            geolog_encode_ridge = float(geolog_encode_ridge_default)
    else:
        geolog_eps = float(geolog_eps_default)
        geolog_M_preset = str(geolog_M_preset_default)
        geolog_encode_iters = int(geolog_encode_iters_default)
        geolog_encode_ridge = float(geolog_encode_ridge_default)


    

    st.markdown("---")
    st.header("Strength weighting (optional)")
    st.caption("Recipe-aware weight re-scaling:  w' = normalize(w * softplus(a·base_linRGB + b)).  This often helps match Trycolors-like behavior on tints / complementary pairs.")
    # Learned defaults from the provided trials (data.csv)
    LEARNED_A0, LEARNED_A1, LEARNED_A2, LEARNED_B = (-1.01798, -3.69844, 2.37642, 3.71610)

    enable_strength = st.checkbox("Enable strength weighting", value=bool(st.session_state.get("strength_enable", False)), key="strength_enable")  # default False

    strength_gamma = st.slider("Strength effect (γ)", min_value=0.0, max_value=2.0, value=float(st.session_state.get("strength_gamma", 1.0)), step=0.05, key="strength_gamma")

    colS1, colS2 = st.columns(2)
    with colS1:
        strength_a0 = st.number_input("a0 (R)", value=float(st.session_state.get("strength_a0", LEARNED_A0)), step=0.1, key="strength_a0")
        strength_a1 = st.number_input("a1 (G)", value=float(st.session_state.get("strength_a1", LEARNED_A1)), step=0.1, key="strength_a1")
    with colS2:
        strength_a2 = st.number_input("a2 (B)", value=float(st.session_state.get("strength_a2", LEARNED_A2)), step=0.1, key="strength_a2")
        strength_b  = st.number_input("b (bias)", value=float(st.session_state.get("strength_b", LEARNED_B)), step=0.1, key="strength_b")

    if st.button("Reset strength params to learned defaults"):
        st.session_state["strength_enable"] = False
        st.session_state["strength_a0"] = float(LEARNED_A0)
        st.session_state["strength_a1"] = float(LEARNED_A1)
        st.session_state["strength_a2"] = float(LEARNED_A2)
        st.session_state["strength_b"]  = float(LEARNED_B)
        st.session_state["strength_gamma"] = 1.0
        st.rerun()


    st.markdown("---")
    st.header("Log-poly calibrator")

    cal_kind = st.selectbox(
        "Calibrator type",
        ["Per-channel log-poly (legacy)", "3D log-poly (cross-channel, recommended)"],
        index=int(st.session_state.get("cal_kind_index", 1)),
        key="cal_kind",
        help="3D mode fits cross-channel terms in log-linearRGB and is often much closer to Trycolors than independent per-channel polynomials.",
    )
    # Persist index to avoid Streamlit warnings if list changes.
    try:
        st.session_state["cal_kind_index"] = 1 if str(cal_kind).startswith("3D") else 0
    except Exception:
        pass

    if str(cal_kind).startswith("3D"):
        cal_deg = st.number_input("Degree", min_value=1, max_value=3, value=int(st.session_state.get("cal_deg", 2)), step=1, key="cal_deg")
        cal_ridge = st.number_input(
            "Ridge (λ)",
            value=float(st.session_state.get("cal_ridge", 1e-3)),
            min_value=0.0,
            max_value=1.0,
            step=1e-4,
            format="%.6f",
            key="cal_ridge",
            help="Ridge stabilizes the multivariate fit; try 1e-4–1e-2.",
        )
    else:
        cal_deg = st.number_input("Degree", min_value=1, max_value=7, value=int(st.session_state.get("cal_deg", 1)), step=1, key="cal_deg")
        cal_ridge = 0.0

    cal_fit_csv = st.file_uploader("Fit CSV (Trycolors format)", type=["csv"], key="cal_fit_csv")
    col1, col2 = st.columns(2)
    with col1:
        fit_clicked = st.button("Fit log-poly")
    with col2:
        clear_cal = st.button("Clear")

    cal_json_file = st.file_uploader("Load calibrator JSON", type=["json"], key="cal_json")
    if cal_json_file is not None:
        try:
            cal_text = cal_json_file.getvalue().decode("utf-8", errors="ignore")
            st.session_state.calibrator = load_calibrator_any(cal_text)
            st.success("Loaded calibrator JSON.")
        except Exception as e:
            st.error(f"Failed to load calibrator: {e}")

    if clear_cal:
        st.session_state.calibrator = None
        st.info("Cleared calibrator.")

    if fit_clicked:
        if cal_fit_csv is None:
            st.error("Upload a CSV first.")
        else:
            try:
                text = cal_fit_csv.getvalue().decode("utf-8", errors="ignore")
                rows = parse_trycolors_csv(text)
                if not rows:
                    raise ValueError("No usable labeled rows found (api_hex + base_hexes + weights).")

                # Build samples: base prediction (engine) -> true
                P: List[Tuple[float, float, float]] = []
                T: List[Tuple[float, float, float]] = []
                for r in rows:
                    pred_lin = predict_lin(
                        r["bases"], r["weights"],
                        engine=engine,
                        ks_eps=float(ks_eps),
                        yn_n=float(yn_n),
                        calibrator=None,
                        nn_model=None,
                        nn_scale=0.0,
                    )
                    P.append(pred_lin)
                    T.append(hex_to_linear_rgb(r["api_hex"]))

                Pn = np.array(P, dtype=float)
                Tn = np.array(T, dtype=float)

                eps = 1e-6

                if str(cal_kind).startswith("3D"):
                    # Multivariate ridge regression in log-linearRGB with cross terms.
                    X = np.log(np.clip(Pn, eps, 1.0))  # Sx3
                    Y = np.log(np.clip(Tn, eps, 1.0))  # Sx3
                    x_mins = [float(np.min(X[:, ch])) for ch in range(3)]
                    x_maxs = [float(np.max(X[:, ch])) for ch in range(3)]

                    Phi = Calibrator3D._phi_matrix(X, int(cal_deg))  # SxF
                    ridge = float(cal_ridge)
                    F = int(Phi.shape[1])
                    A = Phi.T @ Phi + ridge * np.eye(F, dtype=float)
                    B = Phi.T @ Y
                    W = np.linalg.solve(A, B)  # Fx3

                    # Compute palette hash from all bases used in training rows
                    all_bases = set()
                    for r in rows:
                        for b in r["bases"]:
                            all_bases.add(clean_hex(b))
                    palette_hash = _compute_palette_hash(list(all_bases))

                    cal_tmp = Calibrator3D(
                        engine=str(engine),
                        ks_eps=float(ks_eps),
                        yn_n=float(yn_n),
                        hybrid_t=float(st.session_state.get("hybrid_t", 0.425)),
                        degree=int(cal_deg),
                        ridge=float(ridge),
                        W=W.tolist(),
                        x_mins=x_mins,
                        x_maxs=x_maxs,
                        palette_hash=palette_hash,
                    )
                else:
                    coeffs: List[List[float]] = []
                    x_mins: List[float] = []
                    x_maxs: List[float] = []
                    for ch in range(3):
                        X = np.log(np.clip(Pn[:, ch], eps, 1.0))
                        Y = np.log(np.clip(Tn[:, ch], eps, 1.0))
                        # Record training range for safe clamping at inference time.
                        x_mins.append(float(np.min(X)))
                        x_maxs.append(float(np.max(X)))
                        c = np.polyfit(X, Y, int(cal_deg))
                        coeffs.append([float(x) for x in c.tolist()])

                    all_bases = set()
                    for r in rows:
                        for b in r["bases"]:
                            all_bases.add(clean_hex(b))
                    palette_hash = _compute_palette_hash(list(all_bases))

                    cal_tmp = Calibrator(
                        engine=str(engine),
                        ks_eps=float(ks_eps),
                        yn_n=float(yn_n),
                        hybrid_t=float(st.session_state.get("hybrid_t", 0.425)),
                        degree=int(cal_deg),
                        coeffs=coeffs,
                        x_mins=x_mins,
                        x_maxs=x_maxs,
                        palette_hash=palette_hash,
                    )

                st.session_state.calibrator = cal_tmp

                # Fit diagnostics (train-set RMSE): engine-only vs +calibrator
                rmse_before = float(np.mean([rmse_lin(P[i], T[i]) for i in range(len(P))])) if P else float("nan")
                rmse_after = float(np.mean([rmse_lin(cal_tmp.apply(P[i]), T[i]) for i in range(len(P))])) if P else float("nan")
                st.session_state.cal_fit_report = {
                    "n_rows": int(len(P)),
                    "engine": str(engine),
                    "ks_eps": float(ks_eps),
                    "yn_n": float(yn_n),
                    "degree": int(cal_deg),
                    "rmse_before": rmse_before,
                    "rmse_after": rmse_after,
                }

                st.success("Log-poly calibrator fitted.")
                st.caption(f"Fit diagnostics (train): mean RMSE before={rmse_before:.6f}, after={rmse_after:.6f}")

                # If the calibrator is clearly harmful on the training set, default it to OFF.
                if np.isfinite(rmse_before) and np.isfinite(rmse_after) and rmse_after > rmse_before * 1.01:
                    st.warning(
                        "Calibrator worsened the training RMSE. "
                        "Try degree=1–2, collect more representative training data, "
                        "or disable the calibrator for this palette/recipe regime."
                    )
                    st.session_state.use_calibrator = False
            except Exception as e:
                st.error(f"Fit failed: {e}")

    cal_obj = st.session_state.get("calibrator")
    if isinstance(cal_obj, (Calibrator, Calibrator3D)):
        st.download_button(
            "Download calibrator JSON",
            data=cal_obj.to_json().encode("utf-8"),
            file_name="logpoly_calibrator.json",
            mime="application/json",

        )

    # Enable/disable calibrator (useful for quick A/B)
    use_calibrator = st.checkbox(
        "Enable log-poly calibrator",
        value=False,  # default OFF
        disabled=not isinstance(st.session_state.get("calibrator"), (Calibrator, Calibrator3D)),
        help="If enabled, the fitted/loaded calibrator is applied after the forward mixer. Note: changing engine/params after fitting will disable it for safety.",
        key="use_calibrator",
    )

    rep = st.session_state.get("cal_fit_report")
    if isinstance(rep, dict):
        try:
            rb = float(rep.get("rmse_before", float("nan")))
            ra = float(rep.get("rmse_after", float("nan")))
            nrows = int(rep.get("n_rows", 0))
            st.caption(f"Calibrator fit report (train): RMSE {rb:.6f} → {ra:.6f} on {nrows} rows.")
        except Exception:
            pass


    st.markdown("---")
    st.header("NN residual")

    if not TORCH_AVAILABLE:
        st.info("Install PyTorch to enable NN residual.")
        use_nn = False
        nn_scale = 0.0
    else:
        nn_scale = st.slider("NN residual scale", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        use_nn = st.checkbox("Enable NN residual", value=False)

        nn_fit_csv = st.file_uploader("Train NN CSV (Trycolors format)", type=["csv"], key="nn_fit_csv")
        train_nn_clicked = st.button("Train residual NN")

        def train_residual_nn(
            rows: List[Dict[str, Any]],
            engine: str,
            ks_eps: float,
            yn_n: float,
            calibrator: Optional[Calibrator],
            epochs: int = 500,
            lr: float = 1e-3,
        ) -> "ResidualNN":
            model = ResidualNN()
            opt = optim.Adam(model.parameters(), lr=float(lr))

            samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
            all_bases = set()
            for r in rows:
                # Base prediction (engine + calibrator), then learn delta to GT
                pred_lin = predict_lin(
                    r["bases"], r["weights"],
                    engine=engine,
                    ks_eps=ks_eps,
                    yn_n=yn_n,
                    calibrator=calibrator,
                    nn_model=None,
                    nn_scale=0.0,
                )
                x = torch.tensor(pred_lin, dtype=torch.float32)
                y = torch.tensor(hex_to_linear_rgb(r["api_hex"]), dtype=torch.float32)
                samples.append((x, y))
                for b in r["bases"]:
                    all_bases.add(clean_hex(b))

            if not samples:
                raise ValueError("No usable labeled rows to train on.")

            model.train()
            for ep in range(int(epochs)):
                total = 0.0
                for x, y in samples:
                    opt.zero_grad()
                    delta = model(x)
                    loss = ((x + delta - y) ** 2).mean()
                    loss.backward()
                    opt.step()
                    total += float(loss.item())
                # keep it quiet in Streamlit; show occasional status
                if ep in (0, 49, 99, 199, 399) or (ep + 1 == epochs):
                    st.caption(f"NN epoch {ep+1}/{epochs}, mean loss={total/len(samples):.6f}")

            model.eval()
            # Attach training metadata for runtime safety checks.
            cal_hash = None
            try:
                if calibrator is not None:
                    cal_hash = hashlib.sha1(calibrator.to_json().encode('utf-8')).hexdigest()
            except Exception:
                cal_hash = None
            model._meta = {
                'engine': str(engine),
                'ks_eps': float(ks_eps),
                'yn_n': float(yn_n),
                'hybrid_t': float(st.session_state.get('hybrid_t', 0.4)),
                'calibrator_hash': cal_hash,
                'palette_hash': _compute_palette_hash(list(all_bases)),  # NEW: store training palette hash
            }
            return model

        if train_nn_clicked:
            if nn_fit_csv is None:
                st.error("Upload a CSV first.")
            else:
                try:
                    if not isinstance(st.session_state.get("calibrator"), (Calibrator, Calibrator3D)):
                        st.warning("No calibrator loaded/fitted. NN will learn residual on the *uncalibrated* engine output.")
                    text = nn_fit_csv.getvalue().decode("utf-8", errors="ignore")
                    rows = parse_trycolors_csv(text)
                    st.session_state.nn_model = train_residual_nn(
                        rows=rows,
                        engine=engine,
                        ks_eps=float(ks_eps),
                        yn_n=float(yn_n),
                        calibrator=st.session_state.get("calibrator"),
                    )
                    st.success("Residual NN trained.")
                except Exception as e:
                    st.error(f"NN training failed: {e}")

    st.markdown("---")
    swatch_size = st.slider("Swatch size", 100, 600, 220, 10)

calibrator_active: Optional[Any] = st.session_state.get("calibrator") if (isinstance(st.session_state.get("calibrator"), (Calibrator, Calibrator3D)) and st.session_state.get("use_calibrator", False)) else None

# Disable calibrator if the current engine/parameters differ from those used at fit time.
if calibrator_active is not None:
    mism: List[str] = []
    if getattr(calibrator_active, "engine", None) and str(calibrator_active.engine) != str(engine):
        mism.append(f"engine={calibrator_active.engine} (fitted) vs {engine} (current)")
    km_like_engines = ("KM", "YN-KM", "Hybrid (KM ⊕ Linear)", "Hybrid (YN-KM ⊕ Linear)")
    yn_like_engines = ("YN-KM", "Hybrid (YN-KM ⊕ Linear)")
    # ks_eps matters for any engine that uses KM/YN-KM under the hood; yn_n matters for YN-KM variants.
    if str(engine) in km_like_engines:
        if abs(float(getattr(calibrator_active, "ks_eps", ks_eps)) - float(ks_eps)) > 1e-12:
            mism.append(f"KS_EPS={getattr(calibrator_active, 'ks_eps', None)} (fitted) vs {ks_eps} (current)")
    if str(engine) in yn_like_engines:
        if abs(float(getattr(calibrator_active, "yn_n", yn_n)) - float(yn_n)) > 1e-12:
            mism.append(f"Yule–Nielsen n={getattr(calibrator_active, 'yn_n', None)} (fitted) vs {yn_n} (current)")
    if str(engine).startswith("Hybrid"):
        if abs(float(getattr(calibrator_active, "hybrid_t", hybrid_t)) - float(hybrid_t)) > 1e-12:
            mism.append(f"Hybrid t={getattr(calibrator_active, 'hybrid_t', None)} (fitted) vs {hybrid_t} (current)")
    if mism:
        st.sidebar.warning(
            "Calibrator settings mismatch → calibrator disabled. "
            "Re-fit the calibrator with the current engine/params. "
            f"Details: {' | '.join(mism)}"
        )
        calibrator_active = None

# NN residual: only activate if toggled on and (optionally) compatible with current settings.
nn_active = st.session_state.get("nn_model") if (TORCH_AVAILABLE and use_nn and isinstance(st.session_state.get("nn_model"), ResidualNN)) else None
if nn_active is not None and hasattr(nn_active, "_meta"):
    meta = getattr(nn_active, "_meta")
    if not isinstance(meta, dict):
        meta = {}

    mism2: List[str] = []

    if str(meta.get("engine", "")) and str(meta.get("engine")) != str(engine):
        mism2.append(f"engine={meta.get('engine')} (trained) vs {engine} (current)")

    km_like_engines = ("KM", "YN-KM", "Hybrid (KM ⊕ Linear)", "Hybrid (YN-KM ⊕ Linear)")
    yn_like_engines = ("YN-KM", "Hybrid (YN-KM ⊕ Linear)")

    # Only compare params that matter for the current engine.
    if str(engine) in km_like_engines:
        if abs(float(meta.get("ks_eps", ks_eps)) - float(ks_eps)) > 1e-12:
            mism2.append("KS_EPS")
    if str(engine) in yn_like_engines:
        if abs(float(meta.get("yn_n", yn_n)) - float(yn_n)) > 1e-12:
            mism2.append("Yule–Nielsen n")
    if str(engine).startswith("Hybrid"):
        if abs(float(meta.get("hybrid_t", hybrid_t)) - float(hybrid_t)) > 1e-12:
            mism2.append("Hybrid blend t")

    if mism2:
        st.sidebar.warning(
            "NN residual settings mismatch → NN disabled. "
            "Retrain the NN with the current engine/params (and calibrator, if used). "
            f"Details: {' | '.join(mism2)}"
        )
        nn_active = None

# ---------------- Inverse mode: Get Mix Recipe ----------------

st.markdown("<hr style='margin:6px 0;border-top:1px solid #ffe0b2;'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin:0 0 4px 0;font-size:14px;color:#e65100;'>Get Mix Recipe</h3>", unsafe_allow_html=True)

inv_left, inv_right = st.columns([1, 1])

with inv_left:
    target_hex = st.color_picker("Target color", "#8B5CF6", key="target_hex")

    st.markdown("**Search constraints**")
    max_colors = st.slider("Max colors in recipe", min_value=1, max_value=6, value=2, step=1)  # increased to 6

    mode_label = st.selectbox(
        "Inverse search mode",
        ["Exhaustive parts grid", "Dominant + micro-tints (Trycolors-like)"],
        index=0,
        help=(
            "Exhaustive mode enumerates all integer part allocations summing to MaxParts "
            "(can be slow for large MaxParts). "
            "Dominant+micro-tints assumes one dominant color and small tint amounts; "
            "this enables high precision (e.g., MaxParts=200 → 0.5% steps) without "
            "enumerating all compositions."
        ),
    )

    if mode_label.startswith("Dominant"):
        search_mode = "dominant_micro"
        max_parts = st.slider("Max parts (precision) — fine grid", min_value=20, max_value=400, value=200, step=10, key="max_parts_micro")
        tint_cap = st.slider("Tint cap (max parts per tint color)", min_value=1, max_value=40, value=10, step=1, key="tint_cap")
    else:
        search_mode = "exhaustive"
        tint_cap = 0
        max_parts = st.slider("Max parts (precision)", min_value=2, max_value=40, value=12, step=1, key="max_parts_exhaustive")

    prefilter_top_n = st.slider(
        "Prefilter palette to top-N nearest colors (0 = off)",
        min_value=0,
        max_value=60,
        value=12,
        step=1,
        help=(
            "Speed knob. Even when enabled, the app keeps a few palette extremes "
            "(white/black + approximate cyan/magenta/yellow corners) to reduce the risk "
            "of excluding useful tint pigments."
        ),
    )
    top_k = st.slider("Keep top-K solutions", min_value=1, max_value=20, value=5, step=1)

    # Loss metric is now hard-coded to "de00" (perceptual). No user control in client build.
    loss_metric = "de00"

with inv_right:

    st.subheader("Candidate palette")

    # ---- Palette presets (matching Trycolors / client's Netlify dropdown) ----
    PALETTE_PRESETS: Dict[str, List[Dict[str, str]]] = {
        "Cadmium (calibrator default)": [
            {"name": "Cadmium Yellow Light", "hex": "#FEE100"},
            {"name": "Cadmium Red Light",    "hex": "#DE290C"},
            {"name": "Ultramarine Blue",     "hex": "#19123F"},
            {"name": "Titanium White",       "hex": "#F7F5F1"},
            {"name": "Mars Black",           "hex": "#232222"},
        ],
        "Default Artist Palette (20 colors)": [
            {"name": "Cadmium Yellow",       "hex": "#FDE100"},
            {"name": "Yellow Ochre",         "hex": "#D2A51E"},
            {"name": "Orange",               "hex": "#E87511"},
            {"name": "Burnt Sienna",         "hex": "#8F3E1F"},
            {"name": "Cadmium Red Medium",   "hex": "#B01B0F"},
            {"name": "Cadmium Red Light",    "hex": "#DE290C"},
            {"name": "Alizarin Crimson",     "hex": "#731524"},
            {"name": "Dioxazine Purple",     "hex": "#3C1361"},
            {"name": "Ultramarine Blue",     "hex": "#1B3F8B"},
            {"name": "Cerulean Blue",        "hex": "#2C73A8"},
            {"name": "Viridian Green",       "hex": "#00796B"},
            {"name": "Chromium Oxide Green", "hex": "#4D7D4D"},
            {"name": "Sap Green",            "hex": "#507D2A"},
            {"name": "Raw Umber",            "hex": "#7B5B3A"},
            {"name": "Burnt Umber",          "hex": "#6E3B21"},
            {"name": "Raw Sienna",           "hex": "#D2691E"},
            {"name": "Naples Yellow",        "hex": "#E8C85A"},
            {"name": "Titanium White",       "hex": "#F7F5F1"},
            {"name": "Ivory Black",          "hex": "#1A1A1A"},
            {"name": "Payne's Gray",         "hex": "#536878"},
        ],
        "Common Color Names & Values (13)": [
            {"name": "Red",     "hex": "#FF0000"}, {"name": "Orange",  "hex": "#FF7F50"},
            {"name": "Yellow",  "hex": "#FFD700"}, {"name": "Green",   "hex": "#2E8B57"},
            {"name": "Blue",    "hex": "#0047AB"}, {"name": "Purple",  "hex": "#6A0DAD"},
            {"name": "Pink",    "hex": "#FF00AB"}, {"name": "Brown",   "hex": "#7B3F00"},
            {"name": "Cyan",    "hex": "#00CED1"}, {"name": "Lime",    "hex": "#32CD32"},
            {"name": "Magenta", "hex": "#C71585"}, {"name": "White",   "hex": "#FFFFFF"},
            {"name": "Black",   "hex": "#000000"},
        ],
        "Gamblin Artist Oil (subset)": [
            {"name": "Cadmium Yellow Light",  "hex": "#FDE100"},
            {"name": "Cadmium Yellow Medium", "hex": "#F5B800"},
            {"name": "Cadmium Orange",        "hex": "#E87511"},
            {"name": "Cadmium Red Light",     "hex": "#DE290C"},
            {"name": "Cadmium Red Medium",    "hex": "#B01B0F"},
            {"name": "Alizarin Crimson",      "hex": "#731524"},
            {"name": "Dioxazine Purple",      "hex": "#3C1361"},
            {"name": "Ultramarine Blue",      "hex": "#1B3F8B"},
            {"name": "Cerulean Blue",         "hex": "#2C73A8"},
            {"name": "Phthalo Blue",          "hex": "#000F89"},
            {"name": "Phthalo Green",         "hex": "#123524"},
            {"name": "Viridian",              "hex": "#00796B"},
            {"name": "Chromium Oxide Green",  "hex": "#4D7D4D"},
            {"name": "Yellow Ochre",          "hex": "#D2A51E"},
            {"name": "Raw Sienna",            "hex": "#D2691E"},
            {"name": "Burnt Sienna",          "hex": "#8F3E1F"},
            {"name": "Raw Umber",             "hex": "#7B5B3A"},
            {"name": "Burnt Umber",           "hex": "#6E3B21"},
            {"name": "Titanium White",        "hex": "#F7F5F1"},
            {"name": "Ivory Black",           "hex": "#1A1A1A"},
        ],
    }

    preset_choice = st.selectbox(
        "Load palette preset",
        ["-- Keep current --"] + list(PALETTE_PRESETS.keys()),
        key="palette_preset_sel",
    )
    if st.button("Load preset", key="load_preset_btn"):
        if preset_choice in PALETTE_PRESETS:
            st.session_state.palette_rows = [dict(r) for r in PALETTE_PRESETS[preset_choice]]
            st.rerun()

    def _init_palette() -> None:
        if "palette_rows" not in st.session_state:
            # Default palette aligned to the bundled Trycolors recipe dataset (Cadmium palette).
            st.session_state.palette_rows = [
                {"name": "Titanium White",       "hex": "#F7F5F1"},
                {"name": "Mars Black",           "hex": "#232222"},
                {"name": "Cadmium Yellow Light", "hex": "#FEE100"},
                {"name": "Cadmium Red Light",    "hex": "#DE290C"},
                {"name": "Ultramarine Blue",     "hex": "#19123F"},
            ]

    def _add_palette_row() -> None:
        st.session_state.palette_rows.append({"name": "", "hex": "#808080"})

    def _remove_palette_row(idx: Optional[int] = None) -> None:
        if not st.session_state.palette_rows:
            return
        if idx is None:
            st.session_state.palette_rows.pop()
        else:
            if 0 <= idx < len(st.session_state.palette_rows):
                st.session_state.palette_rows.pop(idx)

    _init_palette()

    pal_controls = st.columns([1, 1, 1])
    with pal_controls[0]:
        if st.button("Add color"):
            _add_palette_row()
    with pal_controls[1]:
        if st.button("Remove last"):
            _remove_palette_row()
    with pal_controls[2]:
        if st.button("Reset palette"):
            # Default palette aligned to the bundled Trycolors recipe dataset (Cadmium palette).
            st.session_state.palette_rows = [
                {"name": "Titanium White",       "hex": "#F7F5F1"},
                {"name": "Mars Black",           "hex": "#232222"},
                {"name": "Cadmium Yellow Light", "hex": "#FEE100"},
                {"name": "Cadmium Red Light",    "hex": "#DE290C"},
                {"name": "Ultramarine Blue",     "hex": "#19123F"},
            ]

    # Import palette
    st.markdown("**Import palette**")
    pal_file = st.file_uploader("Palette CSV (hex per line, or csv with hex column)", type=["csv", "txt"], key="palette_file")
    pal_trycolors = st.file_uploader("Trycolors CSV (extract unique base_hexes)", type=["csv"], key="palette_trycolors_file")
    import_cols = st.columns([1, 1])
    with import_cols[0]:
        if st.button("Load palette file"):
            if pal_file is None:
                st.error("Upload a palette file first.")
            else:
                try:
                    text = pal_file.getvalue().decode("utf-8", errors="ignore")
                    pal_list = parse_palette_csv(text)
                    if not pal_list:
                        raise ValueError("No valid hex colors found in palette file.")
                    st.session_state.palette_rows = [{"name": "", "hex": h} for h in pal_list]
                    st.success(f"Loaded palette: {len(pal_list)} colors.")
                except Exception as e:
                    st.error(f"Failed to load palette: {e}")
    with import_cols[1]:
        if st.button("Extract from Trycolors CSV"):
            if pal_trycolors is None:
                st.error("Upload a Trycolors CSV first.")
            else:
                try:
                    text = pal_trycolors.getvalue().decode("utf-8", errors="ignore")
                    pal_list = extract_palette_from_trycolors_csv(text)
                    if not pal_list:
                        raise ValueError("No base_hexes found in Trycolors CSV.")
                    st.session_state.palette_rows = [{"name": "", "hex": h} for h in pal_list]
                    st.success(f"Extracted palette: {len(pal_list)} colors.")
                except Exception as e:
                    st.error(f"Failed to extract palette: {e}")

    # Palette editor
    st.markdown(f"**Palette editor ({len(st.session_state.palette_rows)} colors)**")
    st.caption("Tip: Names are optional but help maintain name↔hex parity in the recipe output.")
    to_del: List[int] = []
    for i, row in enumerate(st.session_state.palette_rows):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            row["name"] = st.text_input(f"Name {i+1}", value=str(row.get("name", "")), key=f"pal_name_{i}")
        with c2:
            row["hex"] = st.color_picker(f"Hex {i+1}", row.get("hex", "#808080"), key=f"pal_hex_{i}")
        with c3:
            if st.button("✖", key=f"pal_del_{i}"):
                to_del.append(i)
    for idx in sorted(to_del, reverse=True):
        _remove_palette_row(idx)

run_inv = st.button("Run Get Mix Recipe (inverse)")

if run_inv:
    try:
        palette_hexes = [clean_hex(r.get("hex", "")) for r in st.session_state.palette_rows]
        palette_hexes = [h for h in palette_hexes if h is not None]  # type: ignore[assignment]
        if not palette_hexes:
            raise ValueError("Palette is empty.")

        # Compute palette hash for provenance and gating
        current_palette_hash = _compute_palette_hash(palette_hexes)

        # Calibrator is palette-specific. If the current palette is not within the calibrator's training palette,
        # disable it for this inverse run to avoid severe degradation.
        _cal_for_inv = calibrator_active
        if _cal_for_inv is not None:
            cal_pal_hash = getattr(_cal_for_inv, "palette_hash", None)
            if cal_pal_hash and cal_pal_hash != current_palette_hash:
                st.warning(
                    f"Loaded calibrator was trained on a different palette (hash mismatch). "
                    f"Calibrator disabled for this inverse run."
                )
                _cal_for_inv = None
            elif not cal_pal_hash:
                # No palette hash stored; we can only warn if the palette isn't a subset.
                if not calibrator_allows_hexes(palette_hexes):
                    st.warning(
                        "Loaded calibrator does not have palette hash, and current palette contains colors "
                        "outside the calibrator's training palette. Calibrator disabled for this inverse run."
                    )
                    _cal_for_inv = None

        # NN residual: check palette hash match if available
        _nn_for_inv = nn_active
        if _nn_for_inv is not None and hasattr(_nn_for_inv, "_meta"):
            nn_pal_hash = _nn_for_inv._meta.get("palette_hash")
            if nn_pal_hash and nn_pal_hash != current_palette_hash:
                st.warning(
                    f"NN residual was trained on a different palette (hash mismatch). "
                    f"NN disabled for this inverse run."
                )
                _nn_for_inv = None

        with st.spinner("Searching..."):
            sols = inverse_mix_recipe(
                target_hex=target_hex,
                palette_hexes=palette_hexes,
                engine=engine,
                ks_eps=float(ks_eps),
                yn_n=float(yn_n),
                calibrator=_cal_for_inv,
                nn_model=_nn_for_inv,
                nn_scale=float(nn_scale),
                max_colors=int(max_colors),
                max_parts=int(max_parts),
                prefilter_top_n=int(prefilter_top_n),
                top_k=int(top_k),
                search_mode=str(search_mode),
                tint_cap=int(tint_cap),
                loss_metric=loss_metric,
            )

        # Perceptual metrics (Trycolors-style): ΔE00 and Match%
        for _s in sols:
            try:
                _de = delta_e00(hex_to_lab(_s["target_hex"]), hex_to_lab(_s["pred_hex"]))
                _s["dE00"] = float(_de)
                _s["match_pct"] = float(match_percent_from_de00(_de))
            except Exception:
                _s["dE00"] = float("nan")
                _s["match_pct"] = float("nan")

        if not sols:
            st.warning("No solutions found (unexpected).")
        else:
            best = sols[0]

            # Log the run
            try:
                cal_hash = None
                if _cal_for_inv:
                    cal_hash = hashlib.sha1(_cal_for_inv.to_json().encode('utf-8')).hexdigest()
                nn_hash = None
                if _nn_for_inv and hasattr(_nn_for_inv, "_meta"):
                    nn_hash = _nn_for_inv._meta.get("palette_hash")
                _log_recipe_run(
                    target_hex=target_hex,
                    palette_hexes=palette_hexes,
                    palette_hash=current_palette_hash,
                    engine=engine,
                    ks_eps=float(ks_eps),
                    yn_n=float(yn_n),
                    hybrid_t=float(hybrid_t),
                    max_colors=int(max_colors),
                    max_parts=int(max_parts),
                    prefilter_top_n=int(prefilter_top_n),
                    top_k=int(top_k),
                    search_mode=str(search_mode),
                    tint_cap=int(tint_cap),
                    loss_metric=loss_metric,
                    calibrator_hash=cal_hash,
                    nn_hash=nn_hash,
                    best_solution=best,
                )
            except Exception as e:
                st.warning(f"Could not log recipe run: {e}")

            dE_best = float(best.get("dE00", float("nan")))
            match_best = float(best.get("match_pct", float("nan")))

            # ---- Trycolors-style recipe display ----

            # Build recipe rows with name lookup
            recipe_rows = []
            for h, p in zip(best["bases"], best["parts"]):
                _nm = ""
                try:
                    for _row in st.session_state.get("palette_rows", []):
                        if clean_hex(_row.get("hex", "")) == clean_hex(h):
                            _nm = str(_row.get("name", "")).strip()
                            break
                except Exception:
                    _nm = ""
                pct = 100.0 * float(p) / float(best["total_parts"])
                recipe_rows.append({"name": _nm, "hex": h, "pct": pct, "parts": int(p)})

            # Prominent match percentage (large, colored)
            if np.isfinite(match_best):
                match_color = "#22c55e" if match_best >= 90 else ("#eab308" if match_best >= 75 else "#ef4444")
                st.markdown(
                    f'<div style="text-align:right;font-size:2.2em;font-weight:800;color:{match_color};'
                    f'line-height:1.2">{match_best:.1f}%<br>'
                    f'<span style="font-size:0.35em;font-weight:400;color:#888">match</span></div>',
                    unsafe_allow_html=True,
                )

            # Target → Result swatches
            colA, colB = st.columns([1, 1])
            with colA:
                show_swatch("Target", best["target_hex"], size=int(swatch_size))
            with colB:
                show_swatch("Result", best["pred_hex"], size=int(swatch_size))

            # Recipe ingredients — Trycolors style: Name #HEX   Percentage%
            st.markdown("#### Recipe")
            for row in recipe_rows:
                _display_name = row["name"] if row["name"] else "—"
                _hex_upper = row["hex"].upper()
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:6px 12px;'
                    f'margin:4px 0;border-radius:6px;background:#1e1e2e;border:1px solid #333">'
                    f'<div style="width:32px;height:32px;border-radius:5px;background:{row["hex"]};'
                    f'border:1px solid #555;flex-shrink:0"></div>'
                    f'<div style="flex:1"><strong>{_display_name}</strong>'
                    f'<br><span style="color:#888;font-size:0.85em">{_hex_upper}</span></div>'
                    f'<div style="font-size:1.3em;font-weight:700;color:#22c55e">{row["pct"]:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Error row
            if np.isfinite(dE_best):
                err_pct = max(0.0, 100.0 - match_best) if np.isfinite(match_best) else float("nan")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:6px 12px;'
                    f'margin:4px 0;border-radius:6px;background:#2a1a1a;border:1px solid #633">'
                    f'<div style="flex:1;color:#ef4444"><strong>Error</strong> (ΔE: {dE_best:.2f})</div>'
                    f'<div style="font-size:1.3em;font-weight:700;color:#ef4444">{err_pct:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Copy Recipe — plain text in Trycolors format
            recipe_text_lines = []
            for row in recipe_rows:
                _n = row["name"] if row["name"] else row["hex"]
                recipe_text_lines.append(f"{_n} {row['hex'].upper()} {row['pct']:.1f}%")
            if np.isfinite(dE_best):
                recipe_text_lines.append(f"Error {err_pct:.1f}%")
            recipe_text = "\n".join(recipe_text_lines)

            st.code(recipe_text, language=None)
            st.caption("Copy the recipe text above — same format as Trycolors.")

            # ---- Detailed technical view (collapsed) ----
            with st.expander("Technical details"):
                st.write(f"RMSE (linear RGB): `{best['rmse']:.6f}`")
                if np.isfinite(dE_best):
                    st.write(f"ΔE00 (CIEDE2000): `{dE_best:.2f}`  |  Match: `{match_best:.1f}%`")
                st.code(
                    "bases=" + str(best["bases"]) +
                    "\nparts=" + str(best["parts"]) +
                    "\nweights=" + str(best["weights"]) +
                    f"\npred={best['pred_hex']}  rmse={best['rmse']:.6f}"
                )
                if PANDAS_AVAILABLE:
                    st.dataframe(pd.DataFrame(recipe_rows))

            # ---- Top solutions table ----
            st.markdown("### All solutions")
            if PANDAS_AVAILABLE:
                _top_rows = []
                for i, s in enumerate(sols):
                    # Build Trycolors-style recipe string for each solution
                    _r_parts = []
                    for _h, _p in zip(s["bases"], s["parts"]):
                        _n2 = ""
                        try:
                            for _row in st.session_state.get("palette_rows", []):
                                if clean_hex(_row.get("hex", "")) == clean_hex(_h):
                                    _n2 = str(_row.get("name", "")).strip()
                                    break
                        except Exception:
                            _n2 = ""
                        _pct2 = 100.0 * float(_p) / float(s["total_parts"])
                        _label2 = _n2 if _n2 else _h
                        _r_parts.append(f"{_label2} {_pct2:.1f}%")
                    _top_rows.append({
                        "rank": i + 1,
                        "match_%": round(float(s.get("match_pct", float("nan"))), 1),
                        "ΔE00": round(float(s.get("dE00", float("nan"))), 2),
                        "pred_hex": s["pred_hex"],
                        "recipe": " + ".join(_r_parts),
                    })
                st.dataframe(pd.DataFrame(_top_rows))
            else:
                st.write(sols)

    except Exception as e:
        st.error(f"Inverse search failed: {e}")


# ---------------- Batch evaluation ----------------
st.markdown("<hr style='margin:6px 0;border-top:1px solid #ffe0b2;'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin:0 0 4px 0;font-size:13px;color:#bf360c;'>Batch Evaluation</h3>", unsafe_allow_html=True)
st.caption("Upload a Trycolors-style CSV and compute mean RMSE for the currently selected engine + optional calibrator + optional NN residual.")

eval_csv = st.file_uploader("Evaluation CSV", type=["csv"], key="eval_csv")
run_eval = st.button("Run evaluation")

if run_eval:
    if eval_csv is None:
        st.error("Upload a CSV first.")
    else:
        try:
            text = eval_csv.getvalue().decode("utf-8", errors="ignore")
            rows = parse_trycolors_csv(text)
            # Calibrator is palette-specific. Disable it if evaluation CSV uses colors outside its training palette.
            _cal_for_eval = calibrator_active
            try:
                _all_hex: List[str] = []
                for _r in rows:
                    _all_hex.extend(list(_r.get("bases", [])))
                if _cal_for_eval is not None:
                    cal_pal_hash = getattr(_cal_for_eval, "palette_hash", None)
                    if cal_pal_hash:
                        current_eval_hash = _compute_palette_hash(_all_hex)
                        if cal_pal_hash != current_eval_hash:
                            st.warning(
                                "Evaluation CSV uses a palette different from the calibrator's training palette. "
                                "Calibrator disabled for this evaluation run."
                            )
                            _cal_for_eval = None
                    elif not calibrator_allows_hexes(_all_hex):
                        st.warning(
                            "Evaluation CSV contains colors outside the calibrator's training palette. "
                            "Calibrator disabled for this evaluation run."
                        )
                        _cal_for_eval = None
            except Exception:
                _cal_for_eval = calibrator_active

            # NN residual: check palette hash if available
            _nn_for_eval = nn_active
            if _nn_for_eval is not None and hasattr(_nn_for_eval, "_meta"):
                nn_pal_hash = _nn_for_eval._meta.get("palette_hash")
                if nn_pal_hash:
                    current_eval_hash = _compute_palette_hash(_all_hex)
                    if nn_pal_hash != current_eval_hash:
                        st.warning(
                            "Evaluation CSV uses a palette different from the NN's training palette. "
                            "NN disabled for this evaluation run."
                        )
                        _nn_for_eval = None

            df, summary, group = batch_evaluate(
                rows=rows,
                engine=engine,
                ks_eps=float(ks_eps),
                yn_n=float(yn_n),
                calibrator=_cal_for_eval,
                nn_model=_nn_for_eval,
                nn_scale=float(nn_scale),
            )
            st.markdown("### Summary")
            st.json(summary)

            st.markdown("### Per-group RMSE")
            if PANDAS_AVAILABLE:
                st.dataframe(group)
            else:
                st.write(group)

            st.markdown("### Per-row results")
            if PANDAS_AVAILABLE:
                st.dataframe(df)
                st.download_button(
                    "Download results CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="trycolors_eval_results.csv",
                    mime="text/csv",

                )
            else:
                st.write(df)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")


# ---------------- Forward mode: interactive mix ----------------
st.markdown("<hr style='margin:6px 0;border-top:1px solid #ffe0b2;'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin:0 0 4px 0;font-size:14px;color:#e65100;'>Forward Mix</h3>", unsafe_allow_html=True)


# Quick reset (clears Streamlit widget state for this section)
c_reset = st.columns([1, 5])
with c_reset[0]:
    reset_fwd = st.button("Reset forward editor", key="fwd_reset_editor")
if reset_fwd:
    # Clear per-row widget keys so pasted recipes / programmatic updates actually propagate.
    for i in range(8):
        for k in (f"fwd_hex_{i}", f"fwd_w_{i}"):
            if k in st.session_state:
                del st.session_state[k]
    st.session_state.fwd_rows = [
        {"hex": "#FF2B2B", "weight": 50.0},
        {"hex": "#FFFFFF", "weight": 50.0},
    ]
    st.session_state.fwd_n_bases = 2
    st.session_state.fwd_weight_units = "Percent"
    st.session_state.fwd_paste_text = ""
    st.success("Forward editor reset.")

# ---- Helpers (local to UI) ----
def _parse_forward_recipe_text(text: str) -> List[Dict[str, Any]]:
    """
    Robustly parse a pasted Trycolors recipe (or similar).

    Supported formats:

    (A) Single-line (preferred):
        - "Ultramarine Blue #19123F 94.6%"
        - "#F7F5F1, 1.7"
        - "#DE290C : 3.1"

    (B) Multi-line blocks (common when copying from UI cards):
        Ultramarine Blue
        #19123F
        94.6%

    Parsing rules:
    - Any line containing a HEX code (#RRGGBB or RRGGBB) is treated as a "hex line".
    - Weight is preferably parsed from the same line (percent token or last numeric token).
    - If a hex line contains no weight, the next line containing a numeric/percent token is used as the weight.
    - If we can't find a weight, default weight = 1.0.

    Returns rows: [{"hex": "#RRGGBB", "weight": float, "raw_line": str}, ...]
    """
    rows: List[Dict[str, Any]] = []
    if not text:
        return rows

    pending_hex: Optional[str] = None
    pending_raw: str = ""

    def _extract_weight(s: str) -> Optional[float]:
        # Prefer explicit percent tokens, e.g., "94.6%"
        m_pct = re.search(r"([-+]?(?:\d+\.\d+|\d+))\s*%", s)
        if m_pct:
            try:
                return float(m_pct.group(1))
            except Exception:
                return None

        # Otherwise use the last numeric token that is NOT embedded in an alphanumeric code (e.g., avoid "PB29").
        nums = re.findall(r"(?<![A-Za-z0-9])([-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)", s)
        if nums:
            try:
                return float(nums[-1])
            except Exception:
                return None
        return None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m_hex = re.search(r"#?[0-9A-Fa-f]{6}", line)
        if m_hex:
            hx = clean_hex(m_hex.group(0))
            if not hx:
                continue

            # Try to extract a weight from the SAME line (remove the hex first to avoid confusing parsing).
            rest = (line[:m_hex.start()] + " " + line[m_hex.end():]).strip()
            w = _extract_weight(rest)
            if w is None:
                # No weight found on this line → remember the hex and wait for a weight line.
                pending_hex = hx
                pending_raw = raw
            else:
                rows.append({"hex": hx, "weight": float(w), "raw_line": raw})
                pending_hex = None
                pending_raw = ""
        else:
            # No hex on this line; if we have a pending hex, try to interpret this as a weight line.
            if pending_hex is not None:
                w = _extract_weight(line)
                if w is not None:
                    rows.append({"hex": pending_hex, "weight": float(w), "raw_line": (pending_raw + "\n" + raw).strip()})
                    pending_hex = None
                    pending_raw = ""

    # If the last hex had no weight, default it.
    if pending_hex is not None:
        rows.append({"hex": pending_hex, "weight": 1.0, "raw_line": pending_raw})

    return rows



def _ensure_fwd_rows_len(n: int) -> None:
    """Ensure st.session_state.fwd_rows exists and has length n."""
    if "fwd_rows" not in st.session_state or not isinstance(st.session_state.fwd_rows, list):
        st.session_state.fwd_rows = [{"hex": "#FF2B2B", "weight": 50.0}, {"hex": "#FFFFFF", "weight": 50.0}]

    # Clean up / normalize
    cleaned: List[Dict[str, Any]] = []
    for r in st.session_state.fwd_rows:
        hx = clean_hex(r.get("hex", "")) or "#808080"
        try:
            w = float(r.get("weight", 1.0))
        except Exception:
            w = 1.0
        cleaned.append({"hex": hx, "weight": float(w)})

    st.session_state.fwd_rows = cleaned

    # Resize
    if len(st.session_state.fwd_rows) < n:
        for _ in range(n - len(st.session_state.fwd_rows)):
            st.session_state.fwd_rows.append({"hex": "#808080", "weight": 1.0})
    elif len(st.session_state.fwd_rows) > n:
        st.session_state.fwd_rows = st.session_state.fwd_rows[:n]


# ---- Paste Trycolors recipe ----
with st.expander("Paste Trycolors recipe (optional)", expanded=True):
    st.caption(
        "Paste the recipe shown in Trycolors (lines containing HEX + percentage). "
        "Example:\n"
        "• Ultramarine Blue #19123F 94.6%\n"
        "• Cadmium Red Light #DE290C 3.1%\n"
        "• Titanium White #F7F5F1 1.7%\n"
        "• Cadmium Yellow Light #FEE100 0.5%"
    )
    paste_text = st.text_area("Recipe text", value="", height=140, key="fwd_paste_text")
    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        paste_is_percent = st.checkbox("Treat weights as %", value=True, key="fwd_paste_is_percent")
    with cB:
        apply_paste = st.button("Load pasted recipe", key="fwd_apply_paste")
    with cC:
        st.caption("Tip: Any line with a hex code will be parsed; the last number on the line is used as the weight.")

    if apply_paste:
        parsed = _parse_forward_recipe_text(paste_text)
        if not parsed:
            st.warning("No usable lines found (need at least one '#RRGGBB' per line).")
        else:
            # Clamp to a reasonable max for UI
            n_new = max(1, min(8, len(parsed)))
            st.session_state.fwd_n_bases = n_new
            st.session_state.fwd_rows = [{"hex": r["hex"], "weight": float(r["weight"])} for r in parsed[:n_new]]
            # IMPORTANT: Update widget state so the editor fields refresh to the pasted values.
            for i, r0 in enumerate(st.session_state.fwd_rows):
                st.session_state[f"fwd_hex_{i}"] = r0["hex"]
                st.session_state[f"fwd_w_{i}"] = float(r0["weight"])
            if paste_is_percent:
                st.session_state.fwd_weight_units = "Percent"
            st.success(f"Loaded {n_new} bases from pasted recipe.")


# ---- Manual editor ----
if "fwd_n_bases" not in st.session_state:
    st.session_state.fwd_n_bases = 4
if "fwd_weight_units" not in st.session_state:
    st.session_state.fwd_weight_units = "Percent"

n_bases = st.slider("Number of bases", min_value=1, max_value=8, value=int(st.session_state.fwd_n_bases), step=1, key="fwd_n_bases")

weight_units = st.selectbox(
    "Weight units",
    ["Percent", "Fraction", "Raw (will normalize)"],
    index=["Percent", "Fraction", "Raw (will normalize)"].index(st.session_state.fwd_weight_units) if st.session_state.fwd_weight_units in ["Percent", "Fraction", "Raw (will normalize)"] else 0,
    key="fwd_weight_units",
)

_ensure_fwd_rows_len(int(n_bases))

st.markdown("**Bases + weights**")
rows = st.session_state.fwd_rows

# Display editor as rows
for i in range(int(n_bases)):
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        rows[i]["hex"] = st.color_picker(f"Base {i+1}", rows[i]["hex"], key=f"fwd_hex_{i}")
    with c2:
        # Use a single stable widget config (Streamlit can be sensitive to changing widget params across reruns).
        rows[i]["weight"] = st.number_input(
            f"Weight {i+1}",
            value=float(rows[i]["weight"]),
            min_value=0.0,
            max_value=1000.0,
            step=0.1,
            format="%.6f",
            key=f"fwd_w_{i}",
        )
        if weight_units == "Percent":
            st.caption("Units: percent (0–100).")
        elif weight_units == "Fraction":
            st.caption("Units: fraction (0–1).")
        else:
            st.caption("Units: raw (will normalize).")
    with c3:
        st.image(make_swatch(rows[i]["hex"], size=64), use_column_width=False)

# Optional target (for RMSE)
st.markdown("**Optional target (for RMSE diagnostics)**")
fwd_target_hex = st.color_picker("Target (compare forward output against this)", "#1E2448", key="fwd_target_hex")
show_rmse = st.checkbox("Compute RMSE vs target", value=True, key="fwd_show_rmse")

# Quick warning: KM/YN-KM + tiny KS_EPS + near-zero base channels can collapse a channel (common for yellows/reds with B≈0).
if str(engine) in ("KM", "YN-KM"):
    try:
        _bases_tmp = [clean_hex(r["hex"]) for r in rows[: int(n_bases)]]
        _bases_tmp = [b for b in _bases_tmp if b is not None]  # type: ignore[assignment]
        _bases_lin_tmp = [hex_to_linear_rgb(b) for b in _bases_tmp]
        _min_ch = min(min(rgb) for rgb in _bases_lin_tmp) if _bases_lin_tmp else 1.0
        if _min_ch < 5e-5 and float(ks_eps) < 1e-3:
            st.warning(
                "One or more selected bases has a channel near 0 (in linear RGB). "
                "With a very small KS_EPS, KM/YN-KM can drive that channel to ~0 in the mixture even for tiny tint weights. "
                "Try increasing KS_EPS (e.g., 1e-3 → 2e-2) and/or switching to KM (or YN-KM with n≈1) before relying on the calibrator/NN."
            )
    except Exception:
        pass

with st.expander("Auto-tune KM parameters (diagnostic for this ONE recipe)", expanded=False):
    st.caption(
        "This runs a small grid search to find KS_EPS (and optionally n for YN-KM) that minimizes RMSE vs the target "
        "for the currently entered recipe. This is meant for diagnosis; it can overfit a single case."
    )
    if not show_rmse:
        st.info("Enable 'Compute RMSE vs target' to use auto-tune.")
    else:
        tgt0 = clean_hex(fwd_target_hex)
        if tgt0 is None:
            st.warning("Invalid target hex.")
        elif str(engine) == "Linear":
            st.info("Auto-tune applies to KM/YN-KM only (Linear has no KS_EPS/n).")
        else:
            if st.button("Run auto-tune", key="fwd_autotune"):
                try:
                    # Prepare recipe (same logic as forward mix)
                    _bases = [clean_hex(r["hex"]) for r in rows[: int(n_bases)]]
                    if any(b is None for b in _bases):
                        raise ValueError("One or more bases is invalid.")
                    _bases_hex = [b for b in _bases if b is not None]  # type: ignore[assignment]

                    _w_in = [float(r["weight"]) for r in rows[: int(n_bases)]]
                    if weight_units == "Percent":
                        _w = [x / 100.0 for x in _w_in]
                    else:
                        _w = _w_in

                    _tgt_lin = hex_to_linear_rgb(tgt0)
                    _bases_lin = [hex_to_linear_rgb(h) for h in _bases_hex]
                    _w_norm = normalize_weights(_w)

                    # Evaluate without calibrator/NN to isolate the base engine behavior.
                    eps_grid = np.logspace(-6, math.log10(0.2), 24)
                    if str(engine) == "YN-KM":
                        n_grid = np.linspace(0.8, 2.5, 18)
                    else:
                        n_grid = np.array([float(yn_n)])

                    best = None
                    for epsv in eps_grid:
                        if str(engine) == "KM":
                            pred = mix_km(_bases_lin, _w_norm, float(epsv))
                            e = rmse_lin(pred, _tgt_lin)
                            cand = (e, float(epsv), float("nan"), linear_rgb_to_hex(pred))
                            if (best is None) or (cand[0] < best[0]):
                                best = cand
                        else:
                            for nv in n_grid:
                                pred = mix_ynkm(_bases_lin, _w_norm, float(nv), float(epsv))
                                e = rmse_lin(pred, _tgt_lin)
                                cand = (e, float(epsv), float(nv), linear_rgb_to_hex(pred))
                                if (best is None) or (cand[0] < best[0]):
                                    best = cand

                    if best is None:
                        st.warning("Auto-tune failed to find any candidate (unexpected).")
                    else:
                        e_best, eps_best, n_best, hex_best = best
                        st.markdown("**Best parameters found (base engine only):**")
                        if str(engine) == "KM":
                            st.write(f"KS_EPS ≈ `{eps_best:.6g}` → pred `{hex_best}` with RMSE `{e_best:.6f}`")
                        else:
                            st.write(f"KS_EPS ≈ `{eps_best:.6g}`, n ≈ `{n_best:.3f}` → pred `{hex_best}` with RMSE `{e_best:.6f}`")
                        st.caption("Tip: If the best KS_EPS is orders of magnitude larger than your current value, your model is likely hitting the 'near-zero channel' pathology.")
                except Exception as e:
                    st.error(f"Auto-tune failed: {e}")

mix_clicked = st.button("Mix (forward)")

if mix_clicked:
    try:
        bases = [clean_hex(r["hex"]) for r in rows[: int(n_bases)]]
        if any(b is None for b in bases):
            raise ValueError("One or more bases is invalid.")
        bases_hex = [b for b in bases if b is not None]  # type: ignore[assignment]

        w_in = [float(r["weight"]) for r in rows[: int(n_bases)]]
        if weight_units == "Percent":
            w = [x / 100.0 for x in w_in]
        else:
            w = w_in

        # Merge duplicates (same HEX entered multiple times) to keep the recipe well-formed.
        try:
            bases_hex, w = merge_duplicate_bases(bases_hex, w)
        except Exception:
            pass


        # Stage 0: engine-only prediction (no calibrator / no NN)
        engine_lin = predict_lin(
            bases_hex=bases_hex,
            weights=w,
            engine=engine,
            ks_eps=float(ks_eps),
            yn_n=float(yn_n),
            calibrator=None,
            nn_model=None,
            nn_scale=0.0,
        )
        engine_hex = linear_rgb_to_hex(engine_lin)

        # Stage 1: apply calibrator (if active + compatible with the bases)
        _cal_for_fwd = calibrator_active
        if _cal_for_fwd is not None:
            cal_pal_hash = getattr(_cal_for_fwd, "palette_hash", None)
            if cal_pal_hash:
                current_pal_hash = _compute_palette_hash(bases_hex)
                if cal_pal_hash != current_pal_hash:
                    st.warning(
                        "Loaded calibrator was trained on a different palette. "
                        "Calibrator disabled for this forward mix."
                    )
                    _cal_for_fwd = None
            elif not calibrator_allows_hexes(bases_hex):
                st.warning(
                    "Loaded calibrator does not have palette hash, and current bases contain colors "
                    "outside the calibrator's training palette. Calibrator disabled for this forward mix."
                )
                _cal_for_fwd = None

        if _cal_for_fwd is not None:
            cal_lin = _cal_for_fwd.apply(engine_lin)
            cal_hex = linear_rgb_to_hex(cal_lin)   # FIX: assign cal_hex
        else:
            cal_lin = engine_lin
            cal_hex = engine_hex   # if calibrator off, cal_hex is same as engine_hex

        # Stage 2: apply NN residual (if active)
        if nn_active is not None:
            nn_active.eval()
            with torch.no_grad():
                x = torch.tensor(cal_lin, dtype=torch.float32)
                delta = nn_active(x).cpu().numpy().astype(float)
            nn_lin = (
                clamp01(float(cal_lin[0] + float(nn_scale) * float(delta[0]))),
                clamp01(float(cal_lin[1] + float(nn_scale) * float(delta[1]))),
                clamp01(float(cal_lin[2] + float(nn_scale) * float(delta[2]))),
            )
        else:
            nn_lin = cal_lin
        nn_hex = linear_rgb_to_hex(nn_lin)

        # Final output = after all active stages
        out_lin = nn_lin
        out_hex = nn_hex

        # Calibrator clamp diagnostic (only meaningful when calibrator is active)
        clamp_info: Optional[str] = None
        if _cal_for_fwd is not None:
            eps_c = 1e-6
            raw_x = [math.log(max(float(engine_lin[ch]), eps_c)) for ch in range(3)]
            flags: List[str] = []
            names = ["R", "G", "B"]
            for ch in range(3):
                mn = float(_cal_for_fwd.x_mins[ch]) if getattr(_cal_for_fwd, "x_mins", None) else float("nan")
                mx = float(_cal_for_fwd.x_maxs[ch]) if getattr(_cal_for_fwd, "x_maxs", None) else float("nan")
                out_of = (raw_x[ch] < mn) or (raw_x[ch] > mx)
                flags.append(f"{names[ch]}: x={raw_x[ch]:.3f} range=[{mn:.3f},{mx:.3f}] {'CLAMP' if out_of else 'in-range'}")
            clamp_info = " | ".join(flags)

        st.markdown("**Forward results (stage-by-stage)**")
        colS0, colS1, colS2 = st.columns(3)
        with colS0:
            show_swatch("Engine only", engine_hex, size=int(swatch_size))
        with colS1:
            lab = "+ Calibrator" if _cal_for_fwd is not None else "Calibrator (OFF)"
            show_swatch(lab, cal_hex, size=int(swatch_size))
        with colS2:
            lab2 = "+ NN residual" if nn_active is not None else "NN residual (OFF)"
            show_swatch(lab2, nn_hex, size=int(swatch_size))

        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            show_swatch("Final result", out_hex, size=int(swatch_size))
        with colB:
            if show_rmse:
                tgt = clean_hex(fwd_target_hex)
                if tgt is not None:
                    st.markdown("**Diagnostics**")
                    st.caption(f"Engine: {engine} | KS_EPS={float(ks_eps):g} | n={float(yn_n):g}")
                    st.caption(
                        f"Calibrator: {'ON' if _cal_for_fwd is not None else 'OFF'} | "
                        f"NN residual: {'ON' if nn_active is not None else 'OFF'} (scale={float(nn_scale):g})"
                    )
                    if clamp_info:
                        st.caption("Calibrator clamp status: " + clamp_info)
                    st.write(f"Target: `{tgt}`")
                    tgt_lin = hex_to_linear_rgb(tgt)

                    rmse_engine = rmse_lin(engine_lin, tgt_lin)
                    rmse_cal = rmse_lin(cal_lin, tgt_lin)
                    rmse_final = rmse_lin(out_lin, tgt_lin)

                    st.write(f"RMSE (linear RGB) — engine: `{rmse_engine:.6f}`")
                    st.write(f"RMSE (linear RGB) — +calibrator: `{rmse_cal:.6f}`")
                    st.write(f"RMSE (linear RGB) — final: `{rmse_final:.6f}`")

                    # Perceptual metrics (ΔE00 / Match%) for parity with Trycolors UI
                    try:
                        de_engine = float(delta_e00(hex_to_lab(engine_hex), hex_to_lab(tgt)))
                        de_cal = float(delta_e00(hex_to_lab(cal_hex), hex_to_lab(tgt)))
                        de_final = float(delta_e00(hex_to_lab(out_hex), hex_to_lab(tgt)))
                        st.write(f"ΔE00 — engine: `{de_engine:.2f}`  (Match `{match_percent_from_de00(de_engine):.1f}%`)")
                        st.write(f"ΔE00 — +calibrator: `{de_cal:.2f}`  (Match `{match_percent_from_de00(de_cal):.1f}%`)")
                        st.write(f"ΔE00 — final: `{de_final:.2f}`  (Match `{match_percent_from_de00(de_final):.1f}%`)")
                    except Exception:
                        pass

                    rmse_post = rmse_hex(out_hex, tgt)
                    st.caption(f"After HEX rounding (final): {rmse_post:.6f}")
                else:
                    st.warning("Invalid target hex.")
        with colC:
            # Display both nominal weights (user inputs) and effective weights (after strength model).
            w_norm_in = normalize_weights(w)

            # Defensive: merge duplicates again for display (should already be merged).
            try:
                bases_disp, w_norm_in = merge_duplicate_bases(bases_hex, w_norm_in)
            except Exception:
                bases_disp = list(bases_hex)

            # Compute the *effective* weights used internally after the strength model.
            w_eff = list(w_norm_in)
            strength_on = bool(st.session_state.get("strength_enable", False))
            gamma = float(st.session_state.get("strength_gamma", 1.0))
            if strength_on and gamma > 0.0:
                try:
                    a = (
                        float(st.session_state.get("strength_a0", -1.01798)),
                        float(st.session_state.get("strength_a1", -3.69844)),
                        float(st.session_state.get("strength_a2",  2.37642)),
                    )
                    bb = float(st.session_state.get("strength_b", 3.71610))
                    bases_lin_disp = [hex_to_linear_rgb(h) for h in bases_disp]
                    w_eff = apply_strength_weights(bases_lin_disp, list(w_norm_in), a=a, b=bb, gamma=gamma)
                except Exception:
                    w_eff = list(w_norm_in)

            st.markdown("**Recipe weights (nominal vs effective)**")
            recipe_rows = []
            for hx, wi_in, wi_eff in zip(bases_disp, w_norm_in, w_eff):
                recipe_rows.append({
                    "base_hex": hx,
                    "w_in": float(wi_in),
                    "percent_in": 100.0 * float(wi_in),
                    "w_eff": float(wi_eff),
                    "percent_eff": 100.0 * float(wi_eff),
                })

            if PANDAS_AVAILABLE:
                st.dataframe(pd.DataFrame(recipe_rows))
            else:
                st.write(recipe_rows)

            if strength_on and gamma > 0.0:
                try:
                    delta_max = max(abs(float(a) - float(b)) for a, b in zip(w_eff, w_norm_in))
                    st.caption(f"Strength weighting is ON (γ={gamma:.2f}). Max |w_eff − w_in| = {delta_max:.3f}.")
                except Exception:
                    pass

            st.code(
                "engine=" + str(engine) +
                "\nbases=" + str(bases_disp) +
                "\nweights_in=" + str(w_in) +
                "\nweights_norm_in=" + str(w_norm_in) +
                "\nweights_eff=" + str(w_eff) +
                "\nengine=" + engine_hex +
                "\ncalibrated=" + cal_hex +
                "\nfinal=" + out_hex
            )


    except Exception as e:
        st.error(f"Forward mix failed: {e}")