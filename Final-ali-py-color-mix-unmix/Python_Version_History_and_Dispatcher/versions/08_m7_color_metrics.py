
from __future__ import annotations
import math
import numpy as np

# ---------- sRGB / CIELAB / OKLab / CIEDE2000 utilities ----------

def clean_hex(h: str) -> str:
    s = str(h).strip().upper()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4:
        s = "#" + "".join(c * 2 for c in s[1:])
    if len(s) != 7:
        raise ValueError(f"Invalid hex: {h}")
    return s

def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(c: float) -> float:
    c = float(np.clip(c, 0.0, 1.0))
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055

def hex_to_linear_rgb(h: str) -> np.ndarray:
    h = clean_hex(h).lstrip("#")
    vals = [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    return np.array([_srgb_to_linear(v) for v in vals], dtype=float)

def linear_rgb_to_hex(rgb: np.ndarray) -> str:
    srgb = [_linear_to_srgb(float(x)) for x in np.clip(rgb, 0, 1)]
    vals = [max(0, min(255, int(round(x * 255)))) for x in srgb]
    return "#{:02X}{:02X}{:02X}".format(*vals)

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=float)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=float)

def linear_rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return _RGB_TO_XYZ @ np.asarray(rgb, dtype=float)

def xyz_to_linear_rgb(xyz: np.ndarray) -> np.ndarray:
    return _XYZ_TO_RGB @ np.asarray(xyz, dtype=float)

def _f_lab(t: float) -> float:
    d = 6 / 29
    return t ** (1/3) if t > d ** 3 else t / (3 * d * d) + 4 / 29

def _finv_lab(f: float) -> float:
    d = 6 / 29
    return f ** 3 if f > d else 3 * d * d * (f - 4 / 29)

def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(xyz, dtype=float) / _D65
    fx, fy, fz = _f_lab(float(x)), _f_lab(float(y)), _f_lab(float(z))
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dtype=float)

def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = [float(x) for x in lab]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    return np.array([_finv_lab(fx), _finv_lab(fy), _finv_lab(fz)], dtype=float) * _D65

def hex_to_lab(h: str) -> np.ndarray:
    return xyz_to_lab(linear_rgb_to_xyz(hex_to_linear_rgb(h)))

def lab_to_hex(lab: np.ndarray) -> str:
    return linear_rgb_to_hex(xyz_to_linear_rgb(lab_to_xyz(lab)))

_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6309787005],
], dtype=float)
_M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
], dtype=float)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)

def linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = _M1 @ np.asarray(rgb, dtype=float)
    return _M2 @ np.cbrt(np.maximum(lms, 0.0))

def oklab_to_linear_rgb(ok: np.ndarray) -> np.ndarray:
    lms_ = _M2_INV @ np.asarray(ok, dtype=float)
    lms = lms_ ** 3
    return _M1_INV @ lms

def hex_to_oklab(h: str) -> np.ndarray:
    return linear_rgb_to_oklab(hex_to_linear_rgb(h))

def oklab_to_hex(ok: np.ndarray) -> str:
    return linear_rgb_to_hex(oklab_to_linear_rgb(ok))

def ciede2000(lab1, lab2) -> float:
    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]
    avg_L = (L1 + L2) / 2
    C1 = math.sqrt(a1*a1 + b1*b1)
    C2 = math.sqrt(a2*a2 + b2*b2)
    avg_C = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.sqrt(a1p*a1p + b1*b1)
    C2p = math.sqrt(a2p*a2p + b2*b2)
    avg_Cp = (C1p + C2p) / 2
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if C1p != 0 else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if C2p != 0 else 0.0
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p <= h1p:
        dhp = h2p - h1p + 360
    else:
        dhp = h2p - h1p - 360
    dHp = 2 * math.sqrt(C1p*C2p) * math.sin(math.radians(dhp/2))
    avg_Lp = (L1 + L2) / 2
    avg_Cpp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        avg_hp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        avg_hp = (h1p + h2p + 360) / 2
    else:
        avg_hp = (h1p + h2p - 360) / 2
    T = 1 - 0.17*math.cos(math.radians(avg_hp-30)) + 0.24*math.cos(math.radians(2*avg_hp)) + 0.32*math.cos(math.radians(3*avg_hp+6)) - 0.20*math.cos(math.radians(4*avg_hp-63))
    d_ro = 30 * math.exp(-((avg_hp - 275) / 25) ** 2)
    R_C = 2 * math.sqrt((avg_Cpp**7) / (avg_Cpp**7 + 25**7))
    S_L = 1 + (0.015 * ((avg_Lp - 50)**2)) / math.sqrt(20 + ((avg_Lp - 50)**2))
    S_C = 1 + 0.045 * avg_Cpp
    S_H = 1 + 0.015 * avg_Cpp * T
    R_T = -math.sin(math.radians(2*d_ro)) * R_C
    return float(math.sqrt((dLp/S_L)**2 + (dCp/S_C)**2 + (dHp/S_H)**2 + R_T*(dCp/S_C)*(dHp/S_H)))

def de00_hex(a: str, b: str) -> float:
    return ciede2000(hex_to_lab(a), hex_to_lab(b))

def match_pct(de: float) -> float:
    return max(0.0, 100.0 - float(de))
