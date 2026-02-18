# app_trycolors_lab_logpoly_nn_recipe.py
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from itertools import combinations
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
st.set_page_config(page_title="Trycolors Reverse Engineering Lab", layout="wide")


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
    st.image(make_swatch(hex_color, size=size), use_container_width=False)


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


# ============================================================
# 3) Log-poly calibrator
# ============================================================

@dataclass
class Calibrator:
    ks_eps: float
    yn_n: float
    degree: int
    coeffs: List[List[float]]

    def apply(self, rgb_lin: Tuple[float, float, float]) -> Tuple[float, float, float]:
        eps = 1e-6
        out: List[float] = []
        for ch in range(3):
            x = math.log(max(float(rgb_lin[ch]), eps))
            y = float(np.polyval(self.coeffs[ch], x))
            out.append(clamp01(math.exp(y)))
        return (out[0], out[1], out[2])

    def to_json(self) -> str:
        return json.dumps({
            "kind": "log_poly",
            "ks_eps": float(self.ks_eps),
            "yn_n": float(self.yn_n),
            "degree": int(self.degree),
            "coeffs": self.coeffs,
        }, indent=2)

    @staticmethod
    def from_json(text: str) -> "Calibrator":
        d = json.loads(text)
        if d.get("kind") not in (None, "log_poly"):
            raise ValueError(f"Unsupported calibrator kind: {d.get('kind')}")
        return Calibrator(
            ks_eps=float(d["ks_eps"]),
            yn_n=float(d["yn_n"]),
            degree=int(d["degree"]),
            coeffs=d["coeffs"],
        )


# ============================================================
# 3b) Residual NN (optional)
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
    w = normalize_weights(weights)

    if engine == "Linear":
        lin = mix_linear_rgb(bases_lin, w)
    elif engine == "KM":
        lin = mix_km(bases_lin, w, float(ks_eps))
    else:  # "YN-KM"
        lin = mix_ynkm(bases_lin, w, float(yn_n), float(ks_eps))

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
        results.append({
            "row": i,
            "group_id": r.get("group_id", ""),
            "base_hexes": ";".join(r["bases"]),
            "weights": ";".join([f"{x:g}" for x in r["weights"]]),
            "api_hex": r["api_hex"],
            "pred_hex": pred,
            "rmse": rmse_hex(pred, r["api_hex"]),
        })

    if PANDAS_AVAILABLE:
        df = pd.DataFrame(results)
        summary = {"mean_rmse": float(df["rmse"].mean()) if len(df) else float("nan")}
        group = df.groupby("group_id")["rmse"].mean().reset_index() if len(df) else pd.DataFrame(columns=["group_id", "rmse"])
        return df, summary, group

    # fallback
    mean = float(sum(r["rmse"] for r in results) / len(results)) if results else float("nan")
    return results, {"mean_rmse": mean}, {}


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
) -> List[Dict[str, Any]]:
    """
    Discrete brute-force search over:
      - subsets of palette colors up to max_colors
      - integer part allocations that sum to max_parts

    Returns a list of best solutions (sorted by rmse asc).
    """
    tgt = clean_hex(target_hex)
    if tgt is None:
        raise ValueError("Invalid target hex.")
    target_lin = hex_to_linear_rgb(tgt)

    pal = [clean_hex(h) for h in palette_hexes]
    pal = [h for h in pal if h is not None]  # type: ignore[assignment]
    if not pal:
        raise ValueError("Palette is empty.")

    max_colors = int(max(1, max_colors))
    max_parts = int(max(2, max_parts))
    top_k = int(max(1, top_k))

    # Prefilter palette to keep search reasonable (optional)
    if prefilter_top_n and prefilter_top_n > 0 and prefilter_top_n < len(pal):
        pal_lin = [hex_to_linear_rgb(h) for h in pal]
        dists = [rmse_lin(x, target_lin) for x in pal_lin]
        idxs = list(range(len(pal)))
        idxs.sort(key=lambda i: dists[i])
        pal = [pal[i] for i in idxs[: int(prefilter_top_n)]]

    best: List[Dict[str, Any]] = []

    def consider(sol: Dict[str, Any]) -> None:
        nonlocal best
        best.append(sol)
        best.sort(key=lambda x: x["rmse"])
        if len(best) > top_k:
            best = best[:top_k]

    # Search
    for k in range(1, max_colors + 1):
        for idxs in combinations(range(len(pal)), k):
            bases = [pal[i] for i in idxs]
            for parts in compositions(max_parts, k):
                weights = list(parts)  # normalize inside predict
                pred_lin = predict_lin(
                    bases, weights,
                    engine=engine,
                    ks_eps=float(ks_eps),
                    yn_n=float(yn_n),
                    calibrator=calibrator,
                    nn_model=nn_model,
                    nn_scale=float(nn_scale),
                )
                e = rmse_lin(pred_lin, target_lin)
                # quick prune if we already have better solutions
                if best and len(best) >= top_k and e >= best[-1]["rmse"]:
                    continue
                pred_hex = linear_rgb_to_hex(pred_lin)
                consider({
                    "rmse": float(e),
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

st.title("Trycolors Reverse Engineering Lab")
st.caption("Forward mixer (Linear / KM / YN-KM) + log-poly calibration + optional NN residual + inverse 'Get Mix Recipe' search.")


# ---------------- Sidebar: global params ----------------
with st.sidebar:
    st.header("Engine + params")
    engine = st.selectbox("Engine", ["YN-KM", "KM", "Linear"], index=0)
    ks_eps = st.number_input("KS_EPS (KM / YN-KM)", value=1e-6, min_value=1e-9, max_value=1e-2, step=1e-6, format="%.9f")
    yn_n = st.number_input("Yule–Nielsen n (YN-KM)", value=1.5, min_value=0.0, max_value=10.0, step=0.1)

    st.markdown("---")
    st.header("Log-poly calibrator")

    cal_deg = st.number_input("Degree", min_value=1, max_value=7, value=3, step=1)
    cal_fit_csv = st.file_uploader("Fit CSV (Trycolors format)", type=["csv"], key="cal_fit_csv")
    col1, col2 = st.columns(2)
    with col1:
        fit_clicked = st.button("Fit log-poly", use_container_width=True)
    with col2:
        clear_cal = st.button("Clear", use_container_width=True)

    cal_json_file = st.file_uploader("Load calibrator JSON", type=["json"], key="cal_json")
    if cal_json_file is not None:
        try:
            cal_text = cal_json_file.getvalue().decode("utf-8", errors="ignore")
            st.session_state.calibrator = Calibrator.from_json(cal_text)
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
                coeffs: List[List[float]] = []
                for ch in range(3):
                    X = np.log(np.clip(Pn[:, ch], eps, 1.0))
                    Y = np.log(np.clip(Tn[:, ch], eps, 1.0))
                    c = np.polyfit(X, Y, int(cal_deg))
                    coeffs.append([float(x) for x in c.tolist()])

                st.session_state.calibrator = Calibrator(
                    ks_eps=float(ks_eps),
                    yn_n=float(yn_n),
                    degree=int(cal_deg),
                    coeffs=coeffs,
                )
                st.success("Log-poly calibrator fitted.")

            except Exception as e:
                st.error(f"Fit failed: {e}")

    cal_obj = st.session_state.get("calibrator")
    if isinstance(cal_obj, Calibrator):
        st.download_button(
            "Download calibrator JSON",
            data=cal_obj.to_json().encode("utf-8"),
            file_name="logpoly_calibrator.json",
            mime="application/json",
            use_container_width=True,
        )

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
        train_nn_clicked = st.button("Train residual NN", use_container_width=True)

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
            return model

        if train_nn_clicked:
            if nn_fit_csv is None:
                st.error("Upload a CSV first.")
            else:
                try:
                    if not isinstance(st.session_state.get("calibrator"), Calibrator):
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

calibrator_active: Optional[Calibrator] = st.session_state.get("calibrator") if isinstance(st.session_state.get("calibrator"), Calibrator) else None
nn_active = st.session_state.get("nn_model") if (TORCH_AVAILABLE and use_nn and isinstance(st.session_state.get("nn_model"), ResidualNN)) else None


# ---------------- Inverse mode: Get Mix Recipe ----------------
st.markdown("---")
st.header("Get Mix Recipe (inverse search)")

st.write(
    "This mode matches the flow described in the client video: "
    "you choose a **target color** and a **candidate palette**; the app computes the best discrete mix recipe "
    "(integer parts) under your constraints."
)

inv_left, inv_right = st.columns([1, 1])

with inv_left:
    target_hex = st.color_picker("Target color", "#8B5CF6", key="target_hex")

    st.markdown("**Search constraints**")
    max_colors = st.slider("Max colors in recipe", min_value=1, max_value=4, value=2, step=1)
    max_parts = st.slider("Max parts (precision)", min_value=2, max_value=30, value=12, step=1)
    prefilter_top_n = st.slider("Prefilter palette to top-N nearest colors (0 = off)", min_value=0, max_value=60, value=12, step=1)
    top_k = st.slider("Keep top-K solutions", min_value=1, max_value=20, value=5, step=1)

with inv_right:
    st.subheader("Candidate palette")

    def _init_palette() -> None:
        if "palette_rows" not in st.session_state:
            st.session_state.palette_rows = [
                {"hex": "#FFFFFF"},
                {"hex": "#000000"},
                {"hex": "#FF2B2B"},
                {"hex": "#2B6BFF"},
                {"hex": "#2BFF7A"},
            ]

    def _add_palette_row() -> None:
        st.session_state.palette_rows.append({"hex": "#808080"})

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
        if st.button("Add color", use_container_width=True):
            _add_palette_row()
    with pal_controls[1]:
        if st.button("Remove last", use_container_width=True):
            _remove_palette_row()
    with pal_controls[2]:
        if st.button("Reset palette", use_container_width=True):
            st.session_state.palette_rows = [
                {"hex": "#FFFFFF"},
                {"hex": "#000000"},
                {"hex": "#FF2B2B"},
                {"hex": "#2B6BFF"},
                {"hex": "#2BFF7A"},
            ]

    # Import palette
    st.markdown("**Import palette**")
    pal_file = st.file_uploader("Palette CSV (hex per line, or csv with hex column)", type=["csv", "txt"], key="palette_file")
    pal_trycolors = st.file_uploader("Trycolors CSV (extract unique base_hexes)", type=["csv"], key="palette_trycolors_file")
    import_cols = st.columns([1, 1])
    with import_cols[0]:
        if st.button("Load palette file", use_container_width=True):
            if pal_file is None:
                st.error("Upload a palette file first.")
            else:
                try:
                    text = pal_file.getvalue().decode("utf-8", errors="ignore")
                    pal_list = parse_palette_csv(text)
                    if not pal_list:
                        raise ValueError("No valid hex colors found in palette file.")
                    st.session_state.palette_rows = [{"hex": h} for h in pal_list]
                    st.success(f"Loaded palette: {len(pal_list)} colors.")
                except Exception as e:
                    st.error(f"Failed to load palette: {e}")
    with import_cols[1]:
        if st.button("Extract from Trycolors CSV", use_container_width=True):
            if pal_trycolors is None:
                st.error("Upload a Trycolors CSV first.")
            else:
                try:
                    text = pal_trycolors.getvalue().decode("utf-8", errors="ignore")
                    pal_list = extract_palette_from_trycolors_csv(text)
                    if not pal_list:
                        raise ValueError("No base_hexes found in Trycolors CSV.")
                    st.session_state.palette_rows = [{"hex": h} for h in pal_list]
                    st.success(f"Extracted palette: {len(pal_list)} colors.")
                except Exception as e:
                    st.error(f"Failed to extract palette: {e}")

    # Palette editor
    st.markdown(f"**Palette editor ({len(st.session_state.palette_rows)} colors)**")
    to_del: List[int] = []
    for i, row in enumerate(st.session_state.palette_rows):
        c1, c2 = st.columns([5, 1])
        with c1:
            row["hex"] = st.color_picker(f"Palette {i+1}", row.get("hex", "#808080"), key=f"pal_{i}")
        with c2:
            if st.button("✖", key=f"pal_del_{i}"):
                to_del.append(i)
    for idx in sorted(to_del, reverse=True):
        _remove_palette_row(idx)

run_inv = st.button("Run Get Mix Recipe (inverse)", type="primary", use_container_width=True)

if run_inv:
    try:
        palette_hexes = [clean_hex(r.get("hex", "")) for r in st.session_state.palette_rows]
        palette_hexes = [h for h in palette_hexes if h is not None]  # type: ignore[assignment]
        if not palette_hexes:
            raise ValueError("Palette is empty.")

        with st.spinner("Searching..."):
            sols = inverse_mix_recipe(
                target_hex=target_hex,
                palette_hexes=palette_hexes,
                engine=engine,
                ks_eps=float(ks_eps),
                yn_n=float(yn_n),
                calibrator=calibrator_active,
                nn_model=nn_active,
                nn_scale=float(nn_scale),
                max_colors=int(max_colors),
                max_parts=int(max_parts),
                prefilter_top_n=int(prefilter_top_n),
                top_k=int(top_k),
            )

        if not sols:
            st.warning("No solutions found (unexpected).")
        else:
            best = sols[0]
            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                show_swatch("Target", best["target_hex"], size=int(swatch_size))
            with colB:
                show_swatch("Best match", best["pred_hex"], size=int(swatch_size))
            with colC:
                st.markdown("**Best recipe**")
                st.write(f"RMSE (linear RGB): `{best['rmse']:.6f}`")
                recipe_rows = []
                for h, p in zip(best["bases"], best["parts"]):
                    recipe_rows.append({
                        "base_hex": h,
                        "parts": int(p),
                        "percent": 100.0 * float(p) / float(best["total_parts"]),
                    })
                if PANDAS_AVAILABLE:
                    st.dataframe(pd.DataFrame(recipe_rows), use_container_width=True)
                else:
                    st.write(recipe_rows)

                st.code(
                    "bases=" + str(best["bases"]) +
                    "\nparts=" + str(best["parts"]) +
                    "\nweights=" + str(best["weights"]) +
                    f"\npred={best['pred_hex']}  rmse={best['rmse']:.6f}"
                )

            st.markdown("### Top solutions")
            if PANDAS_AVAILABLE:
                df_s = pd.DataFrame([{
                    "rank": i + 1,
                    "rmse": s["rmse"],
                    "pred_hex": s["pred_hex"],
                    "bases": ";".join(s["bases"]),
                    "parts": ";".join(map(str, s["parts"])),
                } for i, s in enumerate(sols)])
                st.dataframe(df_s, use_container_width=True)
            else:
                st.write(sols)

    except Exception as e:
        st.error(f"Inverse search failed: {e}")


# ---------------- Batch evaluation ----------------
st.markdown("---")
st.header("Batch evaluation vs reference CSV")
st.caption("Upload a Trycolors-style CSV and compute mean RMSE for the currently selected engine + optional calibrator + optional NN residual.")

eval_csv = st.file_uploader("Evaluation CSV", type=["csv"], key="eval_csv")
run_eval = st.button("Run evaluation", use_container_width=True)

if run_eval:
    if eval_csv is None:
        st.error("Upload a CSV first.")
    else:
        try:
            text = eval_csv.getvalue().decode("utf-8", errors="ignore")
            rows = parse_trycolors_csv(text)
            df, summary, group = batch_evaluate(
                rows=rows,
                engine=engine,
                ks_eps=float(ks_eps),
                yn_n=float(yn_n),
                calibrator=calibrator_active,
                nn_model=nn_active,
                nn_scale=float(nn_scale),
            )
            st.markdown("### Summary")
            st.json(summary)

            st.markdown("### Per-group RMSE")
            if PANDAS_AVAILABLE:
                st.dataframe(group, use_container_width=True)
            else:
                st.write(group)

            st.markdown("### Per-row results")
            if PANDAS_AVAILABLE:
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download results CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="trycolors_eval_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.write(df)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")


# ---------------- Forward mode: interactive mix ----------------
st.markdown("---")
st.header("Forward mix (given bases + weights)")

st.write(
    "This is the forward mixer: you provide base colors and weights, and the app computes the mixed result. "
    "For the client-requested workflow (given target + palette → compute weights), use **Get Mix Recipe** above."
)

f1, f2 = st.columns([1, 1])
with f1:
    base_a = st.color_picker("Base A", "#FF2B2B", key="base_a")
    w_a = st.slider("wA", 0.0, 1.0, 0.8, key="w_a")
with f2:
    base_b = st.color_picker("Base B", "#FFFFFF", key="base_b")
    w_b = st.slider("wB", 0.0, 1.0, 0.2, key="w_b")

mix_clicked = st.button("Mix (forward)", type="secondary")

if mix_clicked:
    try:
        out_hex = predict_hex(
            bases_hex=[base_a, base_b],
            weights=[w_a, w_b],
            engine=engine,
            ks_eps=float(ks_eps),
            yn_n=float(yn_n),
            calibrator=calibrator_active,
            nn_model=nn_active,
            nn_scale=float(nn_scale),
        )
        show_swatch("Mixed result", out_hex, size=int(swatch_size))
        st.code(f"engine={engine}\nbases={[base_a, base_b]}\nweights={[w_a, w_b]}\nresult={out_hex}")
    except Exception as e:
        st.error(f"Forward mix failed: {e}")
