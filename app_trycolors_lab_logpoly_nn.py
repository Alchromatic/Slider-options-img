# app_trycolors_lab_logpoly_nn_final.py
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
st.set_page_config(
    page_title="Trycolors Reverse Engineering Lab", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Note: For iframe embedding, run Streamlit with these flags:
# streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false
# Or create .streamlit/config.toml with:
# [server]
# enableCORS = false
# enableXsrfProtection = false

# ============================================================
# 1) Color utilities
# ============================================================

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

def srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def linear_to_srgb(c: float) -> float:
    c = clamp01(c)
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055

def clean_hex(h: str) -> Optional[str]:
    if not h:
        return None
    s = str(h).strip().upper()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4:
        s = "#" + "".join(ch * 2 for ch in s[1:])
    if len(s) != 7:
        return None
    if any(ch not in "0123456789ABCDEF" for ch in s[1:]):
        return None
    return s

def hex_to_rgb255(h: str) -> Tuple[int, int, int]:
    h = clean_hex(h)
    if h is None:
        raise ValueError("Invalid hex")
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def rgb255_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, int(r))),
        max(0, min(255, int(g))),
        max(0, min(255, int(b))),
    )

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

def normalize_weights(w: Sequence[float]) -> List[float]:
    s = float(sum(w))
    if s <= 0:
        raise ValueError("Weights sum to 0")
    return [float(x) / s for x in w]

def rmse_hex(h1: str, h2: str) -> float:
    a = hex_to_linear_rgb(h1)
    b = hex_to_linear_rgb(h2)
    return math.sqrt(((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) / 3.0)

def make_swatch(hex_color: str, size: int = 180) -> Image.Image:
    r, g, b = hex_to_rgb255(hex_color)
    return Image.new("RGB", (size, size), (r, g, b))


# ============================================================
# 2) Mixing engines
# ============================================================

def mix_linear_rgb(bases, w):
    """Simple linear RGB mixing (additive light) - good default for screen colors"""
    w = normalize_weights(w)
    return tuple(clamp01(sum(b[i] * wi for b, wi in zip(bases, w))) for i in range(3))

def _ks_from_R(R, eps):
    """Convert reflectance to K/S ratio (Kubelka-Munk)"""
    R = max(R, eps)
    return (1 - R) ** 2 / (2 * R)

def _R_from_ks(KS):
    """Convert K/S ratio back to reflectance"""
    return max(0.0, (1 + KS) - math.sqrt(KS*KS + 2*KS))

def mix_km(bases, w, eps):
    """Kubelka-Munk subtractive mixing for paint/pigment simulation"""
    w = normalize_weights(w)
    out = []
    # Use a higher minimum reflectance floor (5%) to prevent extreme darkening
    # when mixing colors with zero channel values
    min_reflectance = max(eps, 0.05)
    for ch in range(3):
        KS = sum(_ks_from_R(max(b[ch], min_reflectance), min_reflectance) * wi for b, wi in zip(bases, w))
        out.append(clamp01(_R_from_ks(KS)))
    return tuple(out)

def mix_ynkm(bases, w, n, eps):
    if n <= 0:
        return mix_km(bases, w, eps)
    fwd = lambda R: clamp01(R) ** (1/n)
    inv = lambda Rp: clamp01(Rp) ** n
    bases_yn = [(fwd(r), fwd(g), fwd(b)) for r, g, b in bases]
    mix = mix_km(bases_yn, w, eps)
    return tuple(inv(x) for x in mix)


# ============================================================
# 3) Log-poly calibrator
# ============================================================

@dataclass
class Calibrator:
    ks_eps: float
    yn_n: float
    degree: int
    coeffs: List[List[float]]

    def apply(self, rgb):
        eps = 1e-6
        out = []
        for ch in range(3):
            x = math.log(max(rgb[ch], eps))
            y = np.polyval(self.coeffs[ch], x)
            out.append(clamp01(math.exp(y)))
        return tuple(out)

    def to_json(self) -> str:
        return json.dumps({
            "ks_eps": self.ks_eps,
            "yn_n": self.yn_n,
            "degree": self.degree,
            "coeffs": self.coeffs
        }, indent=2)

    @staticmethod
    def from_json(text: str) -> "Calibrator":
        d = json.loads(text)
        return Calibrator(
            ks_eps=d["ks_eps"],
            yn_n=d["yn_n"],
            degree=d["degree"],
            coeffs=d["coeffs"]
        )


# ============================================================
# 3b) Residual NN (optional)
# ============================================================

if TORCH_AVAILABLE:
    class ResidualNN(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3),
            )

        def forward(self, x):
            return self.net(x)


# ============================================================
# 4) CSV parsing
# ============================================================

def parse_csv(text: str) -> List[Dict[str, Any]]:
    import csv
    rdr = csv.DictReader(io.StringIO(text))
    rows = []
    for r in rdr:
        bases = [clean_hex(x) for x in str(r.get("base_hexes","")).split(";")]
        weights = [float(x) for x in str(r.get("weights","")).split(";")]
        api = clean_hex(r.get("api_hex",""))
        if not api or any(b is None for b in bases):
            continue
        rows.append({
            "group_id": r.get("group_id",""),
            "bases": bases,
            "weights": weights,
            "api_hex": api
        })
    return rows


# ============================================================
# 5) Prediction + batch evaluation
# ============================================================

def predict_one(bases, weights, ks_eps, yn_n, calibrator=None, nn_model=None, use_linear=False):
    bases_lin = [hex_to_linear_rgb(h) for h in bases]
    w = normalize_weights(weights)

    if use_linear:
        # Simple linear RGB mixing (additive)
        lin = mix_linear_rgb(bases_lin, w)
    else:
        # YN-KM subtractive mixing
        lin = mix_ynkm(bases_lin, w, yn_n, ks_eps)

    if calibrator and not use_linear:
        lin = calibrator.apply(lin)

    if nn_model is not None and not use_linear:
        with torch.no_grad():
            delta = nn_model(torch.tensor(lin, dtype=torch.float32)).numpy()
        lin = tuple(clamp01(lin[i] + 0.25 * delta[i]) for i in range(3))

    return linear_rgb_to_hex(lin)


def batch_evaluate(rows, ks_eps, yn_n, calibrator, nn_model=None):
    results = []
    for r in rows:
        pred = predict_one(
            r["bases"], r["weights"],
            ks_eps, yn_n,
            calibrator, nn_model
        )
        results.append({
            "group_id": r["group_id"],
            "api_hex": r["api_hex"],
            "pred_hex": pred,
            "rmse": rmse_hex(pred, r["api_hex"])
        })

    if PANDAS_AVAILABLE:
        df = pd.DataFrame(results)
        summary = {"mean_rmse": float(df["rmse"].mean())}
        group = df.groupby("group_id")["rmse"].mean().reset_index()
        return df, summary, group

    mean = sum(r["rmse"] for r in results) / len(results)
    return results, {"mean_rmse": mean}, {}


def train_residual_nn(rows, calibrator, ks_eps, yn_n, epochs=500, lr=1e-3):
    model = ResidualNN()
    opt = optim.Adam(model.parameters(), lr=lr)

    samples = []
    for r in rows:
        pred = predict_one(r["bases"], r["weights"], ks_eps, yn_n, calibrator)
        x = torch.tensor(hex_to_linear_rgb(pred), dtype=torch.float32)
        y = torch.tensor(hex_to_linear_rgb(r["api_hex"]), dtype=torch.float32)
        samples.append((x, y))

    for ep in range(epochs):
        total = 0.0
        for x, y in samples:
            opt.zero_grad()
            delta = model(x)
            loss = ((x + delta - y) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item()
        if ep % 100 == 0:
            print(f"Epoch {ep}: loss={total/len(samples):.6f}")

    return model


# ============================================================
# 6) UI
# ============================================================

st.title("Trycolors Reverse Engineering Lab")
st.caption("YN-KM + permanent log-poly calibration (+ optional NN residual)")

with st.sidebar:
    st.subheader("Mixing Parameters")
    
    # Mixing method selector
    mix_method = st.selectbox(
        "Mixing Method",
        ["YN-KM (Subtractive/Paint)", "Linear RGB (Additive/Light)"],
        index=0,
        help="YN-KM simulates paint mixing (subtractive). Linear RGB is simple additive mixing."
    )
    use_linear_mix = "Linear" in mix_method
    
    ks_eps = st.number_input("KS_EPS", value=1e-6, format="%.9f", disabled=use_linear_mix)
    yn_n = st.number_input("YN n", value=1.5, disabled=use_linear_mix)

    st.markdown("---")
    st.subheader("Log-poly calibrator")

    cal_deg = st.number_input("Degree", 1, 7, 3)
    fit_csv = st.file_uploader("Fit CSV", type=["csv"])
    if st.button("Fit log-poly") and fit_csv:
        rows = parse_csv(fit_csv.read().decode())
        P, T = [], []
        for r in rows:
            pred = predict_one(r["bases"], r["weights"], ks_eps, yn_n)
            P.append(hex_to_linear_rgb(pred))
            T.append(hex_to_linear_rgb(r["api_hex"]))

        P = np.array(P)
        T = np.array(T)
        coeffs = []
        eps = 1e-6
        for ch in range(3):
            X = np.log(np.clip(P[:,ch], eps, 1))
            Y = np.log(np.clip(T[:,ch], eps, 1))
            coeffs.append(np.polyfit(X, Y, cal_deg).tolist())

        st.session_state.calibrator = Calibrator(
            ks_eps=ks_eps,
            yn_n=yn_n,
            degree=cal_deg,
            coeffs=coeffs
        )
        st.success("Log-poly calibrator fitted")

    if "calibrator" in st.session_state:
        st.download_button(
            "Download calibrator JSON",
            st.session_state.calibrator.to_json(),
            "logpoly_calibrator.json",
            "application/json"
        )

    if TORCH_AVAILABLE:
        st.markdown("---")
        st.subheader("Residual NN")
        if st.button("Train residual NN") and fit_csv:
            rows = parse_csv(fit_csv.read().decode())
            st.session_state.nn_model = train_residual_nn(
                rows,
                st.session_state.calibrator,
                ks_eps,
                yn_n
            )
            st.success("Residual NN trained")

        use_nn = st.checkbox("Enable NN residual", value=False)
    else:
        use_nn = False


st.markdown("---")
st.subheader("Batch evaluation")

eval_csv = st.file_uploader("Evaluation CSV", type=["csv"])
if st.button("Run evaluation") and eval_csv:
    rows = parse_csv(eval_csv.read().decode())
    df, summary, group = batch_evaluate(
        rows,
        ks_eps,
        yn_n,
        st.session_state.get("calibrator"),
        st.session_state.get("nn_model") if use_nn else None
    )
    st.json(summary)
    if PANDAS_AVAILABLE:
        st.dataframe(group)
        st.dataframe(df)


st.markdown("---")
st.subheader("Interactive mix")

# CSV Upload for populating dropdown
mix_csv = st.file_uploader("Load mix presets (CSV)", type=["csv"], key="mix_presets_csv")
if mix_csv:
    csv_content = mix_csv.read().decode()
    st.session_state.mix_presets = parse_csv(csv_content)

# Dropdown to select from CSV presets
preset_options = ["-- Manual Input --"]
if "mix_presets" in st.session_state and st.session_state.mix_presets:
    for i, row in enumerate(st.session_state.mix_presets):
        bases_str = " + ".join(row["bases"])
        weights_str = ":".join([f"{w:.2f}" for w in row["weights"]])
        preset_options.append(f"{row['group_id']} | {bases_str} ({weights_str}) → {row['api_hex']}")

# Initialize active_preset tracking
if "active_preset_idx" not in st.session_state:
    st.session_state.active_preset_idx = 0

selected_preset = st.selectbox(
    "Select preset from CSV", 
    preset_options, 
    key="preset_selector"
)

# Determine current preset data
current_preset = None
if selected_preset != "-- Manual Input --" and "mix_presets" in st.session_state:
    idx = preset_options.index(selected_preset) - 1
    if 0 <= idx < len(st.session_state.mix_presets):
        current_preset = st.session_state.mix_presets[idx]
        # Track if preset changed to force widget updates
        if st.session_state.active_preset_idx != idx + 1:
            st.session_state.active_preset_idx = idx + 1
            st.rerun()
else:
    if st.session_state.active_preset_idx != 0:
        st.session_state.active_preset_idx = 0

# Set values based on preset or defaults
if current_preset:
    preset_bases = current_preset["bases"]
    preset_weights = current_preset["weights"]
    preset_target = current_preset["api_hex"]
    preset_num_bases = len(preset_bases)
else:
    preset_bases = ["#FF2B2B", "#FFFFFF"]
    preset_weights = [0.8, 0.2]
    preset_target = "#FF9C9C"
    preset_num_bases = 2

# Number of base colors selector
num_bases = st.number_input(
    "Number of base colors", 
    min_value=2, 
    max_value=6, 
    value=preset_num_bases
)

# Dynamic base color pickers and weight sliders
col1, col2 = st.columns(2)

bases = []
weights = []

with col1:
    st.markdown("**Base Colors**")
    for i in range(num_bases):
        if i < len(preset_bases):
            default_color = preset_bases[i]
        else:
            default_color = "#FFFFFF"
        # Use unique key with preset index to force refresh on preset change
        color = st.color_picker(
            f"Base {chr(65+i)}", 
            default_color, 
            key=f"base_{i}_p{st.session_state.active_preset_idx}"
        )
        bases.append(color)

with col2:
    st.markdown("**Weights**")
    for i in range(num_bases):
        if i < len(preset_weights):
            default_weight = preset_weights[i]
        else:
            default_weight = 0.5
        weight = st.slider(
            f"w{chr(65+i)}", 
            0.0, 1.0, 
            default_weight, 
            key=f"weight_{i}_p{st.session_state.active_preset_idx}"
        )
        weights.append(weight)

# Target color picker for comparison
st.markdown("---")
st.markdown("**Target Color (for comparison)**")
target_col1, target_col2 = st.columns([1, 3])
with target_col1:
    target_color = st.color_picker(
        "Target", 
        preset_target, 
        key=f"target_color_p{st.session_state.active_preset_idx}"
    )
with target_col2:
    st.markdown(f"Target hex: `{target_color}`")

# Show current mix inputs for debugging (collapsible)
with st.expander("Debug: Current inputs"):
    st.write(f"Bases: {bases}")
    st.write(f"Weights: {weights}")
    st.write(f"Target: {target_color}")
    st.write(f"Sum of weights: {sum(weights)}")
    st.write(f"Mixing method: {'Linear RGB' if use_linear_mix else 'YN-KM'}")

# Mix button and results
if st.button("Mix", key="mix_button"):
    # Validate weights
    if sum(weights) <= 0:
        st.error("Weights must sum to a positive value!")
    else:
        try:
            out = predict_one(
                bases, weights,
                ks_eps, yn_n,
                st.session_state.get("calibrator"),
                st.session_state.get("nn_model") if use_nn else None,
                use_linear=use_linear_mix
            )
            
            # Calculate RMSE between prediction and target
            error = rmse_hex(out, target_color)
            
            # Display results side by side
            st.markdown("### Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.markdown("**Predicted**")
                st.image(make_swatch(out, 120), caption=out)
            
            with res_col2:
                st.markdown("**Target**")
                st.image(make_swatch(target_color, 120), caption=target_color)
            
            with res_col3:
                st.markdown("**Comparison**")
                # Create a side-by-side comparison swatch
                comparison = Image.new("RGB", (120, 120))
                pred_half = make_swatch(out, 60)
                target_half = make_swatch(target_color, 60)
                comparison.paste(pred_half, (0, 30))
                comparison.paste(target_half, (60, 30))
                st.image(comparison, caption="Pred | Target")

            st.markdown("---")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("RMSE (Linear RGB)", f"{error:.6f}")
            with metric_col2:
                # Calculate a rough match percentage (inverse of error)
                match_pct = max(0, (1 - error) * 100)
                st.metric("Approx Match %", f"{match_pct:.1f}%")
        except Exception as e:
            st.error(f"Error during mixing: {str(e)}")