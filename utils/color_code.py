color-algopy
Sanwar Sunny shared this file. Want to do more with it?

    # app_trycolors_approx.py
    ﻿
    from __future__ import annotations
    import io
    import math
    import warnings
    from typing import List, Tuple, Dict, Optional
    ﻿
    import csv
    import random
    ﻿
    import numpy as np
    from PIL import Image
    import streamlit as st
    ﻿
    # ---- Optional PyTorch (for Trycolors-Approx) ----
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
    ﻿
    # =================================================
    # 1. Color utilities
    # =================================================
    ﻿
    KS_EPS: float = 1e-6
    ﻿
    def set_ks_eps(eps: float) -> None:
        global KS_EPS
        KS_EPS = float(eps)
    ﻿
    def clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))
    ﻿
    def srgb_to_linear(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4
    ﻿
    def linear_to_srgb(c: float) -> float:
        c = clamp01(c)
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1 / 2.4)) - 0.055
    ﻿
    def hex_to_rgb255(h: str) -> Tuple[int, int, int]:
        h = h.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) != 6:
            raise ValueError(f"Bad hex: {h}")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    ﻿
    def rgb255_to_hex(r: int, g: int, b: int) -> str:
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return "#{:02X}{:02X}{:02X}".format(r, g, b)
    ﻿
    def hex_to_linear_rgb(h: str) -> Tuple[float, float, float]:
        r8, g8, b8 = hex_to_rgb255(h)
        return (
            srgb_to_linear(r8 / 255.0),
            srgb_to_linear(g8 / 255.0),
            srgb_to_linear(b8 / 255.0),
        )
    ﻿
    def linear_rgb_to_hex(rgb_lin: Tuple[float, float, float]) -> str:
        r_lin, g_lin, b_lin = rgb_lin
        r = int(round(clamp01(linear_to_srgb(r_lin)) * 255))
        g = int(round(clamp01(linear_to_srgb(g_lin)) * 255))
        b = int(round(clamp01(linear_to_srgb(b_lin)) * 255))
        return rgb255_to_hex(r, g, b)
    ﻿
    def normalize_weights(w: List[float], tol: float = 1e-12) -> List[float]:
        s = float(sum(w))
        if s <= tol:
            raise ValueError(f"Weights sum ≈ 0: {w}")
        return [wi / s for wi in w]
    ﻿
    def rmse_hex(pred_hex: str, true_hex: str) -> float:
        Rp, Gp, Bp = hex_to_linear_rgb(pred_hex)
        Rt, Gt, Bt = hex_to_linear_rgb(true_hex)
        return math.sqrt(((Rp-Rt)**2 + (Gp-Gt)**2 + (Bp-Bt)**2)/3.0)
    ﻿
    ﻿
    # ---------- Evaluation helper: batch RMSE vs Trycolors CSV ----------
    def evaluate_model_on_csv(text: str, model_predict):
        """
        text: contents of a CSV file (Trycolors outputs)
        model_predict: function(base_hexes, weights) -> predicted_hex
    ﻿
        Returns (results_list, mean_rmse), where each result dict has:
          - group_id
          - base_hexes
          - weights
          - true_hex
          - pred_hex
          - rmse  (RMSE in linear RGB)
        """
        rdr = csv.DictReader(io.StringIO(text))
        results = []
        rmse_sum = 0.0
        count = 0
    ﻿
        for r in rdr:
            api_hex = (r.get("api_hex") or "").strip()
            if not api_hex:
                continue
    ﻿
            base_hexes = [h.strip() for h in (r.get("base_hexes") or "").split(";") if h.strip()]
            weights = [float(w) for w in (r.get("weights") or "").split(";") if w.strip()]
    ﻿
            # Use your chosen engine (Linear / KM / YN-KM / Trycolors-Approx)
            pred_hex = model_predict(base_hexes, weights)
    ﻿
            e = rmse_hex(pred_hex, api_hex)
            rmse_sum += e
            count += 1
    ﻿
            results.append({
                "group_id": (r.get("group_id") or "").strip(),
                "base_hexes": base_hexes,
                "weights": weights,
                "true_hex": api_hex,
                "pred_hex": pred_hex,
                "rmse": e,
            })
    ﻿
        mean_rmse = rmse_sum / count if count else None
        return results, mean_rmse
    ﻿
    ﻿
    ##############################################################
    ﻿
    def make_swatch(hex_color: str, size: int = 200) -> Image.Image:
        r, g, b = hex_to_rgb255(hex_color)
        img = Image.new("RGB", (size, size), (r, g, b))
        return img
    ﻿
    def show_swatch(label: str, hex_color: str, size: int = 200):
        st.markdown(f"**{label}: {hex_color}**")
        st.image(make_swatch(hex_color, size=size), use_column_width=False)
    ﻿
    ﻿
    # =================================================
    # 2. Mixing engines: Linear / KM / YN-KM
    # =================================================
    ﻿
    def mix_linear_rgb(bases: List[Tuple[float,float,float]], w: List[float]) -> Tuple[float,float,float]:
        w = normalize_weights(w)
        r = sum(b[0]*wi for b, wi in zip(bases, w))
        g = sum(b[1]*wi for b, wi in zip(bases, w))
        b = sum(b[2]*wi for b, wi in zip(bases, w))
        return (clamp01(r), clamp01(g), clamp01(b))
    ﻿
    def _ks_from_R(R: float, eps: Optional[float]=None) -> float:
        if eps is None:
            eps = KS_EPS
        R = max(R, eps)
        return (1 - R) ** 2 / (2 * R)
    ﻿
    def _R_from_ks(KS: float) -> float:
        return max(0.0, (1 + KS) - math.sqrt(KS**2 + 2*KS))
    ﻿
    def mix_kubelka_munk(bases: List[Tuple[float,float,float]], w: List[float]) -> Tuple[float,float,float]:
        w = normalize_weights(w)
        def ch(j: int) -> float:
            KS = 0.0
            for bi, wi in zip(bases, w):
                KS += _ks_from_R(bi[j]) * wi
            return clamp01(_R_from_ks(KS))
        return (ch(0), ch(1), ch(2))
    ﻿
    def mix_kubelka_munk_yn(bases: List[Tuple[float,float,float]], w: List[float], n: float=1.5) -> Tuple[float,float,float]:
        if n <= 0:
            return mix_kubelka_munk(bases, w)
        def yn_fwd(R):  return R ** (1.0 / n)
        def yn_inv(Rp): return clamp01(Rp) ** n
        bases_yn = [(yn_fwd(r), yn_fwd(g), yn_fwd(b)) for (r,g,b) in bases]
        mix_yn = mix_kubelka_munk(bases_yn, w)
        return (yn_inv(mix_yn[0]), yn_inv(mix_yn[1]), yn_inv(mix_yn[2]))
    ﻿
    ﻿
    # =================================================
    # 3. Trycolors-Approx latent model (PyTorch)
    # =================================================
    ﻿
    if TORCH_AVAILABLE:
    ﻿
        class LatentMixer(nn.Module):
            """
            Simple surrogate for Trycolors Algorithm 4:
            RGB -> latent(7) with encoder, strength net for weights,
            latent mix -> RGB via decoder.
            """
            def __init__(self, latent_dim: int = 7, hidden: int = 32):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(3, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, latent_dim),
                )
                self.strength_net = nn.Sequential(
                    nn.Linear(3, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                    nn.Softplus()  # positive strength
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 3),
                    nn.Sigmoid(),  # 0..1 RGB
                )
    ﻿
            def forward(self, rgb: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                """
                rgb: (N,3) linear RGB
                weights: (N,)
                returns: (3,) mixed linear RGB
                """
                # normalize weights then modulate by tint strength
                w = weights / (weights.sum() + 1e-9)
                strength = self.strength_net(rgb).squeeze(-1)  # (N,)
                eff = w * strength
                eff = eff / (eff.sum() + 1e-9)
    ﻿
                z = self.encoder(rgb)                # (N,latent)
                z_mix = (eff.unsqueeze(1) * z).sum(0, keepdim=True)  # (1,latent)
                rgb_mix = self.decoder(z_mix).squeeze(0)             # (3,)
                return rgb_mix
    ﻿
        def train_latent_model(
            rows: List[Dict],
            epochs: int = 200,
            lr: float = 1e-3,
            latent_dim: int = 7,
            hidden: int = 32,
            device: str | None = None,
        ):
            """
            rows: output of parse_csv (only labeled rows will be used)
            returns (model, history_str)
            """
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
    ﻿
            # collect training cases
            cases = []
            for r in rows:
                api_hex = r["api_hex"]
                if not api_hex:
                    continue
                try:
                    bases_lin = [hex_to_linear_rgb(h) for h in r["base_hexes"]]
                    w = [float(x) for x in r["weights"]]
                    w = normalize_weights(w)
                    tgt = hex_to_linear_rgb(api_hex)
                    cases.append((bases_lin, w, tgt))
                except Exception:
                    continue
    ﻿
            if not cases:
                raise ValueError("No labeled rows (with valid api_hex) to train on.")
    ﻿
            model = LatentMixer(latent_dim=latent_dim, hidden=hidden).to(device)
            opt = optim.Adam(model.parameters(), lr=lr)
    ﻿
            history = []
            for ep in range(1, epochs+1):
                random.shuffle(cases)
                total = 0.0
                for bases_lin, w, tgt in cases:
                    rgb = torch.tensor(bases_lin, dtype=torch.float32, device=device)
                    ww  = torch.tensor(w, dtype=torch.float32, device=device)
                    target = torch.tensor(tgt, dtype=torch.float32, device=device)
                    opt.zero_grad()
                    pred = model(rgb, ww)
                    loss = ((pred - target) ** 2).mean()
                    loss.backward()
                    opt.step()
                    total += loss.item()
                mean_loss = total / len(cases)
                history.append(mean_loss)
            hist_str = f"epochs={epochs}, final MSE={history[-1]:.6f}"
            return model, hist_str, history
    ﻿
    # =================================================
    # 4. CSV helpers
    # =================================================
    ﻿
    def parse_csv(text: str) -> List[Dict]:
        out: List[Dict] = []
        rdr = csv.DictReader(io.StringIO(text))
        for r in rdr:
            gid = (r.get("group_id") or "").strip()
            if not gid or gid.startswith("#"):
                continue
            base_hexes = [h.strip() for h in (r.get("base_hexes") or "").split(";") if h.strip()]
            weights = [float(x) for x in (r.get("weights") or "").split(";") if x.strip()]
            api_hex = (r.get("api_hex") or "").strip()
            out.append({"group_id": gid, "base_hexes": base_hexes, "weights": weights, "api_hex": api_hex})
        return out
    ﻿
    # =================================================
    # 5. Streamlit UI
    # =================================================
    ﻿
    st.set_page_config(page_title="Mini Color Mixer (Trycolors Approx)", layout="wide")
    st.title("Mini Color Mixer — Linear / KM / YN-KM / Trycolors-Approx")
    ﻿
    # ---- Sidebar configuration ----
    with st.sidebar:
        st.header("Base model")
        engine_options = ["Linear", "KM", "YN-KM"]
        if TORCH_AVAILABLE:
            engine_options.append("Trycolors-Approx")
        engine = st.selectbox("Engine", engine_options)
    ﻿
        ks_eps = st.number_input(
            "KS_EPS (for KM / YN-KM)",
            value=1e-6, min_value=1e-9, max_value=0.01, step=1e-6, format="%.8f"
        )
        set_ks_eps(ks_eps)
        yn_n = st.number_input("Yule–Nielsen n", value=1.5, min_value=0.0, max_value=10.0, step=0.1)
    ﻿
        st.markdown("---")
        st.header("Trycolors-Approx model")
    ﻿
        if not TORCH_AVAILABLE:
            st.info("Install PyTorch to enable the Trycolors-Approx neural mixer.")
        else:
            latent_dim = st.number_input("Latent dim", min_value=3, max_value=16, value=7, step=1)
            hidden_dim = st.number_input("Hidden size", min_value=8, max_value=128, value=32, step=8)
            lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
            epochs = st.number_input("Train epochs", min_value=10, max_value=1000, value=200, step=10)
    ﻿
            train_file = st.file_uploader("CSV for training (Trycolors outputs)", type=["csv"], key="train_csv")
            if st.button("Train / Retrain Trycolors-Approx"):
                if not train_file:
                    st.error("Upload a labeled CSV first.")
                else:
                    if not TORCH_AVAILABLE:
                        st.error("PyTorch not installed.")
                    else:
                        text = train_file.read().decode("utf-8", errors="ignore")
                        rows = parse_csv(text)
                        try:
                            model, hist_str, history = train_latent_model(
                                rows,
                                epochs=int(epochs),
                                lr=float(lr),
                                latent_dim=int(latent_dim),
                                hidden=int(hidden_dim),
                            )
                            st.session_state.latent_model = model
                            st.session_state.latent_history = history
                            st.success("Training finished.")
                            st.caption(hist_str)
                        except Exception as e:
                            st.error(f"Training failed: {e}")
    ﻿
        st.markdown("---")
        swatch_size = st.slider("Swatch size", 100, 600, 220, 10)
    ﻿
    # ---- Interactive mix ----
    st.subheader("Interactive mix")
    ﻿
    def _init_rows():
        if "rows" not in st.session_state:
            st.session_state.rows = [
                {"hex": "#FF2B2B", "w": 0.8},
                {"hex": "#FFFFFF", "w": 0.2},
            ]
    ﻿
    def _add_row():
        st.session_state.rows.append({"hex": "#808080", "w": 0.0})
    ﻿
    def _remove_row(idx: Optional[int] = None):
        if not st.session_state.rows:
            return
        if idx is None:
            st.session_state.rows.pop()
        elif 0 <= idx < len(st.session_state.rows):
            st.session_state.rows.pop(idx)
    ﻿
    _init_rows()
    ﻿
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Add row"):
            _add_row()
    with c2:
        if st.button("Remove last") and st.session_state.rows:
            _remove_row()
    with c3:
        if st.button("Reset to red + white"):
            st.session_state.rows = [
                {"hex": "#FF2B2B", "w": 0.8},
                {"hex": "#FFFFFF", "w": 0.2},
            ]
    ﻿
    st.markdown("**Bases and weights**")
    ﻿
    hdr = st.columns([3,3,3,1])
    hdr[0].markdown("_Color_")
    hdr[1].markdown("_Weight slider_")
    hdr[2].markdown("_Weight exact_")
    hdr[3].markdown("_Del_")
    ﻿
    to_delete = []
    for i, row in enumerate(st.session_state.rows):
        c1, c2, c3, c4 = st.columns([3,3,3,1])
        with c1:
            row["hex"] = st.color_picker(f"Color {i+1}", row["hex"], key=f"hex_{i}")
        with c2:
            row["w"] = st.slider(f"W{i+1}", 0.0, 1.0, float(row["w"]), 0.01, key=f"ws_{i}")
        with c3:
            row["w"] = float(st.number_input(f"w{i+1}", 0.0, 1.0, float(row["w"]), 0.001, key=f"wn_{i}"))
        with c4:
            if st.button("✖", key=f"del_{i}"):
                to_delete.append(i)
    ﻿
    for idx in sorted(to_delete, reverse=True):
        _remove_row(idx)
    ﻿
    opts1, opts2 = st.columns([1,1])
    with opts1:
        normalize = st.checkbox("Normalize weights", value=True)
    with opts2:
        show_calibrated = st.checkbox("Apply calibrator to KM / Trycolors-Approx", value=False)
    ﻿
    mix_clicked = st.button("Mix colors", type="primary")
    ﻿
    if mix_clicked:
        try:
            base_hexes = [r["hex"].strip() for r in st.session_state.rows if r["hex"].strip()]
            weights = [float(r["w"]) for r in st.session_state.rows]
            if len(base_hexes) == 0 or len(base_hexes) != len(weights):
                raise ValueError("Provide the same number of colors and weights.")
            if sum(weights) <= 0:
                raise ValueError("Weights sum to 0.")
    ﻿
            bases_lin = [hex_to_linear_rgb(h) for h in base_hexes]
            w = normalize_weights(weights) if normalize else weights
    ﻿
            if engine == "Linear":
                lin = mix_linear_rgb(bases_lin, w)
            elif engine == "KM":
                lin = mix_kubelka_munk(bases_lin, w)
            elif engine == "YN-KM":
                lin = mix_kubelka_munk_yn(bases_lin, w, n=float(yn_n))
            else:  # Trycolors-Approx
                if not TORCH_AVAILABLE:
                    raise RuntimeError("PyTorch not available.")
                model = st.session_state.get("latent_model")
                if model is None:
                    raise RuntimeError("Trycolors-Approx model not trained yet.")
                device = next(model.parameters()).device
                model.eval()
                with torch.no_grad():
                    rgb = torch.tensor(bases_lin, dtype=torch.float32, device=device)
                    ww  = torch.tensor(w, dtype=torch.float32, device=device)
                    lin_t = model(rgb, ww)
                    lin = tuple(float(v) for v in lin_t.cpu().numpy())
    ﻿
            out_hex = linear_rgb_to_hex(lin)
            show_swatch(f"{engine} result", out_hex, size=swatch_size)
    ﻿
            # Optional: calibrated KM (if model & show_calibrated)
            if show_calibrated and st.session_state.get("calib_coeffs") and engine in ("KM","YN-KM","Trycolors-Approx"):
                kind, payload = st.session_state.calib_coeffs
                if kind == "linear":
                    corr = apply_channel_polys(lin, payload)
                else:
                    coeffs, m = payload
                    corr = apply_log_poly(lin, coeffs, mode=m)
                cal_hex = linear_rgb_to_hex(corr)
                show_swatch("Calibrated", cal_hex, size=swatch_size)
    ﻿
            st.code(
                "Bases=" + str(base_hexes) +
                "\nWeights=" + str(w) +
                f"\nEngine={engine} -> {out_hex}"
            )
        except Exception as e:
            st.error(f"Mix failed: {e}")
    ﻿
    st.markdown("---")
    st.subheader("Evaluate model vs Trycolors CSV")
    ﻿
    eval_file = st.file_uploader("Upload Trycolors CSV", type=["csv"], key="eval_csv")
    ﻿
    if eval_file is not None:
        text = eval_file.read().decode("utf-8", errors="ignore")
    ﻿
        def predict_fn(base_hexes, weights):
            bases_lin = [hex_to_linear_rgb(h) for h in base_hexes]
            w = normalize_weights(weights)
    ﻿
            if engine == "Linear":
                lin = mix_linear_rgb(bases_lin, w)
            elif engine == "KM":
                lin = mix_kubelka_munk(bases_lin, w)
            elif engine == "YN-KM":
                lin = mix_kubelka_munk_yn(bases_lin, w, n=float(yn_n))
            elif engine == "Trycolors-Approx":
                if not TORCH_AVAILABLE:
                    raise RuntimeError("PyTorch not available.")
                model = st.session_state.get("latent_model")
                if model is None:
                    raise RuntimeError("Trycolors-Approx model not trained yet.")
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    rgb = torch.tensor(bases_lin, dtype=torch.float32, device=device)
                    ww  = torch.tensor(w, dtype=torch.float32, device=device)
                    lin_t = model(rgb, ww)
                    lin = tuple(float(v) for v in lin_t.cpu().numpy())
            else:
                raise RuntimeError(f"Unknown engine: {engine}")
    ﻿
            return linear_rgb_to_hex(lin)
    ﻿
        results, mean_rmse = evaluate_model_on_csv(text, predict_fn)
    ﻿
        if mean_rmse is None:
            st.warning("No labeled rows with api_hex in this CSV.")
        else:
            st.write(f"**Mean RMSE vs Trycolors:** {mean_rmse:.5f}")
    ﻿
        # Show first ~50 rows for inspection
        if results:
            st.dataframe(results[:50])
    ﻿
    st.caption("Tip: use your Trycolors CSV with the Trycolors-Approx model to see how close you can get to their behavior.")

