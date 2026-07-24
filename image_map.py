#!/usr/bin/env python3
"""
image_map.py
============

PixPlot-style **similarity map** for the Reference Image Library.

The map-build job (admin-triggered, background thread — same job table as the
ingestion tool):

    1. Embeds every artwork's thumbnail with a small CNN
       (torchvision MobileNetV3-Small, 576-dim features, CPU-friendly) and
       stores the vector in ``image_library.embedding``.
    2. Projects all vectors to 2-D with t-SNE (cosine metric, PCA init) so
       visually similar works land near each other, k-means clusters them into
       "hotspots", normalises coordinates to [0, 1], and stores
       ``map_x`` / ``map_y`` / ``map_cluster`` per row.

The viewer (``art-map.html``) then just fetches ``GET /api/library/map`` and
draws the already-baked gallery thumbnails at those positions.

torch / torchvision / sklearn are imported lazily inside the job so server
boot stays fast and the model (~10 MB download, cached) only loads when an
admin actually rebuilds the map.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import image_library_db as db
import bucket_storage as bs
from image_sources import BAKED_THUMB_DIR, PROXY_CACHE_DIR, render_cached_image

_BATCH = 16  # images per forward pass


# ---------------------------------------------------------------------------
# Thumbnail resolution — reuse the gallery's baked/cached files where possible
# ---------------------------------------------------------------------------

def _thumb_file(art: Dict[str, Any]) -> Optional[str]:
    """Return a local JPEG path for this artwork's thumbnail, producing one via
    the same pipeline the gallery proxy uses if nothing is on disk yet."""
    baked = os.path.join(BAKED_THUMB_DIR, f"{art['id']}.jpg")
    if os.path.isfile(baked) and os.path.getsize(baked) > 0:
        return baked

    cache_path = os.path.join(PROXY_CACHE_DIR, f"{art['id']}_thumb.jpg")
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    # Bucket rows: pull the pre-generated thumb from object storage.
    if art.get("storage_key") and art.get("thumb_key") and bs.is_configured():
        try:
            data = bs.get_bytes(art["thumb_key"])
            os.makedirs(PROXY_CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as fh:
                fh.write(data)
            return cache_path
        except Exception:
            return None

    candidates = [u for u in (art.get("thumb_url"), art.get("image_url")) if u]
    from image_sources import IMAGE_LIBRARY_DIR
    local_abs = (
        os.path.join(IMAGE_LIBRARY_DIR, art["local_path"]) if art.get("local_path") else None
    )
    return render_cached_image(cache_path, candidates, "thumb", local_abs)


# ---------------------------------------------------------------------------
# CNN embeddings
# ---------------------------------------------------------------------------

_model_singleton = None


def _get_model():
    """Lazy-load MobileNetV3-Small as a feature extractor (features + avgpool,
    no classifier head → 576-dim vectors)."""
    global _model_singleton
    if _model_singleton is None:
        import torch
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        net = mobilenet_v3_small(weights=weights)
        net.eval()
        _model_singleton = (net, weights.transforms())
    return _model_singleton


def _embed_batch(paths: List[str]) -> List[Optional[List[float]]]:
    """Embed a batch of image files. Returns one vector (or None) per path."""
    import torch
    from PIL import Image

    net, preprocess = _get_model()
    tensors, ok_idx = [], []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                tensors.append(preprocess(im.convert("RGB")))
            ok_idx.append(i)
        except Exception:
            continue

    out: List[Optional[List[float]]] = [None] * len(paths)
    if not tensors:
        return out
    with torch.no_grad():
        x = torch.stack(tensors)
        feats = net.avgpool(net.features(x)).flatten(1)  # (B, 576)
    for j, i in enumerate(ok_idx):
        out[i] = [round(float(v), 5) for v in feats[j]]
    return out


# ---------------------------------------------------------------------------
# 2-D layout + clusters
# ---------------------------------------------------------------------------

def _compute_layout(ids: List[int], vectors: List[List[float]]) -> List[Tuple[int, float, float, int]]:
    """t-SNE the vectors to 2-D, k-means them into hotspots, normalise to
    [0, 1]. Returns (id, x, y, cluster) rows."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-8)
    n = len(ids)

    if n >= 8:
        perplexity = float(min(30, max(5, (n - 1) // 4)))
        pts = TSNE(
            n_components=2, metric="cosine", init="pca",
            perplexity=perplexity, random_state=42,
        ).fit_transform(X)
    else:  # too few points for t-SNE — PCA is enough
        pts = PCA(n_components=min(2, n)).fit_transform(X)
        if pts.shape[1] < 2:
            pts = np.column_stack([pts, np.zeros(n)])

    k = int(min(12, max(2, n // 40))) if n >= 4 else 1
    clusters = (
        KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if k > 1 else np.zeros(n, dtype=int)
    )

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    pts = (pts - lo) / span  # → [0, 1] both axes

    return [
        (ids[i], float(round(pts[i, 0], 5)), float(round(pts[i, 1], 5)), int(clusters[i]))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

def run_map_job(job_id: int, force: bool = False) -> None:
    """Background worker: embed artworks missing a vector (all of them when
    ``force``), then recompute and store the 2-D layout. Progress is reported
    through the shared image_library_jobs row."""
    db.update_job(job_id, status="running", message="Listing artworks…")
    try:
        todo = db.artworks_for_embedding(force=force)
        done = failed = 0
        db.update_job(job_id, requested=len(todo),
                      message=f"Embedding {len(todo)} artworks…" if todo else "Embeddings up to date.")

        for start in range(0, len(todo), _BATCH):
            batch = todo[start:start + _BATCH]
            paths = [_thumb_file(a) or "" for a in batch]
            vecs = _embed_batch(paths)
            for art, vec in zip(batch, vecs):
                if vec is None:
                    failed += 1
                    continue
                db.set_embedding(art["id"], vec)
                done += 1
            db.update_job(job_id, fetched=done + failed, saved=done, failed=failed,
                          message=f"Embedded {done}/{len(todo)}…")

        pairs = db.load_embeddings()
        if len(pairs) < 2:
            db.update_job(job_id, status="error",
                          message="Not enough embedded artworks to build a map.")
            return
        db.update_job(job_id, message=f"Computing 2-D layout for {len(pairs)} artworks…")
        layout = _compute_layout([p[0] for p in pairs], [p[1] for p in pairs])
        db.save_map_layout(layout)
        db.update_job(
            job_id, status="done", saved=done, failed=failed,
            message=f"Map ready: {len(layout)} artworks placed"
                    + (f" ({failed} could not be embedded)." if failed else "."),
        )
    except Exception as e:  # noqa: BLE001
        db.update_job(job_id, status="error", message=f"Error: {e}")


__all__ = ["run_map_job"]
