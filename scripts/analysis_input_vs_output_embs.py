#!/usr/bin/env python
# coding: utf-8
"""
Single-file experiment harness for comparing embedding spaces between models.

Features:
- Model families: Qwen3, Llama3, Gemma3 (base + instruct variants).
- Filter out models with tied input/output embeddings.
- For every *untied* model:
    * Sample a subset of tokens from its vocab.
    * Compute input vs output embedding similarity with multiple metrics:
        - Distance matrix correlation (cosine).
        - Linear CKA.
        - Orthogonal Procrustes residual.
        - kNN Jaccard overlap.
        - LocSim-n (local similarity via softmax on n-NN).
        - LocSim-inf (local similarity via sparsemax on (approx.) all tokens).
- Token-alignment helpers for future cross-model comparisons (string overlap).
- Results:
    * Written to JSONL (one line per metric & model).
    * Printed summary: aggregates per family/variant + per-model tables.

This is intentionally a single big script so you can hack on it easily.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Dataclasses & basic result container
# -----------------------------------------------------------------------------


@dataclass
class MetricResult:
    """
    Container for a single metric's result.

    Attributes
    ----------
    name: str
        Metric name, e.g. "dist_corr_cosine", "locsim_n_20".
    primary: float
        Primary similarity score scaled to [0, 1], higher = more similar.
    raw: Dict[str, float]
        Raw scalar values (e.g. Pearson r, residual, etc.).
    per_token: Optional[np.ndarray]
        Optional per-token scores (e.g. local similarities). Shape [N].
    meta: Dict[str, Any]
        Additional metadata (e.g. hyperparameters, N, dims).
    """

    name: str
    primary: float
    raw: Dict[str, float]
    per_token: Optional[np.ndarray] = None
    meta: Dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Core linear algebra helpers
# -----------------------------------------------------------------------------


def pairwise_cosine_similarity(X: Tensor) -> Tensor:
    """Pairwise cosine similarities for rows of X. Returns [N, N]."""
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return Xn @ Xn.T


def pairwise_cosine_distance(X: Tensor) -> Tensor:
    """Pairwise cosine distances 1 - cosine_similarity for rows of X."""
    return 1.0 - pairwise_cosine_similarity(X)


# -----------------------------------------------------------------------------
# Global metrics
# -----------------------------------------------------------------------------
def metric_distance_matrix_corr(
    X: Tensor,
    Y: Tensor,
    cache: Dict[str, Any],
    name: str = "dist_corr_cosine",
    recon_topk: int = 200,
    recon_tau: float = 1.0,
) -> MetricResult:
    simX = cache.get("cosine_sim_X")
    simY = cache.get("cosine_sim_Y")
    if simX is None or simY is None:
        simX = pairwise_cosine_similarity(X)
        simY = pairwise_cosine_similarity(Y)
        cache["cosine_sim_X"] = simX
        cache["cosine_sim_Y"] = simY

    N = simX.shape[0]
    iu = torch.triu_indices(N, N, offset=1, device=simX.device)
    vX = simX[iu[0], iu[1]].detach().cpu().numpy()
    vY = simY[iu[0], iu[1]].detach().cpu().numpy()

    vx_mean = vX.mean()
    vy_mean = vY.mean()
    num = np.sum((vX - vx_mean) * (vY - vy_mean))
    den = math.sqrt(float(np.sum((vX - vx_mean) ** 2) * np.sum((vY - vy_mean) ** 2))) + 1e-12
    r = float(num / den)
    primary = 0.5 * (r + 1.0)

    # --- real recon errors from global similarity weights ---
    eps = 1e-12
    Xf = X.float()
    Yf = Y.float()

    mse_y_from_x = []
    mse_x_from_y = []
    rel_y_from_x = []
    rel_x_from_y = []

    k = min(recon_topk, N - 1)

    for i in range(N):
        # X -> Y using weights from simX row i
        sxi = simX[i].clone()
        sxi[i] = -1e9
        idxX = torch.topk(sxi, k=k, largest=True).indices
        wX = torch.softmax(sxi[idxX] / recon_tau, dim=0)

        y_hat = (wX.unsqueeze(0) @ Yf[idxX]).squeeze(0)
        mse_y = torch.mean((Yf[i] - y_hat) ** 2)
        denom_y = torch.mean(Yf[i] ** 2).clamp_min(eps)
        mse_y_from_x.append(mse_y)
        rel_y_from_x.append(mse_y / denom_y)

        # Y -> X using weights from simY row i
        syi = simY[i].clone()
        syi[i] = -1e9
        idxY = torch.topk(syi, k=k, largest=True).indices
        wY = torch.softmax(syi[idxY] / recon_tau, dim=0)

        x_hat = (wY.unsqueeze(0) @ Xf[idxY]).squeeze(0)
        mse_x = torch.mean((Xf[i] - x_hat) ** 2)
        denom_x = torch.mean(Xf[i] ** 2).clamp_min(eps)
        mse_x_from_y.append(mse_x)
        rel_x_from_y.append(mse_x / denom_x)

    my = torch.stack(mse_y_from_x).mean()
    mx = torch.stack(mse_x_from_y).mean()
    ry = torch.stack(rel_y_from_x).mean()
    rx = torch.stack(rel_x_from_y).mean()
    mse_bi = 0.5 * (my + mx)
    rel_bi = 0.5 * (ry + rx)

    return MetricResult(
        name=name,
        primary=primary,
        raw={
            "pearson_r": r,
            "primary_similarity": primary,
            "recon_topk": float(k),
            "recon_tau": float(recon_tau),
            "recon_mse_y_from_x": float(my.item()),
            "recon_mse_x_from_y": float(mx.item()),
            "recon_mse_bidirectional": float(mse_bi.item()),
            "recon_sim_from_mse_bidirectional": float((1.0 / (1.0 + mse_bi)).item()),
            "recon_relmse_y_from_x": float(ry.item()),
            "recon_relmse_x_from_y": float(rx.item()),
            "recon_relmse_bidirectional": float(rel_bi.item()),
            "recon_sim_from_relmse_bidirectional": float((1.0 / (1.0 + rel_bi)).item()),
        },
        per_token=None,
        meta={"N": int(N)},
    )


def metric_cka(
    X: Tensor,
    Y: Tensor,
    cache: Dict[str, Any],
    name: str = "cka_linear",
) -> MetricResult:
    # Center
    muX = X.mean(dim=0, keepdim=True)
    muY = Y.mean(dim=0, keepdim=True)
    Xc = X - muX
    Yc = Y - muY

    # CKA
    K = Xc.T @ Xc
    L = Yc.T @ Yc
    num = torch.norm(Yc.T @ Xc, p="fro") ** 2
    den = torch.norm(K, p="fro") * torch.norm(L, p="fro") + 1e-12
    cka_val = float((num / den).item())
    primary = cka_val

    # --- real recon errors via bidirectional least squares (with intercept) ---
    eps = 1e-12
    Xcf = Xc.float()
    Ycf = Yc.float()
    muYf = muY.float()
    muXf = muX.float()

    A = torch.linalg.lstsq(Xcf, Ycf).solution  # [d_x, d_y]
    Yhat = (Xcf @ A) + muYf
    mse_y = torch.mean((Y.float() - Yhat) ** 2)
    rel_y = mse_y / torch.mean(Y.float() ** 2).clamp_min(eps)

    B = torch.linalg.lstsq(Ycf, Xcf).solution  # [d_y, d_x]
    Xhat = (Ycf @ B) + muXf
    mse_x = torch.mean((X.float() - Xhat) ** 2)
    rel_x = mse_x / torch.mean(X.float() ** 2).clamp_min(eps)

    mse_bi = 0.5 * (mse_y + mse_x)
    rel_bi = 0.5 * (rel_y + rel_x)

    return MetricResult(
        name=name,
        primary=primary,
        raw={
            "cka": cka_val,
            "primary_similarity": primary,
            "recon_mse_y_from_x": float(mse_y.item()),
            "recon_mse_x_from_y": float(mse_x.item()),
            "recon_mse_bidirectional": float(mse_bi.item()),
            "recon_sim_from_mse_bidirectional": float((1.0 / (1.0 + mse_bi)).item()),
            "recon_relmse_y_from_x": float(rel_y.item()),
            "recon_relmse_x_from_y": float(rel_x.item()),
            "recon_relmse_bidirectional": float(rel_bi.item()),
            "recon_sim_from_relmse_bidirectional": float((1.0 / (1.0 + rel_bi)).item()),
        },
        per_token=None,
        meta={"d_x": int(X.shape[1]), "d_y": int(Y.shape[1])},
    )


def metric_procrustes_residual(
    X: Tensor,
    Y: Tensor,
    cache: Dict[str, Any],
    name: str = "procrustes_residual",
) -> MetricResult:
    muX = X.mean(dim=0, keepdim=True)
    muY = Y.mean(dim=0, keepdim=True)
    Xc = (X - muX).float()
    Yc = (Y - muY).float()

    eps = 1e-12

    # X -> Y
    Mxy = Xc.T @ Yc  # [d_x, d_y]
    U, S, Vh = torch.linalg.svd(Mxy, full_matrices=False)
    Rxy = U @ Vh  # [d_x, d_y]
    Yhat = (Xc @ Rxy) + muY.float()

    resid_y = torch.norm(Y.float() - Yhat, p="fro") / (torch.norm(Y.float(), p="fro") + eps)
    mse_y = torch.mean((Y.float() - Yhat) ** 2)
    rel_y = mse_y / torch.mean(Y.float() ** 2).clamp_min(eps)

    # Y -> X
    Myx = Yc.T @ Xc  # [d_y, d_x]
    U2, S2, Vh2 = torch.linalg.svd(Myx, full_matrices=False)
    Ryx = U2 @ Vh2  # [d_y, d_x]
    Xhat = (Yc @ Ryx) + muX.float()

    resid_x = torch.norm(X.float() - Xhat, p="fro") / (torch.norm(X.float(), p="fro") + eps)
    mse_x = torch.mean((X.float() - Xhat) ** 2)
    rel_x = mse_x / torch.mean(X.float() ** 2).clamp_min(eps)

    resid_bi = 0.5 * (resid_y + resid_x)
    mse_bi = 0.5 * (mse_y + mse_x)
    rel_bi = 0.5 * (rel_y + rel_x)

    primary = float((1.0 / (1.0 + resid_bi)).item())

    return MetricResult(
        name=name,
        primary=primary,
        raw={
            "residual_y_from_x": float(resid_y.item()),
            "residual_x_from_y": float(resid_x.item()),
            "residual_bidirectional": float(resid_bi.item()),
            "primary_similarity": primary,
            "recon_mse_y_from_x": float(mse_y.item()),
            "recon_mse_x_from_y": float(mse_x.item()),
            "recon_mse_bidirectional": float(mse_bi.item()),
            "recon_sim_from_mse_bidirectional": float((1.0 / (1.0 + mse_bi)).item()),
            "recon_relmse_y_from_x": float(rel_y.item()),
            "recon_relmse_x_from_y": float(rel_x.item()),
            "recon_relmse_bidirectional": float(rel_bi.item()),
            "recon_sim_from_relmse_bidirectional": float((1.0 / (1.0 + rel_bi)).item()),
            "singular_values_mean_xy": float(S.mean().item()),
            "singular_values_mean_yx": float(S2.mean().item()),
        },
        per_token=None,
        meta={"d_x": int(X.shape[1]), "d_y": int(Y.shape[1])},
    )


# -----------------------------------------------------------------------------
# Local baseline metric: kNN Jaccard overlap
# -----------------------------------------------------------------------------


def metric_knn_jaccard(
    X: Tensor,
    Y: Tensor,
    cache: Dict[str, Any],
    k: int = 10,
    name: Optional[str] = None,
) -> MetricResult:
    """
    kNN Jaccard overlap between local neighbourhoods.

    For each token i, computes its k nearest neighbours in X and Y (cosine
    distance), and returns the Jaccard similarity between the neighbour sets.

    Primary similarity = mean Jaccard in [0, 1].
    """
    if name is None:
        name = f"knn_jaccard_k={k}"

    DX = cache.get("cosine_dist_X")
    DY = cache.get("cosine_dist_Y")
    if DX is None or DY is None:
        DX = pairwise_cosine_distance(X)
        DY = pairwise_cosine_distance(Y)
        cache["cosine_dist_X"] = DX
        cache["cosine_dist_Y"] = DY

    N = DX.shape[0]
    jaccs: List[float] = []
    for i in range(N):
        dxi = DX[i]
        dyi = DY[i]
        idx_x = torch.topk(dxi, k=k + 1, largest=False).indices
        idx_x = idx_x[idx_x != i][:k]
        idx_y = torch.topk(dyi, k=k + 1, largest=False).indices
        idx_y = idx_y[idx_y != i][:k]

        set_x = set(idx_x.detach().cpu().tolist())
        set_y = set(idx_y.detach().cpu().tolist())
        inter = len(set_x & set_y)
        union = len(set_x | set_y)
        j = inter / union if union > 0 else 1.0
        jaccs.append(j)

    jaccs_arr = np.asarray(jaccs, dtype=np.float32)
    primary = float(jaccs_arr.mean())

    return MetricResult(
        name=name,
        primary=primary,
        raw={
            "mean_jaccard": primary,
            "std_jaccard": float(jaccs_arr.std()),
            "k": float(k),
        },
        per_token=jaccs_arr,
        meta={"N": int(N)},
    )


# -----------------------------------------------------------------------------
# LocSim-n and LocSim-inf
# -----------------------------------------------------------------------------


def jsd_sparse_torch(
    idx_p: Tensor,
    p: Tensor,
    idx_q: Tensor,
    q: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    Sparse Jensen-Shannon divergence between two distributions.

    idx_p, idx_q : 1D LongTensors of indices of non-zero entries.
    p, q        : probabilities on those indices, summing to 1.

    Returns
    -------
    jsd : scalar tensor in [0, 1].
    """
    device = p.device
    all_idx = torch.unique(torch.cat([idx_p, idx_q]))
    P = torch.zeros(all_idx.shape[0], device=device)
    Q = torch.zeros_like(P)

    for j, idx in enumerate(all_idx):
        mask_p = idx_p == idx
        if mask_p.any():
            P[j] = p[mask_p][0]
        mask_q = idx_q == idx
        if mask_q.any():
            Q[j] = q[mask_q][0]

    M = 0.5 * (P + Q)

    def kl(a: Tensor, b: Tensor) -> Tensor:
        a_safe = a + eps
        b_safe = b + eps
        return torch.sum(a_safe * (torch.log(a_safe) - torch.log(b_safe)))

    jsd = 0.5 * kl(P, M) + 0.5 * kl(Q, M)
    jsd_norm = jsd / math.log(2.0)
    return jsd_norm


def sparsemax(z: Tensor) -> Tensor:
    """
    Sparsemax over a 1D tensor (logits).

    Reference: Martins & Astudillo (2016), From Softmax to Sparsemax.
    """
    # Sort in descending order
    z_sorted, _ = torch.sort(z, descending=True)
    cssv = torch.cumsum(z_sorted, dim=0) - 1
    rhos = torch.arange(1, z.numel() + 1, device=z.device, dtype=z.dtype)
    cond = z_sorted - cssv / rhos > 0

    if not cond.any():
        tau = cssv[-1] / rhos[-1]
    else:
        k = int(cond.nonzero()[-1]) + 1
        tau = cssv[k - 1] / k

    return torch.clamp(z - tau, min=0.0)


def metric_locsim_n(
    X: Tensor,
    Y: Tensor,
    cache: Dict[str, Any],
    n: int = 20,
    tau: float = 1.0,
    name: Optional[str] = None,
) -> MetricResult:
    if name is None:
        name = f"locsim_n_{n}"

    DX = cache.get("cosine_dist_X")
    DY = cache.get("cosine_dist_Y")
    if DX is None or DY is None:
        DX = pairwise_cosine_distance(X)
        DY = pairwise_cosine_distance(Y)
        cache["cosine_dist_X"] = DX
        cache["cosine_dist_Y"] = DY

    N = DX.shape[0]
    sims: List[Tensor] = []

    mse_y_from_x: List[Tensor] = []
    mse_x_from_y: List[Tensor] = []
    rel_y_from_x: List[Tensor] = []
    rel_x_from_y: List[Tensor] = []

    eps = 1e-12

    # compute in float32 for stability
    Xf = X.float()
    Yf = Y.float()

    for i in range(N):
        dxi = DX[i]
        dyi = DY[i]

        idxX = torch.topk(dxi, k=n + 1, largest=False).indices
        idxX = idxX[idxX != i][:n]
        idxY = torch.topk(dyi, k=n + 1, largest=False).indices
        idxY = idxY[idxY != i][:n]

        pX = torch.softmax(-dxi[idxX] / tau, dim=0)  # weights from X
        pY = torch.softmax(-dyi[idxY] / tau, dim=0)  # weights from Y

        jsd = jsd_sparse_torch(idxX, pX, idxY, pY)
        sims.append(1.0 - jsd)

        # X -> Y recon using X-weights
        y_hat = (pX.unsqueeze(0) @ Yf[idxX]).squeeze(0)
        mse_y = torch.mean((Yf[i] - y_hat) ** 2)
        denom_y = torch.mean(Yf[i] ** 2).clamp_min(eps)
        mse_y_from_x.append(mse_y)
        rel_y_from_x.append(mse_y / denom_y)

        # Y -> X recon using Y-weights
        x_hat = (pY.unsqueeze(0) @ Xf[idxY]).squeeze(0)
        mse_x = torch.mean((Xf[i] - x_hat) ** 2)
        denom_x = torch.mean(Xf[i] ** 2).clamp_min(eps)
        mse_x_from_y.append(mse_x)
        rel_x_from_y.append(mse_x / denom_x)

    sims_t = torch.stack(sims)
    primary = float(sims_t.mean().item())

    my = torch.stack(mse_y_from_x)
    mx = torch.stack(mse_x_from_y)
    ry = torch.stack(rel_y_from_x)
    rx = torch.stack(rel_x_from_y)

    mse_bi = 0.5 * (my.mean() + mx.mean())
    rel_bi = 0.5 * (ry.mean() + rx.mean())

    return MetricResult(
        name=name,
        primary=primary,
        raw={
            "mean_locsim": primary,
            "std_locsim": float(sims_t.std().item()),
            "n": float(n),
            "tau": float(tau),
            "recon_mse_y_from_x": float(my.mean().item()),
            "recon_mse_x_from_y": float(mx.mean().item()),
            "recon_mse_bidirectional": float(mse_bi.item()),
            "recon_sim_from_mse_bidirectional": float((1.0 / (1.0 + mse_bi)).item()),
            "recon_relmse_y_from_x": float(ry.mean().item()),
            "recon_relmse_x_from_y": float(rx.mean().item()),
            "recon_relmse_bidirectional": float(rel_bi.item()),
            "recon_sim_from_relmse_bidirectional": float((1.0 / (1.0 + rel_bi)).item()),
        },
        per_token=sims_t.detach().cpu().numpy(),
        meta={"N": int(N)},
    )


def metric_locsim_inf(
    X: Tensor,
    Y: Tensor,
    cache: Dict[str, Any],
    tau: float = 1.0,
    topk: Optional[int] = None,  # recommend using this
    name: str = "locsim_inf",
) -> MetricResult:
    DX = cache.get("cosine_dist_X")
    DY = cache.get("cosine_dist_Y")
    if DX is None or DY is None:
        DX = pairwise_cosine_distance(X)
        DY = pairwise_cosine_distance(Y)
        cache["cosine_dist_X"] = DX
        cache["cosine_dist_Y"] = DY

    N = DX.shape[0]
    sims: List[Tensor] = []

    mse_y_from_x: List[Tensor] = []
    mse_x_from_y: List[Tensor] = []
    rel_y_from_x: List[Tensor] = []
    rel_x_from_y: List[Tensor] = []

    eps = 1e-12

    for i in range(N):
        dxi = DX[i]
        dyi = DY[i]

        if topk is not None and topk < N:
            # take topk+1 then drop self
            idxX = torch.topk(dxi, k=topk + 1, largest=False).indices
            idxX = idxX[idxX != i][:topk]
            idxY = torch.topk(dyi, k=topk + 1, largest=False).indices
            idxY = idxY[idxY != i][:topk]

            zX = -dxi[idxX] / tau
            zY = -dyi[idxY] / tau
            pX = sparsemax(zX)
            pY = sparsemax(zY)

            jsd = jsd_sparse_torch(idxX, pX, idxY, pY)

            # recon
            y_hat = (pX.unsqueeze(0) @ Y[idxX]).squeeze(0)
            x_hat = (pY.unsqueeze(0) @ X[idxY]).squeeze(0)

        else:
            # full support (O(N^2)); exclude self by forcing its logit very negative
            idx_all = torch.arange(N, device=dxi.device)
            zX = -dxi / tau
            zY = -dyi / tau
            zX[i] = zX.min() - 1e6
            zY[i] = zY.min() - 1e6

            pX = sparsemax(zX)
            pY = sparsemax(zY)

            jsd = jsd_sparse_torch(idx_all, pX, idx_all, pY)

            y_hat = (pX.unsqueeze(0) @ Y).squeeze(0)
            x_hat = (pY.unsqueeze(0) @ X).squeeze(0)

        sims.append(1.0 - jsd)

        mse_y = torch.mean((Y[i] - y_hat) ** 2)
        mse_x = torch.mean((X[i] - x_hat) ** 2)

        denom_y = torch.mean(Y[i] ** 2).clamp_min(eps)
        denom_x = torch.mean(X[i] ** 2).clamp_min(eps)

        mse_y_from_x.append(mse_y)
        mse_x_from_y.append(mse_x)
        rel_y_from_x.append(mse_y / denom_y)
        rel_x_from_y.append(mse_x / denom_x)

    sims_t = torch.stack(sims)
    primary = float(sims_t.mean().item())

    my = torch.stack(mse_y_from_x)
    mx = torch.stack(mse_x_from_y)
    ry = torch.stack(rel_y_from_x)
    rx = torch.stack(rel_x_from_y)

    mse_bi = 0.5 * (my.mean() + mx.mean())
    rel_bi = 0.5 * (ry.mean() + rx.mean())

    return MetricResult(
        name=name if topk is None else f"{name}_topk={topk}",
        primary=primary,
        raw={
            "mean_locsim_inf": primary,
            "std_locsim_inf": float(sims_t.std().item()),
            "tau": float(tau),
            "topk": float(topk or N),
            "recon_mse_y_from_x": float(my.mean().item()),
            "recon_mse_x_from_y": float(mx.mean().item()),
            "recon_mse_bidirectional": float(mse_bi.item()),
            "recon_sim_from_mse_bidirectional": float((1.0 / (1.0 + mse_bi)).item()),
            "recon_relmse_y_from_x": float(ry.mean().item()),
            "recon_relmse_x_from_y": float(rx.mean().item()),
            "recon_relmse_bidirectional": float(rel_bi.item()),
            "recon_sim_from_relmse_bidirectional": float((1.0 / (1.0 + rel_bi)).item()),
        },
        per_token=sims_t.detach().cpu().numpy(),
        meta={"N": int(N)},
    )


# -----------------------------------------------------------------------------
# Token alignment between models with different vocabularies
# -----------------------------------------------------------------------------


def default_token_filter(token_str: str) -> bool:
    """
    Heuristic filter to drop obvious special tokens.

    You can customize this depending on the models (e.g. sentencepiece vs BPE).
    """
    s = token_str
    if s.startswith("<") and s.endswith(">"):
        return False
    if s in {"<pad>", "<s>", "</s>", "<unk>", "<bos>", "<eos>"}:
        return False
    return True


def build_token_alignment(
    tokenizer_a,
    tokenizer_b,
    max_tokens: Optional[int] = None,
    filter_fn: Optional[Callable[[str], bool]] = default_token_filter,
) -> Tuple[List[str], Tensor, Tensor]:
    """
    Build a token alignment between two tokenizers based on string overlap.

    Returns
    -------
    token_strings : list[str]
        Canonical token strings shared by both vocabularies.
    token_ids_a : LongTensor [N]
        Corresponding vocab ids in tokenizer_a.
    token_ids_b : LongTensor [N]
        Corresponding vocab ids in tokenizer_b.
    """
    vocab_a = tokenizer_a.get_vocab()  # dict: str -> id
    vocab_b = tokenizer_b.get_vocab()

    set_a = set(vocab_a.keys())
    set_b = set(vocab_b.keys())
    common = set_a & set_b

    if filter_fn is not None:
        common = {s for s in common if filter_fn(s)}

    if not common:
        raise ValueError("No overlapping tokens between tokenizers.")

    token_strings = sorted(common)

    if max_tokens is not None and len(token_strings) > max_tokens:
        token_strings = token_strings[:max_tokens]

    token_ids_a = torch.tensor([vocab_a[s] for s in token_strings], dtype=torch.long)
    token_ids_b = torch.tensor([vocab_b[s] for s in token_strings], dtype=torch.long)

    return token_strings, token_ids_a, token_ids_b


def build_token_subset_single(
    tokenizer,
    max_tokens: Optional[int] = None,
    filter_fn: Optional[Callable[[str], bool]] = default_token_filter,
) -> Tuple[List[str], Tensor]:
    """
    Build a token subset for a single tokenizer (for within-model comparisons).

    Returns
    -------
    token_strings : list[str]
    token_ids : LongTensor [N]
    """
    vocab = tokenizer.get_vocab()
    toks = [s for s in vocab.keys() if filter_fn(s)]
    toks = sorted(toks)
    if max_tokens is not None and len(toks) > max_tokens:
        toks = toks[:max_tokens]
    token_ids = torch.tensor([vocab[s] for s in toks], dtype=torch.long)
    return toks, token_ids


# -----------------------------------------------------------------------------
# Embedding extraction helpers for HuggingFace causal LM models
# -----------------------------------------------------------------------------


def get_input_embeddings_tensor(model: PreTrainedModel) -> Tensor:
    emb_mod = model.get_input_embeddings()
    if emb_mod is None or not hasattr(emb_mod, "weight"):
        raise ValueError("Input embeddings module has no `.weight` attribute.")
    return emb_mod.weight.data


def get_output_embeddings_tensor(model: PreTrainedModel) -> Optional[Tensor]:
    """
    Try to get the real output embedding / LM head matrix [V, d], or None.
    """
    if hasattr(model, "get_output_embeddings"):
        out_mod = model.get_output_embeddings()
        if out_mod is not None and hasattr(out_mod, "weight"):
            return out_mod.weight.data

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.data

    for attr in ("output_layer", "score", "classifier"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            if hasattr(m, "weight"):
                return m.weight.data

    return None


def embeddings_are_tied(model: PreTrainedModel) -> bool:
    """
    Returns True if input and output embeddings share the same underlying
    weight tensor (data_ptr equality). If model has no explicit output
    embeddings, returns False (treated as untied).
    """
    in_mod = model.get_input_embeddings()
    out_mod = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if in_mod is None or out_mod is None:
        return False

    w_in = getattr(in_mod, "weight", None)
    w_out = getattr(out_mod, "weight", None)
    if w_in is None or w_out is None:
        return False

    try:
        return w_in.data_ptr() == w_out.data_ptr()
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Model family definitions (Qwen3, Llama3, Gemma3; base + instruct)
# -----------------------------------------------------------------------------

VariantKind = Literal["base", "instruct"]


class FamilySpec(Dict[str, List[str]]):
    base: List[str]
    instruct: List[str]


MODEL_FAMILIES: Dict[str, FamilySpec] = {
    "llama3": {
        "base": [
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
        ],
        "instruct": [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
        ],
    },
    "gemma3": {
        "base": [
            "google/gemma-3-1b-pt",
            "google/gemma-3-4b-pt",
            "google/gemma-3-12b-pt",
            "google/gemma-3-27b-pt",
        ],
        "instruct": [
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
        ],
    },
    "qwen3": {
        "base": [
            "Qwen/Qwen3-0.6B-Base",
            "Qwen/Qwen3-1.7B-Base",
            "Qwen/Qwen3-4B-Base",
            "Qwen/Qwen3-8B-Base",
            "Qwen/Qwen3-14B-Base",
            "Qwen/Qwen3-30B-A3B-Base",
        ],
        "instruct": [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-30B-A3B",
        ],
    },
    "smollm3": {
        "base": [
            "HuggingFaceTB/SmolLM3-3B-Base",  # base :contentReference[oaicite:8]{index=8}
        ],
        "instruct": [
            "HuggingFaceTB/SmolLM3-3B",  # instruct/chat :contentReference[oaicite:9]{index=9}
        ],
        # (no "think" key; SmolLM3 toggles think/no_think in prompting)
    },
}


# -----------------------------------------------------------------------------
# Metric suite runner
# -----------------------------------------------------------------------------


def run_all_metrics(
    X: Tensor,
    Y: Tensor,
    n_loc: int = 20,
    k_knn: int = 10,
    locinf_topk: Optional[int] = 200,
) -> List[MetricResult]:
    """
    Run a default suite of global + local metrics on aligned embeddings.

    Parameters
    ----------
    X, Y : [N, d_x], [N, d_y]
        Row-aligned embedding matrices (same canonical tokens).
    n_loc : int
        n for LocSim-n.
    k_knn : int
        k for kNN Jaccard.
    locinf_topk : Optional[int]
        topk approximation for LocSim-inf (None means exact over all tokens).
    """
    cache: Dict[str, Any] = {}
    results: List[MetricResult] = []

    print("    [*] Running embedding similarity metrics...")
    results.append(metric_distance_matrix_corr(X, Y, cache))
    print("        - distance matrix correlation")
    results.append(metric_cka(X, Y, cache))
    print("        - CKA")
    results.append(metric_procrustes_residual(X, Y, cache))
    print("        - Procrustes residual")
    results.append(metric_knn_jaccard(X, Y, cache, k=k_knn))
    print("        - kNN Jaccard")
    results.append(metric_locsim_n(X, Y, cache, n=n_loc))
    print("        - LocSim-n")
    results.append(metric_locsim_n(X, Y, cache, n=5))
    print("        - LocSim-5")
    results.append(metric_locsim_n(X, Y, cache, n=100))
    print("        - LocSim-100")
    # results.append(metric_locsim_inf(X, Y, cache))
    # print("        - LocSim-inf (exact)")
    if locinf_topk is not None:
        results.append(metric_locsim_inf(X, Y, cache, topk=locinf_topk))
        print(f"        - LocSim-inf (topk={locinf_topk})")

        results.append(metric_locsim_inf(X, Y, cache, topk=20))
        print(f"        - LocSim-inf (topk=20)")

        results.append(metric_locsim_inf(X, Y, cache, topk=1000))
        print(f"        - LocSim-inf (topk=1000)")

    return results


# -----------------------------------------------------------------------------
# Top-level evaluation helpers
# -----------------------------------------------------------------------------


def load_model_and_tokenizer(model_id: str) -> Tuple[PreTrainedModel, Any]:
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers is not installed.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def evaluate_input_vs_output_for_model(
    model: PreTrainedModel,
    tokenizer,
    model_id: str,
    family: str,
    variant_kind: VariantKind,
    max_tokens: int = 5000,
    n_loc: int = 20,
    k_knn: int = 10,
    locinf_topk: Optional[int] = 200,
) -> List[Dict[str, Any]]:
    """
    For a single model with untied embeddings:
    - sample a subset of tokens,
    - compute input vs output embedding similarity metrics,
    - return a list of flat result dicts for logging / JSON.
    """
    token_strings, token_ids = build_token_subset_single(tokenizer, max_tokens=max_tokens)
    token_ids = token_ids.to(device)

    in_emb_full = get_input_embeddings_tensor(model).to(device)
    out_emb_full = get_output_embeddings_tensor(model)
    if out_emb_full is None:
        raise ValueError(f"Model {model_id} appears to have no output embeddings.")
    out_emb_full = out_emb_full.to(device)

    X = in_emb_full[token_ids]
    Y = out_emb_full[token_ids]

    results = run_all_metrics(X, Y, n_loc=n_loc, k_knn=k_knn, locinf_topk=locinf_topk)
    flat_results: List[Dict[str, Any]] = []

    for res in results:
        row: Dict[str, Any] = {
            "family": family,
            "variant": variant_kind,
            "model_id": model_id,
            "comparison": "input_vs_output",
            "num_tokens": int(X.shape[0]),
            "dim_in": int(X.shape[1]),
            "dim_out": int(Y.shape[1]),
            "metric": res.name,
            "primary": res.primary,
        }
        for k, v in res.raw.items():
            row[f"raw_{k}"] = v
        if res.meta is not None:
            for k, v in res.meta.items():
                row[f"meta_{k}"] = v
        flat_results.append(row)

    return flat_results


# -----------------------------------------------------------------------------
# Full family loop (base + instruct, skip tied embeddings)
# -----------------------------------------------------------------------------


def run_full_suite(
    output_path: str = "embedding_similarity_results.jsonl",
    max_tokens: int = 5000,
    n_loc: int = 20,
    k_knn: int = 10,
    locinf_topk: Optional[int] = 200,
) -> List[Dict[str, Any]]:
    """
    Loop over Qwen3, Llama3, Gemma3 (base + instruct variants), skip tied models,
    compute input vs output metrics per model, and write JSONL results.

    Returns
    -------
    all_rows : list[dict]
        All result rows (including skipped markers) as dictionaries.
    """
    all_rows: List[Dict[str, Any]] = []

    for family_name, spec in MODEL_FAMILIES.items():
        for variant_kind_str, model_ids in spec.items():
            variant_kind: VariantKind = variant_kind_str  # type: ignore
            print(f"\n=== Family: {family_name} | variant: {variant_kind} ===", flush=True)

            for model_id in model_ids:
                print(f"\n[+] Loading model: {model_id}", flush=True)
                try:
                    model, tokenizer = load_model_and_tokenizer(model_id)
                except Exception as e:
                    print(f"    [!] Failed to load {model_id}: {e}")
                    all_rows.append(
                        {
                            "family": family_name,
                            "variant": variant_kind,
                            "model_id": model_id,
                            "comparison": "input_vs_output",
                            "skipped": True,
                            "reason": f"load_error: {e}",
                        }
                    )
                    continue

                # Filter out models with tied embeddings
                if embeddings_are_tied(model):
                    print(f"    [↷] Skipping {model_id}: tied input/output embeddings.")
                    all_rows.append(
                        {
                            "family": family_name,
                            "variant": variant_kind,
                            "model_id": model_id,
                            "comparison": "input_vs_output",
                            "skipped": True,
                            "reason": "tied_embeddings",
                        }
                    )
                    del model
                    del tokenizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                print(f"    [✓] Evaluating untied model {model_id}...")

                try:
                    model_rows = evaluate_input_vs_output_for_model(
                        model=model,
                        tokenizer=tokenizer,
                        model_id=model_id,
                        family=family_name,
                        variant_kind=variant_kind,
                        max_tokens=max_tokens,
                        n_loc=n_loc,
                        k_knn=k_knn,
                        locinf_topk=locinf_topk,
                    )
                    all_rows.extend(model_rows)
                    print_summary(model_rows)
                except Exception as e:
                    print(f"    [!] Error during evaluation of {model_id}: {e}")
                    all_rows.append(
                        {
                            "family": family_name,
                            "variant": variant_kind,
                            "model_id": model_id,
                            "comparison": "input_vs_output",
                            "skipped": True,
                            "reason": f"eval_error: {e}",
                        }
                    )

                # Free GPU memory between models
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Write results to JSONL
    print(f"\nWriting results to {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print("Done.")
    return all_rows


# -----------------------------------------------------------------------------
# Pretty-print summary to facilitate research questions
# -----------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if isinstance(x, (float, int)):
        return f"{x:.4f}"
    return str(x)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    """
    Print a structured summary of results:

    - Per (family, variant) aggregate mean primary per metric across models.
    - Per model, per metric primary + key raw values.
    """
    # Filter out skipped rows, split metrics vs skipped
    metric_rows = [r for r in rows if not r.get("skipped", False)]
    skipped_rows = [r for r in rows if r.get("skipped", False)]

    if skipped_rows:
        print("\n=== Skipped models ===")
        for r in skipped_rows:
            print(f"  {r['family']} | {r['variant']} | {r['model_id']} -> {r['reason']}")

    if not metric_rows:
        print("\nNo metric rows to summarize.")
        return

    # Aggregate by (family, variant, metric)
    agg: Dict[Tuple[str, str, str], List[float]] = {}
    for r in metric_rows:
        key = (r["family"], r["variant"], r["metric"])
        agg.setdefault(key, []).append(float(r["primary"]))

    print("\n=== Family / Variant aggregate means (primary) ===")
    # sort nicely
    for family, variant, metric in sorted(agg.keys()):
        vals = agg[(family, variant, metric)]
        mean_val = sum(vals) / len(vals)
        print(f"  {family:8s} | {variant:8s} | {metric:20s} -> mean={mean_val:.4f} (n={len(vals)})")

    # Group by model to print detailed tables
    by_model: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in metric_rows:
        key = (r["family"], r["variant"], r["model_id"])
        by_model.setdefault(key, []).append(r)

    print("\n=== Per-model metric tables ===")
    for family, variant, model_id in sorted(by_model.keys()):
        rows_m = by_model[(family, variant, model_id)]
        print(f"\n--- {family} | {variant} | {model_id} ---")
        # Simple table header
        print(f"{'metric':25s} {'primary':>10s}  raw")
        print("-" * 60)
        for r in sorted(rows_m, key=lambda x: x["metric"]):
            metric = r["metric"]
            primary = r["primary"]
            # pick a couple of interesting raw keys
            raw_keys = [k for k in r.keys() if k.startswith("raw_")]
            # stable order
            raw_keys.sort()
            raw_str_parts = []
            for rk in raw_keys:
                raw_str_parts.append(f"{rk.replace('raw_', '')}={_fmt(r[rk])}")
            raw_str = ", ".join(raw_str_parts)
            print(f"{metric:25s} {_fmt(primary):>10s}  {raw_str}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    rows = run_full_suite()
    print_summary(rows)
