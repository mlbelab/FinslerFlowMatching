from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .coupling import sinkhorn_coupling
from .models import VelocityNet


def carre_du_champ(X: np.ndarray, k: int = 256, k_bw: int = 8, gamma: float = 1.0,
                   rank: int = 0, sigma_min: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    k = int(min(k, n - 1))
    dist, idx = cKDTree(X).query(X, k=k + 1)
    eps = np.maximum(dist[:, min(k_bw, k)], 1e-12)
    w = np.exp(-dist ** 2 / (eps[:, None] * eps[idx]))
    p = w / w.sum(1, keepdims=True)
    nb = X[idx]
    mean = np.einsum("nk,nkd->nd", p, nb)
    dev = nb - mean[:, None, :]
    lam, vec = np.linalg.eigh(np.einsum("nk,nkd,nke->nde", p, dev, dev))
    nn = dist[:, 1]
    nn2 = nn ** 2 / 9.0
    scale = gamma * np.minimum(nn2, np.quantile(nn2, 0.9)) / np.maximum(lam[:, -1], 1e-300)
    lam = np.clip(lam, 0.0, None) * scale[:, None]
    if 0 < rank < d:
        lam[:, : d - rank] = 0.0
    floor = sigma_min * float(np.median(nn[nn > 0]))
    return vec, lam + floor ** 2


def _subset_idx(mask: np.ndarray, max_pts: int, gen: torch.Generator,
                device: torch.device) -> torch.Tensor:
    idx = torch.as_tensor(np.where(mask)[0], device=device)
    if len(idx) > max_pts:
        idx = idx[torch.randperm(len(idx), generator=gen, device=device)[:max_pts]]
    return idx


def _sym(V: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bj,bkj->bik", V, L, V)


def train_cdcfm(X: np.ndarray, p0: np.ndarray, p1: np.ndarray, vec: np.ndarray,
                lam: np.ndarray, cfg, device: torch.device, dtype: torch.dtype,
                blur_frac: float = 0.1, max_pts: int = 200, seed: int = 0):
    Xt = torch.as_tensor(X, dtype=dtype, device=device)
    V = torch.as_tensor(vec, dtype=dtype, device=device)
    L = torch.as_tensor(lam, dtype=dtype, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 91)
    i0 = _subset_idx(p0, max_pts, gen, device)
    i1 = _subset_idx(p1, max_pts, gen, device)
    pi = sinkhorn_coupling(torch.cdist(Xt[i0], Xt[i1]) ** 2, blur_frac=blur_frac)
    joint = (pi / pi.sum()).reshape(-1)
    n1 = len(i1)
    tol = 1e-10 * float(L.max())
    d = Xt.shape[1]
    torch.manual_seed(cfg.seed + 2)
    v_net = VelocityNet(d, width=cfg.width, depth=cfg.depth).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(v_net.parameters(), lr=cfg.lr)
    draw = torch.Generator(device=device).manual_seed(seed + 17)
    hist = []
    for _ in range(cfg.phase2_iters):
        kk = torch.multinomial(joint, cfg.batch_size, replacement=True, generator=draw)
        a, b = i0[kk // n1], i1[kk % n1]
        V0, L0 = V[a], L[a]
        root0 = _sym(V0, L0.clamp_min(0.0).sqrt())
        iroot0 = _sym(V0, torch.where(L0 > tol, L0.clamp_min(tol).rsqrt(),
                                      torch.zeros_like(L0)))
        M = root0 @ _sym(V[b], L[b]) @ root0
        mL, mV = torch.linalg.eigh(0.5 * (M + M.transpose(1, 2)))
        Bmap = iroot0 @ _sym(mV, mL.clamp_min(0.0).sqrt()) @ iroot0
        z = torch.randn(cfg.batch_size, d, generator=draw, device=device, dtype=dtype)
        delta = torch.einsum("bij,bj->bi", root0, z)
        bdelta = torch.einsum("bij,bj->bi", Bmap, delta)
        t = torch.rand(cfg.batch_size, 1, generator=draw, device=device, dtype=dtype)
        x0, x1 = Xt[a], Xt[b]
        x_t = (1 - t) * x0 + t * x1 + (1 - t) * delta + t * bdelta
        target = x1 - x0 + bdelta - delta
        loss = ((v_net(t, x_t) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), cfg.grad_clip)
        opt.step()
        hist.append(float(loss))
    p = joint.reshape(len(i0), n1)
    ent = float(-(p * torch.log(p.clamp_min(1e-30))).sum() / np.log(p.numel()))
    return v_net, hist, {"n0": int(len(i0)), "n1": int(n1), "pi_entropy_frac": ent}
