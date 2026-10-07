"""Path A: the three phases, and the deterministic pushforward.

1. **Geodesic interpolant.**  Fit ``phi`` by minimising ``E F(x_t, xdot_t)^2``.
2. **Geometry-aware OT.**  ``C_ij`` = the mean Finsler energy along the *frozen*
   interpolant at ``K`` quadrature nodes, then entropic Sinkhorn for ``pi*``.
3. **Distillation.**  Regress ``v_theta`` onto ``xdot_t`` under ``pi*``, then RK4 from
   ``p_0``.

Phase 2 is where the geometry actually decides something: the interpolant only says what
a path between a *given* pair costs, and the coupling is what turns that into a choice of
which pairs to transport at all.
"""
from __future__ import annotations

import numpy as np
import ot                                              # POT: entropic + exact OT
import torch

from scripts.method.config import Run
from scripts.method.metric import Metric
from scripts.method.nets import Coupler, GeoPathNet, VelocityNet, interpolant


def train_geodesic(metric: Metric, coupler: Coupler, seed: int, cfg: Run,
                   iters: int | None = None, log_every: int = 250):
    """Phase 1: fit the interpolant by minimising the Finsler energy."""
    iters = cfg.geo_iters if iters is None else iters
    torch.manual_seed(seed)
    phi = GeoPathNet(metric.d, **cfg.net_kw).to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(phi.parameters(), lr=cfg.geo_lr)
    for it in range(iters):
        x0, x1 = coupler.sample(cfg.batch)
        t = torch.rand(cfg.batch, 1, **cfg.torch_kw)
        x_t, x_dot = interpolant(phi, t, x0, x1, create_graph=True)
        loss = metric.F2(x_t, x_dot).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(phi.parameters(), cfg.grad_clip)
        opt.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"    iter {it:5d}  L_geo {float(loss):.5f}")
    return phi


def finsler_cost(phi, metric: Metric, X0, X1, cfg: Run,
                 K: int | None = None, chunk: int | None = None):
    """``C_ij = (1/K) sum_k F(x_{t_k}, xdot_{t_k})^2`` along the frozen interpolant."""
    K = cfg.ot_k if K is None else K
    chunk = cfg.ot_chunk if chunk is None else chunk
    n0, n1 = len(X0), len(X1)
    a = X0[:, None, :].expand(n0, n1, metric.d).reshape(-1, metric.d)
    b = X1[None, :, :].expand(n0, n1, metric.d).reshape(-1, metric.d)
    nodes = (torch.arange(K, dtype=torch.float64) + 0.5) / K          # midpoints
    C = torch.empty(n0 * n1, **cfg.torch_kw)
    for s in range(0, n0 * n1, chunk):
        e = min(s + chunk, n0 * n1)
        acc = torch.zeros(e - s, **cfg.torch_kw)
        for tk in nodes.tolist():
            t = torch.full((e - s, 1), tk, **cfg.torch_kw)
            with torch.enable_grad():
                x_t, x_dot = interpolant(phi, t, a[s:e], b[s:e], create_graph=False)
            acc = acc + metric.F2(x_t.detach(), x_dot.detach())
        C[s:e] = acc / K
    return C.reshape(n0, n1)


def build_ot(phi, metric: Metric, X0_t, X1_t, seed: int, cfg: Run,
             max_pts: int | None = None):
    """Phase 2: entropic OT under the Finsler cost, reg a fraction of the median cost.

    Returns the coupler and a diagnostic dict.  ``pi_entropy_frac`` is the coupling's
    entropy against the uniform product: 1 means the geometry bought no structure at all,
    0 means a permutation.
    """
    max_pts = cfg.ot_max_pts if max_pts is None else max_pts
    g = torch.Generator(device="cpu").manual_seed(seed + 91)
    i0 = torch.randperm(len(X0_t), generator=g)[:max_pts].to(cfg.device)
    i1 = torch.randperm(len(X1_t), generator=g)[:max_pts].to(cfg.device)
    A, B = X0_t[i0], X1_t[i1]
    C = finsler_cost(phi, metric, A, B, cfg)
    Cn = C.detach().double().cpu().numpy()
    reg = float(cfg.ot_blur_frac * np.median(Cn) + 1e-12)
    pi = ot.sinkhorn(np.full(len(A), 1 / len(A)), np.full(len(B), 1 / len(B)), Cn, reg,
                     method="sinkhorn_log", numItermax=cfg.ot_n_iter)
    pi = torch.as_tensor(pi, **cfg.torch_kw)
    p = pi / pi.sum()
    ent = float(-(p * torch.log(p.clamp_min(1e-30))).sum() / np.log(p.numel()))
    return Coupler(A, B, pi=pi, seed=seed), {"C_median": float(C.median()), "reg": reg,
                                             "pi_entropy_frac": ent}


def train_cfm(phi, coupler: Coupler, d: int, seed: int, cfg: Run,
              iters: int | None = None, log_every: int = 250):
    """Phase 3: distil the frozen geodesic field under ``pi*`` into ``v_theta``."""
    iters = cfg.cfm_iters if iters is None else iters
    torch.manual_seed(seed + 1)
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    v_net = VelocityNet(d, **cfg.net_kw).to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(v_net.parameters(), lr=cfg.cfm_lr)
    for it in range(iters):
        x0, x1 = coupler.sample(cfg.batch)
        t = torch.rand(cfg.batch, 1, **cfg.torch_kw)
        with torch.enable_grad():
            x_t, x_dot = interpolant(phi, t, x0, x1, create_graph=False)
        loss = ((v_net(t.detach(), x_t.detach()) - x_dot.detach()) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), cfg.grad_clip)
        opt.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"    iter {it:5d}  L_CFM {float(loss):.5f}")
    for p in phi.parameters():
        p.requires_grad_(True)
    return v_net


@torch.no_grad()
def integrate(v_net, x0, cfg: Run, n_steps: int | None = None):
    """RK4 on ``dx/dt = v_theta(t, x)`` from 0 to 1 -> ``(n_steps+1, B, d)``."""
    n_steps = cfg.n_steps if n_steps is None else n_steps
    dt = 1.0 / n_steps
    x = x0.clone()
    traj = [x.clone()]
    for k in range(n_steps):
        t = torch.full((len(x), 1), k * dt, device=x.device, dtype=x.dtype)
        k1 = v_net(t, x)
        k2 = v_net(t + dt / 2, x + dt / 2 * k1)
        k3 = v_net(t + dt / 2, x + dt / 2 * k2)
        k4 = v_net(t + dt, x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.clone())
    return torch.stack(traj)


def path_a(metric: Metric, X0_t, X1_t, seed: int, cfg: Run, verbose: bool = False):
    """Phases 1–3 at one ``(metric, seed)`` -> the pushforward of ``p_0``.

    Stops *before* any evaluation: the tuning pass calls this one, and a selection
    routine with a test number already in hand is one edit away from ranking on it.
    ``phi`` and ``coupler`` are returned because Path B reuses them frozen.
    """
    log = 250 if verbose else 0
    phi = train_geodesic(metric, Coupler(X0_t, X1_t, seed=seed), seed, cfg, log_every=log)
    coupler, diag = build_ot(phi, metric, X0_t, X1_t, seed, cfg)
    v_net = train_cfm(phi, coupler, metric.d, seed, cfg, log_every=log)
    traj = integrate(v_net, X0_t, cfg).cpu().numpy().astype(np.float64)
    return {"phi": phi, "coupler": coupler, "v_net": v_net, "traj": traj, "ot": diag}


def cuda_warm_up(metric: Metric, X0_t, X1_t, cfg: Run, iters: int = 20) -> bool:
    """Twenty discarded iterations, so every grid point is scored under the same numerics.

    A CUDA process resolves its kernel selection and lazy initialisation on the first
    training run, so that run is scored under slightly different numerics from every
    later one.  Running a throwaway fit first puts the whole sweep on one footing,
    which matters most at the least-conditioned corner (smallest rho with the strongest
    1-form, where ``C_rho`` is nearest singular).

    Pass the *least-conditioned* metric, since it is the one that sets the requirement.
    Nothing here is kept.  Returns whether it ran.
    """
    if cfg.device.type != "cuda":
        return False
    phi = train_geodesic(metric, Coupler(X0_t, X1_t, seed=cfg.tune_seed), cfg.tune_seed,
                         cfg, iters=iters, log_every=0)
    cp, _ = build_ot(phi, metric, X0_t, X1_t, cfg.tune_seed, cfg)
    integrate(train_cfm(phi, cp, metric.d, cfg.tune_seed, cfg, iters=iters, log_every=0),
              X0_t, cfg)
    return True
