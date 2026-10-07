"""Curly Flow Matching (Petrović et al., NeurIPS 2025) as a benchmark method.

This is a faithful **standalone adapter** around the authors' own single-marginal
code — their core primitives (``get_xt_xt_dot``, ``get_u_xt``, ``coupling``) and network
(``MLP``) are used directly, copied verbatim into :mod:`scripts.core.vendor` so this
module needs no checkout of their release on the machine — and it
replicate the two-stage ``DeepCycleModule`` training recipe (their single-marginal
cell module).  We deliberately do NOT use their Lightning/Hydra orchestration so the
model trains on the *same* :class:`~scripts.core.datasets.Dataset` (p₀→p₁ over the
50-D PCA cloud) and is scored by the *same* W2 (interior + final τ-bands) and fate
concordance as our Path A / Path B — an apples-to-apples benchmark.

Curly-FM's algorithm (single marginal):
  Stage 1  — a geodesic net φ defines μ_t = (1-t)x0 + t·x1 + t(1-t)·φ([x0,x1,t]); its
             time-derivative μ̇_t is trained to match the kNN-smoothed RNA-velocity
             reference field u(x) (cosine + L2 loss).  This is the curl-aware analogue
             of our Finsler geodesic — it bends the interpolant along the *observed
             velocity* rather than along the Randers metric derived from P.
  Stage 2  — the transport plan is the coupling that minimises drift discrepancy
             ‖μ̇_t − u‖² (exact ``linear_sum_assignment``); a flow net v_θ is then
             distilled with ‖v_θ − μ̇_t‖².
Inference is the deterministic ODE dx/dt = v_θ(t,x) from p₀ (their DeepCycle setting).

Two entry points, one training loop
-----------------------------------
:func:`train_curly_from_field` is the general one: it takes the reference positions
and the reference field explicitly.  :func:`train_curly` is the Pancreas convenience
wrapper that pulls the RNA velocity projected to PCA (``PancreasMeta.velocity_pca``)
out of the metadata — exactly the input Curly-FM expects on real data.

On the synthetic suite there is no measured velocity, so the arm supplies
b̂(x_i) = the kernel-smoothed empirical flux read out of P
(:meth:`~scripts.core.geometry.FinslerGeometry.drift`).  That is the same field our
Finsler geometry is built from, which makes the comparison information-matched: both
methods get P and nothing else.  Note the consequence for scoring — the velocity
metric ``cos(v_θ, b̂)`` uses that same field as its yardstick, so Curly-FM regresses
directly onto the reference and its cosine is near-ceiling *by construction*; read
its W2 columns, not its cosine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# the authors' own code (single-marginal primitives + network), copied verbatim into
# scripts.core.vendor so this module needs no checkout on the machine -- see that
# package's docstring for the provenance of the vendored release
from .vendor.curly_release import MLP, coupling, get_u_xt, get_xt_xt_dot


@dataclass
class CurlyConfig:
    geo_iters: int = 2500      # stage-1 (geodesic net) iterations
    vel_iters: int = 2500      # stage-2 (flow net) iterations
    batch_size: int = 256      # stage-1 batch
    coupling_batch: int = 128  # stage-2 batch (all-pairs assignment cost ~ b²)
    width: int = 256
    k: int = 20                # kNN for the reference-velocity estimator
    # Reference (DeepCycle) velocity-data hyperparameters — configs/model/deepcycle.yaml.
    # With alpha=1 the L2 term ‖u − α·μ̇_t‖² is unsatisfiable when the reference velocity
    # |u| ≪ |x1−x0| (it just straightens the path); alpha≈0.1 lets the direction (cosine)
    # term bend the geodesic along the field.  This is Curly-FM's *true* setting here.
    sigma: float = 0.01        # interpolant noise scale
    alpha: float = 0.1         # magnitude weight in the geo loss
    lr: float = 1e-3
    log_every: int = 250       # 0 silences the loop (the synth suite runs ~100 cells)
    seed: int = 0
    # Which network the two stages train.  "release" is their own ``MLP`` — three SELU
    # layers with ``t`` concatenated raw — and is the default, so every published number
    # in this repository stays measured on their architecture.  "shared" swaps in the
    # engine's :class:`~scripts.core.models.PhiNet` / ``VelocityNet``, which is what the
    # notebooks use: there all five arms must differ by method and not by capacity, and
    # this arm is the only one that would otherwise carry its own class.  Everything
    # else about the arm -- the losses, the kNN field smoother, the drift-discrepancy
    # coupling, the two-stage recipe -- is theirs either way.
    net: str = "release"
    depth: int = 4             # hidden layers; only ``net="shared"`` reads it


class CurlyVelAdapter(torch.nn.Module):
    """Wrap Curly-FM's flow MLP v(cat[x,t]) as v_net(t, x) — the interface our
    ``integrate_flow`` / ``evaluate.loto_wasserstein`` expect (so the benchmark is
    scored by the identical harness as Path A/B)."""

    def __init__(self, mlp: torch.nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return self.mlp(torch.cat([x, t], dim=-1))


class _SharedNetAdapter(torch.nn.Module):
    """Present a shared ``PhiNet`` / ``VelocityNet`` under Curly-FM's calling convention.

    Their primitives take the network as one callable of a single concatenated tensor
    with ``t`` **last** — ``geodesic_model(cat([x0, x1, t], -1))`` in ``get_xt`` — while
    ours take ``(t, x0, x1)`` and embed ``t`` sinusoidally.  This adapter is the whole
    difference; no loss, coupling or smoother is touched.

    It also flattens, because ``coupling`` evaluates the geodesic net on a rank-3
    ``(B, B, ·)`` grid of all pairs while the training loops evaluate it at rank 2, and
    the engine nets assert a ``(B, 1)`` time column.
    """

    def __init__(self, net: torch.nn.Module, d: int, n_inputs: int):
        super().__init__()
        self.net, self.d, self.n_inputs = net, int(d), int(n_inputs)

    def forward(self, z: Tensor) -> Tensor:
        assert z.shape[-1] == self.n_inputs * self.d + 1, (
            f"expected {self.n_inputs} x {self.d} coordinates plus t, got {z.shape}")
        flat = z.reshape(-1, z.shape[-1])
        xs = torch.split(flat[:, :-1], self.d, dim=-1)
        out = self.net(flat[:, -1:], *xs)
        return out.reshape(*z.shape[:-1], out.shape[-1])


def _curly_nets(cfg: "CurlyConfig", d: int, device, dtype):
    """The stage-1 geodesic net and the stage-2 flow net, under ``cfg.net``."""
    if cfg.net == "release":
        return (MLP(dim=2 * d, out_dim=d, w=cfg.width, time_varying=True).to(device, dtype),
                MLP(dim=d, out_dim=d, w=cfg.width, time_varying=True).to(device, dtype))
    assert cfg.net == "shared", f"cfg.net must be 'release' or 'shared', got {cfg.net!r}"
    from .models import PhiNet, VelocityNet  # local: keeps the release path import-free
    geo = PhiNet(d, width=cfg.width, depth=cfg.depth).to(device, dtype)
    vel = VelocityNet(d, width=cfg.width, depth=cfg.depth).to(device, dtype)
    return _SharedNetAdapter(geo, d, 2), _SharedNetAdapter(vel, d, 1)


def _sample(idx: Tensor, X: Tensor, batch: int, gen: torch.Generator) -> Tensor:
    sel = idx[torch.randint(0, len(idx), (batch,), generator=gen, device=X.device)]
    return X[sel]


def _tail(hist: list[float], n: int = 100) -> float:
    """Mean of the last ``n`` iterations — matches the convergence report of our arms."""
    return float(np.mean(hist[-n:])) if hist else float("nan")


def train_curly_from_field(
    X_ref: np.ndarray,
    velocity: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    device,
    dtype,
    cfg: CurlyConfig | None = None,
) -> tuple[CurlyVelAdapter, dict]:
    """Two-stage Curly-FM on an explicit reference field; returns the flow adapter.

    Parameters
    ----------
    X_ref, velocity
        ``(N, d)`` reference positions and the field sampled at them.  Curly-FM's own
        adaptive-bandwidth kNN smoother (:func:`get_u_xt`, k = ``cfg.k``) extends this
        per-cell field to the off-lattice points ``x_t`` the interpolant visits — the
        step that turns per-cell RNA velocity into a field u(x) in their pipeline.
    src_idx, tgt_idx
        Integer indices into ``X_ref`` giving the p₀ and p₁ marginals.  Passing indices
        rather than separate arrays keeps the reference cloud and the transport
        endpoints in one coordinate frame, which :func:`get_u_xt` requires.
    """
    cfg = cfg or CurlyConfig()
    X_ref = np.asarray(X_ref)
    velocity = np.asarray(velocity)
    assert X_ref.ndim == 2, f"X_ref must be (N, d), got {X_ref.shape}"
    assert velocity.shape == X_ref.shape, (
        f"velocity {velocity.shape} must match X_ref {X_ref.shape}")
    assert len(src_idx) > 0 and len(tgt_idx) > 0, "empty source/target marginal"
    assert X_ref.shape[0] > cfg.k, (
        f"need more than k={cfg.k} reference cells for the kNN field estimator, "
        f"got {X_ref.shape[0]}")

    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    d = X_ref.shape[1]
    X = torch.as_tensor(X_ref, dtype=dtype, device=device)
    train_x = X                                             # reference positions
    train_vel = torch.as_tensor(velocity, dtype=dtype, device=device)
    src = torch.as_tensor(np.asarray(src_idx), device=device)
    tgt = torch.as_tensor(np.asarray(tgt_idx), device=device)

    geo, vel = _curly_nets(cfg, d, device, dtype)
    geo_opt = torch.optim.AdamW(geo.parameters(), lr=cfg.lr)
    vel_opt = torch.optim.Adam(vel.parameters(), lr=cfg.lr)
    say = (lambda m: print(m)) if cfg.log_every else (lambda m: None)

    # --- Stage 1: geodesic net matches the reference velocity field ---
    say("  -- Curly-FM stage 1 (geodesic net ↦ reference field) --")
    hist_geo: list[float] = []
    for it in range(cfg.geo_iters):
        x0 = _sample(src, X, cfg.batch_size, gen)
        x1 = _sample(tgt, X, cfg.batch_size, gen)
        t = torch.rand(cfg.batch_size, device=device, dtype=dtype)
        xt, mu_t_dot, _ = get_xt_xt_dot(t, x0, x1, geo, sigma=cfg.sigma)
        ut = get_u_xt(xt, train_x, train_vel, k=cfg.k)
        cos = 1.0 - F.cosine_similarity(ut, mu_t_dot).mean()
        l2 = torch.mean((ut - cfg.alpha * mu_t_dot) ** 2)
        loss = cos + l2
        geo_opt.zero_grad(); loss.backward(); geo_opt.step()
        hist_geo.append(float(loss))
        if cfg.log_every and (it % cfg.log_every == 0 or it == cfg.geo_iters - 1):
            say(f"    [curly-geo] iter {it:5d}  L={float(loss):.5f} "
                f"(cos={float(cos):.4f})")

    # --- Stage 2: drift-discrepancy coupling + flow distillation ---
    say("  -- Curly-FM stage 2 (drift-discrepancy coupling ↦ flow net) --")
    for p in geo.parameters():
        p.requires_grad_(False)
    hist_vel: list[float] = []
    for it in range(cfg.vel_iters):
        x0 = _sample(src, X, cfg.coupling_batch, gen)
        x1 = _sample(tgt, X, cfg.coupling_batch, gen)
        # Curly-FM transport plan: minimise ‖μ̇_t − u‖² via linear assignment
        x0, x1 = coupling(x0, x1, cfg.coupling_batch, train_x, train_vel, geo,
                          sigma=cfg.sigma)
        t = torch.rand(x0.shape[0], device=device, dtype=dtype)
        xt, mu_t_dot, _ = get_xt_xt_dot(t, x0, x1, geo, sigma=cfg.sigma)
        vt = vel(torch.cat([xt.detach(), t[:, None]], dim=-1))
        loss = torch.mean((vt - mu_t_dot.detach()) ** 2)
        vel_opt.zero_grad(); loss.backward(); vel_opt.step()
        hist_vel.append(float(loss))
        if cfg.log_every and (it % cfg.log_every == 0 or it == cfg.vel_iters - 1):
            say(f"    [curly-vel] iter {it:5d}  L_CFM={float(loss):.6f}")

    diag = {"geo_iters": cfg.geo_iters, "vel_iters": cfg.vel_iters,
            "loss_geo": _tail(hist_geo), "loss_vel": _tail(hist_vel),
            "alpha": cfg.alpha, "sigma": cfg.sigma, "k": cfg.k, "net": cfg.net,
            "n_ref": int(X_ref.shape[0])}
    return CurlyVelAdapter(vel).eval(), diag


def train_curly(ds, meta, device, dtype, cfg: CurlyConfig | None = None
                ) -> tuple[CurlyVelAdapter, dict]:
    """Pancreas convenience wrapper: reference field = RNA velocity in PCA space."""
    return train_curly_from_field(
        ds.X, meta.velocity_pca, np.where(ds.p0)[0], np.where(ds.p1)[0],
        device, dtype, cfg)
