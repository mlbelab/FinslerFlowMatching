"""Path A — Phase 2: geometry-aware optimal-transport coupling π*.

After Phase 1 has frozen the geodesic interpolant φ_{η*}, we no longer couple the
endpoints independently.  Instead we measure the *true manifold distance* between
every source/target pair by Monte-Carlo quadrature of the Finsler energy along the
learned geodesic, and solve a regularised (entropic) optimal-transport problem for
the coupling π*:

    C_ij = (1/K) Σ_{k=1..K} F( x_{t_k,η*}(x0_i, x1_j), ẋ_{t_k,η*}(x0_i, x1_j) )²   (cost)

    π* = argmin_{π ∈ Π(p0, p1)}  Σ_ij π_ij C_ij + 2σ² · KL(π ‖ p0 ⊗ p1).          (OT)

With uniform empirical marginals a = p0, b = p1 the KL-to-product regulariser is,
up to an additive constant, the negative Shannon entropy of π, so this is exactly
the Sinkhorn entropic-OT problem with blur (entropic regularisation) reg = 2σ².
We solve it with POT's ``ot.sinkhorn`` (already a project dependency).

Phase 3 (velocity distillation) then draws endpoint pairs (x0, x1) ~ π* instead of
from the independent product coupling: :class:`OTCoupler` exposes the same
``.sample(batch)`` interface as :class:`scripts.core.train.EndpointCoupler`, so the
existing distillation loop (``train.train_phase2``) is reused verbatim.
"""
from __future__ import annotations

import numpy as np
import ot  # POT: entropic optimal transport (Sinkhorn)
import torch
from torch import Tensor

from .geometry import FinslerGeometry
from .models import PhiNet
from .train import interpolant_and_velocity


# --------------------------------------------------------------------------- #
#  Geodesic manifold-distance cost matrix  C_ij
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _subset(X: Tensor, mask: np.ndarray, max_pts: int, gen: torch.Generator) -> Tensor:
    """Coordinates of the endpoint set ``mask``, subsampled to at most ``max_pts``."""
    idx = torch.as_tensor(np.where(mask)[0], device=X.device)
    if len(idx) > max_pts:
        sel = torch.randperm(len(idx), generator=gen, device=X.device)[:max_pts]
        idx = idx[sel]
    return X[idx]


def geodesic_cost_matrix(
    phi: PhiNet,
    geom: FinslerGeometry,
    X0: Tensor,
    X1: Tensor,
    K: int = 8,
    chunk: int = 8192,
) -> Tensor:
    """C_ij = (1/K) Σ_k F(x_{t_k}, ẋ_{t_k})²  over the frozen geodesic  ->  (n0, n1).

    We evaluate the Finsler energy at K midpoint quadrature nodes t_k = (k+½)/K and
    average.  Pairs are flattened to (n0·n1, d) and processed in chunks to bound the
    memory of the all-pairs interpolant evaluation.  ẋ needs ∂_t φ, so the frozen φ
    is evaluated under ``enable_grad`` (create_graph=False — no back-prop here).
    """
    n0, n1, d = X0.shape[0], X1.shape[0], X0.shape[1]
    device, dtype = X0.device, X0.dtype

    # all-pairs endpoint tensors, flattened:  x0rep[m] = X0[i], x1rep[m] = X1[j]
    x0rep = X0[:, None, :].expand(n0, n1, d).reshape(-1, d)     # (n0·n1, d)
    x1rep = X1[None, :, :].expand(n0, n1, d).reshape(-1, d)     # (n0·n1, d)
    M = x0rep.shape[0]

    t_nodes = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K   # midpoints
    cost = torch.zeros(M, device=device, dtype=dtype)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        a, b = x0rep[s:e], x1rep[s:e]
        acc = torch.zeros(e - s, device=device, dtype=dtype)
        for tk in t_nodes:
            t = tk.expand(e - s, 1)
            with torch.enable_grad():
                x_t, x_dot = interpolant_and_velocity(phi, t, a, b, create_graph=False)
            acc = acc + geom.F2(x_t.detach(), x_dot.detach())
        cost[s:e] = acc / K
    return cost.reshape(n0, n1)


# --------------------------------------------------------------------------- #
#  Entropic OT (Sinkhorn) coupling π*
# --------------------------------------------------------------------------- #
def sinkhorn_coupling(
    C: Tensor, sigma: float | None = None, blur_frac: float = 0.1,
    n_iter: int = 20000,
) -> Tensor:
    """π* = argmin_π ⟨π, C⟩ + 2σ² KL(π ‖ p0⊗p1), uniform marginals  ->  (n0, n1).

    reg = 2σ² is the entropic blur.  If ``sigma`` is None we set it adaptively so
    that reg = ``blur_frac`` · median(C) — this keeps the Sinkhorn problem well
    conditioned regardless of the dataset's absolute Finsler-energy scale (the
    median cost varies a lot between Arch / Cycle / Bifurcation).

    ``n_iter`` is a generous cap rather than a budget: POT stops as soon as the
    marginal violation drops below its ``stopThr`` (1e-9), which for these problem
    sizes is typically a few hundred iterations.  It was raised from 5000 because a
    small ``blur_frac`` (a sharp coupling — exactly what the sweep tends to prefer)
    makes the iteration converge slowly and the old cap returned a coupling whose
    marginals were still visibly off.
    """
    Cnp = C.detach().cpu().numpy().astype(np.float64)
    n0, n1 = Cnp.shape
    a = np.full(n0, 1.0 / n0)
    b = np.full(n1, 1.0 / n1)
    if sigma is None:
        reg = float(blur_frac * np.median(Cnp) + 1e-12)
    else:
        reg = 2.0 * float(sigma) ** 2
    # log-domain Sinkhorn: stable even for small reg (avoids kernel underflow /
    # divide-by-zero that plagues the vanilla exp-domain iteration).
    pi = ot.sinkhorn(a, b, Cnp, reg, method="sinkhorn_log", numItermax=n_iter)
    return torch.as_tensor(pi, dtype=C.dtype, device=C.device)


# --------------------------------------------------------------------------- #
#  Sampler over the coupling π* (drop-in for EndpointCoupler)
# --------------------------------------------------------------------------- #
class OTCoupler:
    """Sample endpoint pairs (x0, x1) ~ π* — same interface as EndpointCoupler.

    π* is flattened into a joint categorical over the (i, j) source/target index
    grid; each ``sample`` call draws ``batch`` pairs with replacement.
    """

    def __init__(self, X0: Tensor, X1: Tensor, pi: Tensor, seed: int = 0):
        assert pi.shape == (X0.shape[0], X1.shape[0]), "π shape must match (n0, n1)"
        self.X0 = X0
        self.X1 = X1
        self.n1 = X1.shape[0]
        self.device = X0.device
        flat = pi.reshape(-1)
        self.joint = flat / flat.sum()                         # normalised joint p(i,j)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

    def sample(self, batch: int) -> tuple[Tensor, Tensor]:
        idx = torch.multinomial(self.joint, batch, replacement=True, generator=self.gen)
        i = torch.div(idx, self.n1, rounding_mode="floor")
        j = idx % self.n1
        return self.X0[i], self.X1[j]


# --------------------------------------------------------------------------- #
#  Convenience builder: Phase 2 end-to-end
# --------------------------------------------------------------------------- #
def build_ot_coupler(
    phi: PhiNet,
    geom: FinslerGeometry,
    X: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    K: int = 8,
    sigma: float | None = None,
    blur_frac: float = 0.1,
    max_pts: int = 200,
    seed: int = 0,
) -> tuple[OTCoupler, dict]:
    """Run Path-A Phase 2: cost matrix -> Sinkhorn π* -> :class:`OTCoupler`.

    Source/target clouds are subsampled to ``max_pts`` each to keep the all-pairs
    geodesic cost and the Sinkhorn solve tractable.  Returns the coupler plus a
    small diagnostics dict (cost/coupling statistics) for logging.  ``blur_frac`` sets
    the entropic regulariser to ``blur_frac·median(C)`` when ``sigma`` is unset —
    smaller ⇒ a sharper (less diffuse) coupling, which tightens the terminal marginal.
    """
    Xt = torch.as_tensor(X, dtype=dtype, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 91)
    X0 = _subset(Xt, p0, max_pts, gen)
    X1 = _subset(Xt, p1, max_pts, gen)

    phi.eval()
    C = geodesic_cost_matrix(phi, geom, X0, X1, K=K)
    pi = sinkhorn_coupling(C, sigma=sigma, blur_frac=blur_frac)

    # transport efficiency diagnostic: how peaked is π vs the uniform product?
    joint = (pi / pi.sum()).reshape(-1)
    entropy = float(-(joint * torch.log(joint + 1e-30)).sum())
    max_entropy = float(np.log(joint.numel()))
    diag = {
        "n0": int(X0.shape[0]),
        "n1": int(X1.shape[0]),
        "C_median": float(C.median()),
        "C_min": float(C.min()),
        "pi_entropy_frac": entropy / max_entropy,   # 1 = uniform/product, <1 = focused
    }
    return OTCoupler(X0, X1, pi, seed=seed), diag


# --------------------------------------------------------------------------- #
#  Standard (Euclidean) OT coupling — the baseline matching for CFM / MFM
# --------------------------------------------------------------------------- #
def build_standard_ot_coupler(
    X: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    sigma: float | None = None,
    blur_frac: float = 0.1,
    max_pts: int = 200,
    seed: int = 0,
) -> tuple[OTCoupler, dict]:
    """Standard entropic OT with the **Euclidean** cost ``C_ij = ‖x0_i − x1_j‖²``.

    This is the geometry-free matching used by the baseline methods (CFM, MFM): it
    couples the source/target clouds by plain squared-Euclidean optimal transport —
    no Finsler geodesic cost.  Same Sinkhorn solver and :class:`OTCoupler` sampler as
    :func:`build_ot_coupler`, so it is a drop-in for Phase 3 / ``train_cfm``.

    ``blur_frac`` is forwarded to :func:`sinkhorn_coupling` exactly as in
    :func:`build_ot_coupler` (it used to be pinned at the 0.1 default), so the
    baselines' entropic blur is tunable on the same footing as FFM's.
    """
    Xt = torch.as_tensor(X, dtype=dtype, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 91)
    X0 = _subset(Xt, p0, max_pts, gen)
    X1 = _subset(Xt, p1, max_pts, gen)

    C = torch.cdist(X0, X1) ** 2                              # squared-Euclidean cost
    pi = sinkhorn_coupling(C, sigma=sigma, blur_frac=blur_frac)

    joint = (pi / pi.sum()).reshape(-1)
    entropy = float(-(joint * torch.log(joint + 1e-30)).sum())
    max_entropy = float(np.log(joint.numel()))
    diag = {
        "n0": int(X0.shape[0]),
        "n1": int(X1.shape[0]),
        "C_median": float(C.median()),
        "C_min": float(C.min()),
        "pi_entropy_frac": entropy / max_entropy,
    }
    return OTCoupler(X0, X1, pi, seed=seed), diag
