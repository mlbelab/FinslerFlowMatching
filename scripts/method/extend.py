"""Other ways to extend the moment fields off the samples.

The paper extends ``m`` and ``C`` off the observed states with a Nadaraya-Watson smoother
whose bandwidth is the median positive nearest-neighbour distance (:class:`~scripts.
method.metric.TrueFW`), and notes that a network or an SVR would do as well.  Two
properties of that extension bear on the d = 2 instability: the window holds ~6
effective neighbours at d = 2 (hundreds above d = 10), and where the kernel has no mass
every moment decays to zero, ``a -> lambda``, so the metric is cheapest exactly where
there is no data -- which is where the withheld marginal sits.

Each object below changes **one** of those properties and inherits the rest from
``TrueFW``, so an arm built on it differs from the shipped metric by that change alone.
All of them take ``rho`` and ``lam`` as absolute values, so a caller can hold both at the
shipped metric's and vary nothing else:

``graph_denoise``  the per-point moments averaged over a kNN graph before smoothing;
                   bandwidth and far field untouched
``NeffFW``         the bandwidth chosen per query so the Gaussian weights carry a fixed
                   Kish effective sample size; the far field no longer collapses
``BarrierFW``      a conformal factor, 1 where the shipped kernel has mass and
                   ``phi_max`` where it has none; moments and window untouched
``StraightPhi``    the zero geodesic correction, so Phase 3 distils straight paths under
                   whatever coupling Phase 2 built

Nothing in the shipped pipeline imports this module.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from scripts.method.metric import TrueFW


def graph_denoise(X, vals, k: int = 16, steps: int = 5, alpha: float = 0.5) -> np.ndarray:
    """``vals`` diffused ``steps`` times over the symmetrised kNN graph of ``X``.

    One step is ``v <- (1 - alpha) v + alpha W v`` with ``W`` the row-normalised adjacency,
    i.e. ``(I - alpha L_rw)^steps`` -- the explicit heat-kernel form of graph-Laplacian
    smoothing.  Every step is a convex combination of neighbouring rows, so second moments
    stay PSD and nothing has to be projected back.
    """
    X = np.asarray(X, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    n = len(X)
    _, nn = cKDTree(X).query(X, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    A = sp.csr_matrix((np.ones(n * k), (rows, nn[:, 1:].reshape(-1))), shape=(n, n))
    A = A.maximum(A.T)
    W = sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A
    flat = vals.reshape(n, -1)
    for _ in range(steps):
        flat = (1.0 - alpha) * flat + alpha * (W @ flat)
    return flat.reshape(vals.shape)


def kish_neff(k: torch.Tensor) -> torch.Tensor:
    """``(sum_i k_i)^2 / sum_i k_i^2`` per row -- how many points a weighting averages."""
    return k.sum(1) ** 2 / (k ** 2).sum(1).clamp_min(torch.finfo(k.dtype).tiny)


class NeffFW(TrueFW):
    """The FW metric with a per-query bandwidth that fixes the Kish effective sample size.

    ``h(x)`` is found by bisection in ``log h`` so the weights ``exp(-|x - x_i|^2 / 4h^2)``
    have ``n_eff`` effective points, and the weights are shifted so the nearest point
    carries weight 1.  The shift cancels in the normalised average, so on the data this
    is a plain fixed-neighbour-count smoother; off the data the window widens instead of
    emptying, and the ``eps_den`` floor never takes over.  ``h`` is held fixed when
    differentiating, so Phase 1 sees the gradient of the smoother at a frozen window.

    There is no closed-form ``mobility_divergence`` for a moving bandwidth, so Path B is
    refused rather than run on the fixed-bandwidth formula.
    """

    name = "FFM-neff"

    def __init__(self, X, D_pts, b0_pts, cfg, rho: float, lam: float | None = None,
                 lam_mult: float = 0.3, n_eff: float = 100.0, bisect_iters: int = 30):
        self.n_eff, self.bisect_iters = float(n_eff), int(bisect_iters)
        super().__init__(X, D_pts, b0_pts, cfg, rho=rho, lam=lam, lam_mult=lam_mult)

    def _kernel(self, x):
        sqd = torch.clamp((x ** 2).sum(1, keepdim=True) + (self.X ** 2).sum(1)[None, :]
                          - 2.0 * (x @ self.X.T), min=0.0)
        with torch.no_grad():
            s = sqd.detach()
            s0 = s.min(dim=1, keepdim=True).values
            r = s - s0
            lo = torch.full_like(s0, float(np.log(1e-4 * self.eps_kernel)))
            hi = torch.log(1e2 * torch.sqrt(r.max(dim=1, keepdim=True).values) + 1e-30)
            hi = torch.maximum(hi, lo + 1.0)
            for _ in range(self.bisect_iters):
                mid = 0.5 * (lo + hi)
                small = kish_neff(torch.exp(-r / (4.0 * torch.exp(2.0 * mid))))[:, None] \
                    < self.n_eff
                lo = torch.where(small, mid, lo)
                hi = torch.where(small, hi, mid)
            h2 = torch.exp(lo + hi)
        return torch.exp(-(sqd - s0) / (4.0 * h2))

    def mobility_divergence(self, x):
        raise NotImplementedError("NeffFW has a moving bandwidth; Path B is not defined on it")


class BarrierFW(TrueFW):
    """``Phi(x) F_FW(x, v)`` with ``Phi = 1 + (phi_max - 1) exp(-S(x) / kappa)``.

    ``S(x) = sum_i k(x, x_i)`` is the shipped kernel's mass, so ``S >= 1`` on every
    sample (its own term) and ``Phi`` is 1 there to within ``(phi_max - 1) e^{-1/kappa}``;
    where the kernel is empty ``Phi -> phi_max``.  A conformal factor on a Randers norm
    is again a Randers norm with the same admissibility number (see
    :class:`~scripts.method.metric.ConformalFW`), so the 1-form is untouched and only
    *length* off the data becomes expensive.  The mobility is not rescaled.
    """

    name = "FFM-barrier"

    def __init__(self, X, D_pts, b0_pts, cfg, rho: float, lam: float | None = None,
                 lam_mult: float = 0.3, phi_max: float = 30.0, kappa: float = 0.1):
        self.phi_max, self.kappa = float(phi_max), float(kappa)
        super().__init__(X, D_pts, b0_pts, cfg, rho=rho, lam=lam, lam_mult=lam_mult)

    def barrier(self, k):
        return 1.0 + (self.phi_max - 1.0) * torch.exp(-k.sum(1) / self.kappa)

    def tensors(self, x):
        k = self._kernel(x)
        D = self._smooth(k, self.D_pts)
        b = self._smooth(k, self.b0_pts) + self.eps_b
        Ci = torch.linalg.inv(D + self.rho * self.I[None])
        a2 = torch.einsum("bi,bij,bj->b", b, Ci, b).clamp(min=0.0) + self.lam ** 2
        p = self.barrier(k)
        return (p ** 2 * a2)[:, None, None] * Ci, -p[:, None] * torch.einsum("bij,bj->bi",
                                                                             Ci, b)


class StraightPhi(torch.nn.Module):
    """``phi = 0``: the interpolant is the chord.  Kept a function of ``t`` so the exact
    time derivative in :func:`~scripts.method.nets.interpolant` is defined (and zero)."""

    def forward(self, t, x0, x1):
        return 0.0 * t + torch.zeros_like(x0)
