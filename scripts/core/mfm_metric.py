"""Metric Flow Matching (Kapusniak et al.) metrics as a benchmark for our pipeline.

This is a faithful **adapter** around the authors' own geometry code, copied verbatim
into :mod:`scripts.core.vendor` so nothing here reads a checkout on the machine (their
release is https://github.com/kksniak/metric-flow-matching).  MFM learns a geodesic
interpolant + flow
under a *data-manifold metric* built purely from the point cloud — **no velocity and
no transition matrix P** (that is the clean contrast with our Finsler FFM, which is
built from P, and with Curly-FM, which is built from the RNA velocity).

Two metrics, following the reference (``mfm/geo_metrics/``):

  * **LAND** (2-D here) — the diagonal, anisotropic local-covariance metric.  We import
    the authors' ``land_metric_tensor`` verbatim.  The squared metric length of a
    velocity is ``F² = Σ_d v_d² · M_dd(x)^α`` with
    ``M_dd(x) = 1 / (Σ_n w_n(x)·(x−x_n)_d² + ρ)`` and Gaussian weights of bandwidth γ.
  * **RBF** (10-D / 50-D here) — the learned *conformal* metric.  We reimplement
    ``mfm/geo_metrics/rbf.py`` self-contained (KMeans centres + a non-negative RBF head
    trained so ``h(x)≈1`` on data, then ``M(x) = 1/(h(x)+ρ)^α``), giving
    ``F² = M(x)·‖v‖²``.  (Reimplemented rather than imported only to drop the repo's
    PyTorch-Lightning/Wandb training scaffolding — the maths is identical.)

Both objects expose exactly the surface our Path-A trainer needs: ``.d`` and a
differentiable ``.F2(x, v)`` — so ``scripts.core.train.train_phase1`` /
``build_ot_coupler`` treat them like a drop-in ``FinslerGeometry``.
"""
from __future__ import annotations

import inspect

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor

# --- the authors' LAND metric, copied verbatim into scripts.core.vendor rather than
#     imported from a checkout on this machine (see that package's docstring) ---
from .vendor.mfm_land import land_metric_tensor


# --------------------------------------------------------------------------- #
#  LAND metric (2-D)
# --------------------------------------------------------------------------- #
class LandMetric:
    """Diagonal LAND metric F²(x,v) = Σ_d v_d² · land_metric_tensor(x)_d^α.

    ``land_metric_tensor(x, samples, gamma, rho)`` returns the inverse diagonal
    ``1/(Σ_n w_n (x−x_n)² + ρ)``: large along the data (cheap), ≈1/ρ in voids
    (expensive) — the manifold-adherence metric MFM learns geodesics under.
    """

    def __init__(self, X: np.ndarray, gamma: float = 0.2, rho: float = 1e-3,
                 alpha: float = 1.0, device="cpu", dtype=torch.float32) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.samples = torch.as_tensor(X, dtype=dtype, device=self.device)
        self.d = int(self.samples.shape[1])
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.name = "LAND"

    def F2(self, x: Tensor, v: Tensor) -> Tensor:
        assert x.shape == v.shape and x.dim() == 2
        M = land_metric_tensor(x, self.samples, self.gamma, self.rho) ** self.alpha
        return (v ** 2 * M).sum(-1)                          # (B,)


# --------------------------------------------------------------------------- #
#  RBF metric (>2-D)
# --------------------------------------------------------------------------- #
class RbfMetric:
    """Conformal RBF metric F²(x,v) = M(x)·‖v‖²  with  M(x) = 1/(h(x)+ρ)^α.

    ``h(x) = Σ_k W_k exp(-½ λ_k ‖x−C_k‖²)`` is a non-negative RBF network (KMeans
    centres ``C_k``, per-cluster bandwidths ``λ_k = 0.5/(κ σ_k)²``) trained so h≈1 on
    the data cloud (self-contained reimplementation of ``mfm/geo_metrics/rbf.py``).
    """

    def __init__(self, X: np.ndarray, n_centers: int = 100, kappa: float = 1.0,
                 rho: float = 1e-3, alpha: float = 1.0, lr: float = 1e-2,
                 steps: int = 300, device="cpu", dtype=torch.float32,
                 seed: int = 0) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.name = "RBF"
        Xt = torch.as_tensor(X, dtype=dtype, device=self.device)
        self.d = int(Xt.shape[1])
        K = min(n_centers, Xt.shape[0])

        # --- KMeans centres + per-cluster bandwidths (rbf.py::on_train_start) ---
        Xnp = np.asarray(X, dtype=np.float64)
        km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(Xnp)
        C = km.cluster_centers_
        labels = km.labels_
        sigmas = np.zeros((K, 1))
        for k in range(K):
            pts = Xnp[labels == k]
            var = ((pts - C[k]) ** 2).mean(axis=0) if len(pts) else np.ones(self.d)
            sigmas[k, 0] = np.sqrt(var.mean())
        sigmas[sigmas < 1e-8] = 1e-8
        self.C = torch.as_tensor(C, dtype=dtype, device=self.device)          # (K,d)
        self.lamda = torch.as_tensor(0.5 / (kappa * sigmas) ** 2,             # (K,1)
                                     dtype=dtype, device=self.device)

        # --- fit non-negative weights W so h(x)≈1 on the data (rbf.py training) ---
        W = torch.rand(K, 1, dtype=dtype, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((1.0 - self._h(Xt, W)) ** 2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                W.clamp_(min=1e-4)                            # on_before_zero_grad
        self.W = W.detach()
        self._last_loss = float(loss)

    def _h(self, x: Tensor, W: Tensor) -> Tensor:
        dist2 = torch.cdist(x, self.C) ** 2                  # (B,K)
        phi = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])   # (B,K,1)
        return (W * phi).sum(dim=1)                          # (B,1)

    def F2(self, x: Tensor, v: Tensor) -> Tensor:
        assert x.shape == v.shape and x.dim() == 2
        h = self._h(x, self.W)                               # (B,1)
        M = 1.0 / (h + self.rho) ** self.alpha               # (B,1) conformal
        return (M * (v ** 2).sum(-1, keepdim=True)).squeeze(-1)   # (B,)


# --------------------------------------------------------------------------- #
#  Factory: LAND in 2-D, RBF otherwise (per the project spec)
# --------------------------------------------------------------------------- #
def mfm_metric_kind(X: np.ndarray) -> str:
    """``"land"`` in a 2-D data space, ``"rbf"`` otherwise — the reference's own rule.

    Exposed so a caller can know *which* metric it is about to get without
    reimplementing the dimension test.
    """
    return "land" if int(np.asarray(X).shape[1]) == 2 else "rbf"


def _accepted(cls, params: dict) -> dict:
    names = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {k: v for k, v in params.items() if k in names}


def make_mfm_metric(X: np.ndarray, device="cpu", dtype=torch.float32,
                    seed: int = 0, params: dict | None = None, kind: str = "auto"):
    """LAND metric for a 2-D data space, RBF metric otherwise.

    ``params`` (optional) overrides the metric hyperparameters — e.g. ``{"gamma": 0.5}``
    for LAND or ``{"n_centers": 200, "kappa": 2.0}`` for RBF — so the sweep and the tuned
    eval can inject the best-found values.

    The two metrics share ``rho`` and ``alpha`` but not their bandwidth parameterisation
    (LAND has ``gamma``; RBF has ``n_centers`` / ``kappa``), so a key meant for the other
    metric is **dropped** rather than raising: one caller supplies one ``params`` dict for
    clouds of every dimension, and which class it lands in is a property of the data.  A
    key that neither class accepts is still an error — that is a typo, not a dispatch.

    ``kind`` overrides the dimension rule.  The rule is a *default*, not a property of
    MFM: the authors' per-cloud yamls set ``velocity_metric`` explicitly, and not always
    to what the rule would infer from the dimension, so reproducing one of their configs
    requires being able to name the family.  Left at ``"auto"`` the behaviour is
    unchanged.
    """
    params = dict(params or {})
    unknown = set(params) - set(_accepted(LandMetric, params)) - set(
        _accepted(RbfMetric, params))
    assert not unknown, f"no MFM metric accepts {sorted(unknown)}"
    assert kind in ("auto", "land", "rbf"), f"unknown metric kind {kind!r}"
    resolved = mfm_metric_kind(X) if kind == "auto" else kind
    if resolved == "land":
        return LandMetric(X, device=device, dtype=dtype, **_accepted(LandMetric, params))
    return RbfMetric(X, device=device, dtype=dtype, seed=seed,
                     **_accepted(RbfMetric, params))
