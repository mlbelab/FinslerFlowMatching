"""The container every driver trains on, and the dense builder for its ``P``.

A :class:`Dataset` bundles
    X          : (N, d)  point-cloud coordinates
    P          : (N, N)  row-stochastic transition matrix (directional / asymmetric)
    tau        : (N,)    ground-truth pseudotime (for colouring / sourcing)
    v          : (N, d)  ground-truth unit velocity field (only used to *build* P)
    name       : str

This module used to carry three synthetic generators as well -- an Arch (curvature), a
Cycle (non-zero curl) and a Bifurcation (unequal fates).  The paper's synthetic
evidence is now the single :mod:`route_bench` **Sheet**, which builds its own cloud, so
the generators were removed; they are recoverable from git history.  What remains is
used by the real-data drivers: the container itself, and the dense
:func:`build_transition_matrix` that :mod:`scripts.core.transition` mirrors sparsely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist


@dataclass
class Dataset:
    name: str
    X: np.ndarray      # (N, d)  point-cloud coordinates
    P: np.ndarray      # (N, N)  row-stochastic transition matrix
    tau: np.ndarray    # (N,)    pseudotime in [0, 1] (cyclic for Cycle); USED ONLY
                       #         for colouring plots — never enters training.
    v: np.ndarray      # (N, d)  unit velocity field; USED ONLY to *build* P (it
                       #         supplies the directional bias) — never in training.
    p0: np.ndarray     # (N,) bool  source endpoint set (defines the marginal p_0)
    p1: np.ndarray     # (N,) bool  target endpoint set (defines the marginal p_1)

    def __post_init__(self) -> None:
        N, d = self.X.shape
        assert self.P.shape == (N, N), f"P must be {(N, N)}, got {self.P.shape}"
        assert self.tau.shape == (N,), f"tau must be {(N,)}, got {self.tau.shape}"
        assert self.v.shape == (N, d), f"v must be {(N, d)}, got {self.v.shape}"
        assert self.p0.shape == (N,) and self.p1.shape == (N,), "p0/p1 must be (N,)"
        assert self.p0.any() and self.p1.any(), "p0 and p1 must be non-empty"
        # P may be a dense ndarray (synthetic sets) or a scipy-sparse row-stochastic
        # matrix (CellRank velocity+similarity kernel); handle both uniformly.
        if sp.issparse(self.P):
            row_sums = np.asarray(self.P.sum(axis=1)).ravel()
            p_min = self.P.min()
        else:
            row_sums = self.P.sum(axis=1)
            p_min = self.P.min()
        assert np.allclose(row_sums, 1.0, atol=1e-6), "P rows must sum to 1"
        assert p_min >= -1e-12, "P must be non-negative"


def _unit(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise normalise to unit length."""
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / (n + eps)


def build_transition_matrix(
    X: np.ndarray,
    v: np.ndarray,
    k: int = 15,
    lam: float = 4.0,
    self_weight: float = 0.05,
) -> np.ndarray:
    """Directional, row-stochastic transition matrix over a point cloud.

    For each ordered neighbour pair (i -> j) with displacement d_ij = x_j - x_i,

        P_ij  ∝  exp(-||d_ij||^2 / (2 σ^2)) · exp( λ · <v_i, d_ij> / ||d_ij|| )

    restricted to the k nearest neighbours of i.  The Gaussian factor keeps
    transitions local; the directional factor tilts probability mass along the
    local velocity field v_i.  A large λ produces a strongly *asymmetric* P
    (large P^asym), which is exactly what the Cycle benchmark needs.

    σ is set to the median k-NN distance (data-adaptive bandwidth).
    """
    N = X.shape[0]
    assert v.shape == X.shape
    dist = cdist(X, X)                                   # (N, N)

    # data-adaptive Gaussian bandwidth: median distance to the k-th neighbour
    knn_sorted = np.sort(dist, axis=1)
    sigma = float(np.median(knn_sorted[:, min(k, N - 1)])) + 1e-9

    W = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))        # (N, N) locality

    diffs = X[None, :, :] - X[:, None, :]               # (N, N, d): d_ij = x_j - x_i
    dnorm = np.linalg.norm(diffs, axis=-1) + 1e-12      # (N, N)
    align = np.einsum("id,ijd->ij", v, diffs) / dnorm  # cos-similarity * |d| / |d|
    D = np.exp(lam * align)                             # (N, N) directional tilt

    # k-nearest-neighbour mask (exclude self)
    mask = np.zeros((N, N), dtype=bool)
    nn_idx = np.argsort(dist, axis=1)[:, 1 : k + 1]
    rows = np.repeat(np.arange(N), k)
    mask[rows, nn_idx.reshape(-1)] = True

    P = W * D * mask
    P[np.arange(N), np.arange(N)] += self_weight        # small holding probability
    P = P / P.sum(axis=1, keepdims=True)
    return P
