"""The empirical transition matrix ``P`` over a point cloud.

The one piece of ``scripts/experiments/synth_suite/data.py`` that was never about that suite's
datasets.  It is the sparse, chunked implementation; the dense original still lives in
:func:`scripts.core.datasets.build_transition_matrix` and is identical in value, but it
materialises an (N, N) array and so is unusable at benchmark sizes.

``P`` is the *only* thing the geometry is derived from -- :class:`~scripts.core.geometry.
FinslerGeometry` reads its symmetric part for the diffusion tensor and its first moment
for the drift -- so the neighbourhood scale below is a modelling choice, not an
implementation detail.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist

#: neighbourhood of the k-NN graph, and the mass held on the diagonal.  Shared by every
#: caller so that two clouds built at different sizes are still comparable.
P_KNN_K = 15
P_SELF_WEIGHT = 0.05


def _unit(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / (n + eps)


# --------------------------------------------------------------------------- #
#  Transition matrix — chunked k-NN reimplementation of the dense original
# --------------------------------------------------------------------------- #
def build_transition_matrix(
    X: np.ndarray,
    v: np.ndarray,
    k: int = P_KNN_K,
    lam: float = 4.0,
    self_weight: float = P_SELF_WEIGHT,
    chunk: int = 512,
) -> sp.csr_matrix:
    """Directional, row-stochastic P over a point cloud (sparse, chunked).

    Identical in value to :func:`scripts.core.datasets.build_transition_matrix`:

        P_ij  ∝  exp(-‖d_ij‖² / (2σ²)) · exp( λ · ⟨v_i, d_ij⟩ / ‖d_ij‖ )

    over the k nearest neighbours j of i (self excluded), plus a ``self_weight``
    holding mass on the diagonal, then row-normalised.  σ is the median distance to
    the k-th neighbour, **recomputed on whatever point set is passed in** — this is
    the property that makes the rebuild on a thinned subset honest: neighbourhoods
    legitimately widen as cells are removed, exactly as they would if the velocity
    pipeline were re-run on fewer cells.
    """
    X = np.asarray(X, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    N = X.shape[0]
    assert v.shape == X.shape, f"v {v.shape} must match X {X.shape}"
    assert N > k + 1, f"need N > k+1 = {k + 1} points to build a {k}-NN graph, got {N}"

    # --- pass 1: data-adaptive bandwidth σ = median distance to the k-th neighbour
    kth = np.empty(N, dtype=np.float64)
    for s in range(0, N, chunk):
        sl = slice(s, min(s + chunk, N))
        Dc = cdist(X[sl], X)                                   # (c, N), self dist = 0
        # (k+1)-th smallest *including* self == k-th neighbour excluding self
        kth[sl] = np.partition(Dc, k, axis=1)[:, k]
    sigma = float(np.median(kth)) + 1e-9

    # --- pass 2: k-NN edges with the Gaussian x directional weight
    rows_all, cols_all, vals_all = [], [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        rows = np.arange(s, e)
        Dc = cdist(X[rows], X)                                 # (c, N)
        Dc[np.arange(e - s), rows] = np.inf                    # exclude self
        nn = np.argpartition(Dc, k - 1, axis=1)[:, :k]         # (c, k) k nearest
        dij = np.take_along_axis(Dc, nn, axis=1)               # (c, k) ‖d_ij‖
        diff = X[nn] - X[rows][:, None, :]                     # (c, k, d)  d_ij
        align = np.einsum("cd,ckd->ck", v[rows], diff) / (dij + 1e-12)
        w = np.exp(-(dij ** 2) / (2.0 * sigma ** 2)) * np.exp(lam * align)
        rows_all.append(np.repeat(rows, k))
        cols_all.append(nn.reshape(-1))
        vals_all.append(w.reshape(-1))

    P = sp.coo_matrix(
        (np.concatenate(vals_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(N, N),
    ).tocsr()
    P = P + sp.diags(np.full(N, self_weight))                  # holding probability
    rs = np.asarray(P.sum(axis=1)).ravel()
    assert rs.min() > 0.0, "a row of P has zero mass before normalisation"
    P = sp.diags(1.0 / rs) @ P
    return P.tocsr()


# --------------------------------------------------------------------------- #
#  Undirected P — the "the cloud is all you have" case
# --------------------------------------------------------------------------- #
def build_uniform_knn_transition(
    X: np.ndarray,
    k: int = P_KNN_K,
    self_weight: float = P_SELF_WEIGHT,
    chunk: int = 512,
) -> sp.csr_matrix:
    """Row-stochastic P that gives each of the k nearest neighbours equal mass.

        P_ij = (1 - s) / k   for the k nearest j of i,     P_ii = s

    No Gaussian falloff and no directional term: the only information in this P is
    *which points are neighbours*, which is the only information a bare point cloud
    carries.  It is the transition matrix to use when the benchmark supplies no
    velocities — the MFM synthetic clouds, where the comparison is against a purely
    Riemannian method and any drift we read out would be an artefact.

    The one thing it is *not* is symmetric.  "j is one of i's k nearest" is not a
    symmetric relation on a cloud of varying density, so ``P^asym`` is small but
    non-zero, and the Randers 1-form built from it is a measure of kNN-graph
    asymmetry rather than of any dynamics.  :func:`asymmetry_scale` quantifies
    exactly that, so a report can state how inert the 1-form is instead of assuming it.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    assert X.ndim == 2, f"X must be (N, d), got {X.shape}"
    assert N > k + 1, f"need N > k+1 = {k + 1} points to build a {k}-NN graph, got {N}"
    assert 0.0 <= self_weight < 1.0, f"self_weight must be in [0, 1), got {self_weight}"

    rows_all, cols_all = [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        rows = np.arange(s, e)
        Dc = cdist(X[rows], X)                                 # (c, N)
        Dc[np.arange(e - s), rows] = np.inf                    # exclude self
        nn = np.argpartition(Dc, k - 1, axis=1)[:, :k]         # (c, k) k nearest
        rows_all.append(np.repeat(rows, k))
        cols_all.append(nn.reshape(-1))

    rows_all = np.concatenate(rows_all)
    cols_all = np.concatenate(cols_all)
    # every edge carries the same weight; duplicates cannot occur because the k nearest
    # of a row are distinct column indices, so the row sum is exactly k before scaling
    P = sp.coo_matrix((np.full(rows_all.shape, (1.0 - self_weight) / k),
                       (rows_all, cols_all)), shape=(N, N)).tocsr()
    P = P + sp.diags(np.full(N, self_weight))
    rs = np.asarray(P.sum(axis=1)).ravel()
    assert np.allclose(rs, 1.0, atol=1e-12), (
        f"uniform rows must already sum to 1, got [{rs.min()}, {rs.max()}]")
    return P.tocsr()


# --------------------------------------------------------------------------- #
#  Velocity-only P — the "the cloud comes with arrows" case
# --------------------------------------------------------------------------- #
#: softmax temperature of the velocity kernel.  4.0 is CellRank's default
#: ``softmax_scale`` for :class:`cellrank.kernels.VelocityKernel`, kept rather than
#: retuned so the kernel is recognisably theirs and not a knob we chose.
VELOCITY_SOFTMAX_SCALE = 4.0


def build_velocity_kernel(
    X: np.ndarray,
    v: np.ndarray,
    k: int = P_KNN_K,
    softmax_scale: float = VELOCITY_SOFTMAX_SCALE,
    self_weight: float = P_SELF_WEIGHT,
    chunk: int = 512,
) -> sp.csr_matrix:
    """Row-stochastic P from a *measured* velocity field, the CellRank way.

        P_ij  ∝  exp( λ · cos(v_i, x_j - x_i) )   over the k nearest j of i,   P_ii = s

    This is :class:`cellrank.kernels.VelocityKernel` in its deterministic mode: the
    correlation between a cell's velocity and the displacement to each neighbour,
    softmaxed over that cell's neighbourhood at ``softmax_scale`` (their default, 4).
    A neighbour the arrow points at gets e^{2λ} times the mass of the one directly
    behind it, so all of the *direction* in P comes from v and none of it from density.

    Difference from :func:`build_transition_matrix`, which is the other directed
    builder here: that one multiplies the same directional term by a Gaussian
    ``exp(-‖d‖²/2σ²)``, so distance both selects the neighbourhood *and* reweights
    inside it.  CellRank's velocity kernel does not — distance enters only through
    which points are neighbours at all — and this function is the faithful form, for
    benchmarks that are scored against a velocity-based method.

    Unlike :func:`build_uniform_knn_transition`, the antisymmetric part here is the
    signal rather than an artefact of kNN asymmetry: on a rotational field
    :func:`asymmetry_scale` runs near 1 and the flux read out of ``P^asym`` recovers
    the field's own direction.  Both numbers are cheap and both should be reported.
    """
    X = np.asarray(X, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    N = X.shape[0]
    assert X.ndim == 2, f"X must be (N, d), got {X.shape}"
    assert v.shape == X.shape, f"v {v.shape} must match X {X.shape}"
    assert N > k + 1, f"need N > k+1 = {k + 1} points to build a {k}-NN graph, got {N}"
    assert 0.0 <= self_weight < 1.0, f"self_weight must be in [0, 1), got {self_weight}"

    rows_all, cols_all, vals_all = [], [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        rows = np.arange(s, e)
        Dc = cdist(X[rows], X)                                 # (c, N)
        Dc[np.arange(e - s), rows] = np.inf                    # exclude self
        nn = np.argpartition(Dc, k - 1, axis=1)[:, :k]         # (c, k) k nearest
        diff = X[nn] - X[rows][:, None, :]                     # (c, k, d)  d_ij
        cos = np.einsum("cd,ckd->ck", _unit(v[rows]), _unit(diff))
        # softmax over the k neighbours, shifted by the row max for numerical safety;
        # the shift cancels in the normalisation and keeps exp() away from overflow
        w = np.exp(softmax_scale * (cos - cos.max(axis=1, keepdims=True)))
        w /= w.sum(axis=1, keepdims=True)
        rows_all.append(np.repeat(rows, k))
        cols_all.append(nn.reshape(-1))
        vals_all.append(((1.0 - self_weight) * w).reshape(-1))

    P = sp.coo_matrix(
        (np.concatenate(vals_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(N, N),
    ).tocsr()
    P = P + sp.diags(np.full(N, self_weight))
    rs = np.asarray(P.sum(axis=1)).ravel()
    assert np.allclose(rs, 1.0, atol=1e-10), (
        f"softmax rows must already sum to 1, got [{rs.min()}, {rs.max()}]")
    return P.tocsr()


def asymmetry_scale(P: sp.csr_matrix) -> float:
    """‖P − Pᵀ‖_F / ‖P + Pᵀ‖_F — how much of P is antisymmetric.

    Zero for a perfectly undirected graph.  Reported alongside any result on a
    uniform-kNN P so the claim "the 1-form has nothing to bite on here" is a measured
    number rather than an assertion.
    """
    A = (P - P.T)
    S = (P + P.T)
    num = float(np.sqrt((A.multiply(A)).sum()))
    den = float(np.sqrt((S.multiply(S)).sum())) + 1e-30
    return num / den
