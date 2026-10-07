"""Scoring primitives shared by every experiment in this repository.

Extracted verbatim from the retired ``scripts/experiments/synth_suite/metrics.py`` when the
benchmark was cut down to the Sheet: these functions never referred to a dataset, only
to point clouds and trajectories, so they belong with the method library rather than
with any one benchmark.  What was left behind there -- the per-band marginal report, the
fate/branch assignment rule, the escape diagnostic -- was written against that suite's
``SynthDataset``/``Split`` abstraction and died with it (recover from the baseline commit
if ever needed).

Two families live here:

*Distribution distances* (:func:`wasserstein`, :func:`rbf_mmd`) compare a transported
cloud with a held-out one.  Both subsample with a *shared* generator seed, so the
subsampling noise is common-mode across arms and differences between arms stay
meaningful.

*Network probes* (:func:`velocity_fidelity`, :func:`score_second_moment`) read the fitted
networks directly instead of their samples.  For the drift that is a convenience; for the
score it is essential, and the docstring of :func:`score_second_moment` explains at length
why the obvious trajectory-covariance test is circular for anything integrated as an SDE.

:func:`local_moment_fidelity` sits between the two families and reads neither the network
nor a transported marginal.  It asks whether the model's own displacements carry the
first and second moment of a **raw, unsmoothed** transition row at a node no arm was
trained on, which is the one question a distance between clouds cannot answer: a cloud
can land in the right place having moved the wrong way.
"""
from __future__ import annotations

import numpy as np
import ot  # POT: exact optimal transport
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from .geometry import FinslerGeometry

#: caps that keep the *exact* OT solve (O(n³ log n)) and the O(n²) MMD tractable
MAX_OT_PTS = 400
MAX_MMD_PTS = 1000
#: draws averaged into a sampling floor.  One split is too noisy to read a gap against:
#: on the corridor references here the split-to-split spread is a tenth of the floor.
N_FLOOR_DRAWS = 20
#: RBF-MMD bandwidth mixture, as multiples of the median pairwise distance
MMD_BANDWIDTH_MULT = (0.25, 0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------------- #
#  Distribution distances
# --------------------------------------------------------------------------- #
def _subsample(A: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform subsample without replacement, or ``A`` itself if already small."""
    A = np.asarray(A, dtype=np.float64)
    if len(A) <= max_n:
        return A
    return A[rng.choice(len(A), max_n, replace=False)]


def floor_splits(n_ref: int, n_draw: int, seed: int = 0,
                 n_draws: int = N_FLOOR_DRAWS):
    """``(a_idx, b_idx)`` index pairs for a *size-matched* two-sample null on a reference.

    A model pushes ``n_draw`` particles and is scored against all ``n_ref`` reference
    cells, so the null that bounds it is a draw of the *same* size against the rest of
    the reference -- not two halves.  Two halves are both smaller than either side of
    the comparison they are meant to bound, and W_p between empirical measures carries
    an O(n^{-1/d}) finite-sample bias, so a half-split floor reads high: on a 2-D
    Gaussian null a 150-vs-150 split scores 0.40 where the 400-vs-300 comparison it
    bounds scores 0.31.  An inflated floor shrinks the denominator of every
    gap-to-floor statement, and can place a model below its own floor.

    ``n_draw`` is capped at half the reference, past which the remainder stops being a
    marginal; :data:`MAX_OT_PTS` then caps both sides identically inside the metric, so
    on a reference of 800 or more the null is exactly the comparison's sample size.
    """
    n = int(min(n_draw, n_ref // 2))
    assert n > 1, f"reference of {n_ref} too small for a floor at n_draw={n_draw}"
    rng = np.random.default_rng(seed)
    for _ in range(max(1, n_draws)):
        p = rng.permutation(n_ref)
        yield p[:n], p[n:]


def mean_floor(ref: np.ndarray, n_draw: int, metric, seed: int = 0,
               n_draws: int = N_FLOOR_DRAWS) -> dict:
    """``metric(a, b, seed)`` averaged over :func:`floor_splits` draws of ``ref``."""
    ref = np.asarray(ref, dtype=np.float64)
    rows = [metric(ref[a], ref[b], k) for k, (a, b) in
            enumerate(floor_splits(len(ref), n_draw, seed, n_draws))]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def wasserstein(A: np.ndarray, B: np.ndarray, p: int = 2,
                max_n: int = MAX_OT_PTS, seed: int = 0) -> float:
    """Exact W_p between two uniform empirical measures (POT's EMD solver).

    ``p=2`` uses the squared-Euclidean ground cost and takes a final square root;
    ``p=1`` uses the Euclidean cost directly.  Both clouds are subsampled to
    ``max_n`` with the *same* generator seed across arms, so the subsampling noise
    is common-mode and differences between arms stay meaningful.  A caller comparing
    against a published number should pass ``max_n`` at or above its cloud size so no
    subsampling happens at all -- the caller does.
    """
    assert p in (1, 2), f"only W1 and W2 are implemented, got p={p}"
    rng = np.random.default_rng(seed)
    A, B = _subsample(A, max_n, rng), _subsample(B, max_n, rng)
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    a = np.full(len(A), 1.0 / len(A))
    b = np.full(len(B), 1.0 / len(B))
    metric = "sqeuclidean" if p == 2 else "euclidean"
    M = ot.dist(A, B, metric=metric)
    # 1e7 is the MFM release's cap and is what a 5000x5000 solve needs headroom for.
    # Raising it cannot move an already-converged answer -- POT stops at optimality, not
    # at the cap -- it only removes the silent truncation POT would otherwise warn about.
    val = float(ot.emd2(a, b, M, numItermax=10_000_000))
    return float(np.sqrt(max(val, 0.0))) if p == 2 else val


def rbf_mmd(A: np.ndarray, B: np.ndarray, max_n: int = MAX_MMD_PTS,
            seed: int = 0) -> float:
    """√(unbiased MMD²) under a 5-bandwidth mixture of RBF kernels.

    k(x, y) = (1/S) Σ_s exp(−‖x−y‖² / (2 h_s²)),  h_s = m_s · σ_med, with σ_med the
    median pairwise distance of the *pooled* sample (the standard median heuristic)
    and m_s ∈ {¼, ½, 1, 2, 4}.  The mixture removes the single-bandwidth failure
    mode where the kernel is far wider or narrower than the separation being tested.

    The estimator is the unbiased MMD² (diagonal terms dropped), so it can come out
    slightly negative when the two samples really are from the same law; we clamp at
    zero before the square root and report the root so the number carries the same
    units as W1/W2.
    """
    rng = np.random.default_rng(seed)
    A, B = _subsample(A, max_n, rng), _subsample(B, max_n, rng)
    m, n = len(A), len(B)
    if m < 2 or n < 2:
        return float("nan")

    Z = np.concatenate([A, B], axis=0)
    D2 = ot.dist(Z, Z, metric="sqeuclidean")                  # (m+n, m+n)
    iu = np.triu_indices(len(Z), k=1)
    sigma_med = float(np.sqrt(np.median(D2[iu])) + 1e-12)     # median *distance*

    K = np.zeros_like(D2)
    for mult in MMD_BANDWIDTH_MULT:
        h2 = (mult * sigma_med) ** 2
        K += np.exp(-D2 / (2.0 * h2))
    K /= len(MMD_BANDWIDTH_MULT)

    Kaa, Kbb, Kab = K[:m, :m], K[m:, m:], K[:m, m:]
    # unbiased: drop the self-similarity diagonal from the within-sample terms
    term_a = (Kaa.sum() - np.trace(Kaa)) / (m * (m - 1))
    term_b = (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1))
    term_ab = Kab.mean()
    mmd2 = float(term_a + term_b - 2.0 * term_ab)
    return float(np.sqrt(max(mmd2, 0.0)))


def distribution_metrics(pred: np.ndarray, true: np.ndarray,
                         seed: int = 0) -> dict[str, float]:
    """The three headline distances between a predicted and a true point cloud."""
    return {
        "W2": wasserstein(pred, true, p=2, seed=seed),
        "W1": wasserstein(pred, true, p=1, seed=seed),
        "MMD": rbf_mmd(pred, true, seed=seed),
    }


# --------------------------------------------------------------------------- #
#  Per-timepoint evaluation of a pushed-forward trajectory
# --------------------------------------------------------------------------- #
def traj_slice(traj: np.ndarray, t: float) -> np.ndarray:
    """The state of every trajectory at model time ``t`` -> (B, d).

    ``traj`` is the (n_steps+1, B, d) array produced by integrating the flow from the
    source cloud; the integrator is uniform in t, so time ``t`` is row ``t·n_steps``.
    """
    assert traj.ndim == 3, f"traj must be (S+1, B, d), got {traj.shape}"
    n_steps = traj.shape[0] - 1
    return np.asarray(traj[int(round(t * n_steps))], dtype=np.float64)




# --------------------------------------------------------------------------- #
#  Drift fidelity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def velocity_fidelity(v_net, geom_ref: FinslerGeometry, traj: np.ndarray,
                      device: torch.device, dtype: torch.dtype = torch.float32,
                      n_time: int = 21, max_pts: int = 256, seed: int = 0,
                      min_drift_norm: float = 1e-8) -> dict[str, float]:
    """Alignment of the learned drift v_θ(t, x) with the empirical flux field b̂(x).

    Sampling: ``n_time`` model times spread uniformly over [0, 1] × ``max_pts``
    trajectories, i.e. the states the flow actually visits — scoring v_θ on a uniform
    box would mostly probe data voids where b̂ ≡ 0 and the comparison is vacuous.
    Points whose reference drift underflows to zero (kernel voids) are dropped and
    counted in ``frac_valid``.
    """
    assert traj.ndim == 3, f"traj must be (S+1, B, d), got {traj.shape}"
    S1, B, d = traj.shape
    rng = np.random.default_rng(seed)
    cols = rng.choice(B, min(max_pts, B), replace=False)
    rows = np.unique(np.round(np.linspace(0, S1 - 1, n_time)).astype(int))

    xs = np.asarray(traj[np.ix_(rows, cols)], dtype=np.float64)   # (T, C, d)
    ts = np.asarray(rows, dtype=np.float64) / (S1 - 1)            # (T,)
    x = torch.as_tensor(xs.reshape(-1, d), dtype=dtype, device=device)
    t = torch.as_tensor(np.repeat(ts, len(cols))[:, None], dtype=dtype, device=device)

    v = v_net(t, x)                                               # (M, d)
    b = geom_ref.drift(x)                                         # (M, d)
    assert v.shape == b.shape, f"v {tuple(v.shape)} vs b {tuple(b.shape)}"

    nb = b.norm(dim=1)
    ok = nb > min_drift_norm
    frac_valid = float(ok.float().mean())
    if int(ok.sum()) < 2:
        return {"cos": float("nan"), "alpha": float("nan"),
                "L2_rel": float("nan"), "frac_valid": frac_valid,
                "n_points": int(ok.sum())}
    v, b = v[ok], b[ok]

    dots = (v * b).sum(1)
    cos = float((dots / (v.norm(dim=1) * b.norm(dim=1) + 1e-30)).mean())
    # α*: the single scalar that best maps the empirical flux onto the model's units
    alpha = float(dots.sum() / ((b * b).sum() + 1e-30))
    # The residual uses max(α*, 0).  For any arm that flows forward α* > 0 and this is
    # the plain least-squares fit; it only bites on a *reversed* field, where the
    # unconstrained fit would flip b̂ and report L2_rel ≈ 0 for a model going exactly
    # the wrong way.  Clamped, that case reports L2_rel = 1, as it should.
    resid = ((v - max(alpha, 0.0) * b) ** 2).sum(1).mean()
    l2_rel = float(resid / ((v ** 2).sum(1).mean() + 1e-30))
    return {"cos": cos, "alpha": alpha, "L2_rel": l2_rel,
            "frac_valid": frac_valid, "n_points": int(v.shape[0])}


# --------------------------------------------------------------------------- #
#  Local moment fidelity against the raw held-out P rows
# --------------------------------------------------------------------------- #
def trajectory_segments(traj: np.ndarray, delta_steps: int = 1,
                        max_segments: int = 20000,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The model's displacement cloud ``(z, Delta)``, read off ``traj`` -> two (R, d).

    ``z = x_eta(t)`` and ``Delta = x_eta(t + delta) - x_eta(t)`` for every step of every
    path, with ``delta = delta_steps / n_steps`` of model time.  Sampling ``t`` at random
    (as the algorithm this implements does) draws from the same population; taking the
    whole integration grid instead just uses every draw the integrator already made, and
    then subsamples once to ``max_segments`` so the kernel weighting below stays a
    matrix and not a loop.
    """
    assert traj.ndim == 3, f"traj must be (S+1, B, d), got {traj.shape}"
    S1, _B, d = traj.shape
    assert 1 <= delta_steps < S1, f"delta_steps must be in [1, {S1 - 1}], got {delta_steps}"
    tr = np.asarray(traj, dtype=np.float64)
    z = tr[:-delta_steps].reshape(-1, d)
    dz = (tr[delta_steps:] - tr[:-delta_steps]).reshape(-1, d)

    ok = np.isfinite(z).all(1) & np.isfinite(dz).all(1)   # a diverged path carries none
    z, dz = z[ok], dz[ok]
    if len(z) > max_segments:
        pick = np.random.default_rng(seed).choice(len(z), max_segments, replace=False)
        z, dz = z[pick], dz[pick]
    return z, dz


def _raw_moments(x_i: np.ndarray, cols: np.ndarray,
                 vals: np.ndarray, X_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sum_j P_ij D_ij, sum_j P_ij D_ij D_ij^T)`` for one row, ``D_ij = x_j - x_i``."""
    D = X_all[cols] - x_i                                       # (k, d)
    return vals @ D, (D * vals[:, None]).T @ D


def local_moment_fidelity(traj: np.ndarray, X_nodes: np.ndarray, P_rows, X_all,
                          h: float | None = None, h_mult: float = 3.0,
                          delta_steps: int = 1, max_segments: int = 20000,
                          min_ess: float = 5.0, max_dist_h: float = 3.0,
                          chunk: int = 128, seed: int = 0,
                          return_nodes: bool = False) -> dict:
    """Do the model's own displacements have P's first and second moment, locally?

    The reference is the **raw** transition row at a held-out node — no kernel in front
    of it, no geometry object, nothing the model was trained through::

        b_P(x_i) = sum_j P_ij (x_j - x_i)
        C_P(x_i) = sum_j P_ij (x_j - x_i)(x_j - x_i)^T

    The model side is the same two moments of :func:`trajectory_segments`, localised at
    ``x_i`` by a Gaussian kernel ``k_h(x_i, z_r) = exp(-||x_i - z_r||^2 / 2h^2)``
    normalised to weights ``w_r``::

        b_theta(x_i) = sum_r w_r Delta_r
        C_theta(x_i) = sum_r w_r Delta_r Delta_r^T

    and the two scores are the cosine between the pair, the second in the Frobenius
    inner product, ``tr(C_theta^T C_P) / (||C_theta||_F ||C_P||_F)``.  Both are in
    [-1, 1] and both are scale-free by construction, which is the point: model time and
    P's per-transition step are different units, so anything that is not a cosine would
    be reporting that mismatch rather than the geometry.  ``R2`` is a similarity between
    two PSD matrices and so is non-negative; it reads as *shape* agreement — the
    directions the model spreads in against the directions the data spreads in.

    **How this differs from** :func:`velocity_fidelity`, which also reports a cosine.
    That one compares the learned field ``v_theta`` with the *kernel-smoothed* flux
    ``b_0``, at states the flow visits, on the full cloud.  Here the reference is one
    unsmoothed row of ``P``, the query points are held-out **nodes** rather than model
    states, and the model side is estimated from the sampled trajectories rather than
    read out of the network.  So this scores a model that has no ``v_theta`` to read (any
    sampler at all), and it scores it where the data says something and the model was
    never shown the answer.  The two agreeing is a check; the two disagreeing localises
    the failure to the smoothing.

    **Coverage is a result, not a nuisance.**  A node the model's trajectories never
    approach has no local estimate, and a node whose weight collapses onto one or two
    segments has an outer product rather than an average.  Both are excluded and counted
    in ``frac_covered``: a node is covered when its nearest segment is within
    ``max_dist_h * h`` and the weights' effective sample size ``(sum w)^2 / sum w^2``
    reaches ``min_ess``.  On a route-selection cloud a low ``frac_covered`` over the
    withheld region *is* the finding — the model did not go there — and reading a high
    R2 over the handful of nodes it did reach without that fraction beside it would be
    reading a survivor.

    ``h`` defaults to ``h_mult`` times the median nearest-neighbour spacing of ``X_all``,
    so it scales with the cloud rather than with its units; pass the reference
    geometry's own bandwidth to make the two diagnostics agree about what "local" means.
    """
    X_nodes = np.asarray(X_nodes, dtype=np.float64)
    X_all = np.asarray(X_all, dtype=np.float64)
    assert X_nodes.ndim == 2 and X_all.ndim == 2, "nodes and cloud must be (n, d)"
    assert X_nodes.shape[1] == X_all.shape[1] == traj.shape[2], (
        f"node/cloud/traj dimensions disagree: {X_nodes.shape[1]}, {X_all.shape[1]}, "
        f"{traj.shape[2]}")
    P = sp.csr_matrix(P_rows)
    assert P.shape == (len(X_nodes), len(X_all)), (
        f"P_rows must be (n_nodes, N) = {(len(X_nodes), len(X_all))}, got {P.shape}")

    if h is None:
        # k=2 because the first neighbour of a cloud point is itself
        nn = cKDTree(X_all).query(X_all, k=2)[0][:, 1]
        h = float(h_mult * np.median(nn))
    assert h > 0.0, "the localisation bandwidth must be positive"

    z, dz = trajectory_segments(traj, delta_steps, max_segments, seed)
    n_nodes = len(X_nodes)
    nan_out = {"R1": float("nan"), "R2": float("nan"), "R1_std": float("nan"),
               "R2_std": float("nan"), "frac_covered": 0.0, "n_nodes": n_nodes,
               "n_covered": 0, "h": h, "n_segments": int(len(z)),
               "delta": float(delta_steps / (traj.shape[0] - 1))}
    if len(z) < 2:
        return nan_out

    r1, r2, covered = np.full(n_nodes, np.nan), np.full(n_nodes, np.nan), np.zeros(
        n_nodes, dtype=bool)
    zz = (z ** 2).sum(1)
    for lo in range(0, n_nodes, chunk):
        blk = X_nodes[lo:lo + chunk]
        sqd = np.maximum((blk ** 2).sum(1)[:, None] + zz[None, :] - 2.0 * blk @ z.T, 0.0)
        for k in range(len(blk)):
            i = lo + k
            # weights are formed relative to the nearest segment, so a node whose whole
            # neighbourhood sits in the kernel's tail still gets meaningful *relative*
            # weights instead of a row that underflows to zero; whether the node is
            # close enough to be scored at all is the separate max_dist_h test.
            d2 = sqd[k]
            d2min = float(d2.min())
            a = np.exp(-(d2 - d2min) / (2.0 * h * h))
            Z = float(a.sum())
            ess = Z * Z / float((a * a).sum())
            if np.sqrt(d2min) > max_dist_h * h or Z <= 0.0 or ess < min_ess:
                continue
            covered[i] = True

            w = a / Z
            b_th = w @ dz
            C_th = (dz * w[:, None]).T @ dz
            row = slice(P.indptr[i], P.indptr[i + 1])
            b_P, C_P = _raw_moments(X_nodes[i], P.indices[row], P.data[row], X_all)

            nb = np.linalg.norm(b_th) * np.linalg.norm(b_P)
            if nb > 1e-30:
                r1[i] = float(b_th @ b_P / nb)
            nC = np.linalg.norm(C_th) * np.linalg.norm(C_P)
            if nC > 1e-30:
                r2[i] = float((C_th * C_P).sum() / nC)

    out = dict(nan_out)
    out["n_covered"] = int(covered.sum())
    out["frac_covered"] = float(covered.mean())
    for nm, arr in (("R1", r1), ("R2", r2)):
        good = np.isfinite(arr)
        out[f"n_{nm}"] = int(good.sum())
        if good.any():
            out[nm] = float(arr[good].mean())
            out[f"{nm}_std"] = float(arr[good].std())
            out[f"{nm}_q"] = [float(q) for q in
                              np.quantile(arr[good], (0.1, 0.25, 0.5, 0.75, 0.9))]
        if return_nodes:
            out[f"{nm}_nodes"] = arr
    return out


# --------------------------------------------------------------------------- #
#  Second-moment fidelity read off the learned score network
# --------------------------------------------------------------------------- #
def _bures(A: Tensor, B: Tensor) -> Tensor:
    """Squared Bures distance d_B²(A,B) = tr A + tr B - 2 tr[(A^½ B A^½)^½]  ->  (M,).

    A, B are batched symmetric PSD (M, d, d).  The middle term is the *fidelity*
    tr[(A^½ B A^½)^½]; it is symmetric in A and B even though the expression is not
    manifestly so, because A^½BA^½ and B^½AB^½ are similar.  We form A^½ by symmetric
    eigendecomposition (d is small) and re-symmetrise the product before its own
    ``eigh``, since the numerical A^½BA^½ is symmetric only to round-off.
    """
    ea, Va = torch.linalg.eigh(A)
    A_half = Va @ torch.diag_embed(torch.sqrt(torch.clamp(ea, min=0.0))) \
        @ Va.transpose(-1, -2)
    C = A_half @ B @ A_half
    C = 0.5 * (C + C.transpose(-1, -2))
    ec = torch.clamp(torch.linalg.eigvalsh(C), min=0.0)
    fidelity = torch.sqrt(ec).sum(-1)                       # tr[(A^½BA^½)^½]
    tr = lambda Z: Z.diagonal(dim1=-2, dim2=-1).sum(-1)     # noqa: E731
    return torch.clamp(tr(A) + tr(B) - 2.0 * fidelity, min=0.0)


def score_second_moment(s_net, geom_ref: FinslerGeometry, traj: np.ndarray,
                        sigma: float, device: torch.device,
                        dtype: torch.dtype = torch.float32,
                        n_time: int = 9, max_pts: int = 64,
                        t_lo: float = 0.1, t_hi: float = 0.9,
                        eig_floor: float = 1e-6,
                        seed: int = 0) -> dict[str, float]:
    """Does the learned score encode the data's anisotropic second moment?

    **Why not measure the trajectories.**  The obvious second-moment test — take the
    transported ensemble, detrend it locally, and compare its covariance to Σ — is
    circular for anything integrated as an SDE.  Affine detrending removes the drift
    and its shear by construction, so what is left *is* the increment the sampler
    injected, i.e. a hyperparameter.  Any ODE arm with the same increment bolted on
    scores identically, and the scale is tunable to 1 by a single scalar.  Measured
    against the real Σ field the shape score's isotropic-null floor is 0.82-0.88,
    which is where Path B sat: the statistic could not resolve the thing it named.

    **What this measures instead.**  Path B's conditional score is exactly linear in
    the displacement,

        ∇_x log p_t(x|x0,x1) = -[σ² t(1-t)]^{-1} M_t^{-1} (x - μ_t) ,

    so the *Jacobian* of the score is the negative precision of the conditional and is
    independent of x.  A network that has learned it therefore satisfies
    ∇_x s_φ = -M_t^{-1}/(σ²t(1-t)), and we can invert that relation to read the model's
    own mobility straight out of the weights, with no sampling anywhere:

        M̂_θ(t, x) = [ -sym ∇_x s_φ(t, x) ]^{-1} / ( σ² t (1-t) )

    (``sym`` because a finite network's Jacobian need not be symmetric, while the true
    precision is).  This is compared to the P-derived mobility M_P = ρI + Σ_P from the
    *same* full-cloud reference geometry every other metric uses.  Both are
    trace-normalised — so the comparison is of anisotropic *shape*, immune to the units
    mismatch between P's per-transition-step scale and the model's [0,1] time — and
    scored by the **Bures** distance, the Wasserstein-2 distance between the two
    centred Gaussians.  With unit traces d_B² = 2(1 - fidelity), so d_B ∈ [0, √2].

    **Only the shape is reported, and that is not a hedge.**  The magnitude that
    trace-normalisation discards, ``trace_ratio`` = tr M̂_θ / tr M_P, is kept in the
    returned dict as a diagnostic but is deliberately *not* tabulated, because measured
    on Cycle it runs 5.1e1 / 9.9e2 / 2.6e5 at σ = 0.40 / 0.15 / 0.05 — four orders of
    magnitude, monotone in σ.  Two things drive that.  The estimator is derived from the
    *conditional* score, but s_φ is fit to the conditional target in expectation over
    the coupling and therefore converges to the *marginal* score, whose precision is the
    population-scale one rather than the bridge width σ²t(1-t)M; dividing by σ²t(1-t)
    then inflates it by that ratio.  On top of that a small σ makes the true score
    steeper than the network can represent, so the fitted precision is flat and M̂_θ
    inflates further.  A number that mobile is not comparable across σ and would be
    indefensible in a table.  Both effects are pure *scale*: they leave the eigenvector
    frame and the eigenvalue ratios alone, which is exactly what the trace-normalised
    Bures distance reads, and d_B moves only 0.32 → 0.54 over the same 8× in σ.
    ``frac_valid`` is reported alongside d_B instead, and is itself informative: it is
    the fraction of probed states at which -sym ∇s is positive-definite at all, i.e.\
    where the learned score is locally the gradient of a genuine log-density.

    **How to read it.**  For the M_t-shaped bridge, M_P *is* the training target (up to
    the time-locking of M at μ_t versus evaluation at x), so a small d_B confirms the
    fit succeeded rather than discovering something.  The informative contrasts are
    (i) ``pathb_iso``, trained on M ≡ I, whose d_B should track the anisotropy of Σ,
    and (ii) the degradation along the thinning ladder and under a rebuilt P, where the
    reference is no longer what the network was trained on.

    Times are restricted to [``t_lo``, ``t_hi``]: the score target carries a
    1/(σ²t(1-t)) factor, the networks are only ever fit on [t_eps, 1-t_eps], and near
    the endpoints the Jacobian diverges and its inverse is meaningless.  Points whose
    -sym∇s is not positive-definite (a trained net carries no such guarantee) are
    dropped and counted in ``frac_valid``.
    """
    assert traj.ndim == 3, f"traj must be (S+1, B, d), got {traj.shape}"
    assert sigma > 0.0, "the score second moment is undefined at sigma = 0"
    S1, B, d = traj.shape
    nan = {"bures": float("nan"), "bures_std": float("nan"),
           "affinity": float("nan"), "trace_ratio": float("nan"),
           "frac_valid": 0.0, "n_points": 0}

    # Same point selection as velocity_fidelity — the states the flow actually visits
    # — but restricted to the interior of the time window.
    rng = np.random.default_rng(seed)
    cols = rng.choice(B, min(max_pts, B), replace=False)
    rows = np.unique(np.round(np.linspace(0, S1 - 1, n_time)).astype(int))
    ts_all = rows.astype(np.float64) / (S1 - 1)
    keep = (ts_all >= t_lo) & (ts_all <= t_hi)
    rows, ts = rows[keep], ts_all[keep]
    if rows.size == 0:
        return nan

    xs = np.asarray(traj[np.ix_(rows, cols)], dtype=np.float64)   # (T, C, d)
    x = torch.as_tensor(xs.reshape(-1, d), dtype=dtype, device=device)
    t = torch.as_tensor(np.repeat(ts, len(cols))[:, None], dtype=dtype, device=device)
    if not torch.isfinite(x).all():
        return nan                                    # a diverged run has no geometry

    # --- Jacobian of the score, one backward pass per output component.  Samples are
    # independent, so summing over the batch before differentiating gives every row's
    # gradient at once: ∂/∂x_b[m] Σ_n s_a(x_n) = ∂s_a(x_m)/∂x_b.
    x_req = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        s = s_net(t, x_req)                                       # (M, d)
        assert s.shape == x.shape, f"s {tuple(s.shape)} vs x {tuple(x.shape)}"
        J = torch.stack(
            [torch.autograd.grad(s[:, a].sum(), x_req, retain_graph=(a < d - 1),
                                 create_graph=False)[0]
             for a in range(d)], dim=1)                           # (M, d, d)  J[:,a,b]
    J = J.detach()

    with torch.no_grad():
        # -sym ∇s = M^{-1}/(σ²t(1-t))  ->  precision, positive-definite if learned
        A = -0.5 * (J + J.transpose(-1, -2))                      # (M, d, d)
        tt = (t * (1.0 - t)).squeeze(1)                           # (M,)
        evals, evecs = torch.linalg.eigh(A.double())
        ok = torch.isfinite(evals).all(-1) & (evals.min(-1).values > eig_floor)
        frac_valid = float(ok.double().mean())
        if int(ok.sum()) < 2:
            return {**nan, "frac_valid": frac_valid, "n_points": int(ok.sum())}

        # M̂ = A^{-1} / (σ² t(1-t)), built from the eigendecomposition
        inv = evecs[ok] @ torch.diag_embed(1.0 / evals[ok]) @ evecs[ok].transpose(-1, -2)
        scale = (sigma ** 2) * tt[ok].double()                    # (K,)
        M_hat = inv / scale[:, None, None]                        # (K, d, d)
        M_hat = 0.5 * (M_hat + M_hat.transpose(-1, -2))

        # reference mobility M_P = ρI + Σ_P at the same points
        M_P = geom_ref.metric_tensor_inv(x[ok]).double()          # (K, d, d)
        M_P = 0.5 * (M_P + M_P.transpose(-1, -2))

        tr = lambda Z: Z.diagonal(dim1=-2, dim2=-1).sum(-1)       # noqa: E731
        tr_hat, tr_P = tr(M_hat), tr(M_P)
        trace_ratio = float((tr_hat / (tr_P + 1e-30)).mean())

        # trace-normalise both to unit trace -> pure shape comparison, d_B ∈ [0, √2]
        d2 = _bures(M_hat / (tr_hat[:, None, None] + 1e-30),
                    M_P / (tr_P[:, None, None] + 1e-30))
        dB = torch.sqrt(d2)
        return {"bures": float(dB.mean()), "bures_std": float(dB.std(unbiased=False)),
                "affinity": float((1.0 - 0.5 * d2).mean()),
                "trace_ratio": trace_ratio, "frac_valid": frac_valid,
                "n_points": int(M_hat.shape[0])}

