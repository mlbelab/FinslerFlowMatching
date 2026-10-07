"""The erythroid cloud, its three latent-time marginals, and the P rebuilt without the middle.

Ported from Petrović et al. (2025), *Curly Flow Matching*, §E.2 and
``notebooks/{2d,nd}_mouse_erythroid.ipynb``.  9815 mouse gastrulation cells subset to
the erythroid lineage (Pijuan-Sala et al., 2019), pre-processed by UniTVelo in
:mod:`scripts.experiments.erythroid.preprocess` — see that module for why it runs in a
separate interpreter and for the d = 2 / d > 2 space definitions.

The task
--------
Their cell 7 bins UniTVelo's unified ``latent_time`` into **three equal-quantile
marginals** and withholds the middle one::

    edges  = np.quantile(latent, np.linspace(0, 1, 4))
    bin_id = np.digitize(latent, edges[1:-1], right=False)
    train_ts = [0, 2];  test_ts = [1]

so the model sees marginal 0 and marginal 2, is integrated from 0 for half of its time
axis, and is scored at that midpoint against marginal 1.  Equal-quantile bins make the
three marginals equal in size by construction (3272 / 3272 / 3271 at N = 9815).

What is rebuilt, and why
------------------------
Everything our side reads is rebuilt on the **survivors** — marginals 0 and 2 only: the
transition matrix ``P``, the kNN bandwidth inside it, and hence the Finsler geometry,
whose Σ and drift are moments of that ``P``.  A ``P`` built on all three marginals and
then sliced would carry the withheld cells' neighbourhoods in its bandwidth and their
velocities in its flux, at exactly the midpoint we are scored at.

The reference has no equivalent step because Curly-FM never builds a graph over the
cloud: it reads the per-cell velocity vectors directly and only ever queries
:func:`~scripts.experiments.erythroid.reference_release.get_ut_knn_gaussian` with the
*training* marginals as the neighbour pool.  Note the asymmetry this leaves in the
baseline's favour: Curly-FM is handed the raw 9815×d velocity field, ours sees it only
after the kernel has coarsened it into a 15-neighbour row-stochastic ``P``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from scripts.core.paths import data_dir
from scripts.core.protocol import TrainingSet
from scripts.core.transition import (P_KNN_K, VELOCITY_SOFTMAX_SCALE,
                                     asymmetry_scale, build_velocity_kernel)

#: where :mod:`scripts.experiments.erythroid.preprocess` writes its output.  The three
#: ``.npz`` are shipped under ``<public>/data/`` -- 9 MB, so this benchmark runs from a
#: fresh clone without the TensorFlow-pinned UniTVelo environment.
CACHE_DIR = os.environ.get("ERYTHROID_CACHE", data_dir("erythroid_cache"))

#: the dimensions their Table 4 reports.  2 is the standardised UMAP; 20 and 50 are
#: unstandardised PCA prefixes.  See :mod:`scripts.experiments.erythroid.preprocess`.
DIMS = (2, 20, 50)
#: their ``num_times`` — three equal-quantile latent-time bins.
NUM_TIMES = 3
#: their ``train_ts`` / ``test_ts``: the middle marginal is the held-out target.
TRAIN_BINS = (0, 2)
HOLDOUT_BIN = 1

#: The middle marginal is cut once more, into a **selection** slice supplying the
#: ``W2_intermediate`` term of :mod:`scripts.core.selection` and a **scored** slice every
#: reported number is measured on.
#:
#: 10 % and not 1 %, measured rather than guessed: scoring the 15 saved t = 1/2 slabs
#: against a fixed slice reproduces the full-marginal ordering at Spearman 0.990 / 0.999 /
#: 0.999 (d = 2 / 20 / 50) at 10 %, against 0.955 / 0.971 / 0.977 at 1 %.  A small slice
#: also biases the distance upward — harmless for ranking, since the slice is fixed, but
#: it is why the scored slice is the large half.
HOLDOUT_VAL_FRACTION = 0.10
HOLDOUT_SPLIT_SEED = 0
#: latent-time strata the val slice is drawn proportionally from: the middle marginal is a
#: contiguous quantile band, so an unstratified draw can land at one end of it.
N_HOLDOUT_STRATA = 10

#: kNN width of *our* transition matrix: the engine-wide ``P_KNN_K``, kept rather than
#: raised to their 30 so P is the same object here as everywhere else in the paper.  Their
#: k = 30 is the width of *their* reference field estimator and stays at 30 in evaluation.
P_K = P_KNN_K


@dataclass
class ErythroidCloud:
    """One dimension's view of the cloud, with the marginals already separated."""

    dim: int
    X: np.ndarray                    # (N, d) all cells
    velocity: np.ndarray             # (N, d) UniTVelo field in the same space
    latent_time: np.ndarray          # (N,)   UniTVelo unified time
    bin_id: np.ndarray               # (N,)   0 / 1 / 2, equal-quantile
    celltype: np.ndarray             # (N,)   lineage stage strings, figures only
    umap: np.ndarray                 # (N, 2) raw UMAP, figures only
    diag: dict = field(default_factory=dict)

    def marginal(self, b: int) -> np.ndarray:
        return np.flatnonzero(self.bin_id == b)

    @property
    def shown(self) -> np.ndarray:
        """Indices of everything a model is allowed to see: marginals 0 and 2."""
        return np.flatnonzero(np.isin(self.bin_id, TRAIN_BINS))

    @property
    def target(self) -> np.ndarray:
        """The withheld middle marginal — evaluation only."""
        return self.marginal(HOLDOUT_BIN)

    @property
    def target_val(self) -> np.ndarray:
        """The 10 % of the withheld marginal the tuner may rank on."""
        return split_holdout(self)[0]

    @property
    def target_test(self) -> np.ndarray:
        """The 90 % every reported number is scored against."""
        return split_holdout(self)[1]


def load_cloud(dim: int, cache_dir: str = CACHE_DIR) -> ErythroidCloud:
    """Read one dimension's cache and cut the three equal-quantile marginals.

    The quantile edges are computed on the **full** cloud, as theirs are: the binning is
    a property of the dataset, not of any split, and recomputing it on survivors would
    silently change which cells the middle marginal contains.
    """
    path = os.path.join(cache_dir, f"erythroid_d{dim}.npz")
    assert os.path.exists(path), (
        f"missing {path}\nRun the UniTVelo step first:\n"
        f"    conda run -n unitvelo python -m scripts.experiments.erythroid.preprocess")
    z = np.load(path, allow_pickle=True)

    X = np.asarray(z["X"], dtype=np.float64)
    V = np.asarray(z["velocity"], dtype=np.float64)
    latent = np.asarray(z["latent_time"], dtype=np.float64)
    assert X.shape == V.shape, f"X {X.shape} vs velocity {V.shape}"
    assert X.shape[1] == dim, f"cache says d={X.shape[1]}, asked for {dim}"
    assert latent.shape == (X.shape[0],), f"latent_time {latent.shape} vs X {X.shape}"

    edges = np.quantile(latent, np.linspace(0, 1, NUM_TIMES + 1))
    bin_id = np.digitize(latent, edges[1:-1], right=False)
    counts = np.bincount(bin_id, minlength=NUM_TIMES)
    assert (counts > 0).all(), f"an empty latent-time bin: {counts}"

    return ErythroidCloud(
        dim=dim, X=X, velocity=V, latent_time=latent, bin_id=bin_id,
        celltype=np.asarray(z["celltype"]), umap=np.asarray(z["umap"], dtype=np.float64),
        diag={"n_cells": int(X.shape[0]), "bin_counts": counts.tolist(),
              "quantile_edges": edges.tolist(),
              "velocity_norm_median": float(np.median(np.linalg.norm(V, axis=1)))},
    )


def build_training_set(cloud: ErythroidCloud, subset: np.ndarray | None = None,
                       k: int = P_K,
                       softmax_scale: float = VELOCITY_SOFTMAX_SCALE) -> TrainingSet:
    """Assemble what an arm may see, with ``P`` **rebuilt** on exactly those cells.

    ``subset`` restricts further — the selection split passes the tuning 85 % here, so the
    swept arm's geometry is rebuilt on its own reduced pool rather than borrowing
    neighbourhood structure from cells it is about to be ranked against.  ``None`` means
    all shown cells, i.e. the refit used for the reported numbers.

    ``P`` is :func:`~scripts.core.transition.build_velocity_kernel`: CellRank's
    ``VelocityKernel`` in deterministic mode, ``P_ij ∝ exp(λ cos(v_i, x_j - x_i))`` over
    the k nearest neighbours at the engine-wide λ = 4.  This cloud arrives with a measured
    velocity, which is the condition that makes the velocity kernel the right builder.
    """
    idx = cloud.shown if subset is None else np.asarray(subset, dtype=np.int64)
    assert idx.ndim == 1 and len(idx) > k + 1, f"need > k+1 cells, got {len(idx)}"
    assert not np.isin(cloud.bin_id[idx], HOLDOUT_BIN).any(), (
        "a withheld middle-marginal cell reached the training set")

    X = cloud.X[idx]
    V = cloud.velocity[idx]
    P = build_velocity_kernel(X, V, k=k, softmax_scale=softmax_scale)

    bins = cloud.bin_id[idx]
    p0 = bins == TRAIN_BINS[0]
    p1 = bins == TRAIN_BINS[1]
    assert p0.sum() > 0 and p1.sum() > 0, "a training marginal came out empty"

    diag = {
        "n_train": int(len(idx)),
        "n_p0": int(p0.sum()), "n_p1": int(p1.sum()),
        "P_k": k, "P_softmax_scale": softmax_scale,
        "P_asymmetry": float(asymmetry_scale(P)),
        "cos_J_velocity": float(_flux_alignment(P, X, V)),
    }
    return TrainingSet(
        name=f"Erythroid-d{cloud.dim}", idx=idx, X=X, P=sp.csr_matrix(P),
        p0=p0, p1=p1, rho=0.0, holdout_band=HOLDOUT_BIN,
        p_variant="velocity_kernel", diag=diag, velocity=V,
    )


def _flux_alignment(P: sp.csr_matrix, X: np.ndarray, V: np.ndarray) -> float:
    """Mean cos(J_i, v_i): does P's antisymmetric first moment point where v points?

    ``J_i = Σ_j P^asym_ij (x_j - x_i)`` is the only channel through which our geometry can
    learn the direction of the field, so this is the precondition for any claim that the
    drift did something.  Reported per run, never assumed.
    """
    A = (P - P.T) * 0.5
    J = A @ X - sp.diags(np.asarray(A.sum(axis=1)).ravel()) @ X
    nj = np.linalg.norm(J, axis=1)
    nv = np.linalg.norm(V, axis=1)
    ok = (nj > 1e-12) & (nv > 1e-12)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.sum(J[ok] * V[ok], axis=1) / (nj[ok] * nv[ok])))


def split_shown(cloud: ErythroidCloud, val_fraction: float,
                split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val partition of the shown cells, for hyper-parameter selection.

    Stratified **by marginal**: every arm here is endpoint-coupled, so removing an
    unbalanced share of one marginal changes the coupling rather than just the sample
    size.  Returns global indices, both drawn from ``cloud.shown``.
    """
    from scripts.core.protocol import marginal_val_split

    shown = cloud.shown
    is_p0 = cloud.bin_id[shown] == TRAIN_BINS[0]
    train_mask = marginal_val_split(is_p0, val_fraction, split_seed)
    return shown[train_mask], shown[~train_mask]


def split_holdout(cloud: ErythroidCloud,
                  val_fraction: float = HOLDOUT_VAL_FRACTION,
                  split_seed: int = HOLDOUT_SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Cut the withheld middle marginal into a selection slice and a scored slice.

    Returns ``(val, test)`` as global indices, disjoint and together exactly
    ``cloud.target``.  Neither ever enters a training set, so this partition changes
    *selection and scoring* only.

    Stratified by **latent-time decile within the band**: the marginal is one contiguous
    quantile slab of a developmental time axis, so a uniform draw can concentrate the val
    slice near one edge of it.  :func:`scripts.core.protocol.marginal_val_split` is not
    reused, since it partitions a boolean membership rather than a continuous band.

    The seed is a *split* seed and must not depend on the arm, the grid point or the model
    seed, or two rows of the sweep would be ranked against different reference clouds.
    """
    assert 0.0 < val_fraction < 1.0, f"val_fraction must be in (0, 1), got {val_fraction}"
    tgt = cloud.target
    assert len(tgt) >= 2 * N_HOLDOUT_STRATA, f"middle marginal too small: {len(tgt)}"

    rng = np.random.default_rng(split_seed)
    order = np.argsort(cloud.latent_time[tgt], kind="stable")
    picks = []
    for stratum in np.array_split(order, N_HOLDOUT_STRATA):
        n_v = max(1, int(round(val_fraction * len(stratum))))
        picks.append(rng.choice(stratum, size=n_v, replace=False))

    is_val = np.zeros(len(tgt), dtype=bool)
    is_val[np.concatenate(picks)] = True
    return tgt[is_val], tgt[~is_val]
