"""The Pancreas cloud, its three latent-time marginals, and the P rebuilt without the middle.

scVelo's endocrinogenesis set (Bastidas-Ponce et al., 2019; ``endocrinogenesis_day15``):
3696 cells, dynamical-mode velocities, top-50 PCA.  The heavy step — ``recover_dynamics``
plus the CellRank kernels — was run once by :mod:`scripts.core.scvelo_data` and cached, so
everything here reads a 5 MB ``.npz`` and needs neither scVelo nor CellRank installed.

The task
--------
The **erythroid protocol**: a held-out middle marginal, and a ``P`` rebuilt on the
survivors rather than sliced out of a full-cloud one.  ``latent_time`` is
cut into three equal-quantile marginals, the middle one is **withheld**, and a model
trained on marginals 0 and 2 is scored at :math:`t = 1/2` against it.  At N = 3696 the
bins are 1232 each and they land on the biology rather than across it::

    bin 0  (t <= 0.294)   Ductal 916 + Ngn3 low 259 + Ngn3 high 57     the progenitor pool
    bin 1  (0.294..0.816) Ngn3 high 585 + Pre-endocrine 543 + 104 fated   *** withheld ***
    bin 2  (t >= 0.816)   Beta 534 + Alpha 470 + Epsilon 110 + Delta 69   the terminal fates

So the held-out marginal is exactly the endocrine differentiation step — Ngn3-high
progenitors committing through the pre-endocrine stage — and not an arbitrary slab.  A
method that reproduces it has recovered the transition; one that chords from ductal cells
to islets straight through PCA space has not.

Why not the cluster-marginal setup
------------------------------------
That driver takes the marginals from *cluster* labels (Ductal → the four fates) and hands
the geometry the **full 3696-cell CellRank ``P``**.  Both are fine for what it does, and
neither is usable here.  Cluster marginals leave no withheld middle to score against, and
a ``P`` built over all cells carries the withheld cells' neighbourhoods in its bandwidth
and their velocities in its flux — into precisely the region we are scored at.  So the
cached ``ds.P`` is **discarded** by :func:`load_cloud` and ``P`` is rebuilt on the
survivors by :func:`build_training_set`, which is what closes that leak.  Everything else
downstream — bandwidth, flux, Σ, the metric — is a function of that rebuilt ``P`` and is
therefore rebuilt with it.

Four spaces, four searches
--------------------------
The notebook reports :data:`DIMS` = ``(2, 10, 20, 50)``.  The last three are nested PCA
prefixes; ``d = 2`` is the z-scored UMAP chart with ``velocity_umap``, which is a
different object and is labelled as one.  **Every column is searched in its own space**,
including that one.

An earlier version of this tree ran the search at the three prefixes and let ``d = 2``
inherit ``d = 50``'s point, on the evidence that ours had picked the same ``rho_mult`` at
all three.  Searching it settled the question the other way: the UMAP column's argmin is
nowhere near the prefixes', and its score surface is *ragged in* ``rho`` rather than
monotone (see :mod:`scripts.experiments.pancreas.tune`).  The inheritance had been reasoning from
three points of one kind about a fourth of a different kind, which a z-scored 2-D
embedding is; it was retired rather than argued for.

The asymmetry this leaves is the one the velocity-kernel benchmarks and
:mod:`scripts.experiments.erythroid` also run under, and it is in the baseline's favour: Curly-FM
is handed the raw per-cell velocity field, ours sees it only after the kernel has
coarsened it into a 15-neighbour row-stochastic ``P``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from scripts.core.metrics import local_moment_fidelity
from scripts.core.paths import data_dir
from scripts.core.protocol import TrainingSet, marginal_val_split
from scripts.core.scvelo_data import load_pancreas
from scripts.core.transition import (P_KNN_K, VELOCITY_SOFTMAX_SCALE, asymmetry_scale,
                                     build_velocity_kernel)

#: where :func:`scripts.core.scvelo_data.load_pancreas` wrote its cache.  Shipped under
#: ``<public>/data/`` -- 5 MB, and it is what makes this notebook runnable from a fresh
#: clone with no scVelo and no h5ad.  Overridable so the tree can be smoke-tested against
#: a synthetic cloud without the real data, the same escape hatch ``ERYTHROID_CACHE``
#: gives that benchmark.
CACHE_DIR = os.environ.get("PANCREAS_CACHE", data_dir("pancreas_cache"))

#: the cache's own build parameters — they name the file and nothing else reads them.
N_PCS, VEL_WEIGHT, VELO_MODE = 50, 0.8, "dynamical"

#: PCA prefixes the notebook reports.  Nested by construction (``X_10 == X_50[:, :10]``),
#: since PCA axes are ordered by explained variance, so those three columns differ by
#: *how much* of the same space a method was given and not by which space.
PCA_DIMS = (10, 20, 50)

#: ``d = 2`` is the **UMAP embedding**, z-scored per axis, with ``velocity_umap`` as its
#: field — the erythroid tree's convention for its own 2-D column, and the chart this
#: dataset is conventionally read in.  It is *not* a prefix of the PCA space and
#: nothing here pretends it is: it is a different chart of the same cells, reported beside
#: the prefixes because it is the space the biology is usually read in and the only one a
#: reader can check by eye.  z-scoring is what makes ``rho`` and ``sigma`` — an absolute
#: floor on squared displacements and an absolute noise level — comparable to the
#: unstandardised prefixes at all.
UMAP_DIM = 2

#: every column the notebook reports, in table order.
DIMS = (UMAP_DIM,) + PCA_DIMS

#: the columns the hyper-parameter search runs at — all of them.  Kept as its own name,
#: rather than folded into :data:`DIMS`, because it is what ``tune.py`` fans out over and
#: what ``verify`` pins: a column that is reported but not searched would be a hole in the
#: record, and the two names being equal is the check that there is none.
TUNED_DIMS = DIMS

#: three equal-quantile ``latent_time`` bins; the middle one is the held-out target.
NUM_TIMES = 3
TRAIN_BINS = (0, 2)
HOLDOUT_BIN = 1

#: The withheld marginal is cut once more, into a **selection** slice and a **scored**
#: slice.  This is what supplies the midpoint term of :mod:`scripts.core.selection`: the
#: tuner ranks on 10 % of the target and every reported number is measured on the disjoint
#: 90 %.  Selection therefore sees route information and the notebook says so; what it
#: never sees is a *reported* cell, which is the guarantee ``verify`` asserts.
HOLDOUT_VAL_FRACTION = 0.10
HOLDOUT_SPLIT_SEED = 0
#: latent-time strata the val slice is drawn proportionally from.  The middle marginal is
#: one contiguous quantile band of a developmental axis, so an unstratified draw can land
#: at one end of it and rank hyper-parameters on the start of the transition alone.
N_HOLDOUT_STRATA = 10

#: the leak-free control's split of the *kept* marginals — no withheld cell is involved.
SHOWN_VAL_FRACTION = 0.15
SHOWN_SPLIT_SEED = 0

#: kNN width of our transition matrix: the engine-wide 15, as everywhere else.
P_K = P_KNN_K


@dataclass
class PancreasCloud:
    """One space's view of the cloud, with the marginals already separated."""

    dim: int
    X: np.ndarray                    # (N, d) PCA prefix, or z-scored UMAP at d = 2
    velocity: np.ndarray             # (N, d) RNA velocity in the same space
    latent_time: np.ndarray          # (N,)   scVelo dynamical latent time in [0, 1]
    bin_id: np.ndarray               # (N,)   0 / 1 / 2, equal-quantile
    celltype: np.ndarray             # (N,)   cluster labels; scoring + figures
    umap: np.ndarray                 # (N, 2) figures only
    diag: dict = field(default_factory=dict)

    def marginal(self, b: int) -> np.ndarray:
        return np.flatnonzero(self.bin_id == b)

    @property
    def shown(self) -> np.ndarray:
        """Indices of everything a model may see: marginals 0 and 2."""
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


def load_cloud(dim: int, cache_dir: str = CACHE_DIR) -> PancreasCloud:
    """Read the cache, take the ``dim``-column PCA prefix, cut the three marginals.

    ``load_pancreas`` returns from its cache before it looks at an h5ad
    (:mod:`scripts.core.scvelo_data`), which is why the raw file path below is never
    opened and why this tree needs no single-cell stack.  The assert fires first so a
    missing cache is a message and not a five-minute scVelo import.

    The quantile edges are computed on the **full** cloud: the binning is a property of
    the dataset, not of a split, and recomputing it on survivors would silently change
    which cells the withheld marginal contains.
    """
    assert dim == UMAP_DIM or 1 <= dim <= N_PCS, (
        f"dim must be {UMAP_DIM} (UMAP) or in [1, {N_PCS}] (PCA prefix), got {dim}")
    cache = os.path.join(cache_dir, f"pancreas_d{N_PCS}_vw{VEL_WEIGHT}_{VELO_MODE}.npz")
    assert os.path.exists(cache), (
        f"missing {cache}\nBuild it once with the single-cell stack installed:\n"
        f"    python -c \"from scripts.core.scvelo_data import load_pancreas; \"\n"
        f"      \"load_pancreas('<endocrinogenesis_day15.h5ad>', cache_dir='{cache_dir}')\"")
    ds, meta = load_pancreas("", n_pcs=N_PCS, vel_weight=VEL_WEIGHT, mode=VELO_MODE,
                             cache_dir=cache_dir)

    umap = np.asarray(meta.umap, dtype=np.float64)
    if dim == UMAP_DIM:
        # the UMAP chart, z-scored per axis.  The *same* affine map is applied to the
        # field, so a velocity still points where it pointed -- dividing coordinates and
        # velocities by different numbers would rotate the flow relative to the cloud.
        scale = umap.std(axis=0)
        assert (scale > 0).all(), f"a degenerate UMAP axis: {scale}"
        X = (umap - umap.mean(axis=0)) / scale
        V = np.asarray(meta.velocity_umap, dtype=np.float64) / scale
    else:
        X = np.asarray(ds.X[:, :dim], dtype=np.float64)
        V = np.asarray(meta.velocity_pca[:, :dim], dtype=np.float64)
    latent = np.asarray(meta.latent_time, dtype=np.float64)
    assert X.shape == V.shape, f"X {X.shape} vs velocity {V.shape}"
    assert latent.shape == (X.shape[0],), f"latent_time {latent.shape} vs X {X.shape}"
    # ``ds.P`` is the full-cloud CellRank matrix and is deliberately left behind here;
    # see the module docstring.  ``build_training_set`` builds the only P this tree uses.

    edges = np.quantile(latent, np.linspace(0, 1, NUM_TIMES + 1))
    bin_id = np.digitize(latent, edges[1:-1], right=False)
    counts = np.bincount(bin_id, minlength=NUM_TIMES)
    assert (counts > 0).all(), f"an empty latent-time bin: {counts}"

    return PancreasCloud(
        dim=dim, X=X, velocity=V, latent_time=latent, bin_id=bin_id,
        celltype=np.asarray(meta.clusters), umap=umap,
        diag={"n_cells": int(X.shape[0]), "bin_counts": counts.tolist(),
              "space": "umap_zscored" if dim == UMAP_DIM else "pca_prefix",
              "quantile_edges": edges.tolist(),
              "velocity_norm_median": float(np.median(np.linalg.norm(V, axis=1)))},
    )


def build_training_set(cloud: PancreasCloud, subset: np.ndarray | None = None,
                       k: int = P_K,
                       softmax_scale: float = VELOCITY_SOFTMAX_SCALE) -> TrainingSet:
    """Assemble what an arm may see, with ``P`` **rebuilt** on exactly those cells.

    ``subset`` restricts further — the leak-free control passes its training 85 % here, so
    the swept arm's geometry is rebuilt on its own reduced pool rather than borrowing
    neighbourhood structure from cells it is about to be ranked against.  ``None`` means
    "all shown cells", the refit the reported numbers use.

    ``P`` is :func:`~scripts.core.transition.build_velocity_kernel`: CellRank's
    ``VelocityKernel`` in deterministic mode, :math:`P_{ij} \\propto \\exp(\\lambda
    \\cos(v_i, x_j - x_i))` over the k nearest neighbours.  This cloud arrives *with* a
    measured field, which is the condition that selects the velocity kernel over the
    uniform kNN one a purely geometric kernel uses.
    """
    idx = cloud.shown if subset is None else np.asarray(subset, dtype=np.int64)
    assert idx.ndim == 1 and len(idx) > k + 1, f"need > k+1 cells, got {len(idx)}"
    assert not np.isin(cloud.bin_id[idx], HOLDOUT_BIN).any(), (
        "a withheld middle-marginal cell reached the training set")

    X, V = cloud.X[idx], cloud.velocity[idx]
    P = build_velocity_kernel(X, V, k=k, softmax_scale=softmax_scale)

    bins = cloud.bin_id[idx]
    p0, p1 = bins == TRAIN_BINS[0], bins == TRAIN_BINS[1]
    assert p0.sum() > 0 and p1.sum() > 0, "a training marginal came out empty"

    diag = {
        "n_train": int(len(idx)), "n_p0": int(p0.sum()), "n_p1": int(p1.sum()),
        "P_k": k, "P_softmax_scale": softmax_scale,
        # both cheap, both reported rather than assumed: how directed P actually is, and
        # how well its flux recovers the field it was built from.
        "P_asymmetry": float(asymmetry_scale(P)),
        "cos_J_velocity": float(_flux_alignment(P, X, V)),
    }
    return TrainingSet(
        name=f"Pancreas-d{cloud.dim}", idx=idx, X=X, P=sp.csr_matrix(P),
        p0=p0, p1=p1, rho=0.0, holdout_band=HOLDOUT_BIN,
        p_variant="velocity_kernel", diag=diag, velocity=V,
    )


# --------------------------------------------------------------------------- #
#  The evaluation-only full-cloud transition matrix, and R_1 / R_2 against it
# --------------------------------------------------------------------------- #
#: keyed by ``(dim, k, softmax_scale)``.  A 3696-cell velocity kernel is a second of
#: work, but the ruler cell asks for it once per arm per dimension and the rows must be
#: byte-identical across arms or the comparison is between two references.
_FULL_P_CACHE: dict[tuple, sp.csr_matrix] = {}


def full_transition(cloud: PancreasCloud, k: int = P_K,
                    softmax_scale: float = VELOCITY_SOFTMAX_SCALE) -> sp.csr_matrix:
    """``P`` over **every** cell, including the withheld marginal — scoring only.

    This is the one object in this module built on cells the models never see, and it
    exists for exactly one reason: :func:`holdout_moment_fidelity` needs a reference row
    *at* a held-out node, and :attr:`TrainingSet.P` has no such row by construction —
    :func:`build_training_set` asserts it.  It is the pancreas counterpart of the Sheet's
    ``ds.P_full``.

    **It must never reach a model.**  Nothing in this module passes it to
    :class:`TrainingSet`, and ``verify`` fails if a training set's ``P`` ever matches this
    one's shape.  The same rule the module docstring states about the cached CellRank
    matrix applies here and for the same reason: a ``P`` carrying the withheld cells'
    neighbourhoods and velocities would hand a method the answer to the question it is
    being asked.  Reading it *after* training, as a yardstick, leaks nothing.

    Same construction as the training ``P`` — same kernel, same ``k``, same tilt — so the
    only difference between the two is which cells are in them, which is what makes the
    comparison a measurement rather than an artefact of two kernels.
    """
    key = (cloud.dim, int(k), float(softmax_scale))
    if key not in _FULL_P_CACHE:
        P = build_velocity_kernel(cloud.X, cloud.velocity, k=k,
                                  softmax_scale=softmax_scale)
        _FULL_P_CACHE[key] = sp.csr_matrix(P)
    return _FULL_P_CACHE[key]


def holdout_moment_fidelity(traj: np.ndarray, cloud: PancreasCloud,
                            h: float | None = None, seed: int = 0, **kw) -> dict:
    """``R_1`` / ``R_2`` at the withheld cells, against the raw rows of
    :func:`full_transition`.

    The nodes are ``cloud.target_test`` — the 90 % of the middle marginal every reported
    number is scored on — so neither side of this comparison was available during
    training or during selection.  The reference rows are unsmoothed: no geometry object,
    no kernel in front of ``P``, nothing the model was fitted through.  See
    :func:`scripts.core.metrics.local_moment_fidelity` for what the two cosines are and
    why coverage is reported beside them.

    The bandwidth defaults to three median nearest-neighbour spacings of the full cloud,
    the same rule the Sheet's adapter uses, so ``h`` scales with the space rather than
    with its units — which matters here, where ``d = 2`` is a z-scored chart and the
    prefixes are not.
    """
    idx = cloud.target_test
    if h is None:
        # k=2 because the first neighbour of a cloud point is itself
        h = float(3.0 * np.median(cKDTree(cloud.X).query(cloud.X, k=2)[0][:, 1]))
    return local_moment_fidelity(traj, cloud.X[idx], full_transition(cloud)[idx],
                                 cloud.X, h=h, seed=seed, **kw)


def _flux_alignment(P: sp.csr_matrix, X: np.ndarray, V: np.ndarray) -> float:
    """Mean ``cos(J_i, v_i)``: does P's antisymmetric first moment point where v points?

    ``J_i = Σ_j P^asym_ij (x_j - x_i)`` is the only channel through which our geometry can
    learn the direction of the field, so this is the precondition for any claim that the
    drift did something.  Same construction as the erythroid tree's, and reported per run.
    """
    A = (P - P.T) * 0.5
    J = A @ X - sp.diags(np.asarray(A.sum(axis=1)).ravel()) @ X
    nj, nv = np.linalg.norm(J, axis=1), np.linalg.norm(V, axis=1)
    ok = (nj > 1e-12) & (nv > 1e-12)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.sum(J[ok] * V[ok], axis=1) / (nj[ok] * nv[ok])))


def split_shown(cloud: PancreasCloud, val_fraction: float = SHOWN_VAL_FRACTION,
                split_seed: int = SHOWN_SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val partition of the *kept* cells.

    Two jobs, both load-bearing.  The train half is what a sweep cell fits on, with ``P``
    and the geometry rebuilt on it; the val half supplies the **endpoint** term of
    :mod:`scripts.core.selection`, so that term is measured out of sample rather than on
    cells the arm was trained to reach.  The notebook then refits the chosen point on all
    kept cells, so the split costs selection 15 % of the data and costs the reported
    number nothing.

    Stratified by marginal, for the reason :func:`scripts.core.protocol.marginal_val_split`
    gives: every arm here is endpoint-coupled, so removing an unbalanced share of one
    marginal changes the coupling rather than just the sample size.  Returns global
    indices, both drawn from ``cloud.shown``; the withheld marginal is in neither.
    """
    shown = cloud.shown
    is_p0 = cloud.bin_id[shown] == TRAIN_BINS[0]
    train_mask = marginal_val_split(is_p0, val_fraction, split_seed)
    return shown[train_mask], shown[~train_mask]


def split_holdout(cloud: PancreasCloud, val_fraction: float = HOLDOUT_VAL_FRACTION,
                  split_seed: int = HOLDOUT_SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Cut the withheld marginal into a selection slice and a scored slice.

    Returns ``(val, test)`` as global indices, disjoint and together exactly
    ``cloud.target``.  Neither ever enters a training set — the whole marginal is withheld
    from training under both protocols — so this partition changes *selection and scoring*
    only, which is the entire content of the selection protocol.

    Stratified by **latent-time decile within the band**, so the val slice is a fair
    miniature of what is scored rather than a draw concentrated at one edge of the
    transition.  :func:`~scripts.core.protocol.marginal_val_split` is not reused: it
    partitions a boolean membership into two strata, which is the right rule for
    :func:`split_shown` and the wrong one for a continuous band cut into ten.

    The seed is a *split* seed and must not depend on the arm, the grid point or the model
    seed, or two rows of the sweep would be ranked against different reference clouds.
    """
    assert 0.0 < val_fraction < 1.0, f"val_fraction must be in (0, 1), got {val_fraction}"
    tgt = cloud.target
    assert len(tgt) >= 2 * N_HOLDOUT_STRATA, f"middle marginal too small: {len(tgt)}"

    rng = np.random.default_rng(split_seed)
    order = np.argsort(cloud.latent_time[tgt], kind="stable")   # local, time-ordered
    picks = []
    for stratum in np.array_split(order, N_HOLDOUT_STRATA):
        n_v = max(1, int(round(val_fraction * len(stratum))))
        picks.append(rng.choice(stratum, size=n_v, replace=False))

    is_val = np.zeros(len(tgt), dtype=bool)
    is_val[np.concatenate(picks)] = True
    return tgt[is_val], tgt[~is_val]
