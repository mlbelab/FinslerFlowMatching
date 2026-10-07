"""The R2 hindbrain as a three-marginal task: the spaces, the splits, the training set.

The spaces
----------
``d = 2`` is the **UMAP chart, z-scored per axis** and ``d = 20`` / ``d = 50`` are raw
prefixes of the one 50-component PCA :mod:`scripts.core.itracer_data` fits.  That is the
convention of every other benchmark here — erythroid's
:func:`~scripts.experiments.erythroid.preprocess.build_spaces`, the pancreas' and
*C. elegans*' ``UMAP_DIM`` — and it is what lets a hyper-parameter tuned on the mouse
erythroid cloud at ``d = 2`` mean the same thing on a human organoid at ``d = 2``.  The
three columns match the three the erythroid tree searched, so :data:`.bench.POINT_OF` is
the identity and the transplant is across the dataset only.

The prefixes are nested, so the three columns are three truncations of one fit rather than
three fits that could disagree about the cloud.  The pseudotime is **not** recomputed per
column: it is the first 20 components' diffusion map, once
(:data:`scripts.core.itracer_data.PT_PCS`), so widening the reduction to reach ``d = 50``
cannot move the task between columns of the table.

The splits
----------
Selection needs cells that scoring never sees, so the cloud is cut twice.
:func:`split_shown` partitions the two *endpoint* thirds into a train and a val half,
stratified by marginal: a sweep cell fits on the train half with ``P`` and the geometry
rebuilt on it, and the val half supplies the endpoint term of
:mod:`scripts.core.selection`.  :func:`split_holdout` cuts the withheld middle third into a
**selection** slice and a **scored** slice, stratified by pseudotime decile; the tuner
ranks on the former and every reported number is measured on the latter.  Neither slice
ever enters a training set — the whole middle third is withheld from training under every
protocol — so this partition moves selection and scoring only.

The fractions are the Pancreas' :data:`SHOWN_VAL_FRACTION` = 0.15 and
:data:`HOLDOUT_VAL_FRACTION` = 0.10, and not the 0.20/0.20 the 250-cell scGESTALT chain
needs: the three marginals here are 967 cells each, so 10 % of the withheld third is 97
cells to rank on and 870 to report on, and both numbers are large enough that neither the
ranking nor the table is reading noise.

The task
--------
The tertiles of the NPC -> neuron diffusion pseudotime.  ``p_0`` is the early third, ``p_1``
the late third, and the middle third is withheld from everything: from the cells an arm
trains on, from the kNN support of ``K_expr``, from the OT coupling and from the geometry.
The pseudotime is used exactly twice — to cut the tertiles, and to place the withheld third
on the model's time axis at :func:`t_mid` — and it is never an argument to
:func:`~scripts.core.itracer_data.lineage_kernel`.

The ablations
-------------
:data:`P_VARIANTS` is the 2 x 2 of ``{lineage off, on} x {velocity off, on}`` on one
kernel, reached by setting each factor's temperature to zero — at which point that factor
is identically 1 and drops out of the product exactly.  Running our arm on all four is what
turns "the lineage helps" from an assumption into a measurement, and adds the question it
immediately raises: whether a recorded lineage buys anything a *cheaper* directional signal
does not already give, since RNA velocity comes out of the same reads as the expression.
Because each ablation is a temperature rather than a second function, no two of the four
kernels can differ in their support, their bandwidth or their normalisation.

The velocity factor is scVelo's stochastic estimate refitted on whichever cells the
training set is allowed to see (:func:`~scripts.core.itracer_data.velocity_alignment`), so
it is held to the same withholding standard as ``P`` itself.  It is not the pseudotime by
another name: it is read off spliced and unspliced counts, which the tertile cut never
enters.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from scripts.core.itracer_data import (BETA, K_SIGMA, K_SUPPORT, N_PCS, N_THIRDS, NU,
                                       lineage_kernel, load_cloud, velocity_alignment)
from scripts.core.protocol import TrainingSet, marginal_val_split

#: the three marginals, as codes rather than as literal thirds, so nothing downstream has
#: to remember that 1 is the withheld one.
SOURCE, HOLDOUT, TARGET = 0, 1, 2

#: the dimension whose space is the z-scored UMAP chart.  Every other entry of
#: :data:`DIMS` is a PCA prefix.  See :func:`space`.
UMAP_DIM = 2

#: the benchmark spaces: the three the erythroid tree tuned at, so the transplanted point
#: is a transplant across the dataset and not also across the dimension.
DIMS = (UMAP_DIM, 20, 50)
assert max(DIMS) <= N_PCS, f"the cache holds {N_PCS} components, DIMS asks for {max(DIMS)}"

#: fraction of each endpoint third held out of training for the endpoint term of
#: selection, and fraction of the withheld third the tuner is allowed to rank on.
SHOWN_VAL_FRACTION = 0.15
HOLDOUT_VAL_FRACTION = 0.10

#: split seeds.  Fixed per cloud: two rows of a sweep ranked against different val sets
#: are not comparable, so these must not depend on the arm, the point or the model seed.
SHOWN_SPLIT_SEED, HOLDOUT_SPLIT_SEED = 20260910, 20260911

#: integration resolution.  Divisible by three, which :func:`scripts.core.arms.run_arm`
#: asserts so that ``t = k/3`` lands exactly on a step.
N_STEPS = 300

#: the four transition matrices the benchmark compares, as the
#: ``(lineage, velocity)`` temperature pair each is built at.  Both factors of
#: :func:`~scripts.core.itracer_data.lineage_kernel` are :math:`e^{\\theta \\cdot s}` over a
#: score the data supplies, so a temperature of 0 makes that factor identically 1 and it
#: drops out of the product *exactly*.  The four entries are therefore the 2 x 2 of
#: {lineage off, on} x {velocity off, on} on one kernel:
#:
#: ============== ============ =========== =====================================
#: variant        :math:`\\beta` :math:`\\nu` what ``P`` is allowed to know
#: ============== ============ =========== =====================================
#: expr_only      0            0           the expression neighbourhood, alone
#: lineage_kernel BETA         0           + who a cell's recorded relatives are
#: velocity_only  0            NU          + which way a cell is currently moving
#: lineage_velo.  BETA         NU          both
#: ============== ============ =========== =====================================
#:
#: That exactness is the reason an ablation here is a temperature and not a second
#: function.  All four ``P``s share their kNN support, their self-tuning bandwidth, their
#: directedness and their row normalisation, so the *only* thing that differs between two
#: of these columns is which factors are switched on.  Hand-written alternatives would each
#: have to be argued to be the same in every other respect; these cannot fail to be.
P_VARIANTS: dict[str, tuple[float, float]] = {
    "expr_only":        (0.0,  0.0),
    "lineage_kernel":   (BETA, 0.0),
    "velocity_only":    (0.0,  NU),
    "lineage_velocity": (BETA, NU),
}


def space(cloud, dim: int) -> np.ndarray:
    """The whole cloud in the space an arm at ``dim`` is trained and scored in.

    :data:`UMAP_DIM` gets the UMAP chart z-scored per axis, every other ``dim`` the leading
    ``dim`` principal components.  The z-scoring is the whole point of the ``d = 2``
    branch: the two UMAP axes have no common unit, and a blur or a bandwidth transplanted
    from another cloud is only comparable on a standardised chart.
    """
    if dim == UMAP_DIM:
        U = np.asarray(cloud.umap, dtype=np.float64)
        return (U - U.mean(axis=0)) / U.std(axis=0)
    assert dim <= cloud.X.shape[1], f"asked for d={dim}, the cache has {cloud.X.shape[1]} PCs"
    return np.asarray(cloud.X[:, :dim], dtype=np.float64)


def t_mid(cloud) -> float:
    """Model time of the withheld third: its mean pseudotime, rescaled to ``[0, 1]``.

    A flow's time runs linearly from ``p_0`` to ``p_1``, so the withheld population has to
    be placed on that segment rather than assumed to sit at a half.  The three marginals
    are equal tertiles of a rank, which puts it at ``0.5`` to four decimals — a coincidence
    of the cut, and the assertion is what says so rather than the constant.
    """
    mean = {b: float(cloud.s[cloud.marginal(b)].mean()) for b in (SOURCE, HOLDOUT, TARGET)}
    t = (mean[HOLDOUT] - mean[SOURCE]) / (mean[TARGET] - mean[SOURCE])
    assert 0.0 < t < 1.0, (
        f"the withheld third is not between the two endpoints: t = {t:.4f} "
        f"(mean pseudotime {mean[SOURCE]:.4f} / {mean[HOLDOUT]:.4f} / {mean[TARGET]:.4f})")
    return float(t)


def snap(t: float, n_steps: int = N_STEPS) -> float:
    """``t`` moved to the nearest integration step, which is where it can be scored."""
    return round(t * n_steps) / n_steps


def split_shown(cloud) -> tuple[np.ndarray, np.ndarray]:
    """``(train, val)`` cloud indices over the two endpoint thirds, stratified by marginal.

    A sweep cell trains on ``train`` alone — ``P`` and the geometry rebuilt on it — and the
    endpoint term of the objective is scored against the ``p_1`` cells of ``val``.
    """
    shown = cloud.shown
    keep = marginal_val_split(cloud.third[shown] == SOURCE, SHOWN_VAL_FRACTION,
                              SHOWN_SPLIT_SEED)
    return shown[keep], shown[~keep]


def split_holdout(cloud) -> tuple[np.ndarray, np.ndarray]:
    """``(selection, scored)`` cloud indices over the withheld middle third.

    Stratified by pseudotime decile rather than drawn at random: the middle third is the
    thing being predicted and it is spread along the very axis the flow is scored on, so an
    unstratified draw can under-sample one end of it and rank the sweep against a shifted
    target.  The tuner sees only ``selection``; every reported number — the arms, the
    sampling floor and both controls — is measured on ``scored``, so no hyper-parameter was
    chosen against the cells it is reported on.
    """
    held = cloud.marginal(HOLDOUT)
    order = np.argsort(cloud.s[held], kind="stable")
    rng = np.random.default_rng(HOLDOUT_SPLIT_SEED)
    sel: list[int] = []
    for bin_ in np.array_split(order, 10):
        n = int(min(max(round(HOLDOUT_VAL_FRACTION * len(bin_)), 1), len(bin_) - 1))
        sel += list(rng.permutation(bin_)[:n])
    mask = np.zeros(len(held), dtype=bool)
    mask[np.asarray(sel, dtype=int)] = True
    return held[mask], held[~mask]


def build_training_set(cloud, dim: int, subset: np.ndarray | None = None,
                       p_variant: str = "lineage_kernel") -> TrainingSet:
    """Everything an arm may see: the two endpoint thirds, and ``P`` rebuilt over them.

    ``P`` is :func:`~scripts.core.itracer_data.lineage_kernel` refitted on the kept cells
    *in this space* — so no row of it was fitted at a withheld cell, no edge of its kNN
    support runs through one, and its self-tuning bandwidth never saw one either.

    ``subset`` narrows the kept cells further and is how a sweep cell trains on the train
    half of :func:`split_shown`: the kernel is rebuilt on that half too, so the val cells
    are absent from the geometry and not merely from the loss.

    ``p_variant`` selects the ``(beta, nu)`` temperature pair from :data:`P_VARIANTS`,
    which is the ablation: every variant is the *same call* with one or both factors turned
    off, so the difference between two columns is those factors and nothing else.
    ``K_lineage_mean_on_edges`` and ``K_velocity_mean_on_edges`` are exactly 1.0 wherever a
    factor is off, and are what a reader should check before believing a comparison.

    The velocity alignment is refitted on ``idx`` too (:func:`velocity_alignment`), so the
    third factor is held to the same standard as the other two: no withheld cell smoothed
    it, and in a sweep cell no val cell did either.  It is built only when ``nu`` is
    non-zero — an expression-only or lineage-only column never touches the velocity
    artifact, and so does not require it to exist.
    """
    assert p_variant in P_VARIANTS, \
        f"unknown P variant {p_variant!r}; expected one of {sorted(P_VARIANTS)}"
    idx = cloud.shown if subset is None else np.sort(np.asarray(subset, dtype=int))
    assert not (cloud.third[idx] == HOLDOUT).any(), "a withheld cell reached the arms"

    beta, nu = P_VARIANTS[p_variant]
    X = space(cloud, dim)[idx]
    C = velocity_alignment(idx) if nu else None
    P, K, L, V = lineage_kernel(X, cloud.barcode[idx], cloud.scar[idx], beta=beta,
                                C=C, nu=nu)
    third = cloud.third[idx]
    p0, p1 = third == SOURCE, third == TARGET
    edge = K > 0
    return TrainingSet(
        name=f"iTracer-R2-d{dim}", idx=idx, X=X, P=sp.csr_matrix(P), p0=p0, p1=p1,
        rho=0.0, holdout_band=HOLDOUT, p_variant=p_variant,
        diag={"n_train": int(len(idx)), "n_p0": int(p0.sum()), "n_p1": int(p1.sum()),
              "n_withheld": int((cloud.third == HOLDOUT).sum()),
              "k_support": K_SUPPORT, "k_sigma": K_SIGMA, "beta": beta, "nu": nu,
              "support_density": float(edge.mean()),
              "P_nnz_per_row": float((P > 0).sum() / P.shape[0]),
              "K_lineage_mean_on_edges": float(L[edge].mean()),
              "K_velocity_mean_on_edges": float(V[edge].mean()),
              "velocity_align_on_edges": None if C is None else float(C[edge].mean()),
              "direct_p0_p1_edges": int((P[np.ix_(p0, p1)] > 0).sum()),
              "direct_p0_p1_possible": int(p0.sum() * p1.sum())})


__all__ = ["DIMS", "HOLDOUT", "HOLDOUT_VAL_FRACTION", "N_STEPS", "N_THIRDS", "P_VARIANTS",
           "SOURCE", "SHOWN_VAL_FRACTION", "TARGET", "UMAP_DIM", "build_training_set",
           "load_cloud", "snap", "space", "split_holdout", "split_shown", "t_mid",
           "velocity_alignment"]
