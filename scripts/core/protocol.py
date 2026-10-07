"""What an arm is allowed to see, and how a dataset is seeded.

Two survivors of ``scripts/experiments/synth_suite/protocol.py``, plus one addition.  Everything
else there -- :class:`Split`, the per-band 70/30 draw, the thinning ladder -- was written
against that suite's four-band ``SynthDataset`` and went with it; the Sheet benchmark
builds its own splits in :mod:`scripts.experiments.sheet.split`.  What is here is
dataset-agnostic and is the contract between *any* benchmark and
:mod:`scripts.core.arms`: :func:`dataset_salt`, :class:`TrainingSet`, and
:func:`marginal_val_split`, the train/val partition a marginal benchmark tunes on.  It
lives here rather than inside that package because the rule is dataset-agnostic and an
experiment may not import another experiment: the Sheet's own selection split is a
different object (one fixed 70/15/15 partition of a cloud with a withheld band), and the
circles benchmark uses no split at all -- it is a qualitative port of a reference that
tunes nothing.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


def dataset_salt(name: str) -> int:
    """A stable per-dataset seed offset.

    Emphatically *not* ``hash(name)``: CPython salts string hashing per process
    (PEP 456), so ``hash("Cycle")`` differs on every interpreter start unless
    ``PYTHONHASHSEED`` is pinned.  Seeding the split with it made ``make_split(ds, 0)``
    draw a *different* 70/30 partition in every process — which silently defeated the
    protocol's central assumption that every arm in a cell is scored against the same
    held-out third.  ``crc32`` is fixed by the standard, so this is reproducible across
    processes, machines and Python versions.
    """
    return zlib.crc32(name.encode()) % (2 ** 31)


def marginal_val_split(is_p0: np.ndarray, val_fraction: float,
                       split_seed: int) -> np.ndarray:
    """A train mask over shown cells, stratified by marginal.

    ``is_p0`` is the (M,) boolean that says which shown cells belong to the source
    marginal; the complement is the target.  Each marginal is partitioned separately,
    so a val fraction of 0.15 removes 15 % of *each* rather than 15 % of the union --
    an unstratified draw can leave the two marginals unbalanced, and every arm here is
    endpoint-coupled, so an unbalanced pair changes the coupling itself rather than just
    the sample size.

    The seed is the *split* seed, not the model seed.  On a benchmark that redraws its
    cloud per model seed the partition is redrawn with it (the cloud is new, so there is
    nothing to hold fixed), but within one cloud it must not depend on the arm or on the
    grid point, or two rows of the sweep would be ranked against different val sets.

    Kept general rather than folded into its one caller
    because the rule is the answer to a question any future ported benchmark with a
    selection grid will ask, and because writing it twice is how two benchmarks end up
    stratifying differently without anyone noticing.
    """
    is_p0 = np.asarray(is_p0, dtype=bool)
    assert is_p0.ndim == 1, f"is_p0 must be (M,), got {is_p0.shape}"
    assert 0.0 < val_fraction < 1.0, f"val_fraction must be in (0, 1), got {val_fraction}"

    rng = np.random.default_rng(split_seed)
    train = np.ones(is_p0.shape, dtype=bool)
    for member in (is_p0, ~is_p0):
        idx = np.flatnonzero(member)
        assert len(idx) >= 2, "a marginal with fewer than two cells cannot be split"
        # round rather than floor: at val_fraction 0.15 and n = 2000 both give 300, but
        # floor silently yields an empty val set for a small smoke-run marginal.
        n_val = int(min(max(round(val_fraction * len(idx)), 1), len(idx) - 1))
        train[rng.permutation(idx)[:n_val]] = False
    return train


# --------------------------------------------------------------------------- #
#  Training set assembly
# --------------------------------------------------------------------------- #
@dataclass
class TrainingSet:
    """Everything a method arm is allowed to see, plus rebuild diagnostics."""

    name: str
    idx: np.ndarray                 # global indices of the training cells
    X: np.ndarray                   # (M, d)
    P: sp.csr_matrix                # (M, M) row-stochastic
    p0: np.ndarray                  # (M,) bool  source marginal  (band T0 ∩ pool)
    p1: np.ndarray                  # (M,) bool  target marginal  (band T3 ∩ pool)
    rho: float
    holdout_band: int | None
    p_variant: str
    diag: dict = field(default_factory=dict)
    # (M, d) measured per-cell velocity, when the benchmark supplies one.  Only the
    # arms whose published input *is* a velocity field read it (``curly``); ours never
    # does — we see it only after it has been coarsened into P — so leaving it set
    # cannot advantage us.  ``None`` on every cloud that carries no arrows.
    velocity: np.ndarray | None = None

    def __post_init__(self) -> None:
        M = self.X.shape[0]
        assert self.P.shape == (M, M)
        if self.velocity is not None:
            assert self.velocity.shape == self.X.shape, (
                f"velocity {self.velocity.shape} must match X {self.X.shape}")
        assert self.p0.shape == (M,) and self.p1.shape == (M,)
        assert self.p0.sum() > 0 and self.p1.sum() > 0, "empty source/target marginal"
        rs = np.asarray(self.P.sum(axis=1)).ravel()
        assert np.allclose(rs, 1.0, atol=1e-8), "P_train rows must sum to 1"
