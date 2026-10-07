"""Train / validation / test partition of a route-selection cloud.

Why route-bench needs its own splitter
--------------------------------------
:mod:`scripts.core.protocol` splits each *timepoint band* 70/30.  A route
cloud has no bands: it has four **regions**, two of which the model is never allowed
to see at all (see :mod:`scripts.experiments.sheet.datasets`).  So "70/15/15 of the dataset" has to
mean two different things at once, and this module makes both explicit.

    region            train      val      test     what the split is for
    ----------------------------------------------------------------------------
    SOURCE (p_0)      70%        15%      15%      model input / held-out marginals
    TARGET (p_1)      70%        15%      15%      model input / held-out marginals
    GAP                --        50%      50%      never an input; evaluation only
    DECOY              --        50%      50%      never an input; evaluation only

The withheld regions get **no train share**, because a train share of a region no arm
may look at is not a thing that exists.  Splitting them 50/50 instead of 15/15 is not
generosity, it is the only way the numbers mean anything: the Sheet's crossing
corridor holds 223 cells, so a 15% test slice would be 33 points and its own sampling
floor would be computed from 16 vs 16 — noise reported to three decimals.  At 50/50 it
is 111 vs 112, which is a reference cloud.  Every table states the reference size next
to the number.

What the split actually buys
----------------------------
1. ``P_train`` is rebuilt on the **70% train cells only** — not on all shown cells as
   :func:`scripts.experiments.sheet.datasets._finish` does.  The k-NN graph a model is handed is
   therefore genuinely built from its own training sample, bandwidth included.
2. ``val`` and ``test`` reference clouds are **disjoint**, so a hyper-parameter chosen
   on val has not seen the cells the paper reports on.
3. The flow is integrated from the *train* source cells (arms already do this via
   ``ts.p0``), so no evaluation cell is ever pushed forward.

The selection objective, and what it does and does not claim
------------------------------------------------------------
:func:`val_objective` is the **one** hyper-parameter score in this repository, and every
benchmark uses the same rule (:mod:`scripts.experiments.erythroid` and
:mod:`scripts.experiments.pancreas` all reproduce it against their own marginals):

    ``mean( W2(flow at t=1,               TARGET_val),
            W2(flow over the gap window,  GAP_val) )``

i.e. the user-facing endpoint metric averaged with the intermediate one, the latter
scored against the **val half** of the withheld region.  Val and test halves are disjoint
(:func:`split_route_eval`, asserted in the verification suite in the research repository), so the reported
number is still measured on cells the tuner never saw: this is a valid train/val/test
protocol in the ordinary machine-learning sense.

What it is **not** is blind.  The selected hyper-parameters have seen where the route
goes, so the claim this benchmark supports is *how well can an arm do when it is tuned
for the route*, and not the stronger *can an arm find the route from* ``P`` *alone*.  An
earlier revision carried a second, leak-free objective built only from shown cells in
order to support that stronger reading; it was removed in favour of one rule everywhere,
and git history has it.  The disjointness of val and test is what the whole protocol now
rests on, and the verification suite in the research repository asserts it directly.

Selection never puts a withheld cell into *training* — only into ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from scripts.core.metrics import traj_slice, wasserstein
from scripts.core.protocol import TrainingSet, dataset_salt
from scripts.core.selection import OBJECTIVE_TERMS as _OBJECTIVE_TERMS
from scripts.core.selection import objective_from as _objective_from
from scripts.core.selection import with_objective

from .datasets import (
    DECOY,
    GAP,
    P_KNN_K,
    P_SELF_WEIGHT,
    SOURCE,
    TARGET,
    RouteDataset,
    build_transition_matrix,
)
from .protocol import RouteEval, corridor_mask, gap_window, pooled_slices

#: 70/15/15 over the regions a model is allowed to see
SHOWN_FRACS = (0.70, 0.15, 0.15)
#: 0/50/50 over the withheld regions — see the module docstring
WITHHELD_FRACS = (0.0, 0.5, 0.5)

SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
#  The split
# --------------------------------------------------------------------------- #
@dataclass
class RouteSplit:
    """A per-region partition of one cloud into train / val / test cells."""

    name: str
    seed: int
    train_idx: np.ndarray                  # global indices, shown regions only
    val_idx: np.ndarray                    # global indices, every region
    test_idx: np.ndarray                   # global indices, every region
    region: np.ndarray = field(repr=False)  # the cloud's region labels, for masking
    n_total: int = 0

    def __post_init__(self) -> None:
        parts = [self.train_idx, self.val_idx, self.test_idx]
        for nm, arr in zip(SPLIT_NAMES, parts):
            assert arr.ndim == 1 and arr.size > 0, f"{nm} split is empty"
        allidx = np.concatenate(parts)
        assert len(np.unique(allidx)) == len(allidx), "a cell landed in two splits"
        assert len(allidx) == self.n_total, (
            f"split covers {len(allidx)} of {self.n_total} cells")
        # the contract the whole benchmark rests on, restated at the split level
        assert not np.isin(self.region[self.train_idx], (GAP, DECOY)).any(), (
            "a withheld cell reached the train split")

    def indices(self, which: str) -> np.ndarray:
        assert which in SPLIT_NAMES, f"unknown split {which!r}, expected {SPLIT_NAMES}"
        return {"train": self.train_idx, "val": self.val_idx,
                "test": self.test_idx}[which]

    def mask(self, which: str) -> np.ndarray:
        """Boolean mask over the whole cloud for one split."""
        m = np.zeros(self.n_total, dtype=bool)
        m[self.indices(which)] = True
        return m

    def counts(self) -> dict[str, dict[str, int]]:
        """``{split: {region: n}}`` — goes straight into the protocol table."""
        names = {SOURCE: "source", TARGET: "target", GAP: "gap", DECOY: "decoy"}
        out: dict[str, dict[str, int]] = {}
        for which in SPLIT_NAMES:
            reg = self.region[self.indices(which)]
            out[which] = {nm: int((reg == code).sum())
                          for code, nm in names.items() if (self.region == code).any()}
            out[which]["all"] = int(len(self.indices(which)))
        return out


def make_split(ds: RouteDataset, seed: int = 0) -> RouteSplit:
    """Region-stratified 70/15/15 (shown) / 0-50-50 (withheld) partition.

    Stratified rather than global so that the val and test slices carry the same
    region composition as the cloud: a global draw would let a seed hand the test set
    twice as many gap cells as the val set and make the two columns incomparable.
    """
    rng = np.random.default_rng([seed, dataset_salt(ds.name)])
    buckets: dict[str, list[np.ndarray]] = {k: [] for k in SPLIT_NAMES}

    for code in (SOURCE, TARGET, GAP, DECOY):
        idx = np.flatnonzero(ds.region == code)
        if idx.size == 0:                                   # the Sheet has no decoy
            continue
        idx = rng.permutation(idx)
        shown = code in (SOURCE, TARGET)
        f_tr, f_va, _ = SHOWN_FRACS if shown else WITHHELD_FRACS
        n_tr = int(round(f_tr * len(idx)))
        n_va = int(round(f_va * len(idx)))
        if shown:                                           # keep all three non-empty
            n_tr = min(max(n_tr, 1), len(idx) - 2)
            n_va = min(max(n_va, 1), len(idx) - n_tr - 1)
        buckets["train"].append(idx[:n_tr])
        buckets["val"].append(idx[n_tr:n_tr + n_va])
        buckets["test"].append(idx[n_tr + n_va:])

    parts = {k: np.sort(np.concatenate(v)) for k, v in buckets.items()}
    return RouteSplit(name=ds.name, seed=int(seed), region=ds.region,
                      n_total=int(len(ds.X)), **{f"{k}_idx": v for k, v in parts.items()})


# --------------------------------------------------------------------------- #
#  What the model sees
# --------------------------------------------------------------------------- #
def split_training_set(ds: RouteDataset, split: RouteSplit,
                       p_variant: str = "directional") -> TrainingSet:
    """``(X, P, p_0, p_1)`` over the train split only, with ``P`` rebuilt on it.

    The rebuild is the point.  ``ds.P_train`` was built on *all* shown cells; handing
    that to a model trained on 70% of them would leak the other 30% through the graph
    — both through the edges themselves and through the median-k-NN bandwidth σ, which
    is a global statistic of whatever point set it is computed on.
    """
    idx = split.train_idx
    region = ds.region[idx]
    p0 = region == SOURCE
    p1 = region == TARGET
    assert p0.sum() > 0 and p1.sum() > 0, "empty source/target marginal in the train split"

    P = build_transition_matrix(ds.X[idx], ds._v_gen[idx], k=P_KNN_K, lam=ds.lam,
                                self_weight=P_SELF_WEIGHT)
    assert isinstance(P, sp.csr_matrix) and P.shape == (len(idx), len(idx))

    diag = {
        "n_train": int(len(idx)), "n_p0": int(p0.sum()), "n_p1": int(p1.sum()),
        "n_val": int(len(split.val_idx)), "n_test": int(len(split.test_idx)),
        "n_withheld_regions": int(np.isin(ds.region, (GAP, DECOY)).sum()),
        "split_seed": int(split.seed), "split_counts": split.counts(),
        "lam": float(ds.lam), "d": int(ds.d),
    }
    return TrainingSet(name=ds.name, idx=idx, X=ds.X[idx], P=P, p0=p0, p1=p1,
                       rho=0.0, holdout_band=None, p_variant=p_variant, diag=diag)


# --------------------------------------------------------------------------- #
#  What the model is scored against
# --------------------------------------------------------------------------- #
def split_route_eval(ds: RouteDataset, split: RouteSplit, which: str) -> RouteEval:
    """The val or test reference clouds — same windows as the full protocol.

    ``t_lo``/``t_hi`` come from the **full** gap, not from the split's half of it: the
    time window is a property of where the hole is on the manifold, and letting it
    wobble with the split would make val and test score different stretches of the
    route.
    """
    assert which in ("val", "test"), f"scored splits are val/test, got {which!r}"
    m = split.mask(which)
    t_lo, t_hi = gap_window(ds)
    corr = corridor_mask(ds) & m
    return RouteEval(
        name=ds.name,
        endpoint_ref=np.asarray(ds.X[(ds.region == TARGET) & m], dtype=np.float64),
        gap_ref=np.asarray(ds.X[(ds.region == GAP) & m], dtype=np.float64),
        corridor_ref=np.asarray(ds.X[corr], dtype=np.float64),
        corridor_label=("|u| <= corridor" if ds.name == "Sheet" else "whole gap"),
        t_lo=t_lo, t_hi=t_hi,
    )


# --------------------------------------------------------------------------- #
#  The selection objective
# --------------------------------------------------------------------------- #
#: the objective itself is defined once, in :mod:`scripts.core.selection`, and re-exported
#: here because this module is where the Sheet's callers already look for it.  What lives
#: below is only the Sheet's *marginals* -- which cells each of the two terms is scored
#: against; the arithmetic and the reasoning are shared with the other four benchmarks.
OBJECTIVE_TERMS = _OBJECTIVE_TERMS
objective_from = _objective_from


def val_objective(traj: np.ndarray, ds: RouteDataset, split: RouteSplit,
                  seed: int = 0) -> dict[str, float]:
    """Hyper-parameter selection score (lower is better), plus the terms it averages.

    Endpoint W2 averaged with the intermediate W2, the latter scored against the **val
    half of the withheld region** (the crossing corridor on the Sheet, the whole gap
    elsewhere, exactly as :func:`scripts.experiments.sheet.evaluate.evaluate_route` does on
    test).  Val and test halves are disjoint, so the *reported* number is still measured
    on cells the tuner never saw — this is a valid protocol, not a leak into the score.
    What it is not is blind; see the module docstring for what that costs the claim.

    The individual terms are returned alongside ``"objective"`` so a finished sweep can
    be re-ranked, or a new term added, without retraining anything.
    """
    assert traj.ndim == 3, f"traj must be (S+1, B, d), got {traj.shape}"
    m = split.mask("val")
    tgt_ref = np.asarray(ds.X[(ds.region == TARGET) & m], dtype=np.float64)
    assert len(tgt_ref) > 1, "val target marginal is too small to score"
    rev = split_route_eval(ds, split, "val")

    terms = {
        "W2_endpoint": wasserstein(traj_slice(traj, 1.0), tgt_ref, p=2, seed=seed),
        "W2_intermediate": wasserstein(
            pooled_slices(traj, rev.t_lo, rev.t_hi), rev.corridor_ref, p=2, seed=seed),
    }
    return with_objective(terms)
