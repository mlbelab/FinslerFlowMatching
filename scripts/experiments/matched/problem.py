"""The two datasets behind one interface: moments in, val and test scores out.

One controlled synthetic case and one existing real-data case, and deliberately no more:
the review asks for a matched comparison, not a wider benchmark.  Both are *existing*
clouds with *existing* splits -- nothing here re-partitions anything, and nothing here
invents a ruler.  A :class:`Problem` is built by calling the tree's own splitter, so the
cells this experiment trains on are the same cells the published table trains on and the
cells it is scored against are the same cells the published table is scored against.

The two stages differ exactly as the two trees already differ
-------------------------------------------------------------
``stage="tune"`` builds what a sweep cell may see: the Sheet's 70 % train split, the
Pancreas's 85 % ``split_shown`` train half, ``P`` rebuilt on each, and the *validation*
marginals of :mod:`scripts.core.selection`.  ``stage="bench"`` builds what the reported
run uses: the same Sheet split (its tuner and its notebook agree), the Pancreas's whole
shown set (its notebook refits the chosen point on all kept cells), and the **test**
marginals, which are disjoint from the validation ones.

Each tree keeps its own ruler
-----------------------------
The Sheet scores with :func:`scripts.core.metrics.wasserstein` and the Pancreas tuner with
:func:`scripts.method.rulers.euclidean_w2`; the Pancreas bench scores with ``wasserstein``
through ``distribution_metrics``.  Those choices are copied rather than harmonised, so an
``ffm_full`` row here lands on the same ruler as the published FFM row and the two can be
compared directly.  Harmonising them would have made this tree internally tidier and
externally incomparable.

The intermediate term, and why the Sheet reports two of them
------------------------------------------------------------
"Midpoint :math:`W_2`" is unambiguous on the Pancreas: the flow at :math:`t=1/2` against
the withheld marginal.  On the Sheet the established intermediate metric is the flow
*pooled over the gap window* against the crossing corridor, because the corridor is a
stretch of the route and not an instant.  Both are reported: ``W2_mid`` is the literal
:math:`t=1/2` slice on both datasets, and ``W2_window`` is the Sheet's own pooled figure
beside it.  Selection uses each tree's published objective and neither of the extra
columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch

from scripts.core.metrics import mean_floor, traj_slice, wasserstein
from scripts.core.selection import with_objective
from scripts.method import euclidean_w2, moments_from
from scripts.method.config import Run

#: the two clouds this experiment runs on, as the CLI spells them
KEYS = ("sheet", "pancreas20")

#: the PCA prefix the Pancreas column is taken in.  One dimension, because the review asks
#: for the existing 20-component case and not for a new sweep over prefixes.
PANCREAS_DIM = 20

STAGES = ("tune", "bench")


@dataclass
class Problem:
    """One cloud, one split, the moments of ``P``, and the two scoring callables."""

    key: str
    stage: str
    X: np.ndarray                                   # (N, d) train cells
    P: sp.csr_matrix                                # rebuilt on the train cells
    b0_pts: np.ndarray                              # m_i  = sum_j P_ij (x_j - x_i)
    D_pts: np.ndarray                               # D~_i = sum_j P_ij dx dx^T
    X0_t: torch.Tensor
    X1_t: torch.Tensor
    val: Callable[[np.ndarray, int], dict]
    test: Callable[[np.ndarray, int], dict]
    refs: dict = field(default_factory=dict, repr=False)
    diag: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    def floors(self, seed: int = 0) -> dict:
        """Each test reference scored against a same-sized draw from itself.

        The number a :math:`W_2` between two samples of the *same* distribution reports at
        this sample size.  A paired difference smaller than the floor is not a difference.
        The draw is sized to the arm's own cloud -- every arm pushes ``len(X0_t)``
        particles -- rather than to half the reference, which would bound the comparison
        with a smaller and therefore larger number (:func:`scripts.core.metrics.floor_splits`).
        """
        out = {}
        n_draw = int(len(self.X0_t))
        for name, ref in self.refs.items():
            out[name] = (mean_floor(ref, n_draw,
                                    lambda a, b, k: {"W2": wasserstein(a, b, p=2, seed=k)},
                                    seed)["W2"]
                         if len(ref) > 3 else float("nan"))
        return out


# --------------------------------------------------------------------------- #
#  Sheet
# --------------------------------------------------------------------------- #
def _build_sheet(stage: str, cfg: Run) -> Problem:
    from scripts.experiments.sheet.datasets import make_dataset
    from scripts.experiments.sheet.protocol import pooled_slices
    from scripts.experiments.sheet.split import (make_split, split_route_eval,
                                                 split_training_set, val_objective)

    ds = make_dataset("Sheet")
    split = make_split(ds, seed=cfg.split_seed)
    ts = split_training_set(ds, split)                 # P rebuilt on the 70 %
    X = np.asarray(ts.X, dtype=np.float64)
    P = sp.csr_matrix(ts.P)
    b0_pts, D_pts = moments_from(P, X)

    X_t = cfg.tensor(X)
    X0_t = X_t[torch.as_tensor(ts.p0, device=cfg.device)]
    X1_t = X_t[torch.as_tensor(ts.p1, device=cfg.device)]

    rev = split_route_eval(ds, split, "test")

    def val(traj, seed=0):
        return val_objective(traj, ds, split, seed=seed)

    def test(traj, seed=0):
        return {
            "W2_mid": wasserstein(traj_slice(traj, 0.5), rev.corridor_ref, p=2, seed=seed),
            "W2_window": wasserstein(pooled_slices(traj, rev.t_lo, rev.t_hi),
                                     rev.corridor_ref, p=2, seed=seed),
            "W2_end": wasserstein(traj_slice(traj, 1.0), rev.endpoint_ref, p=2, seed=seed),
        }

    return Problem(
        key="sheet", stage=stage, X=X, P=P, b0_pts=b0_pts, D_pts=D_pts,
        X0_t=X0_t, X1_t=X1_t, val=val, test=test,
        refs={"W2_mid": rev.corridor_ref, "W2_end": rev.endpoint_ref},
        diag={"n_train": int(len(ts.idx)), "n_p0": int(ts.p0.sum()),
              "n_p1": int(ts.p1.sum()), "n_mid_ref": int(len(rev.corridor_ref)),
              "n_end_ref": int(len(rev.endpoint_ref)), "d": int(X.shape[1])})


# --------------------------------------------------------------------------- #
#  Pancreas, 20-component PCA
# --------------------------------------------------------------------------- #
def _build_pancreas(stage: str, cfg: Run) -> Problem:
    from scripts.experiments.pancreas.datasets import (TRAIN_BINS, build_training_set,
                                                       load_cloud, split_holdout,
                                                       split_shown)

    cloud = load_cloud(PANCREAS_DIM)
    i_train, i_val = split_shown(cloud)
    # tune: the 85 % train half, exactly as a sweep cell sees it.  bench: every shown
    # cell, exactly as the notebook refits the chosen point.
    subset = i_train if stage == "tune" else None
    ts = build_training_set(cloud, subset=subset)      # P rebuilt on whatever it got

    X = np.asarray(ts.X, dtype=np.float64)
    P = sp.csr_matrix(ts.P)
    b0_pts, D_pts = moments_from(P, X)

    X_t = cfg.tensor(X)
    X0_t = X_t[torch.as_tensor(ts.p0, device=cfg.device)]
    X1_t = X_t[torch.as_tensor(ts.p1, device=cfg.device)]

    # validation marginals: the val slice of p_1 among kept cells (out of sample), and the
    # 10 % selection slice of the withheld marginal, which nothing ever trains on.
    i_end = i_val[cloud.bin_id[i_val] == TRAIN_BINS[1]]
    assert len(i_end) > 1, f"val slice of p1 is too small to score ({len(i_end)})"
    i_sel, _ = split_holdout(cloud)

    # test marginals: the published ones -- the 90 % of the withheld bin, and the whole
    # p_1 marginal.  Copied from ``pancreas.train.Scorer`` rather than reinvented.
    mid_ref = cloud.X[cloud.target_test]
    end_ref = cloud.X[cloud.marginal(TRAIN_BINS[1])]

    def val(traj, seed=0):
        return with_objective({
            "W2_endpoint": euclidean_w2(traj[-1], cloud.X[i_end], seed=seed),
            "W2_intermediate": euclidean_w2(traj[traj.shape[0] // 2], cloud.X[i_sel],
                                            seed=seed),
        })

    def test(traj, seed=0):
        return {"W2_mid": wasserstein(traj_slice(traj, 0.5), mid_ref, p=2, seed=seed),
                "W2_end": wasserstein(traj_slice(traj, 1.0), end_ref, p=2, seed=seed)}

    return Problem(
        key="pancreas20", stage=stage, X=X, P=P, b0_pts=b0_pts, D_pts=D_pts,
        X0_t=X0_t, X1_t=X1_t, val=val, test=test,
        refs={"W2_mid": mid_ref, "W2_end": end_ref},
        diag={"n_train": int(len(ts.idx)), "n_p0": int(ts.p0.sum()),
              "n_p1": int(ts.p1.sum()), "n_mid_ref": int(len(mid_ref)),
              "n_end_ref": int(len(end_ref)), "d": int(X.shape[1]),
              "n_val_end": int(len(i_end)), "n_val_sel": int(len(i_sel)),
              "P_asymmetry": ts.diag["P_asymmetry"]})


_BUILDERS = {"sheet": _build_sheet, "pancreas20": _build_pancreas}


def build(key: str, stage: str, cfg: Run) -> Problem:
    """The cloud, its split, the two moments of ``P``, and the two scoring callables."""
    assert key in KEYS, f"unknown dataset {key!r}; this tree has {KEYS}"
    assert stage in STAGES, f"unknown stage {stage!r}; expected {STAGES}"
    return _BUILDERS[key](stage, cfg)


def cache_for(key: str) -> str | None:
    """A path ``preflight_cell`` must find before a job starts, or ``None``.

    The Sheet is generated from a seed and has no cache; the Pancreas reads a PCA slab
    that a compute node may or may not be able to see, and finding that out in the first
    second of a 124-task array is the difference between one error message and 124.
    """
    if key != "pancreas20":
        return None
    from scripts.experiments.pancreas.datasets import CACHE_DIR
    return CACHE_DIR
