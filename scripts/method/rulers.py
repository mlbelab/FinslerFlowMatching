"""The one ruler here that knows about the geometry.

``W_2`` is Euclidean: it measures where the cloud landed, in the ambient space, and a
method can score well on it by cutting a corner the manifold does not have.

* **``W_{2,F}``** -- the same optimal-transport distance to the reference cloud, but with
  the ground cost replaced by the Finsler cost ``c_F`` along a frozen interpolant.  A
  detour that leaves the manifold, or one that runs against ``b_0``, is charged for.
  It is always reported beside a Euclidean control run through the *same exact solver on
  the same clouds*, so any reordering between the two columns is the ground cost talking
  and not the OT.

The second ruler the notebooks print beside it -- the local moment cosines ``R_1``/``R_2``
against the raw held-out transition rows -- lives in
:func:`scripts.core.metrics.local_moment_fidelity`, because it reads only samples and a
``P`` and so has nothing to do with this module's judge.
"""
from __future__ import annotations

import numpy as np
import ot
import scipy.sparse as sp
import torch

from scripts.method.config import Run
from scripts.method.phases import finsler_cost

#: predicted cells entering the exact ``W_{2,F}`` solve.  The cost matrix is dense and
#: every entry is a K-node quadrature through the interpolant, so this is the one place
#: in the notebook where subsampling buys something real.
WF_MAX_PRED = 400


def finsler_w2(pred, ref, judge, cfg: Run, max_pred: int = WF_MAX_PRED, seed: int = 0):
    """``W_{2,F}(pred, ref)`` under the judge's geodesic cost, by exact OT.

    ``judge = (phi, metric)`` is fixed once for a whole column, so every row is scored by
    the identical cost function — a per-arm judge would let a method be graded by its own
    geometry.
    """
    phi, metric = judge
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, dtype=np.float64)
    if len(pred) > max_pred:
        pred = pred[rng.choice(len(pred), max_pred, replace=False)]
    A = cfg.tensor(pred)
    B = cfg.tensor(np.asarray(ref, dtype=np.float64))
    C = np.maximum(finsler_cost(phi, metric, A, B, cfg).double().cpu().numpy(), 0.0)
    w2sq = ot.emd2(np.full(len(A), 1.0 / len(A)), np.full(len(B), 1.0 / len(B)),
                   np.ascontiguousarray(C), numItermax=1_000_000)
    return float(np.sqrt(max(w2sq, 0.0)))


def euclidean_w2(pred, ref, max_pred: int = WF_MAX_PRED, seed: int = 0):
    """The same OT problem with ``||x - y||^2`` in place of ``c_F^2`` — the control."""
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, dtype=np.float64)
    if len(pred) > max_pred:
        pred = pred[rng.choice(len(pred), max_pred, replace=False)]
    ref = np.asarray(ref, dtype=np.float64)
    C = ot.dist(pred, ref, metric="sqeuclidean")
    return float(np.sqrt(max(ot.emd2(np.full(len(pred), 1.0 / len(pred)),
                                     np.full(len(ref), 1.0 / len(ref)), C,
                                     numItermax=1_000_000), 0.0)))
