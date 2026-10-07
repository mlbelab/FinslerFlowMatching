"""What a model is shown, and what it is scored against, on a route-selection cloud.

This is the thin adapter between :mod:`scripts.experiments.sheet.datasets` and the existing method
arms in :mod:`scripts.core.arms`.  It deliberately does *not* re-implement
the arms: an arm only ever consumes a
:class:`~scripts.core.protocol.TrainingSet` -- ``(X, P, p_0, p_1)`` -- so
handing it the route-bench training view makes the numbers directly comparable with the
synthetic suite, and any difference in the tables is a difference in the *data*, not in
the pipeline.

The two halves of the protocol
------------------------------
**Shown.**  ``X[train_mask]`` and ``P_train``, with ``p_0 = SOURCE`` and
``p_1 = TARGET``.  There is no ``rho`` ladder and no held-out third of each band here:
route-bench withholds a contiguous *region* of the manifold rather than a random
fraction of a timepoint, so the whole of ``p_0`` and ``p_1`` is training data and the
whole of the GAP is evaluation data.  ``rho=0`` / ``holdout_band=None`` are passed
through only because ``TrainingSet`` records them for provenance.

**Scored.**  Two references:

``endpoint``      the TARGET cells, against the flow at ``t = 1``.
``intermediate``  the GAP cells, against the flow pooled over the *time window the gap
                  occupies*, ``[t_lo, t_hi] = [min prog, max prog]`` over GAP cells.

The window matters.  A single slice at ``t = 0.5`` is a thin sheet of mass being
compared against a finite-width band of cells, so its W2 would be dominated by the band
width rather than by whether the model went the right way.  Pooling the model over the
same span of the route that the band covers makes the two clouds comparable in extent.
This inherits the suite's constant-speed convention (``BAND_TIMES = k/3`` assumes model
time is proportional to progress along the route); it is an assumption, and it is the
same assumption for every arm.

**The Sheet's corridor.**  On the Sheet the gap strip is 1144 cells spread over the full
lateral extent, but only the cells near ``u = 0`` are where the dynamics actually send
mass across (``chi(u) = exp(-u^2/2u_c^2)`` is the crossing probability).  Scoring against
the whole strip would reward a model that smears across the entire front.  So the Sheet's
intermediate reference is restricted to ``|u| <= SHEET_CORRIDOR`` -- the high-probability
corridor.  The prediction is *not* restricted: the question is whether the model's mass
is in the corridor, so mass that is elsewhere must be allowed to cost something.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.core.protocol import TrainingSet

from .datasets import GAP, SHEET_CORRIDOR, SOURCE, TARGET, RouteDataset

#: number of trajectory slices pooled across the gap's time window
N_GAP_SLICES = 9


def training_set(ds: RouteDataset, p_variant: str = "directional") -> TrainingSet:
    """The exact ``(X, P, p_0, p_1)`` an arm is allowed to see."""
    idx = ds.train_idx
    region = ds.region[idx]
    p0 = region == SOURCE
    p1 = region == TARGET
    assert p0.sum() > 0 and p1.sum() > 0, "empty source/target marginal"
    assert not (p0 & p1).any(), "a cell cannot be in both marginals"
    diag = {
        "n_train": int(len(idx)),
        "n_p0": int(p0.sum()),
        "n_p1": int(p1.sum()),
        "n_withheld": int(len(ds.X) - len(idx)),
        "lam": float(ds.lam),
        "d": int(ds.d),
    }
    return TrainingSet(name=ds.name, idx=idx, X=ds.X[idx], P=ds.P_train,
                       p0=p0, p1=p1, rho=0.0, holdout_band=None,
                       p_variant=p_variant, diag=diag)


def gap_window(ds: RouteDataset) -> tuple[float, float]:
    """``(t_lo, t_hi)``: the span of route progress the withheld gap covers."""
    prog = ds.prog[ds.region == GAP]
    lo, hi = float(prog.min()), float(prog.max())
    assert 0.0 <= lo < hi <= 1.0, f"degenerate gap window ({lo}, {hi}) on {ds.name}"
    return lo, hi


def corridor_mask(ds: RouteDataset) -> np.ndarray:
    """Boolean mask over the *whole* cloud: the high-probability part of the gap.

    Only the Sheet narrows it.  On a one-dimensional cloud the gap *is* already the
    corridor -- there is nowhere else to be -- so the guard returns the whole gap and
    the caller needs no special case.
    """
    gap = ds.region == GAP
    if ds.name != "Sheet":
        return gap
    assert ds.uv is not None, "the Sheet must carry its (u, w) parameters"
    return gap & (np.abs(ds.uv[:, 0]) <= SHEET_CORRIDOR)


@dataclass(frozen=True)
class RouteEval:
    """The fixed reference clouds one dataset is scored against."""

    name: str
    endpoint_ref: np.ndarray          # (n1, d)  the TARGET cells
    gap_ref: np.ndarray               # (ng, d)  every GAP cell
    corridor_ref: np.ndarray          # (nc, d)  the high-probability part of the gap
    corridor_label: str               # what the restriction was, for the caption
    t_lo: float
    t_hi: float

    def __post_init__(self) -> None:
        d = self.endpoint_ref.shape[1]
        for nm in ("gap_ref", "corridor_ref"):
            arr = getattr(self, nm)
            assert arr.ndim == 2 and arr.shape[1] == d, f"{nm} must be (n, {d})"
            assert len(arr) > 1, f"{nm} is too small to score against"


def route_eval(ds: RouteDataset) -> RouteEval:
    """Assemble the evaluation references for one cloud."""
    t_lo, t_hi = gap_window(ds)
    corr = corridor_mask(ds)
    label = (f"|u| <= {SHEET_CORRIDOR}" if ds.name == "Sheet" else "whole gap")
    return RouteEval(
        name=ds.name,
        endpoint_ref=np.asarray(ds.X[ds.region == TARGET], dtype=np.float64),
        gap_ref=np.asarray(ds.X[ds.region == GAP], dtype=np.float64),
        corridor_ref=np.asarray(ds.X[corr], dtype=np.float64),
        corridor_label=label, t_lo=t_lo, t_hi=t_hi,
    )


def pooled_slices(traj: np.ndarray, t_lo: float, t_hi: float,
                  n_slices: int = N_GAP_SLICES) -> np.ndarray:
    """The flow's states over ``[t_lo, t_hi]``, stacked into one cloud -> (K*B, d).

    ``traj`` is the ``(n_steps+1, B, d)`` array every arm returns; the integrator is
    uniform in t, so row ``round(t*n_steps)`` is time ``t``.
    """
    assert traj.ndim == 3, f"traj must be (S+1, B, d), got {traj.shape}"
    assert 0.0 <= t_lo < t_hi <= 1.0
    n_steps = traj.shape[0] - 1
    rows = np.unique(np.round(np.linspace(t_lo, t_hi, n_slices) * n_steps).astype(int))
    return np.asarray(traj[rows].reshape(-1, traj.shape[2]), dtype=np.float64)
