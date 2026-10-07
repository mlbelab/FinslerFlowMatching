"""The Sheet: a curved sheet in R^3 with a single lateral crossing corridor.

Contract
--------
The dataset is a point cloud ``X`` partitioned into **regions**:

    SOURCE  (p_0)   shown to the model
    TARGET  (p_1)   shown to the model
    GAP             *withheld* -- evaluation only

The model is handed ``X[train_mask]`` and a transition matrix ``P_train`` rebuilt on
exactly those survivors.  Nothing about the GAP reaches it: not a point, not an edge,
not a bandwidth.  ``P_full`` and ``_v_gen`` exist on the full cloud as ground truth for
scoring and for the figures, and are never model inputs -- the same discipline
:mod:`scripts.core.transition` applies to its own ``_v_gen``.

``DECOY`` remains in the region vocabulary even though the Sheet does not use it.  It
marks "withheld, and also the *wrong* route", which is a distinct thing from "withheld"
and is what the region colours and the figure legends are written against; keeping the
code lets a future cloud add a decoy without reopening every consumer.

Why the Sheet is shaped the way it is
-------------------------------------
A doubly-curved sheet in R^3, long in ``u`` and short in ``w``.  p_0 and p_1 sit on the
two sides of the *short* axis, so the crossing distance is small and the Euclidean
shortcut is cheap.  The flow is engineered to be approximately 1-D: in a narrow
corridor at ``u ~ 0`` the field points straight across (+w), and away from it the field
points *along* the fronts, perpendicular to the crossing direction, so off-corridor
mass must first travel laterally to the corridor and only then cross.  A model that
ignores P crosses everywhere at once; a model that reads it produces an hourglass.  The
sheet's saddle curvature also means a straight chord across ``w`` leaves the surface,
so the geometry has something to say as well.

This is the one route-selection cloud the paper reports.  Two others (a Clock ring with
congruent gaps, and a ForkedArc bifurcation) were built and swept alongside it and are
recoverable from git history; they were cut because the Sheet is the cloud on which the
Euclidean and the dynamical answers differ *visibly* as well as numerically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from scripts.core.transition import build_transition_matrix

# --------------------------------------------------------------------------- #
#  Region codes
# --------------------------------------------------------------------------- #
#: Re-exported, not defined: four benchmarks label their clouds with these integers, so
#: they live in :mod:`scripts.core.present` where an experiment that is not the Sheet can
#: read them without importing the Sheet.  Kept importable from here because every module
#: in this package spells ``from .datasets import SOURCE, TARGET, GAP, DECOY``.
from scripts.core.present import (      # noqa: F401
    DECOY,
    GAP,
    REGION_LABELS,
    REGION_NAMES,
    SOURCE,
    TARGET,
)

REGION_COLOURS = {
    SOURCE: "#4c72b0",     # blue
    TARGET: "#c44e52",     # red
    GAP: "#55a868",        # green — the thing to be recovered
    DECOY: "#c9a227",      # ochre — withheld, but not the answer
}

#: regions the model is allowed to see
TRAIN_REGIONS = (SOURCE, TARGET)

DATASET_NAMES = ("Sheet",)

P_KNN_K = 15
P_SELF_WEIGHT = 0.05


# --------------------------------------------------------------------------- #
#  Container
# --------------------------------------------------------------------------- #
@dataclass
class RouteDataset:
    """A cloud, its regions, and the two transition matrices (train view + truth)."""

    name: str
    X: np.ndarray                 # (N, d)   coordinates
    region: np.ndarray            # (N,)     one of SOURCE/TARGET/GAP/DECOY
    prog: np.ndarray              # (N,)     progress along the *intended* route
    lam: float                    # directional tilt used to build P
    P_train: sp.csr_matrix        # (n_train, n_train)  what the model sees
    P_full: sp.csr_matrix         # (N, N)              ground truth, never an input
    _v_gen: np.ndarray = field(repr=False)   # (N, d) generator tangent; NOT an input
    #: parameter-space coordinates, for plotting a curved sheet flat.  None in 2-D.
    uv: np.ndarray | None = field(default=None, repr=False)
    #: per-cell fate label on a cloud that bifurcates, ``None`` on one that does not.
    #: Ground truth for scoring which lineage the transported mass reached, in the same
    #: sense as ``_v_gen``: it is never an input, and no arm is handed it.
    fate: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        N, d = self.X.shape
        for nm in ("region", "prog"):
            arr = getattr(self, nm)
            assert arr.shape == (N,), f"{nm} must be {(N,)}, got {arr.shape}"
        if self.fate is not None:
            assert self.fate.shape == (N,), f"fate must be {(N,)}, got {self.fate.shape}"
        assert self._v_gen.shape == (N, d), "the generator field must match X"
        assert set(np.unique(self.region)) <= {SOURCE, TARGET, GAP, DECOY}
        for r in (SOURCE, TARGET, GAP):
            assert (self.region == r).sum() > 0, f"region {REGION_NAMES[r]} is empty"

        n_tr = int(self.train_mask.sum())
        assert self.P_full.shape == (N, N), f"P_full must be {(N, N)}"
        assert self.P_train.shape == (n_tr, n_tr), f"P_train must be {(n_tr, n_tr)}"
        for nm, P in (("P_full", self.P_full), ("P_train", self.P_train)):
            rs = np.asarray(P.sum(axis=1)).ravel()
            assert np.allclose(rs, 1.0, atol=1e-6), f"{nm} rows must sum to 1"
            assert P.data.min() >= -1e-12, f"{nm} must be non-negative"

        # The whole point of the benchmark: no withheld cell is reachable by the model.
        assert not np.isin(self.region[self.train_mask], (GAP, DECOY)).any(), (
            "a withheld cell leaked into the training view")

        # Unit tangents, so that lam means the same thing at any cloud size.
        nrm = np.linalg.norm(self._v_gen, axis=1)
        assert np.allclose(nrm, 1.0, atol=1e-6), "the generator field must be unit-norm"

    # -- views ---------------------------------------------------------------- #
    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def train_mask(self) -> np.ndarray:
        """Boolean mask of the cells the model is allowed to see."""
        return np.isin(self.region, TRAIN_REGIONS)

    @property
    def train_idx(self) -> np.ndarray:
        return np.flatnonzero(self.train_mask)

    def region_mask(self, r: int) -> np.ndarray:
        return self.region == r

    def training_view(self) -> tuple[np.ndarray, sp.csr_matrix]:
        """``(X_train, P_train)`` — precisely the pair handed to a model."""
        return self.X[self.train_mask], self.P_train

    def counts(self) -> dict[str, int]:
        return {REGION_NAMES[r]: int((self.region == r).sum())
                for r in (SOURCE, GAP, TARGET, DECOY)}

    def summary(self) -> str:
        c = self.counts()
        return (f"{self.name:10s} N={len(self.X):5d}  d={self.d}  "
                f"train={int(self.train_mask.sum()):5d}  "
                + "  ".join(f"{k}={v}" for k, v in c.items() if v))


def _finish(name: str, X: np.ndarray, v: np.ndarray, region: np.ndarray,
            prog: np.ndarray, lam: float,
            uv: np.ndarray | None = None,
            fate: np.ndarray | None = None) -> RouteDataset:
    """Normalise the field, build both transition matrices, and box it up.

    ``P_train`` is built by re-running the *same* kernel on the survivors, not by
    slicing ``P_full``: the bandwidth sigma is the median distance to the k-th
    neighbour and must be recomputed once the gap is gone, exactly as re-running a
    velocity pipeline on a cell subset would.  Slicing would silently leave the
    model a graph whose scale still knows about the deleted region.
    """
    v = np.asarray(v, dtype=np.float64)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    train = np.isin(region, TRAIN_REGIONS)
    P_full = build_transition_matrix(X, v, k=P_KNN_K, lam=lam,
                                     self_weight=P_SELF_WEIGHT)
    P_train = build_transition_matrix(X[train], v[train], k=P_KNN_K, lam=lam,
                                      self_weight=P_SELF_WEIGHT)
    return RouteDataset(name=name, X=X, region=region, prog=prog,
                        lam=lam, P_train=P_train, P_full=P_full, _v_gen=v, uv=uv,
                        fate=fate)


# --------------------------------------------------------------------------- #
#  Sheet — a curved sheet in R^3 with a single lateral corridor
# --------------------------------------------------------------------------- #
SHEET_L = 1.50        # lateral half-extent (the long axis, u)
SHEET_W = 0.45        # crossing half-extent (the short axis, w)
SHEET_GAP = 0.13      # half-width of the withheld strip around w = 0
SHEET_AMP = 0.50      # curvature amplitude of the saddle
SHEET_CORRIDOR = 0.28 # 1-sigma half-width of the crossing corridor, in u
SHEET_FLOOR = 0.12    # residual forward drift outside the corridor
SHEET_REVERSAL = 0.30 # width of the lateral sign flip, as a fraction of SHEET_W

#: If True the lateral component reverses at the midline: funnel in on the p_0 side,
#: fan out on the p_1 side (an hourglass).  If False the field funnels inward on *both*
#: sides.  True is the coherent choice, because p_1 is sampled across the full lateral
#: extent: a pure funnel would make it cheap to reach u ~ 0 and expensive to leave it,
#: while the marginal constraint still forces mass out to the edges -- cost and
#: marginals would then disagree, and every model would be penalised for obeying the
#: boundary condition.  With the reversal, the cheapest transport *is* the one that
#: matches both marginals.
SHEET_FAN_OUT = True


def _sheet_z(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Saddle height.  Curved in *both* parameters, so a chord leaves the surface."""
    return SHEET_AMP * ((u / SHEET_L) ** 2 - 0.5 * (w / SHEET_W) ** 2)


def _sheet_dz(u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(dz/du, dz/dw)`` — the surface tangent basis is ``(1,0,z_u)``, ``(0,1,z_w)``."""
    return (2.0 * SHEET_AMP * u / SHEET_L ** 2,
            -SHEET_AMP * w / SHEET_W ** 2)


def make_sheet(N: int = 4000, seed: int = 1, lam: float = 6.0,
               noise_sd: float = 0.012) -> RouteDataset:
    """Curved sheet in R^3; p_0 and p_1 on the two sides of the short axis.

    The flow is built in parameter space and then pushed onto the surface:

        chi(u)  = exp(−u² / 2 u_c²)              corridor indicator, u_c = 0.28
        v_w     = chi + floor                    cross the sheet (only near u = 0)
        v_u     = −tanh(u/u_c) · (1 − chi) · tanh(−w/w_s)

    ``tanh(−w/w_s)`` flips sign at the midline, so the lateral component points
    *inward* on the p_0 side (a funnel) and *outward* on the p_1 side (a fan): mass
    converges to the corridor, crosses, then spreads again.  ``floor`` keeps a small
    forward drift everywhere so no cell is a perfect trap.  Set ``SHEET_FAN_OUT`` to
    False to drop the reversal and funnel inward on both sides; see the constant's
    comment for why the reversal is the coherent default.

    The resulting field is approximately 1-D in the sense the brief asks for: on the
    lateral boundary it is perpendicular to the crossing direction, and only along the
    midline u ~ 0 does it point at p_1.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-SHEET_L, SHEET_L, size=N)
    w = rng.uniform(-SHEET_W, SHEET_W, size=N)

    z_u, z_w = _sheet_dz(u, w)
    e_u = np.stack([np.ones_like(u), np.zeros_like(u), z_u], axis=1)   # dX/du
    e_w = np.stack([np.zeros_like(w), np.ones_like(w), z_w], axis=1)   # dX/dw

    X = np.stack([u, w, _sheet_z(u, w)], axis=1)
    X = X + rng.normal(0.0, noise_sd, size=X.shape)      # thickness around the sheet

    chi = np.exp(-0.5 * (u / SHEET_CORRIDOR) ** 2)
    v_w = chi + SHEET_FLOOR
    # -tanh(u/u_c) points at the corridor from either side; (1 - chi) switches the
    # lateral push off once you are in it.  The third factor is the midline reversal:
    # a narrow tanh in w, so the funnel is still at ~75% strength where p_0 ends
    # rather than having already decayed away.
    swing = np.tanh(-w / (SHEET_REVERSAL * SHEET_W)) if SHEET_FAN_OUT else 1.0
    v_u = -np.tanh(u / SHEET_CORRIDOR) * (1.0 - chi) * swing
    v = v_u[:, None] * e_u + v_w[:, None] * e_w          # push onto the tangent plane

    region = np.full(N, GAP, dtype=np.int64)
    region[w <= -SHEET_GAP] = SOURCE
    region[w >= SHEET_GAP] = TARGET

    prog = (w + SHEET_W) / (2.0 * SHEET_W)
    return _finish("Sheet", X, v, region, prog, lam,
                   uv=np.stack([u, w], axis=1))


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
_MAKERS = {"Sheet": make_sheet}
assert tuple(_MAKERS) == DATASET_NAMES, "the maker table and the registry disagree"


def make_dataset(name: str, **kw) -> RouteDataset:
    if name not in _MAKERS:
        raise KeyError(f"unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    return _MAKERS[name](**kw)


if __name__ == "__main__":
    for nm in DATASET_NAMES:
        print(make_dataset(nm).summary())
