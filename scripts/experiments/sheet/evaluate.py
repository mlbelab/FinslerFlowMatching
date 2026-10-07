"""Scoring one trained arm on one route-selection cloud.

Four families, in the order the tables report them:

1. **Endpoint transport** -- W2 and RBF-MMD between the flow at ``t = 1`` and the
   TARGET cells.  This is the marginal the model was trained to hit, so it is the
   sanity column: an arm that fails here has not solved the transport problem at all
   and its intermediate number means nothing.
2. **Intermediate transport** -- the same two distances between the flow pooled over
   the gap's time window and the withheld GAP cells (on the Sheet, the
   high-probability corridor of them).  Nothing in training saw these cells, so this
   is the actual test.
3. **Drift fidelity** -- ``cos(v_theta(t,x), b_0(x))`` against the P-derived flux
   ``b_0(x) = sum_i k_eps(x,x_i) J_i / sum_i k_eps(x,x_i)`` evaluated on the **full**
   cloud and the **full** P.  The reference is identical for every arm -- that is the
   whole point -- and it is the only place the withheld region's dynamics enter the
   score as a field rather than as a point cloud.
4. **Route selection** -- where the transported mass physically ends up, by nearest
   neighbour in the full cloud: in the gap (right), in the decoy (wrong way round),
   or off the manifold entirely (cut the corner).  This is a diagnostic rather than a
   headline number; the two tables the brief asks for are 1 and 2.
5. **Local moment fidelity** -- ``R1`` and ``R2``, the cosines between the model's own
   displacement moments and the *raw* held-out transition rows, node by node
   (:func:`holdout_moment_fidelity`).  Family 3 needs a ``v_theta`` and compares against
   a smoothed field; this one needs only samples and compares against ``P`` itself.

Every distance also gets a **floor**: the same distance between a draw from the
reference cloud and the rest of it, sized to the cloud the arm itself pushes and
averaged over draws (:func:`scripts.core.metrics.floor_splits`).  No model can beat the
sampling noise between two samples of one finite cloud, so the floor is what "perfect"
reads as in these units.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from scripts.core.metrics import (local_moment_fidelity, mean_floor, rbf_mmd,
                                  traj_slice, velocity_fidelity, wasserstein)
from scripts.core.geometry import FinslerGeometry

from .datasets import DECOY, GAP, SOURCE, TARGET, RouteDataset
from .protocol import RouteEval, corridor_mask, pooled_slices

#: nearest-neighbour distance beyond which a transported point counts as "off the
#: manifold", expressed as a multiple of the cloud's own median nearest-neighbour
#: spacing.  Generous on purpose: the interesting failure (cutting across the empty
#: disc) is an order of magnitude out, not a factor of five.
OFF_MANIFOLD_MULT = 5.0

_GEOM_CACHE: dict[tuple, FinslerGeometry] = {}
_TREE_CACHE: dict[str, tuple[cKDTree, float]] = {}


def reference_geometry(ds: RouteDataset, device: torch.device,
                       dtype: torch.dtype = torch.float32) -> FinslerGeometry:
    """Geometry on the FULL ``(X, P_full)`` -- the fixed yardstick for ``b_0``.

    Never a model input.  Only :meth:`FinslerGeometry.drift` is read, and ``b_0``
    depends solely on the antisymmetric flux ``J_pts`` and the kernel bandwidth, not
    on ``rho``, ``c`` or either ablation switch, so the defaults here do not leak a
    training hyper-parameter into the metric.
    """
    key = (ds.name, str(device), str(dtype))
    if key not in _GEOM_CACHE:
        _GEOM_CACHE[key] = FinslerGeometry(ds.X, ds.P_full, device=device, dtype=dtype)
    return _GEOM_CACHE[key]


def _tree(ds: RouteDataset) -> tuple[cKDTree, float]:
    """KD-tree over the full cloud plus its median nearest-neighbour spacing."""
    if ds.name not in _TREE_CACHE:
        tree = cKDTree(ds.X)
        # k=2 because the first neighbour of a cloud point is itself
        nn = tree.query(ds.X, k=2)[0][:, 1]
        _TREE_CACHE[ds.name] = (tree, float(np.median(nn)))
    return _TREE_CACHE[ds.name]


# --------------------------------------------------------------------------- #
#  Distances and their sampling floors
# --------------------------------------------------------------------------- #
def _distances(pred: np.ndarray, ref: np.ndarray, seed: int = 0) -> dict[str, float]:
    """The two headline distances, W2 and the RBF-MMD, in the reference's units."""
    return {"W2": wasserstein(pred, ref, p=2, seed=seed),
            "MMD": rbf_mmd(pred, ref, seed=seed)}


def sampling_floor(ref: np.ndarray, n_draw: int, seed: int = 0) -> dict[str, float]:
    """A same-sized draw from ``ref`` against the rest of it -- the irreducible value.

    Size-matched to the arm's own draw rather than a half-split: see
    :func:`scripts.core.metrics.floor_splits` for why a half-split reads high.
    """
    return mean_floor(ref, n_draw, lambda a, b, k: _distances(a, b, seed=k), seed)


def floors(rev: RouteEval, n_end: int, n_mid: int,
           seed: int = 0) -> dict[str, dict[str, float]]:
    """Each reference's floor at the sample size it is actually scored against."""
    return {"endpoint": sampling_floor(rev.endpoint_ref, n_end, seed),
            "intermediate": sampling_floor(rev.corridor_ref, n_mid, seed),
            "gap_full": sampling_floor(rev.gap_ref, n_mid, seed)}


# --------------------------------------------------------------------------- #
#  Route selection
# --------------------------------------------------------------------------- #
def route_shares(ds: RouteDataset, cloud: np.ndarray) -> dict[str, float]:
    """Fraction of ``cloud`` sitting in each region of the full manifold.

    A point is assigned the region of its nearest cloud cell, unless that neighbour is
    further than ``OFF_MANIFOLD_MULT`` median spacings away, in which case it is
    ``off_manifold``: a straight-line shortcut across the Sheet leaves the surface, which
    is not "in" any region and must not be silently credited to the nearest arc.
    """
    tree, spacing = _tree(ds)
    dist, idx = tree.query(np.asarray(cloud, dtype=np.float64), k=1)
    off = dist > OFF_MANIFOLD_MULT * spacing
    reg = ds.region[idx]
    n = float(len(cloud))
    out = {"off_manifold": float(off.sum()) / n}
    for code, nm in ((SOURCE, "source"), (TARGET, "target"),
                     (GAP, "gap"), (DECOY, "decoy")):
        out[nm] = float(((reg == code) & ~off).sum()) / n
    return out


# --------------------------------------------------------------------------- #
#  Local moment fidelity at held-out nodes
# --------------------------------------------------------------------------- #
#: node groups the moment cosines are reported over, widest first.  Three rather than
#: one because the number only means something next to the region it was taken in: over
#: every test node it is dominated by the two marginals, which every arm gets right;
#: over the corridor it is the crossing itself, which is the question.
MOMENT_GROUPS = ("test", "gap", "corridor")


def moment_node_mask(ds: RouteDataset, split, group: str) -> np.ndarray:
    """Boolean mask over the whole cloud: the ``group`` subset of ``split``'s test cells.

    Every one of these is a cell no arm was trained on, so its row of ``P_full`` is a
    held-out row in the strict sense -- not merely a row about a held-out *region*.
    """
    assert group in MOMENT_GROUPS, f"unknown node group {group!r}"
    m = split.mask("test")
    if group == "gap":
        m = m & (ds.region == GAP)
    elif group == "corridor":
        m = m & corridor_mask(ds)
    return m


def holdout_moment_fidelity(traj: np.ndarray, ds: RouteDataset, split,
                            groups: tuple[str, ...] = MOMENT_GROUPS,
                            h: float | None = None,
                            seed: int = 0, **kw) -> dict[str, dict]:
    """:func:`~scripts.core.metrics.local_moment_fidelity` per node group -> nested dict.

    The reference rows come from ``ds.P_full``, which is ground truth and never a model
    input, and the nodes come from the test half of the split, so neither side of the
    comparison was available during training or during selection.  The bandwidth is
    shared across groups when it is left to default, because a per-group bandwidth would
    make the corridor row and the test row scores of different kernels.
    """
    if h is None:
        # k=2 because the first neighbour of a cloud point is itself
        h = float(3.0 * np.median(cKDTree(ds.X).query(ds.X, k=2)[0][:, 1]))
    P_full = sp.csr_matrix(ds.P_full)
    out = {}
    for g in groups:
        idx = np.flatnonzero(moment_node_mask(ds, split, g))
        assert len(idx), f"node group {g!r} is empty; the split cannot score it"
        out[g] = local_moment_fidelity(traj, ds.X[idx], P_full[idx], ds.X, h=h,
                                       seed=seed, **kw)
    return out


# --------------------------------------------------------------------------- #
#  The whole evaluation of one arm
# --------------------------------------------------------------------------- #
def evaluate_route(traj: np.ndarray, ds: RouteDataset, rev: RouteEval,
                   v_net=None, device: torch.device | None = None,
                   dtype: torch.dtype = torch.float32, seed: int = 0) -> dict:
    """Every metric for one arm's trajectory on one cloud."""
    assert traj.ndim == 3, f"traj must be (S+1, B, d), got {traj.shape}"
    assert traj.shape[2] == ds.d, (
        f"trajectory is {traj.shape[2]}-dimensional but {ds.name} is {ds.d}-dimensional")

    end_pred = traj_slice(traj, 1.0)
    mid_pred = pooled_slices(traj, rev.t_lo, rev.t_hi)

    out: dict = {
        "endpoint": _distances(end_pred, rev.endpoint_ref, seed=seed),
        # the Sheet restricts this reference to the crossing corridor; the other two
        # clouds' corridor_ref *is* the whole gap, so the column means the same thing
        "intermediate": _distances(mid_pred, rev.corridor_ref, seed=seed),
        "intermediate_full_gap": _distances(mid_pred, rev.gap_ref, seed=seed),
        "floors": floors(rev, len(end_pred), len(mid_pred), seed=seed),
        "route": {"intermediate": route_shares(ds, mid_pred),
                  "endpoint": route_shares(ds, end_pred)},
        "window": {"t_lo": rev.t_lo, "t_hi": rev.t_hi,
                   "corridor": rev.corridor_label,
                   "n_corridor": int(len(rev.corridor_ref)),
                   "n_gap": int(len(rev.gap_ref))},
    }
    if v_net is not None:
        assert device is not None, "a device is required to score the drift"
        out["drift"] = velocity_fidelity(v_net, reference_geometry(ds, device, dtype),
                                         traj, device, dtype, seed=seed)
    return out
