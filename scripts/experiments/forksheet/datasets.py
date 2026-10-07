"""The ForkSheet: a curved bifurcation drawn on a curved sheet in R^3.

Contract
--------
Identical to the Sheet's (:mod:`scripts.experiments.sheet.datasets`), which is the point:
the cloud is a :class:`~scripts.experiments.sheet.datasets.RouteDataset`, so the split, the
protocol, the evaluation and every arm are the Sheet's unchanged and any difference in a
table is a difference in the *data*.

    SOURCE  (p_0)   shown -- the trunk **and the first part of the fork**
    TARGET  (p_1)   shown -- the terminal stretch of *both* arms
    GAP             withheld -- exactly the stretch over which the two arms resolve

One extra piece of ground truth rides along: ``ds.fate``, the lineage label of every cell.
Like ``_v_gen`` it is never a model input; it exists so that one question can be asked of a
trajectory: *did each cell end up on its own arm*.  :func:`fate_table` answers it by
putting a ball round each arm's p_1 cells (:func:`fate_balls`), opened outward so that
running past the tip is not a routing error (:func:`fate_inside`), and cross-tabulating
where each cell landed against the arm it belongs to.  There is no third label to argue
about -- a cell is in one arm's region, another's, or in neither.

Why the cloud is shaped the way it is
-------------------------------------
**The fork opens inside p_0.**  ``FORK_S0 > FORK_SPLIT``, so by the time p_0 ends the two
lineages are already ~0.16 apart -- eight times the point noise.  The model is therefore
told *that* a split happens and shown its first millimetres; what is withheld is where the
two arms go, which is the part a bridge has to invent.

**Both arms are curved, and they are not each other's mirror.**  One arm hugs the trunk
and then plunges away from it, the other leaves early and turns to run alongside it again
(:func:`_arm`); both sit on top of a trunk that is itself a sine rather than a straight
line.  A straight-line interpolant is wrong for both arms, and wrong in a different way
for each, so an arm cannot recover one branch by symmetry once it has the other.

**The mass split is a knob, and it defaults to equal.**  ``shares`` sets what fraction of
the cells takes each arm; at the default ``(0.5, 0.5)`` it is exactly half and half.  Equal
mass is the clean *default* -- the reference share is 0.5, so any deviation in a scored
share is the model's -- but it is also the easiest case, since a lazy arm that splits its
mass evenly is right for the wrong reason.  Turning the knob is what tells the two apart:
see :func:`make_fork_sheet` and :func:`_stratify` for what "70/30" is guaranteed to mean.

At whatever the setting, the labels are a stratified sequence and not a coin flip, so the
share holds inside p_0, inside the gap and inside p_1 alike and never has to be corrected
for the draw.

**The sheet underneath is doubly curved.**  ``z(u, w)`` is a saddle plus a twist, so a
straight chord between two cells leaves the surface, the two arms sit at different heights
by the time they separate, and the geometry has something to say about the route in
addition to the dynamics.  The fork is a curve *on* that surface: it is built in the
``(u, w)`` chart and lifted, and its tangent is pushed through the same map.
"""
from __future__ import annotations

import numpy as np

from scripts.experiments.sheet.datasets import (  # noqa: F401
    GAP,
    P_KNN_K,
    P_SELF_WEIGHT,
    SOURCE,
    TARGET,
    RouteDataset,
    _finish,
)

DATASET_NAMES = ("ForkSheet",)

# --------------------------------------------------------------------------- #
#  The sheet the fork is drawn on
# --------------------------------------------------------------------------- #
FORK_L = 1.50         # half-extent of the trunk axis u
FORK_W = 1.00         # lateral scale the height map is written in
FORK_AMP = 0.80       # saddle amplitude: curvature along both chart axes
FORK_TILT = 0.25      # twist, so the two arms do not descend by the same amount

# --------------------------------------------------------------------------- #
#  The fork drawn on it
# --------------------------------------------------------------------------- #
FORK_SPLIT = 0.12     # progress at which the arms separate
FORK_S0 = 0.45        # end of p_0   (> FORK_SPLIT: p_0 holds a resolved fork)
FORK_S1 = 0.70        # start of p_1 (both arms)
FORK_TRUNK_AMP = 0.30   # the trunk is a sine, so nothing in the cloud is straight
FORK_TRUNK_FREQ = 1.70
#: lateral half-width of a band.  The cloud is a Y-shaped *ribbon* on the sheet and not a
#: wire: the two bands overlap while the arms are closer than ``2 * FORK_BAND`` and are
#: resolved after, which puts the visible split just inside p_0 and gives the withheld
#: stretch a real width to be scored across.
FORK_BAND = 0.05

#: ``(fate, direction, amplitude, exponent, s_ness)`` — see :func:`_arm`.  The two entries
#: differ in *shape* and never in weight: how much mass each carries is the ``shares``
#: argument of :func:`make_fork_sheet` and nothing here.  One arm hangs by the trunk and
#: then plunges away (``s_ness = 0``, pure power), the other leaves early and flattens out
#: at its tip (``s_ness = 1``, pure S): both are curved along their whole length, in
#: opposite senses, and neither is the other reflected.
FORK_FATES = ((0, -1.0, 0.95, 2.6, 0.0),
              (1, +1.0, 0.85, 2.0, 1.0))

#: default mass split, one entry per :data:`FORK_FATES` row, in the same order.
FORK_SHARES = (0.5, 0.5)

FATE_NAMES = {0: "lower arm", 1: "upper arm"}


def _fork_z(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Sheet height: a saddle in ``(u, w)`` plus a twist that breaks the ``w`` mirror."""
    return (FORK_AMP * ((u / FORK_L) ** 2 - 0.5 * (w / FORK_W) ** 2)
            + FORK_TILT * (u / FORK_L) * (w / FORK_W))


def _fork_dz(u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(dz/du, dz/dw)`` — the tangent basis is ``(1,0,z_u)``, ``(0,1,z_w)``."""
    return (2.0 * FORK_AMP * u / FORK_L ** 2 + FORK_TILT * w / (FORK_L * FORK_W),
            -FORK_AMP * w / FORK_W ** 2 + FORK_TILT * u / (FORK_L * FORK_W))


def _trunk(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The spine in the chart, ``(w, dw/ds)``.  A sine, so the trunk is not an axis."""
    return (FORK_TRUNK_AMP * np.sin(FORK_TRUNK_FREQ * s),
            FORK_TRUNK_AMP * FORK_TRUNK_FREQ * np.cos(FORK_TRUNK_FREQ * s))


def _arm(tau: np.ndarray, expo: np.ndarray, sness: np.ndarray
         ) -> tuple[np.ndarray, np.ndarray]:
    """An arm's departure from the trunk, ``(g, dg/dtau)``, on the unit interval.

    A power ``tau^p`` and a smoothstep ``tau^2 (3 - 2 tau)``, mixed by ``s_ness``.  The two
    ends of that mix are curved in opposite senses -- the power bends away from the trunk
    harder the further it goes, the smoothstep bends away and then turns to run alongside
    it -- so an arm at ``s_ness = 0`` and an arm at ``s_ness = 1`` cannot be mapped onto
    each other by any reflection.  Both vanish to first order at ``tau = 0`` (for ``p > 1``)
    so the arm leaves the trunk tangentially and the generator field has no kink at the
    split.
    """
    pw, sm = tau ** expo, tau ** 2 * (3.0 - 2.0 * tau)
    d_pw, d_sm = expo * tau ** (expo - 1.0), 6.0 * tau * (1.0 - tau)
    return ((1.0 - sness) * pw + sness * sm,
            (1.0 - sness) * d_pw + sness * d_sm)


def _shares(shares) -> np.ndarray:
    """Validate a mass split and normalise it.  ``(70, 30)`` and ``(0.7, 0.3)`` are equal.

    Accepting unnormalised weights is not laziness: "70/30" is how an imbalance is said out
    loud, and a call site that has to divide by 100 first will eventually forget to.
    """
    a = np.asarray(shares, dtype=np.float64)
    assert a.shape == (len(FORK_FATES),), \
        f"need one share per fate, {len(FORK_FATES)}; got {np.shape(shares)}"
    assert (a > 0.0).all(), f"every fate needs some mass; got {shares}"
    return a / a.sum()


def _stratify(n: int, p: np.ndarray) -> np.ndarray:
    """Fate index for each of ``n`` cells **in progress order**, at shares ``p``.

    Fate ``k``'s *j*-th cell is due at sequence position ``(j - 1/2) / p_k`` and the ranks
    are handed out in order of due time.  Two things follow, and both are the reason this
    is not a multinomial draw:

    * after any prefix of length ``m``, fate ``k`` holds within one cell of ``m * p_k`` --
      so the requested share is the share inside p_0, inside the gap and inside p_1
      separately, and a deviation in a scored share is the model's and never the draw's;
    * at ``(0.5, 0.5)`` it is exactly alternation, i.e. the equal-mass cloud is unchanged
      by this generalisation.

    The floor-then-largest-remainder step is what makes the totals come out to ``n``
    exactly rather than to ``n`` minus a rounding error.
    """
    counts = np.floor(n * p).astype(np.int64)
    rem = n - int(counts.sum())
    if rem:
        counts[np.argsort(counts - n * p)[:rem]] += 1
    assert counts.sum() == n, (counts, n)
    due = np.concatenate([(np.arange(c) + 0.5) / pk for c, pk in zip(counts, p)])
    lab = np.repeat(np.arange(len(p)), counts)
    return lab[np.argsort(due, kind="stable")]


def make_fork_sheet(N: int = 3000, seed: int = 3, lam: float = 6.0,
                    noise_sd: float = 0.012, shares=FORK_SHARES) -> RouteDataset:
    """A two-fate bifurcation on a doubly curved sheet; both arms curve, 50/50 mass.

    In the chart, a cell of fate ``f`` at progress ``s`` sits at

        u = L (2s - 1)
        w = trunk(s) + dir_f * amp_f * g_f(tau) + band,  tau = clip((s - s_s)/(1 - s_s))

    and is lifted to ``(u, w, z(u, w))``.  Its generator tangent is ``dX/ds``, taken
    analytically through the same lift -- the lateral offset within a band is a property
    of the cell and not of ``s``, so it contributes nothing to the tangent and the field
    runs *along* each band.  ``P`` is therefore tilted along each cell's own arm, and a
    model that reads ``P`` can tell the two lineages apart wherever they are resolved.

    ``shares`` is the mass split, one weight per :data:`FORK_FATES` row and in that order;
    it is normalised, so ``(70, 30)`` and ``(0.7, 0.3)`` say the same thing.  Only the
    labelling changes -- the two arms keep their geometry, so a cloud at ``(0.1, 0.9)`` is
    the same Y with one branch drawn thin.  That thinning is real and is part of what an
    imbalanced setting tests: at ``N = 3000`` and ``(0.1, 0.9)`` the minority arm holds
    ~90 cells in p_1, its band is sparse against ``FORK_BAND``, and ``P``'s neighbourhoods
    reach across the fork more readily than they do at equal mass.  Raise ``N`` if the
    question is the coupling rather than the sampling.
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, size=N)

    pick = np.empty(N, dtype=np.int64)
    pick[np.argsort(s)] = _stratify(N, _shares(shares))
    fate = np.array([f[0] for f in FORK_FATES], dtype=np.int64)[pick]
    direction = np.array([f[1] for f in FORK_FATES])[pick]
    amp = np.array([f[2] for f in FORK_FATES])[pick]
    expo = np.array([f[3] for f in FORK_FATES])[pick]
    sness = np.array([f[4] for f in FORK_FATES])[pick]

    tau = np.clip((s - FORK_SPLIT) / (1.0 - FORK_SPLIT), 0.0, 1.0)
    dtau = np.where(s > FORK_SPLIT, 1.0 / (1.0 - FORK_SPLIT), 0.0)
    w_trunk, dw_trunk = _trunk(s)
    g, dg = _arm(tau, expo, sness)

    u = FORK_L * (2.0 * s - 1.0)
    w = w_trunk + direction * amp * g + rng.uniform(-FORK_BAND, FORK_BAND, N)
    dw = dw_trunk + direction * amp * dg * dtau
    du = np.full(N, 2.0 * FORK_L)

    z_u, z_w = _fork_dz(u, w)
    e_u = np.stack([np.ones_like(u), np.zeros_like(u), z_u], axis=1)
    e_w = np.stack([np.zeros_like(w), np.ones_like(w), z_w], axis=1)

    X = np.stack([u, w, _fork_z(u, w)], axis=1)
    X = X + rng.normal(0.0, noise_sd, size=X.shape)
    v = du[:, None] * e_u + dw[:, None] * e_w

    region = np.full(N, GAP, dtype=np.int64)
    region[s <= FORK_S0] = SOURCE
    region[s >= FORK_S1] = TARGET

    return _finish("ForkSheet", X, v, region, s.copy(), lam,
                   uv=np.stack([u, w], axis=1), fate=fate)


# --------------------------------------------------------------------------- #
#  Fate of a transported cloud
# --------------------------------------------------------------------------- #
#: ball radius, as a fraction of the distance between the two arms' p_1 centres.
#:
#: Not a tuning knob.  Two balls of radius ``d/2`` whose centres are ``d`` apart touch at
#: exactly one point: any larger and a cell can be inside both and the assignment needs a
#: tie-break, any smaller and a dead band opens between them for no benefit.  Measured on
#: the default cloud, ``d/2`` puts **every** p_1 cell of each arm inside its own ball, not
#: one inside the other's, and none of p_0 inside either -- so perfect data scores exactly
#: 1 and a model that does not move still scores 0, with nothing fitted to make that so.
#: A quarter of ``d``, the other natural guess, leaves 43% of the real p_1 cells outside
#: both balls and caps the metric at 0.57.
FATE_BALL_FRAC = 0.5

#: quantile of p_1, by progress, whose two ends define an arm's outward direction.
FATE_AXIS_Q = 0.1

#: an arm's lineage counts as *resolved* at a progress once the two bands no longer
#: overlap, i.e. once the centre lines are more than this many band half-widths apart.
FATE_RESOLVED_MULT = 2.0


def fate_balls(ds: RouteDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(fates, centres, axes, radius)`` -- one region per arm, around its p_1 cells.

    The centre is the arm's p_1 **centroid** and not its far tip.  The tip is at one end
    of a stretch that is as long as the arms are far apart -- p_1 runs the terminal third
    of each arm, ~1.1 in extent against 1.37 between the centres -- so a ball hung there
    reaches back over only part of its own arm however the radius is set: at ``d/2`` from
    the tips, 23% of the real p_1 cells are outside both balls.  From the centroid an
    arm's own cells reach 0.61 and a ball of 0.685 holds all of them.

    The radius is measured off the **closest** pair of centres, so no two regions overlap
    however many arms a cloud has.

    ``axes`` is each arm's outward unit direction, the centroid of its far
    :data:`FATE_AXIS_Q` of p_1 by progress minus that of its near one.  It exists only so
    that :func:`fate_inside` knows which way *further along this arm* is.
    """
    assert ds.fate is not None, f"{ds.name} carries no fate labels"
    tgt = ds.region == TARGET
    fates = np.array([int(f) for f in sorted(np.unique(ds.fate))])
    centres, axes = [], []
    for f in fates:
        m = tgt & (ds.fate == f)
        pts, s = ds.X[m], ds.prog[m]
        centres.append(pts.mean(axis=0))
        a = (pts[s >= np.quantile(s, 1.0 - FATE_AXIS_Q)].mean(axis=0)
             - pts[s <= np.quantile(s, FATE_AXIS_Q)].mean(axis=0))
        axes.append(a / np.linalg.norm(a))
    centres, axes = np.stack(centres), np.stack(axes)
    d = min(float(np.linalg.norm(centres[i] - centres[j]))
            for i in range(len(fates)) for j in range(i + 1, len(fates)))
    return fates, centres, axes, FATE_BALL_FRAC * d


def fate_inside(pts: np.ndarray, centres: np.ndarray, axes: np.ndarray,
                r: float) -> np.ndarray:
    """``(n, n_fates)`` membership: each ball, with its **outward half run to infinity**.

    Overshoot is not a routing error.  A cell that sails straight out past the end of its
    own arm chose that arm and nothing else, and how far past it went is a distance, which
    the endpoint W_2 in the main table already measures in the data's own units.  A closed
    ball charges it as *outside*, i.e. scores it the same as a cell that stalled in the
    trunk -- and on this cloud that is not a corner case: the region beyond p_1 is where
    95% of FFM's outside mass sits, because the Finsler geodesic arrives at full speed
    (terminal ``|dx/dt|`` ~ 4.5 against OT-CFM's ~1.0) and Phase 3's distillation residual
    scales with it.

    So the ball is cut in half at its centre, perpendicular to the arm, and the outer half
    is extended along the axis without bound: inside means *within ``r`` of the arm's
    outward ray*, plus the near hemisphere so that the p_1 cells behind the centroid still
    count.  Undershoot and lateral flight are still caught, since both leave the cylinder.
    Opening the far end cannot create an overlap -- the arms diverge, so the two rays only
    separate -- and :func:`fate_table` asserts it.
    """
    v = np.asarray(pts, dtype=np.float64)[:, None, :] - centres[None]
    t = (v * axes[None]).sum(axis=2)
    perp = np.linalg.norm(v - t[..., None] * axes[None], axis=2)
    return (perp <= r) & ((t >= 0.0) | (np.linalg.norm(v, axis=2) <= r))


def _separation(s: np.ndarray) -> np.ndarray:
    """Chart distance between the two arms' centre lines at progress ``s``.

    The trunk cancels -- it is common to both -- so this is the lateral spread of the
    fates' offsets, zero before :data:`FORK_SPLIT` and monotone after.
    """
    tau = np.clip((np.asarray(s, dtype=np.float64) - FORK_SPLIT) / (1.0 - FORK_SPLIT),
                  0.0, 1.0)
    off = np.stack([d * a * _arm(tau, e, n)[0] for _, d, a, e, n in FORK_FATES])
    return off.max(axis=0) - off.min(axis=0)


def fate_resolved(ds: RouteDataset, idx: np.ndarray) -> np.ndarray:
    """Which of the cells ``idx`` already sit on a distinguishable arm.

    Deep in the trunk a cell carries a lineage label but is not yet *on* either arm: the
    two bands overlap, and the cell would have been drawn in the same place under either
    label.  Asking a model to route such a cell correctly is asking it to read a coin
    flip, so per-cell scoring is restricted to the cells for which the answer is legible
    from the data at all.  Analytic rather than empirical -- it is a property of the
    generator, not of the draw.
    """
    return _separation(ds.prog[np.asarray(idx, dtype=np.int64)]) > \
        FATE_RESOLVED_MULT * FORK_BAND


def fate_scored(ds: RouteDataset, src: np.ndarray, cloud: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(keep, true, where)`` -- the per-cell answer that :func:`fate_table` averages.

    ``keep`` is over all of ``src`` (:func:`fate_resolved`); ``true`` and ``where`` are over
    the kept cells only.  Both are **positions** into ``fate_balls``' ``fates`` and not
    lineage labels, so a cell is correctly routed exactly where ``where == true``;
    ``where == len(fates)`` is *outside*.  Exposed so a figure can colour the very points
    the table counts, rather than re-deriving the rule beside it and drifting from it.
    """
    fates, centres, axes, r = fate_balls(ds)
    src = np.asarray(src, dtype=np.int64)
    cloud = np.asarray(cloud, dtype=np.float64)
    assert src.shape == (len(cloud),), \
        f"one source index per transported point; got {src.shape} and {cloud.shape}"

    keep = fate_resolved(ds, src)
    assert keep.any(), "no source cell has its arm resolved; nothing to score"
    true = np.searchsorted(fates, ds.fate[src][keep])

    inside = fate_inside(cloud[keep], centres, axes, r)
    assert inside.sum(axis=1).max() <= 1, "the balls overlap; FATE_BALL_FRAC is above 0.5"
    where = np.where(inside.any(axis=1), inside.argmax(axis=1), len(fates))
    return keep, true, where


def fate_table(ds: RouteDataset, src: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Where each arm's cells ended up: rows the true arm, columns each ball then *outside*.

    Rows sum to one.  ``src`` are full-cloud indices of the transported cells, in the
    order of ``cloud``'s rows -- ``ts.idx[ts.p0]`` for a trajectory endpoint, since
    ``integrate`` starts from ``x0`` and never reorders.

    Conditioning on the true arm is what makes this more than a marginal.  A share is
    blind to a permutation -- an arm that sends every lower cell to the upper ball and
    vice versa reproduces the target split exactly -- and here that lands wholly off the
    diagonal.  Each row is already a rate within one lineage, so the rows can be read
    against each other at any ``shares`` setting without reweighting: a perfect transport
    is the identity whether the split is 50/50 or 10/90.

    *outside* is one column and not two failures: it is mass that committed to no arm,
    whether it stopped short or left the manifold sideways.  Mass that ran *past* the end
    of its arm is not outside -- see :func:`fate_inside`.
    """
    fates = fate_balls(ds)[0]
    _keep, true, where = fate_scored(ds, src, cloud)

    rows = []
    for i, f in enumerate(fates):
        m = true == i
        assert m.any(), f"arm {f} has no resolved source cell"
        rows.append([float((where[m] == j).mean()) for j in range(len(fates) + 1)])
    return np.array(rows)


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
_MAKERS = {"ForkSheet": make_fork_sheet}
assert tuple(_MAKERS) == DATASET_NAMES, "the maker table and the registry disagree"


def make_dataset(name: str, **kw) -> RouteDataset:
    if name not in _MAKERS:
        raise KeyError(f"unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    return _MAKERS[name](**kw)


if __name__ == "__main__":
    for nm in DATASET_NAMES:
        print(make_dataset(nm).summary())
