"""Local intrinsic dimension of a point cloud: four estimators, one shared neighbourhood.

Why this is in the repository
-----------------------------
Every real-data benchmark here reports the *same* cells at ``d = 2 / 10 / 20 / 50``, and the
tables move a great deal between those columns — the pancreas mid-:math:`W_2` roughly
doubles from d = 2 to d = 10 and the floors move with it.  Some of that is the transport
getting harder and some of it is the *representation*: a PCA prefix that is already wider
than the manifold is adding coordinates that carry noise, and a prefix that is narrower
than the manifold is folding it.  Neither the tables nor the floors distinguish the two.

A local intrinsic-dimension estimate does, cheaply, and before any model is fit.  It is a
property of the cloud, so it is computed once near the top of a notebook and never enters
a model input; nothing here reads a marginal, a split or a transition matrix.

The four estimators
-------------------
They are picked to fail differently, which is the only reason to run four:

``twonn_local``  Facco et al. (2017).  Reads only the *two* nearest neighbours of a cell —
    the shortest possible baseline, so it is the least sensitive to curvature and the most
    sensitive to duplicate cells and to measurement noise.  "Local" here means the
    Pareto MLE is pooled over a cell's ``k``-neighbourhood rather than over the whole
    cloud, which is what turns a global scalar into a per-cell number.
``knn_mle``      Levina & Bickel (2005) with the MacKay & Ghahramani (2005) ``k − 2``
    bias correction.  Uses the whole distance ladder :math:`T_1 \\dots T_k`, so it
    averages away the noise TwoNN sees and pays for it by seeing curvature over the whole
    neighbourhood — it biases *down* when the manifold curves inside radius :math:`T_k`.
``tle``          Amsaleg et al. (2019), the Tight Local intrinsic dimensionality
    Estimator.  Built for exactly the regime the other two are weakest in: a
    neighbourhood small enough that ``k`` is not large.  It extracts
    :math:`O(k^2)` distance measurements from the same ``k`` neighbours by taking every
    *pair*, so its variance at fixed ``k`` is much lower than the MLE's.
``lpca_pr``      Local PCA, summarised by the **participation ratio**
    :math:`(\\sum \\lambda)^2 / \\sum \\lambda^2` of the local covariance spectrum.  The
    only one of the four that is not a distance-ladder estimator: it asks how many
    directions the neighbourhood actually spreads in.  It is the one that answers "how
    many PCA columns would I need", which is the question the ``d`` columns pose, and it
    is also the one with a hard ceiling — see the caveat below.

Two caveats that must travel with any number this module returns
----------------------------------------------------------------
1. **The participation ratio cannot exceed the neighbourhood size.**  A cloud of
   :math:`k + 1` points spans at most :math:`k` dimensions, so ``lpca_pr`` is bounded by
   ``k`` no matter how wide the ambient space is.  At ``k = 50`` in a 2000-gene space
   that bound is *binding* if the data really is high-dimensional, and a PR that sits
   near ``k`` should be read as "at least ``k``", not as an estimate.  The distance-ladder
   estimators have no such ceiling.
2. **All four are biased down on curved, noisy, finite samples**, and they are biased down
   by different amounts.  The comparison that means something is *between representations
   at a fixed estimator*, which is why :func:`histograms` puts the four representations
   side by side inside one estimator rather than the other way round.

Neighbourhood convention
------------------------
One ``k`` for all four estimators (:data:`K_NEIGHBOURS`), one set of sampled cells
(:func:`sample_cells`), one k-NN graph per representation (:func:`prepare`).  Distances
are Euclidean in whatever coordinates the representation supplies, self is excluded, and
the ladder is sorted ascending.  ``twonn_local`` is the one estimator whose *measurement*
is not at scale ``k``: its ratio :math:`\\mu = r_2 / r_1` is always the two nearest
neighbours in the full cloud, and ``k`` only sets how many such ratios are pooled.  That
is what the estimator is; it is noted rather than papered over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

#: neighbours per cell, shared by every estimator so the four are comparable.  50 is large
#: enough for the pair-based TLE and the local PCA to have something to work with and
#: small enough that a 3.7k-cell cloud still has ~70 disjoint neighbourhoods in it.
K_NEIGHBOURS = 50

#: cells the estimators are evaluated at.  Every estimator and every representation uses
#: the *same* draw (:func:`sample_cells`), so a difference between two histograms is a
#: difference in the geometry and not in which cells were looked at.
N_SAMPLE = 1500

#: fixed, because the sample is a reported quantity like any other.
SAMPLE_SEED = 0

#: floor for every division and logarithm below.  Single-cell clouds contain exact
#: duplicate cells (two cells with identical counts in the retained genes), which put a
#: zero in the distance ladder; they are floored here and *counted*, not dropped silently.
EPS = 1e-12


# --------------------------------------------------------------------------- #
#  The shared neighbourhood
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Local:
    """One representation's k-NN geometry, evaluated at the sampled cells.

    ``dist`` and ``nbr`` are the ladder at the *sampled* cells only; ``mu`` is TwoNN's
    ratio at **every** cell, because a sampled cell's neighbourhood pools the ratios of
    its neighbours and those are not themselves sampled.
    """

    name: str
    X: np.ndarray                 # (N, d) the representation itself
    idx: np.ndarray               # (m,)   sampled cell indices into X
    dist: np.ndarray              # (m, k) ascending distances, self excluded
    nbr: np.ndarray               # (m, k) neighbour indices, matching dist
    mu: np.ndarray                # (N,)   r_2 / r_1 at every cell (TwoNN's ratio)
    n_duplicate: int              # cells whose nearest neighbour sits at distance 0

    @property
    def k(self) -> int:
        return int(self.dist.shape[1])

    @property
    def m(self) -> int:
        return int(len(self.idx))

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


def sample_cells(n: int, n_sample: int = N_SAMPLE, seed: int = SAMPLE_SEED) -> np.ndarray:
    """``min(n_sample, n)`` distinct cell indices, sorted, from a fixed generator.

    Sorted so that two calls with the same seed are identical *and* so the returned order
    does not depend on the RNG's internal permutation order, which changes between numpy
    versions in a way ``default_rng`` does not promise to preserve.
    """
    assert n > 0, f"empty cloud: n = {n}"
    m = min(int(n_sample), int(n))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=m, replace=False))


def _drop_self(D: np.ndarray, I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove each row's own index from a ``(N, k+1)`` k-NN result, giving ``(N, k)``.

    Not simply ``[:, 1:]``: with duplicate cells the query is not guaranteed to be its own
    first neighbour, and on a cloud with a large duplicate block it can be pushed out of
    the window entirely.  Both cases are handled — if self is absent the last (farthest)
    column is dropped instead, which keeps every row the same width.
    """
    n, kp1 = I.shape
    keep = I != np.arange(n)[:, None]
    missing = keep.all(axis=1)
    keep[missing, -1] = False
    assert (keep.sum(axis=1) == kp1 - 1).all(), "self-removal left a ragged neighbourhood"
    return D[keep].reshape(n, kp1 - 1), I[keep].reshape(n, kp1 - 1)


def prepare(name: str, X: np.ndarray, idx: np.ndarray,
            k: int = K_NEIGHBOURS) -> Local:
    """Build the k-NN ladder for one representation, once, for all four estimators.

    The graph is built over the **whole** cloud and then sliced at ``idx``: a sampled
    cell's neighbours are its true neighbours, not its nearest fellow-samples.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    n = X.shape[0]
    assert X.ndim == 2, f"{name}: X must be (N, d), got {X.shape}"
    assert k + 2 < n, f"{name}: need n > k + 2 = {k + 2} cells, got {n}"
    idx = np.asarray(idx, dtype=np.int64)
    assert idx.ndim == 1 and idx.min() >= 0 and idx.max() < n, f"{name}: bad sample index"

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    D, I = _drop_self(*nn.kneighbors(X))
    assert (np.diff(D, axis=1) >= -1e-9).all(), f"{name}: k-NN distances are not sorted"

    mu = D[:, 1] / np.maximum(D[:, 0], EPS)
    return Local(name=name, X=X, idx=idx, dist=D[idx], nbr=I[idx], mu=mu,
                 n_duplicate=int((D[:, 0] <= EPS).sum()))


# --------------------------------------------------------------------------- #
#  Estimator 1 — local TwoNN
# --------------------------------------------------------------------------- #
#: Facco et al. discard the largest 10 % of the mu ratios before fitting, on the grounds
#: that the tail is where the locally-uniform-density assumption breaks first.  Kept at
#: their value; it also absorbs the ratios that a near-duplicate cell blows up.
#:
#: This is a **Type-II censoring** of the sample and the estimator below is corrected for
#: it.  Feeding a truncated sample to the plain MLE ``n / sum(log mu)`` instead — which is
#: what a naive reading of the two recipes together produces — shrinks the denominator
#: without shrinking the count and overestimates by ~35 % at every dimension we tested.
TWONN_DISCARD = 0.10


def twonn_local(loc: Local, discard: float = TWONN_DISCARD) -> np.ndarray:
    """Per-cell TwoNN (Facco et al., 2017), pooled over each cell's k-neighbourhood.

    For every cell :math:`j` the ratio :math:`\\mu_j = r_2(j) / r_1(j)` is Pareto(1, d)
    distributed under a locally uniform density, so the MLE over a set of :math:`n` such
    ratios is :math:`\\hat d = n / \\sum_j \\log \\mu_j`.  The set used at sampled cell
    :math:`i` is :math:`\\{i\\} \\cup \\mathrm{kNN}(i)`, which is what makes the estimate
    local; the ratios themselves are always two-nearest-neighbour ratios in the full
    cloud, per the module docstring.

    Ratios of exactly 1 (a duplicate cell: :math:`r_1 = r_2`) carry no information and
    would send :math:`\\hat d` to infinity, so they are dropped and the count is reduced
    with them.  A cell left with fewer than two usable ratios returns ``nan``.

    With ``discard > 0`` the largest ratios are censored rather than deleted:
    :math:`x_j = \\log \\mu_j` is :math:`\\mathrm{Exp}(d)`, so keeping the :math:`r`
    smallest of :math:`n` is a Type-II censored sample and its MLE is

    .. math::
        \\hat d = r \\big/ \\big[ \\textstyle\\sum_{j \\le r} x_{(j)}
                                  + (n - r)\\, x_{(r)} \\big] ,

    which reduces to :math:`n / \\sum_j x_j` at ``discard = 0``.  See
    :data:`TWONN_DISCARD` for what happens if the censoring term is left out.
    """
    assert 0.0 <= discard < 1.0, f"discard must be in [0, 1), got {discard}"
    pool = np.concatenate([loc.idx[:, None], loc.nbr], axis=1)      # (m, k+1)
    lm = np.log(np.maximum(loc.mu[pool], 1.0))
    lm = np.where(np.isfinite(lm) & (lm > EPS), lm, np.nan)
    lm = np.sort(lm, axis=1)                                        # nan sorts last

    n_ok = (~np.isnan(lm)).sum(axis=1)
    n_keep = np.maximum(2, np.floor(n_ok * (1.0 - discard)).astype(np.int64))
    take = (np.arange(lm.shape[1])[None, :] < n_keep[:, None]) & ~np.isnan(lm)

    cnt = take.sum(axis=1)
    kept = np.where(take, lm, 0.0)
    # x_(r), the largest ratio that survived — every censored one is known only to exceed it
    x_r = kept.max(axis=1)
    total = kept.sum(axis=1) + (n_ok - cnt) * x_r
    return np.where((cnt >= 2) & (total > EPS), cnt / np.maximum(total, EPS), np.nan)


# --------------------------------------------------------------------------- #
#  Estimator 2 — kNN maximum likelihood
# --------------------------------------------------------------------------- #
def knn_mle(loc: Local) -> np.ndarray:
    """Levina & Bickel (2005) MLE with the MacKay & Ghahramani (2005) correction.

    .. math::
        \\hat m_k(i) = \\Big[ \\tfrac{1}{k-2} \\sum_{j=1}^{k-1}
                              \\log \\tfrac{T_k(i)}{T_j(i)} \\Big]^{-1}

    The :math:`k-2` denominator rather than Levina & Bickel's :math:`k-1` is the
    inverse-of-the-mean form, which is unbiased for :math:`1/d`; the difference is a few
    percent at ``k = 50`` and is stated because the two forms are both in circulation.
    """
    assert loc.k >= 4, f"kNN-MLE needs k >= 4, got {loc.k}"
    D = np.maximum(loc.dist, EPS)
    s = np.log(D[:, -1:] / D[:, :-1]).sum(axis=1)
    return np.where(s > EPS, (loc.k - 2) / np.maximum(s, EPS), np.nan)


# --------------------------------------------------------------------------- #
#  Estimator 3 — TLE
# --------------------------------------------------------------------------- #
#: measurements below this fraction of the neighbourhood radius are dropped, per the
#: reference implementation.  Both members of a dropped ``(s, t)`` pair go together.
TLE_EPSILON = 1e-4


def _tle_point(nn: np.ndarray, dists: np.ndarray, epsilon: float) -> float:
    """TLE at one query from its ``k`` neighbours — Amsaleg et al. (2019), Alg. 2.

    Transcribed from the reference implementation in ``skdim.id.TLE`` (Bac et al.,
    scikit-dimension, BSD-3), which is itself a port of the authors' MATLAB.  The one
    deliberate change is the small-distance guard at the end: the reference slices the
    *rows* of a ``(1, k)`` array where it means to drop columns, which is a no-op
    whenever no distance is below ``epsilon`` — the normal case — and wrong when one is.
    Here the sub-``epsilon`` distances are masked out of the :math:`s_2` sum instead.
    """
    dists = np.asarray(dists, dtype=np.float64).reshape(1, -1)
    k = dists.shape[1]
    r = dists[0, -1]                                   # neighbourhood radius
    assert r > 0.0, "degenerate neighbourhood: every k-NN distance is zero"

    V = squareform(pdist(np.asarray(nn, dtype=np.float64)))   # (k, k) neighbour-neighbour
    Di = np.tile(dists.T, (1, k))                             # u_i down the rows
    Dj = Di.T                                                 # u_j across the columns
    Z2 = 2.0 * Di ** 2 + 2.0 * Dj ** 2 - V ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 2.0 * (r ** 2 - Di ** 2)
        A = Di ** 2 + V ** 2 - Dj ** 2
        B = Di ** 2 + Z2 - Dj ** 2
        S = r * (np.sqrt(np.maximum(A ** 2 + 4.0 * V ** 2 * (r ** 2 - Di ** 2), 0.0)) - A) / denom
        T = r * (np.sqrt(np.maximum(B ** 2 + 4.0 * Z2 * (r ** 2 - Di ** 2), 0.0)) - B) / denom

        # boundary 1: rows at the radius itself, where the denominator vanishes
        at_r = (dists == r).ravel()
        S[at_r, :] = (r * V[at_r, :] ** 2
                      / (r ** 2 + V[at_r, :] ** 2 - Dj[at_r, :] ** 2))
        T[at_r, :] = r * Z2[at_r, :] / (r ** 2 + Z2[at_r, :] - Dj[at_r, :] ** 2)
        # boundary 2/3: a neighbour sitting on the query collapses s and t to the other leg
        di0 = (Di == 0)
        S[di0], T[di0] = Dj[di0], Dj[di0]
        dj0 = (Dj == 0)
        S[dj0] = T[dj0] = r * V[dj0] / (r + V[dj0])

    # boundary 4: coincident neighbours contribute nothing; park them at r so that
    # log(x / r) = 0 and subtract their count from the measurement total.
    v0 = (V == 0)
    np.fill_diagonal(v0, False)
    S[v0] = T[v0] = r
    n_v0 = int(v0.sum())

    tiny = (T < epsilon) | (S < epsilon)
    np.fill_diagonal(tiny, False)
    n_tiny = int(tiny.sum())
    S[tiny] = T[tiny] = r

    S = np.log(np.maximum(S, EPS) / r)
    T = np.log(np.maximum(T, EPS) / r)
    np.fill_diagonal(S, 0.0)
    np.fill_diagonal(T, 0.0)

    ladder = dists.ravel()
    usable = ladder >= epsilon
    n_small = int((~usable).sum())
    s2 = float(np.log(np.maximum(ladder[usable], EPS) / r).sum())

    total = float(S.sum() + T.sum() + 2.0 * s2)
    n_meas = k ** 2 - n_tiny - n_small - n_v0
    if n_meas <= 0 or total >= -EPS:
        return float("nan")
    return -2.0 * n_meas / total


def tle(loc: Local, epsilon: float = TLE_EPSILON) -> np.ndarray:
    """TLE at every sampled cell.  ``O(k^2)`` per cell, so it is the slowest of the four."""
    out = np.empty(loc.m, dtype=np.float64)
    for a in range(loc.m):
        out[a] = _tle_point(loc.X[loc.nbr[a]], loc.dist[a], epsilon)
    return out


# --------------------------------------------------------------------------- #
#  Estimator 4 — local PCA, participation ratio
# --------------------------------------------------------------------------- #
#: cumulative-variance level for the companion rank reported next to the PR.
LPCA_LEVEL = 0.95


def lpca_pr(loc: Local, level: float = LPCA_LEVEL,
            chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Participation ratio of the local covariance spectrum, and the ``level``-variance rank.

    The neighbourhood is :math:`\\{i\\} \\cup \\mathrm{kNN}(i)`, centred on its own mean
    rather than on the query — a covariance about the query would count the offset between
    the query and the local centroid as a direction of spread.

    Returns ``(pr, n_level)``.  ``pr`` is :math:`(\\sum \\lambda)^2 / \\sum \\lambda^2`,
    which is ``1`` for a rank-one neighbourhood and ``q`` for ``q`` equal eigenvalues, and
    is the smooth answer to "how many directions".  ``n_level`` is the integer rank needed
    to reach ``level`` of the variance, the discrete answer, reported beside it because
    the two disagree in an informative way when the spectrum has a heavy tail.

    Both are capped at the neighbourhood size — see caveat 1 in the module docstring.
    """
    pool = np.concatenate([loc.idx[:, None], loc.nbr], axis=1)      # (m, k+1)
    n = pool.shape[1]
    pr = np.empty(loc.m, dtype=np.float64)
    n_level = np.empty(loc.m, dtype=np.float64)

    for s in range(0, loc.m, chunk):
        Y = loc.X[pool[s:s + chunk]]                                # (c, k+1, d)
        Y = Y - Y.mean(axis=1, keepdims=True)
        # the Gram matrix has the same non-zero spectrum as the covariance and is (k+1)^2
        # rather than d^2, which is the difference between 51x51 and 2000x2000 per cell
        if loc.d >= n:
            ev = np.linalg.eigvalsh(np.einsum("cid,cjd->cij", Y, Y) / (n - 1))
        else:
            ev = np.linalg.eigvalsh(np.einsum("cid,cie->cde", Y, Y) / (n - 1))
        ev = np.clip(ev, 0.0, None)[:, ::-1]                        # descending
        tot = ev.sum(axis=1)
        pr[s:s + chunk] = np.where(
            tot > EPS, tot ** 2 / np.maximum((ev ** 2).sum(axis=1), EPS), np.nan)
        cum = np.cumsum(ev, axis=1) / np.maximum(tot[:, None], EPS)
        n_level[s:s + chunk] = np.where(tot > EPS, (cum < level).sum(axis=1) + 1, np.nan)

    return pr, n_level


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def summarise(values: dict[str, np.ndarray], locs: dict[str, Local],
              extra: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per representation: ``mean`` and ``median`` first, then the spread.

    ``dropped`` is the number of sampled cells the estimator could not return a finite
    number for; it is a column rather than a footnote because a representation where it
    is not zero is a representation whose mean is over a different set of cells.
    """
    rows = []
    for name, v in values.items():
        loc, v = locs[name], np.asarray(v, dtype=np.float64)
        ok = np.isfinite(v)
        w = v[ok]
        assert w.size, f"{name}: every estimate is non-finite"
        row = {"ambient d": loc.d, "cells": loc.m, "k": loc.k,
               "mean": w.mean(), "median": float(np.median(w)), "sd": w.std(),
               "p10": float(np.percentile(w, 10)), "p90": float(np.percentile(w, 90)),
               "dropped": int((~ok).sum()), "dup cells": loc.n_duplicate}
        if extra and name in extra:
            row.update(extra[name])
        rows.append(pd.Series(row, name=name))
    return pd.DataFrame(rows)


#: above this ratio between the widest and the narrowest panel's spread, ``histograms``
#: stops sharing the x-axis.  Four panels on one axis is the better figure whenever the
#: panels are on comparable scales; it stops being one when the widest is several times
#: the narrowest, because then every other panel is a spike two pixels across.
SHARE_X_MAX_RATIO = 2.5


def histograms(values: dict[str, np.ndarray], title: str, stem=None,
               bins: int = 40, colour: str = "#4c72b0", fs: int = 8,
               clip: tuple[float, float] = (0.5, 99.5), share: str | bool = "auto"):
    """One panel per representation, each with its mean and median marked.

    ``share`` controls the x-axis.  Sharing it is what lets the panels be compared by eye
    and is preferred, but it only works while their spreads are of the same order: an
    ambient panel running to 90 puts three PCA panels that live inside ``[4, 20]`` into
    slivers, and a sliver shows neither the location the table already gives nor the shape
    the figure is for.  ``"auto"`` shares when the widest panel's clipped spread is within
    :data:`SHARE_X_MAX_RATIO` of the narrowest and gives each panel its own range and bin
    edges otherwise; ``True``/``False`` force it.  The choice is written into the x-label
    so a reader is never guessing which of the two figures they are looking at.

    ``clip`` trims each range to a percentile window so that a handful of near-duplicate
    cells cannot squash a panel into its first bin; the mean and median printed on each
    panel are over *all* finite values, not the trimmed ones.
    """
    import matplotlib.pyplot as plt

    names = list(values)
    finite = {n: np.asarray(values[n])[np.isfinite(values[n])] for n in names}

    def _range(w):
        a, b = np.percentile(w, list(clip))
        return (float(w.min()), float(w.max()) + 1e-9) if not b > a else (a, b)

    spans = {n: _range(w) for n, w in finite.items()}
    widths = [b - a for a, b in spans.values()]
    if share == "auto":
        share = max(widths) <= SHARE_X_MAX_RATIO * min(widths)
    if share:
        lo = min(a for a, _ in spans.values())
        hi = max(b for _, b in spans.values())
        spans = {n: (lo, hi) for n in names}

    fig, axes = plt.subplots(1, len(names), figsize=(2.15 * len(names), 2.25),
                             constrained_layout=True, sharex=bool(share), sharey=bool(share))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        w = finite[name]
        a, b = spans[name]
        edges = np.linspace(a, b, bins + 1)
        ax.hist(w, bins=edges, color=colour, alpha=0.85, lw=0)
        ax.set_xlim(a, b)
        mean, med = w.mean(), float(np.median(w))
        ax.axvline(mean, color="#c44e52", lw=1.1, ls="--")
        ax.axvline(med, color="#55a868", lw=1.1, ls="-")
        ax.set_title(name, fontsize=fs)
        # these distributions skew either way depending on the estimator, so the corner
        # with room in it is not always the same one; put the label over the lighter half
        left_mass = (w < 0.5 * (a + b)).mean()
        x, ha = (0.97, "right") if left_mass > 0.5 else (0.03, "left")
        ax.text(x, 0.95, f"mean {mean:.1f}\nmed  {med:.1f}", transform=ax.transAxes,
                ha=ha, va="top", fontsize=fs - 2, family="monospace")
        ax.tick_params(labelsize=fs - 2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("cells", fontsize=fs - 1)
    xlabel = "local ID" + ("" if share else " (own scale per panel)")
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=fs - 1)
    fig.suptitle(title, fontsize=fs + 1)

    if stem is not None:
        import pathlib
        stem = pathlib.Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    return fig
