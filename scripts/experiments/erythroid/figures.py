"""Two figures: what the cloud is, and where each arm routed its mass.

**E-data** — the UMAP, twice.  Left, coloured by the five erythroid lineage stages, which
is the biology the benchmark is about; right, coloured by the three equal-quantile
latent-time marginals with the withheld middle one in orange, which is the *task*.

**E-compare** — one row per dimension, one column per arm: the two shown marginals set
back, the withheld middle marginal in orange at full strength, and a few of the arm's
routes drawn through it.

**E-paths** — the d = 2 routes on their own, at a size where an individual route is
readable.  E-compare answers "did the mass land in the right place"; this one answers
"how did it get there", which is the question a *geometry* claim is about.  The ink is
the Pancreas notebook's — :func:`~scripts.experiments.sheet.paper_figures._draw_paths`,
with the start columns picked at t = 0 alone and handed to every panel — laid out two
columns wide in :data:`PATH_BLOCKS` order, so the two ours panels share the last row.
Only d = 2 is drawn, because there the plane *is* the model space.

The palette is :mod:`scripts.experiments.sheet.paper_figures`' ``PAPER_REGIONS``,
imported not restated: source is violet, target is blue, the thing to be recovered is
orange.

A panel at d = 20 / 50 plots the **first two principal components** of the unstandardised
PCA prefix the arm was trained and scored in.  It is a projection and is labelled as one;
pushing the prediction back to the UMAP would be worse, since a UMAP has no inverse and
such a panel would show a fitted embedding rather than the prediction.  The scored
numbers in E1 are computed in the full space regardless of what is drawn.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402

from scripts.experiments.sheet.datasets import GAP, SOURCE, TARGET          # noqa: E402
from scripts.experiments.sheet.paper_figures import (                       # noqa: E402
    PAPER_REGIONS, _PATH_CASING, _PATH_INK, _draw_paths, _lateral_starts,
)
from .datasets import DIMS, HOLDOUT_BIN, TRAIN_BINS, load_cloud     # noqa: E402
from .train import ARM_BLOCK, ARM_LABELS, ARMS, TRAJ_DIR    # noqa: E402

DRAWN_BLOCKS = ("harness", "ours")


def _drawn(arms):
    keep = tuple(a for a in arms if ARM_BLOCK.get(a) in DRAWN_BLOCKS)
    assert keep, f"none of {tuple(arms)} is in a drawn block {DRAWN_BLOCKS}"
    return keep


def _panel_label(arm: str) -> str:
    return ARM_LABELS[arm].replace(" (harness)", "")

#: routes drawn per panel.  More becomes a ribbon at this size.
N_TRAJ_PANEL = 8
#: cells scattered per marginal, out of ~3272: drawn in full they are a solid block.
N_SCATTER = 1100


def _thin(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


def _spread(n: int, k: int) -> np.ndarray:
    """``k`` indices spread evenly over ``n``: the saved slab's routes are themselves a
    ``linspace`` over the source marginal's ordering, so ``[:k]`` off the front would draw
    only the first ``k/64`` of the source cloud."""
    return np.linspace(0, n - 1, min(k, n)).astype(int)


def _scatter(ax, X, colour, s, alpha, zorder) -> None:
    ax.scatter(X[:, 0], X[:, 1], s=s, alpha=alpha, color=colour, linewidths=0,
               zorder=zorder)


def _bare(ax) -> None:
    """No frame, no ticks — the Sheet's and the circles' panel style."""
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


# --------------------------------------------------------------------------- #
#  E-data: the UMAP
# --------------------------------------------------------------------------- #
def figure_data(cloud, out_dir: str, stem: str = "erythroid_umap") -> str:
    """The UMAP by lineage stage, and the same UMAP by latent-time marginal."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    U = cloud.umap

    types = list(dict.fromkeys(cloud.celltype.tolist()))
    ramp = plt.get_cmap("viridis")(np.linspace(0.08, 0.94, len(types)))
    for colour, name in zip(ramp, types):
        m = cloud.celltype == name
        axes[0].scatter(U[m, 0], U[m, 1], s=3.2, alpha=0.75, color=colour,
                        linewidths=0, label=f"{name} ({int(m.sum())})")
    axes[0].set_title("Erythroid lineage "
                      f"($N = {len(U)}$ cells)", fontsize=11)
    axes[0].legend(loc="best", frameon=False, fontsize=7.5, markerscale=2.2)

    key = {TRAIN_BINS[0]: (PAPER_REGIONS[SOURCE], "$p_0$: bin 0 (shown)", 0.5, 3.2),
           HOLDOUT_BIN: (PAPER_REGIONS[GAP], "withheld: bin 1 (target)", 0.95, 4.4),
           TRAIN_BINS[1]: (PAPER_REGIONS[TARGET], "$p_1$: bin 2 (shown)", 0.5, 3.2)}
    for b in (TRAIN_BINS[0], TRAIN_BINS[1], HOLDOUT_BIN):
        colour, label, alpha, size = key[b]
        m = cloud.bin_id == b
        axes[1].scatter(U[m, 0], U[m, 1], s=size, alpha=alpha, color=colour,
                        linewidths=0, label=f"{label}  $n={int(m.sum())}$")
    edges = cloud.diag["quantile_edges"]
    axes[1].set_title("Three equal-quantile UniTVelo latent-time bins\n"
                      f"edges at $t = {edges[1]:.3f},\\ {edges[2]:.3f}$", fontsize=11)
    axes[1].legend(loc="best", frameon=False, fontsize=8, markerscale=2.2)

    for ax in axes:
        _bare(ax)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
#  E-compare: predictions per dimension
# --------------------------------------------------------------------------- #
def panel(ax, cloud, traj_npz, dim: int) -> None:
    """One (dimension, arm) cell, drawn in the first two coordinates of the model space."""
    X = cloud.X
    i_src, i_end = cloud.marginal(TRAIN_BINS[0]), cloud.marginal(TRAIN_BINS[1])
    i_tgt = cloud.target
    _scatter(ax, X[i_src][_thin(len(i_src), N_SCATTER)], PAPER_REGIONS[SOURCE],
             2.6, 0.38, 1)
    _scatter(ax, X[i_end][_thin(len(i_end), N_SCATTER, seed=3)], PAPER_REGIONS[TARGET],
             2.6, 0.38, 1)
    _scatter(ax, X[i_tgt][_thin(len(i_tgt), N_SCATTER, seed=2)], PAPER_REGIONS[GAP],
             4.0, 0.90, 2)

    saved = np.asarray(traj_npz["paths"])
    paths = saved[:, _spread(saved.shape[1], N_TRAJ_PANEL)]
    for j in range(paths.shape[1]):
        p = paths[:, j]
        ax.plot(p[:, 0], p[:, 1], lw=2.6, alpha=0.9, color=_PATH_CASING,
                solid_capstyle="round", zorder=4)
        ax.plot(p[:, 0], p[:, 1], lw=1.1, alpha=1.0, color=_PATH_INK,
                solid_capstyle="round", zorder=5)
    _bare(ax)


def figure_compare(root: str, dims, arms, out_dir: str, seed: int = 0,
                   stem: str = "erythroid_compare") -> str:
    arms = _drawn(arms)
    traj_dir = os.path.join(root, TRAJ_DIR)

    nrow, ncol = len(dims), len(arms)
    fig = plt.figure(figsize=(3.4 * ncol, 3.4 * nrow))
    for i, dim in enumerate(dims):
        cloud = load_cloud(dim)
        for j, arm in enumerate(arms):
            ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1)
            path = os.path.join(traj_dir, f"d{dim}_{arm}_s{seed}.npz")
            assert os.path.exists(path), (
                f"missing {path}; the test stage writes one slab per (dim, arm, seed)")
            with np.load(path) as z:
                panel(ax, cloud, z, dim)
            if i == 0:
                ax.set_title(_panel_label(arm), fontsize=11, pad=4)
            if j == 0:
                axis = "z-scored UMAP" if dim == 2 else "first 2 PCs"
                ax.text(-0.05, 0.5, f"$d = {dim}$\n({axis})", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=5, color=PAPER_REGIONS[SOURCE],
                   label="bin 0: source (shown)"),
        plt.Line2D([], [], marker="o", ls="", ms=5, color=PAPER_REGIONS[TARGET],
                   label="bin 2: endpoint (shown)"),
        plt.Line2D([], [], marker="o", ls="", ms=6, color=PAPER_REGIONS[GAP],
                   label="withheld $t = 1/2$ marginal (target)"),
        plt.Line2D([], [], ls="-", lw=1.4, color=_PATH_INK, label="trajectories"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
#  E-paths: the d = 2 routes, full time axis
# --------------------------------------------------------------------------- #
#: routes drawn per E-paths panel: the Pancreas' count, one per lateral quantile.
N_TRAJ_2D = 5
#: E-paths panel columns.  Six panels down one column is a strip too tall to place.
PATH_NCOL = 2

PATH_BLOCKS = (
    ("Data", ("__data__",)),
    (r"Deterministic ($\sigma = 0$)", ("cfm", "mfm_land", "curly", "ffm")),
    (r"Stochastic ($\sigma > 0$)", ("pathb",)),
)


def _path_cells(arms):
    """The panels of E-paths in :data:`PATH_BLOCKS` order, flattened across the blocks."""
    keep = set(arms)
    missing = keep - {a for _h, g in PATH_BLOCKS for a in g}
    assert not missing, (
        f"PATH_BLOCKS places every arm in a block, but {sorted(missing)} is in none; a "
        f"new arm has to be declared deterministic or stochastic before it can be drawn")
    return [a for _h, group in PATH_BLOCKS for a in group
            if a == "__data__" or a in keep]


def panel_paths(ax, cloud, paths: np.ndarray) -> None:
    """One arm's d = 2 routes over the three marginals, in the Pancreas' path style."""
    X = cloud.X
    i_src, i_end, i_tgt = (cloud.marginal(TRAIN_BINS[0]),
                           cloud.marginal(TRAIN_BINS[1]), cloud.target)
    _scatter(ax, X[i_src][_thin(len(i_src), N_SCATTER)], PAPER_REGIONS[SOURCE],
             1.4, 0.42, 1)
    _scatter(ax, X[i_end][_thin(len(i_end), N_SCATTER, seed=3)], PAPER_REGIONS[TARGET],
             1.4, 0.42, 1)
    _scatter(ax, X[i_tgt][_thin(len(i_tgt), N_SCATTER, seed=2)], PAPER_REGIONS[GAP],
             2.4, 1.0, 2)
    if paths is not None:
        _draw_paths(ax, paths, three_d=False)


def figure_paths(root: str, arms, out_dir: str, seed: int = 0, dim: int = 2,
                 stem: str = "erythroid_paths") -> str:
    """The full t = 0 → 1 routes, one panel per arm on a :data:`PATH_NCOL`-wide grid."""
    arms = _drawn(arms)
    traj_dir = os.path.join(root, TRAJ_DIR)
    cloud = load_cloud(dim)

    slabs = {}
    for arm in arms:
        path = os.path.join(traj_dir, f"d{dim}_{arm}_s{seed}.npz")
        assert os.path.exists(path), (
            f"missing {path}; the test stage writes one slab per (dim, arm, seed)")
        with np.load(path) as z:
            slabs[arm] = np.asarray(z["paths"], dtype=np.float64)

    n_src = min(p.shape[1] for p in slabs.values())
    cols = _lateral_starts(next(iter(slabs.values()))[0, :n_src],
                           min(N_TRAJ_2D, n_src))

    lo, hi = cloud.X[:, :2].min(axis=0), cloud.X[:, :2].max(axis=0)
    pad = 0.04 * np.maximum(hi - lo, 1e-9)

    cells = _path_cells(arms)
    ncol = PATH_NCOL
    nrow = -(-len(cells) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.95 * ncol, 2.25 * nrow),
                             layout="constrained")
    axes = np.asarray(axes).reshape(-1)
    for n, arm in enumerate(cells):
        ax = axes[n]
        panel_paths(ax, cloud, None if arm == "__data__" else slabs[arm][:, cols])
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        label = "the withheld marginal" if arm == "__data__" else _panel_label(arm)
        ax.set_title(label, fontsize=9, pad=3)
        if n == 0:
            ax.set_ylabel(f"$d = {dim}$", fontsize=9)
    for ax in axes[len(cells):]:
        ax.set_axis_off()

    dot = dict(marker="o", ls="", ms=3)
    fig.legend(handles=[
        plt.Line2D([], [], **dot, color=PAPER_REGIONS[SOURCE], label="$p_0$ (shown)"),
        plt.Line2D([], [], **dot, color=PAPER_REGIONS[TARGET], label="$p_1$ (shown)"),
        plt.Line2D([], [], **dot, color=PAPER_REGIONS[GAP],
                   label="withheld $t = 1/2$ marginal (scored)"),
        plt.Line2D([], [], marker="o", ls="", ms=4, color="#ffffff",
                   markeredgecolor=_PATH_INK, label="$t{=}0$"),
        plt.Line2D([], [], marker="*", ls="", ms=7, color="#ffffff",
                   markeredgecolor=_PATH_INK, label="$t{=}1$"),
    ], loc="outside lower center", ncol=5, frameon=False, fontsize=7,
        handletextpad=0.3, columnspacing=1.5)
    return _save(fig, out_dir, stem)


def _save(fig, out_dir: str, stem: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, stem)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
