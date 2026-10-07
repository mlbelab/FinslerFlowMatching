"""Three figures, all on the R2 UMAP: the task, where the mass landed, and the routes.

**I-data** — the 2901-cell UMAP once, coloured by the three pseudotime tertiles: violet
``p_0``, blue ``p_1``, and the withheld middle third in orange.  That is the whole task,
and it is the only picture of the cloud the notebook draws.

**I-compare** — one row per dimension, one column per arm: the two shown thirds set back,
the withheld third in orange at full strength, the arm's prediction at ``t_mid`` in ink,
and a few of its routes.

**I-paths** — the routes on their own at a size where one route is readable, in the
erythroid notebook's layout and with the Pancreas' path ink
(:func:`~scripts.experiments.sheet.paper_figures._draw_paths`): white casing, dark line,
open dot at ``t = 0`` and a star at ``t = 1``.

At ``d = 2`` — the only dimension drawn — there is nothing to project: the model space
*is* the UMAP chart, z-scored per axis (:func:`~scripts.experiments.itracer.bench.space`),
so :func:`to_umap` just undoes the z-scoring and the panels are the trained coordinates
themselves, exactly as in the erythroid figures.  Above ``d = 2`` it falls back to a
Gaussian regression onto the chart, which is a projection and inherits the UMAP's
distortions; that path is unused by the notebook.

Only ``d = 2`` is drawn, the same convention as the erythroid notebook: at ``d = 10`` the
first two principal coordinates are not a picture of the method.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                          # noqa: E402

from scripts.core.itracer_data import N_THIRDS                              # noqa: E402
from scripts.core.present import GAP, PAPER_REGIONS, SOURCE, TARGET         # noqa: E402
from scripts.experiments.sheet.paper_figures import (                       # noqa: E402
    _PATH_CASING, _PATH_INK, _draw_paths, _lateral_starts,
)

from .bench import ARM_LABEL, UMAP_DIM, traj_path                           # noqa: E402

#: the ink of a third, and what to call it.  Orange is the thing to be recovered, here as
#: in every other figure in the paper.
THIRD_COLOUR = {0: PAPER_REGIONS[SOURCE], 1: PAPER_REGIONS[GAP], 2: PAPER_REGIONS[TARGET]}
THIRD_LABEL = {0: r"$p_0$: early third (shown)",
               1: "withheld: middle third (scored)",
               2: r"$p_1$: late third (shown)"}

#: neighbours behind the PCA -> UMAP lift, used only above ``UMAP_DIM``.  The kernel's own
#: support, so a prediction is placed by the same neighbourhood size the geometry was read
#: at.
LIFT_K = 15

#: routes per panel.  More becomes a ribbon at this size.
N_TRAJ_PANEL = 5
N_TRAJ_2D = 5

#: I-paths panel columns: the widest layout that leaves at most one blank cell.  Four
#: panels want 2 x 2 and five want 3 x 2; a fixed 2 would give the five-panel figure three
#: rows and a hole, which reads as a missing arm rather than as a layout.
def path_ncol(n_cells: int) -> int:
    return min(3, n_cells) if n_cells > 4 else 2


def to_umap(cloud, Z: np.ndarray, dim: int, k: int = LIFT_K) -> np.ndarray:
    """Place points of the ``dim``-dimensional model space on the UMAP.

    At ``UMAP_DIM`` this is exact and free: the model space is the chart z-scored per
    axis, so multiplying the axis scales back on and re-centring returns the trained
    coordinates unchanged.  Above it, Gaussian regression over the cloud at each query
    point's own bandwidth — ``h`` is its distance to the ``k``-th nearest cell and every
    cell is weighted ``exp(-d^2/h^2)``.  A hard ``k``-NN average would be cheaper but is
    discontinuous: the neighbour set flips as a trajectory moves and the lifted route
    comes out visibly jagged, which is an artefact of the lift and not of the flow.
    ``Z`` may be ``(n, dim)`` or ``(steps, n, dim)``; the shape comes back with the last
    axis replaced by 2.
    """
    Z = np.asarray(Z, dtype=np.float64)
    flat = Z.reshape(-1, Z.shape[-1])
    assert flat.shape[1] == dim, (Z.shape, dim)
    if dim == UMAP_DIM:
        U = np.asarray(cloud.umap, dtype=np.float64)
        return flat.reshape(Z.shape) * U.std(axis=0) + U.mean(axis=0)
    X = np.asarray(cloud.X[:, :dim], dtype=np.float64)
    d2 = np.maximum(((flat[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0)
    h2 = np.partition(d2, k - 1, axis=1)[:, k - 1][:, None] + 1e-12
    w = np.exp(-d2 / h2)
    out = (w @ cloud.umap) / w.sum(1)[:, None]
    return out.reshape(*Z.shape[:-1], 2)


def _bare(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _scatter(ax, Y, colour, s, alpha, zorder, **kw) -> None:
    ax.scatter(Y[:, 0], Y[:, 1], s=s, alpha=alpha, color=colour, linewidths=0,
               zorder=zorder, **kw)


def _marginals(ax, cloud, faint: float = 0.35) -> None:
    """The two shown thirds set back, the withheld third in orange on top."""
    U = cloud.umap
    for b in (0, 2):
        _scatter(ax, U[cloud.marginal(b)], THIRD_COLOUR[b], 2.6, faint, 1)
    _scatter(ax, U[cloud.marginal(1)], THIRD_COLOUR[1], 4.0, 0.90, 2)


def _slab(dim: int, arm: str, seed: int, point: str, smoke: bool) -> dict:
    path = traj_path(dim, arm, seed, point, smoke)
    assert os.path.exists(path), (
        f"missing {path}; `scripts.experiments.itracer.bench` writes one slab per "
        f"(point, dim, arm, seed)")
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


# --------------------------------------------------------------------------- #
#  I-data
# --------------------------------------------------------------------------- #
def figure_data(cloud, out_dir: str, stem: str = "itracer_umap") -> str:
    """The cloud, once: which cells are ``p_0``, which are ``p_1``, which are withheld."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    U = cloud.umap
    for b in range(N_THIRDS):
        m = cloud.third == b
        ax.scatter(U[m, 0], U[m, 1], s=4.4, alpha=0.85, linewidths=0,
                   color=THIRD_COLOUR[b], label=f"{THIRD_LABEL[b]}  $n={int(m.sum())}$")
    ax.set_title(f"iTracer R2 hindbrain, {len(U)} lineage-recorded cells\n"
                 f"tertiles of the NPC $\\to$ neuron pseudotime at "
                 f"$s = {cloud.edges[0]:.3f},\\ {cloud.edges[1]:.3f}$", fontsize=11)
    ax.legend(loc="best", frameon=False, fontsize=8, markerscale=2.4)
    _bare(ax)
    fig.tight_layout()
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
#  I-compare
# --------------------------------------------------------------------------- #
def panel(ax, cloud, slab: dict, dim: int) -> None:
    _marginals(ax, cloud)
    cols = _lateral_starts(np.asarray(slab["paths"][0], dtype=np.float64),
                           min(N_TRAJ_PANEL, slab["paths"].shape[1]))
    paths = to_umap(cloud, slab["paths"][:, cols], dim)
    for j in range(paths.shape[1]):
        p = paths[:, j]
        ax.plot(p[:, 0], p[:, 1], lw=2.0, alpha=0.9, color=_PATH_CASING,
                solid_capstyle="round", zorder=4)
        ax.plot(p[:, 0], p[:, 1], lw=0.9, alpha=1.0, color=_PATH_INK,
                solid_capstyle="round", zorder=5)
    _scatter(ax, to_umap(cloud, slab["mid"], dim), _PATH_INK, 5.0, 0.8, 6)
    _bare(ax)


def figure_compare(cloud, dims, arms, out_dir: str, seed: int = 0,
                   point: str = "erythroid", smoke: bool = False,
                   stem: str = "itracer_compare") -> str:
    """One row per dimension, one column per arm, at one of the two hyper-parameter sets.

    The row label names the space and not only its width, because only ``UMAP_DIM`` is
    drawn where it was trained: the rows above it are lifted onto the chart by
    :func:`to_umap` and carry that projection's distortions.
    """
    dims, arms = tuple(dims), tuple(arms)
    nrow, ncol = len(dims), len(arms)
    fig = plt.figure(figsize=(3.4 * ncol, 3.4 * nrow))
    for i, dim in enumerate(dims):
        for j, arm in enumerate(arms):
            ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1)
            panel(ax, cloud, _slab(dim, arm, seed, point, smoke), dim)
            if i == 0:
                ax.set_title(ARM_LABEL[arm], fontsize=11, pad=4)
            if j == 0:
                kind = "UMAP" if dim == UMAP_DIM else "PCs, lifted"
                ax.text(-0.05, 0.5, f"$d = {dim}$\n({kind})", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)

    dot = dict(marker="o", ls="", ms=5)
    fig.legend(handles=[
        plt.Line2D([], [], **dot, color=THIRD_COLOUR[0], label="$p_0$ (shown)"),
        plt.Line2D([], [], **dot, color=THIRD_COLOUR[2], label="$p_1$ (shown)"),
        plt.Line2D([], [], **dot, color=THIRD_COLOUR[1],
                   label="withheld middle third (scored)"),
        plt.Line2D([], [], **dot, color=_PATH_INK,
                   label=r"prediction at $t_{\mathrm{mid}}$"),
        plt.Line2D([], [], ls="-", lw=1.4, color=_PATH_INK, label="trajectories"),
    ], loc="lower center", ncol=5, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    return _save(fig, out_dir, f"{stem}_{point}")


# --------------------------------------------------------------------------- #
#  I-paths
# --------------------------------------------------------------------------- #
def figure_paths(cloud, arms, out_dir: str, seed: int = 0, dim: int = UMAP_DIM,
                 point: str = "erythroid", smoke: bool = False,
                 stem: str = "itracer_paths") -> str:
    """The full ``t = 0 -> 1`` routes, one panel per arm beside the cloud they cross."""
    arms = tuple(arms)
    slabs = {a: _slab(dim, a, seed, point, smoke) for a in arms}
    n_src = min(s["paths"].shape[1] for s in slabs.values())
    cols = _lateral_starts(np.asarray(next(iter(slabs.values()))["paths"][0, :n_src],
                                      dtype=np.float64), min(N_TRAJ_2D, n_src))

    U = cloud.umap
    lo, hi = U.min(axis=0), U.max(axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-9)

    cells = ("__data__",) + arms
    ncol = path_ncol(len(cells))
    nrow = -(-len(cells) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.95 * ncol, 2.35 * nrow),
                             layout="constrained")
    axes = np.asarray(axes).reshape(-1)
    for n, arm in enumerate(cells):
        ax = axes[n]
        _marginals(ax, cloud, faint=0.42)
        if arm != "__data__":
            _draw_paths(ax, to_umap(cloud, slabs[arm]["paths"][:, cols], dim),
                        three_d=False)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        _bare(ax)
        ax.set_title("the withheld third" if arm == "__data__" else ARM_LABEL[arm],
                     fontsize=9, pad=3)
        if n == 0:
            ax.set_ylabel(f"$d = {dim}$", fontsize=9)
    for ax in axes[len(cells):]:
        ax.set_axis_off()

    dot = dict(marker="o", ls="", ms=3)
    fig.legend(handles=[
        plt.Line2D([], [], **dot, color=THIRD_COLOUR[0], label="$p_0$ (shown)"),
        plt.Line2D([], [], **dot, color=THIRD_COLOUR[2], label="$p_1$ (shown)"),
        plt.Line2D([], [], **dot, color=THIRD_COLOUR[1],
                   label="withheld middle third (scored)"),
        plt.Line2D([], [], marker="o", ls="", ms=4, color="#ffffff",
                   markeredgecolor=_PATH_INK, label="$t{=}0$"),
        plt.Line2D([], [], marker="*", ls="", ms=7, color="#ffffff",
                   markeredgecolor=_PATH_INK, label="$t{=}1$"),
    ], loc="outside lower center", ncol=5, frameon=False, fontsize=7,
        handletextpad=0.3, columnspacing=1.5)
    return _save(fig, out_dir, f"{stem}_{point}")


def _save(fig, out_dir: str, stem: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, stem)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
