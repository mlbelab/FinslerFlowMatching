"""The diagnostic dataset figure for the route-selection benchmark.

This is the *diagnostic* renderer, with its own palette and its own region drawing; the
paper's submission figures come from :mod:`scripts.experiments.sheet.paper_figures`.  The two are kept
apart on purpose — this one is allowed to be busy and annotated, because its job is to
let a reader check that the cloud is what it claims to be.

Four panels, answering four questions in order:

    1. what is the cloud, and which cells belong to which region?
    2. what does the flow (i.e. P) actually say?
    3. what does the model *see* -- cloud and P-graph, gap removed?
    4. what is the shortcut it must be prevented from taking?

Panel 3 draws the edges of ``P_train`` rather than just its points, because "the gap
is withheld" is a statement about the graph, not about the scatter: if a k-NN edge
still bridged the hole the deletion would be cosmetic.

Run:  python -m scripts.experiments.sheet.plots
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from scripts.core.paths import experiment_root, rel

from .datasets import (
    DATASET_NAMES,
    DECOY,
    GAP,
    REGION_COLOURS,
    REGION_LABELS,
    SHEET_CORRIDOR,
    SHEET_GAP,
    SHEET_L,
    SHEET_W,
    SOURCE,
    TARGET,
    RouteDataset,
    _sheet_z,
    make_dataset,
)

DEFAULT_OUT = os.path.join(experiment_root("route_bench_paper"), "diagnostic")
QUIVER_N = 130
EDGE_MAX = 4000


def _save(fig, stem: str) -> None:
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {rel(stem)}.pdf / .png")


def _thin(n: int, k: int, seed: int = 0) -> np.ndarray:
    """``k`` indices out of ``n``, or all of them if there are fewer."""
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, k, replace=False)


def _regions(ax, ds: RouteDataset, cols=(0, 1), size: float = 4.0,
             only: tuple[int, ...] | None = None, alpha: float = 0.85,
             ghost: tuple[int, ...] = ()) -> None:
    """Scatter coloured by region; ``ghost`` regions are drawn hollow and faint."""
    order = (SOURCE, TARGET, GAP, DECOY)
    for r in order:
        m = ds.region_mask(r)
        if not m.any() or (only is not None and r not in only and r not in ghost):
            continue
        pts = ds.X[m][:, cols]
        if r in ghost:
            ax.scatter(*pts.T, s=size + 3, facecolors="none",
                       edgecolors=REGION_COLOURS[r], linewidths=0.35, alpha=0.45,
                       label=f"{REGION_LABELS[r]} — withheld")
        else:
            ax.scatter(*pts.T, s=size, c=REGION_COLOURS[r], lw=0, alpha=alpha,
                       label=REGION_LABELS[r])


def _field(ax, ds: RouteDataset, cols=(0, 1), n: int = QUIVER_N, scale: float = 22.0,
           seed: int = 0, colour_by_region: bool = True) -> None:
    """A thinned quiver of the generator tangent — the direction P is tilted towards."""
    idx = _thin(len(ds.X), n, seed)
    pts, vec = ds.X[idx][:, cols], ds._v_gen[idx][:, cols]
    c = ([REGION_COLOURS[r] for r in ds.region[idx]] if colour_by_region else "#222222")
    ax.quiver(pts[:, 0], pts[:, 1], vec[:, 0], vec[:, 1], color=c,
              width=0.0045 if colour_by_region else 0.0026,
              scale=scale, alpha=0.9 if colour_by_region else 0.6)


def _train_graph(ax, ds: RouteDataset, cols=(0, 1), seed: int = 0) -> None:
    """The training cloud with the *edges* of P_train, plus the hole left behind."""
    Xt = ds.X[ds.train_mask]
    P = ds.P_train.tocoo()
    keep = P.row != P.col
    r, c = P.row[keep], P.col[keep]
    sel = _thin(len(r), EDGE_MAX, seed)
    seg = np.stack([Xt[r[sel]][:, cols], Xt[c[sel]][:, cols]], axis=1)
    ax.add_collection(LineCollection(seg, colors="0.40", linewidths=0.3, alpha=0.6,
                                     zorder=0))
    for reg in (SOURCE, TARGET):
        m = ds.region_mask(reg)
        ax.scatter(*ds.X[m][:, cols].T, s=4, c=REGION_COLOURS[reg], lw=0, zorder=2)
    for reg in (GAP, DECOY):
        m = ds.region_mask(reg)
        if m.any():
            ax.scatter(*ds.X[m][:, cols].T, s=6, facecolors="none",
                       edgecolors=REGION_COLOURS[reg], linewidths=0.3, alpha=0.35,
                       zorder=1)


def _legend(ax, **kw) -> None:
    ax.legend(fontsize=7, frameon=False, markerscale=2.2,
              loc=kw.pop("loc", "upper right"), **kw)


def _clean(ax, equal: bool = True) -> None:
    if equal:
        ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _count_bridging_edges(ds: RouteDataset) -> int:
    """How many P_train edges join p_0 directly to p_1 — the hole's integrity check."""
    reg = ds.region[ds.train_mask]
    P = ds.P_train.tocoo()
    off = P.row != P.col
    return int((reg[P.row[off]] != reg[P.col[off]]).sum())


# --------------------------------------------------------------------------- #
#  2. Sheet
# --------------------------------------------------------------------------- #
def figure_sheet(ds: RouteDataset, out_dir: str) -> None:
    # 2 x 2 rather than 1 x 4: the domain is 3.0 x 0.9, so every panel needs a wide,
    # short cell to be drawn at *true* aspect.  Squeezing it into a square panel would
    # stretch the crossing 3.3x and make it look long -- the opposite of the point.
    fig = plt.figure(figsize=(15.0, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.07,
                          left=0.03, right=0.985, top=0.87, bottom=0.06)
    u, w = ds.uv[:, 0], ds.uv[:, 1]

    # -- 1. the sheet in R^3 --------------------------------------------------- #
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    gu, gw = np.meshgrid(np.linspace(-SHEET_L, SHEET_L, 26),
                         np.linspace(-SHEET_W, SHEET_W, 10))
    ax0.plot_wireframe(gu, gw, _sheet_z(gu, gw), color="0.62", lw=0.55, alpha=0.7)
    for r in (SOURCE, TARGET, GAP):
        m = ds.region_mask(r)
        ax0.scatter(*ds.X[m].T, s=3, c=REGION_COLOURS[r], lw=0, alpha=0.75,
                    label=REGION_LABELS[r])
    # true relative extents, so the "short axis" reads as short rather than as a
    # square panel: the box is (2L, 2W, z-range) = (3.0, 0.9, 0.75)
    ax0.set_box_aspect((2 * SHEET_L, 2 * SHEET_W,
                        float(_sheet_z(SHEET_L, 0.0) - _sheet_z(0.0, SHEET_W))),
                       zoom=1.95)   # a flat box_aspect leaves mplot3d a lot of slack
    ax0.view_init(elev=24, azim=-66)
    # the frame adds nothing here (every tick is suppressed anyway) and at this zoom
    # its spines sprawl across the cell, so drop it and keep wireframe + cloud
    ax0.set_axis_off()
    ax0.text2D(0.5, 0.03, "$x = u$ (long axis),  $y = w$ (the crossing),  "
                          "$z$ = saddle height", transform=ax0.transAxes,
               ha="center", fontsize=7.5, color="0.35")
    ax0.legend(fontsize=8, frameon=False, markerscale=2.5,
               loc="upper left", bbox_to_anchor=(-0.02, 0.32))
    ax0.set_title("Sheet: a saddle in $\\mathbb{R}^3$ — $p_0$ / $p_1$ face each other "
                  "across the $\\bf{short}$ axis", fontsize=9.5, y=0.99)

    # -- 2. parameter space + the corridor field ------------------------------- #
    ax1 = fig.add_subplot(gs[0, 1])
    idx = _thin(len(u), 300, 0)
    du = ds._v_gen[:, 0]                      # dX/du component is (1, 0, z_u)
    dw = ds._v_gen[:, 1]                      # dX/dw component is (0, 1, z_w)
    ax1.axhspan(-SHEET_GAP, SHEET_GAP, color=REGION_COLOURS[GAP], alpha=0.10, lw=0)
    ax1.axvspan(-SHEET_CORRIDOR, SHEET_CORRIDOR, color="0.5", alpha=0.10, lw=0)
    ax1.quiver(u[idx], w[idx], du[idx], dw[idx],
               color=[REGION_COLOURS[r] for r in ds.region[idx]],
               width=0.0035, scale=34, alpha=0.9)
    box = dict(fc="white", ec="none", alpha=0.8, pad=1.2)
    ax1.text(0.0, SHEET_W * 0.94, "corridor", ha="center", va="top", fontsize=8,
             color="0.25", bbox=box)
    ax1.text(-SHEET_L * 0.60, -SHEET_W * 0.72,
             "on the flanks the field is ∥ the fronts:\nmass must move sideways first",
             ha="center", fontsize=8, color="0.25", bbox=box)
    ax1.text(SHEET_L * 0.58, SHEET_W * 0.72, "past the midline it fans out again",
             ha="center", fontsize=8, color="0.25", bbox=box)
    ax1.set_title("the flow in parameter space $(u, w)$ — ≈1-D: only near $u=0$ "
                  "does it point at $p_1$", fontsize=9.5)
    ax1.set_ylabel("$w$ (crossing)", fontsize=8)

    # -- 3. what the model sees, in parameter space ---------------------------- #
    ax2 = fig.add_subplot(gs[1, 0])
    Xt_uv = ds.uv[ds.train_mask]
    P = ds.P_train.tocoo()
    keep = P.row != P.col
    sel = _thin(int(keep.sum()), EDGE_MAX, 0)
    seg = np.stack([Xt_uv[P.row[keep][sel]], Xt_uv[P.col[keep][sel]]], axis=1)
    ax2.add_collection(LineCollection(seg, colors="0.55", linewidths=0.25, alpha=0.5))
    for r in (SOURCE, TARGET):
        m = ds.region_mask(r)
        ax2.scatter(ds.uv[m, 0], ds.uv[m, 1], s=3, c=REGION_COLOURS[r], lw=0)
    m = ds.region_mask(GAP)
    ax2.scatter(ds.uv[m, 0], ds.uv[m, 1], s=5, facecolors="none",
                edgecolors=REGION_COLOURS[GAP], linewidths=0.3, alpha=0.35)
    ax2.set_title(f"what the model sees ({int(ds.train_mask.sum())} cells) and its "
                  f"$P_{{\\rm train}}$ graph — edges bridging the strip: "
                  f"{_count_bridging_edges(ds)}", fontsize=9.5)
    ax2.set_ylabel("$w$ (crossing)", fontsize=8)
    ax2.set_xlabel("$u$  (long axis)", fontsize=8)

    # -- 4. the trap ----------------------------------------------------------- #
    ax3 = fig.add_subplot(gs[1, 1])
    for r in (SOURCE, TARGET, GAP):
        m = ds.region_mask(r)
        ax3.scatter(ds.uv[m, 0], ds.uv[m, 1], s=3, c=REGION_COLOURS[r], lw=0,
                    alpha=0.30)
    for u0 in np.linspace(-SHEET_L * 0.88, SHEET_L * 0.88, 9):
        ax3.plot([u0, u0], [-SHEET_W * 0.92, SHEET_W * 0.92], "k--", lw=1.0,
                 alpha=0.8)
    ax3.plot([], [], "k--", lw=1.0, label="what a metric-blind model does:\n"
                                          "cross everywhere at once")
    # the true route: in from both flanks, across at u ~ 0, out to both flanks again
    for u_start in (-SHEET_L * 0.86, SHEET_L * 0.86):
        ax3.annotate("", xy=(0.0, 0.0), xytext=(u_start, -SHEET_W * 0.84),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="#1a7f37",
                                     alpha=0.9))
        ax3.annotate("", xy=(u_start, SHEET_W * 0.84), xytext=(0.0, 0.0),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="#1a7f37",
                                     alpha=0.9))
    ax3.plot([], [], color="#1a7f37", lw=1.5,
             label="what $P$ says: an hourglass through $u\\approx0$")
    ax3.set_xlim(-SHEET_L, SHEET_L); ax3.set_ylim(-SHEET_W, SHEET_W)
    ax3.legend(fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none",
               loc="lower center", ncol=2)
    ax3.set_title("the trap: the crossing is equally $\\bf{short}$ everywhere — "
                  f"gap width {2 * SHEET_GAP:.2f} vs lateral extent "
                  f"{2 * SHEET_L:.1f}", fontsize=9.5)
    ax3.set_xlabel("$u$  (long axis)", fontsize=8)
    ax3.set_ylabel("$w$ (crossing)", fontsize=8)

    for a in (ax1, ax2, ax3):                 # true aspect: the strip must look thin
        a.set_aspect("equal")
        a.set_xlim(-SHEET_L, SHEET_L)
        a.set_ylim(-SHEET_W, SHEET_W)
        a.set_xticks([]); a.set_yticks([])

    fig.suptitle("Dataset 2 — Sheet.  A curved sheet with one lateral corridor: mass "
                 "must funnel to $u\\approx0$, cross, then fan out.", fontsize=12)
    _save(fig, os.path.join(out_dir, "sheet_dataset"))


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--datasets", nargs="*", default=list(DATASET_NAMES))
    args = p.parse_args(argv)

    fns = {"Sheet": figure_sheet}
    for name in args.datasets:
        ds = make_dataset(name)
        print(" ", ds.summary())
        fns[name](ds, args.out)


if __name__ == "__main__":
    main()
