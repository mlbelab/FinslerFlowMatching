"""Submission figures for the Sheet: the cloud itself, and transported trajectories.

Three artefacts, all under ``<root>/paper/figures/``:

``Sheet_overview``
    The cloud in $\\mathbb{R}^3$ next to its $(u, w)$ parameter plane, next to the
    reference flow $b_0$.  The 3-D panel is the honest picture -- the sheet is a saddle,
    so a straight crossing in $w$ leaves it -- and the 2-D panel is the one every later
    figure is drawn in, so showing them side by side is what licenses the projection.
    The third panel is why the corridor exists: outside $|u| \\le 0.28$ the field runs
    *along* the fronts, not across them.

``Sheet_paths_main`` / ``Sheet_paths_appendix``
    Three transported trajectories per method, each method shown twice -- in
    $\\mathbb{R}^3$ (top row) and in the plane (bottom row).  Three, not five: a 3-D
    panel this small cannot carry more without the curves reading as a single ribbon.
    Columns are grouped into a deterministic and a stochastic block, matching the two
    halves of the main table (:mod:`scripts.experiments.sheet.paper_focus`).  ``main`` carries the
    five compared methods, ``appendix`` the geometry ablation.

Every panel is drawn from the *same* three start cells -- spread across the sheet's
long axis by a rule that looks only at $t = 0$ (:func:`_lateral_starts`) -- so a
difference between panels is a difference between methods and not between the cells
they happened to be handed.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                            # noqa: E402,F401

from scripts.core.present import (                                 # noqa: E402
    FAINT,
    INK,
    PAPER_REGIONS,
)

from .datasets import (                                            # noqa: E402
    DECOY,
    GAP,
    SHEET_L,
    SHEET_W,
    SOURCE,
    TARGET,
    _sheet_z,
    make_dataset,
)
from .paper_focus import (                                         # noqa: E402
    ABLATION_BLOCKS,
    ABLATION_LABELS_SHORT,
    MAIN_BLOCKS,
    MAIN_HEADINGS,
)
from .paper_report import _save                                     # noqa: E402
from .paper_run import DEFAULT_OUT                                 # noqa: E402
from .protocol import corridor_mask                                # noqa: E402

#: paths per panel.  Fewer than the 2-D-only figures use: a 3-D panel of this size
#: turns five curves into one ribbon, and the point of the panel is that they differ.
N_TRAJ_PANEL = 3

#: one camera for every 3-D panel in the paper.  Chosen to show the saddle curvature and
#: to keep the three bands across $w$ separated, without foreshortening the long axis.
VIEW_ELEV, VIEW_AZIM = 30.0, -50.0

#: aliases, so the private spellings this module has always used keep working
_INK, _FAINT = INK, FAINT

#: ``zoom`` for a 3-D panel: it enlarges the projected box inside its axes, discarding
#: the empty corners of the bounding cube without touching the proportions.  The value
#: was measured, not guessed -- rendering the panel and looking for ink on the border
#: pixels of the axes rectangle puts the first clipping at ``2.6`` for both cell shapes,
#: so ``2.35`` keeps a margin.  Judging this by eye is unreliable here: the sheet's far
#: short edge projects to a near-vertical line and reads as a crop when it is not.
PANEL_ZOOM = OVERVIEW_ZOOM = 2.35

#: The paper palette is :data:`scripts.core.present.PAPER_REGIONS`, imported above and
#: re-exported here because every figure in this package -- and, before the restructure,
#: the notebook figure layer -- reached for it under this name.  It is deliberately *not*
#: :data:`scripts.experiments.sheet.datasets.REGION_COLOURS`: that dictionary styles a
#: dozen diagnostic plots whose look is not being reviewed here, and restyling them
#: silently would make every existing artefact disagree with the ones in the submission.
#: the wireframe must read as scaffolding, not as a fifth region, so it is the only
#: fully desaturated line in the figure
_SURFACE_WIRE = "#b0b0b0"

#: every trajectory is drawn in the same ink, with a white casing under it.  Per-method
#: colours would add five more hues on top of four region colours, and they would buy
#: nothing: a panel here holds exactly one method and its title says which.  Near-black
#: rather than the old blue-black: with an indigo p_1 in the same panel, a navy path
#: read as another region.  The casing is what keeps it legible over the corridor.
_PATH_INK = "#15161A"
_PATH_CASING = "#ffffff"


# --------------------------------------------------------------------------- #
#  Shared scaffolding
# --------------------------------------------------------------------------- #
def _limits(ds, pad_frac: float = 0.04):
    """Axis limits fixed to the cloud, never to the paths.

    One divergent SDE sample must not rescale a panel until the manifold is a dot, so
    the limits come from ``X`` and the paths are clipped by them.
    """
    lo, hi = ds.X.min(0), ds.X.max(0)
    pad = pad_frac * np.maximum(hi - lo, 1e-9)
    return lo - pad, hi + pad


def _style_3d(ax, lo, hi, zoom: float = 1.0) -> None:
    """A 3-D box with the chartjunk removed: no ticks, no filled panes, pale grid.

    ``zoom`` only enlarges the rendered box inside its axes -- a long thin slab seen at
    an angle leaves most of a square panel empty otherwise.  It does not touch the
    proportions, which stay true: the sheet really is long, narrow and gently curved,
    and stretching ``w`` or ``z`` to fill the panel would misrepresent how cheap the
    crossing is.
    """
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo), zoom=zoom)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    # no box, no panes, no ticks: a bounding cube around a thin slab is mostly empty
    # air, and the wireframe of the manifold is a better depth cue than a wire cube
    ax.set_axis_off()


def _style_2d(ax, lo, hi) -> None:
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(_FAINT)


def _surface(ax, alpha: float = 0.55, lw: float = 0.4) -> None:
    """The analytic saddle the Sheet is sampled from, as a bare wireframe.

    Drawn from :func:`scripts.experiments.sheet.datasets._sheet_z` rather than fitted to the cloud:
    it is the ground-truth manifold, and its job is to make the curvature legible in a
    static projection -- without it the scatter alone reads as a flat slab.

    A *filled* translucent surface was the obvious choice and is wrong here: Matplotlib
    composites it over the scatter regardless of depth, which washes the region colours
    out to pastel and destroys the one thing these panels have to show.
    """
    u = np.linspace(-SHEET_L, SHEET_L, 49)
    w = np.linspace(-SHEET_W, SHEET_W, 13)
    U, W = np.meshgrid(u, w)
    ax.plot_wireframe(U[::4, ::6], W[::4, ::6], _sheet_z(U, W)[::4, ::6],
                      color=_SURFACE_WIRE, linewidth=lw, alpha=alpha, zorder=0)


def _regions(ds, corridor):
    """``(mask, colour, is_corridor)`` in draw order, corridor last so it sits on top.

    Only the *scored* part of the withheld strip is drawn.  On the Sheet the corridor
    is a strict subset of the gap (``|u| <= SHEET_CORRIDOR``) and the unscored remainder
    is dropped
    rather than shown as a tint of the same orange.  The tint was the more literal
    picture but the worse one: a pale band abutting the corridor reads at figure scale as
    one wide orange region, i.e. as the thing the intermediate metric measures, when it
    is exactly the part that is *not* measured.  Dropping it leaves the corridor as an
    isolated island between the two marginals, which is the geometry the score is about.
    """
    assert corridor.shape == (len(ds.X),), (corridor.shape, ds.X.shape)
    out = []
    for reg in (SOURCE, TARGET, DECOY):
        m = ds.region == reg
        if m.any():
            out.append((m, PAPER_REGIONS[reg], False))
    if corridor.any():
        out.append((corridor, PAPER_REGIONS[GAP], True))
    return out


def _scatter_cloud(ax, ds, corridor, three_d: bool, s: float = 1.6,
                   alpha_scale: float = 1.0) -> None:
    for m, colour, is_corr in _regions(ds, corridor):
        # The corridor is the one thing the reader has to be able to point at, so it is
        # the only region drawn at full strength.  The marginals sit low enough to stay
        # background and high enough that the violet and the blue do not collapse into
        # one pale wash.
        if is_corr:
            base = 1.0
        else:
            base = 0.34 if colour == PAPER_REGIONS[DECOY] else 0.42
        alpha = float(np.clip(base * alpha_scale, 0.0, 1.0))
        pts = ds.X[m]
        if three_d:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s * (1.9 if is_corr else 1.0),
                       alpha=alpha, color=colour, linewidths=0, depthshade=False,
                       zorder=1)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=s * (1.7 if is_corr else 1.0),
                       alpha=alpha, color=colour, linewidths=0, zorder=1)


def _draw_paths(ax, paths: np.ndarray, three_d: bool, colour: str = _PATH_INK) -> None:
    """``paths`` is ``(steps, n_paths, d)``; start = white dot, end = white star.

    Each curve is stroked twice: a wider white casing first, then the ink.  Without the
    casing a dark curve running along the dark teal corridor -- which is exactly where a
    correct path spends its middle third -- becomes invisible at figure scale.
    """
    assert paths.ndim == 3, paths.shape
    for j in range(paths.shape[1]):
        p = paths[:, j]
        args = (p[:, 0], p[:, 1], p[:, 2]) if three_d else (p[:, 0], p[:, 1])
        ax.plot(*args, lw=3.0, alpha=0.9, color=_PATH_CASING,
                solid_capstyle="round", zorder=4)
        ax.plot(*args, lw=1.4, alpha=1.0, color=colour, solid_capstyle="round",
                zorder=5)
    ends = [(paths[0], 18, "o", 0.8), (paths[-1], 46, "*", 0.7)]
    for pts, size, marker, lw in ends:
        args = ((pts[:, 0], pts[:, 1], pts[:, 2]) if three_d
                else (pts[:, 0], pts[:, 1]))
        ax.scatter(*args, s=size, marker=marker, color=_PATH_CASING,
                   edgecolors=colour, linewidths=lw, zorder=6,
                   **({"depthshade": False} if three_d else {}))


def _legend_handles(ds, corridor, extra_paths: bool = True):
    """Bare labels: which region is shown and which is withheld belongs in the caption.

    The qualifiers these entries used to carry ("(shown)", "(withheld)", "(scored)") are
    protocol, not identity, and a legend that restates the protocol competes with the
    panels for the reader's attention.
    """
    dot = dict(marker="o", ls="", ms=5)
    handles = [
        Line2D([], [], **dot, color=PAPER_REGIONS[SOURCE], label="$p_0$"),
        Line2D([], [], **dot, color=PAPER_REGIONS[TARGET], label="$p_1$"),
    ]
    if corridor.any():
        # only the scored part of the withheld strip is drawn (see :func:`_regions`), so
        # the legend names precisely that.  Where the corridor *is* the whole gap the
        # plainer word is the accurate one, so the wording follows the cloud rather than
        # being hard-coded to the Sheet.
        whole_gap = int(corridor.sum()) == int((ds.region == GAP).sum())
        handles.append(Line2D([], [], **dot, color=PAPER_REGIONS[GAP],
                              label="gap" if whole_gap else "corridor"))
    if (ds.region == DECOY).any():
        handles.append(Line2D([], [], **dot, color=PAPER_REGIONS[DECOY],
                              label="decoy"))
    if extra_paths:
        handles += [
            Line2D([], [], marker="o", ls="", ms=5, color="#ffffff",
                   markeredgecolor=_INK, label="$t{=}0$"),
            Line2D([], [], marker="*", ls="", ms=8, color="#ffffff",
                   markeredgecolor=_INK, label="$t{=}1$"),
        ]
    return handles


def _lateral_starts(x0: np.ndarray, n: int) -> np.ndarray:
    """``n`` start columns spread across the sheet's *long* axis ``u``.

    Not a farthest-point draw over $p_0$: on the Sheet the only interesting variation is
    in $u$.  The reference field funnels a cell at $|u| \\sim 1$ inward until it reaches
    the corridor and only then lets it cross, while a cell already at $u \\sim 0$ crosses
    directly.  A draw spread in $\\mathbb{R}^3$ lands mostly at the lateral extremes and
    shows the same funnel three times; spreading in $u$ shows one funnel from each side
    plus one direct crossing, which is precisely what the corridor metric scores.

    The rule looks only at $t = 0$, so it cannot favour a method, and the columns it
    returns are handed to *every* panel of the figure.
    """
    assert x0.ndim == 2 and x0.shape[0] >= n, (x0.shape, n)
    u = x0[:, 0]
    q = np.linspace(0.10, 0.90, n) if n > 1 else np.array([0.5])
    picked: list[int] = []
    for target in np.quantile(u, q):
        order = np.argsort(np.abs(u - target))
        picked.append(next(int(k) for k in order if int(k) not in picked))
    return np.array(picked, dtype=int)


# --------------------------------------------------------------------------- #
#  Figure 1 — the cloud
# --------------------------------------------------------------------------- #
def overview_figure(ds, out_stem: str, n_arrows: int = 240, seed: int = 0) -> None:
    """3-D cloud, the same cloud in the plane, and the reference flow that defines the route."""
    assert ds.d == 3, f"the overview is written for the 3-D Sheet, got d={ds.d}"
    corridor = corridor_mask(ds)
    lo, hi = _limits(ds)

    # Matplotlib fits a 3-D cube into its axes by the *smaller* of the two axes
    # dimensions, so a wide short cell renders the sheet small no matter how wide the
    # figure is.  The 3-D panel therefore gets a tall cell -- the full height of the
    # figure -- and the two plane panels stack beside it.  Its width share is set to the
    # projected content's own aspect (~1.1:1, measured): a wider cell cannot make the
    # sheet any bigger, it only adds white space on both sides of it.
    fig = plt.figure(figsize=(10.2, 3.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 1.0], hspace=0.02)

    ax0 = fig.add_subplot(gs[:, 0], projection="3d")
    _surface(ax0)
    _scatter_cloud(ax0, ds, corridor, three_d=True, s=2.6, alpha_scale=1.45)
    _style_3d(ax0, lo, hi, zoom=OVERVIEW_ZOOM)
    ax0.set_title(r"(a) $\mathcal{M}\subset\mathbb{R}^3$", fontsize=10.5, pad=-4)

    ax1 = fig.add_subplot(gs[0, 1])
    _scatter_cloud(ax1, ds, corridor, three_d=False, s=2.0)
    _style_2d(ax1, lo, hi)
    ax1.set_title(r"(b) $(u,w)$ plane", fontsize=10.5, pad=3)

    # (c) the generator field, thinned so the arrows stay readable.  This is ground
    # truth (never a model input) and it is the whole reason the corridor is the route:
    # away from u = 0 the field runs along the fronts, not across them.
    ax2 = fig.add_subplot(gs[1, 1])
    _scatter_cloud(ax2, ds, corridor, three_d=False, s=1.2)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ds.X), size=min(n_arrows, len(ds.X)), replace=False)
    v = ds._v_gen[pick, :2]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    ax2.quiver(ds.X[pick, 0], ds.X[pick, 1], v[:, 0], v[:, 1],
               color="#3a3a3a", alpha=0.8, width=0.0035, headwidth=3.6,
               scale=34, zorder=4)
    _style_2d(ax2, lo, hi)
    ax2.set_title(r"(c) reference flow $b_0$", fontsize=10.5, pad=3)
    ax2.set_xlabel("$u$", fontsize=9.5, labelpad=1)
    for ax in (ax1, ax2):
        ax.set_ylabel("$w$", fontsize=9.5, labelpad=1)

    fig.legend(handles=_legend_handles(ds, corridor, extra_paths=False),
               loc="outside lower center", ncol=4, frameon=False, fontsize=9.5,
               handletextpad=0.3, columnspacing=1.6)
    _save(fig, out_stem)


# --------------------------------------------------------------------------- #
#  Figure 2/3 — trajectories, 3-D over 2-D, grouped by noise
# --------------------------------------------------------------------------- #
def blocked_paths_figure(ds, trajs: dict[str, np.ndarray], blocks, labels: dict,
                         out_stem: str, n_paths: int = N_TRAJ_PANEL) -> None:
    """Three rows (heading, 3-D, plane) x one column per arm, grouped into noise blocks.

    The block heading lives in a gridspec row of its own rather than in a
    ``suptitle``/``subfigures`` arrangement.  A ``suptitle`` is placed relative to the
    *subfigure*, not to the axes inside it, so with a 3-D panel -- whose drawn content
    sits well inside its bounding box -- it lands on top of the per-method titles.  A
    real row cannot overlap anything, and it also lets the group rule span exactly the
    columns of its block.
    """
    blocks = [(h, [a for a in g if a in trajs]) for h, g in blocks]
    blocks = [(h, g) for h, g in blocks if g]
    assert blocks, "none of the requested arms have trajectories"
    corridor = corridor_mask(ds)
    lo, hi = _limits(ds)

    # one draw of start cells, reused in every panel of every block
    n_src = min(trajs[a].shape[1] for _h, g in blocks for a in g)
    cols = _lateral_starts(trajs[blocks[0][1][0]][0, :n_src], min(n_paths, n_src))

    # a narrow spacer column between blocks: real space, so the two halves read as two
    # halves even before the reader gets to the headings
    n_panels = sum(len(g) for _h, g in blocks)
    spacer = 0.22
    widths, spans, col = [], [], 0
    for i, (_h, arms) in enumerate(blocks):
        if i:
            widths.append(spacer)
            col += 1
        spans.append((col, col + len(arms)))
        widths += [1.0] * len(arms)
        col += len(arms)

    # the plane panel is drawn at equal aspect, so its *content* is only
    # (w-extent / u-extent) as tall as a column is wide -- roughly a third.  Giving that
    # row a square-ish share would centre a thin strip in a tall empty box, which is
    # where the dead band between the two rows came from; the ratio below is the
    # measured extent ratio plus a little slack.
    plane_ratio = float((hi[1] - lo[1]) / (hi[0] - lo[0])) + 0.14
    fig = plt.figure(figsize=(2.05 * n_panels + spacer * (len(blocks) - 1) + 0.5, 3.05),
                     layout="constrained")
    gs = fig.add_gridspec(3, len(widths), width_ratios=widths,
                          height_ratios=[0.13, 0.84, plane_ratio],
                          hspace=0.04, wspace=0.06)

    for (heading, arms), (c0, c1) in zip(blocks, spans):
        axh = fig.add_subplot(gs[0, c0:c1])
        axh.set_axis_off()
        axh.text(0.5, 0.30, heading, ha="center", va="bottom", fontsize=10.5,
                 color=_INK, transform=axh.transAxes)
        # a rule under the heading spanning only its own columns -- the figure's
        # \cmidrule, so the grouping is visible without reading the words
        axh.plot([0.02, 0.98], [0.14, 0.14], transform=axh.transAxes, color=_FAINT,
                 lw=0.9, clip_on=False)

        for k, arm in enumerate(arms):
            traj = trajs[arm][:, cols]

            ax3 = fig.add_subplot(gs[1, c0 + k], projection="3d")
            _surface(ax3, alpha=0.5, lw=0.35)
            _scatter_cloud(ax3, ds, corridor, three_d=True, s=1.5, alpha_scale=1.35)
            _draw_paths(ax3, traj, three_d=True)
            _style_3d(ax3, lo, hi, zoom=PANEL_ZOOM)
            # A column heading may be stacked ("FFM" over "deterministic").  Matplotlib
            # anchors a title by its *bottom*, so extra lines grow upward and would climb
            # into the group-heading row; each one is paid for by dropping the anchor a
            # line further into the axes, which is empty air here (the box is off and the
            # projected sheet sits well inside it).
            title = labels.get(arm, arm)
            ax3.set_title(title, fontsize=9.5, linespacing=1.0,
                          pad=-6 - 9.5 * title.count("\n"))

            ax2 = fig.add_subplot(gs[2, c0 + k])
            _scatter_cloud(ax2, ds, corridor, three_d=False, s=1.4)
            _draw_paths(ax2, traj, three_d=False)
            _style_2d(ax2, lo, hi)

    fig.legend(handles=_legend_handles(ds, corridor), loc="outside lower center",
               ncol=7, frameon=False, fontsize=9, handletextpad=0.3,
               columnspacing=1.5)
    _save(fig, out_stem)


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def load_trajectories(root: str, cloud: str, seed: int) -> dict:
    out = {}
    for path in sorted(os.listdir(os.path.join(root, "trajectories"))):
        stem = path[:-4]
        if not path.endswith(".npz") or not stem.endswith(f"_s{seed}"):
            continue
        name, _, rest = stem.partition("_")
        if name != cloud:
            continue
        arm = rest.rsplit(f"_s{seed}", 1)[0]
        out[arm] = np.load(os.path.join(root, "trajectories", path))["traj"]
    return out


def build_figures(root: str = DEFAULT_OUT,
                  cloud: str = "Sheet", fig_seed: int = 0,
                  n_paths: int = N_TRAJ_PANEL) -> None:
    ds = make_dataset(cloud)
    figs = os.path.join(root, "paper", "figures")
    os.makedirs(figs, exist_ok=True)

    overview_figure(ds, os.path.join(figs, f"{cloud}_overview"))

    trajs = load_trajectories(root, cloud, fig_seed)
    if not trajs:
        print(f"  [skip] no {cloud} trajectories at seed {fig_seed}")
        return
    # MAIN_HEADINGS, not MAIN_LABELS: in a figure the two FFM columns carry the shared
    # name on one line and their qualifier on the next, which is where the pair reads as
    # one method.  The table form of the same rows keeps the qualifier inline.
    blocked_paths_figure(ds, trajs, MAIN_BLOCKS, MAIN_HEADINGS,
                         os.path.join(figs, f"{cloud}_paths_main"), n_paths=n_paths)
    blocked_paths_figure(ds, trajs, ABLATION_BLOCKS, ABLATION_LABELS_SHORT,
                         os.path.join(figs, f"{cloud}_paths_appendix"),
                         n_paths=n_paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_OUT)
    ap.add_argument("--cloud", default="Sheet")
    ap.add_argument("--fig-seed", type=int, default=0)
    ap.add_argument("--n-paths", type=int, default=N_TRAJ_PANEL)
    args = ap.parse_args()
    build_figures(args.root, args.cloud, args.fig_seed, args.n_paths)


if __name__ == "__main__":
    main()
