"""Drawing primitives for the Sheet figures: the analytic surface, the cloud, trajectories and legends."""
from __future__ import annotations


import numpy as np

from matplotlib.lines import Line2D                                # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                            # noqa: E402,F401

from ffm.core.present import (                                 # noqa: E402
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
)

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

#: The paper palette is :data:`ffm.core.present.PAPER_REGIONS`, imported above and
#: re-exported here because every figure in this package -- and, before the restructure,
#: the notebook figure layer -- reached for it under this name.  It is deliberately *not*
#: :data:`scripts.sheet.datasets.REGION_COLOURS`: that dictionary styles a
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

    Drawn from :func:`scripts.sheet.datasets._sheet_z` rather than fitted to the cloud:
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
