"""Figure primitives: the palette, the block grid, and the flux quiver.

Three things live here, and nothing task-specific does — a notebook draws its own panels
and only borrows the layout.

**The palette is the paper's, imported and not restated.**  ``PAPER_REGIONS`` and the two
greys come from :mod:`scripts.core.present`, which is also what the paper's own figure
code reads, so a notebook figure and the paper's figure are the same ink at the same
strength.  Nothing task-specific is imported at module scope: this package may import an
experiment, but doing so eagerly would drag one benchmark's figure module into every
notebook.  :func:`sheet_primitives` is the one place that reaches for one, and it does so
behind a fence — ``paper_figures`` calls ``matplotlib.use("Agg")`` at module scope, which
in a notebook kills every subsequent inline plot, so the backend is saved and restored
around the import.

**The block grid** is ``blocked_paths_figure``'s geometry: columns gathered into headed
blocks, a real spacer between blocks, and a rule spanning only its own columns.  It is
generic in the number of content rows, because the Sheet wants two (a 3-D view and its
2-D chart) and a PCA cloud wants one.

**The flux quiver** draws ``b_0`` the way the method actually reads it: the *P-weighted*
first moment ``sum_j P_ij (y_j - y_i)`` in whatever 2-D chart ``y`` is, averaged on a
coarse grid.  Both halves of that matter.  An unweighted neighbour centroid is the
boundary drift of the slab rather than the dynamics, and a per-cell arrow is noise —
binning is what the method's own kernel smoothing does, and it is the only way the
structure is visible by eye.  :func:`unit_quiver` is the same field at the paper's
weight — one fixed arrow length, centred on its bin — for panels that sit beside a Sheet
figure and should not disagree with it about what an arrow means.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#: the palette, and the two greys every panel is built on: unremarkable cells, and
#: everything drawn on top
from scripts.core.present import FAINT, INK, PAPER_REGIONS

#: gap between two headed blocks, in units of one panel width
SPACER = 0.22

__all__ = ["PAPER_REGIONS", "FAINT", "INK", "SPACER", "BlockGrid", "VBlockGrid",
           "bin_flux", "flux_quiver", "unit_quiver", "sheet_primitives"]


def sheet_primitives():
    """The Sheet's own drawing helpers, imported lazily and behind the backend fence.

    Returned as the module rather than re-exported name by name: they are private to
    ``paper_figures`` and a notebook that reaches for them is knowingly reaching into the
    paper's figure code.  The import is deferred to here because it is the only thing in
    this package that touches an experiment, and because ``paper_figures`` switches
    matplotlib to ``Agg`` at module scope — which would kill inline plotting in any
    notebook that merely imported this module.  The active backend is restored on the way
    out, so calling this is safe; importing ``paper_figures`` yourself is not.
    """
    backend = matplotlib.get_backend()
    from scripts.experiments.sheet import paper_figures as _pf
    matplotlib.use(backend)
    return _pf


class _PanelGrid:
    """What the two grid orientations share: the figure, its legend and its save."""

    def legend(self, handles, fontsize: int = 5, ncol: int | None = None):
        self.fig.legend(handles=handles, loc="outside lower center",
                        ncol=len(handles) if ncol is None else ncol,
                        frameon=False, fontsize=fontsize, handletextpad=0.3,
                        columnspacing=1.5)
        return self.fig

    def save(self, stem, exts=("png", "pdf"), dpi: int = 300):
        """Write ``stem.ext`` for each extension; returns the paths written."""
        stem.parent.mkdir(parents=True, exist_ok=True)
        out = [stem.with_suffix(f".{e}") for e in exts]
        for p in out:
            self.fig.savefig(p, dpi=dpi, bbox_inches="tight")
        return out


class BlockGrid(_PanelGrid):
    """Headed blocks of columns, ``n`` content rows each, with a spacer between blocks.

    ``blocks = [(heading, [payload, ...]), ...]``.  Row 0 is the headings and is drawn in
    the constructor; content rows are numbered from 1 and are the caller's, reached with
    :meth:`ax`.  Nothing here knows what a payload is.
    """

    def __init__(self, blocks, row_ratios=(1.0,), panel_w: float = 2.05,
                 height: float = 3.05, heading_h: float = 0.13,
                 spacer: float = SPACER, heading_fs: int = 8, pad: float = 0.5,
                 hspace: float = 0.04, wspace: float = 0.06):
        self.blocks = list(blocks)
        n_panels = sum(len(cols) for _h, cols in self.blocks)
        widths, self.spans, col = [], [], 0
        for i, (_h, cols) in enumerate(self.blocks):
            if i:                                    # a real column, so the gap is real
                widths.append(spacer)
                col += 1
            self.spans.append((col, col + len(cols)))
            widths += [1.0] * len(cols)
            col += len(cols)

        self.fig = plt.figure(
            figsize=(panel_w * n_panels + spacer * (len(self.blocks) - 1) + pad, height),
            layout="constrained")
        self.gs = self.fig.add_gridspec(1 + len(row_ratios), len(widths),
                                        width_ratios=widths,
                                        height_ratios=[heading_h, *row_ratios],
                                        hspace=hspace, wspace=wspace)
        for (heading, _cols), (c0, c1) in zip(self.blocks, self.spans):
            axh = self.fig.add_subplot(self.gs[0, c0:c1])
            axh.set_axis_off()
            axh.text(0.5, 0.30, heading, ha="center", va="bottom", fontsize=heading_fs,
                     color=INK, transform=axh.transAxes)
            axh.plot([0.02, 0.98], [0.14, 0.14], transform=axh.transAxes, color=FAINT,
                     lw=0.9, clip_on=False)

    def ax(self, row: int, block: int, k: int, **kw):
        """The axes at content ``row`` (1-based), column ``k`` of ``block``."""
        return self.fig.add_subplot(self.gs[row, self.spans[block][0] + k], **kw)


class VBlockGrid(_PanelGrid):
    """:class:`BlockGrid` turned on its side: one column of panels, headings at the left.

    Same contract — ``blocks = [(heading, [payload, ...]), ...]`` — with rows and columns
    exchanged.  Column 0 carries the block headings, rotated, each with a rule spanning
    only its own rows; content columns are numbered from 1 and are reached with
    :meth:`ax`.  A stack rather than a strip is what a one-column page wants: six panels
    side by side are two inches wide each, six panels down the page are full width.
    """

    def __init__(self, blocks, col_ratios=(1.0,), panel_h: float = 2.05,
                 width: float = 3.05, heading_w: float = 0.13,
                 spacer: float = SPACER, heading_fs: int = 8, pad: float = 0.5,
                 hspace: float = 0.04, wspace: float = 0.06):
        self.blocks = list(blocks)
        n_panels = sum(len(rows) for _h, rows in self.blocks)
        heights, self.spans, row = [], [], 0
        for i, (_h, rows) in enumerate(self.blocks):
            if i:                                    # a real row, so the gap is real
                heights.append(spacer)
                row += 1
            self.spans.append((row, row + len(rows)))
            heights += [1.0] * len(rows)
            row += len(rows)

        self.fig = plt.figure(
            figsize=(width,
                     panel_h * n_panels + spacer * (len(self.blocks) - 1) + pad),
            layout="constrained")
        self.gs = self.fig.add_gridspec(len(heights), 1 + len(col_ratios),
                                        height_ratios=heights,
                                        width_ratios=[heading_w, *col_ratios],
                                        hspace=hspace, wspace=wspace)
        for (heading, _rows), (r0, r1) in zip(self.blocks, self.spans):
            axh = self.fig.add_subplot(self.gs[r0:r1, 0])
            axh.set_axis_off()
            axh.text(0.30, 0.5, heading, ha="center", va="center", rotation=90,
                     fontsize=heading_fs, color=INK, transform=axh.transAxes)
            axh.plot([0.86, 0.86], [0.02, 0.98], transform=axh.transAxes, color=FAINT,
                     lw=0.9, clip_on=False)

    def ax(self, col: int, block: int, k: int, **kw):
        """The axes at content ``col`` (1-based), row ``k`` of ``block``."""
        return self.fig.add_subplot(self.gs[self.spans[block][0] + k, col], **kw)


def bin_flux(Y, J, nu: int = 20, nw: int = 10, min_count: int = 4):
    """Average ``J`` on an ``nu x nw`` grid over ``Y`` -> ``(m, 4)`` of ``(y0, y1, j0, j1)``.

    Bins holding fewer than ``min_count`` cells are dropped rather than drawn: a one-cell
    arrow is a single noisy displacement wearing the same ink as a mean over forty.
    """
    Y, J = np.asarray(Y, dtype=np.float64), np.asarray(J, dtype=np.float64)
    edge = lambda a, n: np.linspace(a.min(), a.max(), n + 1)[1:-1]
    b0 = np.clip(np.digitize(Y[:, 0], edge(Y[:, 0], nu)), 0, nu - 1)
    b1 = np.clip(np.digitize(Y[:, 1], edge(Y[:, 1], nw)), 0, nw - 1)
    key = b0 * nw + b1
    rows = [[Y[m, 0].mean(), Y[m, 1].mean(), *J[m].mean(0)]
            for k in np.unique(key) if (m := key == k).sum() >= min_count]
    return np.asarray(rows, dtype=np.float64)


def flux_quiver(ax, Y, J, nu: int = 20, nw: int = 10, min_count: int = 4, **kw):
    """:func:`bin_flux` straight onto ``ax``.

    ``angles="xy"`` because the two chart axes rarely span the same range, and the default
    (angles in *display* space) would then draw a field that is not the one computed.
    """
    g = bin_flux(Y, J, nu, nw, min_count)
    opts = dict(color=INK, width=0.005, headwidth=4.0, alpha=0.9, angles="xy")
    return ax.quiver(g[:, 0], g[:, 1], g[:, 2], g[:, 3], **{**opts, **kw})


#: the Sheet's arrow ink and weight, restated here so a 2-D chart and the paper's 3-D
#: overview draw the same arrow.  Deliberately lighter than :data:`INK`: these arrows are
#: an input the reader is asked to look *past*, not a result.
ARROW_GREY = "#3a3a3a"


def unit_quiver(ax, Y, J, nu: int = 20, nw: int = 10, min_count: int = 4,
                length: float = 0.05, **kw):
    """:func:`bin_flux`, then every arrow at one fixed length and centred on its bin.

    The Sheet's treatment (``paper_figures`` cell of ``sheet_ffm_paper.ipynb``), in the
    plane.  Two departures from :func:`flux_quiver`, and both are claims about what the
    panel says.  **One length**, ``length`` times the larger axis span: the binned first
    moment has a magnitude, but that magnitude is a property of how far apart cells are
    where the arrow sits, so drawing it would read as a speed the figure is not claiming.
    **Centred** rather than rooted at the bin, so a row of arrows reads as one line
    through the cloud instead of a comb hanging off it.
    """
    g = bin_flux(Y, J, nu, nw, min_count)
    assert len(g), "no bin held enough cells to average; lower nu/nw or min_count"
    Y = np.asarray(Y, dtype=np.float64)
    span = float(np.max(Y.max(axis=0) - Y.min(axis=0)))
    vec = g[:, 2:] * (length * span / (np.linalg.norm(g[:, 2:], axis=1, keepdims=True)
                                       + 1e-12))
    opts = dict(color=ARROW_GREY, width=0.004, headwidth=4.0, headlength=4.5, alpha=0.9,
                angles="xy", scale_units="xy", scale=1.0)
    tail = g[:, :2] - vec / 2
    return ax.quiver(tail[:, 0], tail[:, 1], vec[:, 0], vec[:, 1], **{**opts, **kw})
