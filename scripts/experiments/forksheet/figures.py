"""Drawing helpers for the ForkSheet: the Sheet's, with two things swapped.

The Sheet's ``paper_figures`` is written against a ``RouteDataset`` and a region mask, so
almost all of it applies here unchanged -- the palette, the scatter, the path stroking, the
legend, the 2-D styling and the axis limits.  Reusing it is the point: two clouds drawn by
two figure modules would differ in ink and weight, and a reader would have to work out
whether that meant anything.

Two things are the Sheet's and should not be:

* :func:`_surface` draws *its* saddle, from its own analytic height map and over its own
  extent.  Here the wireframe is :func:`~scripts.experiments.forksheet.datasets._fork_z`
  over the cloud's own chart box.
* :func:`_style_3d` fixes the Sheet's camera.  A fork needs one that opens the two arms
  instead of one that keeps three bands across ``w`` apart.

A third, the start-column rule, is simply gone.  The Sheet's (``_lateral_starts``,
``_spread_starts``) are not re-exported and no fork rule replaces them: the paths figure
draws its columns uniformly at random and hands the same draw to every panel, so no rule of
any kind can be suspected of flattering a method.

:data:`PANEL_ZOOM` is re-measured for the same reason the camera is: it is a property of
the projected bounding box, and this one is not the Sheet's shape.

Everything is handed back as one namespace by :func:`fork_primitives`, so a notebook holds
a single ``pf`` and does not have to know which helper came from where.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np

from .datasets import _fork_dz, _fork_z

#: one camera for every 3-D ForkSheet panel.  Higher and further round than the Sheet's:
#: the two arms separate in ``w``, which the Sheet's near-side-on view foreshortens into a
#: single band, and the saddle still reads at this elevation.
VIEW_ELEV, VIEW_AZIM = 34.0, -64.0

#: wireframe resolution over the chart box -- coarse, because the surface is scaffolding
WIRE_NU, WIRE_NW = 13, 9

#: ``zoom`` for a 3-D panel of the paths figure: it enlarges the projected box inside its
#: axes without touching the proportions.  Not the Sheet's 2.35 -- that was measured
#: against the Sheet's long thin slab, and this cloud's bounding box is twice as deep, so
#: the same zoom cuts both arm tips off.  Measured the same way, by rendering and looking
#: for ink on the border pixels of the axes rectangle: first clipping at 2.0, so 1.7 keeps
#: a margin.  It is a property of the *cell* as much as of the cloud -- a wider cell fits
#: the same box smaller, since Matplotlib scales a 3-D box by the shorter axis -- so a
#: figure with a differently shaped panel measures its own.
PANEL_ZOOM = 1.7


def fork_primitives():
    """The ForkSheet's drawing helpers, in one namespace.

    The Sheet's ``paper_figures`` is imported here rather than at module scope, and behind
    a backend fence: it calls ``matplotlib.use("Agg")`` when it loads, which silently kills
    inline plotting for the rest of a notebook session.  The active backend is saved and
    restored around the import, so calling this is safe where importing ``paper_figures``
    directly is not.
    """
    backend = matplotlib.get_backend()
    from scripts.experiments.sheet import paper_figures as _pf
    matplotlib.use(backend)

    def _surface(ax, ds, alpha: float = 0.55, lw: float = 0.4) -> None:
        """The analytic sheet the fork is drawn on, as a bare wireframe.

        Over the *cloud's* chart box rather than a fixed extent: the fork does not fill a
        rectangle, and a wireframe drawn over the full domain would put most of its ink
        where there are no cells.  A filled surface is deliberately not an option here for
        the same reason it is not on the Sheet -- Matplotlib composites it over the scatter
        regardless of depth and washes the region colours out.
        """
        assert ds.uv is not None, f"{ds.name} carries no chart coordinates"
        lo, hi = ds.uv.min(0), ds.uv.max(0)
        pad = 0.06 * (hi - lo)
        U, W = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], WIRE_NU),
                           np.linspace(lo[1] - pad[1], hi[1] + pad[1], WIRE_NW))
        ax.plot_wireframe(U, W, _fork_z(U, W), color=_pf._SURFACE_WIRE, linewidth=lw,
                          alpha=alpha, zorder=0)

    def _style_3d(ax, lo, hi, zoom: float = 1.0) -> None:
        """The Sheet's 3-D box, then the fork's camera."""
        _pf._style_3d(ax, lo, hi, zoom=zoom)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    return SimpleNamespace(
        _limits=_pf._limits,
        _style_2d=_pf._style_2d,
        _scatter_cloud=_pf._scatter_cloud,
        _draw_paths=_pf._draw_paths,
        _legend_handles=_pf._legend_handles,
        PANEL_ZOOM=PANEL_ZOOM,
        _surface=_surface,
        _style_3d=_style_3d,
        # the height map and its gradient, so a panel can lift a 2-D chart field onto the
        # surface instead of drawing it flat under one
        _fork_z=_fork_z,
        _fork_dz=_fork_dz,
        VIEW_ELEV=VIEW_ELEV,
        VIEW_AZIM=VIEW_AZIM,
    )
