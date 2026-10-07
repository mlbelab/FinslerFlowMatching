"""The Sheet's own table shapes, on top of the shared primitives.

This module used to be a driver as well -- it loaded a flat ``runs/*.json`` tree and
emitted four tables plus a trajectory figure.  :mod:`scripts.experiments.sheet.paper_report` and
:mod:`scripts.experiments.sheet.paper_focus` now own the reporting, and they carry their own
versions of the tables that changed for the submission.

The generic machinery -- ``_index`` / ``_dig`` / ``_agg`` / ``_fmt`` and the markdown and
LaTeX writers -- moved to :mod:`scripts.core.present` when the notebook figure layer
started needing it too, and is re-exported here so that ``from .report import _fmt`` keeps
meaning what it always did.  What is genuinely local is Sheet-shaped:

    _grid       a W2 | MMD block per cloud, plus the raw means for the winner bolding
    _plane      the 2-D view this cloud is drawn in

Cells are averaged over seeds; the spread is the sample standard deviation across
seeds, and is printed as ``+-`` only when more than one seed is present.
"""
from __future__ import annotations

import numpy as np

from scripts.core.present import (      # noqa: F401
    _agg,
    _best_mask,
    _blocked,
    _dig,
    _fmt,
    _index,
    _md_table,
    _tex_table,
)

from .datasets import DATASET_NAMES

#: table column order.  One entry today; kept as a named constant so a second cloud
#: cannot silently reorder every table by landing wherever the registry puts it.
TABLE_ORDER = DATASET_NAMES


# --------------------------------------------------------------------------- #
#  Tables
# --------------------------------------------------------------------------- #
def _grid(idx, datasets, arms, path_builder, prec=3):
    """Rows of formatted strings plus the raw means, for bolding the winner."""
    rows, raw = [], []
    for arm in arms:
        row, rraw = [], []
        for ds in datasets:
            cells = idx.get((ds, arm), [])
            if not cells:
                row += ["--", "--"]
                rraw += [np.nan, np.nan]
                continue
            for key in ("W2", "MMD"):
                m, s, n = _agg(cells, *path_builder(key))
                row.append(_fmt(m, s, n, prec))
                rraw.append(m)
        rows.append(row)
        raw.append(rraw)
    return rows, np.array(raw, dtype=float)


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _plane(ds, X: np.ndarray) -> np.ndarray:
    """The 2-D view a cloud is drawn in.

    The Sheet's ambient coordinates are literally ``(u, w, z)``, so dropping ``z`` *is*
    the parameter plane -- and it is the plane the corridor and the crossing live in.
    A cloud that is already 2-D passes through unchanged.
    """
    return X[:, :2]
