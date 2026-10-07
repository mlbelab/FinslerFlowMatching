"""``scripts.method`` — the notebook implementation of Finsler Flow Matching, as modules.

Why this package exists, and why it is not ``scripts.core``
-----------------------------------------------------------
The engine in ``scripts/core/`` is what the paper's *tables* were produced with: it is
tuned for sweeps, records, resumability and five separate benchmark trees.  The
notebooks are a **second, independent implementation of the same mathematics**, written
so the method can be read top to bottom — the metric, the three phases, the bridge, the
rulers — in plain PyTorch with nothing dispatching on an arm name.  That second reading
is the point of the notebooks, so it is kept.

What was wrong was that the second implementation lived *inside* one notebook, as a
thousand lines of module-level globals.  Nothing in it could be reused by a second
experiment, and the method could not be read apart from the Sheet's own bookkeeping.
``scripts.method`` is that code with every global turned into an explicit argument: the shared
half of a notebook, so a notebook holds only what is task-specific.

Position in the repository
--------------------------
``scripts.method`` is a **sibling** of ``scripts/core/`` and ``scripts/experiments/`` and may import both,
the same permission an experiment has.  Neither imports it.  The engine's verify suites
are therefore untouched by anything in here.

Layout, and the notebook cell each module came from
---------------------------------------------------
``config``   the :class:`~scripts.method.config.Run` dataclass that replaced the globals
``moments``  the two raw spatial moments of ``P`` — the only statistics anything reads
``metric``   the regularised Freidlin–Wentzell Randers metric, and ``sigma``'s convention
``nets``     the MFM SELU stack, the interpolant, the coupler
``phases``   Phase 1 (geodesic) / 2 (Finsler-cost OT) / 3 (distillation), and the sampler
``bridge``   Path B: Schrödinger-bridge matching and the full-SDE drift
``ConstantMobility``  (from ``scripts.core.bridge``) the frozen-``M`` control for Path B
``rulers``   the geometry-aware ``W_{2,F}`` and its Euclidean control
``report``   the ``mean ± sd`` table over model seeds
``tables``   every printed number on disk, so a table re-renders without a re-run
``figures``  the ``paper_figures`` backend fence, the block grid, the ``P``-flux quiver
``tuning``   ``Grid`` — one enumeration shared by an in-notebook sweep and a batch array
"""
from __future__ import annotations

from scripts.core.bridge import ConstantMobility
from scripts.method.bridge import SdeDrift, path_b, train_sbm
from scripts.method.config import Run
from scripts.method.metric import (ConformalFW, FWFactory, Metric, TrueFW,
                                   c_eff_of, sigma_for)
from scripts.method.moments import moments_from, rho_star
from scripts.method.nets import (Coupler, GeoPathNet, PhiNet, VelocityNet, interpolant,
                        timestep_embedding)
from scripts.method.phases import (build_ot, cuda_warm_up, finsler_cost, integrate, path_a,
                          train_cfm, train_geodesic)
from scripts.method.report import BASE_COLUMNS, pct_change, results_table, seed_frame
from scripts.method.rulers import euclidean_w2, finsler_w2
from scripts.method.tables import TableStore, table_summary, table_walltime
from scripts.method.tuning import (Fanout, Grid, cell_tag, format_tuned, on_edge,
                                   preflight_cell, print_ladder, rank, require_complete,
                                   smoke_root)

# ``scripts.method.figures`` is deliberately NOT imported here: it pulls in matplotlib,
# and a headless tuning job has no use for it.  A notebook that plots asks for it by name.

__all__ = [
    "Run",
    "moments_from", "rho_star",
    "Metric", "TrueFW", "ConformalFW", "FWFactory", "sigma_for", "c_eff_of",
    "GeoPathNet", "PhiNet", "VelocityNet", "timestep_embedding", "interpolant",
    "Coupler",
    "train_geodesic", "finsler_cost", "build_ot", "train_cfm", "integrate", "path_a",
    "cuda_warm_up",
    "train_sbm", "SdeDrift", "path_b", "ConstantMobility",
    "finsler_w2", "euclidean_w2",
    "seed_frame", "results_table", "pct_change", "BASE_COLUMNS",
    "TableStore", "table_summary", "table_walltime",
    "Grid", "Fanout", "rank", "on_edge", "format_tuned", "preflight_cell", "cell_tag", "smoke_root",
    "print_ladder",
    "require_complete",
]
