"""``mean ± sd`` over model seeds, and the floor printed beside it.

Two conventions, both deliberate and both worth stating once here rather than in every
notebook that prints a table:

* **The spread is over model seeds and nothing else.**  The split is drawn once and held
  fixed, so a seed varies the initialisation, the batch order and the coupling draw —
  training stochasticity, which is what dominates at this cloud size.  ``ddof = 0``
  because these are the runs, not a sample from a larger pool we are estimating.
* **Every distance column is quoted against a floor**, the same distance between a draw
  from the reference cloud and the rest of it, sized to the cloud the arm itself pushes
  and averaged over draws (:func:`scripts.core.metrics.floor_splits`).  No method can go
  below it, so a number near the floor means *solved*, not *good*, and a table without
  the floor row cannot be read at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: ``column -> path into the metrics dict``.  The two distances every benchmark reports,
#: and the two the selection objective is the mean of — so the table a reader sees and
#: the rule that picked the hyper-parameters are the same numbers.  Anything that says
#: *how* a method missed is task-specific and is appended by the notebook (the Sheet's
#: route breakdown).
BASE_COLUMNS: dict[str, tuple[str, ...]] = {
    "mid W2":   ("intermediate", "W2"),
    "end W2":   ("endpoint", "W2"),
}


def _dig(d, path):
    for k in path:
        d = d[k]
    return d


def seed_frame(runs, name, seeds, columns=BASE_COLUMNS) -> pd.DataFrame:
    """One row per model seed for one arm — the object the ± is computed from.

    Kept separate from :func:`results_table` so a notebook can look at the per-seed
    numbers when a spread is large, instead of only at the summary that hides them.
    """
    return pd.DataFrame([{c: _dig(runs[(name, s)]["metrics"], p)
                          for c, p in columns.items()} for s in seeds])


def results_table(runs, names, seeds, columns=BASE_COLUMNS, floors=None,
                  fmt: str = "{:.4f}") -> pd.DataFrame:
    """``mean ± sd`` over ``seeds``, arms as rows, with an optional leading floor row.

    ``floors`` is the metrics dict of any run (they all carry the same one, since the
    floor is a property of the reference cloud and not of the method); its entries are
    read through the same ``columns`` paths, so a column added above appears in the floor
    row automatically, and a column the floor has no entry for reads ``--`` rather than
    quietly borrowing a neighbour's.
    """
    rows = {}
    for nm in names:
        f = seed_frame(runs, nm, seeds, columns)
        rows[nm] = {c: f"{fmt.format(f[c].mean())} ± {fmt.format(f[c].std(ddof=0))}"
                    for c in columns}
    if floors is not None:                      # last, so it reads as the line to beat
        rows["sampling floor"] = {c: (fmt.format(_dig(floors, p))
                                      if _has(floors, p) else "--")
                                  for c, p in columns.items()}
    return pd.DataFrame(rows).T


def _has(d, path) -> bool:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return False
        d = d[k]
    return True


def pct_change(runs, a: str, b: str, seeds, column=("intermediate", "W2")) -> float:
    """``(b - a) / a`` in per cent on the seed means — how much ``b`` bought over ``a``.

    Reported on the means rather than per seed and averaged, because the seeds are not
    paired across arms: each arm draws its own initialisation, so seed 3 of one is not a
    matched control for seed 3 of another.
    """
    m = lambda nm: float(np.mean([_dig(runs[(nm, s)]["metrics"], column) for s in seeds]))
    return 100.0 * (m(b) - m(a)) / m(a)
