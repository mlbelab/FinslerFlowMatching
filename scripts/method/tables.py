"""Every number a notebook prints, on disk, so a table can be re-read without a re-run.

The tables in these notebooks are computed inside the same cell that trains, and printed.
That coupling is what makes them expensive to touch: changing a column heading, adding a
row, or simply looking up a figure from last week costs a full retrain, and a kernel that
has been restarted has lost the run entirely.  This module breaks the coupling in the only
place it actually exists — persistence — and changes nothing about what is computed.

    TBL = TableStore.open(path, benchmark="pancreas", ...)   # load if present, else empty
    TBL.put_run(dim, arm, seed, metrics=..., seconds=...)    # during the run loop
    TBL.save()
    results_table(TBL.runs(dim), REPORTED, seeds, COLUMNS, floors=TBL.floors(dim))

:meth:`TableStore.runs` returns exactly the ``{(arm, seed): {"metrics": ...}}`` mapping
that :mod:`scripts.method.report` already consumes, and :meth:`TableStore.times` the
``{(arm, seed): seconds}`` one the wall-clock table already consumes.  So the renderers are
the ones that were already there; nothing about a table's *shape* is restated here, and a
notebook switches over by reading its numbers from the store instead of from the live run.

What is stored is **numbers, not rendered text**.  A stored string would freeze the
formatting decision at write time, which is the thing being complained about; a stored
number can be re-rendered into any table, including one that does not exist yet.  The
corollary is that anything derived from a *trajectory* — ``W_{2,F}``, the moment cosines —
has to be computed while the trajectories are alive and put here explicitly
(:meth:`put_custom`), because the trajectories themselves are far too large to keep and
are not what a table wants anyway.

The file is written atomically (temp file, then rename), so an interrupted save leaves the
previous answer intact rather than a truncated one.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.method.report import BASE_COLUMNS, seed_frame

#: bumped when a field changes meaning rather than when one is added.  :meth:`open`
#: refuses a file from a future schema instead of silently mis-reading it.
SCHEMA = 1


def _plain(o: Any) -> Any:
    """NumPy scalars and arrays down to what :mod:`json` will accept, recursively.

    Metrics dicts arrive with a mix of Python floats, ``np.float64`` and the occasional
    ``np.int64`` count, depending on which ruler produced them; normalising here keeps the
    call sites from each having to remember which is which.
    """
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return _plain(o.item())
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    raise TypeError(f"{type(o).__name__} is not a table number; store a float, not an object")


class TableStore:
    """The JSON behind a notebook's tables: open, put, save, render.

    Keyed ``dim -> arm -> seed``, with the per-dimension constants (the floor, sigma, the
    reference size) beside the runs rather than inside each of them, since they are
    properties of the cloud and repeating them per seed would let two copies disagree.
    """

    def __init__(self, path: str, data: dict | None = None):
        self.path = path
        self.data: dict = data if data is not None else {
            "schema": SCHEMA, "meta": {}, "tuned": {}, "dims": {}}

    # -- lifecycle ----------------------------------------------------------------#
    @classmethod
    def open(cls, path: str, **meta) -> "TableStore":
        """Load ``path`` if it exists, else start empty; either way merge ``meta`` in.

        Re-opening is the ordinary case, not the exception: a notebook run with
        ``NB_DIMS=50`` should add d = 50 to what a previous run already measured rather
        than discard it, so nothing here truncates.  ``meta`` is merged on every open so
        the device and seed list describe the *most recent* writer.
        """
        data = None
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
            got = data.get("schema")
            assert got == SCHEMA, (
                f"{path} was written by schema {got}, this is schema {SCHEMA}; "
                f"delete it to re-measure, or check out the matching revision")
        st = cls(path, data)
        st.data.setdefault("meta", {}).update(_plain(meta))
        return st

    def save(self) -> "TableStore":
        """Write atomically, so an interrupted save cannot destroy the previous answer."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=1, sort_keys=False)
        os.replace(tmp, self.path)
        return self

    # -- writing ------------------------------------------------------------------#
    def _dim(self, dim: int) -> dict:
        return self.data["dims"].setdefault(str(dim), {"runs": {}, "custom": {}})

    def put_meta(self, **kw) -> "TableStore":
        self.data.setdefault("meta", {}).update(_plain(kw))
        return self

    def put_tuned(self, tuned: dict) -> "TableStore":
        """The selected points, so a stored table can say what produced it."""
        self.data["tuned"] = _plain(tuned)
        return self

    def put_dim(self, dim: int, **kw) -> "TableStore":
        """Per-dimension constants: ``sigma``, ``n_ref_mid``, ``floors``, whatever else."""
        self._dim(dim).update(_plain(kw))
        return self

    def put_run(self, dim: int, arm: str, seed: int, metrics: dict,
                seconds: float | None = None) -> "TableStore":
        """One cell of the main table: an arm at a seed at a dimension."""
        cell = self._dim(dim)["runs"].setdefault(str(arm), {}).setdefault(str(seed), {})
        cell["metrics"] = _plain(metrics)
        if seconds is not None:
            cell["seconds"] = float(seconds)
        return self

    def put_custom(self, dim: int, block: str, payload: dict) -> "TableStore":
        """A task-specific table that is not ``mean ± sd`` over the two distances.

        ``block`` names it (``"wf"``, ``"moments"``, ...) so a notebook can add one without
        this module needing to know what it is.  The payload is stored verbatim and handed
        back verbatim by :meth:`custom`; only the renderers know its shape.
        """
        self._dim(dim)["custom"][block] = _plain(payload)
        return self

    # -- reading ------------------------------------------------------------------#
    @property
    def meta(self) -> dict:
        return self.data.get("meta", {})

    @property
    def tuned(self) -> dict:
        return {int(d): v for d, v in self.data.get("tuned", {}).items()}

    def dims(self) -> tuple[int, ...]:
        return tuple(sorted(int(d) for d in self.data["dims"]))

    def arms(self, dim: int) -> tuple[str, ...]:
        """Insertion order, which is the order the run loop measured them in."""
        return tuple(self._dim(dim)["runs"])

    def seeds(self, dim: int, arm: str) -> tuple[int, ...]:
        return tuple(sorted(int(s) for s in self._dim(dim)["runs"].get(str(arm), {})))

    def has(self, dim: int, arm: str, seed: int) -> bool:
        return str(seed) in self._dim(dim)["runs"].get(str(arm), {})

    def runs(self, dim: int) -> dict:
        """``{(arm, seed): {"metrics": ...}}`` — what :mod:`scripts.method.report` eats.

        The same mapping the live run loop builds, minus the nets and the trajectories, so
        every existing table renderer works against a stored run unchanged.
        """
        return {(arm, int(s)): {"metrics": cell["metrics"]}
                for arm, by_seed in self._dim(dim)["runs"].items()
                for s, cell in by_seed.items()}

    def times(self, dim: int) -> dict:
        """``{(arm, seed): seconds}`` — what the wall-clock table eats."""
        return {(arm, int(s)): cell["seconds"]
                for arm, by_seed in self._dim(dim)["runs"].items()
                for s, cell in by_seed.items() if "seconds" in cell}

    def floors(self, dim: int) -> dict | None:
        """The sampling floor, in the shape ``results_table(floors=...)`` expects."""
        return self._dim(dim).get("floors")

    def const(self, dim: int, key: str, default=None):
        """One per-dimension constant put by :meth:`put_dim` — ``sigma``, a count, ...

        Named rather than reached for through :attr:`data` so a notebook printing a header
        does not have to know how the file is nested.
        """
        return self._dim(dim).get(key, default)

    def custom(self, dim: int, block: str) -> dict | None:
        return self._dim(dim)["custom"].get(block)


# --------------------------------------------------------------------------- #
#  Renderers that read the store instead of a live run
# --------------------------------------------------------------------------- #
def table_summary(store: TableStore, arms: Iterable[str], seeds: Iterable[int],
                  dims: Iterable[int] | None = None, column: str = "mid W2",
                  columns: dict = BASE_COLUMNS, floor_key: str = "intermediate",
                  floor_metric: str = "W2") -> pd.DataFrame:
    """One column per dimension, one row per arm, with the floor as the last row.

    The cross-dimension read that only makes sense once every dimension is in one place,
    which before this module it never was — the notebook had to hold all of them live in a
    single kernel to print it.

    An arm the store has no run for at some dimension reads ``NaN`` in that column rather
    than taking the whole table down with a ``KeyError``: a benchmark whose blocks are not
    all measured at every dimension is the ordinary case, not a corrupt store.
    """
    dims = store.dims() if dims is None else tuple(dims)
    seeds = tuple(seeds)
    out = {}
    for d in dims:
        runs = store.runs(d)
        col = {nm: (seed_frame(runs, nm, seeds, columns)[column].mean()
                    if all((nm, s) in runs for s in seeds) else np.nan)
               for nm in arms}
        fl = (store.floors(d) or {}).get(floor_key, {}).get(floor_metric)
        if fl is not None:
            col["sampling floor"] = fl
        out[d] = col
    return pd.DataFrame(out)


def table_walltime(store: TableStore, arms: Iterable[str], seeds: Iterable[int],
                   dims: Iterable[int] | None = None) -> pd.DataFrame:
    """Mean seconds per arm per dimension, with the all-arms-one-seed total beneath.

    Missing cells read ``NaN`` for the reason given in :func:`table_summary`.  The total
    row still sums what *is* there, so it is the cost of the runs the store holds rather
    than an estimate of the runs it does not.
    """
    dims = store.dims() if dims is None else tuple(dims)
    arms, seeds = tuple(arms), tuple(seeds)

    def _mean(d, nm):
        got = [store.times(d)[(nm, s)] for s in seeds if (nm, s) in store.times(d)]
        return float(np.mean(got)) if len(got) == len(seeds) else np.nan

    frame = pd.DataFrame({f"d = {d}": {nm: _mean(d, nm) for nm in arms} for d in dims})
    frame.loc["one seed, all arms"] = [sum(store.times(d).values()) / len(seeds)
                                       for d in dims]
    return frame
