"""One enumeration of a search, shared by an in-notebook loop and a batch fan-out.

A grid written twice is a grid that will eventually disagree with itself: the notebook
sweeps ``for rho ... for lam ...`` and the cluster submits ``--index 0..N``, and if those
two orderings are written independently then a tuned point reported by one was measured
by the other.  :class:`Grid` is therefore the *only* place a search is spelled out.  It
is an ordered, index-addressable list of points — cell ``i`` is the same point in every
process, on every machine, forever — so a batch array is just ``seq 0 (len-1)`` and the
notebook loop is ``for point in grid``.

The last axis varies fastest, i.e. ``Grid(rho=..., lam=...)`` enumerates in exactly the
order a nested ``for rho: for lam:`` would.  That matters beyond tidiness: the sweep
consumes a seeded RNG, so reordering the enumeration changes the numbers.

:class:`Fanout` stacks per-arm grids behind one flat index, because arms do not share a
search space — ``ffm`` sweeps ``(rho, lam)``, a baseline sweeps its own one knob, and
``cfm`` sweeps nothing at all but still needs a cell so its row exists.
"""
from __future__ import annotations

import itertools
import os

import pandas as pd


class Grid:
    """``Grid(rho=[...], lam=[...])`` -> an ordered list of ``{"rho":…, "lam":…}``.

    An empty grid (no axes) has exactly one point, ``{}`` — an arm with nothing to select
    is one cell, not zero, so it still gets run, recorded and ranked beside the others.
    """

    def __init__(self, **axes):
        self.axes = {k: tuple(v) for k, v in axes.items()}
        self.points = [dict(zip(self.axes, combo))
                       for combo in itertools.product(*self.axes.values())]

    @classmethod
    def of(cls, points) -> "Grid":
        """A grid from an explicit ordered point list, for a ladder that is not a product.

        Almost every search here *is* a product and should be written as one.  The
        exception is a ladder with a point appended that its own units cannot express —
        the erythroid tree's ``pathb`` sweeps a bridge width in kernel bandwidths but
        must also measure the reference's *absolute* σ, which is a different number of
        bandwidths at every dimension.  Quoting the nearest rung as though it were the
        reference's setting would be the alternative, and it would be false.

        ``axes`` is then reconstructed from the points, in first-appearance order, so
        :func:`on_edge` and :meth:`__str__` keep working: an axis the appended point does
        not carry simply has one value fewer, and a key only *it* carries is a
        single-value axis, which ``on_edge`` already ignores.
        """
        g = cls.__new__(cls)
        g.points = [dict(p) for p in points]
        keys = dict.fromkeys(k for p in g.points for k in p)
        g.axes = {k: tuple(dict.fromkeys(p[k] for p in g.points if k in p))
                  for k in keys}
        return g

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, i):
        return self.points[i]

    def label(self, i: int) -> str:
        p = self.points[i]
        return " ".join(f"{k}={_num(v)}" for k, v in p.items()) or "(no free knobs)"

    def __str__(self):
        ax = ", ".join(f"{k}[{len(v)}]" for k, v in self.axes.items()) or "-"
        return f"Grid({ax}) = {len(self)} point{'s' * (len(self) != 1)}"


class Fanout:
    """Per-arm grids behind one flat index — one cell is one batch job.

    ``cells()`` yields ``(index, arm, point)`` in a fixed order; ``[i]`` is the inverse.
    Arms keep insertion order, so appending an arm never renumbers the ones before it and
    a half-finished search stays valid.
    """

    def __init__(self, grids: dict[str, Grid]):
        self.grids = dict(grids)
        self._flat = [(arm, p) for arm, g in self.grids.items() for p in g]

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, i: int) -> tuple[str, dict]:
        return self._flat[i]

    def cells(self):
        return [(i, arm, p) for i, (arm, p) in enumerate(self._flat)]

    def describe(self) -> str:
        lines = [f"{len(self)} cells over {len(self.grids)} arms"]
        i = 0
        for arm, g in self.grids.items():
            lines.append(f"  [{i:>3}..{i + len(g) - 1:>3}]  {arm:<10} {g}")
            i += len(g)
        return "\n".join(lines)


def cell_tag(wave: int, name: str, point: dict, dim: int | None = None) -> str:
    """The filename a sweep record is filed under: *what* it measured, not where it ran.

    Filing a cell under its flat index looks natural — the index is what a scheduler
    addresses it by — and is wrong the first time a ladder is widened.  Adding two rungs
    to a baseline's grid shifts every later arm's index by two, so ``--index 11`` now
    resolves to a point that has never been measured while ``w1_i011.json`` still holds
    the *other* arm's record: the run is refused as "already recorded", and forcing it
    would overwrite a good cell with an unrelated one.  Keying on content makes both
    failures unrepresentable — a widened grid simply has two files that do not exist yet,
    and every record already on disk keeps naming the point it measured.

    The tag is readable rather than hashed, because a sweep directory is something a
    human reads: ``w1_Arch-mfm_land_land_gamma0.5.json``.
    """
    parts = [f"w{wave}"] + ([f"d{dim}"] if dim is not None else [])
    parts.append(name.replace("/", "-"))
    parts += [f"{k}{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool)
              else f"{k}{v}" for k, v in sorted(point.items())]
    slug = "_".join(parts)
    return "".join(c if (c.isalnum() or c in "._+-") else "-" for c in slug)


def preflight_cell(waves: dict, wave: int, index: int, dim: int | None, dims,
                   root: str, path_for, force: bool = False,
                   cache: str | None = None) -> tuple[str, dict, str]:
    """Everything that can fail about one sweep cell, checked before the first tensor.

    A search is submitted as a few hundred identical jobs, so a mistake in the invocation
    — a stale cell count, a dimension that was never searched, a data cache the compute
    node cannot see — is a mistake in *every* job at once.  Checked here, the array dies
    in its first second with a message naming the fix; checked implicitly by the first
    tensor operation that trips over it, it dies after the model has been built and the
    cloud loaded, once per cell, with an ``IndexError``.

    Returns the ``(arm, point, path)`` the cell resolves to, so the caller does not index
    the fanout a second time.  ``cache`` is a path that must exist (the tree's data
    cache); ``path_for(arm, point)`` says where this cell's record will be written and is
    a *callable* rather than a path, because the path is a function of the point and the
    point is not known until the index has been resolved here — see :func:`cell_tag`.  An
    existing record is refused unless ``force``: re-running a finished cell is nearly
    always an accident of resubmission, and silently overwriting it loses the record that
    was already ranked.
    """
    assert wave in waves, f"unknown wave {wave}; this tree has {sorted(waves)}"
    # ``dims`` is empty for a benchmark that lives in one space (the Sheet, the Arch), so
    # those trees have no ``--dim`` at all and pass ``None``.  The two real-data trees
    # search each space independently and must reject a space that was never searched.
    dims = tuple(dims)
    if dims:
        assert dim in dims, f"d={dim} was not searched; this tree searches {dims}"
    else:
        assert dim is None, f"this tree searches one space; --dim {dim} means nothing here"

    n = len(waves[wave])
    assert 0 <= index < n, (
        f"--index {index} is out of range for wave {wave}, which has {n} cells "
        f"(0..{n - 1}); `plan --count {wave}` prints that number")

    if cache is not None:
        assert os.path.exists(cache), (
            f"data cache not found at {cache}; this node cannot see it, or step 0 has "
            f"not been run")

    name, point = waves[wave][index]
    cell_path = path_for(name, point)
    if os.path.exists(cell_path) and not force:
        raise SystemExit(
            f"cell already recorded at {cell_path}; pass --force to re-measure it")
    os.makedirs(os.path.dirname(cell_path) or ".", exist_ok=True)
    probe = cell_path + ".probe"
    with open(probe, "w") as fh:                     # writable *now*, not after training
        fh.write("")
    os.remove(probe)

    return name, point, cell_path


def smoke_root(root: str, default_root: str, name: str, smoke: bool) -> str:
    """The tree a sweep cell writes into, with a wiring check sent to its own sibling.

    ``NB_SMOKE=1`` runs a search cell at a meaningless budget, which is exactly what you
    want before submitting four hundred of them — and exactly what must not be ranked.
    A smoke record is shaped like a real one and lands under a content-addressed name, so
    dropped into the reported ``sweep/`` it is indistinguishable from the cell it
    impersonates and ``collect`` will pick an argmin off it.  So a smoke cell is
    redirected to ``<name>_smoke``, the same suffix
    :func:`~scripts.core.paths.experiment_root` gives a smoke run tree.

    An **explicit** ``--root`` is honoured either way: a caller who named the directory
    has said where the records go, and second-guessing that would make ``--root`` mean
    two things.  Only the default is redirected.
    """
    from scripts.core.paths import experiment_root
    if not smoke or os.path.abspath(root) != os.path.abspath(default_root):
        return root
    return experiment_root(name, smoke=True)


def rank(rows, key: str, columns=None, ascending: bool = True):
    """Sort the sweep records on ``key`` -> ``(dataframe, best row as a plain dict)``.

    Returns the frame as well as the winner because a selection that is only ever shown
    as its argmin cannot be sanity-checked: a grid whose top three are within noise, or
    whose argmin sits on an edge, is visible in the table and invisible in the answer.
    """
    df = pd.DataFrame(list(rows)).sort_values(key, ascending=ascending)
    assert len(df), "nothing to rank -- no sweep cell produced a record"
    best = df.iloc[0].to_dict()          # read BEFORE the display filter, so the winner
    if columns is not None:              # keeps every field and not just the shown ones
        df = df[[c for c in columns if c in df.columns]]
    return df, best


def on_edge(grid: Grid, point: dict) -> list[str]:
    """The axes whose selected value is an endpoint — the grid may be too narrow.

    Reported rather than acted on: widening is a decision (and, per the MFM tree's rule,
    one applied to the baselines first), not something a ranking function should do.
    """
    return [k for k, vals in grid.axes.items()
            if len(vals) > 1 and point.get(k) in (vals[0], vals[-1])]


def print_ladder(df, best: dict, grid: Grid, *, header: str, key: str, terms,
                 flat_pct: float, n_have: int, n_want: int) -> None:
    """One arm's ranked ladder, its winner, and whether the ladder was wide enough.

    Shared by every tuner rather than copied into each, because the *edge caution* is the
    one part of a search that has to read the same everywhere: it is what licenses a
    widened ladder, and four independently-worded versions of it would eventually
    disagree about what "flat" means.

    An endpoint winner is only a *narrow* ladder if the score is still falling as it
    leaves.  If the tail has gone flat, another rung buys a coin flip between values that
    differ by less than the run-to-run spread — so the slope at the winner's inward
    neighbour is printed, against ``flat_pct``, instead of a bare "argmin on the edge".
    """
    cols = ["point", *terms, key]
    short = df[[c for c in cols if c in df.columns]].to_string(index=False,
                                                               float_format="%.4f")
    print(f"\n  {header}  ({n_have}/{n_want} cells, ranked on `{key}`)")
    print("    " + short.replace("\n", "\n    "))

    if not (edge := on_edge(grid, best["point"])):
        return
    print(f"    caution: the argmin sits on an endpoint in {edge}")
    rows = df.to_dict("records")
    for axis in edge:
        vals = grid.axes[axis]
        step = vals[1] if best["point"][axis] == vals[0] else vals[-2]
        inward = {**best["point"], axis: step}
        hit = [r for r in rows if r["point"] == inward]
        if not hit:
            continue
        drop = (hit[0][key] - best[key]) / abs(best[key]) * 100
        verdict = "flat; converged" if abs(drop) < flat_pct else "a gradient; widen"
        print(f"      {axis}: {key} {best[key]:.4f} at the edge against "
              f"{hit[0][key]:.4f} one rung in ({drop:+.1f}%) -- {verdict}")


def require_complete(missing, *, partial: bool = False,
                     quiet: bool = False) -> None:
    """Refuse to publish a ``TUNED`` literal from a half-finished sweep.

    ``missing`` is ``[(label, n_have, n_want), ...]`` for every arm that is short of
    cells.  The failure mode this exists to stop is quiet and expensive: with three of
    sixteen cells present, ``collect`` still ranks, still writes ``tuned.json`` and still
    prints a literal that *looks* like an answer -- and the argmin of a third of a ladder
    is very often on an endpoint, which then reads as "widen the grid" rather than as
    "you have not run it yet".  Pasting that into a notebook costs a full reported grid
    before anyone notices.

    ``partial=True`` is the escape hatch, for reading a search that is still in flight.
    It prints the same shortfall and writes nothing that the caller does not choose to.
    ``quiet`` silences even that, and is for the one caller that is neither: wave 2's
    ``run``, which reads wave 1's answer to build its own cell.  That read is complete by
    construction -- ``run`` asserts the parent arm is present -- and it happens while
    wave 2 is by definition unfinished, so warning about wave 2 there would fire on every
    cell of every wave-2 job and mean nothing.
    """
    if not missing:
        return
    short = "; ".join(f"{label} {have}/{want}" for label, have, want in missing)
    total = sum(want - have for _, have, want in missing)
    if partial:
        if not quiet:
            print(f"\n  --partial: {total} cell(s) still missing ({short}); the points "
                  f"below are the best of what has finished, not the search's answer")
        return
    raise SystemExit(
        f"\nrefusing to write a tuned point: {total} cell(s) of the sweep have not "
        f"finished ({short}).\nRun them -- `plan` prints the loop -- or pass `--partial` "
        f"to rank what is there without publishing it.")


def format_tuned(tuned: dict, name: str = "TUNED", indent: int = 4) -> str:
    """The selected points as a paste-ready Python literal.

    Hardcoding a tuned point into a notebook is only honest if the literal is *emitted*
    by the search rather than typed from a log, so this is the one authorised way to
    produce it and ``verify`` checks the pasted copy against the search's own record.
    """
    pad = " " * indent
    lines = [f"{name} = {{"]
    for outer, inner in tuned.items():
        if isinstance(inner, dict) and inner and all(isinstance(v, dict)
                                                     for v in inner.values()):
            lines.append(f"{pad}{_num(outer)}: {{")
            for arm, pt in inner.items():
                lines.append(f"{pad * 2}{arm!r}: {_pt(pt)},")
            lines.append(f"{pad}}},")
        else:
            lines.append(f"{pad}{_num(outer)}: {_pt(inner)},")
    lines.append("}")
    return "\n".join(lines)


def _pt(p: dict) -> str:
    return "{" + ", ".join(f"{k!r}: {_num(v)}" for k, v in p.items()) + "}"


def _num(v) -> str:
    """Floats at full precision in scientific form; anything else through ``repr``.

    ``%.6e`` and not ``repr(float)``: a tuned rho is a multiple of a measured scale and
    reads as noise at seventeen digits, but rounding it to four would make the pasted
    literal a *different* point from the one that was measured.
    """
    return f"{v:.6e}" if isinstance(v, float) else repr(v)
