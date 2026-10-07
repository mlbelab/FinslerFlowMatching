"""The paper's hyper-parameter table, emitted from the ``tuned.json`` files themselves.

    python -m scripts.experiments.tuned_table            # every experiment
    python -m scripts.experiments.tuned_table pancreas   # one of them

Each experiment's tuner publishes its argmin to a ``tuned.json`` beside itself, and every
driver in this tree reads its point from that file -- so the file *is* what was run.  The
appendix table that reports those points is the one artefact that used to be typed out by
hand, which means it was the one place a number could disagree with the code without any
run failing.  This module closes that: the rows below are read from the same files the
drivers read, formatted, and printed for pasting into the manuscript.

Three things are deliberate.

* **The paths come from each tuner's own ``TUNED_FILE``** rather than being re-derived
  here, so there is no second opinion about which file is authoritative.
* **The row labels come from each experiment's own ``ARM_LABELS``**, the same dict its
  results tables label their rows with, so an arm cannot be called one thing in the
  results and another in the appendix.
* **The nesting is discovered, not declared.**  The five files are nested three different
  ways -- by dimension, not at all, and by dataset then coupling -- and a dimension added
  to a search should appear in the table without anyone editing this file.

Three notebooks also carry the point as a ``TUNED`` literal, which they pass to the
drivers instead of reading the file.  Nothing in a run compares the two, so
:func:`check_notebooks` does, and it runs before anything is emitted: the appendix reports
one point per arm and cannot be regenerated while there are two.

:data:`SYMBOLS` is the one hand-maintained thing here, and a key missing from it is an
assertion rather than a fallback: a new hyper-parameter needs a symbol chosen for it, and
the alternative is a table that quietly prints a raw identifier next to a number.

The output carries a digest of the files it was built from (:func:`digest`), so a table
already in the manuscript can be checked against the tree without re-reading every row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass

from scripts.core.paths import PUBLIC_ROOT, rel

#: LaTeX for every key a ``tuned.json`` can carry.  The ``_mult`` names are multipliers of
#: a measured scale of the data and not absolute values, which the subscript keeps visible:
#: reporting them as bare symbols would invite a reader to set them as absolutes.
SYMBOLS = {
    "rho_mult": r"\rho_{\mathrm{mult}}",
    "lam_mult": r"\lambda_{\mathrm{mult}}",
    "width_mult": r"w_{\mathrm{mult}}",
    "a_mult": r"a_{\mathrm{mult}}",
    "blur_frac": r"\varepsilon_{\mathrm{blur}}",
    "curly_alpha": r"\alpha_{\mathrm{curl}}",
    "land_rho": r"\rho_{\mathrm{LAND}}",
    "land_gamma": r"\gamma_{\mathrm{LAND}}",
}

#: what a cell says when the arm has no searched hyper-parameter at all.  OT-CFM on the
#: pancreas is the case: its one knob is fixed by the protocol, so an empty point is a
#: statement about the search and not a missing number.
NOTHING_SEARCHED = "---"


@dataclass(frozen=True)
class Table:
    """One experiment's block of the appendix table."""

    key: str
    title: str
    path: str
    labels: dict[str, str]
    order: tuple[str, ...] | None = None
    #: the notebook that carries this experiment's point as a ``TUNED`` literal, if one
    #: does.  The notebooks pass that literal to the drivers rather than reading the file
    #: -- deliberately, since a record is reusable only at the point it was run at
    #: (:func:`scripts.experiments.pancreas.noise.tuned_point` gives the reasoning) -- so
    #: the two copies have to be checked against each other somewhere, and generating the
    #: table that reports them is the moment where it costs nothing.
    notebook: str | None = None


def tables() -> list[Table]:
    """Every published ``tuned.json``, with the labels and paths its own code uses.

    Imported here rather than at module scope because each ``tune`` module pulls in the
    training stack, and this is a text generator that is run by hand.
    """
    from scripts.experiments.erythroid.train import ARM_LABELS as ERY_LABELS, ARMS as ERY_ARMS
    from scripts.experiments.erythroid.tune import TUNED_FILE as ERY_FILE
    from scripts.experiments.itracer.bench import ARM_LABEL as ITR_LABELS, ARMS as ITR_ARMS
    from scripts.experiments.itracer.tune import TUNED_FILE as ITR_FILE
    from scripts.experiments.matched.tune import TUNED_FILE as MAT_FILE
    from scripts.experiments.matched.variants import ARM_LABELS as MAT_LABELS, ARMS as MAT_ARMS
    from scripts.experiments.pancreas.train import ARM_LABELS as PAN_LABELS, REPORT_ORDER
    from scripts.experiments.pancreas.tune import TUNED_FILE as PAN_FILE
    from scripts.experiments.sheet.paper_focus import MAIN_BLOCKS, MAIN_LABELS as SHT_LABELS
    from scripts.experiments.sheet.tune import TUNED_FILE as SHT_FILE

    # the Sheet reports its rows in blocks; the appendix is one list, so they are
    # concatenated in the order the results table prints them
    sheet_arms = tuple(a for _, block in MAIN_BLOCKS for a in block)

    nb = lambda name: os.path.join(PUBLIC_ROOT, "notebooks", f"{name}_ffm_paper.ipynb")

    return [
        Table("sheet", "Sheet (route selection)", SHT_FILE, SHT_LABELS, sheet_arms,
              nb("sheet")),
        Table("pancreas", "Pancreas", PAN_FILE, PAN_LABELS, REPORT_ORDER, nb("pancreas")),
        Table("erythroid", "Erythroid", ERY_FILE, ERY_LABELS, tuple(ERY_ARMS),
              nb("erythroid")),
        Table("itracer", "iTracer (lineage)", ITR_FILE, ITR_LABELS, tuple(ITR_ARMS)),
        Table("matched", "Matched-budget variants", MAT_FILE, MAT_LABELS, tuple(MAT_ARMS)),
    ]


# --------------------------------------------------------------------------- #
#  Reading a tuned.json of unknown depth
# --------------------------------------------------------------------------- #
def _is_point(node) -> bool:
    """A leaf: ``knob -> value``, possibly empty because nothing was searched."""
    return isinstance(node, dict) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in node.values())


def columns(tuned: dict, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], dict]:
    """``tuned.json`` -> ``{key path: {arm: point}}``, at whatever depth the arms sit.

    The arm level is the deepest dict whose children are all points.  An all-empty level
    would be indistinguishable from a level of arms that searched nothing, so it is
    rejected rather than guessed at -- that shape does not occur and would mean the file
    records a search in which nothing was ever selected.
    """
    kids = list(tuned.values())
    assert kids, f"empty block at {'/'.join(path) or '<root>'}"
    if all(_is_point(v) for v in kids):
        assert any(kids), (
            f"every arm at {'/'.join(path) or '<root>'} has an empty point, so this level "
            "cannot be told from a level of nested blocks")
        return {path: tuned}
    out: dict[tuple[str, ...], dict] = {}
    for k, v in tuned.items():
        assert isinstance(v, dict) and not _is_point(v), (
            f"{'/'.join(path + (k,))} mixes a tuned point with nested blocks")
        out.update(columns(v, path + (k,)))
    return out


def _column_order(paths):
    """Dimensions ascend numerically; anything else keeps a stable lexicographic order."""
    numeric = all(len(p) == 1 and p[0].lstrip("-").isdigit() for p in paths)
    return sorted(paths, key=(lambda p: int(p[0])) if numeric else None)


# --------------------------------------------------------------------------- #
#  Formatting
# --------------------------------------------------------------------------- #
def _num(v) -> str:
    """``%g``, with the exponent form spelled the way LaTeX wants it."""
    s = f"{float(v):g}"
    if "e" not in s:
        return s
    mantissa, exponent = s.split("e")
    return f"{mantissa} \\times 10^{{{int(exponent)}}}"


def cell(point: dict) -> str:
    """One tuned point as a single math cell, knobs in a fixed order."""
    if not point:
        return NOTHING_SEARCHED
    parts = []
    for k in sorted(point):
        assert k in SYMBOLS, (
            f"no LaTeX symbol for hyper-parameter {k!r}; add one to "
            f"{__name__}.SYMBOLS rather than letting the table print the identifier")
        parts.append(f"{SYMBOLS[k]} = {_num(point[k])}")
    return "$" + r",\ ".join(parts) + "$"


def heading(path: tuple[str, ...], title: str) -> str:
    """A column heading from the key path: a dimension reads as one, everything else
    keeps the key the file uses, so a reader can find the block being quoted."""
    if not path:
        return title
    if len(path) == 1 and path[0].lstrip("-").isdigit():
        return f"$d = {path[0]}$"
    return "\\texttt{" + " / ".join(path).replace("_", r"\_") + "}"


def _rows(table: Table) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """``(column headings, [(row label, cells)])`` for one experiment."""
    with open(table.path) as fh:
        blocks = columns(json.load(fh))
    paths = _column_order(list(blocks))
    arms = table.order or tuple(sorted({a for b in blocks.values() for a in b}))

    unlabelled = [a for a in arms if a not in table.labels]
    assert not unlabelled, (
        f"{table.key}: {unlabelled} appear in {rel(table.path)} but not in the label map "
        "its results tables use; the appendix would name them differently")

    rows = []
    for arm in arms:
        if not any(arm in blocks[p] for p in paths):
            continue
        rows.append((table.labels[arm],
                     [cell(blocks[p][arm]) if arm in blocks[p] else NOTHING_SEARCHED
                      for p in paths]))
    return [heading(p, table.title) for p in paths], rows


def tex(table: Table) -> str:
    """One ``tabular`` for one experiment, ready to paste."""
    heads, rows = _rows(table)
    spec = "l" + "l" * len(heads)
    body = "\n".join(label + " & " + " & ".join(cells) + " \\\\"
                     for label, cells in rows)
    return (f"% {table.title} -- generated from {rel(table.path)}\n"
            "\\begin{tabular}{" + spec + "}\n\\toprule\n"
            "Method & " + " & ".join(heads) + " \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n")


# --------------------------------------------------------------------------- #
#  The notebook copies have to say the same thing
# --------------------------------------------------------------------------- #
def notebook_tuned(path: str) -> dict:
    """The ``TUNED`` literal out of a notebook, keyed the way the JSON file keys it."""
    with open(path) as fh:
        src = "\n".join("".join(c["source"]) for c in json.load(fh)["cells"]
                        if c["cell_type"] == "code")
    m = re.search(r"^TUNED = \{.*?^\}", src, re.M | re.S)
    assert m, f"{rel(path)} has no TUNED literal"
    ns: dict = {}
    exec(m.group(0), ns)                          # noqa: S102 -- our own text
    # the notebooks key by ``int`` dimension and the files by ``str``; a round trip
    # through JSON puts both in the files' form so the comparison is of values alone
    return json.loads(json.dumps(ns["TUNED"]))


def _differences(nb: dict, published: dict, path: tuple[str, ...] = ()) -> list[str]:
    """Every leaf the two copies disagree on, named by its path."""
    out = []
    for k in sorted(set(nb) | set(published), key=str):
        a, b = nb.get(k, "<absent>"), published.get(k, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += _differences(a, b, path + (str(k),))
        elif a != b:
            out.append(f"{'/'.join(path + (str(k),))}: notebook {a} vs file {b}")
    return out


def check_notebooks(chosen: list[Table] | None = None) -> None:
    """Fail unless every notebook's ``TUNED`` literal equals its published file.

    The table below reports one point per arm, so the two copies of that point have to be
    the same number.  Nothing in a run enforces it: a notebook hands its literal to the
    drivers, and a literal that has fallen behind the tuner still trains, still scores and
    still fills a table -- it just fills it with a different point than the appendix
    claims.  Checking here means the appendix cannot be regenerated while they disagree.
    """
    for t in chosen if chosen is not None else tables():
        if t.notebook is None or not os.path.exists(t.notebook):
            continue
        with open(t.path) as fh:
            published = json.load(fh)
        diffs = _differences(notebook_tuned(t.notebook), published)
        assert not diffs, (
            f"{rel(t.notebook)} and {rel(t.path)} select different points, so there is no "
            f"single point to report for {t.key}: " + "; ".join(diffs[:6]))


def digest(chosen: list[Table] | None = None) -> str:
    """A short hash of the files a table was built from, printed beside it.

    Cheap way to tell a pasted table apart from the tree it claims to report: re-run this
    and compare the line, rather than re-reading every cell.
    """
    h = hashlib.sha256()
    for t in chosen if chosen is not None else tables():
        with open(t.path, "rb") as fh:
            h.update(t.key.encode())
            h.update(fh.read())
    return h.hexdigest()[:12]


def report(chosen: list[Table] | None = None) -> str:
    """Every requested block, plus the digest line -- once the copies agree."""
    chosen = list(chosen if chosen is not None else tables())
    check_notebooks(chosen)
    return ("\n".join(tex(t) for t in chosen)
            + f"\n% tuned.json digest: {digest(chosen)}\n")


def main(argv=None) -> None:
    known = {t.key: t for t in tables()}
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", nargs="*", choices=sorted(known) or None,
                    help="one or more experiments; default is all of them")
    ap.add_argument("--out", help="write here instead of printing")
    a = ap.parse_args(argv)

    text = report([known[k] for k in a.experiment] if a.experiment else None)
    if not a.out:
        print(text, end="")
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(text)
    print(f"hyper-parameter table -> {rel(a.out)}")


if __name__ == "__main__":
    main()
