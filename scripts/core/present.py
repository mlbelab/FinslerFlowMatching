"""What a result *looks like*: the region codes, the paper palette, the method's name,
and the table writers.

This exists because three different kinds of caller need exactly the same ink and the
same column formatting, and the dependency rule forbids two of the three pairings:

* an **experiment** (``scripts.experiments.sheet.paper_report`` and friends) writes the
  paper's ``.tex`` and ``.md`` tables;
* **the method's figure layer** (:mod:`scripts.method.figures`) draws the notebook panels;
* every other **experiment** — circles, mfm, erythroid — reuses the Sheet's palette so
  that orange means "the withheld thing to be recovered" in every figure in the paper.

Before the restructure the palette and the table writers lived under
``experiments/sheet/``, so ``scripts.method`` imported an experiment and three
experiments imported a fourth.  Both are now gone: this module is in the engine, which
imports nothing above it, and everyone reads from here.

Nothing here computes anything.  It is names, colours and string formatting, kept in one
file so a figure and the table beside it cannot disagree about either.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

# --------------------------------------------------------------------------- #
#  Region codes — what a cell *is* in a withheld-marginal benchmark
# --------------------------------------------------------------------------- #
#: The four roles a cell can play.  Defined here rather than in a dataset module because
#: four benchmarks label their clouds with them and the codes have to be the same
#: integers in all four, or a palette lookup silently returns the wrong colour.
SOURCE, TARGET, GAP, DECOY = 0, 1, 2, 3

REGION_NAMES = {SOURCE: "source", TARGET: "target", GAP: "gap", DECOY: "decoy"}
REGION_LABELS = {
    SOURCE: r"$p_0$ (shown)",
    TARGET: r"$p_1$ (shown)",
    GAP: "gap — evaluation only",
    DECOY: "decoy — wrong route, also withheld",
}

# --------------------------------------------------------------------------- #
#  The paper palette
# --------------------------------------------------------------------------- #
#: the two greys every panel is built on: unremarkable cells, and everything on top
INK = "#222222"
FAINT = "#b8b8b8"

#: Paper palette.
PAPER_REGIONS = {
    # The two marginals are the cool pair and the withheld strip is the warm one, so
    # "the answer" separates from "the data" by hue *and* by temperature.  Within the
    # cool pair the split is by *lightness*, not hue: a violet and a blue-violet of the
    # same lightness are hard to tell apart once alpha washes them out, so p_0 is taken
    # dark and p_1 is taken bright and clearly blue.
    SOURCE: "#431075",     # deep violet   -- p_0, shown
    TARGET: "#2E6BF6",     # clear blue    -- p_1, shown
    GAP: "#FF8904",        # orange        -- withheld; the thing to be recovered
    # The decoy is neutral rather than a fourth hue.  A decoy is by definition the
    # region a reader must *not* mistake for the answer, so giving it a saturated colour
    # of its own next to the orange corridor would invite the exact confusion it exists
    # to test for.  (The Sheet has no decoy; the code path is kept for clouds that do.)
    DECOY: "#9AA0A6",      # neutral grey  -- withheld, but not the answer
}

# --------------------------------------------------------------------------- #
#  What the method is called
# --------------------------------------------------------------------------- #
#: what *both* of our arms are called in the paper.  Path B is not a second method: it is
#: the same geometry with the SDE switched on.  It used to be given an acronym of its own
#: (``FSBM``), which read as a third method sitting beside FFM -- exactly the reading the
#: pair exists to rule out -- so the two now share a name and differ by a qualifier.
PATHB_NAME = "FFM"
#: the two qualifiers, kept as their own constants because the tables need them *inline*
#: and the figures need them on a second line.  The two forms cannot be one string: a
#: newline inside a ``tabular`` cell is a LaTeX error, and :func:`_tex_table` writes one
#: physical row per label.
QUAL_DET = "deterministic"
QUAL_NOISY = "with noise"

#: inline forms -- tables, LaTeX cells, fixed-width text columns, markdown bullets
FFM_LABEL = f"{PATHB_NAME}, {QUAL_DET}"
PATHB_LABEL = f"{PATHB_NAME}, {QUAL_NOISY}"


def two_line(qualifier: str, ours: bool = False) -> str:
    """``FFM`` on the first line, the qualifier under it -- **figure headings only**.

    A column heading is the one place the pair reads best stacked: the eye picks up the
    shared first line as "the same method twice" before it reads the qualifiers.  Never
    hand the result to a table writer -- see the note on :data:`QUAL_DET`.
    """
    return f"{PATHB_NAME}{' (ours)' if ours else ''}\n{qualifier}"


# --------------------------------------------------------------------------- #
#  Records -> cells -> mean+-sd strings
# --------------------------------------------------------------------------- #
def _index(recs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """``(dataset, arm) -> [record per seed]``."""
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        out[(r["spec"]["dataset"], r["spec"]["arm"])].append(r)
    return out


def _dig(rec: dict, *path: str) -> float:
    node = rec["metrics"]
    for p in path:
        node = node[p]
    return float(node)


def _agg(cells: list[dict], *path: str) -> tuple[float, float, int]:
    """Mean, sample sd and count of one metric across the seeds of a cell."""
    vals = np.array([_dig(r, *path) for r in cells], dtype=float)
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), sd, len(vals)


def _fmt(mean: float, sd: float, n: int, prec: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    if n <= 1:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f}±{sd:.{prec}f}"


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def _blocked(body_labels, blocks):
    """``(heading_or_None, label, row_index)`` for every body row, in order.

    ``blocks`` is ``[(heading, n_rows), ...]`` partitioning the rows; the heading is
    attached to the first row of its block and is ``None`` everywhere else.  With
    ``blocks=None`` this degenerates to one unheaded block, which is why every caller
    that does not want grouping can keep passing nothing.
    """
    if blocks is None:
        blocks = [(None, len(body_labels))]
    total = sum(n for _h, n in blocks)
    assert total == len(body_labels), (
        f"blocks cover {total} rows but the table has {len(body_labels)}")
    out, r = [], 0
    for heading, n in blocks:
        for k in range(n):
            out.append((heading if k == 0 else None, body_labels[r], r))
            r += 1
    return out


def _md_table(header_top, header_sub, rows, body_labels, floor=None,
              blocks=None) -> str:
    head = "| Method | " + " | ".join(
        f"{d} {c}" for d, cs in header_top for c in cs) + " |"
    ncol = sum(len(cs) for _d, cs in header_top)
    rule = "|" + "---|" * (1 + ncol)
    lines = [head, rule]
    for heading, label, r in _blocked(body_labels, blocks):
        if heading is not None:
            lines.append(f"| **{heading}** | " + " | ".join([""] * ncol) + " |")
        lines.append("| " + label + " | " + " | ".join(rows[r]) + " |")
    if floor is not None:
        lines.append("| *sampling floor* | " + " | ".join(f"*{x}*" for x in floor) + " |")
    return "\n".join(lines) + "\n"


def _tex_table(label, col_groups, rows, body_labels, floor=None,
               best: np.ndarray | None = None, lower_is_better=True,
               blocks=None, placement="t", resize=False) -> str:
    """A LaTeX table and nothing else.

    The caption is emitted **empty** on purpose: the ``.tex`` files are pasted
    straight into the manuscript, and prose written by this script would be prose
    the paper did not write.  The legend that explains each table lives in the
    ``.md`` twin, which is what is read here.  ``\\caption{}`` is kept rather than
    dropped so the table still takes a number and ``\\label`` still resolves.
    """
    ncol = sum(len(cs) for _d, cs in col_groups)
    spec = "l" + "".join("r" * len(cs) for _d, cs in col_groups)
    head1, head2, i = [], [], 1
    for dname, cs in col_groups:
        head1.append(f"\\multicolumn{{{len(cs)}}}{{c}}{{{dname}}}")
        head2 += list(cs)
        i += len(cs)
    cmid = []
    start = 2
    for _d, cs in col_groups:
        cmid.append(f"\\cmidrule(lr){{{start}-{start + len(cs) - 1}}}")
        start += len(cs)

    body = []
    for heading, label_, r in _blocked(body_labels, blocks):
        if heading is not None:
            # a rule only *between* blocks: the header already ends in a \midrule
            if body:
                body.append("\\midrule")
            body.append(f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\emph{{{heading}}}}} \\\\")
        cells = []
        for c, txt in enumerate(rows[r]):
            if best is not None and best[r, c]:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt.replace("±", " $\\pm$ "))
        body.append(label_ + " & " + " & ".join(cells) + " \\\\")
    if floor is not None:
        body.append("\\midrule\n\\emph{sampling floor} & "
                    + " & ".join(f"\\emph{{{x}}}".replace("±", " $\\pm$ ")
                                 for x in floor) + " \\\\")
    # a many-cloud table runs past \textwidth; shrinking beats silently overfull boxes
    open_box, close_box = ("\\resizebox{\\textwidth}{!}{%\n", "}\n") if resize else ("", "")
    return (
        f"\\begin{{table}}[{placement}]\n\\centering\n\\small\n" + open_box
        + f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
        " & " + " & ".join(head1) + " \\\\\n" + "".join(cmid) + "\n"
        "Method & " + " & ".join(head2) + " \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n" + close_box
        + f"\\caption{{}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )


def _best_mask(raw: np.ndarray, lower_is_better=True) -> np.ndarray:
    """Column-wise winner mask, NaN-safe."""
    mask = np.zeros(raw.shape, dtype=bool)
    for c in range(raw.shape[1]):
        col = raw[:, c]
        if not np.isfinite(col).any():
            continue
        r = int(np.nanargmin(col) if lower_is_better else np.nanargmax(col))
        mask[r, c] = True
    return mask
