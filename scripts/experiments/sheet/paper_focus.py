"""The two submission tables: a main-text method comparison and an appendix ablation.

This is a *view* over the same test records :mod:`scripts.experiments.sheet.paper_report` reads --
nothing is retrained and no number is recomputed, so a row here always equals the
corresponding row of ``tables.md``.  What differs is the framing:

F1  **Main text.**  Distribution accuracy only -- intermediate and endpoint $W_2$ and
    their RBF-MMD counterparts.  The route share is deliberately *absent*: it is a
    diagnostic of *where* the mass went, and the main table's job is *how close* the
    transported cloud is.  The route breakdown stays in the appendix (T2).
    Rows are split into a deterministic block ($\\sigma = 0$, an ODE pushforward) and a
    stochastic block ($\\sigma > 0$, an SDE), because the two answer different questions
    and bolding a single winner across them would compare an ODE against an SDE.  The
    split follows :data:`scripts.core.arms.NOISY_ARMS`, which is what the
    training code itself branches on.  Curly-FM is in the deterministic block because
    on this cloud it *is* deterministic: its interpolant noise runs at the zero its own
    authors report every result at (Petrović et al. 2025, App. G), there being no
    reference noise level for a synthetic saddle.

F2  **Appendix ablation.**  The same four columns over our own variants: what the
    Randers 1-form buys over a Riemannian metric and over plain Euclidean geodesics,
    and what the anisotropic mobility $M_t$ buys over an isotropic bridge.

Naming: ``ffm`` / ``pathb`` are the *implementation* names.  In the paper both are
**FFM** -- one method, the SDE off and on -- and they are told apart by a qualifier,
``deterministic`` against ``with noise``.  :data:`PATHB_NAME`, :data:`QUAL_DET` and
:data:`QUAL_NOISY` are the single place that decides all of it; :data:`FFM_LABEL` /
:data:`PATHB_LABEL` are the inline forms for tables and :func:`two_line` the stacked
form for figure headings.  ``scripts.core.labels.ARM_LABELS`` spells the same strings
again (the engine may not import an experiment) and ``verify`` pins that the two agree.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from scripts.core.arms import NOISY_ARMS
from scripts.core.labels import ARM_LABELS
from scripts.core.paths import rel

#: What *both* of our arms are called in the paper -- re-exported, not defined.  Path B is
#: not a second method: it is the same geometry with the SDE switched on, so the two share
#: a name and differ by a qualifier.  The strings live in :mod:`scripts.core.present`
#: because the figure layer and the other benchmarks need them too, and neither may import
#: this module; they are importable from here because that is where they were first
#: written and every caller in this package still spells it that way.
from scripts.core.present import (      # noqa: F401
    FFM_LABEL,
    PATHB_LABEL,
    PATHB_NAME,
    QUAL_DET,
    QUAL_NOISY,
    two_line,
)

from .datasets import DATASET_NAMES
from .paper_report import load_records
from .paper_run import DEFAULT_OUT, SPLIT_SEED
from .report import _agg, _best_mask, _fmt, _index, _md_table, _tex_table

#: the four accuracy columns, in the order they are printed.  All lower-is-better,
#: which is why the bold mask below needs no per-column direction flag.
FOCUS_COLS = (
    (("intermediate", "W2"), r"mid $W_2\downarrow$"),
    (("endpoint", "W2"), r"end $W_2\downarrow$"),
    (("intermediate", "MMD"), r"mid MMD$\downarrow$"),
    (("endpoint", "MMD"), r"end MMD$\downarrow$"),
)

#: F1.  Baselines first, ours last within each block.
MAIN_BLOCKS = (
    (r"Deterministic ($\sigma = 0$)", ("cfm", "mfm_land", "curly", "ffm")),
    (r"Stochastic ($\sigma > 0$)", ("pathb",)),
)
#: the deterministic/stochastic split is not hand-maintained here -- it has to agree
#: with the flag the training code actually branches on, or the caption lies
assert set(MAIN_BLOCKS[1][1]) == set(NOISY_ARMS), (MAIN_BLOCKS[1][1], NOISY_ARMS)
assert not set(MAIN_BLOCKS[0][1]) & set(NOISY_ARMS), MAIN_BLOCKS[0][1]

MAIN_LABELS = {
    "cfm": "OT-CFM",
    "mfm_land": "MFM (LAND)",
    "ffm": f"{FFM_LABEL} (ours)",
    "curly": "Curly-FM",
    "pathb": f"{PATHB_LABEL} (ours)",
}
#: the same rows as *figure column headings*: identical words, qualifier stacked under
#: the name.  A separate dict rather than a flag on the one above, because that one also
#: feeds ``_tex_table`` and a ``\n`` there is a syntax error, not a line break.
MAIN_HEADINGS = {**MAIN_LABELS,
                 "ffm": two_line(QUAL_DET, ours=True),
                 "pathb": two_line(QUAL_NOISY, ours=True)}

#: F2.  Row labels name the *ablated component*, not the arm, so the table reads as a
#: knock-out study rather than as five unrelated models.
ABLATION_BLOCKS = (
    (f"{FFM_LABEL}: geometry the interpolant minimises", ("ffm", "ffm_riem", "ffm_eucl")),
    (f"{PATHB_LABEL}: geometry that drives the bridge", ("pathb", "pathb_iso")),
)
ABLATION_LABELS = {
    "ffm": r"Randers $F = \sqrt{v^\top G v} + c\,\beta^\top v$ (full)",
    "ffm_riem": r"Riemannian, drop the 1-form ($c = 0$)",
    "ffm_eucl": r"Euclidean, drop the metric ($G \equiv I$)",
    "pathb": r"Anisotropic mobility $M_t = \rho I + \Sigma$ (full)",
    "pathb_iso": r"Isotropic mobility ($M_t \equiv I$)",
}
#: same rows, names short enough to survive a three-cloud table
ABLATION_LABELS_SHORT = {
    "ffm": r"Randers (full)",
    "ffm_riem": r"Riemannian ($c = 0$)",
    "ffm_eucl": r"Euclidean ($G \equiv I$)",
    "pathb": r"Anisotropic $M_t$ (full)",
    "pathb_iso": r"Isotropic ($M_t \equiv I$)",
}


# --------------------------------------------------------------------------- #
#  Grid
# --------------------------------------------------------------------------- #
def focus_grid(idx, datasets, arms) -> dict:
    """Rows of ``mid W2 / end W2 / mid MMD / end MMD`` per cloud, plus the floor row.

    Returns formatted strings and the raw means side by side; the means exist only so
    the caller can bold a winner without re-parsing ``"0.162±0.006"``.
    """
    rows, raw = [], []
    for arm in arms:
        row, rraw = [], []
        for ds in datasets:
            cells = idx.get((ds, arm), [])
            for path, _hdr in FOCUS_COLS:
                if not cells:
                    row.append("--")
                    rraw.append(np.nan)
                    continue
                mean, sd, n = _agg(cells, *path)
                row.append(_fmt(mean, sd, n, 3))
                rraw.append(mean)
        rows.append(row)
        raw.append(rraw)

    # the floor is a property of the cloud, not of the arm: any populated cell carries
    # the same value, so take the first one that exists
    floor = []
    for ds in datasets:
        cells = next((idx[(ds, a)] for a in arms if idx.get((ds, a))), None)
        for path, _hdr in FOCUS_COLS:
            floor.append("--" if cells is None
                         else _fmt(*_agg(cells, "floors", *path)))

    raw = np.array(raw, float)
    assert raw.shape == (len(arms), len(datasets) * len(FOCUS_COLS)), raw.shape
    assert len(floor) == raw.shape[1], (len(floor), raw.shape)
    return {"rows": rows, "raw": raw, "floor": floor}


def block_best(raw: np.ndarray, sizes) -> np.ndarray:
    """Column winner *within each block*, not across the whole table.

    A deterministic ODE and a stochastic SDE are not competing for the same prize, and
    a single bold spanning both blocks would silently declare that they are.
    """
    mask = np.zeros(raw.shape, bool)
    start = 0
    for n in sizes:
        sl = slice(start, start + n)
        if n:
            mask[sl] = _best_mask(raw[sl], lower_is_better=True)
        start += n
    assert start == raw.shape[0], (start, raw.shape)
    return mask


# --------------------------------------------------------------------------- #
#  Assembly
# --------------------------------------------------------------------------- #
def _present(idx, blocks, datasets):
    """Drop arms with no records, and blocks left empty, so a partial grid still renders."""
    out = []
    for heading, arms in blocks:
        keep = tuple(a for a in arms if any(idx.get((d, a)) for d in datasets))
        if keep:
            out.append((heading, keep))
    return out


def _table_pair(idx, datasets, blocks, labels, label, floor=True):
    """One table rendered twice -- LaTeX for the paper, markdown for reading here."""
    arms = [a for _h, group in blocks for a in group]
    sizes = [len(group) for _h, group in blocks]
    grid = focus_grid(idx, datasets, arms)
    body = [labels.get(a, ARM_LABELS.get(a, a)) for a in arms]
    hdr = [(d, [h for _p, h in FOCUS_COLS]) for d in datasets]
    md_blocks = [(h, len(g)) for h, g in blocks]
    fl = grid["floor"] if floor else None
    tex = _tex_table(label, hdr, grid["rows"], body, floor=fl,
                     best=block_best(grid["raw"], sizes), blocks=md_blocks,
                     resize=len(datasets) > 1)
    md = _md_table(hdr, None, grid["rows"], body, floor=fl, blocks=md_blocks)
    return tex, md


def build_focus(root: str = DEFAULT_OUT,
                datasets=("Sheet",), stem: str = "focus") -> str:
    """Write ``<root>/paper/focus/<stem>.{tex,md}`` and return the tex."""
    recs = load_records(root)
    if not recs:
        raise SystemExit(f"no records under {root}/runs")
    idx = _index(recs)
    datasets = [d for d in datasets if any(k[0] == d for k in idx)]
    assert datasets, "none of the requested clouds have records"
    seeds = sorted({r["spec"]["seed"] for r in recs})

    # The legend lives in the markdown only: the .tex is pasted into the manuscript,
    # and a caption written here would be a caption the paper did not write.
    note = (f"Test split of a fixed 70/15/15 partition (split seed {SPLIT_SEED}); "
            f"mean ± s.d. over {len(seeds)} model seeds. "
            "*mid* scores the withheld region the model never saw, *end* the target "
            "marginal it was trained to hit. Lower is better throughout; the sampling "
            "floor is the distance between a draw from the reference cloud and the rest "
            "of it, sized to the cloud the arm itself pushes and averaged over draws, "
            "and is the value no method can beat.")

    main_tex, main_md = _table_pair(
        idx, datasets, _present(idx, MAIN_BLOCKS, datasets), MAIN_LABELS,
        f"tab:{stem}-methods")

    abl_tex, abl_md = _table_pair(
        idx, datasets, _present(idx, ABLATION_BLOCKS, datasets),
        # the full row names describe the ablation but only fit a one-cloud table
        ABLATION_LABELS if len(datasets) == 1 else ABLATION_LABELS_SHORT,
        f"tab:{stem}-ablation")

    out_dir = os.path.join(root, "paper", "focus")
    os.makedirs(out_dir, exist_ok=True)
    # \toprule/\midrule come from booktabs; the wide variant wraps in \resizebox
    preamble = ("% \\usepackage{booktabs}"
                + ("  % \\usepackage{graphicx}  (for \\resizebox)\n"
                   if len(datasets) > 1 else "\n"))
    tex = preamble + main_tex + "\n" + abl_tex
    md = ("# Submission tables — test split "
          f"({', '.join(datasets)})\n\n" + note + "\n\n"
          f"## F1 Main text — method comparison  `tab:{stem}-methods`\n\n"
          "Deterministic methods integrate an ODE, stochastic methods an SDE; the "
          "winner is bolded within each block, since the two are not the same model "
          f"class. The two {PATHB_NAME} rows are one method: *{QUAL_NOISY}* is our "
          f"Finsler Schrödinger bridge and *{QUAL_DET}* is the same geometry with the "
          "SDE switched off.\n\n" + main_md
          + f"\n## F2 Appendix — geometry ablation  `tab:{stem}-ablation`\n\n"
          "Each row removes one piece of the data-derived geometry and changes nothing "
          "else: the same interpolant, the same coupling, the same schedule and the "
          "same tuning budget. Winners are bolded within each family.\n\n" + abl_md)
    with open(os.path.join(out_dir, f"{stem}.tex"), "w") as fh:
        fh.write(tex)
    with open(os.path.join(out_dir, f"{stem}.md"), "w") as fh:
        fh.write(md)
    print(f"  wrote {rel(out_dir)}/{stem}.tex")
    print(f"  wrote {rel(out_dir)}/{stem}.md")
    return tex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_OUT)
    ap.add_argument("--datasets", nargs="+", default=["Sheet"], choices=DATASET_NAMES)
    ap.add_argument("--stem", default="focus", help="output file stem")
    ap.add_argument("--print", action="store_true", dest="show")
    args = ap.parse_args()
    tex = build_focus(args.root, tuple(args.datasets), args.stem)
    if args.show:
        print(tex)


if __name__ == "__main__":
    main()
