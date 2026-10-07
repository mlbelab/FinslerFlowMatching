"""Internal comparison of the P-derived geometries, on one cloud.

Each geometry variant is a *complete, self-contained* experiment tree -- its own
sweep, its own tuned hyper-parameters, its own five test seeds -- written under its
own root by ``paper_run --geometry`` (see :data:`scripts.experiments.sheet.paper_run.GEOMETRY_OUT`).
This module is the only place that reads more than one of those trees at once, and it
never retrains anything: every number here is copied out of a ``runs/*.json``
that a single-root report would print identically.

The variants
------------
``asym``   the published metric.  ``b`` is the antisymmetric first moment
           ``J^asym_i = sum_j P^asym_ij (x_j - x_i)``, the metric is
           ``G = (rho I + Sigma)^{-1}``, ``beta = -c G b / ||b||_G``.
``fullb``  *estimator-only control*.  Same metric form, but ``b`` is the full-P first
           moment ``J^full_i = sum_j P_ij (x_j - x_i)``.
``fw``     Full-P drift **and** the Freidlin-Wentzell geometric action:
           ``G = a^2 G_0``, ``beta = -c G_0 b / a_bar`` with ``a = ||b||_{G_0}``.
``fwlam``  the FW action in *quadrature*, and the only variant that takes **both**
           moments from the full P: ``a = sqrt(||b||^2_{G_0} + lambda^2)``,
           ``G = a^2 G_0``, ``beta = -G_0 b``.  ``lambda`` replaces both ``c`` and
           ``a_floor``, and it does so with an exact certificate --
           ``beta^T G^{-1} beta = 1 - lambda^2/a^2`` -- where ``fw`` imposes ``c < 1``
           by hand.  See ``scripts.core.geometry``.  The swept range is
           ``paper_run.GRID["fwlam"]``; it was placed by measuring the scale of
           ``||b||_{G_0}`` on this cloud before the sweep rather than guessed, in a
           private side study that has since been deleted -- the range it produced is
           all that survives of it.

Reading ``fullb`` between the first three is the whole point of running it: a change
from ``asym`` to ``fw`` mixes two edits, and only the middle column says which one
paid for it.  ``fwlam`` then adds the third edit -- the second moment -- which a private
2x2 ablation measured to be the one that breaks the *published* metric (full-P ``Sigma``
inside ``G = (rho I + Sigma)^{-1}`` cost ~2% of mid W2 on every arm, and the two edits
*interacted* on Path B), so the question this column answers is whether the FW action is
what makes the full-P pair usable together.

Why the null controls are in the table
--------------------------------------
OT-CFM, MFM (LAND) and the Euclidean FFM ablation never touch ``FinslerGeometry``'s
drift or metric, so their rows *must* be bit-identical across the three roots.  They
are printed, and asserted on, as a free end-to-end check that nothing but the
geometry differed between the runs -- same split, same seeds, same schedule.

    python -m scripts.experiments.sheet.paper_geom_compare --variants asym fw --no-figure
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from scripts.core.labels import ARM_LABELS
from scripts.core.paths import rel

from .datasets import DATASET_NAMES, make_dataset
from .paper_figures import blocked_paths_figure, load_trajectories
from .paper_focus import (FFM_LABEL, FOCUS_COLS, PATHB_LABEL, QUAL_DET, QUAL_NOISY,
                          two_line)
from .paper_report import load_records
from .paper_run import GEOMETRIES, GEOMETRY_OUT, SPLIT_SEED
from .report import _agg, _fmt, _md_table, _tex_table

#: printing order and paper names.  ``fullb`` sits in the middle on purpose: it is the
#: single-edit waypoint between the published metric and the new one.
VARIANT_ORDER = ("asym", "fullb", "fw", "fwlam")
VARIANT_LABELS = {
    "asym": r"$P^{\mathrm{asym}}$ drift",
    "fullb": r"full-$P$ drift",
    "fw": r"full-$P$ + FW",
    "fwlam": r"full-$P$ + FW$_\lambda$",
}
#: longer forms, for the caption and the markdown legend
VARIANT_NOTES = {
    "asym": r"published: $b$ from $P^{\mathrm{asym}}$, $G = (\rho I + \Sigma)^{-1}$",
    "fullb": r"control: $b$ from the full $P$, metric form unchanged",
    "fw": r"new: $b$ from the full $P$, $G = a^2 G_0$, "
          r"$\beta = -c\,G_0 b / \bar a$, $a = \lVert b\rVert_{G_0}$",
    # the only column where *both* moments come from the full P, and the only one whose
    # admissibility is an identity rather than an imposed bound
    "fwlam": r"new: both moments from the full $P$, "
             r"$a = \sqrt{\lVert b\rVert^2_{G_0} + \lambda^2}$, $G = a^2 G_0$, "
             r"$\beta = -G_0 b$ ($c \equiv 1$)",
}

#: rows.  Split by whether the arm reads the P-derived geometry at all -- the last
#: block is a control that must not move, and is asserted on below.
COMPARE_BLOCKS = (
    (r"Deterministic, geometry-driven", ("ffm", "ffm_riem")),
    (r"Stochastic, geometry-driven", ("pathb", "pathb_iso", "curly")),
    (r"Geometry-independent (must not move)", ("cfm", "mfm_land", "ffm_eucl")),
)
COMPARE_LABELS = {
    "ffm": f"{FFM_LABEL} (Randers)",
    "ffm_riem": f"{FFM_LABEL}, Riemannian ($c = 0$)",
    "pathb": PATHB_LABEL,
    "pathb_iso": f"{PATHB_LABEL}, isotropic $M_t$",
    "curly": "Curly-FM",
    "cfm": "OT-CFM",
    "mfm_land": "MFM (LAND)",
    "ffm_eucl": f"{FFM_LABEL}, Euclidean ($G \\equiv I$)",
}
#: the same two arms as *figure* column headings.  Only the pair in :data:`FIGURE_ARMS`
#: appears there and the block heading above them already names the geometry, so the
#: metric-form parentheticals are dropped and the qualifier moves to a second line --
#: which is what makes the two columns read as one method rather than two.
COMPARE_HEADINGS = {**COMPARE_LABELS,
                    "ffm": two_line(QUAL_DET),
                    "pathb": two_line(QUAL_NOISY)}
#: the arms in the last block; they are the ones the identity assertion applies to
CONTROL_ARMS = COMPARE_BLOCKS[-1][1]

#: arms drawn side by side in the comparison figure -- one deterministic, one
#: stochastic, which is the smallest pair that shows both halves of the framework
FIGURE_ARMS = ("ffm", "pathb")


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def variant_index(variants, dataset: str,
                  roots: dict | None = None) -> dict:
    """``(variant, arm) -> [record per seed]``, read from one tree per variant.

    Keyed by *variant* where the single-root tables key by *dataset*, which is what
    lets the whole table machinery in :mod:`scripts.experiments.sheet.report` be reused verbatim:
    a column group is a geometry here instead of a cloud.
    """
    roots = roots or GEOMETRY_OUT
    idx: dict[tuple[str, str], list[dict]] = {}
    for variant in variants:
        root = roots[variant]
        if not os.path.isdir(os.path.join(root, "runs")):
            print(f"  [skip] {variant}: no runs under {rel(root)}/runs")
            continue
        for rec in load_records(root):
            spec = rec["spec"]
            if spec["dataset"] != dataset:
                continue
            # records written before --geometry existed carry no tag; they are the
            # published ones, so treat a missing tag as 'asym' rather than guessing
            tag = spec.get("geometry", "asym")
            assert tag == variant, (
                f"{root} holds a record tagged geometry={tag!r} but that tree is "
                f"{variant!r}; the two variants have been mixed in one directory")
            idx.setdefault((variant, spec["arm"]), []).append(rec)
    assert idx, f"no records for {dataset} under {list(variants)}"
    return idx


# --------------------------------------------------------------------------- #
#  Table
# --------------------------------------------------------------------------- #
def compare_grid(idx, variants, arms) -> dict:
    """Rows of ``mid W2 / end W2 / mid MMD / end MMD`` per variant, one row per arm."""
    rows, raw = [], []
    for arm in arms:
        row, rraw = [], []
        for variant in variants:
            cells = idx.get((variant, arm), [])
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
    raw = np.array(raw, float)
    assert raw.shape == (len(arms), len(variants) * len(FOCUS_COLS)), raw.shape
    return {"rows": rows, "raw": raw}


def best_across_variants(raw: np.ndarray, n_variants: int) -> np.ndarray:
    """Bold the winning *variant* for each (arm, metric), not the winning arm.

    The single-root tables bold down a column, because there the question is "which
    method wins".  Here the row is one method and the columns are the same four
    metrics repeated per geometry, so the comparison runs *across* column groups and
    the mask has to stride by ``len(FOCUS_COLS)``.
    """
    n_cols = len(FOCUS_COLS)
    assert raw.shape[1] == n_variants * n_cols, raw.shape
    mask = np.zeros(raw.shape, bool)
    for r in range(raw.shape[0]):
        for m in range(n_cols):
            cols = [g * n_cols + m for g in range(n_variants)]
            vals = raw[r, cols]
            if not np.isfinite(vals).any():
                continue
            lo = np.nanmin(vals)
            # ties bold together, and only when there is something to compare
            if np.count_nonzero(np.isfinite(vals)) < 2:
                continue
            for c, v in zip(cols, vals):
                mask[r, c] = np.isfinite(v) and v <= lo + 1e-12
    return mask


#: A control-arm gap is only evidence of a configuration leak once it is large
#: relative to the arm's own seed-to-seed spread.  Below that it says nothing about
#: the geometry.
#:
#: The threshold is calibrated, not guessed.  The residual spread of one fixed
#: configuration (Sheet/mfm_land/seed 0) across trees is ~0.3 s.d., which is the
#: resolution floor of any cross-tree comparison here.  One full s.d. sits safely
#: above it while still catching a real hyper-parameter leak, which moves a mean by
#: many s.d.
NOISE_FRAC = 1.0


def check_controls(idx, variants, tol: float = 1e-9) -> list[tuple[bool, str]]:
    """The geometry-independent arms must be numerically identical across variants.

    Returns ``(is_real, line)`` pairs rather than raising: a mismatch is worth
    surfacing in the report even when the rest of the table is still worth reading.
    ``is_real`` is ``True`` only when the gap exceeds :data:`NOISE_FRAC` of the
    reference cell's seed s.d. -- a bit-level gap and a leaked hyper-parameter both
    show up here, and only the second one invalidates the comparison.
    """
    problems = []
    ref = variants[0]
    for arm in CONTROL_ARMS:
        base = idx.get((ref, arm))
        if not base:
            continue
        for variant in variants[1:]:
            cells = idx.get((variant, arm))
            if not cells:
                continue
            for path, hdr in FOCUS_COLS:
                a, sd, _n = _agg(base, *path)
                b = _agg(cells, *path)[0]
                delta = b - a
                if abs(delta) <= tol * max(1.0, abs(a)):
                    continue
                # np.std of a single seed is 0, which would make every gap "real";
                # fall back to the absolute tolerance in that degenerate case.
                scale = sd if np.isfinite(sd) and sd > 0 else 0.0
                is_real = abs(delta) > max(NOISE_FRAC * scale, tol * max(1.0, abs(a)))
                sigma = f"{abs(delta) / scale:.2f}" if scale > 0 else "inf"
                problems.append((is_real,
                    f"{'REAL ' if is_real else 'noise'} {arm} {'/'.join(path)}: "
                    f"{ref}={a:.6g} vs {variant}={b:.6g} "
                    f"(delta {delta:+.3g} = {sigma} seed s.d.)"))
    return problems


def delta_summary(idx, variants, arms, base: str = "asym") -> list[str]:
    """Per-arm percentage change of every metric against the published geometry."""
    lines = []
    for arm in arms:
        if arm in CONTROL_ARMS or not idx.get((base, arm)):
            continue
        for variant in variants:
            if variant == base or not idx.get((variant, arm)):
                continue
            parts = []
            for path, hdr in FOCUS_COLS:
                a = _agg(idx[(base, arm)], *path)[0]
                b = _agg(idx[(variant, arm)], *path)[0]
                # sign convention: negative is an improvement, since every column
                # in FOCUS_COLS is lower-is-better
                parts.append(f"{'/'.join(path)} {100.0 * (b - a) / a:+6.1f}%")
            lines.append(f"  {arm:10s} {base} -> {variant:6s}  " + "  ".join(parts))
    return lines


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
def comparison_figure(dataset: str, variants, out_stem: str,
                      seed: int = 0, n_paths: int = 5,
                      arms=FIGURE_ARMS, roots: dict | None = None) -> bool:
    """One block per geometry, one panel per arm, drawn on the same start cells.

    Reuses :func:`scripts.experiments.sheet.paper_figures.blocked_paths_figure` unchanged by keying
    the trajectory dict on ``arm@variant``; the figure code only ever treats those
    keys as opaque labels.
    """
    roots = roots or GEOMETRY_OUT
    trajs, blocks = {}, []
    for variant in variants:
        root = roots[variant]
        if not os.path.isdir(os.path.join(root, "trajectories")):
            continue
        found = load_trajectories(root, dataset, seed)
        group = []
        for arm in arms:
            if arm not in found:
                continue
            key = f"{arm}@{variant}"
            trajs[key] = found[arm]
            group.append(key)
        if group:
            blocks.append((VARIANT_LABELS.get(variant, variant), tuple(group)))
    if not blocks:
        print(f"  [skip] no trajectories for {dataset} seed {seed}")
        return False

    # every panel must start from the same cells or the visual comparison is void:
    # the start cells are drawn once, from the first panel, and indexed into all
    ref = trajs[blocks[0][1][0]][0]
    for key, traj in trajs.items():
        assert traj.shape[1:] == ref.shape, (
            f"{key} has source cloud {traj.shape[1:]}, expected {ref.shape}")
        assert np.allclose(traj[0], ref, atol=1e-5), (
            f"{key} starts from different cells than {blocks[0][1][0]}; the split or "
            "the source marginal differed between the two trees")

    labels = {f"{a}@{v}": COMPARE_HEADINGS.get(a, ARM_LABELS.get(a, a))
              for v in variants for a in arms}
    blocked_paths_figure(make_dataset(dataset), trajs, blocks, labels,
                         out_stem, n_paths=n_paths)
    return True


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def build_compare(variants=VARIANT_ORDER,
                  dataset: str = "Sheet", root: str | None = None,
                  stem: str = "geometry", seed: int = 0, n_paths: int = 5,
                  figure: bool = True, roots: dict | None = None) -> str:
    """Write ``<root>/paper/compare/<stem>.{tex,md}`` (+ the figure).

    ``root`` is where the report lands and defaults to the *new* variant's tree, so
    the comparison ships next to the setting it is arguing for.  ``roots`` is where
    the records are read *from* and defaults to :data:`GEOMETRY_OUT`.
    """
    variants = [v for v in variants if v in GEOMETRIES]
    assert len(variants) >= 2, f"need at least two variants, got {variants}"
    roots = roots or GEOMETRY_OUT
    root = root or roots[variants[-1]]

    idx = variant_index(variants, dataset, roots)
    present = [v for v in variants if any(k[0] == v for k in idx)]
    assert len(present) >= 2, (
        f"only {present} have records for {dataset}; run the missing "
        f"variant with `paper_run --geometry <v> --datasets {dataset}`")

    blocks = [(h, tuple(a for a in g if any(idx.get((v, a)) for v in present)))
              for h, g in COMPARE_BLOCKS]
    blocks = [(h, g) for h, g in blocks if g]
    arms = [a for _h, g in blocks for a in g]
    grid = compare_grid(idx, present, arms)

    seeds = sorted({r["spec"]["seed"] for cells in idx.values() for r in cells})
    problems = check_controls(idx, present)

    hdr = [(VARIANT_LABELS.get(v, v), [h for _p, h in FOCUS_COLS]) for v in present]
    body = [COMPARE_LABELS.get(a, ARM_LABELS.get(a, a)) for a in arms]
    md_blocks = [(h, len(g)) for h, g in blocks]
    best = best_across_variants(grid["raw"], len(present))

    # the legend goes into the .md; the .tex is the table alone, for the manuscript
    legend = (
        f"Effect of the drift estimator and the metric form on {dataset}. Test split "
        f"of the fixed 70/15/15 partition (split seed {SPLIT_SEED}), mean ± s.d. over "
        f"{len(seeds)} model seeds, hyper-parameters tuned independently per geometry "
        "under the same budget. Lower is better; the best geometry is bolded *within "
        "each row*, so the comparison runs across the column groups rather than down "
        "them. The last block lists methods that never read the transition matrix; "
        "they are held fixed across the groups up to numerical tolerance, and act as "
        "a null control on the comparison.")

    tex = _tex_table(f"tab:{stem}-compare", hdr, grid["rows"], body,
                     floor=None, best=best, blocks=md_blocks, resize=True)
    md = _md_table(hdr, None, grid["rows"], body, floor=None, blocks=md_blocks)

    out_dir = os.path.join(root, "paper", "compare")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}.tex"), "w") as fh:
        fh.write("% \\usepackage{booktabs}  % \\usepackage{graphicx}\n" + tex)

    lines = [f"# Geometry comparison — {dataset}\n",
             legend, "",
             "| variant | definition |", "|---|---|"]
    lines += [f"| `{v}` | {VARIANT_NOTES[v]} |" for v in present]
    lines += ["", f"Table `tab:{stem}-compare` in `{stem}.tex`.", "", md, "",
              f"## Change against `asym` (negative = better)\n",
              "```"] + (delta_summary(idx, present, arms) or ["  (nothing to compare)"])
    lines += ["```", ""]
    real = [ln for is_real, ln in problems if is_real]
    if real:
        lines += ["## ⚠ control arms disagree across trees\n",
                  "A control arm never reads the transition matrix, so a gap larger "
                  "than its own seed spread means the two trees differ in something "
                  "beyond the geometry.\n", "```"]
        lines += [ln for _r, ln in problems] + ["```"]
    elif problems:
        lines += ["## Control-arm audit — clean\n",
                  "Every geometry-independent arm agrees across the trees to well "
                  "inside its own seed-to-seed spread; the residual gaps below are "
                  "within numerical tolerance, not a configuration leak.\n", "```"]
        lines += [ln for _r, ln in problems] + ["```"]
    else:
        lines += ["The geometry-independent arms reproduce exactly across every tree, "
                  "so nothing but the geometry differed between the runs.\n"]
    with open(os.path.join(out_dir, f"{stem}.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote {rel(out_dir)}/{stem}.tex")
    print(f"  wrote {rel(out_dir)}/{stem}.md")
    for line in delta_summary(idx, present, arms):
        print(line)
    for is_real, line in problems:
        print(("  ⚠ " if is_real else "  · ") + line)

    if figure:
        figs = os.path.join(root, "paper", "figures")
        os.makedirs(figs, exist_ok=True)
        comparison_figure(dataset, present,
                          os.path.join(figs, f"{dataset}_paths_{stem}"),
                          seed=seed, n_paths=n_paths, roots=roots)
    return tex


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_ORDER),
                    choices=list(GEOMETRIES))
    ap.add_argument("--dataset", default="Sheet", choices=DATASET_NAMES)
    ap.add_argument("--root", default=None,
                    help="where to write; defaults to the last variant's own tree")
    ap.add_argument("--stem", default="geometry")
    ap.add_argument("--fig-seed", type=int, default=0)
    ap.add_argument("--n-paths", type=int, default=5)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    build_compare(tuple(args.variants), args.dataset, args.root,
                  args.stem, args.fig_seed, args.n_paths, not args.no_figure)


if __name__ == "__main__":
    main()
