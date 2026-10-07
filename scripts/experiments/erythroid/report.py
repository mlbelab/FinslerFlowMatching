"""E1 / E2: the erythroid table in four blocks and the sweep it came from.

**E1** is the evidence table: one sub-table per dimension, one row per arm, grouped into
the four blocks of :mod:`~scripts.experiments.erythroid.train`.  Columns are
``Cos. Dist`` / ``L2`` / ``W1`` in their Table 4 order, plus the true ``W2`` beside
their column rather than replacing it, since their block cannot be recomputed.

**E2** is the selection grid, printed in full with the objective's two terms beside their
average at every point, a mark on the argmin, and every grid point where the objective
ranked differently from the reported metrics.

The objective is :mod:`scripts.core.selection`'s one rule.  Every reported number is
measured on the disjoint 90 % of the withheld marginal.
"""
from __future__ import annotations

import os

import numpy as np

from scripts.core import sensitivity as sens
from scripts.core.paths import data_dir
from scripts.core.selection import OBJECTIVE_TERMS

from .datasets import DIMS
from .train import (ARM_BLOCK, ARM_LABELS, ARMS, BLOCK_PAIRS, INHERIT, PUBLISHED,
                  PUBLISHED_METHODS, SELECTION_LABEL, TEST_DIR, _override_tag,
                  anchor_point, grid_points, grid_spec, load_records, selection_score)

#: the four blocks of E1, in report order.
BLOCKS = ("published", "release", "harness", "ours")
BLOCK_HEADS = {
    "published": "published   Petrovic et al. (2025) Table 4 -- their code, their data, "
                 "transcribed",
    "release":   "release     their training code at their published hyper-parameters, "
                 "on OUR cache",
    "harness":   "harness     baselines through our trainer at their published budget",
    "ours":      "ours        Path A / Path B through the same trainer and budget",
}

#: the reported statistics, in their Table 4 row order, plus ``w2_true``: the same
#: distance at ``power=2``, added rather than substituted because their block was
#: measured at ``power=1`` and we do not have their samples.
METRICS = ("cos_dist", "l2", "w2", "w2_true")
METRIC_LABELS = {"cos_dist": "Cos. Dist", "l2": "L2",
                 "w2": 'W1 (their "W2")', "w2_true": "W2 (true)"}
#: their table prints the d > 2 L2 row divided by 10^3.
L2_SCALE = {2: 1.0, 20: 1e3, 50: 1e3}


def _agg(records: list[dict], key: str) -> tuple[float, float] | None:
    """Mean ± std over seeds, or ``None`` if a record predates the statistic."""
    if any(key not in r["metrics"] for r in records):
        return None
    vals = np.array([r["metrics"][key] for r in records], dtype=float)
    return float(vals.mean()), float(vals.std())


def collect(root: str, dims, arms) -> dict:
    """Test records grouped by (dim, arm), plus the tuned override each came from."""
    recs = load_records(os.path.join(root, TEST_DIR))
    out: dict = {}
    for dim in dims:
        for arm in arms:
            cells = [r for r in recs if r["dim"] == dim and r["arm"] == arm]
            if cells:
                out[(dim, arm)] = cells
    return out


def _fmt(mean: float, std: float, scale: float = 1.0, width: int = 16) -> str:
    """Fixed-point where it fits, scientific where it does not."""
    m, s = mean / scale, std / scale
    prec = 3 if abs(m) < 100 else 1
    out = f"{m:.{prec}f} ± {s:.{prec}f}"
    if len(out) > width - 1:
        out = f"{m:.3g} ± {s:.2g}"
    return out


#: width of the arm-name column; must clear the longest entry of
#: :data:`scripts.experiments.erythroid.train.ARM_LABELS`, asserted in
#: the verification suite in the research repository.
ARM_W = 27
#: width of one metric column.
CELL_W = 18


def _metric_head(dim: int) -> str:
    """Column headings for one dimension, with their x10^3 note where it applies."""
    scale = L2_SCALE.get(dim, 1.0)
    out = ""
    for metric in METRICS:
        label = METRIC_LABELS[metric]
        if metric == "l2" and scale != 1.0:
            label += " (x10^3)"
        out += f"{label:>{CELL_W}}"
    return out


def _cell(value, metric: str, dim: int) -> str:
    """One ``mean ± std`` cell, or a marker saying *why* there is no number."""
    scale = L2_SCALE.get(dim, 1.0) if metric == "l2" else 1.0
    if value is None:
        return f"{'--':>{CELL_W}}"
    return f"{_fmt(value[0], value[1], scale, CELL_W):>{CELL_W}}"


def _arm_row(dim: int, arm: str, got: dict) -> str:
    """One measured arm's four metrics."""
    cells = got.get((dim, arm))
    row = f"  {ARM_LABELS[arm]:<{ARM_W}}"
    for metric in METRICS:
        if not cells:
            row += f"{'(not run)':>{CELL_W}}"
            continue
        agg = _agg(cells, metric)
        row += f"{'(backfill)':>{CELL_W}}" if agg is None else _cell(agg, metric, dim)
    return row


def table_e1(root: str, dims, arms) -> str:
    """Their Table 4's four statistics, one sub-table per dimension, four blocks deep."""
    got = collect(root, dims, arms)

    lines = ["", "E1  Erythroid: their Table 4 statistics, in four blocks", ""]

    for dim in dims:
        head = f"  {'d = ' + str(dim):<{ARM_W}}" + _metric_head(dim)
        lines += [head, "-" * len(head)]
        for block in BLOCKS:
            block_arms = [a for a in arms if ARM_BLOCK.get(a) == block]
            if block == "published":
                lines.append(f"[{BLOCK_HEADS[block]}]")
                for method in PUBLISHED_METHODS:
                    row = f"  {method:<{ARM_W}}"
                    for metric in METRICS:
                        row += _cell((PUBLISHED.get((dim, method)) or {}).get(metric),
                                     metric, dim)
                    lines.append(row)
                continue
            if not block_arms:
                continue
            lines.append(f"[{BLOCK_HEADS[block]}]")
            for arm in block_arms:
                lines.append(_arm_row(dim, arm, got))
            if block == "harness":
                missing = [ARM_LABELS[a] for a in block_arms
                           if BLOCK_PAIRS.get(a, "sentinel") is None]
                if missing:
                    lines.append(f"    ({', '.join(missing)}: no release cell -- the MFM "
                                 f"release ships no erythroid config)")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  E2 — the grid
# --------------------------------------------------------------------------- #
def _point_scores(cells) -> dict[str, float] | None:
    """Mean selection score per grid point, or ``None`` if the records predate the key."""
    by_point: dict[str, list] = {}
    for r in cells:
        by_point.setdefault(_override_tag(r["override"]), []).append(r)
    try:
        return {tag: float(np.mean([selection_score(r) for r in rs]))
                for tag, rs in by_point.items()}
    except AssertionError:
        return None


def _point_terms(cells, key: str) -> dict[str, float]:
    """Mean of one stored objective *term* per grid point, for the breakdown columns."""
    by_point: dict[str, list] = {}
    for r in cells:
        if key in r["metrics"]:
            by_point.setdefault(_override_tag(r["override"]), []).append(
                float(r["metrics"][key]))
    return {tag: float(np.mean(vs)) for tag, vs in by_point.items()}


def table_e2(root: str, dims, arms) -> str:
    """The full selection grid, the objective's two terms beside it, and disagreements."""
    sweep = load_records(os.path.join(root, "sweep")) or load_records(data_dir("erythroid_sweep"))
    test = load_records(os.path.join(root, TEST_DIR))
    lines = ["", f"E2  Selection grid (ranked on {SELECTION_LABEL})",
             "    the two term columns are what the objective averages; they rank nothing",
             "    on their own", ""]

    for dim in dims:
        for arm in arms:
            cells = [r for r in sweep if r["dim"] == dim and r["arm"] == arm]
            if not cells:
                continue
            scores = _point_scores(cells)
            if scores is None:
                lines += [f"  d = {dim}  {ARM_LABELS[arm]}",
                          "      (records predate the objective; re-run "
                          "`backfill --stage sweep`, or `sweep --force`)", ""]
                continue
            terms = {k: _point_terms(cells, k) for k in OBJECTIVE_TERMS}
            best = min(scores, key=scores.get)
            ranks = arm not in INHERIT
            head = (f"  d = {dim}  {ARM_LABELS[arm]}" if ranks else
                    f"  d = {dim}  {ARM_LABELS[arm]}  (inherits "
                    f"{ARM_LABELS[INHERIT[arm]]}'s point; this grid ranks nothing)")
            lines.append(head)
            lines.append(f"      {'point':<30} {'objective':>14}" + "".join(
                f"   {k:>14}" for k in OBJECTIVE_TERMS))
            for tag in sorted(scores, key=scores.get):
                mark = " <- picked" if (ranks and tag == best) else ""
                row = f"      {tag:<30} {scores[tag]:14.4f}"
                for k in OBJECTIVE_TERMS:
                    v = terms[k].get(tag)
                    row += f"   {'--':>14}" if v is None else f"   {v:14.4f}"
                lines.append(f"{row}{mark}")
            if ranks:
                lines += _search_gain(arm, dim, scores, best)
                edge = _edge_warning(arm, dim, best)
                if edge:
                    lines.append(f"      {edge}")
            elif best != _override_tag(_inherited_point(sweep, dim, arm)):
                lines.append(f"      (its own argmin would have been {best}; the "
                             "reported column does not use it)")
            lines.append("")

    lines += ["  There is no published grid to match: the reference hard-codes its",
              "  erythroid hyper-parameters, so this grid is ours.",
              "  A single cell is noisy -- GPU training does not reproduce at a fixed seed",
              "  -- so the grid resolves nothing finer than the five-seed std E1 prints.",
              "  A swept cell trains on the 85% split of the kept cells with P and the",
              "  geometry rebuilt on it; the chosen point is refit on all kept cells for",
              "  the five reported seeds.",
              "  Eight rungs per live axis for every arm, searched per dimension.  Ours",
              "  turns two (rho_mult, lam_mult) so it gets 8 x 8; each baseline has exactly",
              "  one live axis and gets 8.  Every arm's published-or-default value is in",
              "  its grid, which is the `no-search rung` line below each ladder.",
              "  The lambda ladder runs 0.01 .. 30, a slid copy of the Pancreas spacing:",
              "  below that the Randers margin lambda^2/(a^2 + lambda^2) is a few ulp of",
              "  float32 on this cloud and F vanishes on a direction."]
    if test:
        lines += _disagreements(sweep, test, dims, arms)
    return "\n".join(lines)


def _search_gain(arm: str, dim: int, scores: dict[str, float], best: str) -> list[str]:
    """What the grid bought this arm: its selected score against its no-search rung."""
    anchor = _override_tag(anchor_point(arm, dim))
    curve = np.array(sorted(scores.values()), dtype=float)
    out = [f"      spread: min {curve[0]:.4f}  median {float(np.median(curve)):.4f}  "
           f"max {curve[-1]:.4f}"]
    if anchor in scores:
        gain = scores[anchor] - scores[best]
        out.append(f"      no-search rung {anchor} scores {scores[anchor]:.4f}; the "
                   f"search bought {gain:+.4f}")
    else:
        out.append(f"      (no-search rung {anchor} not run; search gain not measurable)")
    return out


def _inherited_point(sweep, dim: int, arm: str) -> dict:
    """The override an inheriting arm actually runs at: its parent's argmin."""
    parent = INHERIT[arm]
    cells = [r for r in sweep if r["dim"] == dim and r["arm"] == parent]
    by_point: dict[str, list] = {}
    for r in cells:
        by_point.setdefault(_override_tag(r["override"]), []).append(r)
    if not by_point:
        return {}
    return min(by_point.values(),
               key=lambda rs: float(np.mean([selection_score(r)
                                             for r in rs])))[0]["override"]


def _edge_warning(arm: str, dim: int, best_tag: str) -> str:
    """Flag an argmin sitting on the first or last rung of any axis of its arm's grid.

    Reads ``grid_spec(arm, dim)`` and not ``GRID``, since ``mfm_land``'s knob is chosen
    per dimension.  The point is recovered by reconstructing candidate tags rather than
    by splitting ``best_tag``, which has no unambiguous separator.
    """
    spec = grid_spec(arm, dim)
    if not spec:
        return ""
    hits = [(key, edge) for key, values in spec.items() for edge in (values[0], values[-1])
            if any(_override_tag(p) == best_tag and p.get(key) == edge
                   for p in grid_points(arm, dim))]
    if not hits:
        return ""
    where = ", ".join(f"{key} = {edge:g}" for key, edge in hits)
    return (f"NOTE: argmin on a grid edge ({where}); widen by a half-decade rung and "
            f"re-run before reporting.")


def _disagreements(sweep, test, dims, arms) -> list[str]:
    """Grid points where the proxy's pick is not the pick the reported metric would make."""
    out = ["", "  Selection-vs-reported-metric disagreements:"]
    found = False
    for dim in dims:
        for arm in arms:
            cells = [r for r in test if r["dim"] == dim and r["arm"] == arm]
            tags = {_override_tag(r["override"]) for r in cells}
            if len(tags) < 2:
                continue
            found = True
            by = {}
            for r in cells:
                by.setdefault(_override_tag(r["override"]), []).append(r)
            best_w2 = min(by, key=lambda t: np.mean([r["metrics"]["w2"] for r in by[t]]))
            picked = _override_tag(cells[0]["override"])
            if best_w2 != picked:
                cost = (np.mean([r["metrics"]["w2"] for r in by[picked]])
                        - np.mean([r["metrics"]["w2"] for r in by[best_w2]]))
                out.append(f"    d={dim} {arm}: selection picked {picked}, W1 prefers "
                           f"{best_w2} (cost {cost:+.4f})")
    if not found:
        out.append("    none measurable -- the test stage runs the argmin only, so there is")
        out.append("    no second grid point at test to compare against.  Re-run `test`")
        out.append("    with --force over the full grid to populate this.")
    return out


# --------------------------------------------------------------------------- #
#  E3 — the grid, read as a sensitivity
# --------------------------------------------------------------------------- #
def sensitivity_panels(root: str, dim: int, arms, tuned: dict,
                       n: int = sens.WINDOW) -> tuple[list[dict], float]:
    """Every swept knob walked over ``n`` rungs around its selected value, off the grid.

    Nothing is trained here.  A one-knob window through a selected point is a set of grid
    cells the sweep already ran, so this is E2 re-read as the question a reader of the
    table actually has -- how much of the reported number is the knob -- rather than a
    second experiment.  Its metric is therefore the selection objective on the validation
    slice and at sweep budget, not the reported test metric.

    Our arm turns two axes, and it gets one block per axis: ``rho_mult`` walked at the
    selected ``lam_mult`` and ``lam_mult`` walked at the selected ``rho_mult``, which are
    two orthogonal slices through the one selected point rather than a path across the
    plane.

    Returns the panels :func:`scripts.core.sensitivity.table` tabulates and the seed spread
    to read them against: the largest per cent std over sweep seeds at any selected point.
    """
    sweep = load_records(os.path.join(root, "sweep")) or load_records(data_dir("erythroid_sweep"))
    panels, noise = [], 0.0
    for arm in arms:
        spec, point = grid_spec(arm, dim), tuned.get(arm) or {}
        cells = [r for r in sweep if r["dim"] == dim and r["arm"] == arm]
        if not spec or not point or not cells:
            continue
        by_point: dict[str, list] = {}
        for r in cells:
            by_point.setdefault(_override_tag(r["override"]), []).append(r)
        scores = {t: float(np.mean([selection_score(r) for r in rs]))
                  for t, rs in by_point.items()}
        here = _override_tag(point)
        assert here in scores, f"d={dim} {arm}: no sweep cell at the selected point {here}"
        at = [selection_score(r) for r in by_point[here]]
        noise = max(noise, float(np.std(at) / np.mean(at) * 100.0) if len(at) > 1 else 0.0)
        for knob, ladder in spec.items():
            walk = [(v, scores.get(_override_tag({**point, knob: v})))
                    for v in sens.window(ladder, point[knob], n)]
            panels.append({"model": ARM_LABELS[arm], "knob": knob,
                           "tuned": float(point[knob]),
                           "values": [v for v, s in walk if s is not None],
                           "scores": [s for _, s in walk if s is not None]})
    return panels, noise
