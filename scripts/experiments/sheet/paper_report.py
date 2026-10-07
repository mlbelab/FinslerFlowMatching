"""Submission tables and figures from the 70/15/15 test runs.

Reads ``outputs/route_bench_paper/runs/*.json`` and never retrains anything, so a
partial grid yields a correspondingly partial report.

Every number here is on the **test** split.  Val numbers live in the same records under
``metrics_val`` and are deliberately not tabulated: they are what the hyper-parameters
were chosen on, so reporting them would be reporting the selection.

What comes out (all under ``<root>/paper/``)
--------------------------------------------
``tables.tex`` / ``tables.md``
    T1  **Main.**  Intermediate W2 (the withheld region — the actual test),
        endpoint W2 (the marginal the arm was trained to hit), and the share of
        intermediate mass on the correct route.
    T2  **Route breakdown.**  gap / decoy / off-manifold, in percent.
    T3  **Drift fidelity.**  cos(v_theta, b_0) and the relative residual after the
        best global rescaling.
    T4  **Wall clock.**  Mean seconds to fit one model, per cloud and arm.
    T5  **MMD.**  The distances of T1 under an RBF-MMD instead of W2 (appendix).
    P   **Protocol.**  Split sizes, reference sizes, and the selected hyper-parameters.

``figures/<cloud>_paths.{pdf,png}``
    Five transported trajectories per method over the faded cloud.  Five, not forty:
    the question these panels answer is *which way did it go*, and a bundle of forty
    obscures exactly that.  Titles carry the method name and nothing else — the numbers
    are in the tables.

``figures/summary_intermediate_w2.{pdf,png}``
    The T1 intermediate column as grouped bars with the per-cloud sampling floor drawn
    as the dashed line no method can pass.

Style is inherited from :mod:`scripts.experiments.sheet.report` (same region colours, same arm
labels) so these figures sit beside the existing ones without a visual seam.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

from scripts.core.arms import MAIN_ARMS                # noqa: E402
from scripts.core.labels import ARM_COLOURS, ARM_LABELS  # noqa: E402
from scripts.core.paths import rel                       # noqa: E402

from .datasets import (                                           # noqa: E402
    DATASET_NAMES,
    DECOY,
    GAP,
    REGION_COLOURS,
    SOURCE,
    TARGET,
    make_dataset,
)
from .paper_run import (                                          # noqa: E402
    DEFAULT_OUT,
    SPLIT_SEED,
    SWEEP_SOURCE,
)
from .protocol import corridor_mask                               # noqa: E402
from .report import _agg, _best_mask, _fmt, _index, _md_table, _plane, _tex_table  # noqa: E402
from .split import make_split                                     # noqa: E402

#: paths drawn per panel.  Five, per the brief: enough to read the route, few enough
#: that the individual curves stay separable.
N_SHOWN_PATHS = 5


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_records(root: str = DEFAULT_OUT) -> list[dict]:
    """Every test record in a run tree."""
    where = os.path.join(root, "runs")
    recs = []
    for path in sorted(glob.glob(os.path.join(where, "*.json"))):
        with open(path) as f:
            recs.append(json.load(f))
    assert recs, f"no run records under {where}"
    return recs


def _timing(cells: list[dict]) -> tuple[float, float, int]:
    v = np.array([float(r["train_s"]) for r in cells], dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


# --------------------------------------------------------------------------- #
#  Tables
# --------------------------------------------------------------------------- #
#: (metric path, column header, decimals, lower-is-better)
MAIN_COLS = (
    (("intermediate", "W2"), r"mid $W_2\downarrow$", 3, True),
    (("endpoint", "W2"), r"end $W_2\downarrow$", 3, True),
    (("route", "intermediate", "gap"), r"route \%$\uparrow$", 1, False),
)


def main_table(idx, datasets, arms) -> dict:
    """T1: the three headline numbers per cloud."""
    rows, raw = [], []
    for arm in arms:
        row, rraw = [], []
        for ds in datasets:
            cells = idx.get((ds, arm), [])
            for path, _hdr, prec, _lo in MAIN_COLS:
                if not cells:
                    row.append("--")
                    rraw.append(np.nan)
                    continue
                m, s, n = _agg(cells, *path)
                if path[0] == "route":                     # a share reads as a percent
                    m, s = 100.0 * m, 100.0 * s
                row.append(_fmt(m, s, n, prec))
                rraw.append(m)
        rows.append(row)
        raw.append(rraw)

    # the floor exists for the two distances only; a route share has no sampling floor
    floor = []
    for ds in datasets:
        cells = next((idx[(ds, a)] for a in arms if idx.get((ds, a))), None)
        floor += ["--" if cells is None else _fmt(*_agg(cells, "floors", b, "W2"))
                  for b in ("intermediate", "endpoint")]
        floor.append("--")
    return {"rows": rows, "raw": np.array(raw, float), "floor": floor}


def mmd_table(idx, datasets, arms) -> dict:
    """T5: the same two distances under an RBF-MMD."""
    rows, raw = [], []
    for arm in arms:
        row, rraw = [], []
        for ds in datasets:
            cells = idx.get((ds, arm), [])
            for block in ("intermediate", "endpoint"):
                if not cells:
                    row.append("--")
                    rraw.append(np.nan)
                    continue
                m, s, n = _agg(cells, block, "MMD")
                row.append(_fmt(m, s, n, 3))
                rraw.append(m)
        rows.append(row)
        raw.append(rraw)
    floor = []
    for ds in datasets:
        cells = next((idx[(ds, a)] for a in arms if idx.get((ds, a))), None)
        floor += ["--" if cells is None else _fmt(*_agg(cells, "floors", b, "MMD"))
                  for b in ("intermediate", "endpoint")]
    return {"rows": rows, "raw": np.array(raw, float), "floor": floor}


def route_table(idx, datasets, arms) -> dict:
    """T2: where the intermediate mass physically ended up, in percent."""
    keys = ("gap", "decoy", "off_manifold")
    rows = []
    for arm in arms:
        row = []
        for ds in datasets:
            cells = idx.get((ds, arm), [])
            for key in keys:
                if not cells:
                    row.append("--")
                    continue
                m, _s, _n = _agg(cells, "route", "intermediate", key)
                row.append(f"{100 * m:.1f}")
        rows.append(row)
    return {"rows": rows, "keys": keys}


def drift_table(idx, datasets, arms) -> dict:
    """T3: alignment of the learned drift with the P-derived flux ``b_0``."""
    rows, raw = [], []
    for arm in arms:
        row, rraw = [], []
        for ds in datasets:
            cells = idx.get((ds, arm), [])
            for key in ("cos", "L2_rel"):
                if not cells:
                    row.append("--")
                    rraw.append(np.nan)
                    continue
                m, s, n = _agg(cells, "drift", key)
                row.append(_fmt(m, s, n, 3))
                rraw.append(m)
        rows.append(row)
        raw.append(rraw)
    return {"rows": rows, "raw": np.array(raw, float)}


#: a seed counts as diverged when its terminal cloud sits this many times further from
#: the target than a same-sized draw from the target sits from the rest of it.  The
#: floor is the only scale on the problem that is not a modelling choice, so the
#: threshold is expressed in units of it rather than as an absolute W2.
DIVERGENCE_MULT = 10.0


def stability_table(idx, datasets, arms, mult: float = DIVERGENCE_MULT) -> dict:
    """T6: median endpoint W2 and how many seeds blew up.

    An arm that diverges on one seed in five reaches an endpoint ``W_2`` an order of
    magnitude above the other four, so the mean and its s.d. describe the outlier rather
    than the method.
    Both facts matter — occasional divergence *is* a property of the arm — so this table
    reports them separately instead of choosing one summary and hiding the other.
    """
    rows, raw = [], []
    for arm in arms:
        row, rraw = [], []
        for ds in datasets:
            cells = idx.get((ds, arm), [])
            if not cells:
                row += ["--", "--"]
                rraw.append(np.nan)
                continue
            vals = np.array([c["metrics"]["endpoint"]["W2"] for c in cells], float)
            thresh = mult * float(np.mean(
                [c["metrics"]["floors"]["endpoint"]["W2"] for c in cells]))
            n_bad = int((vals > thresh).sum())
            row += [f"{np.median(vals):.3f}", f"{n_bad}/{len(vals)}"]
            rraw.append(float(np.median(vals)))
        rows.append(row)
        raw.append(rraw)
    return {"rows": rows, "raw": np.array(raw, float), "mult": mult}


def timing_table(idx, datasets, arms) -> dict:
    """T4: wall-clock seconds to fit one model."""
    rows, raw = [], []
    for arm in arms:
        row, rraw = [], []
        for ds in datasets:
            cells = idx.get((ds, arm), [])
            if not cells:
                row.append("--")
                rraw.append(np.nan)
                continue
            m, s, n = _timing(cells)
            row.append(f"{m:.0f}" if n <= 1 else f"{m:.0f}±{s:.0f}")
            rraw.append(m)
        rows.append(row)
        raw.append(rraw)
    return {"rows": rows, "raw": np.array(raw, float)}


# --------------------------------------------------------------------------- #
#  Protocol block
# --------------------------------------------------------------------------- #
def protocol_block(datasets, root: str) -> str:
    """Split sizes, reference sizes and the tuned point — the reproducibility table."""
    lines = ["| Cloud | d | train | val | test | test $p_1$ | test route ref | hparams from |",
             "|---|---|---|---|---|---|---|---|"]
    for name in datasets:
        ds = make_dataset(name)
        sp = make_split(ds, seed=SPLIT_SEED)
        c = sp.counts()
        m = sp.mask("test")
        n_tgt = int(((ds.region == TARGET) & m).sum())
        n_corr = int((corridor_mask(ds) & m).sum())
        lines.append(f"| {name} | {ds.d} | {c['train']['all']} | {c['val']['all']} | "
                     f"{c['test']['all']} | {n_tgt} | {n_corr} | {SWEEP_SOURCE[name]} |")

    lines.append("\n**Selected hyper-parameters** (chosen on the val split — see "
                 "`scripts.experiments.sheet.split.val_objective`)\n")
    for cloud in sorted(set(SWEEP_SOURCE[d] for d in datasets)):
        path = os.path.join(root, "tuned", f"{cloud}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            tuned = json.load(f)
        lines.append(f"- **{cloud} sweep** (inherited by "
                     f"{', '.join(d for d in datasets if SWEEP_SOURCE[d] == cloud)}): "
                     + "; ".join(f"`{a}` {json.dumps(v)}" for a, v in sorted(tuned.items())))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  -> {rel(stem)}.pdf / .png")


def _spread_starts(starts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Farthest-point sample ``n`` start cells, so five paths cover the source arc.

    A uniform draw of five out of a thousand clusters as often as not, and a clustered
    draw shows five copies of one trajectory rather than the spread of routes the panel
    is meant to show.  Farthest-point sampling is deterministic given its first pick,
    which is the only thing ``seed`` chooses.
    """
    assert starts.ndim == 2 and len(starts) >= n > 0, f"cannot pick {n} of {starts.shape}"
    picks = [int(np.random.default_rng(seed).integers(len(starts)))]
    d = np.linalg.norm(starts - starts[picks[0]], axis=1)
    while len(picks) < n:
        picks.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(starts - starts[picks[-1]], axis=1))
    return np.array(picks, dtype=int)


def paths_figure(ds, trajs: dict[str, np.ndarray], out_stem: str, seed: int = 0,
                 n_paths: int | None = None) -> None:
    """One panel per arm, ``n_paths`` trajectories, method name and nothing else.

    The same start cells are used in every panel (one draw, reused), so a difference
    between panels is a difference between methods and not between the cells they
    happened to be shown.
    """
    n_paths = N_SHOWN_PATHS if n_paths is None else n_paths
    arms = [a for a in MAIN_ARMS if a in trajs]
    assert arms, "no trajectories to plot"
    ncol = min(4, len(arms))
    nrow = int(np.ceil(len(arms) / ncol))

    # Fixed to the cloud's extent: one divergent SDE sample must not rescale a panel
    # until the manifold is a dot.
    pts = _plane(ds, ds.X)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.06 * np.maximum(hi - lo, 1e-9)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
    panel_w = 2.9
    panel_h = float(np.clip(panel_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]),
                            1.0, 4.2)) + 0.42

    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow),
                             squeeze=False, layout="constrained")
    n_src = min(t.shape[1] for t in trajs.values())
    cols = _spread_starts(_plane(ds, trajs[arms[0]][0, :n_src]),
                          min(n_paths, n_src), seed)

    # On the Sheet the gap spans the full width but only |u| <= SHEET_CORRIDOR is
    # scored, so a panel that draws the gap uniformly hides the very thing the route
    # column measures.  Draw the corridor darker wherever it is a strict subset.
    corridor = corridor_mask(ds)
    has_corridor = bool(corridor.any()) and int(corridor.sum()) < int((ds.region == GAP).sum())

    for k, arm in enumerate(arms):
        ax = axes[k // ncol][k % ncol]
        for reg in (SOURCE, TARGET, DECOY, GAP):
            m = ds.region == reg
            if reg == GAP and has_corridor:
                m = m & ~corridor          # the corridor is drawn separately, below
            if not m.any():
                continue
            p = _plane(ds, ds.X[m])
            # the withheld regions are the ones the reader has to locate, so they get
            # a little more ink than the marginals the paths obviously start and end in
            ax.scatter(p[:, 0], p[:, 1], s=1.4,
                       alpha=0.26 if reg in (GAP, DECOY) else 0.14,
                       color=REGION_COLOURS[reg], linewidths=0, zorder=1)
        if has_corridor:
            p = _plane(ds, ds.X[corridor])
            ax.scatter(p[:, 0], p[:, 1], s=1.8, alpha=0.75,
                       color=REGION_COLOURS[GAP], linewidths=0, zorder=2)

        traj = trajs[arm]
        paths = _plane(ds, traj[:, cols].reshape(-1, ds.d)).reshape(
            traj.shape[0], len(cols), 2)
        for j in range(len(cols)):
            ax.plot(paths[:, j, 0], paths[:, j, 1], lw=1.3, alpha=0.9,
                    color=ARM_COLOURS.get(arm, "#222222"), zorder=3,
                    solid_capstyle="round")
        ax.scatter(paths[0, :, 0], paths[0, :, 1], s=16, color="#ffffff",
                   edgecolors="#222222", linewidths=0.7, zorder=4)
        ax.scatter(paths[-1, :, 0], paths[-1, :, 1], s=30, marker="*",
                   color="#ffffff", edgecolors="#222222", linewidths=0.6, zorder=4)

        ax.set_title(ARM_LABELS.get(arm, arm), fontsize=9.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color("#bbbbbb")

    for k in range(len(arms), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    handles = [
        Line2D([], [], marker="o", ls="", ms=5, color=REGION_COLOURS[SOURCE], label="$p_0$"),
        Line2D([], [], marker="o", ls="", ms=5, color=REGION_COLOURS[TARGET], label="$p_1$"),
        Line2D([], [], marker="o", ls="", ms=5, alpha=0.45,
               color=REGION_COLOURS[GAP], label="gap") if has_corridor else
        Line2D([], [], marker="o", ls="", ms=5, color=REGION_COLOURS[GAP], label="gap"),
    ]
    if has_corridor:
        handles.append(Line2D([], [], marker="o", ls="", ms=5,
                              color=REGION_COLOURS[GAP], label="corridor (scored)"))
    if (ds.region == DECOY).any():
        handles.append(Line2D([], [], marker="o", ls="", ms=5,
                              color=REGION_COLOURS[DECOY], label="decoy"))
    handles.append(Line2D([], [], marker="*", ls="", ms=8, color="#ffffff",
                          markeredgecolor="#222222", label="$t{=}1$"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.4)
    _save(fig, out_stem)


def summary_figure(idx, datasets, arms, out_stem: str) -> None:
    """Grouped bars of the intermediate W2, with each cloud's sampling floor."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.5 * len(datasets), 2.9),
                             squeeze=False, layout="constrained")
    for a, name in enumerate(datasets):
        ax = axes[0][a]
        present = [arm for arm in arms if idx.get((name, arm))]
        means = [_agg(idx[(name, arm)], "intermediate", "W2")[0] for arm in present]
        sds = [_agg(idx[(name, arm)], "intermediate", "W2")[1] for arm in present]
        floor = _agg(idx[(name, present[0])], "floors", "intermediate", "W2")[0]
        x = np.arange(len(present))
        ax.bar(x, means, yerr=sds, capsize=2.5, width=0.72, linewidth=0,
               color=[ARM_COLOURS.get(arm, "#777777") for arm in present],
               error_kw={"lw": 0.8, "ecolor": "#444444"})
        ax.axhline(floor, ls="--", lw=1.0, color="#333333")
        ax.text(len(present) - 0.4, floor, " floor", va="bottom", ha="right", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS.get(arm, arm) for arm in present],
                           rotation=38, ha="right", fontsize=7.5)
        ax.set_title(name, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if a == 0:
            ax.set_ylabel("intermediate $W_2$ $\\downarrow$", fontsize=9)
    _save(fig, out_stem)


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def build_report(root: str = DEFAULT_OUT, datasets=DATASET_NAMES, arms=MAIN_ARMS,
                 fig_seed: int = 0, skip_figures: bool = False) -> str:
    recs = load_records(root)
    idx = _index(recs)
    datasets = [d for d in datasets if any(k[0] == d for k in idx)]
    arms = [a for a in arms if any(k[1] == a for k in idx)]
    labels = [ARM_LABELS.get(a, a) for a in arms]
    seeds = sorted({r["spec"]["seed"] for r in recs})
    device = sorted({r["device"] for r in recs})
    # T4 is only interpretable on one accelerator; say so loudly rather than averaging
    # cells from two different GPU models into a single "seconds" number
    if len(device) > 1:
        print(f"  [warn] wall clocks span {len(device)} devices: {', '.join(device)} "
              f"— Table T4 mixes hardware")

    t1 = main_table(idx, datasets, arms)
    t2 = route_table(idx, datasets, arms)
    t3 = drift_table(idx, datasets, arms)
    t4 = timing_table(idx, datasets, arms)
    t5 = mmd_table(idx, datasets, arms)
    t6 = stability_table(idx, datasets, arms)

    hdr1 = [(d, [h for _p, h, _pr, _lo in MAIN_COLS]) for d in datasets]
    hdr2 = [(d, ["gap", "decoy", "off"]) for d in datasets]
    hdr3 = [(d, [r"$\cos\uparrow$", r"$L_2^{\mathrm{rel}}\downarrow$"]) for d in datasets]
    hdr5 = [(d, ["mid MMD", "end MMD"]) for d in datasets]
    hdr6 = [(d, [r"med. end $W_2$", "diverged"]) for d in datasets]

    # column-wise winners.  The route share is the one column where larger is better,
    # so the mask is assembled per column group rather than for the table as a whole.
    best1 = np.zeros(t1["raw"].shape, bool)
    for c in range(t1["raw"].shape[1]):
        lo = MAIN_COLS[c % len(MAIN_COLS)][3]
        best1[:, c] = _best_mask(t1["raw"][:, [c]], lower_is_better=lo)[:, 0]

    selection = ("Hyper-parameters were selected on the val split by averaging the "
                 "endpoint and intermediate $W_2$, the latter against the val half of "
                 "the withheld region (disjoint from the test half reported here). "
                 "Selection therefore saw route information; training did not.")
    note = (f"Test split of a fixed 70/15/15 partition (split seed {SPLIT_SEED}); "
            f"mean $\\pm$ s.d. over {len(seeds)} model seeds. " + selection)

    # The .tex carries the tables alone.  Every legend below is written into the .md
    # twin instead, so nothing this script knows is lost and nothing it wrote ends up
    # in the manuscript as if the paper had written it.
    tex = "\n".join([
        _tex_table("tab:route-main", hdr1, t1["rows"], labels,
                   floor=t1["floor"], best=best1),
        _tex_table("tab:route-breakdown", hdr2, t2["rows"], labels),
        _tex_table("tab:route-drift", hdr3, t3["rows"], labels,
                   best=_best_mask(t3["raw"], lower_is_better=False)),
        _tex_table("tab:route-time", [(d, ["s"]) for d in datasets], t4["rows"],
                   labels, best=_best_mask(t4["raw"])),
        _tex_table("tab:route-mmd", hdr5, t5["rows"], labels,
                   floor=t5["floor"], best=_best_mask(t5["raw"])),
        # no winner is bolded in T6: "fewest divergences" and "lowest median" can
        # disagree, and a single bold would imply the table picks a side
        _tex_table("tab:route-stability", hdr6, t6["rows"], labels),
    ])

    md = "\n".join([
        f"# Route-selection benchmark — test split\n\n{note}\n",
        "## T1 Main  `tab:route-main`\n",
        "*mid* is the withheld region, *end* the target marginal, *route* the share "
        "of intermediate mass on the correct side. The sampling floor is the distance "
        "between a draw from the reference cloud and the rest of it, sized to the cloud "
        "the arm itself pushes and averaged over draws: no method can go below it.\n",
        _md_table(hdr1, None, t1["rows"], labels, floor=t1["floor"]),
        "\n## T2 Route breakdown (% of intermediate mass)  `tab:route-breakdown`\n",
        "*gap* is the correct route, *decoy* the wrong way round on clouds that have "
        "one, *off* outside the manifold. The remainder is mass still sitting on $p_0$ "
        "or already on $p_1$, so the three columns need not sum to 100.\n",
        _md_table(hdr2, None, t2["rows"], labels),
        "\n## T3 Drift fidelity  `tab:route-drift`\n",
        "Against the $P$-derived flux $b_0$ on the full cloud.\n",
        _md_table(hdr3, None, t3["rows"], labels),
        "\n## T4 Training wall clock (s)  `tab:route-time`\n",
        f"Seconds to fit one model on {'/'.join(device)}: all training phases plus the "
        "single terminal pushforward integration; scoring and I/O excluded.\n",
        _md_table([(d, ["s"]) for d in datasets], None, t4["rows"], labels),
        "\n## T5 RBF-MMD  `tab:route-mmd`\n",
        "The MMD counterpart of T1, same protocol.\n",
        _md_table(hdr5, None, t5["rows"], labels, floor=t5["floor"]),
        f"\n## T6 Endpoint stability  `tab:route-stability`\n",
        "The median over seeds, and how many seeds finished further than "
        f"{t6['mult']:.0f}x the endpoint sampling floor from the target. Occasional "
        "divergence dominates the mean in T1 wherever it happens, so it is separated "
        "out here rather than averaged in.\n",
        _md_table(hdr6, None, t6["rows"], labels),
        "\n## Protocol\n",
        protocol_block(datasets, root),
    ])

    out = os.path.join(root, "paper")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tables.tex"), "w") as f:
        f.write(tex)
    with open(os.path.join(out, "tables.md"), "w") as f:
        f.write(md)
    print(f"tables -> {rel(out)}/tables.tex, {rel(out)}/tables.md")

    if not skip_figures:
        figs = os.path.join(out, "figures")
        os.makedirs(figs, exist_ok=True)
        summary_figure(idx, datasets, arms, os.path.join(figs, "summary_intermediate_w2"))
        for name in datasets:
            trajs = {}
            for arm in arms:
                p = os.path.join(root, "trajectories",
                                 f"{name}_{arm}_s{fig_seed}.npz")
                if os.path.exists(p):
                    trajs[arm] = np.load(p)["traj"]
            if not trajs:
                print(f"  [skip] {name}: no trajectories at seed {fig_seed}")
                continue
            paths_figure(make_dataset(name), trajs,
                         os.path.join(figs, f"{name}_paths"), seed=fig_seed)
    return md


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_OUT)
    p.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    p.add_argument("--arms", nargs="+", default=list(MAIN_ARMS))
    p.add_argument("--fig-seed", type=int, default=0)
    p.add_argument("--n-paths", type=int, default=N_SHOWN_PATHS)
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args(argv)
    globals()["N_SHOWN_PATHS"] = args.n_paths
    build_report(args.root, tuple(args.datasets), tuple(args.arms),
                 fig_seed=args.fig_seed, skip_figures=args.skip_figures)


if __name__ == "__main__":
    main()
