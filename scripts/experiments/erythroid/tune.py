"""How the erythroid hyper-parameters were found — the record, and the thing that found them.

    python -m scripts.experiments.erythroid.tune plan            # the grid and the loop
    python -m scripts.experiments.erythroid.tune plan --count 1  # cells per dimension
    python -m scripts.experiments.erythroid.tune run --dim 20 --index 7   # one job
    python -m scripts.experiments.erythroid.tune collect         # ranks; prints TUNED

Same four verbs as every other tree here.  Three things about this one are its own.

**Searched independently per dimension.**  ``--dim`` is required and the flat cell index
restarts at 0 inside each of ``d = 2, 20, 50``, since ``d = 2`` is a standardised UMAP and
``d = 20/50`` are unstandardised PCA prefixes two orders of magnitude wider.  Nothing
inherits a point across a dimension.

**Every rung runs at every sweep seed.**  A cell is ``(dim, arm, point, seed)``: this tree
does not train reproducibly on GPU — two identical runs of one cell gave ``objective``
0.152 and 0.305, bit-identical on CPU — so ``collect`` averages over the seeds present
before it takes an argmin.

**Two waves.**  Wave 1 searches ``ffm``'s metric grid and the baselines' ladders; wave 2
searches ``pathb``'s bridge width at the metric wave 1 chose, so a ``collect`` between the
two is required.  The grid itself is imported from
:data:`~scripts.experiments.erythroid.train.GRID` and never restated, because it carries
the matched-budget assertions.

Selection is the repository's one objective (:mod:`scripts.core.selection`).  Training
never sees the middle marginal under either stage, and ``P``, its bandwidth and the
geometry are rebuilt on whatever the stage's training set is.

The widening rule
-----------------
An argmin on a ladder endpoint means the ladder was too narrow rather than that a point
was chosen — unless the score has gone flat there, past :data:`EDGE_FLAT_PCT`.  ``collect``
prints the slope at the winner's inward neighbour so the call is made on evidence, and the
rule is applied to the **baselines first**.  Because the rung count is asserted equal
across arms, a widened ladder for one arm is a widened ladder for all of them.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from scripts.core.paths import rel
from scripts.core.selection import OBJECTIVE_KEY, OBJECTIVE_TERMS
from scripts.method import (Fanout, Grid, Run, cell_tag, format_tuned, preflight_cell,
                            print_ladder, rank, require_complete, smoke_root)

from .train import (ARMS, DIMS, INHERIT, INHERIT_METRIC, N_STEPS_ERYTHROID, PINNED, ROOT,
                    SWEEP_TRAJ_DIR, _audit_inputs, _override_tag, _write_json, arm_ladder,
                    grid_points, inherited_metric, load_records, run_cell)

# --------------------------------------------------------------------------- #
#  The search
# --------------------------------------------------------------------------- #
#: the arms that actually cost cells: everything that is neither pinned nor inherited.
SEARCHED = tuple(a for a in ARMS if a not in PINNED and a not in INHERIT)
WAVE1_ARMS = tuple(a for a in SEARCHED if a not in INHERIT_METRIC)
WAVE2_ARMS = tuple(a for a in SEARCHED if a in INHERIT_METRIC)

#: seeds every rung is run at: a rung is an average and not a single number, since this
#: tree does not train reproducibly on GPU.
SWEEP_SEEDS = (0, 1)


def _fanout(arms, dim: int) -> Fanout:
    return Fanout({arm: Grid.of([{**point, "seed": seed}
                                 for point in grid_points(arm, dim)
                                 for seed in SWEEP_SEEDS])
                   for arm in arms})


def wave1(dim: int) -> Fanout:
    """The searched grid at one dimension, as a flat, index-addressable fan-out.

    Built from :func:`~scripts.experiments.erythroid.train.grid_points` rather than from a
    restated ladder, so the cell a scheduler addresses and the point ``collect`` ranks are
    the same object.  The seed is the fastest-varying axis and part of the cell, so
    ``ffm``'s 8 x 8 rungs at two seeds is 128 cells and not 64.
    """
    return _fanout(WAVE1_ARMS, dim)


def wave2(dim: int) -> Fanout:
    """Path B's bridge-width ladder, at the metric wave 1 chose for ``ffm``."""
    return _fanout(WAVE2_ARMS, dim)


WAVES = {1: wave1(DIMS[0]), 2: wave2(DIMS[0])}

#: the dimensions searched independently, checked against ``--dim``.
TUNED_DIMS: tuple[int, ...] = tuple(DIMS)

#: the scalar every cell is ranked on.  Defined once, in :mod:`scripts.core.selection`.
SELECTION_KEY = OBJECTIVE_KEY

#: how flat the score has to be at an endpoint before an argmin on the edge counts as
#: converged rather than as too narrow a ladder.  Against this tree's run-to-run spread an
#: edge caution is a prompt to look, not a measurement.
EDGE_FLAT_PCT = 5.0

#: where ``collect`` writes its answer: beside this file, not in the unversioned run tree,
#: because the tuned point is the literal the notebook pastes and ``verify`` diffs.
TUNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


def waves_for(dim: int) -> dict[int, Fanout]:
    """The wave table at one dimension, rebuilt rather than reused: ``mfm_land`` swaps
    which knob it sweeps at ``d = 2``."""
    return {1: wave1(dim), 2: wave2(dim)}


# --------------------------------------------------------------------------- #
#  One cell
# --------------------------------------------------------------------------- #
def cell_path(root: str, dim: int, wave: int, name: str, point: dict) -> str:
    """Where one cell's record lands: keyed by the point it measures, not by the flat
    index it was submitted at, so widening a ladder cannot alias one arm's record onto
    another arm's cell."""
    return os.path.join(root, "sweep",
                        cell_tag(wave, name, point, dim=dim) + ".json")


def run_one(dim: int, wave: int, index: int, cfg: Run, root: str = ROOT,
            force: bool = False) -> dict:
    """Train and score exactly one grid point at one seed, and write its record.

    ``stage="sweep"`` is the whole difference between this and a reported cell: the arm
    trains on the 85 % split of the kept cells, with ``P`` and the geometry rebuilt on
    *that* subset.  A re-run under ``--force`` is audited on its **inputs** and not on its
    score, which drifts on this GPU stack for reasons that certify nothing.
    """
    arm, point, path = preflight_cell(
        waves_for(dim), wave, index, dim, TUNED_DIMS, root,
        lambda n, p: cell_path(root, dim, wave, n, p), force=force)
    point = dict(point)
    seed = point.pop("seed")

    inherit = None
    if arm in INHERIT_METRIC:
        tuned = collect(root, dims=(dim,), quiet=True, partial=True)
        inherit = inherited_metric(arm, tuned.get(dim, {}))
        assert inherit, (
            f"d={dim} {arm} takes its metric from {INHERIT_METRIC[arm]}; run wave 1 and "
            f"`collect` first")

    prior = None
    if force and os.path.exists(path):
        with open(path) as fh:
            prior = json.load(fh)

    traj = os.path.join(root, SWEEP_TRAJ_DIR, f"d{dim}_w{wave}_i{index:03d}.npz")
    rec = run_cell(dim, arm, seed, point, stage="sweep", device=cfg.device,
                   dtype=cfg.dtype, n_steps=N_STEPS_ERYTHROID, smoke=cfg.smoke,
                   traj_path=traj, inherit=inherit)
    rec["wave"], rec["index"], rec["point"] = wave, index, point

    drift = ""
    if prior is not None:
        _audit_inputs(os.path.basename(path), prior, rec)
        was = prior["metrics"].get(SELECTION_KEY)
        if was is not None:
            drift = f"  (was {was:.4f})"
    _write_json(rec, path)

    m = rec["metrics"]
    print(f"[cell] d{dim} w{wave} i{index:>3}  {arm:<14} {point} s{seed}  "
          f"objective {m[SELECTION_KEY]:.4f}{drift}  (end {m['W2_endpoint']:.4f}, "
          f"mid {m['W2_intermediate']:.4f})  [{rec['train_seconds']:.0f}s] -> {path}")
    return rec


# --------------------------------------------------------------------------- #
#  Collect
# --------------------------------------------------------------------------- #
def collect(root: str = ROOT, dims=TUNED_DIMS, quiet: bool = False,
            partial: bool = False) -> dict:
    """Rank every finished cell and return ``{dim: {arm: override}}``.

    The ranking is :func:`~scripts.experiments.erythroid.train.arm_ladder`, which is also
    what :func:`~scripts.experiments.erythroid.train.tuned_points` calls, so the ladder
    printed here and the point the notebook is handed are one ranking.  A dimension with
    **no** cells is skipped rather than fatal; one that has been started and is
    *incomplete* is fatal unless ``partial``.
    """
    sweep = os.path.join(root, "sweep")
    assert os.path.isdir(sweep), (
        f"no sweep records under {sweep}; run the search first "
        f"(`python -m scripts.experiments.erythroid.tune plan` prints the loop)")
    recs = load_records(sweep)

    tuned: dict = {}
    missing: list = []
    for dim in dims:
        started = {w: any(r["dim"] == dim and r.get("wave") == w for r in recs)
                   for w in waves_for(dim)}
        if not started[1]:
            if not quiet:
                print(f"\n  d{dim}  -- not started ({len(wave1(dim))} cells)")
            continue
        tuned[dim] = {}
        for wave, wave_arms in ((1, WAVE1_ARMS), (2, WAVE2_ARMS)):
            for arm in wave_arms:
                inherit = inherited_metric(arm, tuned[dim])
                if arm in INHERIT_METRIC and not inherit:
                    if not quiet:
                        print(f"\n  d{dim}  {arm}  -- waiting on "
                              f"{INHERIT_METRIC[arm]}'s point")
                    continue
                rows, n_want = arm_ladder(recs, dim, arm, inherit)
                n_have = sum(r["n_seeds"] for r in rows)
                want_cells = n_want * len(SWEEP_SEEDS)
                if started[wave] and n_have < want_cells:
                    missing.append((f"d{dim} w{wave} {arm}", n_have, want_cells))
                if not rows:
                    if not quiet:
                        print(f"\n  d{dim}  {arm}  -- no cell yet ({want_cells} cells)")
                    continue
                df, best = rank(rows, SELECTION_KEY)
                tuned[dim][arm] = best["point"]
                if not quiet:
                    print_ladder(df, best, _ladder_of(arm, dim), header=f"d{dim}  {arm}",
                                 key=SELECTION_KEY, terms=OBJECTIVE_TERMS,
                                 flat_pct=EDGE_FLAT_PCT, n_have=n_have,
                                 n_want=want_cells)

        for arm in ARMS:
            if arm in PINNED:
                tuned[dim][arm] = {}
            elif arm in INHERIT and INHERIT[arm] in tuned[dim]:
                tuned[dim][arm] = tuned[dim][INHERIT[arm]]
                if not quiet:
                    print(f"\n  d{dim}  {arm}  inherits {INHERIT[arm]}: "
                          f"{tuned[dim][arm]}")
    require_complete(missing, partial=partial, quiet=quiet)
    return tuned


def _ladder_of(arm: str, dim: int) -> Grid:
    """The arm's ladder without the seed axis — what an *edge* is measured against.

    Winning at seed 0 rather than seed 1 is not a narrow ladder, so the grid handed to the
    printer carries the hyper-parameter axes alone.  Every one of them is checked: an
    argmin pinned to the bottom of ``lam_mult`` while sitting mid-ladder in ``rho_mult``
    is as much a too-narrow search as a one-axis edge is.
    """
    return Grid.of(grid_points(arm, dim))


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def cmd_plan(args) -> None:
    """Print the search and the submission loop, with the scheduler left blank."""
    if args.count is not None:
        counts = {d: len(waves_for(d)[args.count]) for d in args.dims}
        assert len(set(counts.values())) == 1, (
            f"the dimensions no longer have equal cell counts ({counts}); a launcher "
            f"cannot size one array for all of them -- submit them separately")
        print(next(iter(counts.values())))
        return
    for dim in args.dims:
        print(f"=== d{dim} ===")
        for wave, fan in waves_for(dim).items():
            print(f"  wave {wave}")
            print(fan.describe())
        print()
    dims = " ".join(str(d) for d in args.dims)
    total = 0
    print(f"Cells are addressed by (dimension, wave, index) and the index restarts at 0\n"
          f"in each of them.  Wave 2 is Path B at the metric wave 1 chose for ffm, so the\n"
          f"`collect` between the two loops is required.  `<submit>` is your own launcher\n"
          f"-- nothing about it belongs in this repository:\n")
    for wave in sorted(waves_for(args.dims[0])):
        last = len(waves_for(args.dims[0])[wave]) - 1
        total += len(args.dims) * (last + 1)
        print(f"    for d in {dims}; do")
        print(f"      for i in $(seq 0 {last}); do")
        print("        <submit> python -m scripts.experiments.erythroid.tune \\")
        print(f"                 run --dim $d --wave {wave} --index $i")
        print("      done")
        print("    done")
        print(f"    python -m scripts.experiments.erythroid.tune collect"
              f"   # {'wave 1 points' if wave == 1 else 'both waves; paste TUNED'}\n")
    print(f"Serially and without a scheduler it is the same loops with `<submit>` deleted;\n"
          f"budget ~{total} cells.  The cache must exist first:\n"
          f"    python -m scripts.experiments.erythroid.preprocess   # UniTVelo, own env, ~1 h")


def cmd_run(args) -> None:
    args.root = smoke_root(args.root, ROOT, "erythroid_bench", args.smoke)

    preflight_cell(waves_for(args.dim), args.wave, args.index, args.dim, TUNED_DIMS,
                   args.root, lambda n, p: cell_path(args.root, args.dim, args.wave, n, p),
                   force=args.force)
    cfg = Run.from_env(**({"device": torch.device("cpu")} if args.cpu else {}),
                       **({"smoke": True} if args.smoke else {}))
    run_one(args.dim, args.wave, args.index, cfg, root=args.root, force=args.force)


def cmd_collect(args) -> None:
    tuned = collect(args.root, dims=args.dims, partial=args.partial)
    if args.partial:
        print(format_tuned(tuned))
        return
    with open(TUNED_FILE, "w") as fh:
        json.dump({str(k): v for k, v in tuned.items()}, fh, indent=1)
    print(f"\nwrote {rel(TUNED_FILE)}\n\n# paste into notebooks/erythroid_ffm_paper.ipynb\n"
          f"# from scripts/experiments/erythroid/tune.py collect")
    print(format_tuned(tuned))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--dims", type=int, nargs="+", default=list(TUNED_DIMS))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="the grid, the dimensions, and the submission loop")
    p.add_argument("--count", type=int, nargs="?", const=1, choices=sorted(WAVES),
                   help="print only the per-dimension cell count, for an array job")
    p.set_defaults(fn=cmd_plan)

    r = sub.add_parser("run", help="one (arm, point, seed) cell at one dimension")
    r.add_argument("--dim", type=int, required=True, choices=sorted(TUNED_DIMS))
    r.add_argument("--wave", type=int, default=1, choices=sorted(WAVES))
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--smoke", action="store_true",
                   help="tiny budgets, into a separate _smoke tree: proves a "
                        "cluster cell runs before the array is submitted")
    r.add_argument("--force", action="store_true",
                   help="re-measure a cell whose record already exists, and audit it")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("collect", help="rank the cells and print the TUNED literal")
    c.add_argument("--partial", action="store_true",
                   help="rank a search that is still running; writes no tuned.json")
    c.set_defaults(fn=cmd_collect)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
