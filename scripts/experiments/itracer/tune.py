"""The R2 hindbrain's own hyper-parameter search — the record, and the thing that runs it.

    python -m scripts.experiments.itracer.tune plan                    # the grid, the loop
    python -m scripts.experiments.itracer.tune plan --count            # cells per dimension
    python -m scripts.experiments.itracer.tune run --dim 20 --index 7  # one cell = one job
    python -m scripts.experiments.itracer.tune collect                 # rank, write tuned.json

:mod:`~scripts.experiments.itracer.bench` reports the **erythroid** point transplanted
unchanged, and that column stays.  This module adds the other half of the question it
raises: a transplant that loses could be losing because the Finsler geometry does not help
on a human organoid, or because a mouse haematopoiesis number is wrong for one, and only a
search on this cloud can tell the two apart.  So both points are run and both are reported,
side by side — ``--point erythroid`` and ``--point tuned`` — rather than the tuned one
quietly replacing the transplanted one.

Searched independently per dimension
------------------------------------
``--dim`` is required and the flat cell index restarts at 0 inside each of
``d = 2, 20, 50``, because ``d = 2`` is a z-scored UMAP chart and ``d = 20/50`` are
unstandardised PCA prefixes.  Nothing inherits a point across a dimension: ``rho_mult``
floors a metric of squared displacements, and a prefix twenty-five times as wide does not
carry one.

Every rung runs at every sweep seed
-----------------------------------
A cell is ``(dim, arm, point, seed)``.  This is the erythroid tree's rule and it is here for
the erythroid tree's reason: the stack does not train reproducibly on GPU, so a rung is an
average over :data:`SWEEP_SEEDS` and not a single number, and ``collect`` averages before it
takes an argmin.  Two seeds per rung doubles the array and is the cheapest thing that stops
the search from ranking run-to-run noise.

One wave.  The erythroid tree has two because ``pathb`` inherits Path A's metric and its
:math:`\\sigma` ladder can only be searched afterwards; this benchmark runs no ``pathb``,
no ``curly`` and no release arms, so there is nothing to sequence.

One cell is one job, and the cell index is the contract
-------------------------------------------------------
Cell ``i`` at dimension ``d`` is the same ``(arm, point, seed)`` in every process and on
every machine, because :class:`scripts.method.tuning.Fanout` is the only place the search is
enumerated.  A scheduler is therefore only ever told a range of integers; ``plan`` prints
that loop with the submission command left as a placeholder, and nothing about a queue, a
partition or an account appears in this file.  A record is named by the point it measures
(:func:`~scripts.method.tuning.cell_tag`) and not by the flat index it was submitted at, so
widening a ladder renumbers the array without invalidating a single record already on disk.

What is searched, and the asymmetry that is not in our favour
-------------------------------------------------------------
The ladders are the erythroid tree's, rung for rung — see :data:`GRID` and
:data:`GRID_BY_DIM`.  That is the point of them: every transplanted value in
:data:`~scripts.experiments.itracer.bench.ERYTHROID_TUNED` is then a rung of this grid (the
assertion under :data:`ANCHOR` enforces it), so the transplanted column and the searched
column are two points on one surface rather than two incomparable answers, and ``collect``
can say how far apart they are in rungs.

Ours has two interacting knobs and is searched over their full product, 12 x 12 = **144
rungs per dimension**; each baseline has one knob and therefore one ladder of 12.  Ours is
chosen from twelve times as many trained models as either baseline is.  That follows from
the parameterisations and not from a decision about who gets a search, and everything around
it is held identical: the same half-decade spacing, the same objective, the same splits, the
same two seeds, and the same widening rule — an argmin on a ladder endpoint means the ladder
was too narrow unless the score has gone flat there (:data:`EDGE_FLAT_PCT`) — applied to the
**baselines first**.

The kernel ablations
--------------------
``ffm_expr``, ``ffm_velo`` and ``ffm_lin_velo`` are our arm on a ``P`` with the lineage
factor off, the velocity factor on, and both on — the 2 x 2 of
:data:`~scripts.experiments.itracer.datasets.P_VARIANTS` completed around ours.  Each is
searched here on its own 144-rung product rather than inheriting our argmin: ``rho_mult``
floors a metric read off ``P``, and each variant's ``P`` is a different matrix with no
reason to want the same floor.  Handing one ours would let the search's gain be counted as
the factor's, which is the one thing these columns exist to measure.  They quadruple the
FFM half of the array — 144 to 576 rungs per dimension, 288 to 1152 cells — and that is the
cost of an ablation being an ablation.

The *temperatures* are not searched.  ``beta`` and ``nu`` are both fixed at 3.0
(:data:`scripts.core.itracer_data.NU`), equal on purpose: giving one factor a tuned weight
while the other keeps a constant would answer which factor tunes better rather than which
carries more information.  What is searched is the geometry read off each resulting ``P``,
which is the same two knobs for every column.

``cfm`` is searched too, although the request was for ours and MFM.  Its entropic
regulariser is ``blur_frac · median(C)`` and ``blur_frac -> 0`` is the exact-OT coupling the
OT-CFM reference actually uses, so a fixed rung is a *default* and not a tuned value.
Leaving it fixed while ours is searched would be the comparison tuning itself, and it costs
8 rungs at two seeds per dimension.

``mfm_land`` sweeps ``land_gamma`` at ``d = 2`` and ``land_rho`` above it, following the
erythroid tree: MFM's implementation reads ``gamma`` only in its 2-D branch, so a gamma
ladder at ``d = 20`` would be eight identical runs.  The axis therefore changes with the
column, which is why this arm's two columns are not read against each other.

The one selection objective
---------------------------
Each cell writes the two terms :mod:`scripts.core.selection` averages — ``W2_endpoint``
against the ``p_1`` cells of the shown val split, and ``W2_intermediate`` at the withheld
marginal's own model time against the 10 % selection slice of the middle third — and the
averaged scalar beside them.  The intermediate term is read at ``t_mid`` and not at
``t = 1/2``, because the flow has to be scored where the marginal sits.  Storing the terms
rather than only the average makes a re-rank free.

A sweep cell trains on the 85 % split (:func:`~scripts.experiments.itracer.datasets
.split_shown`) with ``P``, its bandwidth and the geometry rebuilt on that subset, so the
endpoint term is out of sample.  The benchmark then refits the chosen point on all 1934
shown cells.  Every *reported* number is measured on the disjoint 90 % of the withheld
third, which the tuner never sees.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from scripts.core.arms import run_arm
from scripts.core.itracer_data import CACHE_DIR
from scripts.core.metrics import traj_slice
from scripts.core.paths import experiment_root, rel
from scripts.core.selection import OBJECTIVE_KEY, OBJECTIVE_TERMS, with_objective
from scripts.method import (Fanout, Grid, Run, cell_tag, euclidean_w2, format_tuned,
                            preflight_cell, print_ladder, rank, require_complete,
                            smoke_root)

from .bench import (ARMS, DTYPE, ERYTHROID_TUNED, MODEL_OF, POINT_OF, P_VARIANT_OF,
                    hparams_for)
from .datasets import (DIMS, N_STEPS, TARGET, build_training_set, load_cloud, snap, space,
                       split_holdout, split_shown, t_mid)

# --------------------------------------------------------------------------- #
#  The search
# --------------------------------------------------------------------------- #
#: How many rungs every swept **axis** gets — per axis and not per arm, so a one-knob
#: baseline and one axis of our two-knob metric have the same freedom along them.
#:
#: Twelve and not the erythroid tree's eight.  That tree's ladders were run rung for rung
#: first and the widening rule was then applied twice, on the evidence ``collect`` prints:
#:
#: * at 8 rungs it fired at ``d = 2`` on two bottom rungs — ``mfm_land.land_gamma`` at
#:   0.0125 with a +12.3 % inward slope, ``ffm.lam_mult`` at 0.01 with +7.9 %;
#: * at 10 it fired once more, again at ``d = 2`` and again on ours: ``lam_mult`` at 0.001
#:   with +8.6 %.  ``land_gamma`` had converged by then, so the second round widened a
#:   ladder only ours was still pinned to.
#: * at 12 it fired at ``d = 2`` on ``ffm_expr.lam_mult``, the *ablation baseline*, at 1e-4
#:   with a +19.7 % inward slope — the steepest edge any round has seen, and the first on an
#:   arm that is not ours.  It had to be widened whichever way it cuts: ``ffm_expr`` is the
#:   row every factor is measured against, an unconverged edge there is a baseline that has
#:   not finished falling, and stopping at a rung that happens to flatter the lineage would
#:   be the comparison choosing where to stop.
#:
#: Every round added two rungs to **every** axis and not to the flagged one, because
#: :data:`N_RUNGS` is the statement that no arm is searched harder than another along an
#: axis; widening ours alone would have been the comparison tuning itself, in our favour.
#: A cell is named by the point it measures, so every round re-used every record already on
#: disk: 480 cells, then 240, then 288, then 1728 for the velocity arms, then 1272 more.
N_RUNGS = 14

#: the dimension-independent ladders, :data:`scripts.experiments.erythroid.train.GRID`
#: extended four rungs, each axis at the end its argmins sit at.
#:
#: ``rho_mult`` — the isotropic floor of M = ρI + Σ in units of ``mean tr Σ/d``.  Its
#: argmins sit in the upper middle (0.1 at ``d = 2``, 0.01 and 0.03 above it) and where it
#: has been flagged at all — ``ffm_expr`` at ``d = 20`` and ``d = 50``, both on the bottom
#: rung — the inward slope was +0.1 % and +0.5 %, flat by :data:`EDGE_FLAT_PCT` and so
#: converged.  It is therefore the one axis extended upwards; the rungs it gained are the
#: near-Euclidean M ≈ ρI end and are there to keep the rung count equal, not because the
#: evidence asked for them.
#: ``lam_mult`` — the quadrature floor λ in units of ``ā = mean ‖b‖_{G_0}``.  Both are
#: half-decade rungs, now over 6.5 decades.  The λ ladder's bottom rung is 1e-5 and this
#: cloud's float64 admissibility floor (:func:`scripts.core.scales.lam_mult_floor`) is 6e-8
#: to 2e-7 across all three spaces and the whole rho ladder, two decades below it —
#: measured, and the reason :data:`scripts.experiments.itracer.bench.DTYPE` is float64
#: rather than the repository default.  In float32 the floor is 1.4e-3 to 3.4e-3, the bottom
#: four λ rungs would be inadmissible, and neither widening could have been applied at all.
#:
#: ``blur_frac`` and not ``sigma``: ``run_cfm`` never reads ``hp.sigma``.  It sets the
#: entropic regulariser to ``blur_frac · median(C)``, and ``blur_frac -> 0`` is the exact-OT
#: coupling their minibatch solver uses, so this ladder spans OT-CFM towards our arm.  The
#: engine default 0.1 is the seventh rung.  It is extended downwards because that is where
#: its argmin sits in all three spaces, not because the baseline needed help.
GRID: dict[str, dict[str, list]] = {
    "ffm": {"rho_mult": [3e-5, 1e-4, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0,
                         10.0, 30.0, 100.0],
            "lam_mult": [1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0,
                         3.0, 10.0, 30.0]},
    "cfm": {"blur_frac": [1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0,
                          3.0, 10.0, 30.0]},
}

#: every ablation gets our arm's grid, rung for rung.  Each is the same model, so its axes
#: are the same axes and there is no other ladder any of them could be given -- but each is
#: searched *independently*, because ``rho_mult`` floors a metric read off ``P`` and each
#: variant's ``P`` is a different matrix.  Reusing our argmin for one would be reading the
#: ablation at a point chosen for the thing being ablated, which would attribute the
#: search's gain to the factor.
for _a in ARMS:
    if _a.startswith("ffm") and _a != "ffm":
        GRID[_a] = {axis: list(rungs) for axis, rungs in GRID["ffm"].items()}

#: the columns whose parameterisation is two interacting knobs rather than one -- our arm
#: and its ablations, which are our arm.  Named so the equal-freedom assertion below can
#: say *why* they are exempt from the one-axis rule instead of hard-coding a string.
TWO_AXIS_ARMS = tuple(a for a in ARMS if a.startswith("ffm"))

#: the arm whose swept axis depends on the space, again the erythroid tree's:
#: :data:`scripts.experiments.erythroid.train.GRID_BY_DIM`.  MFM/LAND uses ``gamma`` only in
#: its 2-D branch, so ``rho`` is swept above ``d = 2``.  The ``d = 2`` rungs are centred on
#: MFM's published 0.125 and the ``d > 2`` rungs on their published rho 1e-3.
#:
#: The two axes were widened at opposite ends in the first round, because that is where
#: their argmins were: ``land_gamma`` at ``d = 2`` won at the bottom on a +12.3 % gradient,
#: ``land_rho`` at ``d = 20`` won at the top (flat, but on an endpoint).  Both converged
#: there — ``land_gamma`` moved to 0.0125 with rungs below it and ``land_rho`` stayed at
#: 0.3 with rungs above it — and neither has been flagged since, so the second and third
#: rounds' rungs are the equal-count tax and not a search, and all four went to the quiet
#: end, which for both axes is now the top.
GRID_BY_DIM: dict[str, dict[int, dict[str, list]]] = {
    "mfm_land": {
        2:  {"land_gamma": [0.000125, 0.0004, 0.00125, 0.004, 0.0125, 0.04, 0.125, 0.4,
                            1.25, 4.0, 12.5, 40.0, 125.0, 400.0]},
        20: {"land_rho": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0,
                          3.0, 10.0, 30.0]},
        50: {"land_rho": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0,
                          3.0, 10.0, 30.0]},
    },
}

assert set(GRID) | set(GRID_BY_DIM) == set(ARMS), \
    "the search and the benchmark disagree about the arms"
assert not (set(GRID) & set(GRID_BY_DIM)), "an arm has one grid, not two"
for _arm, _spec in (list(GRID.items())
                    + [(a, s) for a, d in GRID_BY_DIM.items() for s in d.values()]):
    assert len(_spec) == 1 or _arm in TWO_AXIS_ARMS, (
        f"{_arm} turns {sorted(_spec)} at once; only our arm and its ablation may search "
        f"more than one axis, and only because the extra axis is live")
    for _axis, _rungs in _spec.items():
        assert len(_rungs) == N_RUNGS, (
            f"{_arm}.{_axis} has {len(_rungs)} rungs, not {N_RUNGS}; every swept axis "
            f"gets the same freedom along it")

#: seeds every rung is run at — see the module docstring.  A rung is their average.
SWEEP_SEEDS = (0, 1)


def grid_axes(arm: str, dim: int) -> dict[str, list]:
    """One arm's swept axes at one dimension, without the seed."""
    return GRID[arm] if arm in GRID else GRID_BY_DIM[arm][dim]


def grid_points(arm: str, dim: int) -> list[dict]:
    """One arm's rungs at one dimension, as points — the full product of its axes."""
    return list(Grid(**grid_axes(arm, dim)))


def wave1(dim: int) -> Fanout:
    """The searched grid at one dimension, as a flat, index-addressable fan-out.

    The seed is the fastest-varying axis and part of the cell, so ``ffm``'s 8 x 8 rungs at
    two seeds is 128 cells and not 64.
    """
    return Fanout({arm: Grid.of([{**point, "seed": seed}
                                 for point in grid_points(arm, dim)
                                 for seed in SWEEP_SEEDS])
                   for arm in ARMS})


def waves_for(dim: int) -> dict[int, Fanout]:
    """The wave table at one dimension, rebuilt rather than reused.

    Every caller that resolves an index must go through this and never through a table
    built at some other dimension: ``mfm_land`` sweeps ``land_gamma`` at ``d = 2`` and
    ``land_rho`` above it, so the same index names a different point in a different space.
    """
    assert dim in DIMS, f"d={dim} was not searched; this tree searches {tuple(DIMS)}"
    return {1: wave1(dim)}


#: the wave numbers the CLI accepts.  **Not** for resolving an index — see
#: :func:`waves_for`.  Every dimension has the same cell count, which is what lets a
#: launcher size one array for all three; :func:`cmd_plan` is where that is asserted.
WAVES = waves_for(DIMS[0])

#: the scalar every cell is ranked on.  Defined once, in :mod:`scripts.core.selection`.
SELECTION_KEY = OBJECTIVE_KEY

#: how steep the score has to be at an endpoint before "the argmin sits on the edge" means
#: the ladder was too narrow rather than converged.  Every other tree here uses 5 %; against
#: this stack's run-to-run spread an edge caution is a prompt to look, not a measurement.
EDGE_FLAT_PCT = 5.0

#: **the no-search point** — what each arm runs at if this search does not exist, which is
#: the transplanted erythroid point :mod:`~scripts.experiments.itracer.bench` reports.  Every
#: one of them must be a rung, or the search could not reproduce the column it exists to be
#: compared against.
ANCHOR = {dim: {arm: ERYTHROID_TUNED[POINT_OF[dim]][arm] for arm in ARMS} for dim in DIMS}
for _d, _a in ANCHOR.items():
    for _arm, _pt in _a.items():
        assert _pt in grid_points(_arm, _d), (
            f"d{_d} {_arm}: the transplanted point {_pt} is not a rung of the grid, so the "
            f"search cannot reproduce the column it is compared against")

ROOT = experiment_root("itracer_bench")

#: where ``collect`` writes its answer: *beside this file*, not in the run tree.  The run
#: tree is scratch and is not versioned; the tuned point is a result, so it ships with the
#: source and survives a deleted output directory.
TUNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


def load_tuned() -> dict[int, dict[str, dict]]:
    """The searched point per dimension, or ``{}`` if the search has not been collected."""
    if not os.path.exists(TUNED_FILE):
        return {}
    with open(TUNED_FILE) as fh:
        return {int(k): v for k, v in json.load(fh).items()}


# --------------------------------------------------------------------------- #
#  One cell
# --------------------------------------------------------------------------- #
def cell_path(root: str, dim: int, wave: int, name: str, point: dict) -> str:
    """Where one cell's record lands: keyed by the point it measures and not by the flat
    index it was submitted at, so widening a ladder cannot alias one arm's record onto
    another arm's cell.  One function, so preflight and the write agree."""
    return os.path.join(root, "sweep", cell_tag(wave, name, point, dim=dim) + ".json")


def run_cell(dim: int, wave: int, index: int, cfg: Run, root: str = ROOT,
             cache_dir: str = CACHE_DIR, force: bool = False) -> dict:
    """Train one ``(arm, point, seed)`` on the 85 % split and write the objective's terms."""
    # ``waves_for(dim)`` and not the module-level table: ``mfm_land`` sweeps a different
    # axis at d = 2, so indexing one dimension's fan-out to run another's would file a
    # ``land_gamma`` rung under a ``land_rho`` cell.
    arm, cell, path = preflight_cell(
        waves_for(dim), wave, index, dim, DIMS, root,
        lambda n, p: cell_path(root, dim, wave, n, p), force=force, cache=cache_dir)
    point = dict(cell)
    seed = point.pop("seed")

    cloud = load_cloud(cache_dir=cache_dir)
    i_train, i_val = split_shown(cloud)
    # an ablation's kernel is rebuilt on the same 85 % split with one temperature moved,
    # so its search sees exactly the cells ours does and differs from it in that factor
    # alone -- the velocity alignment included, which is refitted on this split too.
    ts = build_training_set(cloud, dim, subset=i_train, p_variant=P_VARIANT_OF[arm])
    t_star = snap(t_mid(cloud), N_STEPS)

    hp, _ = hparams_for(dim, arm, ts, cfg, point=point)
    t0 = time.time()
    res = run_arm(MODEL_OF[arm], ts, hp, seed, cfg.device, cfg.dtype, n_steps=N_STEPS)
    traj = np.asarray(res.traj, dtype=np.float64)

    # the objective's two references, both in this cell's own space.  ``i_end`` is the val
    # slice of p_1 among the shown cells — the arm never trained on it, and neither did the
    # kernel — and ``i_sel`` is the 10 % selection slice of the withheld third, which
    # nothing trains on under any protocol.  The disjoint 90 % is what the benchmark reports.
    Xd = space(cloud, dim)
    i_end = i_val[cloud.third[i_val] == TARGET]
    assert len(i_end) > 1, f"d{dim}: the val slice of p1 is too small to score ({len(i_end)})"
    i_sel, _ = split_holdout(cloud)

    terms = with_objective({
        "W2_endpoint": euclidean_w2(traj_slice(traj, 1.0), Xd[i_end], seed=seed),
        "W2_intermediate": euclidean_w2(traj_slice(traj, t_star), Xd[i_sel], seed=seed),
    })
    rec = {"dim": dim, "wave": wave, "index": index, "arm": arm, "point": point,
           "model": MODEL_OF[arm], "p_variant": ts.p_variant,
           "seed": int(seed), **terms, "t_mid": t_star, "n_end_cells": int(len(i_end)),
           "n_val_cells": int(len(i_sel)), "n_train": int(len(i_train)),
           "dtype": str(cfg.dtype), "train_s": round(time.time() - t0, 1)}
    with open(path, "w") as fh:                       # preflight made the directory
        json.dump(rec, fh, indent=1)
    print(f"[cell] d{dim} w{wave} i{index:>3}  {arm:<9} {point} s{seed}  "
          f"objective {rec[SELECTION_KEY]:.4f}  (end {rec['W2_endpoint']:.4f}, "
          f"mid {rec['W2_intermediate']:.4f})  [{rec['train_s']:.0f}s] -> {path}")
    return rec


# --------------------------------------------------------------------------- #
#  Collect
# --------------------------------------------------------------------------- #
def sweep_rows(root: str = ROOT) -> list[dict]:
    """Every finished search cell as a plain record, unranked, unaveraged, unfiltered.

    Split out of :func:`collect` so the notebook can draw the search surface without
    re-deriving where the records live or which of them are current: a cell is named by the
    point it measures, so the directory listing *is* the set of measured points.
    """
    sweep = os.path.join(root, "sweep")
    assert os.path.isdir(sweep), (
        f"no sweep records under {sweep}; run the search first "
        f"(`python -m scripts.experiments.itracer.tune plan` prints the loop)")
    rows = []
    for name in sorted(os.listdir(sweep)):
        if name.endswith(".json"):
            with open(os.path.join(sweep, name)) as fh:
                rows.append(json.load(fh))
    return rows


def _point_tag(point: dict) -> str:
    return cell_tag(0, "", point)


def arm_ladder(rows: list[dict], dim: int, arm: str) -> tuple[list[dict], int]:
    """One arm's rungs at one dimension, **seed-averaged**, plus how many the grid wants.

    These are the rows ``collect`` ranks and prints, so the ladder shown and the point
    chosen are one ranking.  Points that are no longer on the grid are dropped and named:
    a widened or narrowed ladder leaves records behind, and silently ranking them would let
    a rung that is no longer part of the search win it.
    """
    live = {_point_tag(p): p for p in grid_points(arm, dim)}
    cells = [r for r in rows if r["dim"] == dim and r["arm"] == arm]
    stale = {_point_tag(r["point"]) for r in cells} - set(live)
    if stale:
        print(f"[tune] d={dim} {arm}: ignoring {len(stale)} off-grid point(s) "
              f"{sorted(stale)}")
    cells = [r for r in cells if _point_tag(r["point"]) in live]

    by_point: dict[str, list[dict]] = {}
    for r in cells:
        by_point.setdefault(_point_tag(r["point"]), []).append(r)
    ladder = []
    for rs in by_point.values():
        ladder.append({"point": rs[0]["point"], "n_seeds": len(rs),
                       **{t: float(np.mean([r[t] for r in rs])) for t in OBJECTIVE_TERMS},
                       SELECTION_KEY: float(np.mean([r[SELECTION_KEY] for r in rs]))})
    return ladder, len(live)


def collect(root: str = ROOT, dims=DIMS, quiet: bool = False,
            partial: bool = False) -> dict:
    """Rank every finished cell and return ``{dim: {arm: point}}``.

    An incomplete dimension is fatal unless ``partial``: ranking twenty rungs of eighty and
    publishing the winner is how a 480-cell search quietly reports the argmin of whichever
    cells the queue happened to finish.
    """
    rows = sweep_rows(root)

    tuned: dict = {}
    missing: list = []
    for dim in dims:
        tuned[dim] = {}
        for arm in ARMS:
            ladder, n_want = arm_ladder(rows, dim, arm)
            n_have = sum(r["n_seeds"] for r in ladder)
            want_cells = n_want * len(SWEEP_SEEDS)
            if not ladder:
                if not quiet:
                    print(f"\n  d{dim}  {arm}  -- no cell yet ({want_cells} cells)")
                continue
            if n_have < want_cells:
                missing.append((f"d{dim} {arm}", n_have, want_cells))
            df, best = rank(ladder, SELECTION_KEY)
            tuned[dim][arm] = best["point"]
            if not quiet:
                print_ladder(df, best, Grid(**grid_axes(arm, dim)),
                             header=f"d{dim}  {arm}", key=SELECTION_KEY,
                             terms=OBJECTIVE_TERMS, flat_pct=EDGE_FLAT_PCT,
                             n_have=n_have, n_want=want_cells)
                _report_move(dim, arm, best["point"], df)
    require_complete(missing, partial=partial, quiet=quiet)
    return tuned


def _report_move(dim: int, arm: str, best: dict, df) -> None:
    """How far the search moved off the transplanted point, in rungs and in score.

    The whole reason both columns are reported is that they may disagree, so ``collect``
    says *how much* rather than leaving the reader to diff two dictionaries: an argmin one
    rung from the anchor with the two scores within noise is a different finding from an
    argmin four rungs away.
    """
    anchor = ANCHOR[dim][arm]
    steps = {k: abs(grid_axes(arm, dim)[k].index(v)
                    - grid_axes(arm, dim)[k].index(anchor[k]))
             for k, v in best.items() if k in anchor}
    hit = df[df["point"].apply(lambda p: p == anchor)]
    at = f"{hit.iloc[0][SELECTION_KEY]:.4f}" if len(hit) else "not measured"
    moved = "on it" if not any(steps.values()) else \
        ", ".join(f"{k} {n} rung{'s' if n != 1 else ''}" for k, n in steps.items())
    print(f"    transplanted point {anchor} scores {at}; the search sits {moved}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def cmd_plan(args) -> None:
    """Print the search and the submission loop, with the scheduler left blank."""
    counts = {d: len(waves_for(d)[1]) for d in args.dims}
    assert len(set(counts.values())) == 1, (
        f"the dimensions no longer have equal cell counts ({counts}); a launcher cannot "
        f"size one array for all of them -- submit them separately")
    n = next(iter(counts.values()))
    if args.count:
        print(n)
        return
    for dim in args.dims:
        print(f"=== d{dim} ===")
        print(waves_for(dim)[1].describe())
        print()
    dims = " ".join(str(d) for d in args.dims)
    print(f"  {n} cells x {len(args.dims)} dimensions ({dims}) = {n * len(args.dims)} jobs")
    print(f"""
Submit as one array over the cell index.  `<submit>` is your own launcher -- nothing about
it belongs in this repository's public half:

    python -m scripts.experiments.itracer.bench --prebuild      # once, before the array
    for d in {dims}; do
      for i in $(seq 0 {n - 1}); do
        <submit> python -m scripts.experiments.itracer.tune run --dim $d --index $i
      done
    done
    python -m scripts.experiments.itracer.tune collect          # ranks, writes tuned.json

Every dimension is searched in its own space; nothing inherits a point across dimensions,
because rho floors a metric of squared displacements and a PCA prefix twenty-five times as
wide does not carry one.""")


def cmd_run(args) -> None:
    # A wiring check writes a real-looking record under a content-addressed name, so it
    # goes to its own tree rather than into the sweep `collect` ranks.
    args.root = smoke_root(args.root, ROOT, "itracer_bench", args.smoke)

    # Widening a ladder renumbers every cell after the new rung, so the way to run the rungs
    # a widening added is to re-submit the whole array: the record is named by the point and
    # not by the index, so the cells that already exist are exactly the ones that were
    # already measured.  Preflight *refuses* those, which is right for an accidental
    # resubmission and wrong for a deliberate one -- hence the flag, and hence it exits 0
    # rather than raising, so a widening does not fill the queue with failures.
    if args.skip_existing and not args.force:
        arm, cell = waves_for(args.dim)[args.wave][args.index]
        path = cell_path(args.root, args.dim, args.wave, arm, cell)
        if os.path.exists(path):
            print(f"d{args.dim} w{args.wave} i{args.index}  {arm} {cell}: already "
                  f"measured, skipping -> {path}")
            return

    # Preflight before anything expensive, so a bad invocation dies in its first second
    # rather than after the cloud has been loaded and the model built.
    preflight_cell(waves_for(args.dim), args.wave, args.index, args.dim, DIMS, args.root,
                   lambda n, p: cell_path(args.root, args.dim, args.wave, n, p),
                   force=args.force, cache=args.cache_dir)
    cfg = Run.from_env(n_steps=N_STEPS, dtype=DTYPE,
                       **({"device": torch.device("cpu")} if args.cpu else {}),
                       **({"smoke": True} if args.smoke else {}))
    run_cell(args.dim, args.wave, args.index, cfg, root=args.root,
             cache_dir=args.cache_dir, force=args.force)


def cmd_collect(args) -> None:
    tuned = collect(args.root, dims=args.dims, partial=args.partial)
    if args.partial:
        # An incomplete search may be *read* but never *published*: writing tuned.json here
        # would leave the benchmark reporting an answer the search had not finished.
        print(format_tuned(tuned))
        return
    with open(TUNED_FILE, "w") as fh:
        json.dump({str(k): v for k, v in tuned.items()}, fh, indent=1)
    print(f"\nwrote {rel(TUNED_FILE)}")
    print(format_tuned(tuned))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    # `--count` prints one integer and nothing else, so a launcher sizes its array off the
    # grid rather than off a number that a widened rung would stale.
    p.add_argument("--count", action="store_true")
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run", help="one (arm, point, seed) cell at one dimension")
    r.add_argument("--wave", type=int, default=1, choices=sorted(WAVES))
    r.add_argument("--dim", type=int, required=True)
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--smoke", action="store_true",
                   help="tiny budgets, into a separate _smoke tree: proves a cluster cell "
                        "runs before the array is submitted")
    r.add_argument("--force", action="store_true",
                   help="re-measure a cell whose record already exists")
    r.add_argument("--skip-existing", action="store_true",
                   help="exit 0 on a cell that already has a record, instead of refusing: "
                        "how the rungs a widening added are run")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("collect")
    c.add_argument("--partial", action="store_true",
                   help="rank a search that is still running; writes no tuned.json")
    c.set_defaults(fn=cmd_collect)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
