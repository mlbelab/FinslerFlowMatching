"""The matched comparison's search: equal ladders, one selection rule, two rankings.

    python -m scripts.experiments.matched.tune plan
    python -m scripts.experiments.matched.tune run --key pancreas20 --index 7
    python -m scripts.experiments.matched.tune collect

One cell is one job and the cell index is the contract, exactly as in the other trees:
cell ``i`` is the same ``(arm, point)`` in every process because
:class:`scripts.method.tuning.Fanout` is the only place the search is enumerated, and the
record is filed under the *point it measured* rather than the index it was submitted at.
Nothing about a queue, a partition or an account appears in this module.

Budget, and why the two arms that matter have exactly the same one
------------------------------------------------------------------
The comparison the review cares about is ``ffm_full`` against ``aniso_fixed``.  Both have
two knobs, both get the **same seven-rung half-decade ladder on each**, both are ranked on
the same objective against the same validation cells with the same trainer and the same
budget: 49 cells each, and no tie-break anywhere that favours one.  Giving the competitor
a fixed ``a = 1`` and reading a poor score as evidence against fixed-time objectives is
the specific failure this arrangement exists to rule out -- ``a`` is a *measured* scale
(:func:`~scripts.experiments.matched.variants.measure_a_star`) that is then searched
around its measurement, on the same spacing and with the same number of rungs as our
:math:`\\lambda`.

The two one-knob arms are one-knob for reasons that are algebraic and not budgetary.
``graph_drift``'s cost never touches :math:`C_\\rho`, so it has no :math:`\\rho`;
``ffm_second``'s :math:`\\lambda` is a global positive factor, so it is absorbed exactly by
:meth:`Metric.calibrate` and by ``reg = blur_frac * median(C)`` and a ladder on it would
return identical rows.  Rather than pad them with a dead axis, each gets a ladder **twice
as fine** on its live one -- thirteen quarter-decade rungs spanning the same range, a
superset of the seven -- so neither is shortchanged on the only thing it can be tuned for.
the verification suite in the research repository asserts both facts rather than asserting them
here in prose.

``ffm_riem`` is not searched.  It is the existing one-form ablation and is reported at
``ffm_full``'s selected point, which is what "the same metric with one term removed"
means; a separately-tuned ablation would be measuring a different model.

Two rankings, one set of cells
------------------------------
Every cell trains the interpolant **once** and then distils it twice -- under one fixed
Euclidean entropic-OT coupling shared verbatim by all arms, and under its own learned path
cost -- so a record carries the objective's two terms for *both* coupling settings.
``collect`` therefore returns two tuned points per arm at no extra compute, and the
common-coupling column is tuned under common coupling rather than handed the point that
won under a different pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from scripts.core.paths import experiment_root, rel
from scripts.core.selection import OBJECTIVE_KEY, OBJECTIVE_TERMS
from scripts.method import (Fanout, Grid, Run, cell_tag, format_tuned, preflight_cell,
                            print_ladder, rank, require_complete, smoke_root)
from scripts.method.nets import Coupler
from scripts.method.phases import integrate, train_cfm, train_geodesic

from . import problem as prob
from .variants import (ARMS, COUPLINGS, KNOBS, MatchedFactory, build_ot_euclidean,
                       matched_path_a)

# --------------------------------------------------------------------------- #
#  The ladders -- half-decades, every rung a multiple of a measured scale
# --------------------------------------------------------------------------- #
#: multiples of ``rho_star = mean_i tr D~_i / d``.  Seven rungs over three decades, which
#: brackets both published points this tree has to be able to reach: the Sheet's
#: ``rho_mult = 1.0`` and Pancreas d = 20's ``0.03``.
RHO_MULTS = (0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0)
#: multiples of the measured ``a_bar = mean ||m||_{C_rho^-1}``.  Same seven rungs, and it
#: brackets the Sheet's ``lam_mult = 0.3`` and Pancreas d = 20's ``1.0``.
LAM_MULTS = (0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0)
#: multiples of the measured ``a_star`` -- graph steps per unit transport distance.  Same
#: seven rungs and the same spacing as ``LAM_MULTS``, which is the budget match: the
#: fixed-time competitor searches its time scale exactly as hard as we search ours.
A_MULTS = (0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0)
#: quarter-decades over the identical range, and a strict superset of the seven above.
#: The two one-knob arms get this on their single live axis.
FINE_MULTS = (0.01, 0.0178, 0.0316, 0.0562, 0.1, 0.178, 0.316, 0.562,
              1.0, 1.78, 3.16, 5.62, 10.0)

WAVE1 = Fanout({
    "ffm_full":    Grid(rho_mult=RHO_MULTS, lam_mult=LAM_MULTS),      # 49
    "aniso_fixed": Grid(rho_mult=RHO_MULTS, a_mult=A_MULTS),          # 49, matched
    "ffm_second":  Grid(rho_mult=FINE_MULTS),                         # 13, one live knob
    "graph_drift": Grid(a_mult=FINE_MULTS),                           # 13, one live knob
})

# --------------------------------------------------------------------------- #
#  Wave 2: two rungs further down in rho, on *both* sides of the matched pair
# --------------------------------------------------------------------------- #
# Wave 1 put the Sheet's ``aniso_fixed`` argmin on the bottom rho rung under common
# coupling, against a +10.2 % gradient -- past ``EDGE_FLAT_PCT``, so the competitor's
# optimum is outside the box wave 1 searched.  Reporting a comparison against a baseline
# whose ladder was still descending is the specific failure this tree exists to avoid, so
# the ladder is extended.
#
# The extension is applied to ``ffm_full`` as well, and by the same two rungs, even though
# *our* rho was flat at its edge on every dataset and coupling.  Widening only the arm that
# flagged would leave the two arms searching boxes of different sizes, and "the competitor
# got a bigger grid" is no better an artefact than "the competitor got a smaller one".
# ``ffm_second`` gets the four quarter-decade rungs that keep its ladder a superset of the
# other two; ``graph_drift`` has no rho and is untouched.
RHO_MULTS_W2 = (0.001, 0.00316)
FINE_MULTS_W2 = (0.001, 0.00178, 0.00316, 0.00562)

WAVE2 = Fanout({
    "ffm_full":    Grid(rho_mult=RHO_MULTS_W2, lam_mult=LAM_MULTS),   # 14
    "aniso_fixed": Grid(rho_mult=RHO_MULTS_W2, a_mult=A_MULTS),       # 14, matched
    "ffm_second":  Grid(rho_mult=FINE_MULTS_W2),                      # 4
})

# --------------------------------------------------------------------------- #
#  Wave 3: two rungs further down again -- Sheet only, and why only the Sheet
# --------------------------------------------------------------------------- #
# Ranking waves 1 and 2 on a three-seed mean put the Sheet's ``aniso_fixed`` argmin back on
# the bottom rho rung, under both couplings.  The pooled ladder says that is a tie inside a
# plateau rather than a descent -- the rung it is measured against has a spread of +-0.087
# on a 0.05 difference -- but "the competitor's optimum sat on the wall of the box" is the
# one reading of this comparison that no amount of argument can retire, so the wall moves.
#
# Sheet cells cost about 41 s, so this is cheap; Pancreas cells cost 129 s, and that is not
# why Pancreas is excluded.  Pancreas is excluded because it *cannot* be measured here: at
# d = 20 the raw second-moment matrix is rank-deficient (smallest eigenvalue -4.6e-14, so
# rho is the only thing making C_rho invertible), and the relative error of the float32
# inverse climbs 4.1e-3 -> 2.2e-2 -> 7.2 across rho_mult 1e-3 -> 3.16e-4 -> 1e-5.  Rungs
# below 1e-3 on Pancreas would return arithmetic, not geometry, and the ranker would be
# free to select them.  At d = 3 the Sheet has no such wall: its float32 inverse error
# stays under 4.6e-4 down to rho_mult 1e-5, and its median condition number has already
# saturated near 58-61, which is the measured reason to expect this wave to come back flat.
#
# Applied to ``ffm_full`` and ``aniso_fixed`` by the same two rungs, as in wave 2.
RHO_MULTS_W3 = (0.0001, 0.000316)
FINE_MULTS_W3 = (0.0001, 0.000178, 0.000316, 0.000562)

WAVE3 = Fanout({
    "ffm_full":    Grid(rho_mult=RHO_MULTS_W3, lam_mult=LAM_MULTS),   # 14
    "aniso_fixed": Grid(rho_mult=RHO_MULTS_W3, a_mult=A_MULTS),       # 14, matched
    "ffm_second":  Grid(rho_mult=FINE_MULTS_W3),                      # 4
})

WAVES = {1: WAVE1, 2: WAVE2, 3: WAVE3}

#: which datasets each wave is submitted for.  Only wave 3 is restricted, and only on the
#: numerical grounds set out above -- never to give one arm a cheaper search than another,
#: which is why the restriction is by *dataset* and never by arm.
WAVE_KEYS: dict[int, tuple[str, ...]] = {1: prob.KEYS, 2: prob.KEYS, 3: ("sheet",)}


def _waves_for(key: str) -> tuple[int, ...]:
    """The waves submitted for one dataset, in order."""
    return tuple(w for w in sorted(WAVES) if key in WAVE_KEYS[w])

#: the ladders a *finished* search is ranked and edge-checked against -- the union of the
#: waves that dataset received, which is the box that was actually searched.  ``WAVES``
#: says what to submit; ``LADDERS`` says what was covered, and ``collect`` reads this one
#: so an edge caution is measured against the real extent of the grid rather than against
#: one round of it.  Keyed by dataset because wave 3 is.
RHO_ALL = tuple(sorted(RHO_MULTS + RHO_MULTS_W2))
FINE_ALL = tuple(sorted(FINE_MULTS + FINE_MULTS_W2))
RHO_SHEET = tuple(sorted(RHO_ALL + RHO_MULTS_W3))
FINE_SHEET = tuple(sorted(FINE_ALL + FINE_MULTS_W3))

_LADDERS_W12 = {                                             # waves 1 + 2
    "ffm_full":    Grid(rho_mult=RHO_ALL, lam_mult=LAM_MULTS),        # 63
    "aniso_fixed": Grid(rho_mult=RHO_ALL, a_mult=A_MULTS),            # 63, matched
    "ffm_second":  Grid(rho_mult=FINE_ALL),                           # 17
    "graph_drift": Grid(a_mult=FINE_MULTS),                           # 13
}
_LADDERS_W123 = {                                            # waves 1 + 2 + 3
    "ffm_full":    Grid(rho_mult=RHO_SHEET, lam_mult=LAM_MULTS),      # 77
    "aniso_fixed": Grid(rho_mult=RHO_SHEET, a_mult=A_MULTS),          # 77, matched
    "ffm_second":  Grid(rho_mult=FINE_SHEET),                         # 21
    "graph_drift": Grid(a_mult=FINE_MULTS),                           # 13
}

LADDERS = {k: (_LADDERS_W123 if 3 in _waves_for(k) else _LADDERS_W12) for k in prob.KEYS}

#: this tree searches each dataset independently, but the axis is a *name* and not an
#: integer dimension, so it is carried in the record path rather than through
#: ``preflight_cell``'s ``--dim``.  ``TUNED_DIMS`` is empty for the same reason it is empty
#: in the Sheet's tuner: there is no integer space to select.
TUNED_DIMS: tuple[int, ...] = ()

SELECTION_KEY = OBJECTIVE_KEY
#: the objective's two terms, once per coupling setting.  Flattened into the record so a
#: finished search can be re-ranked on either column without retraining anything.
TERM_KEYS = {c: tuple(f"{c}_{t}" for t in OBJECTIVE_TERMS) for c in COUPLINGS}
RANK_KEY = {c: f"{c}_{OBJECTIVE_KEY}" for c in COUPLINGS}

#: the model seeds every cell is measured at, and averaged over before ranking.
#:
#: Selection on a single draw turned out to be the weak joint of this comparison.  On the
#: Sheet, one seed put ``ffm_full``'s argmin on the bottom rho rung against an apparent
#: +37 % gradient; re-measuring that rung at four seeds showed it is in fact *worse* than
#: the two rungs above it (0.196 +- 0.021 against 0.167 +- 0.015), and that the ladder is
#: flat there -- the conditioning probe says C_rho's median condition number has saturated
#: by rho_mult ~ 0.01 at d = 3, so the metric has stopped changing and only the noise is
#: left.  A comparison whose conclusion can turn on which arm drew the luckier seed at
#: selection time is not a matched comparison, so every arm is now ranked on a mean.
#:
#: Three seeds, applied identically to every arm, so the budget stays matched.
TUNE_SEEDS = (0, 1, 2)

EDGE_FLAT_PCT = 5.0

ROOT = experiment_root("matched_bench")
TUNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


def smoke_cfg(smoke: bool, cpu: bool) -> Run:
    """A :class:`Run` for the CLI, with ``--smoke`` meaning what the notebooks mean by it.

    ``Run.from_env`` applies keyword overrides *after* its own budget cut, so passing
    ``smoke=True`` to it sets the flag and leaves ``geo_iters`` at 2500 -- a "smoke" run
    that is a full run into a scratch tree.  Going through ``NB_SMOKE`` instead keeps one
    definition of what a smoke budget is, in ``config.py``, rather than restating the
    numbers here where they could drift.
    """
    env = dict(NB_SMOKE="1") if smoke else {}
    if cpu:
        env["NB_CPU"] = "1"
    prior = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return Run.from_env()
    finally:
        for k, v in prior.items():
            os.environ.pop(k) if v is None else os.environ.update({k: v})


# --------------------------------------------------------------------------- #
#  One cell
# --------------------------------------------------------------------------- #
def cell_path(root: str, key: str, wave: int, name: str, point: dict,
              seed: int = 0) -> str:
    """Keyed by the point it measures, under the dataset and the seed it measured it at."""
    return os.path.join(root, "sweep", key, f"s{seed}",
                        cell_tag(wave, name, point) + ".json")


def warm_up(factory: MatchedFactory, X0_t, X1_t, cfg: Run, iters: int = 20) -> bool:
    """Twenty discarded iterations at the least-conditioned corner, before anything kept.

    Puts every cell on the same footing as :func:`scripts.method.phases.cuda_warm_up`
    does for the sweep, mirrored here for a cost object that also has to warm the
    Euclidean Sinkhorn the common column uses.  One cell is one process, so it runs
    per cell.
    """
    if cfg.device.type != "cuda":
        return False
    # Pinned, not read off the current ladder: the warm-up exists so every cell is scored
    # under the same numerics, and a later widening that moved this corner would silently
    # re-base wave 2 against wave 1.  It is a fixed corner of the wave-1 box and stays
    # there.
    cost = factory.at("ffm_full", {"rho_mult": 0.01, "lam_mult": 0.01})
    phi = train_geodesic(cost, Coupler(X0_t, X1_t, seed=cfg.tune_seed), cfg.tune_seed,
                         cfg, iters=iters, log_every=0)
    cp, _ = build_ot_euclidean(X0_t, X1_t, cfg.tune_seed, cfg)
    integrate(train_cfm(phi, cp, cost.d, cfg.tune_seed, cfg, iters=iters, log_every=0),
              X0_t, cfg)
    return True


def run_cell(key: str, wave: int, index: int, cfg: Run, root: str = ROOT,
             force: bool = False, seed: int = 0) -> dict:
    """Train one ``(arm, point)`` on one dataset at one model seed."""
    assert key in prob.KEYS, f"unknown dataset {key!r}; this tree has {prob.KEYS}"
    assert seed in TUNE_SEEDS, f"seed {seed} is not one of {TUNE_SEEDS}"
    assert key in WAVE_KEYS[wave], (
        f"wave {wave} is not submitted for {key!r} (it runs on {WAVE_KEYS[wave]}); "
        f"recording it would put a point on that dataset's ladder that nothing else "
        f"covers, and `collect` would rank against a box it did not search")
    arm, point, path = preflight_cell(
        WAVES, wave, index, None, TUNED_DIMS, root,
        lambda n, p: cell_path(root, key, wave, n, p, seed), force=force,
        cache=prob.cache_for(key))

    pb = prob.build(key, "tune", cfg)
    factory = MatchedFactory(pb.X, pb.D_pts, pb.b0_pts, cfg, pb.X0_t, pb.X1_t)
    warm_up(factory, pb.X0_t, pb.X1_t, cfg)

    t0 = time.time()
    cost = factory.at(arm, point)
    out = matched_path_a(cost, pb.X0_t, pb.X1_t, seed, cfg)

    rec = {"key": key, "wave": wave, "index": index, "arm": arm, "point": point,
           "tag": cell_tag(wave, arm, point), "seed": seed,
           "train_s": round(time.time() - t0, 1),
           "scales": factory.scales(), "n_train": pb.diag["n_train"]}
    for mode in COUPLINGS:
        terms = pb.val(out[mode]["traj"], seed=seed)
        rec.update({f"{mode}_{k}": float(v) for k, v in terms.items()})
        rec[f"{mode}_total_s"] = round(out[mode]["total_s"], 1)
        rec[f"{mode}_pi_entropy_frac"] = out[mode]["ot"]["pi_entropy_frac"]

    with open(path, "w") as fh:                      # preflight made the directory
        json.dump(rec, fh, indent=1)
    print(f"[cell] {key} w{wave} i{index:>3} s{seed}  {arm:<12} {point}  "
          f"objective common {rec[RANK_KEY['common']]:.4f} / specific "
          f"{rec[RANK_KEY['specific']]:.4f}  [{rec['train_s']:.0f}s] -> {path}")
    return rec


# --------------------------------------------------------------------------- #
#  Collect
# --------------------------------------------------------------------------- #
def _pool_seeds(sweep: str) -> tuple[list[dict], int]:
    """One row per grid point, every numeric term averaged over ``TUNE_SEEDS``.

    A point measured at fewer than all the seeds is *dropped*, not averaged over what
    happens to be on disk: a ladder in which some rungs are three-seed means and others are
    single draws would rank the noisier rungs to the top exactly as often as it ranked the
    better ones, which is the failure this pooling exists to remove.  The dropped points
    then show up as an incomplete ladder in ``require_complete``, which is where a missing
    cell belongs.
    """
    by_tag: dict[str, list[dict]] = {}
    n_raw = 0
    for seed in TUNE_SEEDS:
        d = os.path.join(sweep, f"s{seed}")
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(d, name)) as fh:
                r = json.load(fh)
            by_tag.setdefault(r.get("tag", name[:-5]), []).append(r)
            n_raw += 1

    rows = []
    for tag, recs in sorted(by_tag.items()):
        if len({r["seed"] for r in recs}) < len(TUNE_SEEDS):
            continue
        head = recs[0]
        row = {"arm": head["arm"], "point": head["point"], "tag": tag,
               "n_seeds": len(recs)}
        keys = [k for k in head if isinstance(head[k], (int, float))
                and k not in ("seed", "index", "wave")]
        for k in keys:
            row[k] = float(np.mean([r[k] for r in recs]))
            row[f"{k}_sd"] = float(np.std([r[k] for r in recs], ddof=1))
        rows.append(row)
    return rows, n_raw


def collect(root: str = ROOT, keys=prob.KEYS, quiet: bool = False,
            partial: bool = False) -> dict:
    """Rank every finished cell -> ``{key: {coupling: {arm: point}}}``.

    Two rankings over one set of records, because a cell measured both couplings.  The
    completeness guard is the shared one: an unfinished search may be *read* with
    ``partial`` but never published, since the argmin of a third of a ladder is usually on
    an endpoint and then reads as "widen" rather than as "you have not run it".
    """
    tuned: dict = {}
    missing: list = []
    for key in keys:
        sweep = os.path.join(root, "sweep", key)
        if not os.path.isdir(sweep):
            if not quiet:
                print(f"\n=== {key} === no cells yet")
            continue
        rows, n_raw = _pool_seeds(sweep)
        if not quiet:
            print(f"\n=== {key} ===  {len(rows)} points, {n_raw} records "
                  f"({len(TUNE_SEEDS)} seeds each)")
        tuned[key] = {}
        for mode in COUPLINGS:
            tuned[key][mode] = {}
            if not quiet:
                print(f"\n-- coupling: {mode} --")
            for arm, grid in LADDERS[key].items():
                cells = [r for r in rows if r["arm"] == arm]
                if len(cells) < len(grid):
                    # counted once per dataset, not once per coupling
                    if mode == COUPLINGS[0]:
                        missing.append((f"{key} {arm}", len(cells), len(grid)))
                if not cells:
                    continue
                df, best = rank(cells, RANK_KEY[mode])
                tuned[key][mode][arm] = best["point"]
                if not quiet:
                    print_ladder(df, best, grid, header=arm, key=RANK_KEY[mode],
                                 terms=TERM_KEYS[mode], flat_pct=EDGE_FLAT_PCT,
                                 n_have=len(cells), n_want=len(grid))
            # the existing ablation is reported at ours, never separately tuned
            if "ffm_full" in tuned[key][mode]:
                tuned[key][mode]["ffm_riem"] = dict(tuned[key][mode]["ffm_full"])
    require_complete(missing, partial=partial, quiet=quiet)
    return tuned


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def cmd_plan(args) -> None:
    if args.count is not None:
        print(len(WAVES[args.count]))
        return
    for w in sorted(WAVES):
        print(f"--- wave {w} ---")
        print(WAVES[w].describe())
    n = len(WAVES[args.wave])
    keys = WAVE_KEYS[args.wave]
    print(f"\nwave {args.wave}: {n} cells per dataset x {len(keys)} datasets "
          f"({' '.join(keys)}) = {n * len(keys)}.")
    for k in prob.KEYS:
        lad = LADDERS[k]
        print(f"searched in total on {k}: ffm_full {len(lad['ffm_full'])} cells and "
              f"aniso_fixed {len(lad['aniso_fixed'])} -- the matched pair, equal by "
              f"construction.")
    print(f"""
Submit as one array over (dataset, cell).  `<submit>` is your own launcher:

    for k in {' '.join(keys)}; do
      for i in $(seq 0 {n - 1}); do
        <submit> python -m scripts.experiments.matched.tune run --key $k \\
                   --wave {args.wave} --index $i
      done
    done
    python -m scripts.experiments.matched.tune collect""")


def cmd_run(args) -> None:
    args.root = smoke_root(args.root, ROOT, "matched_bench", args.smoke)
    preflight_cell(WAVES, args.wave, args.index, None, TUNED_DIMS, args.root,
                   lambda n, p: cell_path(args.root, args.key, args.wave, n, p, args.seed),
                   force=args.force, cache=prob.cache_for(args.key))
    cfg = smoke_cfg(args.smoke, args.cpu)
    run_cell(args.key, args.wave, args.index, cfg, root=args.root, force=args.force,
             seed=args.seed)


def cmd_collect(args) -> None:
    tuned = collect(args.root, partial=args.partial)
    if args.partial:
        print(format_tuned(tuned))
        return
    with open(TUNED_FILE, "w") as fh:
        json.dump(tuned, fh, indent=1)
    print(f"\nwrote {rel(TUNED_FILE)}")
    print(format_tuned(tuned))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--count", type=int, default=None, metavar="WAVE", choices=sorted(WAVES))
    p.add_argument("--wave", type=int, default=max(WAVES), choices=sorted(WAVES))
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--key", required=True, choices=prob.KEYS)
    r.add_argument("--wave", type=int, default=1, choices=sorted(WAVES))
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--seed", type=int, default=0, choices=TUNE_SEEDS)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--smoke", action="store_true",
                   help="tiny budgets, into a separate _smoke tree")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("collect")
    c.add_argument("--partial", action="store_true")
    c.set_defaults(fn=cmd_collect)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
