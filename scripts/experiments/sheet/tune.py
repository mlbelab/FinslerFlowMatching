"""How the Sheet hyper-parameters were found — the record, and the thing that found them.

    python -m scripts.experiments.sheet.tune plan             # the grid, both waves
    python -m scripts.experiments.sheet.tune run --index 7    # one cell = one job
    python -m scripts.experiments.sheet.tune collect          # rank, print TUNED

The notebook used to carry this search inline: three cells that swept, ranked and then
went straight on to the reported runs.  That is the shape a search takes while it is still
being designed, and the wrong one to publish, for two reasons.  A reader cannot tell which
of the notebook's numbers were *chosen* and which were *measured*, and the twenty-eight
grid cells cost more wall-clock than everything else in the notebook put together, so the
notebook could not be re-executed to check a figure without redoing the search.  The
search moved here; the notebook now pastes the literal this module prints, and its
``verify`` suite diffs the two so they cannot drift.

**One cell is one job, and the cell index is the contract.**  Cell ``i`` of wave ``w`` is
the same (arm, point) in every process and on every machine, because
:class:`scripts.method.tuning.Fanout` is the only place the search is enumerated.  A
scheduler therefore only ever needs a range of integers, and ``plan`` prints that loop
with the submission command left as a placeholder — nothing about a queue, a partition or
an account appears anywhere in this repository.

One space, so no ``--dim``
--------------------------
Unlike the two real-data trees, the Sheet lives in exactly one space: a 3-D saddle at one
fixed 70/15/15 split.  There is nothing to search independently *per* anything, so this
tuner has no ``--dim`` flag and ``TUNED_DIMS`` is empty — the flag would be a knob with
one setting, and :func:`scripts.method.tuning.preflight_cell` rejects it explicitly rather
than accepting and ignoring it.

Two waves, because Path B inherits Path A's point
-------------------------------------------------
``pathb`` reuses the frozen interpolant and the coupling ``ffm`` produced — that is what
makes the two rows one method with the SDE off and on rather than two methods — so its
:math:`\\sigma` ladder can only be searched once :math:`(\\rho, \\lambda)` is fixed.  Wave
1 is everything deterministic (25 cells: ours 15, the three baselines 10), then
``collect``, then wave 2 (``pathb``'s 3 :math:`\\sigma` rungs), then ``collect`` again.

The baselines are swept too, and that is the point
--------------------------------------------------
Giving ours a search and the baselines none would be the comparison tuning itself, so
``mfm_land`` and ``curly`` get their own ladders on the same objective, the same split,
the same seed and the same trainer; ``cfm`` has no knob and still gets a cell, so its row
is produced by the identical path.  Curly-FM's ladder was widened after its argmin hit the
top rung — the repository's widening rule, applied to the baselines first.

The one selection objective
---------------------------
Each cell writes the two terms :mod:`scripts.core.selection` averages — ``W2_endpoint``
against the val cells of :math:`p_1`, and ``W2_intermediate`` at the crossing against the
val half of the withheld strip — and the averaged scalar beside them.  Storing the terms
rather than only the average is what makes a re-rank free.

Every cell trains on the 70 % train split with ``P`` rebuilt on it, and is scored against
val cells only.  The test half of the strip, which every *reported* number is measured on,
is disjoint from the val half and is never read here.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

from scripts.core.arms import HParams, run_arm
from scripts.core.paths import experiment_root, rel
from scripts.core.selection import OBJECTIVE_KEY, OBJECTIVE_TERMS
from scripts.method import (FWFactory, Fanout, Grid, Run, c_eff_of, cell_tag,
                            cuda_warm_up, format_tuned, moments_from, path_a, path_b,
                            preflight_cell, print_ladder, rank, require_complete,
                            sigma_for, smoke_root)

from .datasets import make_dataset
from .split import make_split, split_route_eval, split_training_set, val_objective

# --------------------------------------------------------------------------- #
#  The search
# --------------------------------------------------------------------------- #
#  Every ladder is a multiple of a scale the *data* sets -- ``rho_star`` from the second
#  moment of P, ``a_bar`` from the first, the kernel bandwidth for sigma -- so no rung is
#  ever quoted as a bare number and a rung means the same thing if the cloud is redrawn.
#
#: half-decades around the measured ``rho_star``.
RHO_MULTS = (0.3, 1.0, 3.0)
#: half-decades bracketing the measured ``a_bar``; the argmin has never been on either end.
LAM_MULTS = (0.1, 0.3, 1.0, 3.0, 10.0)
#: Path B's sigma, as a bridge half-width in kernel bandwidths at t = 1/2.  Five rungs and
#: not three: the argmin hit the old bottom rung of 0.3 and the score was still falling as
#: it left (+10.1% one rung in, well past ``EDGE_FLAT_PCT``), so the ladder was extended
#: downwards.  Widening *ours* is the case the "baselines first" rule guards against, so
#: it is licensed here only because no baseline is on a live edge at the same time --
#: ``land_rho`` sits on its bottom rung but flat (+0.7%, converged) and ``curly_alpha``
#: and both ``ffm`` axes are interior.
SIGMA_MULTS = (0.03, 0.1, 0.3, 1.0, 3.0)

#: LAND's floor.  ``land_gamma`` is *not* swept here and that is deliberate: on a 3-D
#: cloud ``make_mfm_metric`` resolves to the conformal RBF metric, whose bandwidths come
#: from its own KMeans clusters and where gamma is dropped -- a gamma ladder returns four
#: identical rows, which is a wasted quarter of the baseline's budget rather than a search.
LAND_RHOS = (1e-3, 3.16e-3, 1e-2, 3.16e-2)
#: Curly-FM's loss weight.  Five rungs and not three: the argmin hit the old top rung of
#: 1.0, so the ladder was extended upwards.  The widening rule is applied to the baselines
#: first, and this is the case it was written for.
CURLY_ALPHAS = (0.01, 0.1, 1.0, 3.16, 10.0)

#: wave 1: everything whose point can be chosen without another arm's answer.
WAVE1 = Fanout({
    "ffm":      Grid(rho_mult=RHO_MULTS, lam_mult=LAM_MULTS),
    "mfm_land": Grid(land_rho=LAND_RHOS),
    "curly":    Grid(curly_alpha=CURLY_ALPHAS),
    "cfm":      Grid(),                       # nothing to select; still one cell, so the
})                                            # row exists and is measured like the others
#: wave 2: Path B, on Path A's chosen metric and its frozen backbone.
WAVE2 = Fanout({"pathb": Grid(width_mult=SIGMA_MULTS)})
WAVES = {1: WAVE1, 2: WAVE2}

#: this tree searches one space, so there is no per-dimension axis.  Kept as an (empty)
#: module constant because it is part of the tuner contract every tree exposes, and
#: ``preflight_cell`` reads it to decide whether ``--dim`` means anything.
TUNED_DIMS: tuple[int, ...] = ()

#: the scalar every cell is ranked on.  Defined once, in :mod:`scripts.core.selection`.
SELECTION_KEY = OBJECTIVE_KEY

#: how steep the score has to be at an endpoint before "the argmin sits on the edge" means
#: the ladder was too narrow rather than converged.  A reporting tolerance, not a noise
#: floor: the run-to-run spread is far below 5 %, so a flatter tail than this is a real
#: ordering that is nonetheless smaller than the five-seed error bars the notebook
#: prints beside it.
EDGE_FLAT_PCT = 5.0

#: engine arm names for the three baselines; ours are ``scripts.method`` calls.
ENGINE_ARM = {"cfm": "cfm", "mfm_land": "mfm_land", "curly": "curly"}
#: ``run_arm`` asserts ``n_steps % 3 == 0``; 102 against the notebook's 100 is a 2 % grid.
BASE_N_STEPS = 102

ROOT = experiment_root("sheet_bench")
#: where ``collect`` writes its answer: *beside this file*, not in the run tree.  The run
#: tree is scratch and is not versioned; the tuned point is a result — it is the literal
#: the notebook pastes and the thing ``verify`` diffs that paste against — so it ships
#: with the source and survives a deleted output directory.  One selection rule, one file.
TUNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


# --------------------------------------------------------------------------- #
#  The cloud, built once per process
# --------------------------------------------------------------------------- #
def build(cfg: Run):
    """The Sheet, its fixed split, the training view of ``P``, and the two moments.

    Everything below the split is a pure function of it, so a cell and the notebook build
    the identical object: the split is drawn at ``cfg.split_seed`` and held there, and
    ``P`` is *rebuilt* on the 70 % rather than sliced out of the full matrix.
    """
    ds = make_dataset("Sheet")
    split = make_split(ds, seed=cfg.split_seed)
    ts = split_training_set(ds, split)

    X = np.asarray(ts.X, dtype=np.float64)
    P = sp.csr_matrix(ts.P)
    b0_pts, D_pts = moments_from(P, X)
    fw = FWFactory(X, D_pts, b0_pts, cfg)

    X_t = cfg.tensor(X)
    X0_t = X_t[torch.as_tensor(ts.p0, device=cfg.device)]
    X1_t = X_t[torch.as_tensor(ts.p1, device=cfg.device)]
    return ds, split, ts, X, P, b0_pts, fw, X0_t, X1_t


# --------------------------------------------------------------------------- #
#  One cell
# --------------------------------------------------------------------------- #
def cell_path(root: str, wave: int, name: str, point: dict) -> str:
    """Where one cell's record lands: keyed by the point it measures, not by the flat
    index it was submitted at, so widening a ladder cannot alias one arm's record onto
    another arm's cell.  One function, so preflight and the write agree."""
    return os.path.join(root, "sweep", cell_tag(wave, name, point) + ".json")


def run_cell(wave: int, index: int, cfg: Run, root: str = ROOT,
             tuned: dict | None = None, force: bool = False) -> dict:
    """Train one (arm, point) on the 70 % split and write the objective's two terms.

    ``tuned`` is wave 1's answer and is required by wave 2 only: ``pathb`` is scored on
    ``ffm``'s metric and its frozen backbone, so a sigma chosen at any other point would
    be measuring a different model.
    """
    arm, point, path = preflight_cell(
        WAVES, wave, index, None, TUNED_DIMS, root,
        lambda n, p: cell_path(root, wave, n, p), force=force)
    ds, split, ts, X, P, b0_pts, fw, X0_t, X1_t = build(cfg)

    # Twenty discarded iterations at the least-conditioned corner, before anything that is
    # kept, so every point on the ladder is scored under the same numerics -- see
    # ``scripts.method.phases.cuda_warm_up``.  One cell is one process, so it runs here
    # rather than once per sweep.
    cuda_warm_up(fw.at(rho=fw.rho_rungs(RHO_MULTS)[0], lam=fw.lam_rungs(LAM_MULTS)[0]),
                 X0_t, X1_t, cfg)

    t0 = time.time()
    traj = _train(arm, point, ds, ts, P, X, b0_pts, fw, X0_t, X1_t, cfg, tuned=tuned)
    terms = val_objective(traj, ds, split, seed=0)

    rec = {
        "wave": wave, "index": index, "arm": arm, "point": point, **terms,
        "n_train": int(len(ts.idx)), "seed": cfg.tune_seed,
        "train_s": round(time.time() - t0, 1),
    }
    with open(path, "w") as fh:                      # preflight made the directory
        json.dump(rec, fh, indent=1)
    print(f"[cell] w{wave} i{index:>3}  {arm:<9} {point}  "
          f"objective {rec[SELECTION_KEY]:.4f}  (end {rec['W2_endpoint']:.4f}, "
          f"mid {rec['W2_intermediate']:.4f})  [{rec['train_s']:.0f}s] -> {path}")
    return rec


def _train(arm: str, point: dict, ds, ts, P, X, b0_pts, fw, X0_t, X1_t, cfg: Run,
           tuned: dict | None):
    """Dispatch one arm to the code path the notebook uses, and return its trajectory.

    Ours go through :mod:`scripts.method` and the baselines through the engine's
    ``run_arm``, which is exactly the split the notebook makes — a point selected through
    a different trainer than the one that consumes it would not be tuned for anything.
    """
    if arm in ENGINE_ARM:
        # Curly-FM's velocity channel is the first moment of P, raw.  Never the generator
        # field: only P is public, and the notebook asserts the same thing.
        ts_base = dataclasses.replace(ts, velocity=b0_pts)
        assert not np.allclose(ts_base.velocity, ds._v_gen[ts.idx]), \
            "a baseline must never see the generator velocity field -- only P is public"
        hp = HParams(**cfg.arm_kw, **point)
        res = run_arm(ENGINE_ARM[arm], ts_base, hp, cfg.tune_seed, cfg.device, cfg.dtype,
                      n_steps=BASE_N_STEPS)
        return np.asarray(res.traj, dtype=np.float64)

    if arm == "ffm":
        mt = fw.at(rho=fw.rho_rungs((point["rho_mult"],))[0],
                   lam=fw.lam_rungs((point["lam_mult"],))[0])
        return path_a(mt, X0_t, X1_t, cfg.tune_seed, cfg)["traj"]

    assert arm == "pathb", f"unknown arm {arm!r}"
    assert tuned is not None, "wave 2 needs wave 1's tuned point: run `collect` first"
    a_point = tuned["ffm"]
    mt = fw.at(rho=fw.rho_rungs((a_point["rho_mult"],))[0],
               lam=fw.lam_rungs((a_point["lam_mult"],))[0])
    det = path_a(mt, X0_t, X1_t, cfg.tune_seed, cfg)         # the backbone pathb inherits
    sigma = sigma_for(mt, point["width_mult"])
    return path_b(mt, det["phi"], det["coupler"], X0_t, sigma, cfg.tune_seed, cfg)["traj"]


# --------------------------------------------------------------------------- #
#  Collect
# --------------------------------------------------------------------------- #
def collect(root: str = ROOT, quiet: bool = False, partial: bool = False) -> dict:
    """Rank every finished cell and return ``{arm: point}``.

    A wave that has **not been started** is skipped: that is the ordinary state between
    the two waves here, since wave 2 cannot run until this function has told it which
    point ``ffm`` won.  A wave that has been started and is *incomplete* is fatal, unless
    ``partial`` -- ranking three rungs of five and publishing the winner is how a search
    quietly reports the argmin of a prefix.
    """
    sweep = os.path.join(root, "sweep")
    assert os.path.isdir(sweep), (
        f"no sweep records under {sweep}; run wave 1 first "
        f"(`python -m scripts.experiments.sheet.tune plan` prints the loop)")
    rows = []
    for name in sorted(os.listdir(sweep)):
        if name.endswith(".json"):
            with open(os.path.join(sweep, name)) as fh:
                rows.append(json.load(fh))

    tuned: dict = {}
    missing: list = []
    for wave, fan in WAVES.items():
        started = any(r["wave"] == wave for r in rows)
        for arm, grid in fan.grids.items():
            cells = [r for r in rows if r["arm"] == arm]
            if started and len(cells) < len(grid):
                missing.append((f"w{wave} {arm}", len(cells), len(grid)))
            if not cells:
                if not quiet:
                    print(f"  {arm:<9} -- no cell yet (wave {wave}, {len(grid)} cells)")
                continue
            df, best = rank(cells, SELECTION_KEY)
            tuned[arm] = best["point"]
            if not quiet:
                print_ladder(df, best, grid, header=arm, key=SELECTION_KEY,
                             terms=OBJECTIVE_TERMS, flat_pct=EDGE_FLAT_PCT,
                             n_have=len(cells), n_want=len(grid))
    require_complete(missing, partial=partial, quiet=quiet)
    return tuned


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def cmd_plan(args) -> None:
    """Print the search and the submission loop, with the scheduler left blank."""
    if args.count is not None:
        print(len(WAVES[args.count]))
        return
    for wave, fan in WAVES.items():
        print(f"\n=== wave {wave} ===")
        print(fan.describe())
    print(f"""
Submit each wave as an array over the cell index.  `<submit>` is your own launcher --
nothing about it belongs in this repository:

    for i in $(seq 0 {len(WAVE1) - 1}); do
      <submit> python -m scripts.experiments.sheet.tune run --wave 1 --index $i
    done
    python -m scripts.experiments.sheet.tune collect          # wave 1's points

    for i in $(seq 0 {len(WAVE2) - 1}); do
      <submit> python -m scripts.experiments.sheet.tune run --wave 2 --index $i
    done
    python -m scripts.experiments.sheet.tune collect          # both waves; paste TUNED

Serially and without a scheduler it is the same two loops with `<submit>` deleted;
budget ~{len(WAVE1) + len(WAVE2)} cells at a few GPU-minutes each.""")


def cmd_run(args) -> None:
    # A wiring check writes a real-looking record under a content-addressed name,
    # so it goes to its own tree rather than into the sweep `collect` ranks.
    args.root = smoke_root(args.root, ROOT, "sheet_bench", args.smoke)

    # Preflight before anything expensive -- before the device is resolved and before wave
    # 2 reads wave 1's records -- so a bad invocation dies in second one.
    preflight_cell(WAVES, args.wave, args.index, None, TUNED_DIMS, args.root,
                   lambda n, p: cell_path(args.root, args.wave, n, p), force=args.force)

    cfg = Run.from_env(**({"device": torch.device("cpu")} if args.cpu else {}),
                       **({"smoke": True} if args.smoke else {}))
    tuned = None
    if args.wave == 2:
        # a *read* of wave 1's answer, not a publication: wave 2 is unfinished by
        # construction here, so the completeness guard is waived and silenced.
        tuned = collect(args.root, quiet=True, partial=True)
        assert "ffm" in tuned, "wave 2 needs wave 1's ffm cells; run wave 1 and `collect`"
    run_cell(args.wave, args.index, cfg, root=args.root, tuned=tuned, force=args.force)


def cmd_collect(args) -> None:
    tuned = collect(args.root, partial=args.partial)
    if args.partial:
        # An incomplete search may be *read* but never *published*: writing tuned.json
        # here would leave the notebook's paste-diff check passing against an answer the
        # search had not finished computing.
        print(format_tuned(tuned))
        return
    with open(TUNED_FILE, "w") as fh:
        json.dump(tuned, fh, indent=1)
    print(f"\nwrote {rel(TUNED_FILE)}\n\n# paste into notebooks/sheet_ffm_paper.ipynb\n"
          f"# from scripts/experiments/sheet/tune.py collect")
    print(format_tuned(tuned))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    # `--count` prints one integer and nothing else, so a launcher can size its array
    # from the grid rather than from a number that a widened rung would stale.
    p.add_argument("--count", type=int, default=None, metavar="WAVE",
                   choices=sorted(WAVES))
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--wave", type=int, default=1, choices=sorted(WAVES))
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--smoke", action="store_true",
                   help="tiny budgets, into a separate _smoke tree: proves a "
                        "cluster cell runs before the array is submitted")
    r.add_argument("--force", action="store_true",
                   help="re-measure a cell whose record already exists")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("collect")
    c.add_argument("--partial", action="store_true",
                   help="rank a search that is still running; writes no tuned.json")
    c.set_defaults(fn=cmd_collect)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
