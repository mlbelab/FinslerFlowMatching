"""How the Pancreas hyper-parameters were found — the record, and the thing that found them.

The notebook hardcodes a tuned point per dimension and per arm.  A hardcoded number is
only honest if the search that produced it is *published and re-runnable*, so this module
is both: read top to bottom it is the protocol, and run it re-does the search.

    python -m scripts.experiments.pancreas.tune plan                    # the grid, both waves
    python -m scripts.experiments.pancreas.tune run --dim 20 --index 7  # one cell = one job
    python -m scripts.experiments.pancreas.tune collect                 # rank, print TUNED

**One cell is one job, and the cell index is the contract.**  Cell ``i`` of wave ``w`` at
dimension ``d`` is the same (arm, point) in every process and on every machine, because
:class:`scripts.method.tuning.Fanout` is the only place the search is enumerated — the notebook's
in-line sweep and this fan-out read the same object.  So a scheduler only ever needs to
be told a range of integers, and ``plan`` prints that loop with the submission command
left as a placeholder.  Nothing about a queue, a partition, a module system or an account
appears anywhere in this repository -- a ``verify`` check greps for the vocabulary --
so the record of the search is portable to whatever queue the reader has.

Two waves, because Path B inherits Path A's point
-------------------------------------------------
``pathb`` reuses the frozen interpolant and the coupling ``ffm`` produced — that is what
makes the two columns one method with the SDE off and on rather than two methods — so its
:math:`\\sigma` ladder can only be searched once :math:`(\\rho, \\lambda)` is fixed.  Hence
wave 1 (everything deterministic, 95 cells per dimension) then ``collect``, then wave 2
(``pathb``'s eight :math:`\\sigma` rungs, 8 cells per dimension) then ``collect`` again.
This is the Sheet's ablation-inherit rule and the notebook's own ordering.  Over the four
searched dimensions that is 380 + 32 jobs; ``plan --count <wave>`` prints the per-wave
number so a launcher sizes its array off the grid rather than off this sentence.

Independently per dimension, all four of them
---------------------------------------------
The search runs at ``TUNED_DIMS`` = :math:`d = 2, 10, 20, 50`, separately at each.
``rho`` floors a metric of squared displacements and ``sigma`` is an absolute noise level,
and an unstandardised PCA prefix twice as wide carries neither across — the erythroid tree
found the same and says so.  The ladders are therefore quoted as multiples of scales the
data itself sets (:math:`\\rho_\\star` from the second moment of ``P``, :math:`\\bar a`
from the first, the kernel bandwidth for :math:`\\sigma`), so the *rungs* transfer even
though the values do not.

An earlier pass searched only the three PCA prefixes and let the 2-D UMAP column inherit
:math:`d = 50`'s point, on the evidence that ours had picked ``rho_mult = 0.03`` at all
three.  Searching it did not confirm that: the UMAP column lands elsewhere, and its
score surface turns out to be **ragged in** ``rho`` rather than monotone, so three
agreeing prefixes were never evidence about a fourth chart of a different kind.  The
inheritance is gone and every reported column has its own fan-out.

Ours searches a product grid and each baseline searches a ladder
----------------------------------------------------------------
This is the one asymmetry in the search that is *not* in the baselines' favour, so it is
stated here rather than left to be read off :data:`WAVE1`.  ``ffm`` has two knobs and they
interact — ``rho`` floors the metric, ``lam`` sets how much of the 1-form survives, and
``c_eff`` is a function of both — so it is searched over the full product, ten
``rho_mult`` rungs by eight ``lam_mult`` rungs, **80 cells per dimension**.  Each baseline
has exactly one knob and therefore one ladder: ``mfm_land`` 8 rungs, ``curly`` 6, ``cfm``
none.  Ours is chosen from ten times as many trained models as any baseline is.

That follows from the parameterisations rather than from a choice about who gets a search
— a one-knob method has nothing to put in a second axis — and everything around it is held
identical: the same half-decade spacing, the same objective, the same split, the same
widening rule.  What makes the asymmetry defensible rather than merely explained is the
last of those: the rule is applied to the *baselines first*, and **no baseline ladder was
ever left with its argmin on an endpoint** (see :data:`LAND_RHOS` and
:data:`CURLY_ALPHAS`, both extended downwards off exactly that condition).  A short ladder
that has converged in the interior has not been shortchanged; a short ladder that stops at
its own argmin has.  Only the first kind is reported.

The one selection objective
---------------------------
Each cell writes the two terms :mod:`scripts.core.selection` averages — ``W2_endpoint``
against the val cells of :math:`p_1`, and ``W2_intermediate`` at :math:`t = 1/2` against
the 10 % selection slice of the withheld marginal — and the averaged scalar beside them.
Storing the terms rather than only the average is what makes a re-rank free: a third term
could be added to the objective without retraining a single cell.

A sweep cell trains on the 85 % split (:func:`~scripts.experiments.pancreas.datasets.split_shown`)
with ``P`` and the geometry rebuilt on that subset, so the endpoint term is measured out of
sample; the notebook then refits the chosen point on all kept cells.  Every *reported*
number is measured on the disjoint 90 % of the withheld marginal, which the tuner never
sees.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from scripts.experiments.pancreas.datasets import (
    CACHE_DIR, TRAIN_BINS, TUNED_DIMS, build_training_set, load_cloud, split_holdout,
    split_shown)
from scripts.core.arms import HParams, run_arm
from scripts.core.paths import experiment_root, rel
from scripts.core.selection import OBJECTIVE_KEY, OBJECTIVE_TERMS, with_objective
from scripts.method import (FWFactory, Fanout, Grid, Run, c_eff_of, cell_tag,
                            euclidean_w2, format_tuned, moments_from, path_a, path_b,
                            preflight_cell, print_ladder, rank, require_complete,
                            sigma_for, smoke_root)

# --------------------------------------------------------------------------- #
#  The search
# --------------------------------------------------------------------------- #
#  WHY EACH LADDER IS AS WIDE AS IT IS.  Every extent below was set by the repository's
#  widening rule -- an argmin on an endpoint means the ladder was too narrow rather than
#  that a point was chosen, unless the score has gone flat there (:data:`EDGE_FLAT_PCT`)
#  -- and the rule is applied to the *baselines first*, since ours is the arm a quietly
#  wider search would flatter.
#
#  Two earlier passes set these extents, and they are kept in full even where the surface
#  that triggered them is gone.  Pass 1 ran before the arms were put onto a single shared
#  architecture, when each baseline still trained the network its own reference ships.
#  Pass 2 re-measured the whole grid on the shared nets.  Both passes carried *two*
#  competing selection protocols, and several of the widenings below were asked for by the
#  leak-free one -- a proxy this repository no longer has, since the collapse to the single
#  objective in :mod:`scripts.core.selection`.  Those rungs stay: a ladder is only ever
#  widened, never narrowed to fit the latest surface, because a rung that has gone interior
#  costs one cell and a rung that is missing costs a re-run.
#
#  Pass 3 is the search under the one objective, and it re-audits every edge from scratch.
#  Until it has run, treat the per-ladder numbers quoted below as the record of how the
#  extent was arrived at rather than as a statement about the current surface.
#
#: half-decade rungs around the measured scales.  Both are multipliers, never absolutes:
#: ``rho_mult`` is in units of ``rho_star`` and ``lam_mult`` in units of ``a_bar``, so a
#: rung means the same thing at d = 10 and d = 50 even though the value does not.
#: It reaches three rungs below :math:`\\rho_\\star` because the search kept asking it to:
#: pass 1 bottomed out at ``0.3`` and pass 2 at ``0.1``, at every dimension, and the score
#: was monotone in it over the whole ladder (d = 50: 14.11 at ``3.0`` down to 12.65 at
#: ``0.1``), so the last two rungs were added together rather than one at a time.  It then
#: had to grow at the *other* end too, and only the UMAP column asked: at d = 2 the argmin
#: landed on the old top ``3.0`` at 1.046 against 1.441 one rung in (+37.8 %), far past
#: :data:`EDGE_FLAT_PCT`.  No baseline was on an edge at that dimension, so the rule fired
#: for ours alone -- the direction it is least comfortable in, and exactly why it is
#: applied mechanically rather than by judgement.  A stiffer metric where the coordinates
#: are a 2-D embedding is the expected shape of it: ``rho_star`` is set by the displacement
#: scale, and UMAP's is not the PCA prefixes'.  The two d = 2 edges together say that this
#: column's surface is **ragged in rho rather than monotone** -- along ``lam_mult = 0.03``
#: it ran 1.49, 0.94, 1.30, 1.83, 1.63, 1.72, 1.26, 1.42, 1.36 across nine rungs (pass 1)
#: -- so an endpoint winner here was never evidence of a trend running off the grid.  That
#: is why the rule is "widen until the argmin comes off the edge" and not "widen while the
#: score improves".
RHO_MULTS = (3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
#: ``lam_mult``, on the same half-decade spacing, in units of :math:`\\bar a`.  Widened
#: downwards twice in pass 1 and once more in pass 2, each time off an argmin sitting on
#: the bottom rung with a gradient well past :data:`EDGE_FLAT_PCT` (+25 %, +11 %, +13.7 %
#: one rung in).  A ninth rung was refused, and mechanically: the winner on the proposed
#: new bottom was a cloud **collapsed** onto the training cells, which
#: :data:`PROXY_REJECT_PCT` is what caught.  A widening that chases a degenerate model is
#: not a wider search, it is a worse one.
LAM_MULTS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
#: Path B's bridge half-width, in kernel bandwidths -- the notebook's own convention.
#: Widened downwards more sharply than any other ladder: pass 1 put the argmin on ``0.3``,
#: the old bottom, at d = 20 and d = 50, and the gradient off that edge is not subtle
#: (d = 50: 13.44 at ``0.3`` against 24.35 at ``1.0`` and 47.82 at ``3.0``).  A bridge
#: whose half-width is a whole kernel bandwidth is simply too wide in an unstandardised
#: 50-dimensional prefix -- the diffusion, not the geometry, then decides where a sample
#: lands.  It bottomed out again at ``0.03`` (d = 20), so two more rungs went on; d = 10
#: was interior throughout.  The top rung ``10.0`` is absurd in a PCA prefix and exists for
#: the UMAP chart alone, which is z-scored, two-dimensional, and whose bandwidth is a
#: different length.
SIGMA_MULTS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
#: the baselines' own single knob each, on the same half-decade spacing.  They are given a
#: search because giving ours one and not theirs would be the comparison tuning itself --
#: and the widening rule applies to them the same way, first.  Both ladders started at the
#: third rung and both were extended two rungs down when the **UMAP** column put its argmin
#: on that bottom edge; the three PCA columns had been interior throughout.  That the chart
#: with the smallest coordinate scale is the one that wanted a smaller regulariser is what
#: one would expect of an absolute knob, and it is the reason each column is ranked in its
#: own space rather than handed d = 50's answer.  ``land_rho`` then took two further rungs
#: down in pass 2, both off endpoint argmins steep enough to count (+6.7 % and +7.8 % one
#: rung in) -- for a baseline, which is the direction the "baselines first" half of the
#: rule exists to protect.
LAND_RHOS = (1e-5, 3.16e-5, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2, 3.16e-2)
CURLY_ALPHAS = (1e-3, 3.16e-3, 0.01, 0.1, 1.0, 3.16)
#: wave 1: everything whose point can be chosen without another arm's answer.
WAVE1 = Fanout({
    "ffm":      Grid(rho_mult=RHO_MULTS, lam_mult=LAM_MULTS),
    "mfm_land": Grid(land_rho=LAND_RHOS),
    "curly":    Grid(curly_alpha=CURLY_ALPHAS),
    "cfm":      Grid(),                       # nothing to select; still one cell, so the
})                                            # row exists and is measured like the others
#: wave 2: Path B, at Path A's chosen metric.
WAVE2 = Fanout({"pathb": Grid(width_mult=SIGMA_MULTS)})
WAVES = {1: WAVE1, 2: WAVE2}

#: the scalar every cell is ranked on.  Defined once, in :mod:`scripts.core.selection`;
#: named here only so a record and a ranker cannot disagree about the spelling.
SELECTION_KEY = OBJECTIVE_KEY

#: how steep the score has to be at an endpoint before "the argmin sits on the edge" means
#: the ladder was too narrow rather than converged.  It is a **reporting tolerance, not a
#: noise floor**, and the difference matters: one cell re-trained three times moved its
#: midpoint term by only 0.012 %
#: (``mfm_land``, d = 20, ``land_rho`` 1e-4: 11.4869 / 11.4883 / 11.4883).  So a 1 % tail
#: *is* a real ordering; it is simply an ordering smaller
#: than the three-seed error bars the notebook reports, and a selection difference that
#: cannot move a reported digit is not worth another rung.  The edges the widening rule was
#: written for are nowhere near it: ``pathb`` at d = 50 went 13.44 to 24.35 across one rung
#: (+81 %), ``curly`` at d = 10 goes 11.58 to 12.32 to 13.95, and ``ffm``'s ``lam_mult``
#: was +25 % one rung in -- all three were widened.
EDGE_FLAT_PCT = 5.0

#: engine arm names for the three baselines; ours are scripts.method calls and have no entry.
ENGINE_ARM = {"cfm": "cfm", "mfm_land": "mfm_land", "curly": "curly"}
#: ``run_arm`` asserts ``n_steps % 3 == 0``; 102 against the notebook's 100 is a 2 % grid.
BASE_N_STEPS = 102

ROOT = experiment_root("pancreas_bench")
#: where ``collect`` writes its answer.  There is one selection rule, so there is one file;
#: ``verify`` diffs the notebook's hardcoded ``TUNED`` against it.
#: where ``collect`` writes its answer: *beside this file*, not in the run tree.  The
#: run tree is scratch and is not versioned; the tuned point is a result — it is the
#: literal the notebook pastes and the thing ``verify`` diffs that paste against — so it
#: ships with the source and survives a deleted output directory.
TUNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


# --------------------------------------------------------------------------- #
#  One cell
# --------------------------------------------------------------------------- #
def cell_path(root: str, dim: int, wave: int, name: str, point: dict) -> str:
    """Where one cell's record lands: keyed by the point it measures, not by the flat
    index it was submitted at, so widening a ladder cannot alias one arm's record onto
    another arm's cell.  One function, so preflight and the write agree."""
    return os.path.join(root, "sweep",
                        cell_tag(wave, name, point, dim=dim) + ".json")


def run_cell(dim: int, wave: int, index: int, cfg: Run, root: str = ROOT,
             tuned: dict | None = None, force: bool = False) -> dict:
    """Train one (arm, point) on the 85 % split and write the objective's two terms.

    ``tuned`` is wave 1's answer and is required by wave 2 only: ``pathb`` is scored at
    ``ffm``'s metric, so a :math:`\\sigma` chosen at any other point would be measuring a
    different model.
    """
    arm, point, path = preflight_cell(
        WAVES, wave, index, dim, TUNED_DIMS, root,
        lambda n, p: cell_path(root, dim, wave, n, p), force=force, cache=CACHE_DIR)
    cloud = load_cloud(dim)
    i_train, i_val = split_shown(cloud)
    ts = build_training_set(cloud, subset=i_train)

    t0 = time.time()
    traj = _train(arm, point, ts, cfg, dim=dim, tuned=tuned)

    # the objective's two marginals.  ``i_end`` is the val slice of p_1 among the *kept*
    # cells -- the arm never trained on it, so the endpoint term is out of sample -- and
    # ``i_sel`` is the 10 % selection slice of the withheld marginal, which nothing ever
    # trains on under any protocol.  The disjoint 90 % is what the notebook reports.
    i_end = i_val[cloud.bin_id[i_val] == TRAIN_BINS[1]]
    assert len(i_end) > 1, f"d{dim}: val slice of p1 is too small to score ({len(i_end)})"
    i_sel, _ = split_holdout(cloud)

    terms = with_objective({
        "W2_endpoint": euclidean_w2(traj[-1], cloud.X[i_end], seed=cfg.tune_seed),
        "W2_intermediate": euclidean_w2(traj[traj.shape[0] // 2], cloud.X[i_sel],
                                        seed=cfg.tune_seed),
    })
    rec = {
        "dim": dim, "wave": wave, "index": index, "arm": arm, "point": point, **terms,
        "n_end_cells": int(len(i_end)), "n_val_cells": int(len(i_sel)),
        "n_train": int(len(i_train)),
        "seed": cfg.tune_seed, "train_s": round(time.time() - t0, 1),
        "P_asymmetry": ts.diag["P_asymmetry"], "cos_J_velocity": ts.diag["cos_J_velocity"],
    }
    with open(path, "w") as fh:                      # preflight made the directory
        json.dump(rec, fh, indent=1)
    print(f"[cell] d{dim} w{wave} i{index:>3}  {arm:<9} {point}  "
          f"objective {rec[SELECTION_KEY]:.4f}  (end {rec['W2_endpoint']:.4f}, "
          f"mid {rec['W2_intermediate']:.4f})  [{rec['train_s']:.0f}s] -> {path}")
    return rec


def _train(arm: str, point: dict, ts, cfg: Run, dim: int, tuned: dict | None):
    """Dispatch one arm to the code path the notebook uses, and return its trajectory.

    Ours go through ``scripts.method`` and the baselines through the engine's ``run_arm``, which is
    exactly the split the notebook makes — a tuned point selected through a different
    trainer than the one that consumes it would not be tuned for anything.
    """
    X = np.asarray(ts.X, dtype=np.float64)
    X0 = cfg.tensor(X[ts.p0])
    X1 = cfg.tensor(X[ts.p1])

    if arm in ENGINE_ARM:
        hp = HParams(**cfg.arm_kw, **point)
        res = run_arm(ENGINE_ARM[arm], ts, hp, cfg.tune_seed, cfg.device, cfg.dtype,
                      n_steps=BASE_N_STEPS)
        return np.asarray(res.traj, dtype=np.float64)

    b0_pts, D_pts = moments_from(ts.P, X)
    if arm == "ffm":
        fw = FWFactory(X, D_pts, b0_pts, cfg, rho_mult=point["rho_mult"],
                       lam_mult=point["lam_mult"])
        return path_a(fw.base, X0, X1, cfg.tune_seed, cfg)["traj"]

    assert arm == "pathb", f"unknown arm {arm!r}"
    assert tuned is not None, "wave 2 needs wave 1's tuned point: run `collect` first"
    a_point = tuned[dim]["ffm"]
    fw = FWFactory(X, D_pts, b0_pts, cfg, rho_mult=a_point["rho_mult"],
                   lam_mult=a_point["lam_mult"])
    det = path_a(fw.base, X0, X1, cfg.tune_seed, cfg)      # the backbone pathb inherits
    sigma = sigma_for(fw.base, point["width_mult"])
    return path_b(fw.base, det["phi"], det["coupler"], X0, sigma, cfg.tune_seed,
                  cfg)["traj"]


# --------------------------------------------------------------------------- #
#  Collect
# --------------------------------------------------------------------------- #
def collect(root: str = ROOT, dims=TUNED_DIMS, quiet: bool = False,
            partial: bool = False) -> dict:
    """Rank every finished cell and return ``{dim: {arm: point}}``.

    A wave that has **not been started** is skipped: that is the ordinary state between
    the two waves here, since wave 2 cannot run until this function has told it which
    point ``ffm`` won.  A wave that has been started and is *incomplete* is fatal, unless
    ``partial`` -- ranking two rungs of nine and publishing the winner is how a 380-cell
    search quietly reports the argmin of whichever cells the queue happened to finish.
    """
    sweep = os.path.join(root, "sweep")
    assert os.path.isdir(sweep), (
        f"no sweep records under {sweep}; run wave 1 first "
        f"(`python -m scripts.experiments.pancreas.tune plan` prints the loop)")
    rows = []
    for name in sorted(os.listdir(sweep)):
        if name.endswith(".json"):
            with open(os.path.join(sweep, name)) as fh:
                rows.append(json.load(fh))

    tuned: dict = {}
    missing: list = []
    for dim in dims:
        tuned[dim] = {}
        for wave, fan in WAVES.items():
            started = any(r["dim"] == dim and r["wave"] == wave for r in rows)
            for arm, grid in fan.grids.items():
                cells = [r for r in rows if r["dim"] == dim and r["arm"] == arm]
                if started and len(cells) < len(grid):
                    missing.append((f"d{dim} w{wave} {arm}", len(cells), len(grid)))
                if not cells:
                    if not quiet:
                        print(f"  d{dim:<3} {arm:<9} -- no cell yet (wave {wave})")
                    continue
                df, best = rank(cells, SELECTION_KEY)
                tuned[dim][arm] = best["point"]
                if not quiet:
                    print_ladder(df, best, grid, header=f"d{dim}  {arm}",
                                 key=SELECTION_KEY, terms=OBJECTIVE_TERMS,
                                 flat_pct=EDGE_FLAT_PCT, n_have=len(cells),
                                 n_want=len(grid))
    require_complete(missing, partial=partial, quiet=quiet)
    return tuned


#: the two arms that are one method with the SDE off and on; everything else is a baseline.
OURS = ("ffm", "pathb")


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
        print(f"  x {len(TUNED_DIMS)} searched dimensions {tuple(TUNED_DIMS)} "
              f"= {len(fan) * len(TUNED_DIMS)} jobs")
    print(f"""
Submit each wave as an array over the cell index.  `<submit>` is your own launcher --
nothing about it belongs in this repository's public half:

    for d in {' '.join(str(d) for d in TUNED_DIMS)}; do
      for i in $(seq 0 {len(WAVE1) - 1}); do
        <submit> python -m scripts.experiments.pancreas.tune run --wave 1 --dim $d --index $i
      done
    done
    python -m scripts.experiments.pancreas.tune collect        # wave 1's points

    for d in {' '.join(str(d) for d in TUNED_DIMS)}; do
      for i in $(seq 0 {len(WAVE2) - 1}); do
        <submit> python -m scripts.experiments.pancreas.tune run --wave 2 --dim $d --index $i
      done
    done
    python -m scripts.experiments.pancreas.tune collect        # both waves; paste TUNED

Every reported column is searched in its own space, the 2-D UMAP chart included; nothing
inherits a point across dimensions.  Serially and without a scheduler it is the same two
loops with `<submit>` deleted; budget ~{len(WAVE1) + len(WAVE2)} cells x
{len(TUNED_DIMS)} dimensions x a few GPU-minutes each.""")


def cmd_run(args) -> None:
    # A wiring check writes a real-looking record under a content-addressed name,
    # so it goes to its own tree rather than into the sweep `collect` ranks.
    args.root = smoke_root(args.root, ROOT, "pancreas_bench", args.smoke)

    # Preflight before anything expensive -- before the device is resolved and before
    # wave 2 reads wave 1's records -- so a bad invocation dies in second one.  ``run_cell``
    # repeats the call; it is a handful of `os.path` calls and it is what resolves the
    # cell, so the alternative would be threading the answer through two signatures to
    # save nothing.
    preflight_cell(WAVES, args.wave, args.index, args.dim, TUNED_DIMS, args.root,
                   lambda n, p: cell_path(args.root, args.dim, args.wave, n, p),
                   force=args.force, cache=CACHE_DIR)

    cfg = Run.from_env(**({"device": torch.device("cpu")} if args.cpu else {}),
                       **({"smoke": True} if args.smoke else {}))
    tuned = None
    if args.wave == 2:
        # a *read* of wave 1's answer, not a publication: wave 2 is unfinished by
        # construction here, so the completeness guard is waived and silenced.
        tuned = collect(args.root, dims=(args.dim,), quiet=True, partial=True)
        assert "ffm" in tuned.get(args.dim, {}), (
            f"wave 2 at d={args.dim} needs wave 1's ffm cells; run wave 1 and `collect`")
    run_cell(args.dim, args.wave, args.index, cfg, root=args.root, tuned=tuned,
             force=args.force)


def cmd_collect(args) -> None:
    tuned = collect(args.root, dims=args.dims, partial=args.partial)
    if args.partial:
        # An incomplete search may be *read* but never *published*: writing tuned.json
        # here would leave the notebook's paste-diff check passing against an answer the
        # search had not finished computing.
        print(format_tuned(tuned))
        return
    with open(TUNED_FILE, "w") as fh:
        json.dump({str(k): v for k, v in tuned.items()}, fh, indent=1)
    print(f"\nwrote {rel(TUNED_FILE)}\n\n# paste into notebooks/pancreas_ffm_paper.ipynb\n"
          f"# from scripts/experiments/pancreas/tune.py collect")
    print(format_tuned(tuned))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--dims", type=int, nargs="+", default=list(TUNED_DIMS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    # `--count` prints one integer and nothing else, so a launcher can size its array
    # from the grid rather than hardcoding a number that a widened rung would stale.
    p.add_argument("--count", type=int, default=None, metavar="WAVE",
                   choices=sorted(WAVES))
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--wave", type=int, default=1, choices=sorted(WAVES))
    r.add_argument("--dim", type=int, required=True)
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
