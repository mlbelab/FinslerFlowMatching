"""OT-CFM, MFM/LAND and four FFM kernels across the withheld middle third of R2's trajectory.

    python -m scripts.experiments.itracer.bench --prebuild        # the cache, once
    python -m scripts.experiments.itracer.bench                   # every cell, cached
    python -m scripts.experiments.itracer.bench --dim 2 --arm ffm --seed 0 --point tuned

The task
--------
:mod:`scripts.core.itracer_data` cuts the 2901 lineage-recorded cells at the tertiles of
their diffusion pseudotime.  The early third is ``p_0``, the late third is ``p_1``, and the
**middle third is withheld from everything**: from the cells an arm trains on, from the kNN
support of ``K_expr``, from the OT coupling and from the geometry.  It appears once, as the
reference the interpolation column is scored against.

Model time is linear from ``p_0`` to ``p_1``, so the withheld marginal is placed at its own
mean pseudotime rescaled onto that segment (:func:`~.datasets.t_mid`).  The three marginals
are equal tertiles of a rank, which puts it at ``0.5`` to four decimals — a coincidence of
the cut, not an assumption.

The six arms
------------
``cfm``
    OT-CFM: straight-line flow matching over an entropic **Euclidean**-OT coupling of the
    two endpoint marginals.  Neither the cloud's shape nor ``P`` enters anywhere.
``mfm_land``
    Metric Flow Matching under the authors' LAND metric — a metric read off the *point
    cloud's* local covariance, with no transition matrix and hence no lineage.
``ffm_expr`` / ``ffm_velo`` / ``ffm`` / ``ffm_lin_velo``
    Ours — the Randers-Finsler geometry read off ``P``, Path A (geodesic interpolant ->
    Finsler-cost OT -> distillation) — on the four transition matrices of
    :data:`~scripts.experiments.itracer.datasets.P_VARIANTS`: the expression kernel alone,
    plus the velocity factor, plus the lineage factor, and plus both.

Same trainer, same architecture, same budgets, same integration grid and the same seeds for
all six (:attr:`scripts.method.config.Run.arm_kw`).  They differ by the geometry and the
coupling, which is what the geometry *is*.

``P`` here is the three-factor kernel of :func:`~scripts.core.itracer_data.lineage_kernel`:
expression neighbours times barcode/scar relatedness times RNA-velocity alignment, each
factor an ``exp(temperature x score)`` that is identically 1 at temperature 0.  There is
still **no pseudotime and no direction indicator**: the pseudotime cuts the marginals and
places ``t_mid``, and is absent from ``P``, so whatever the Finsler geometry finds is in
the cloud, the lineage and the splicing, not in the axis the task was defined on.

What each factor is worth
-------------------------
The four FFM rows are the *same model* on four transition matrices that differ only in
which factors are switched on, so every difference between them is a factor's contribution
and not a difference of method.  :func:`factor_gains` reports the lattice.

``ffm`` minus ``ffm_expr`` is the honest denominator for the whole benchmark: a dataset is
only worth barcoding if the recording buys something over the expression neighbourhood that
comes free with it, and without that column every FFM number here is also compatible with
the Finsler geometry doing all the work by itself.

``ffm_lin_velo`` minus ``ffm_velo`` is the harder version of the same question, and the one
worth quoting.  RNA velocity is a directional signal recovered from reads that were
sequenced anyway; lineage recording is an experiment someone has to design, perform and pay
for.  If the lineage's gain over the expression kernel evaporates once velocity is already
in the kernel, the two are carrying the same information and only one of them was
necessary.  Both deltas are reported, because they can disagree and the disagreement is the
result.

Each ablation is separately tuned (:mod:`~scripts.experiments.itracer.tune` gives each the
same ``rho_mult`` x ``lam_mult`` product, rung for rung).  Reading a searched ``ffm``
against an ablation frozen at ours would credit the lineage with the search — each kernel
is a different geometry and none has a reason to want the same floor.

Two points, side by side
------------------------
:data:`ERYTHROID_TUNED` is ``scripts/experiments/erythroid/tuned.json`` verbatim — the point
the *mouse* benchmark chose, transplanted.  :mod:`~scripts.experiments.itracer.tune` then
searches this cloud on its own, and **both** are reported: ``--point erythroid`` and
``--point tuned``.  A transplant that loses could be losing because the geometry does not
help on a human organoid or because a mouse number is wrong for it, and only running both
tells the two apart; replacing the transplanted column with the searched one would throw
away the comparison that motivated the search.

``rho_mult`` and ``lam_mult`` are *multiples*, resolved by
:func:`scripts.core.scales.resolve` against a geometry built on this cloud, which is what
makes a mouse number quotable on a human organoid at all.  The erythroid tree tuned at
``d = 2 / 20 / 50`` and this benchmark reports the same three, so :data:`POINT_OF` is the
identity: the transplant is across the dataset only, and no column is also borrowing across
a dimension.

The transplant also fixes what ``d`` *means*: ``d = 2`` is the z-scored UMAP chart and
``d = 20`` / ``d = 50`` are raw PCA prefixes, exactly as on the erythroid, pancreas and
*C. elegans* clouds (:func:`~.datasets.space`).  A bandwidth or a blur tuned at erythroid's
``d = 2`` is only quotable here if the two ``d = 2`` spaces are the same kind of object.

What is reported, and what selection was allowed to see
-------------------------------------------------------
Every number in :func:`table` is measured on the **scored** slice of the withheld third —
the 90 % of it that :func:`~.datasets.split_holdout` keeps away from the tuner.  The other
10 % is what ``tune`` ranks on.  Under ``--point erythroid`` nothing was selected here at
all, but the two columns are scored on the same cells regardless, or the transplant and the
search would be compared on different references.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from scripts.core.arms import HParams, run_arm
from scripts.core.metrics import distribution_metrics, traj_slice
from scripts.core.paths import experiment_root
from scripts.core.protocol import TrainingSet
from scripts.core.scales import MULTIPLE_OF
from scripts.core.scales import resolve as resolve_scales
from scripts.method import Run

from .datasets import (DIMS, HOLDOUT, N_STEPS, P_VARIANTS, SOURCE, TARGET, UMAP_DIM,
                       build_training_set, load_cloud, snap, space, split_holdout,
                       split_shown, t_mid)

#: the arms, in table order — the two baselines, then the 2 x 2 of the kernel's factors
#: ordered so that switching one factor on is a step to an adjacent row: nothing,
#: velocity, lineage, both.
ARMS = ("cfm", "mfm_land", "ffm_expr", "ffm_velo", "ffm", "ffm_lin_velo")

#: display names.
ARM_LABEL = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND",
             "ffm_expr": "FFM, expression-only P",
             "ffm_velo": "FFM, + velocity",
             "ffm": "FFM (ours), + lineage",
             "ffm_lin_velo": "FFM, + lineage + velocity"}

#: which :mod:`scripts.core.arms` runner a column is.  The three ablations are **not** new
#: models and deliberately do not appear in that module's runner table: each is ``run_ffm``,
#: the same weights, the same schedule and the same geometry code, handed a different ``P``.
#: Registering one as a runner would make an ablation look like a method.
MODEL_OF = {a: "ffm" if a.startswith("ffm") else a for a in ARMS}

#: which transition matrix a column is built on.  See
#: :data:`scripts.experiments.itracer.datasets.P_VARIANTS`.  ``cfm`` never reads ``P`` and
#: ``mfm_land`` reads only the point cloud, so their entries are inert and are written down
#: anyway — a column with no entry here would be a column whose ``P`` nobody declared.
P_VARIANT_OF = {"cfm": "lineage_kernel", "mfm_land": "lineage_kernel",
                "ffm_expr": "expr_only", "ffm_velo": "velocity_only",
                "ffm": "lineage_kernel", "ffm_lin_velo": "lineage_velocity"}
assert set(P_VARIANT_OF) == set(MODEL_OF) == set(ARMS)
assert set(P_VARIANT_OF.values()) <= set(P_VARIANTS)

#: the ablation lattice, as the pairs :func:`factor_gains` reports.  Written down rather
#: than inferred from the arm names, because the interesting comparison is not only
#: "against nothing": the last row is what the *recording* is worth once a directional
#: signal that costs no extra experiment is already in the kernel, and that is the number a
#: reader deciding whether to barcode a dataset actually needs.
FACTOR_PAIRS = (("+ lineage",              "ffm_expr", "ffm"),
                ("+ velocity",             "ffm_expr", "ffm_velo"),
                ("+ both",                 "ffm_expr", "ffm_lin_velo"),
                ("+ lineage, over velocity", "ffm_velo", "ffm_lin_velo"),
                ("+ velocity, over lineage", "ffm", "ffm_lin_velo"))
assert all(b in ARMS and f in ARMS for _, b, f in FACTOR_PAIRS)

#: model seeds.  The cloud, the reduction, ``P``, the marginals and both splits are all
#: deterministic, so this is the only source of spread in a column.
SEEDS = (0, 1, 2)

#: the working precision, and it is not the repository default.  Measured rather than
#: assumed: this cloud's *float32* admissibility floor
#: (:func:`scripts.core.scales.lam_mult_floor`) is 3.4e-3 at ``d = 2``, 2.5e-3 at
#: ``d = 20`` and 1.4e-3 at ``d = 50``, across the whole ``rho_mult`` ladder.  The bottom
#: ``lam_mult`` rung is 0.01, so float32 would be admissible — by less than half a decade
#: at ``d = 2``, and with the rung *below* it inadmissible in two of the three spaces.
#: That is not enough room for a ladder the widening rule is allowed to extend downwards:
#: a λ edge at 0.01 would have to go unwidened for a reason that is about the float format
#: rather than about the geometry.  In float64 the floor is 6e-8 to 2e-7, five decades
#: clear.  Set for **every** arm and not only for ours, because a precision that differed
#: between columns would be a difference between columns.
DTYPE = torch.float64

#: ``scripts/experiments/erythroid/tuned.json``, verbatim.  Only the three arms this
#: benchmark runs are kept.
ERYTHROID_TUNED: dict[int, dict[str, dict]] = {
    2:  {"cfm": {"blur_frac": 3.0e+01}, "mfm_land": {"land_gamma": 4.0e-02},
         "ffm": {"rho_mult": 3.0e-04, "lam_mult": 1.0e+01}},
    20: {"cfm": {"blur_frac": 1.0e+01}, "mfm_land": {"land_rho": 3.0e-01},
         "ffm": {"rho_mult": 3.0e-04, "lam_mult": 3.0e+01}},
    50: {"cfm": {"blur_frac": 1.0e-02}, "mfm_land": {"land_rho": 1.0e-02},
         "ffm": {"rho_mult": 1.0e-03, "lam_mult": 1.0e+01}},
}

#: an ablation's transplanted point is our arm's, because there is nothing else it could
#: be: the erythroid tree never ran an expression-only, a velocity or a lineage+velocity
#: kernel, so the *only* honest no-search value for those columns is the one their
#: lineage-informed twin was given.  Under ``--point erythroid`` the four FFM rows
#: therefore differ by ``P`` alone, which is the cleanest reading of the lattice the table
#: has; under ``--point tuned`` each has been searched for its own geometry, which is the
#: fairer one.  Both are reported.
for _d in ERYTHROID_TUNED:
    for _a in ARMS:
        if _a.startswith("ffm") and _a != "ffm":
            ERYTHROID_TUNED[_d][_a] = dict(ERYTHROID_TUNED[_d]["ffm"])

#: which erythroid column each benchmark space borrows from.  The identity — see the
#: module docstring — and named rather than assumed, so a future column that has no
#: erythroid counterpart has to say what it is borrowing.
POINT_OF = {d: d for d in DIMS}
assert set(POINT_OF) == set(DIMS)
assert all(set(ERYTHROID_TUNED[POINT_OF[d]]) == set(ARMS) for d in DIMS)

#: the two point sets, in table order — the transplant first, since it is the one that does
#: not depend on anything measured here.
POINTS = ("erythroid", "tuned")

#: display names.
POINT_LABEL = {"erythroid": "erythroid point (transplanted)",
               "tuned": "tuned on this cloud"}

#: the metric switches that define our arm — not hyper-parameters and never searched.
#: :data:`scripts.experiments.erythroid.train.OUR_METRIC` minus the two multiples the point
#: supplies, so ``rho_mult`` / ``lam_mult`` mean here what they mean there.
OUR_METRIC: dict = {"metric_form": "fw_lambda", "c": 1.0,
                    "drift_mode": "full", "sigma_mode": "full"}

#: every ablation carries the *same* switches: each is our arm with a factor of ``P``
#: switched on or off, so a metric form that differed between the rows would confound the
#: thing being measured with the thing being held fixed.
ARM_FIXED: dict[str, dict] = {a: OUR_METRIC for a in ARMS if a.startswith("ffm")}

#: endpoints per side entering the Sinkhorn solve.  Set from the data and not left at the
#: engine default of 200: each marginal is 967 cells, and a cap below that would couple a
#: subsample of the problem rather than the problem.
OT_MAX_PTS = 4096

#: draws behind the cardinality-matched sampling floor.
N_FLOOR_DRAWS = 20

#: complete paths kept per cell for the figures — a fixed thinned set, so two arms in one
#: panel draw the *same* cells.
N_TRAJ_SAVED = 96


def bench_root(smoke: bool = False) -> str:
    return experiment_root("itracer_bench", smoke)


def out_path(dim: int, arm: str, seed: int, point: str = "erythroid",
             smoke: bool = False) -> str:
    return os.path.join(bench_root(smoke), point, f"d{dim}", f"{arm}_s{seed}.json")


def traj_path(dim: int, arm: str, seed: int, point: str = "erythroid",
              smoke: bool = False) -> str:
    return os.path.join(bench_root(smoke), "trajectories",
                        f"{point}_d{dim}_{arm}_s{seed}.npz")


def point_for(dim: int, arm: str, which: str) -> dict:
    """The hyper-parameters one column runs at, from whichever of the two sets is asked for.

    The tuned set is read lazily from :data:`scripts.experiments.itracer.tune.TUNED_FILE` —
    inside the function rather than at module scope, because the tuner imports *this*
    module for the arms and the resolution and a top-level import here would close the
    loop.  A missing or incomplete ``tuned.json`` is a message naming the command that
    writes it, not a ``KeyError`` two minutes into a batch job.
    """
    assert which in POINTS, f"unknown point set {which!r}; expected one of {POINTS}"
    if which == "erythroid":
        return ERYTHROID_TUNED[POINT_OF[dim]][arm]
    from .tune import load_tuned                       # deferred: see the docstring
    tuned = load_tuned()
    assert dim in tuned and arm in tuned[dim], (
        f"no tuned point for d={dim} {arm}; run the search and collect it "
        f"(`python -m scripts.experiments.itracer.tune collect`)")
    return tuned[dim][arm]


def hparams_for(dim: int, arm: str, ts: TrainingSet, cfg: Run,
                point: dict | None = None) -> tuple[HParams, dict]:
    """``(the point this cell runs at, the scales its multiples were resolved against)``.

    The arm's fixed switches go on first and the hyper-parameters second, so the point
    dictionary can never move the switch that defines the arm.  The sweep passes its own
    point through this function and not through a copy of it: a point selected under one
    resolution and consumed under another is not tuned for the model that runs.
    """
    hp = HParams(**{**cfg.arm_kw, "ot_max_pts": OT_MAX_PTS})
    edits = {**ARM_FIXED.get(arm, {}),
             **(ERYTHROID_TUNED[POINT_OF[dim]][arm] if point is None else point)}
    unknown = set(edits) - set(hp.as_dict()) - set(MULTIPLE_OF)
    assert not unknown, f"unknown hyper-parameter(s) {sorted(unknown)}"
    return resolve_scales(ts, hp, edits, cfg.device, cfg.dtype)


def sampling_floor(ref: np.ndarray, n_draw: int, seed: int = 0,
                   n_draws: int = N_FLOOR_DRAWS) -> dict:
    """What a sample of *true* cells scores against the rest of its own marginal.

    A model pushes ``n_draw`` particles, so the number its W2 should be read against is the
    one a same-sized sample of the truth achieves.  The draw is capped at half the
    reference, past which the remainder stops being a marginal; both sizes are recorded.
    """
    ref = np.asarray(ref, dtype=np.float64)
    n = int(min(n_draw, len(ref) // 2))
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_draws):
        p = rng.permutation(len(ref))
        rows.append(distribution_metrics(ref[p[:n]], ref[p[n:]], seed=k))
    return {"n_draw": n, "n_ref": int(len(ref)), "capped": bool(n < n_draw),
            **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}}


def run_cell(dim: int, arm: str, seed: int, point: str = "erythroid", cloud=None,
             cfg: Run | None = None) -> tuple[dict, np.ndarray, float]:
    """``(the record the JSON is written from, the full trajectory, the scored time)``."""
    assert arm in ARMS, f"unknown arm {arm!r}; expected one of {ARMS}"
    cfg = Run.from_env(n_steps=N_STEPS, dtype=DTYPE) if cfg is None else cfg
    cloud = load_cloud() if cloud is None else cloud
    ts = build_training_set(cloud, dim, p_variant=P_VARIANT_OF[arm])
    t_star = snap(t_mid(cloud), N_STEPS)
    pt = point_for(dim, arm, point)

    hp, scales = hparams_for(dim, arm, ts, cfg, point=pt)
    t0 = time.time()
    res = run_arm(MODEL_OF[arm], ts, hp, seed, cfg.device, cfg.dtype, n_steps=N_STEPS)
    train_s = time.time() - t0
    traj = np.asarray(res.traj, dtype=np.float64)
    assert traj.shape[0] == N_STEPS + 1 and traj.shape[1] == int(ts.p0.sum()), traj.shape

    # the withheld third splits into the slice the tuner ranks on and the slice everything
    # is reported on; only the second appears below, under either point.
    Xd = space(cloud, dim)
    i_sel, i_scored = split_holdout(cloud)
    mid_ref = Xd[i_scored]
    end_ref, src_ref = Xd[cloud.marginal(TARGET)], Xd[cloud.marginal(SOURCE)]
    pred_mid, pred_end = traj_slice(traj, t_star), traj_slice(traj, 1.0)

    rec = {
        "benchmark": "itracer_r2", "dim": int(dim), "arm": arm, "seed": int(seed),
        "point": point, "model": MODEL_OF[arm], "p_variant": ts.p_variant,
        "task": {"t_mid": t_star, "n_src": int(ts.p0.sum()), "n_tgt": int(ts.p1.sum()),
                 "n_withheld": int((cloud.third == HOLDOUT).sum()),
                 "n_selection": int(len(i_sel)), "n_scored": int(len(i_scored)),
                 "n_steps": N_STEPS},
        "tuned": pt, "tuned_from_dim": POINT_OF[dim], "fixed": ARM_FIXED.get(arm, {}),
        "hparams": hp.as_dict(), "scales": _jsonable(scales), "smoke": bool(cfg.smoke),
        "ts_diag": ts.diag, "arm_diag": _jsonable(res.diag),
        "interpolation": distribution_metrics(pred_mid, mid_ref, seed=seed),
        "endpoint": distribution_metrics(pred_end, end_ref, seed=seed),
        "floors": {"interpolation": sampling_floor(mid_ref, int(ts.p0.sum())),
                   "endpoint": sampling_floor(end_ref, int(ts.p0.sum()))},
        "controls": {"stay_at_source": distribution_metrics(src_ref, mid_ref, seed=seed),
                     "jump_to_target": distribution_metrics(end_ref, mid_ref, seed=seed)},
        "train_seconds": train_s, "dtype": str(cfg.dtype),
        "device": "CPU" if cfg.device.type == "cpu" else torch.cuda.get_device_name(0),
    }
    return rec, traj, t_star


def save_trajectory(traj: np.ndarray, t_star: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep = np.linspace(0, traj.shape[1] - 1, min(N_TRAJ_SAVED, traj.shape[1])).astype(int)
    np.savez_compressed(path, mid=traj_slice(traj, t_star).astype(np.float32),
                        start=traj[0].astype(np.float32), end=traj[-1].astype(np.float32),
                        paths=traj[:, keep].astype(np.float32), path_idx=keep,
                        t_mid=float(t_star))


def run_all(dims=DIMS, arms=ARMS, seeds=SEEDS, points=POINTS, cfg: Run | None = None,
            force: bool = False) -> list[dict]:
    """Every cell of the grid, skipping the ones already on disk.  Returns the records."""
    cfg = Run.from_env(n_steps=N_STEPS, dtype=DTYPE) if cfg is None else cfg
    cloud, recs = load_cloud(), []
    for point in points:
        for dim in dims:
            for arm in arms:
                for seed in seeds:
                    path = out_path(dim, arm, seed, point, cfg.smoke)
                    if os.path.exists(path) and not force:
                        recs.append(json.load(open(path)))
                        continue
                    rec, traj, t_star = run_cell(dim, arm, seed, point=point,
                                                 cloud=cloud, cfg=cfg)
                    save_trajectory(traj, t_star,
                                    traj_path(dim, arm, seed, point, cfg.smoke))
                    _write_json(rec, path)
                    recs.append(rec)
                    print(f"{point:<9} d={dim:<3} {arm:<9} seed {seed}   "
                          f"mid W2 {rec['interpolation']['W2']:.4f}   "
                          f"end W2 {rec['endpoint']['W2']:.4f}   "
                          f"({rec['train_seconds']:.0f}s)")
    return recs


def load_records(dims=DIMS, arms=ARMS, seeds=SEEDS, points=POINTS,
                 smoke: bool = False) -> list[dict]:
    """Whatever the array has finished, read off disk and nothing trained here."""
    recs = []
    for point in points:
        for dim in dims:
            for arm in arms:
                for seed in seeds:
                    path = out_path(dim, arm, seed, point, smoke)
                    if os.path.exists(path):
                        with open(path) as fh:
                            recs.append(json.load(fh))
    return recs


def table(recs: list[dict], dims=DIMS, arms=ARMS, point: str = "erythroid",
          delta_vs: str | None = None):
    """One point's reported block: W2 per arm and space, mean +- sd over seeds.

    ``withheld`` is the scored slice of the middle third no arm was allowed to see;
    ``endpoint`` is the target marginal every arm was trained to hit, and it is a fit check
    — an arm that misses ``t = 1`` has not earned a reading at ``t_mid``.  The three rows
    below the arms are identical across arms by construction: what a same-sized sample of
    the true withheld cells scores against the rest of itself, and what the two trivial
    answers score.

    ``delta_vs`` names an arm to read every other arm against, adding a third sub-column
    per space: the percentage change in withheld W2 from that arm to this one, negative
    better.  With ``delta_vs="ffm_expr"`` the FFM rows report what each factor of ``P``
    buys over the expression kernel that comes free with the data, and the baseline rows
    report where they stand against that same no-lineage geometry — which is the ablation
    :func:`factor_gains` reports separately, folded into the one reported table.  The
    reference rows are left blank: a sampling floor has no factor switched on.
    """
    import pandas as pd

    rows_p = [r for r in recs if r.get("point", "erythroid") == point]

    def arm_cell(dim, arm, key):
        v = [r[key]["W2"] for r in rows_p if r["dim"] == dim and r["arm"] == arm]
        if not v:
            return ""
        return f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "")

    def ref_cell(dim, *path):
        r = next((r for r in rows_p if r["dim"] == dim), None)
        for k in (path if r is not None else ()):
            r = r[k]
        return "" if r is None else f"{r:.4f}"

    def delta_cell(dim, arm):
        b, _ = _arm_stat(rows_p, point, dim, delta_vs, "interpolation")
        v, _ = _arm_stat(rows_p, point, dim, arm, "interpolation")
        return "" if b is None or v is None else f"{100 * (v - b) / b:+.1f}%"

    # the arm key, not its display label, keeps the column narrow enough to sit next to
    # two W2 columns at three spaces without wrapping the table.
    labs = ("withheld", "endpoint") + (() if delta_vs is None else (f"Δ vs {delta_vs}",))
    cols = [(f"d = {d}", lab) for d in dims for lab in labs]
    rows = {ARM_LABEL[a]: {(f"d = {d}", lab): arm_cell(d, a, key) for d in dims
                           for lab, key in (("withheld", "interpolation"),
                                            ("endpoint", "endpoint"))}
            for a in arms}
    if delta_vs is not None:
        for a in arms:
            rows[ARM_LABEL[a]] |= {(f"d = {d}", labs[2]): delta_cell(d, a) for d in dims}
    rows["sampling floor"] = {
        (f"d = {d}", lab): ref_cell(d, "floors", key, "W2") for d in dims
        for lab, key in (("withheld", "interpolation"), ("endpoint", "endpoint"))}
    for name, key in (("stay at p0", "stay_at_source"), ("jump to p1", "jump_to_target")):
        rows[name] = {(f"d = {d}", "withheld"): ref_cell(d, "controls", key, "W2")
                      for d in dims} | {(f"d = {d}", "endpoint"): "" for d in dims}
    return pd.DataFrame(rows).T.reindex(columns=pd.MultiIndex.from_tuples(cols)).fillna("")


def both_tables(recs: list[dict], dims=DIMS, arms=ARMS, points=POINTS):
    """The two point sets stacked, so the transplant and the search are read together."""
    import pandas as pd

    have = [p for p in points if any(r.get("point", "erythroid") == p for r in recs)]
    return pd.concat({POINT_LABEL[p]: table(recs, dims, arms, p) for p in have},
                     names=["point", "row"]) if have else pd.DataFrame()


def _arm_stat(recs: list[dict], point: str, dim: int, arm: str, key: str):
    v = [r[key]["W2"] for r in recs
         if r.get("point", "erythroid") == point and r["dim"] == dim and r["arm"] == arm]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan) if v else (None, None)


def lineage_gain(recs: list[dict], dims=DIMS, points=POINTS,
                 base: str = "ffm_expr", full: str = "ffm"):
    """What one factor of ``P`` is worth: ``full`` against ``base``, per space.

    Both rows are the same model on two ``P``s that differ in that factor alone, so the
    delta is the factor's contribution and not a difference of method.  It is reported with
    the seed spread beside it because a 3 % improvement on a 4 % standard deviation is not
    a finding — the sign of the delta is only worth reading where ``|delta|`` clears it.
    """
    import pandas as pd

    rows = {}
    for point in points:
        for dim in dims:
            cell = {}
            for lab, key in (("withheld", "interpolation"), ("endpoint", "endpoint")):
                (b, bs) = _arm_stat(recs, point, dim, base, key)
                (f, fs) = _arm_stat(recs, point, dim, full, key)
                if b is None or f is None:
                    cell |= {(lab, base): "", (lab, full): "", (lab, "Δ"): ""}
                    continue
                cell |= {(lab, base): f"{b:.4f} ± {bs:.4f}",
                         (lab, full): f"{f:.4f} ± {fs:.4f}",
                         (lab, "Δ"): f"{100 * (f - b) / b:+.1f}%"}
            rows[(POINT_LABEL[point], f"d = {dim}")] = cell
    return pd.DataFrame(rows).T


def factor_gains(recs: list[dict], dims=DIMS, points=POINTS, pairs=FACTOR_PAIRS,
                 key: str = "interpolation"):
    """The whole ablation lattice as one table: every pair of :data:`FACTOR_PAIRS`, per space.

    One number per cell — the percentage change in withheld-marginal W2 from switching a
    factor on — with the pooled seed spread beside it as ``+-``, so a delta can be read
    against the noise it has to clear.  Negative is better, since these are distances.

    The last two rows are the ones the table exists for.  ``+ lineage`` against the
    expression kernel is the headline, but a recorded lineage is only worth its experiment
    if it survives ``+ lineage, over velocity`` — the same factor added on top of a
    directional signal that falls out of reads already sequenced.  If those two disagree,
    the lineage and the velocity are carrying the same information and only one of them
    had to be paid for.
    """
    import pandas as pd

    rows = {}
    for lab, base, full in pairs:
        cell = {}
        for point in points:
            for dim in dims:
                (b, bs) = _arm_stat(recs, point, dim, base, key)
                (f, fs) = _arm_stat(recs, point, dim, full, key)
                col = (POINT_LABEL[point], f"d = {dim}")
                if b is None or f is None:
                    cell[col] = ""
                    continue
                spread = 100 * np.sqrt(np.nansum([bs ** 2, fs ** 2])) / b
                cell[col] = f"{100 * (f - b) / b:+.1f}% ± {spread:.1f}"
        rows[lab] = cell
    return pd.DataFrame(rows).T


def _write_json(rec: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(_jsonable(rec), fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.dtype):
        return str(o)
    return o


def prebuild(rebuild: bool = False) -> None:
    """Build the cloud cache and print the task, before any array is submitted.

    A few hundred jobs starting at once would otherwise race to write the same ``.npz``,
    and the assertions here are the cheapest place to find out that the cut moved.
    """
    cloud = load_cloud(rebuild=rebuild)
    n_sel, n_scored = (len(x) for x in split_holdout(cloud))
    n_tr, n_val = (len(x) for x in split_shown(cloud))
    print(f"  {len(cloud.idx)} candidate cells, "
          f"{len(np.unique(cloud.family))} scar families, "
          f"{len(np.unique(cloud.barcode))} barcode families; "
          f"cut at s = {cloud.edges[0]:.3f} / {cloud.edges[1]:.3f}")
    print(f"  withheld {int((cloud.third == HOLDOUT).sum())} cells "
          f"({n_sel} for selection, {n_scored} scored); "
          f"shown {n_tr + n_val} ({n_tr} train, {n_val} val)   "
          f"t_mid {snap(t_mid(cloud), N_STEPS):.4f}")
    # the velocity alignment for both cell sets an array task can ask for -- all the shown
    # cells (the benchmark) and the 85 % train split (the sweep).  Built here for the same
    # reason the cloud cache is: a few hundred jobs would otherwise race to write it, and
    # each build reruns scVelo.
    from scripts.core.itracer_data import velocity_alignment
    if any(nu for _, nu in P_VARIANTS.values()):
        for name, cells in (("shown", cloud.shown), ("sweep train", split_shown(cloud)[0])):
            C = velocity_alignment(cells, rebuild=rebuild)
            print(f"  velocity alignment, {name:<11} {C.shape}   "
                  f"cos in [{C.min():+.3f}, {C.max():+.3f}], "
                  f"{100 * (C > 0).mean():.1f}% of pairs forward")

    # all four transition matrices, so each ablation's premise is checked before the array
    # is submitted rather than argued for afterwards: the variants must agree on every
    # structural number and differ only in the factor each switches, which is exactly 1.0
    # wherever it is off.
    for d in DIMS:
        kind = "z-scored UMAP" if d == UMAP_DIM else f"{d} PCs"
        for variant in P_VARIANTS:
            ts = build_training_set(cloud, d, p_variant=variant)
            g = ts.diag
            print(f"  d = {d:<3d} ({kind:<13}) {variant:<16} train {g['n_train']} cells "
                  f"({g['n_p0']} at p0, {g['n_p1']} at p1)   P {ts.P.shape} "
                  f"{g['P_nnz_per_row']:.1f} nnz/row, {g['direct_p0_p1_edges']}/"
                  f"{g['direct_p0_p1_possible']} direct p0->p1 edges, "
                  f"K_lineage {g['K_lineage_mean_on_edges']:.4f}, "
                  f"K_velocity {g['K_velocity_mean_on_edges']:.4f} on edges")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prebuild", action="store_true",
                   help="build and cache the cloud, print the task, then exit")
    p.add_argument("--rebuild", action="store_true",
                   help="with --prebuild: rebuild the cache from the release")
    p.add_argument("--dim", type=int, choices=DIMS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int)
    p.add_argument("--point", choices=POINTS, nargs="+", default=list(POINTS),
                   help="which hyper-parameter set to run: the transplanted erythroid "
                        "point, this cloud's own searched one, or both")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.prebuild:
        prebuild(rebuild=a.rebuild)
        return
    recs = run_all(dims=(a.dim,) if a.dim else DIMS,
                   arms=(a.arm,) if a.arm else ARMS,
                   seeds=(a.seed,) if a.seed is not None else SEEDS,
                   points=tuple(a.point), force=a.force)
    print(both_tables(recs, dims=sorted({r["dim"] for r in recs})).to_string())


if __name__ == "__main__":
    main()
