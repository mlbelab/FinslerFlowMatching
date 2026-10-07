"""Robustness checks of the Pancreas results, as cached run records.

    python -m scripts.experiments.pancreas.review plan              # one job's cells per line
    python -m scripts.experiments.pancreas.review run --cells d2:ffm:s0:none,d2:ffm:s0:p1
    python -m scripts.experiments.pancreas.review report [--save]

One record per cell ``(dim, arm, seed, pert)``, written as ``d2_ffm_s0_none.json``.  A
record carries the point and the budget it was run at and is retrained when either moves,
so the notebook's :func:`ensure` and a batch job filling the same grid never disagree
about what is on disk.  Every number is scored the way the notebook's table is: ``W2`` at
``t = 1/2`` against the 90 % test slice of the withheld marginal, with the subsample drawn
at seed 0 for every run so it is common-mode across arms.

The studies sharing the grid:

**Seed pool.**  At d = 2 the shipped FFM is run at 32 seeds, the three
baselines, FFM at the library default point and the d = 2 fixes below at 20, so the
reported three-seed mean can be placed in the distribution it was drawn from and a fix is
not judged on three draws of a quantity whose seed-to-seed sd is ~15 %.

**Perturbation.**  ``p1..p3`` multiply every coordinate of the cloud by
``1 + 1e-7 * N(0, 1)`` before ``P``, the moments and the metric are built -- an
input change at the level of float32 rounding.  The within-seed spread over ``none`` and
the three draws, pooled over seeds, is the input sensitivity measured.  All draws of
one seed run in one process, so the comparison is same-process; ``rerun`` repeats the
unperturbed cell in a fresh job, which is the floor any other difference is read against.

**Fixes.**  Each holds the shipped ``(rho, lambda)`` and changes one thing:
``ffm_riem`` drops the 1-form and keeps ``a^2`` (the clean ablation), ``ffm_graph`` /
``ffm_neff`` / ``ffm_barrier`` are the extensions in :mod:`scripts.method.extend`,
``ffm_noclip`` removes gradient clipping, untested in the paper.  ``ffm_default``
is FFM at ``rho_mult = 1, lam_mult = 0.3``.  ``ffm_logeuc`` / ``ffm_svr`` / ``ffm_mlp`` /
``ffm_mlp_noise`` / ``ffm_lbarrier`` are the fitted extensions of
:mod:`scripts.method.extend_fit`, at d = 2 and 10 (:mod:`.extensions` reads them).

**Paths.**  Every record carries the mean max perpendicular deviation from the
chord over chord length, and the Finsler length of the trajectory over that of its chord
under the shipped metric of that dimension (one judge for every arm).  ``ffm_straight``
keeps FFM's Finsler-cost coupling and distils straight paths, which separates what the
coupling buys from what the bending buys.

**Band access, fixed protocol.**  :mod:`.band` scores every visible fraction
against whatever part of the withheld marginal stayed hidden, so the reference changes
with the fraction, includes the tuner's selection slice, and the visible sets are not
nested.  Here the 90 % test slice is cut once into a scored half and a pool half
(stratified by latent time); ``band5`` ... ``band45`` make a nested, latent-time-balanced
prefix of the pool -- 5 / 10 / 20 / 35 / 45 % of the whole marginal, the last being the
whole pool -- visible to ``P`` and the geometry, never to the coupling, and every fraction
is scored against the same half.
OT-CFM never reads ``P``, so its row must not move with the fraction.

**No tuning at all.**  Every method at one default point -- FFM at
``rho_mult = 1, lam_mult = 0.3``, OT-CFM, MFM/LAND and Curly-FM at the engine's
``HParams`` defaults -- on the band protocol above, five seeds at d = 2 and three above.  ``ffm_knn`` is the
same FFM on a uniform kNN ``P``, which carries no information beyond the neighbourhoods,
so the gap to ``ffm_default`` is what the velocity kernel itself is worth.

**The search, re-seeded.**  The 80 ``(rho_mult, lam_mult)`` cells of
:mod:`.tune`'s ladders at d = 2, four seeds each, twice: ``t{i}`` replays the tuning
protocol of :func:`.tune.run_cell` (85 % split, the selection objective, the model seed as
the tuning seed) and ``g{i}`` the reported one (all shown cells, W2 on the test slice).
Whether the objective carries any signal about the reported score is then a correlation
across cells, and selecting on one seed can be compared with the cell's four-seed mean.

``ffm_neff`` fixes the effective sample size per query point; ``ffm_wide`` is the other
reading of a fixed-``n_eff`` window -- one global bandwidth, set so the median Kish size on
the samples is the same 100 -- so ``n_eff = 100`` is read both ways.  That is
a wider window at d = 2 (x5.3) and a narrower one above it, where the shipped window
already holds 190 / 538 / 1153 effective points.

**Hardware.**  The 32 shipped-FFM d = 2 seeds again on an A100 and an L40S, in
``outputs/pancreas_review_hw/<gpu>``, read against the V100 pool by seed.

The support diagnostics need no training and are one record per
``(dim, variant)``: the Kish effective sample size of the kernel weights, the share of
Phase-1 chord points and of the scored cells where the shipped kernel has less mass than
the smoother's ``eps_den`` floor, and what unit travel costs there relative to the data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr

from scripts.core.arms import HParams, run_arm
from scripts.core.metrics import distribution_metrics, mean_floor, traj_slice, wasserstein
from scripts.core.transition import build_uniform_knn_transition
from scripts.core.paths import PUBLIC_ROOT, experiment_root, rel, table_file
from scripts.core.selection import OBJECTIVE_KEY, with_objective
from scripts.experiments.matched.variants import RiemannOnly
from scripts.method import (FWFactory, Run, cuda_warm_up, euclidean_w2, integrate,
                            moments_from, path_a)
from scripts.method.extend import BarrierFW, NeffFW, StraightPhi, graph_denoise, kish_neff
from scripts.method.extend_fit import (LearnedBarrierFW, LogEuclidFW, MlpFW, SvrFW,
                                       noise_for_neff)
from scripts.method.metric import TrueFW
from scripts.method.nets import Coupler, interpolant
from scripts.method.phases import build_ot, train_cfm, train_geodesic

from .band import training_set as band_training_set
from .datasets import (N_HOLDOUT_STRATA, TRAIN_BINS, build_training_set, load_cloud,
                       split_holdout, split_shown)
from .tune import LAM_MULTS, RHO_MULTS
from .tune import _train as tune_train

KEY = "pancreas_review"
DIMS = (2, 10, 20, 50)
BASE_N_STEPS = 102

FFM_ARMS = ("ffm", "ffm_default", "ffm_riem", "ffm_noclip", "ffm_graph", "ffm_neff",
            "ffm_wide", "ffm_barrier", "ffm_straight", "ffm_knn", "ffm_logeuc", "ffm_svr",
            "ffm_mlp", "ffm_mlp_noise", "ffm_lbarrier")
#: the fitted extensions of :mod:`scripts.method.extend_fit`, at d = 2 and 10
EXT_ARMS = ("ffm_logeuc", "ffm_svr", "ffm_mlp", "ffm_mlp_noise", "ffm_lbarrier")
ARM_DIMS = {a: (2, 10) for a in EXT_ARMS}
ENGINE_ARMS = ("cfm", "mfm_land", "curly")
#: the baselines at the engine's own ``HParams`` defaults, i.e. with nothing tuned
DEFAULT_ENGINE = {"mfm_default": "mfm_land", "curly_default": "curly"}
ARMS = FFM_ARMS + ENGINE_ARMS + tuple(DEFAULT_ENGINE)
ARM_LABELS = {
    "ffm": "FFM (shipped)",
    "ffm_default": "FFM, default point",
    "ffm_riem": "FFM, one-form off",
    "ffm_noclip": "FFM, no grad clip",
    "ffm_graph": "FFM, graph-denoised moments",
    "ffm_neff": "FFM, n_eff = 100 window",
    "ffm_wide": "FFM, global window, n_eff 100",
    "ffm_barrier": "FFM, off-support barrier",
    "ffm_straight": "FFM coupling, straight paths",
    "ffm_knn": "FFM, default point, uniform kNN P",
    "ffm_logeuc": "FFM, log-Euclidean mean",
    "ffm_svr": "FFM, SVR extension",
    "ffm_mlp": "FFM, MLP extension",
    "ffm_mlp_noise": "FFM, MLP + input noise",
    "ffm_lbarrier": "FFM, learned barrier",
    "mfm_default": "MFM / LAND, untuned",
    "curly_default": "Curly-FM, untuned (alpha 0.01)",
    "cfm": "OT-CFM",
    "mfm_land": "MFM / LAND",
    "curly": "Curly-FM",
}

DEFAULT_POINT = {"rho_mult": 1.0, "lam_mult": 0.3}
GRAPH = {"k": 16, "steps": 5, "alpha": 0.5}
NEFF = {"n_eff": 100.0}
BARRIER = {"phi_max": 30.0, "kappa": 0.1}
MLP = {"iters": 3000, "fit_seed": 0}
LBARRIER = {"phi_max": 30.0, "iters": 3000, "fit_seed": 0, "support_q": 0.05}

PERT_SCALE = 1e-7
PERTS = ("none", "p1", "p2", "p3")
RERUN = "rerun"
#: 45 % of the whole marginal is the entire pool half; the paper's Fig. 5c reads 0 / 5 / 35
BAND_FRACS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.45)
#: nothing tuned anywhere -- the methods at their defaults, and FFM on an
#: information-free kNN P to show what the velocity kernel itself is worth
R4_FRACS = BAND_FRACS
R4_ARMS = ("ffm_default", "cfm", "mfm_default", "curly_default", "ffm_knn")
R4_SEEDS = {2: (0, 1, 2, 3, 4), 10: (0, 1, 2), 20: (0, 1, 2), 50: (0, 1, 2)}
BAND = {f: f"band{round(100 * f)}" for f in sorted(set(BAND_FRACS + R4_FRACS))}
BAND_OF = {c: f for f, c in BAND.items()}
BAND_DIMS = (2, 20, 50)
BAND_ARMS = ("ffm", "cfm", "mfm_land", "curly")
BAND_SPLIT_SEED = 0
GRID = tuple({"rho_mult": r, "lam_mult": l} for r in RHO_MULTS for l in LAM_MULTS)
GRID_OF = {f"g{i}": i for i in range(len(GRID))}
TUNE_OF = {f"t{i}": i for i in range(len(GRID))}
SEARCH_SEEDS = (0, 1, 2, 3)
#: the follow-up wave: the best-looking cells of the search, re-run at seeds it never saw
FRESH_TOP = 8
FRESH_SEEDS = tuple(range(4, 16))
CONDITIONS = PERTS + (RERUN,) + tuple(BAND.values()) + tuple(GRID_OF) + tuple(TUNE_OF)

BASE_SEEDS = (0, 1, 2)
POOL_DIM = 2
POOL = {"ffm": 32, "ffm_default": 20, "ffm_riem": 20, "ffm_noclip": 20, "ffm_graph": 20,
        "ffm_wide": 20, "ffm_barrier": 20, "cfm": 20, "mfm_land": 20, "curly": 20,
        **{a: 20 for a in EXT_ARMS}}
_LOW_D = ("ffm", "ffm_default", "ffm_riem", "ffm_noclip", "ffm_graph", "ffm_neff",
          "ffm_barrier", "mfm_land")
CHAOS_ARMS = {2: _LOW_D + EXT_ARMS, 10: _LOW_D, 20: ("ffm",), 50: ("ffm",)}
#: the learned barrier read off the band protocol too, where it is fitted on the visible
#: cells as well, against the shipped metric on the same splits
LB_BAND = {"dims": (2, 10), "arms": ("ffm", "ffm_lbarrier")}
RERUN_ARMS = ("ffm",)

SUPPORT_ARMS = ("ffm", "ffm_default", "ffm_graph", "ffm_neff", "ffm_wide", "ffm_barrier")
SUPPORT_BUDGET = {"n_chord": 4096, "n_dirs": 32, "seed": 0, "neff": "shifted"}

#: the notebook's averaging window, t = 1/3 .. 2/3 on 11 points
TS_WINDOW = tuple(round(v, 3) for v in np.linspace(1.0 / 3.0, 2.0 / 3.0, 11))


# --------------------------------------------------------------------------- #
#  The grid
# --------------------------------------------------------------------------- #
def cells(dims=DIMS, smoke: bool = False) -> list[tuple[int, str, int, str]]:
    out = set()
    for d in dims:
        out.update((d, a, s, "none") for a in ARMS if d in ARM_DIMS.get(a, DIMS)
                   for s in BASE_SEEDS)
        out.update((d, a, s, p) for a in CHAOS_ARMS.get(d, ()) for s in BASE_SEEDS
                   for p in PERTS)
        if d == POOL_DIM:
            out.update((d, a, s, "none") for a, n in POOL.items() for s in range(n))
            out.update((d, a, s, RERUN) for a in RERUN_ARMS for s in BASE_SEEDS)
        if d in BAND_DIMS:
            out.update((d, a, s, b) for a in BAND_ARMS for s in BASE_SEEDS
                       for b in BAND.values())
        if d == POOL_DIM:
            out.update((d, "ffm", s, c) for s in SEARCH_SEEDS
                       for c in (*GRID_OF, *TUNE_OF))
        out.update((d, a, s, BAND[f]) for a in R4_ARMS for s in R4_SEEDS.get(d, ())
                   for f in R4_FRACS)
        if d in LB_BAND["dims"]:
            out.update((d, a, s, BAND[f]) for a in LB_BAND["arms"] for s in BASE_SEEDS
                       for f in BAND_FRACS)
    if smoke:
        keep = ("none", "p1", BAND[BAND_FRACS[0]], BAND[BAND_FRACS[-1]], "g0", "t0")
        out = {c for c in out if c[2] == 0 and c[3] in keep}
    arm_i = {a: i for i, a in enumerate(ARMS)}
    cond_i = {p: i for i, p in enumerate(CONDITIONS)}
    return sorted(out, key=lambda c: (c[0], arm_i[c[1]], c[2], cond_i[c[3]]))


def token(c) -> str:
    return f"d{c[0]}:{c[1]}:s{c[2]}:{c[3]}"


def parse(tok: str) -> tuple[int, str, int, str]:
    d, arm, s, p = tok.strip().split(":")
    assert d[0] == "d" and s[0] == "s", f"bad cell {tok!r}"
    assert arm in ARMS and p in CONDITIONS, f"bad cell {tok!r}"
    return int(d[1:]), arm, int(s[1:]), p


def root_dir(smoke: bool = False) -> str:
    return experiment_root(KEY, smoke)


def rec_path(root: str, c) -> str:
    return os.path.join(root, "cells", token(c).replace(":", "_") + ".json")


def mid_path(root: str, c) -> str:
    return os.path.join(root, "mid", token(c).replace(":", "_") + ".npz")


def support_path(root: str, dim: int, arm: str) -> str:
    return os.path.join(root, "support", f"d{dim}_{arm}.json")


def point_of(tuned_dim: dict, arm: str, pert: str = "none") -> dict:
    if arm in ENGINE_ARMS:
        pt = {arm: dict(tuned_dim[arm])}
    elif arm in DEFAULT_ENGINE:
        pt = {DEFAULT_ENGINE[arm]: {}, "untuned": True}
    else:
        pt = {"ffm": dict(DEFAULT_POINT if arm in ("ffm_default", "ffm_knn")
                          else tuned_dim["ffm"])}
    pt.update({"ffm_graph": {"graph": GRAPH}, "ffm_neff": {"neff": NEFF},
               "ffm_wide": {"wide": {"median_neff": NEFF["n_eff"]}},
               "ffm_barrier": {"barrier": BARRIER},
               "ffm_noclip": {"grad_clip": None},
               "ffm_knn": {"P": "uniform_knn"},
               "ffm_logeuc": {"extension": "log-euclidean"},
               "ffm_svr": {"extension": "svr", "C": 1.0, "epsilon": 0.1, "floor": "rho I"},
               "ffm_mlp": {"extension": "mlp", **MLP, "drift_cap": "sample max"},
               "ffm_mlp_noise": {"extension": "mlp", **MLP, "drift_cap": "sample max",
                                 "noise": "n_eff 100"},
               "ffm_lbarrier": {"barrier": "learned", **LBARRIER}}.get(arm, {}))
    if pert in PERTS[1:]:
        pt["pert"] = {"scale": PERT_SCALE, "draw": int(pert[1:])}
    if pert in BAND_OF:
        pt["band"] = {"frac": BAND_OF[pert], "protocol": "fixed scored half",
                      "split_seed": BAND_SPLIT_SEED}
    if pert in GRID_OF or pert in TUNE_OF:
        pt["ffm"] = dict(GRID[GRID_OF.get(pert, TUNE_OF.get(pert))])
        pt["search"] = "reported" if pert in GRID_OF else "tuning"
    return pt


def budget_of(cfg: Run) -> dict:
    return {"geo_iters": cfg.geo_iters, "cfm_iters": cfg.cfm_iters, "n_steps": cfg.n_steps,
            "base_n_steps": BASE_N_STEPS, "width": cfg.net_width, "depth": cfg.net_depth,
            "batch": cfg.batch, "lr": cfg.geo_lr, "grad_clip": cfg.grad_clip,
            "ot_k": cfg.ot_k, "ot_max_pts": cfg.ot_max_pts, "blur": cfg.ot_blur_frac,
            "window": list(TS_WINDOW)}


def _fresh(path: str, point: dict, budget: dict) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            prior = json.load(fh)
    except json.JSONDecodeError:
        return False
    return prior.get("point") == point and prior.get("budget") == budget


def missing(cfg: Run, tuned: dict, dims=DIMS, root: str | None = None) -> list[tuple]:
    """Cells not on disk at the current point and budget -- and, once the search grid is
    complete, the follow-up wave it selects."""
    root = root_dir(cfg.smoke) if root is None else root
    budget = budget_of(cfg)
    gone = [c for c in cells(dims, cfg.smoke)
            if not _fresh(rec_path(root, c), point_of(tuned[c[0]], c[1], c[3]), budget)]
    if POOL_DIM in dims and not any(c[3] in GRID_OF or c[3] in TUNE_OF for c in gone):
        gone += [c for c in fresh_cells(load_records(root, dims, cfg.smoke))
                 if not _fresh(rec_path(root, c), point_of(tuned[c[0]], c[1], c[3]), budget)]
    return gone


def _write(rec: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


# --------------------------------------------------------------------------- #
#  One process's shared state
# --------------------------------------------------------------------------- #
def perturbed(cloud, pert: str):
    """The cloud with every coordinate scaled by ``1 + PERT_SCALE * N(0, 1)``."""
    if pert not in PERTS[1:]:
        return cloud
    rng = np.random.default_rng(10_000 + int(pert[1:]))
    return replace(cloud, X=cloud.X * (1.0 + PERT_SCALE * rng.standard_normal(cloud.X.shape)))


def band_halves(cloud) -> tuple[np.ndarray, np.ndarray]:
    """``(scored, pool)``: the test slice cut once into halves within latent-time deciles.

    ``pool`` comes back in visibility order -- one cell per decile in turn -- so the visible
    set of every fraction is a prefix of the next one's and stays balanced along the band.
    """
    tgt = cloud.target_test
    rng = np.random.default_rng(BAND_SPLIT_SEED)
    order = np.argsort(cloud.latent_time[tgt], kind="stable")
    scored, pool = [], []
    for stratum in np.array_split(order, N_HOLDOUT_STRATA):
        p = rng.permutation(stratum)
        scored.append(p[: len(p) // 2])
        pool.append(p[len(p) // 2:])
    turn = [s[i] for i in range(max(map(len, pool))) for s in pool if i < len(s)]
    return tgt[np.sort(np.concatenate(scored))], tgt[np.asarray(turn)]


def band_visible(cloud, frac: float) -> np.ndarray:
    """The first ``frac`` of the whole withheld marginal, in the pool's visibility order."""
    pool = band_halves(cloud)[1]
    n = int(round(frac * len(cloud.target)))
    assert n <= len(pool), f"frac {frac} needs {n} visible cells, the pool has {len(pool)}"
    return np.sort(pool[:n])


def _condition(pert: str) -> str:
    """What a cell's training set depends on: its draw, its band, the tuner's split, or
    nothing.  A grid cell trains on the shipped set; only its metric moves."""
    if pert in PERTS[1:] or pert in BAND_OF:
        return pert
    return "split85" if pert in TUNE_OF else "none"


def wide_scale(X, cfg: Run, n_eff: float, iters: int = 40) -> float:
    """The ``eps_kernel_scale`` whose single, global bandwidth gives the samples a median
    Kish size of ``n_eff`` -- the shipped smoother with its window rescaled, wider where it
    held fewer points than that and narrower where it held more."""
    Xt = torch.as_tensor(X, **cfg.torch_kw)
    sq = torch.cdist(Xt, Xt) ** 2
    nn = sq.clone().fill_diagonal_(float("inf")).min(dim=1).values.sqrt()
    eps0 = float(nn[nn > 0].median())
    lo, hi = np.log(1e-3), np.log(1e3)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        k = torch.exp(-sq / (4.0 * (eps0 * np.exp(mid)) ** 2))
        lo, hi = (mid, hi) if float(kish_neff(k).median()) < n_eff else (lo, mid)
    return float(np.exp(0.5 * (lo + hi)))


def build_metric(arm: str, X, b0, D, cfg: Run, tuned_dim: dict):
    if arm in ("ffm_default", "ffm_knn"):
        return FWFactory(X, D, b0, cfg, **DEFAULT_POINT).base
    fw = FWFactory(X, D, b0, cfg, **tuned_dim["ffm"])
    kw = {"rho": fw.rho, "lam": fw.lam}
    if arm in ("ffm", "ffm_noclip", "ffm_straight"):
        return fw.base
    if arm == "ffm_wide":
        scale = wide_scale(X, cfg, NEFF["n_eff"])
        return TrueFW(X, D, b0, cfg.but(eps_kernel_scale=scale), **kw).calibrate()
    if arm == "ffm_riem":
        return RiemannOnly(X, D, b0, cfg, **kw).calibrate()
    if arm == "ffm_graph":
        return TrueFW(X, graph_denoise(X, D, **GRAPH), graph_denoise(X, b0, **GRAPH), cfg,
                      **kw).calibrate()
    if arm == "ffm_neff":
        return NeffFW(X, D, b0, cfg, **kw, **NEFF).calibrate()
    if arm == "ffm_barrier":
        return BarrierFW(X, D, b0, cfg, **kw, **BARRIER).calibrate()
    if arm == "ffm_logeuc":
        return LogEuclidFW(X, D, b0, cfg, **kw).calibrate()
    if arm == "ffm_svr":
        return SvrFW(X, D, b0, cfg, **kw).calibrate()
    if arm == "ffm_mlp":
        return MlpFW(X, D, b0, cfg, **kw, **MLP).calibrate()
    if arm == "ffm_mlp_noise":
        return MlpFW(X, D, b0, cfg, **kw, **MLP,
                     noise_scale=noise_for_neff(X, cfg, NEFF["n_eff"])).calibrate()
    if arm == "ffm_lbarrier":
        return LearnedBarrierFW(X, D, b0, cfg, **kw, **LBARRIER).calibrate()
    raise KeyError(arm)


class _Scorer:
    def __init__(self, cloud, n_draw: int, ref: np.ndarray | None = None):
        self.mid = cloud.X[cloud.target_test if ref is None else ref]
        self.end = cloud.X[cloud.marginal(TRAIN_BINS[1])]
        dm = lambda a, b, k: distribution_metrics(a, b, seed=k)
        self.floors = {"mid": mean_floor(self.mid, n_draw, dm),
                       "end": mean_floor(self.end, n_draw, dm)}

    def __call__(self, traj) -> dict:
        mid = distribution_metrics(traj_slice(traj, 0.5), self.mid)
        end = distribution_metrics(traj_slice(traj, 1.0), self.end)
        curve = {f"{t:g}": wasserstein(traj_slice(traj, t), self.mid) for t in TS_WINDOW}
        return {"W2_mid": mid["W2"], "W1_mid": mid["W1"], "MMD_mid": mid["MMD"],
                "W2_end": end["W2"], "W2_window": float(np.mean(list(curve.values()))),
                "curve": curve}


class _Ctx:
    """Clouds, training sets, metrics and scorers, each built once per process."""

    def __init__(self, cfg: Run, tuned: dict):
        self.cfg, self.tuned = cfg, tuned
        self._cloud, self._ts, self._metric, self._score = {}, {}, {}, {}
        self.warm = False

    def cloud(self, dim):
        if dim not in self._cloud:
            self._cloud[dim] = load_cloud(dim)
        return self._cloud[dim]

    def ts(self, dim, pert, knn: bool = False):
        key = (dim, _condition(pert), knn)
        if key not in self._ts:
            cloud = self.cloud(dim)
            if key[1] in BAND_OF:
                ts = band_training_set(cloud, band_visible(cloud, BAND_OF[key[1]]))
            elif key[1] == "split85":
                ts = build_training_set(cloud, subset=split_shown(cloud)[0])
            else:
                ts = build_training_set(perturbed(cloud, key[1]))
            if knn:
                ts = replace(ts, P=build_uniform_knn_transition(ts.X), p_variant="uniform_knn")
            X = np.asarray(ts.X, dtype=np.float64)
            b0, D = moments_from(ts.P, X)
            self._ts[key] = (ts, X, b0, D, self.cfg.tensor(X[ts.p0]), self.cfg.tensor(X[ts.p1]))
        return self._ts[key]

    def metric(self, dim, arm, pert):
        key = (dim, arm, pert if pert in GRID_OF else _condition(pert))
        if key not in self._metric:
            _, X, b0, D, _, _ = self.ts(dim, pert, knn=arm == "ffm_knn")
            tuned = self.tuned[dim]
            if pert in GRID_OF:
                tuned = {**tuned, "ffm": GRID[GRID_OF[pert]]}
            self._metric[key] = build_metric(arm, X, b0, D, self.cfg, tuned)
        return self._metric[key]

    def scorer(self, dim, pert="none"):
        key = (dim, pert in BAND_OF)
        if key not in self._score:
            cloud = self.cloud(dim)
            self._score[key] = _Scorer(cloud, int(self.ts(dim, "none")[0].p0.sum()),
                                       band_halves(cloud)[0] if key[1] else None)
        return self._score[key]


# --------------------------------------------------------------------------- #
#  Training and scoring one cell
# --------------------------------------------------------------------------- #
def geo_energy(phi, metric, X0_t, X1_t, cfg: Run, n: int = 4096) -> float:
    """The Phase-1 objective ``E F^2`` of a trained interpolant, on a fixed draw of pairs."""
    g = torch.Generator(device="cpu").manual_seed(7)
    i = torch.randint(len(X0_t), (n,), generator=g).to(X0_t.device)
    j = torch.randint(len(X1_t), (n,), generator=g).to(X1_t.device)
    t = torch.rand(n, 1, generator=g).to(cfg.device, cfg.dtype)
    with torch.enable_grad():
        x_t, x_dot = interpolant(phi, t, X0_t[i], X1_t[j], create_graph=False)
    with torch.no_grad():
        return float(metric.F2(x_t.detach(), x_dot.detach()).mean())


def train(c, ctx: _Ctx):
    dim, arm, seed, pert = c
    cfg = ctx.cfg
    ts, _, _, _, X0_t, X1_t = ctx.ts(dim, pert, knn=arm == "ffm_knn")
    if arm in ENGINE_ARMS or arm in DEFAULT_ENGINE:
        hp = HParams(**cfg.arm_kw, **({} if arm in DEFAULT_ENGINE else ctx.tuned[dim][arm]))
        res = run_arm(DEFAULT_ENGINE.get(arm, arm), ts, hp, seed, cfg.device, cfg.dtype,
                      n_steps=BASE_N_STEPS)
        return np.asarray(res.traj, dtype=np.float64), {"ot": res.diag}

    metric = ctx.metric(dim, arm, pert)
    run_cfg = cfg.but(grad_clip=float("inf")) if arm == "ffm_noclip" else cfg
    if arm == "ffm_straight":
        phi = train_geodesic(metric, Coupler(X0_t, X1_t, seed=seed), seed, run_cfg,
                             log_every=0)
        coupler, ot_diag = build_ot(phi, metric, X0_t, X1_t, seed, run_cfg)
        v_net = train_cfm(StraightPhi(), coupler, metric.d, seed, run_cfg, log_every=0)
        traj = integrate(v_net, X0_t, run_cfg).cpu().numpy().astype(np.float64)
    else:
        det = path_a(metric, X0_t, X1_t, seed, run_cfg)
        phi, traj, ot_diag = det["phi"], det["traj"], det["ot"]
    return traj, {"ot": ot_diag, "geo_energy": geo_energy(phi, metric, X0_t, X1_t, cfg)}


@torch.no_grad()
def path_geometry(traj, judge, cfg: Run, chunk: int = 8192) -> dict:
    """Bend (max perpendicular deviation / chord) and Finsler length / chord length."""
    tr = np.asarray(traj, dtype=np.float64)
    c = tr[-1] - tr[0]
    L = np.linalg.norm(c, axis=1)
    ok = L > 1e-12
    tr, c, L = tr[:, ok], c[ok], L[ok]
    u = c / L[:, None]
    rel_ = tr - tr[0][None]
    along = np.einsum("tbd,bd->tb", rel_, u)
    perp = np.linalg.norm(rel_ - along[..., None] * u[None], axis=2)

    T = tr.shape[0] - 1
    s = (np.arange(T) + 0.5) / T
    chord_x = tr[0][None] + s[:, None, None] * c[None]

    def length(P, V):
        P, V = P.reshape(-1, P.shape[-1]), V.reshape(-1, V.shape[-1])
        F = torch.cat([judge.F(cfg.tensor(P[a:a + chunk]), cfg.tensor(V[a:a + chunk]))
                       for a in range(0, len(P), chunk)])
        return F.reshape(T, -1).sum(0).double().cpu().numpy()

    ratio = length(0.5 * (tr[1:] + tr[:-1]), tr[1:] - tr[:-1]) \
        / length(chord_x, np.broadcast_to(c[None] / T, chord_x.shape))
    return {"bend": float(np.mean(perp.max(axis=0) / L)), "length_ratio": float(np.mean(ratio))}


def _device_name(cfg: Run) -> str:
    return torch.cuda.get_device_name(0) if cfg.device.type == "cuda" else "cpu"


def tuning_cell(c, ctx: _Ctx) -> dict:
    """One cell of the search under the tuning protocol, as :func:`.tune.run_cell` scores it.

    The 85 % split with ``P`` rebuilt on it, the cell trained through the tuner's own
    ``_train``, and the selection objective: ``W2`` to the held-out 15 % of ``p_1`` at
    ``t = 1`` and to the 10 % selection slice of the withheld marginal at the mid step, both
    subsampled at the tuning seed -- which here is the model seed.
    """
    dim, _, seed, cond = c
    cfg = ctx.cfg.but(tune_seed=seed)
    cloud = ctx.cloud(dim)
    _, i_val = split_shown(cloud)
    i_end = i_val[cloud.bin_id[i_val] == TRAIN_BINS[1]]
    i_sel, _ = split_holdout(cloud)
    traj = tune_train("ffm", GRID[TUNE_OF[cond]], ctx.ts(dim, cond)[0], cfg, dim=dim, tuned=None)
    return with_objective({
        "W2_endpoint": euclidean_w2(traj[-1], cloud.X[i_end], seed=seed),
        "W2_intermediate": euclidean_w2(traj[traj.shape[0] // 2], cloud.X[i_sel], seed=seed)})


def run_cells(todo, cfg: Run, tuned: dict, root: str, verbose: bool = True) -> None:
    ctx = _Ctx(cfg, tuned)
    budget = budget_of(cfg)
    for c in sorted(todo, key=lambda c: c[0]):
        dim, arm, seed, pert = c
        if not ctx.warm:
            cuda_warm_up(ctx.metric(dim, "ffm", "none"), *ctx.ts(dim, "none")[4:], cfg)
            ctx.warm = True
        t0 = time.time()
        common = {"dim": dim, "arm": arm, "seed": seed, "pert": pert,
                  "point": point_of(tuned[dim], arm, pert), "budget": budget,
                  "device": _device_name(cfg), "torch": torch.__version__,
                  "smoke": bool(cfg.smoke)}
        if pert in TUNE_OF:
            rec = {**common, **tuning_cell(c, ctx), "train_seconds": time.time() - t0}
            _write(rec, rec_path(root, c))
            if verbose:
                print(f"[review] {token(c)}  objective {rec[OBJECTIVE_KEY]:.4f}"
                      f"  ({rec['train_seconds']:.0f}s)", flush=True)
            continue
        traj, extra = train(c, ctx)
        seconds = time.time() - t0
        rec = {**common, **ctx.scorer(dim, pert)(traj), "floors": ctx.scorer(dim, pert).floors,
               **path_geometry(traj, ctx.metric(dim, "ffm", "none"), cfg), **extra,
               "train_seconds": seconds, "n_steps": int(traj.shape[0] - 1)}
        if dim == POOL_DIM and pert not in GRID_OF:
            os.makedirs(os.path.dirname(mid_path(root, c)), exist_ok=True)
            np.savez_compressed(mid_path(root, c), mid=traj_slice(traj, 0.5).astype(np.float32))
        _write(rec, rec_path(root, c))
        if verbose:
            print(f"[review] {token(c)}  mid W2 {rec['W2_mid']:.4f}  end W2 {rec['W2_end']:.4f}"
                  f"  ({seconds:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
#  Support diagnostics -- no training
# --------------------------------------------------------------------------- #
def support_diagnostics(dim: int, arm: str, ctx: _Ctx) -> dict:
    """Build both metrics with autograd on -- a fitted extension trains a net as it is
    constructed -- then measure without it."""
    ctx.metric(dim, arm, "none"), ctx.metric(dim, "ffm", "none")
    return _support_diagnostics(dim, arm, ctx)


@torch.no_grad()
def _support_diagnostics(dim: int, arm: str, ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    _, _, _, _, X0_t, X1_t = ctx.ts(dim, "none")
    metric, shipped = ctx.metric(dim, arm, "none"), ctx.metric(dim, "ffm", "none")
    cloud = ctx.cloud(dim)
    g = torch.Generator(device="cpu").manual_seed(SUPPORT_BUDGET["seed"])
    n = SUPPORT_BUDGET["n_chord"]
    i = torch.randint(len(X0_t), (n,), generator=g).to(X0_t.device)
    j = torch.randint(len(X1_t), (n,), generator=g).to(X1_t.device)
    t = torch.rand(n, 1, generator=g).to(cfg.device, cfg.dtype)
    dirs = torch.randn(SUPPORT_BUDGET["n_dirs"], metric.d, generator=g)
    dirs = (dirs / dirs.norm(dim=1, keepdim=True)).to(cfg.device, cfg.dtype)
    sets = {"data": metric.X, "chord": (1.0 - t) * X0_t[i] + t * X1_t[j],
            "test": cfg.tensor(cloud.X[cloud.target_test])}

    def weights(p):
        """The variant's kernel weights scaled so the nearest sample has weight 1: Kish
        n_eff is scale-free, and unscaled weights underflow to 0 in the void."""
        if isinstance(metric, NeffFW):
            return metric._kernel(p)
        sqd = torch.clamp((p ** 2).sum(1, keepdim=True) + (metric.X ** 2).sum(1)[None, :]
                          - 2.0 * (p @ metric.X.T), min=0.0)
        return torch.exp(-(sqd - sqd.min(dim=1, keepdim=True).values)
                         / (4.0 * metric.eps_kernel ** 2))

    def stats(P, chunk=1024):
        mass, neff, cost, adm = [], [], [], []
        for a in range(0, len(P), chunk):
            p = P[a:a + chunk]
            mass.append(shipped._kernel(p).sum(1))
            neff.append(kish_neff(weights(p)))
            G, beta = metric.tensors(p)
            vGv = torch.einsum("kd,bde,ke->bk", dirs, G, dirs)
            cost.append((metric.scale * (torch.sqrt(vGv.clamp(min=1e-9)) + beta @ dirs.T))
                        .mean(1))
            adm.append(torch.sqrt((beta * torch.linalg.solve(G, beta)).sum(1).clamp(min=0)))
        return [torch.cat(v).double().cpu().numpy() for v in (mass, neff, cost, adm)]

    out = {k: stats(P) for k, P in sets.items()}
    void = {k: out[k][0] < cfg.eps_den for k in ("chord", "test")}
    base = float(out["data"][2].mean())
    ratio = lambda k, m: float(out[k][2][m].mean() / base) if m.any() else float("nan")
    return {"neff_data": float(np.median(out["data"][1])),
            "neff_chord": float(np.median(out["chord"][1])),
            "neff_test": float(np.median(out["test"][1])),
            "void_chord": float(void["chord"].mean()), "void_test": float(void["test"].mean()),
            "cost_void_chord": ratio("chord", void["chord"]),
            "cost_test": ratio("test", np.ones(len(out["test"][2]), dtype=bool)),
            "adm_mean": float(out["data"][3].mean()), "adm_max": float(out["data"][3].max()),
            "rho": float(metric.rho), "lam": float(metric.lam), "eps_kernel": metric.eps_kernel}


def ensure_support(cfg: Run, tuned: dict, dims=DIMS, root: str | None = None,
                   force: bool = False, arms=SUPPORT_ARMS) -> None:
    root = root_dir(cfg.smoke) if root is None else root
    ctx = None
    for d in dims:
        for arm in arms:
            if d not in ARM_DIMS.get(arm, DIMS):
                continue
            path, pt = support_path(root, d, arm), point_of(tuned[d], arm)
            if not force and _fresh(path, pt, SUPPORT_BUDGET):
                continue
            ctx = ctx or _Ctx(cfg, tuned)
            _write({"dim": d, "arm": arm, "point": pt, "budget": SUPPORT_BUDGET,
                    **support_diagnostics(d, arm, ctx)}, path)


# --------------------------------------------------------------------------- #
#  The notebook's entry
# --------------------------------------------------------------------------- #
def ensure(cfg: Run, tuned: dict, dims=DIMS, root: str | None = None, force: bool = False,
           verbose: bool = True) -> dict:
    """Every cell of the grid, training only what disk cannot answer, then the records."""
    root = root_dir(cfg.smoke) if root is None else root
    dims = tuple(d for d in dims if d in DIMS)
    todo = cells(dims, cfg.smoke) if force else missing(cfg, tuned, dims, root)
    if todo:
        run_cells(todo, cfg, tuned, root, verbose)
    wave2 = missing(cfg, tuned, dims, root)
    if wave2:
        run_cells(wave2, cfg, tuned, root, verbose)
    ensure_support(cfg, tuned, dims, root, force)
    return load_records(root, dims, cfg.smoke)


def ensure_cells(cfg: Run, tuned: dict, todo, root: str | None = None,
                 verbose: bool = True) -> dict:
    """:func:`ensure` for an explicit list of cells -- a notebook that reads one study
    trains (or loads) that study and nothing else of the grid."""
    root = root_dir(cfg.smoke) if root is None else root
    budget = budget_of(cfg)
    todo = list(todo)
    gone = [c for c in todo
            if not _fresh(rec_path(root, c), point_of(tuned[c[0]], c[1], c[3]), budget)]
    if gone:
        run_cells(gone, cfg, tuned, root, verbose)
    runs = {}
    for c in todo:
        with open(rec_path(root, c)) as fh:
            runs[c] = json.load(fh)
    return {"runs": runs, "support": {}, "root": root,
            "dims": tuple(sorted({c[0] for c in todo})), "smoke": cfg.smoke}


def load_records(root: str | None = None, dims=DIMS, smoke: bool = False) -> dict:
    root = root_dir(smoke) if root is None else root
    runs, support = {}, {}
    for c in cells(dims, smoke):
        p = rec_path(root, c)
        if os.path.exists(p):
            with open(p) as fh:
                runs[c] = json.load(fh)
    for d in dims:
        for arm in SUPPORT_ARMS:
            p = support_path(root, d, arm)
            if os.path.exists(p):
                with open(p) as fh:
                    support[(d, arm)] = json.load(fh)
    recs = {"runs": runs, "support": support, "root": root, "dims": tuple(dims),
            "smoke": smoke}
    for c in fresh_cells(recs):
        p = rec_path(root, c)
        if os.path.exists(p):
            with open(p) as fh:
                runs[c] = json.load(fh)
    return recs


# --------------------------------------------------------------------------- #
#  Frames
# --------------------------------------------------------------------------- #
def _w2(runs, d, arm, seeds, pert="none", field="W2_mid") -> dict:
    return {s: runs[(d, arm, s, pert)][field] for s in seeds if (d, arm, s, pert) in runs}


def frame_seeds(recs: dict) -> pd.DataFrame:
    runs = recs["runs"]
    vals = {a: _w2(runs, POOL_DIM, a, range(n)) for a, n in POOL.items()}
    vals = {a: v for a, v in vals.items() if len(v) >= 3}
    if not vals:
        return pd.DataFrame()
    ffm = np.array(list(vals["ffm"].values())) if "ffm" in vals else None
    rows = {}
    for a, by_seed in vals.items():
        v = np.array(list(by_seed.values()))
        first = np.mean([by_seed[s] for s in BASE_SEEDS if s in by_seed])
        trip = np.array([np.mean(x) for x in itertools.combinations(v, 3)])
        rows[ARM_LABELS[a]] = {
            "n": len(v), "mean": v.mean(), "sd": v.std(ddof=1),
            "q2.5": np.percentile(v, 2.5), "median": np.median(v),
            "q97.5": np.percentile(v, 97.5), "seeds 0-2": first,
            "pct among 3-seed means": 100.0 * float((trip <= first).mean()),
            "P(FFM run wins)": (float((ffm[:, None] < v[None, :]).mean())
                                if ffm is not None and a != "ffm" else np.nan)}
    df = pd.DataFrame(rows).T
    df.index.name = f"d = {POOL_DIM} seed pool"
    return df


def frame_chaos(recs: dict) -> pd.DataFrame:
    runs = recs["runs"]
    rows = {}
    for d in recs["dims"]:
        for a in CHAOS_ARMS.get(d, ()):
            M = np.array([[runs[(d, a, s, p)]["W2_mid"] for p in PERTS] for s in BASE_SEEDS
                          if all((d, a, s, p) in runs for p in PERTS)])
            if not len(M):
                continue
            within = float(np.sqrt(M.var(axis=1, ddof=1).mean()))
            pool = np.array(list(_w2(runs, d, a, range(POOL.get(a, 0))).values())) \
                if d == POOL_DIM else M[:, 0]
            seeds = pool if len(pool) > len(M) else M[:, 0]
            seed_sd = float(seeds.std(ddof=1)) if len(seeds) > 1 else np.nan
            dre = [abs(runs[(d, a, s, RERUN)]["W2_mid"] - runs[(d, a, s, "none")]["W2_mid"])
                   for s in BASE_SEEDS if (d, a, s, RERUN) in runs and (d, a, s, "none") in runs]
            rows[(d, ARM_LABELS[a])] = {
                "seeds": len(M), "mid W2": M[:, 0].mean(), "seed sd": seed_sd,
                "seed sd n": len(seeds), "chaos %": 100.0 * within / M.mean(),
                "within / seed sd": within / seed_sd,
                "rerun |dW2|": float(np.mean(dre)) if dre else np.nan}
    df = pd.DataFrame(rows).T
    if len(df):
        df.index.names = ["d", "1e-7 perturbation"]
    return df


def frame_arms(recs: dict) -> pd.DataFrame:
    runs = recs["runs"]
    rows = {}
    for d in recs["dims"]:
        ref = _w2(runs, d, "ffm", BASE_SEEDS)
        ref = np.mean(list(ref.values())) if ref else np.nan
        floors = None
        for a in ARMS:
            got = [runs[(d, a, s, "none")] for s in BASE_SEEDS if (d, a, s, "none") in runs]
            if not got:
                continue
            floors = got[0]["floors"]
            col = lambda f: np.array([r.get(f, np.nan) for r in got], dtype=float)
            mid = col("W2_mid")
            rows[(d, ARM_LABELS[a])] = {
                "n": len(got), "mid W2": mid.mean(), "pop sd": mid.std(ddof=0),
                "vs FFM %": 100.0 * (mid.mean() - ref) / ref, "end W2": col("W2_end").mean(),
                "window W2": col("W2_window").mean(), "bend": col("bend").mean(),
                "F-length / chord": col("length_ratio").mean()}
        if floors is not None:
            rows[(d, "sampling floor")] = {"mid W2": floors["mid"]["W2"],
                                           "end W2": floors["end"]["W2"]}
    df = pd.DataFrame(rows).T
    if len(df):
        df.index.names = ["d", "seeds 0-2"]
    return df


SUPPORT_COLS = {"neff_data": "n_eff data", "neff_chord": "n_eff chord",
                "neff_test": "n_eff scored", "void_chord": "void chord",
                "void_test": "void scored", "cost_void_chord": "cost void / data",
                "cost_test": "cost scored / data", "adm_max": "max |beta|"}


def frame_support(recs: dict) -> pd.DataFrame:
    rows = {(d, ARM_LABELS[a]): {v: r[k] for k, v in SUPPORT_COLS.items()}
            for (d, a), r in sorted(recs["support"].items())}
    df = pd.DataFrame(rows).T
    if len(df):
        df.index.names = ["d", "support"]
    return df


def frame_restarts(recs: dict, ks=(1, 2, 4, 8, 16), n_sub: int = 100) -> pd.DataFrame:
    """W2 of the pooled pushforward of K FFM restarts at d = 2, over random K-subsets.

    Each source cell is pushed by one of the K restarts, assigned at random, and the pooled
    cloud keeps the source order, so W2 reads it through the same subsample as every record
    and K = 1 reproduces the seed pool exactly.
    """
    root, runs = recs["root"], recs["runs"]
    seeds = [s for s in range(POOL["ffm"]) if (POOL_DIM, "ffm", s, "none") in runs
             and os.path.exists(mid_path(root, (POOL_DIM, "ffm", s, "none")))]
    if len(seeds) < 2:
        return pd.DataFrame()
    mids = {s: np.load(mid_path(root, (POOL_DIM, "ffm", s, "none")))["mid"].astype(np.float64)
            for s in seeds}
    cloud = load_cloud(POOL_DIM)
    ref = cloud.X[cloud.target_test]
    n = len(mids[seeds[0]])
    rng = np.random.default_rng(0)
    rows = {}
    def pooled(pick):
        out = np.empty_like(mids[pick[0]])
        for group, s in zip(np.array_split(rng.permutation(n), len(pick)), pick):
            out[group] = mids[s][group]
        return out

    for K in ks:
        if K > len(seeds):
            continue
        picks = ([[s] for s in seeds] if K == 1 else
                 [rng.choice(seeds, K, replace=False) for _ in range(n_sub)])
        v = [wasserstein(pooled(pick), ref) for pick in picks]
        rows[K] = {"subsets": len(v), "mean": np.mean(v), "sd": np.std(v, ddof=1)}
    df = pd.DataFrame(rows).T
    df.index.name = f"pooled FFM restarts, d = {POOL_DIM}"
    return df


def frame_band(recs: dict) -> pd.DataFrame:
    """Mid W2 against the one fixed scored half, by the share of the band shown to ``P``."""
    runs = recs["runs"]
    rows = {}
    pct = {f: f"{100 * f:g}%" for f in BAND_FRACS}
    for d in recs["dims"]:
        floor = None
        for a in BAND_ARMS + LB_BAND["arms"][1:]:
            got = {f: [runs[(d, a, s, c)] for s in BASE_SEEDS if (d, a, s, c) in runs]
                   for f, c in BAND.items()}
            got = {f: np.array([r["W2_mid"] for r in v]) for f, v in got.items() if v}
            if not got:
                continue
            floor = runs[next((d, a, s, c) for c in BAND.values() for s in BASE_SEEDS
                              if (d, a, s, c) in runs)]["floors"]["mid"]["W2"]
            row = {pct[f]: v.mean() for f, v in got.items()}
            row.update({f"{pct[f]} pop sd": v.std(ddof=0) for f, v in got.items()})
            if BAND_FRACS[0] in got:
                base = got[BAND_FRACS[0]].mean()
                row.update({f"{pct[f]} vs 0%": 100.0 * (got[f].mean() - base) / base
                            for f in BAND_FRACS[1:] if f in got})
            rows[(d, ARM_LABELS[a])] = row
        if floor is not None:
            rows[(d, "sampling floor")] = {pct[f]: floor for f in BAND_FRACS}
    df = pd.DataFrame(rows).T
    if len(df):
        df.index.names = ["d", "band shown, one scored half"]
    return df


def _search_matrix(recs: dict):
    """``(reported W2, objective)`` as ``(seed, cell)`` arrays, or ``(None, None)`` until
    every cell of the search is on disk."""
    runs = recs["runs"]
    if POOL_DIM not in recs["dims"]:
        return None, None
    rep = np.array([[runs.get((POOL_DIM, "ffm", s, g), {}).get("W2_mid", np.nan)
                     for g in GRID_OF] for s in SEARCH_SEEDS])
    obj = np.array([[runs.get((POOL_DIM, "ffm", s, t), {}).get(OBJECTIVE_KEY, np.nan)
                     for t in TUNE_OF] for s in SEARCH_SEEDS])
    if np.isnan(rep).any() or np.isnan(obj).any():
        return None, None
    return rep, obj


def fresh_cells(recs: dict) -> list[tuple]:
    """Wave 2 of the search: its ``FRESH_TOP`` best cells by four-seed reported mean, at
    ``FRESH_SEEDS`` -- seeds the ranking never saw, so a winner's curse shows up as a
    regression rather than as a result."""
    rep, _ = _search_matrix(recs)
    if rep is None:
        return []
    top = sorted(int(i) for i in np.argsort(rep.mean(axis=0), kind="stable")[:FRESH_TOP])
    return [(POOL_DIM, "ffm", s, f"g{i}") for i in top for s in FRESH_SEEDS]


def frame_fresh(recs: dict) -> pd.DataFrame:
    """The search's best-looking cells at the seeds that picked them and at fresh ones."""
    rep, _ = _search_matrix(recs)
    picks = fresh_cells(recs)
    runs = recs["runs"]
    if rep is None or not all(c in runs for c in picks):
        return pd.DataFrame()
    rows = {}
    for i in sorted({GRID_OF[c[3]] for c in picks}):
        new = np.array([runs[(POOL_DIM, "ffm", s, f"g{i}")]["W2_mid"] for s in FRESH_SEEDS])
        rows[f"g{i}"] = {**GRID[i], "W2, search seeds": rep[:, i].mean(),
                         "W2, fresh seeds": new.mean(), "fresh sd": new.std(ddof=1),
                         "change %": 100.0 * (new.mean() / rep[:, i].mean() - 1.0)}
    df = pd.DataFrame(rows).T
    df.index.name = (f"best {FRESH_TOP} cells: {len(SEARCH_SEEDS)} search seeds vs "
                     f"{len(FRESH_SEEDS)} fresh")
    return df


def frame_search(recs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What the d = 2 search would have picked, and whether its objective predicts the score.

    ``objective`` is the selection objective at the picked cell (on the picking seed for the
    single-seed rows), ``objective, 4 seeds`` the same cell's mean over all four, ``reported
    rank`` its place among the 80 cells by four-seed mean reported W2 (1 = best).
    """
    runs = recs["runs"]
    rep, obj = _search_matrix(recs)
    if rep is None:
        return pd.DataFrame(), pd.DataFrame()
    rep_m, obj_m = rep.mean(axis=0), obj.mean(axis=0)
    rank = rep_m.argsort().argsort() + 1
    shipped = next((r["point"]["ffm"] for c, r in runs.items()
                    if c[:2] == (POOL_DIM, "ffm") and c[3] == "none"), None)

    def row(i, o):
        return {**GRID[i], "objective": o, "objective, 4 seeds": obj_m[i],
                "reported W2": rep_m[i], "reported rank": int(rank[i])}

    rows = {f"picked on seed {s}": row(int(obj[k].argmin()), obj[k].min())
            for k, s in enumerate(SEARCH_SEEDS)}
    rows[f"picked on {len(SEARCH_SEEDS)}-seed mean"] = row(int(obj_m.argmin()), obj_m.min())
    for label, pt in (("notebook TUNED", shipped), ("library default", DEFAULT_POINT)):
        if pt in GRID:
            rows[label] = row(GRID.index(pt), np.nan)
    rows["best reported"] = row(int(rep_m.argmin()), np.nan)
    picks = pd.DataFrame(rows).T.astype({"reported rank": int})
    picks.index.name = f"d = {POOL_DIM} search, {len(GRID)} cells x {len(SEARCH_SEEDS)} seeds"
    stats = pd.DataFrame({"value": {
        "corr(objective, reported), cell means": float(np.corrcoef(obj_m, rep_m)[0, 1]),
        "Spearman, cell means": float(spearmanr(obj_m, rep_m)[0]),
        "corr(objective, reported), single seed": float(np.mean(
            [np.corrcoef(obj[k], rep[k])[0, 1] for k in range(len(SEARCH_SEEDS))])),
        "variance of cell means, reported": float(rep_m.var(ddof=1)),
        "noise variance of a 4-seed mean": float(rep.var(axis=0, ddof=1).mean()
                                                 / len(SEARCH_SEEDS))}})
    stats.index.name = "search signal"
    return picks, stats


def frame_round4(recs: dict) -> pd.DataFrame:
    """Untuned methods: median mid W2 over seeds against the fixed scored half, by the share
    of band shown.  Medians, because an untuned baseline can
    diverge on a seed (Curly-FM at the engine's alpha = 0.01 does at d = 20) and one such
    seed would set a mean; a fraction is printed only once all its seeds are on disk."""
    runs = recs["runs"]
    rows = {}
    pct = {f: f"{100 * f:g}%" for f in R4_FRACS}
    for d in recs["dims"]:
        seeds = R4_SEEDS.get(d, ())
        for a in R4_ARMS:
            got = {f: np.array([runs[(d, a, s, BAND[f])]["W2_mid"] for s in seeds])
                   for f in R4_FRACS if all((d, a, s, BAND[f]) in runs for s in seeds)}
            if not got:
                continue
            row = {pct[f]: float(np.median(v)) for f, v in got.items()}
            first, last = R4_FRACS[0], R4_FRACS[-1]
            if first in got and last in got:
                row[f"{pct[last]} vs 0%"] = 100.0 * (row[pct[last]] / row[pct[first]] - 1)
            row["diverged"] = int(sum((v > 10 * np.median(got[first])).sum()
                                      for v in got.values()))
            rows[(d, ARM_LABELS[a])] = row
    df = pd.DataFrame(rows).T
    if len(df):
        df.index.names = ["d", "untuned, band shown"]
    return df


#: the d = 2 FFM pool re-run on other GPUs, each in its own tree under
#: ``outputs/pancreas_review_hw/<gpu>`` (``FINSLER_OUT`` pointed there at submission)
HW_TREES = {"A100": "a100", "L40S": "l40s"}


def frame_hardware(recs: dict) -> pd.DataFrame:
    """The same 32 shipped-FFM d = 2 runs on each GPU: distribution, the paper's three-seed
    draw, and how far a seed's outcome carries over from one GPU to another."""
    seeds = range(POOL["ffm"])
    base = _w2(recs["runs"], POOL_DIM, "ffm", seeds)
    trees = {"V100": base}
    for gpu, tree in HW_TREES.items():
        root = os.path.join(PUBLIC_ROOT, "outputs", "pancreas_review_hw", tree, KEY)
        got = {}
        for s in seeds:
            p = rec_path(root, (POOL_DIM, "ffm", s, "none"))
            if os.path.exists(p):
                with open(p) as fh:
                    got[s] = json.load(fh)["W2_mid"]
        if len(got) >= 3:
            trees[gpu] = got
    if len(trees) < 2:
        return pd.DataFrame()
    rows = {}
    for gpu, got in trees.items():
        v = np.array(list(got.values()))
        common = [s for s in got if s in base]
        rows[gpu] = {"n": len(v), "mean": v.mean(), "sd": v.std(ddof=1),
                     "seeds 0-2": np.mean([got[s] for s in BASE_SEEDS if s in got]),
                     "corr with V100 by seed": (float(np.corrcoef([got[s] for s in common],
                                                                  [base[s] for s in common])[0, 1])
                                                if gpu != "V100" else np.nan)}
    df = pd.DataFrame(rows).T
    df.index.name = f"shipped FFM, d = {POOL_DIM}, by GPU"
    return df


FRAME_NAMES = ("seeds", "chaos", "arms", "support", "restarts", "band", "round4", "search",
               "search_signal", "search_fresh", "hardware")


def frames(recs: dict, keep_empty: bool = False):
    fs = (frame_seeds(recs), frame_chaos(recs), frame_arms(recs), frame_support(recs),
          frame_restarts(recs), frame_band(recs), frame_round4(recs), *frame_search(recs),
          frame_fresh(recs), frame_hardware(recs))
    return fs if keep_empty else tuple(f for f in fs if len(f))


def show(*fs) -> None:
    """Print each non-empty frame, rounded, one blank line apart -- the notebook's view."""
    print("\n\n".join(f.round(4).to_string() for f in fs if len(f)))


def header(recs: dict) -> str:
    runs = recs["runs"]
    n_grid = len(cells(recs["dims"], recs["smoke"])) + len(fresh_cells(recs))
    devs = sorted({r.get("device", "?") for r in runs.values()})
    return (f"{len(runs)}/{n_grid} runs, {len(recs['support'])} support records, "
            f"{', '.join(devs) or 'no runs'}")


def save_tables(recs: dict, smoke: bool = False) -> str:
    stem = table_file("pancreas_review", smoke)[:-len(".json")]
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fs = frames(recs, keep_empty=True)
    with open(stem + ".txt", "w") as fh:
        fh.write(header(recs) + "\n\n"
                 + "\n\n".join(f.round(4).to_string() for f in fs if len(f)) + "\n")
    for name, f in zip(FRAME_NAMES, fs):
        if len(f):
            f.to_csv(f"{stem}_{name}.csv")
    return stem + ".txt"


# --------------------------------------------------------------------------- #
#  Batch planning and CLI
# --------------------------------------------------------------------------- #
#: rough seconds per cell, for packing cells into jobs of similar length
_DIM_SECONDS = {2: 40.0, 10: 75.0, 20: 120.0, 50: 270.0}
_ARM_COST = {"cfm": 0.3, "curly": 1.7, "curly_default": 1.7, "ffm_neff": 1.6,
             "ffm_barrier": 1.1, "ffm_svr": 3.0, "ffm_logeuc": 1.5, "ffm_mlp": 1.5,
             "ffm_mlp_noise": 1.5, "ffm_lbarrier": 1.5}


def plan(todo, max_seconds: float = 1200.0) -> list[list[tuple]]:
    """Cells packed into jobs: one ``(dim, arm)`` per job, the perturbation draws of a seed
    kept together so they compare within one process, and each ``rerun`` alone so it is a
    fresh one.  Every other cell is free to go to any job of its ``(dim, arm)``."""
    groups = []
    for (d, arm), grp in itertools.groupby(todo, key=lambda c: (c[0], c[1])):
        grp = list(grp)
        groups += [[c] for c in grp if c[3] == RERUN]
        units = []
        for _, by_seed in itertools.groupby([c for c in grp if c[3] != RERUN],
                                            key=lambda c: c[2]):
            by_seed = list(by_seed)
            draws = [c for c in by_seed if c[3] in PERTS]
            units += ([draws] if draws else []) + [[c] for c in by_seed if c[3] not in PERTS]
        cost = _DIM_SECONDS[d] * _ARM_COST.get(arm, 1.0)
        job, spent = [], 0.0
        for unit in units:
            if job and spent + cost * len(unit) > max_seconds:
                groups.append(job)
                job, spent = [], 0.0
            job += unit
            spent += cost * len(unit)
        if job:
            groups.append(job)
    return groups


def _tuned() -> dict:
    from .noise import tuned_point
    return {d: tuned_point(d) for d in DIMS}


def _dims(s: str) -> tuple[int, ...]:
    return tuple(int(d) for d in s.split(","))


def cmd_plan(args) -> None:
    cfg = Run.from_env()
    todo = cells(_dims(args.dims), cfg.smoke) if args.all else \
        missing(cfg, _tuned(), _dims(args.dims))
    if args.arms:
        todo = [c for c in todo if c[1] in args.arms.split(",")]
    if args.conds:
        todo = [c for c in todo if c[3] in args.conds.split(",")]
    for job in plan(todo, args.max_seconds):
        print(",".join(token(c) for c in job))


def cmd_run(args) -> None:
    cfg = Run.from_env(**({"device": torch.device("cpu")} if args.cpu else {}))
    tuned, root, budget = _tuned(), root_dir(cfg.smoke), budget_of(cfg)
    todo = [parse(t) for t in args.cells.split(",") if t.strip()]
    todo = [c for c in todo if args.force
            or not _fresh(rec_path(root, c), point_of(tuned[c[0]], c[1], c[3]), budget)]
    print(f"[review] {len(todo)} cells to run on {_device_name(cfg)}", flush=True)
    run_cells(todo, cfg, tuned, root)


def cmd_support(args) -> None:
    cfg = Run.from_env(**({"device": torch.device("cpu")} if args.cpu else {}))
    ensure_support(cfg, _tuned(), _dims(args.dims), force=args.force)


def cmd_report(args) -> None:
    smoke = Run.from_env().smoke
    recs = load_records(root_dir(smoke), _dims(args.dims), smoke)
    print(header(recs))
    for f in frames(recs):
        print()
        print(f.round(4).to_string())
    if args.save:
        print(f"\nwrote {rel(save_tables(recs, smoke))}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dims", default=",".join(map(str, DIMS)))
    p.add_argument("--max-seconds", type=float, default=1200.0)
    p.add_argument("--all", action="store_true", help="plan every cell, not only missing ones")
    p.add_argument("--arms", default="", help="only these arms (comma list)")
    p.add_argument("--conds", default="", help="only these conditions, e.g. none")
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--cells", required=True)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("support")
    s.add_argument("--dims", default=",".join(map(str, DIMS)))
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_support)
    q = sub.add_parser("report")
    q.add_argument("--dims", default=",".join(map(str, DIMS)))
    q.add_argument("--save", action="store_true")
    q.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
