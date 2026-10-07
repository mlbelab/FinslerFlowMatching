"""The erythroid robustness check at d = 2, as cached run records.

    python -m scripts.experiments.erythroid.review plan
    python -m scripts.experiments.erythroid.review run --cells ffm:s0:none,ffm:s0:p1
    python -m scripts.experiments.erythroid.review report

Whether the erythroid d = 2 column shares the pancreas input sensitivity (W2 moving by
several per cent under a 1e-7 nudge of the input), and whether the tuned FFM point beats
the library default and reaches the endpoint.  This replays :func:`.train.run_cell`'s
test stage -- same training set, harness hyper-parameters, engine arm, scorer -- with
three additions:

* more seeds: FFM at the tuned point and at the library default (``rho_mult = 1``,
  ``lam_mult = 0.3``) and the three harness baselines at 20 seeds each, so the five-seed
  mean the paper reports can be placed in its distribution;
* ``p1..p3``: every coordinate of the training cloud scaled by ``1 + 1e-7 N(0, 1)``
  before ``P`` and the geometry are built, scored against the unperturbed cloud, three
  draws per seed for FFM at both points and for MFM;
* the endpoint: ``W2`` of the pushforward at ``t = 1`` against the target marginal,
  which the release scorer does not report.

``w2_true`` is the paper's W2 column (the release's ``w2`` is a W1, see the paper's note).
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

from scripts.core.arms import run_arm
from scripts.core.metrics import wasserstein
from scripts.core.paths import experiment_root
from scripts.method import Run

from .datasets import TRAIN_BINS, build_training_set, load_cloud
from .train import (N_STEPS_ERYTHROID, _jsonable, engine_arm, evaluate, hparams_for,
                    scored_target, transport_field)

KEY = "erythroid_review"
DIM = 2
ARMS = ("ffm", "ffm_default", "cfm", "mfm_land", "curly")
ARM_LABELS = {"ffm": "FFM, tuned point", "ffm_default": "FFM, default point",
              "cfm": "OT-CFM (harness)", "mfm_land": "MFM / LAND (harness)",
              "curly": "Curly-FM (harness)"}
ENGINE_OF = {"ffm_default": "ffm"}
DEFAULT_POINT = {"rho_mult": 1.0, "lam_mult": 0.3}
PERT_SCALE = 1e-7
PERTS = ("none", "p1", "p2", "p3")
POOL_SEEDS = tuple(range(20))
CHAOS_ARMS = ("ffm", "ffm_default", "mfm_land")
CHAOS_SEEDS = (0, 1, 2)
#: the seeds the paper's table reports
PAPER_SEEDS = (0, 1, 2, 3, 4)
#: exact EMD on both ~3300-cell clouds, like ``w2_true``; the 400-point default reads high
#: by an arm-dependent amount
END_MAX_N = 10_000


def cells(smoke: bool = False) -> list[tuple[str, int, str]]:
    out = {(a, s, "none") for a in ARMS for s in POOL_SEEDS}
    out |= {(a, s, p) for a in CHAOS_ARMS for s in CHAOS_SEEDS for p in PERTS}
    if smoke:
        out = {c for c in out if c[1] == 0 and c[2] in ("none", "p1")}
    return sorted(out, key=lambda c: (ARMS.index(c[0]), c[1], PERTS.index(c[2])))


def token(c) -> str:
    return f"{c[0]}:s{c[1]}:{c[2]}"


def parse(tok: str):
    arm, s, p = tok.strip().split(":")
    assert arm in ARMS and s[0] == "s" and p in PERTS, f"bad cell {tok!r}"
    return arm, int(s[1:]), p


def root_dir(smoke: bool = False) -> str:
    return experiment_root(KEY, smoke)


def rec_path(root: str, c) -> str:
    return os.path.join(root, "cells", token(c).replace(":", "_") + ".json")


def point_of(tuned_dim: dict, arm: str, pert: str = "none") -> dict:
    pt = dict(DEFAULT_POINT if arm == "ffm_default" else tuned_dim[arm])
    return {**pt, "pert": int(pert[1:])} if pert != "none" else pt


def budget_of(cfg: Run) -> dict:
    return {"n_steps": N_STEPS_ERYTHROID, "dtype": str(cfg.dtype), "dim": DIM,
            "end_w2": "exact"}


def _fresh(path: str, point: dict, budget: dict) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        prior = json.load(fh)
    return prior.get("point") == point and prior.get("budget") == budget


def tuned_points() -> dict:
    """The erythroid ``TUNED`` literal's d = 2 block, read from ``tuned.json``, which the
    notebook asserts its literal against."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")) as fh:
        return {int(d): v for d, v in json.load(fh).items()}[DIM]


def missing(cfg: Run, tuned_dim: dict, root: str | None = None) -> list:
    root = root_dir(cfg.smoke) if root is None else root
    return [c for c in cells(cfg.smoke)
            if not _fresh(rec_path(root, c), point_of(tuned_dim, c[0], c[2]), budget_of(cfg))]


def run_one(c, cfg: Run, tuned_dim: dict, cloud) -> dict:
    """:func:`.train.run_cell` at ``stage = "test"``, perturbing only what training sees."""
    arm, seed, pert = c
    train_cloud = cloud
    if pert != "none":
        rng = np.random.default_rng(10_000 + int(pert[1:]))
        train_cloud = replace(cloud, X=cloud.X * (1.0 + PERT_SCALE
                                                   * rng.standard_normal(cloud.X.shape)))
    ts = build_training_set(train_cloud)
    eng = ENGINE_OF.get(arm, arm)
    over = DEFAULT_POINT if arm == "ffm_default" else tuned_dim[arm]
    t0 = time.time()
    hp, scales = hparams_for(DIM, eng, over, ts, cfg.device, cfg.dtype, cfg.smoke)
    res = run_arm(engine_arm(eng), ts, hp, seed, cfg.device, cfg.dtype,
                  n_steps=N_STEPS_ERYTHROID)
    train_s = time.time() - t0
    field, tag = transport_field(res, hp)
    metrics = evaluate(res.traj, field, cloud, scored_target(cloud), cfg.device, cfg.dtype,
                       seed=seed)
    traj = np.asarray(res.traj, dtype=np.float64)
    end = wasserstein(traj[-1], cloud.X[cloud.marginal(TRAIN_BINS[1])], max_n=END_MAX_N)
    keep = ("w2_true", "w2", "cos_dist", "l2")
    return {"arm": arm, "seed": seed, "pert": pert, "point": point_of(tuned_dim, arm, pert),
            "budget": budget_of(cfg), **{k: float(metrics[k]) for k in keep},
            "end_w2": float(end), "scored_field": tag, "scales": _jsonable(scales),
            "train_seconds": train_s, "smoke": bool(cfg.smoke),
            "device": (torch.cuda.get_device_name(0) if cfg.device.type == "cuda"
                       else "cpu")}


def run_cells(todo, cfg: Run, tuned_dim: dict, root: str, verbose: bool = True) -> None:
    cloud = load_cloud(DIM)
    for c in todo:
        rec = run_one(c, cfg, tuned_dim, cloud)
        path = rec_path(root, c)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
        if verbose:
            print(f"[erythroid review] {token(c)}  W2 {rec['w2_true']:.4f}  "
                  f"end W2 {rec['end_w2']:.4f}  ({rec['train_seconds']:.0f}s)", flush=True)


def ensure(cfg: Run, tuned: dict, root: str | None = None, verbose: bool = True) -> dict:
    """Every cell, training only what disk cannot answer; ``tuned`` is the notebook's
    ``TUNED`` literal (its d = 2 block is used)."""
    root = root_dir(cfg.smoke) if root is None else root
    todo = missing(cfg, tuned[DIM], root)
    if todo:
        run_cells(todo, cfg, tuned[DIM], root, verbose)
    return load_records(root, cfg.smoke)


def load_records(root: str | None = None, smoke: bool = False) -> dict:
    root = root_dir(smoke) if root is None else root
    out = {}
    for c in cells(smoke):
        p = rec_path(root, c)
        if os.path.exists(p):
            with open(p) as fh:
                out[c] = json.load(fh)
    return out


def frame_seeds(recs: dict) -> pd.DataFrame:
    rows = {}
    ffm = np.array([recs[("ffm", s, "none")]["w2_true"] for s in POOL_SEEDS
                    if ("ffm", s, "none") in recs])
    for a in ARMS:
        got = [recs[(a, s, "none")] for s in POOL_SEEDS if (a, s, "none") in recs]
        if len(got) < 3:
            continue
        v = np.array([r["w2_true"] for r in got])
        paper = [recs[(a, s, "none")]["w2_true"] for s in PAPER_SEEDS if (a, s, "none") in recs]
        rows[ARM_LABELS[a]] = {
            "n": len(v), "mean": v.mean(), "sd": v.std(ddof=1), "q2.5": np.percentile(v, 2.5),
            "q97.5": np.percentile(v, 97.5), "seeds 0-4": np.mean(paper),
            "end W2": np.mean([r["end_w2"] for r in got]),
            "P(FFM tuned run wins)": (float((ffm[:, None] < v[None, :]).mean())
                                      if a != "ffm" and len(ffm) else np.nan)}
    df = pd.DataFrame(rows).T
    if len(df):
        df.index.name = f"erythroid d = {DIM}, W2 (paper's column)"
    return df


def frame_chaos(recs: dict) -> pd.DataFrame:
    rows = {}
    for a in CHAOS_ARMS:
        M = np.array([[recs[(a, s, p)]["w2_true"] for p in PERTS] for s in CHAOS_SEEDS
                      if all((a, s, p) in recs for p in PERTS)])
        if not len(M):
            continue
        pool = np.array([recs[(a, s, "none")]["w2_true"] for s in POOL_SEEDS
                         if (a, s, "none") in recs])
        within = float(np.sqrt(M.var(axis=1, ddof=1).mean()))
        rows[ARM_LABELS[a]] = {"seeds": len(M), "W2": M[:, 0].mean(),
                               "chaos %": 100.0 * within / M.mean(),
                               "within / seed sd": within / pool.std(ddof=1)}
    df = pd.DataFrame(rows).T
    if len(df):
        df.index.name = f"erythroid d = {DIM}, 1e-7 perturbation"
    return df


def frames(recs: dict):
    return tuple(f for f in (frame_seeds(recs), frame_chaos(recs)) if len(f))


#: rough seconds per cell, for packing
_SECONDS = {"curly": 60.0}


def plan(todo, max_seconds: float = 900.0) -> list[list]:
    jobs = []
    for _, grp in itertools.groupby(todo, key=lambda c: c[0]):
        units = []
        for _, by_seed in itertools.groupby(list(grp), key=lambda c: c[1]):
            units.append(list(by_seed))
        job, spent = [], 0.0
        for u in units:
            cost = _SECONDS.get(u[0][0], 40.0) * len(u)
            if job and spent + cost > max_seconds:
                jobs.append(job)
                job, spent = [], 0.0
            job += u
            spent += cost
        if job:
            jobs.append(job)
    return jobs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--max-seconds", type=float, default=900.0)
    r = sub.add_parser("run")
    r.add_argument("--cells", required=True)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    cfg = Run.from_env()
    tuned_dim = tuned_points()
    root = root_dir(cfg.smoke)
    if a.cmd == "plan":
        for job in plan(missing(cfg, tuned_dim, root), a.max_seconds):
            print(",".join(token(c) for c in job))
    elif a.cmd == "run":
        todo = [parse(t) for t in a.cells.split(",") if t.strip()]
        todo = [c for c in todo if not _fresh(rec_path(root, c), point_of(tuned_dim, c[0], c[2]),
                                              budget_of(cfg))]
        run_cells(todo, cfg, tuned_dim, root)
    else:
        for f in frames(load_records(root, cfg.smoke)):
            print(f.round(4).to_string())
            print()


if __name__ == "__main__":
    main()
