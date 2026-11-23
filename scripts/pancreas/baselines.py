from __future__ import annotations

import argparse
import glob
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from ffm.core.arms import HParams, run_arm
from ffm.core.paths import experiment_root
from ffm.core.selection import OBJECTIVE_KEY, with_objective
from ffm.method import (FWFactory, Fanout, Grid, Run, cell_tag, euclidean_w2,
                            moments_from, rank, sigma_for)
from ffm.method.metric import Metric

from . import train as tr
from .datasets import (DIMS, TRAIN_BINS, build_training_set, load_cloud, split_holdout,
                       split_shown)

ROOT = experiment_root("pancreas_baselines")
HERE = os.path.dirname(os.path.abspath(__file__))
TUNED_FILE = os.path.join(HERE, "tuned_baselines.json")
SIGMA_MULTS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
WAVE = Fanout({
    "sf2m": Grid(width_mult=SIGMA_MULTS),
    "cdcfm": Grid(cdc_gamma=(0.1, 0.4, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0), cdc_rank=(2, 8, 0),
                  cdc_sigma_min=(0.0, 0.3)),
})
ARMS = tuple(WAVE.grids)
SEEDS = (0, 1, 2, 3, 4)
N_STEPS = 102


def _eps(X, cfg) -> SimpleNamespace:
    return SimpleNamespace(eps_kernel=Metric(np.asarray(X, dtype=np.float64), cfg).eps_kernel)


def cell_path(root: str, dim: int, arm: str, point: dict) -> str:
    return os.path.join(root, "sweep", cell_tag(3, arm, point, dim=dim) + ".json")


def run_cell(dim: int, index: int, cfg: Run, root: str = ROOT, force: bool = False):
    arm, point = WAVE[index]
    path = cell_path(root, dim, arm, point)
    if os.path.exists(path) and not force:
        return None
    cloud = load_cloud(dim)
    i_train, i_val = split_shown(cloud)
    ts = build_training_set(cloud, subset=i_train)
    hp = HParams(**cfg.arm_kw, **tr.engine_point(arm, point, _eps(ts.X, cfg)))
    t0 = time.time()
    traj = np.asarray(run_arm(arm, ts, hp, cfg.tune_seed, cfg.device, cfg.dtype,
                              n_steps=N_STEPS).traj, dtype=np.float64)
    i_end = i_val[cloud.bin_id[i_val] == TRAIN_BINS[1]]
    i_sel, _ = split_holdout(cloud)
    rec = {"dim": dim, "arm": arm, "point": point, "index": index,
           **with_objective({
               "W2_endpoint": euclidean_w2(traj[-1], cloud.X[i_end], seed=cfg.tune_seed),
               "W2_intermediate": euclidean_w2(traj[traj.shape[0] // 2], cloud.X[i_sel],
                                               seed=cfg.tune_seed)}),
           "seed": cfg.tune_seed, "train_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[cell] d{dim} {arm} {point} objective {rec[OBJECTIVE_KEY]:.4f}")
    return rec


def collect(root: str = ROOT, write: bool = True) -> dict:
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(root, "sweep", "*.json"))]
    tuned = {}
    for dim in DIMS:
        for arm in ARMS:
            cells = [r for r in rows if r["dim"] == dim and r["arm"] == arm]
            if len(cells) == len(WAVE.grids[arm]):
                tuned.setdefault(dim, {})[arm] = rank(cells, OBJECTIVE_KEY)[1]["point"]
    if write:
        with open(TUNED_FILE, "w") as fh:
            json.dump({str(k): v for k, v in tuned.items()}, fh, indent=1)
    return tuned


def tuned_points() -> dict:
    base = json.load(open(os.path.join(HERE, "tuned.json")))
    extra = json.load(open(TUNED_FILE)) if os.path.exists(TUNED_FILE) else {}
    return {int(d): {**base[d], **extra.get(d, {})} for d in base}


def context(dims, cfg: Run, tuned: dict, cache_dir: str | None = None) -> dict:
    kw = {} if cache_dir is None else {"cache_dir": cache_dir}
    clouds = {d: load_cloud(d, **kw) for d in dims}
    tsets = {d: build_training_set(clouds[d]) for d in dims}
    tens, metric, sigma = {}, {}, {}
    for d in dims:
        ts = tsets[d]
        X = np.asarray(ts.X, dtype=np.float64)
        X_t = cfg.tensor(X)
        tens[d] = (X_t[torch.as_tensor(ts.p0, device=cfg.device)],
                   X_t[torch.as_tensor(ts.p1, device=cfg.device)])
        b0, D = moments_from(ts.P, X)
        pt = tuned[d]["ffm"]
        metric[d] = FWFactory(X, D, b0, cfg, rho_mult=pt["rho_mult"],
                              lam_mult=pt["lam_mult"]).base
        sigma[d] = sigma_for(metric[d], tuned[d]["pathb"]["width_mult"])
    return {"clouds": clouds, "tsets": tsets, "tens": tens, "metric": metric,
            "sigma": sigma}


def test(dims, seeds, cfg: Run, root: str = ROOT, arms=ARMS) -> None:
    tuned = tuned_points()
    tr.stage_test(root, dims, tuned, seeds, cfg, n_steps=N_STEPS, arms=arms,
                  **context(dims, cfg, tuned))


REFERENCE = experiment_root("pancreas_leakage")


def records(root: str = ROOT, arms=None) -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(root, tr.TEST_DIR, "*.json")):
        r = json.load(open(p))
        if arms is not None and r["arm"] not in arms:
            continue
        m = r["metrics"]
        rows.append({"d": r["dim"], "arm": r["arm"], "seed": r["seed"],
                     "mid W2": m["intermediate"]["W2"], "end W2": m["endpoint"]["W2"],
                     "mid W2F": m["intermediate"].get("W2F", np.nan),
                     "seconds": r["train_seconds"]})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, cols=("mid W2", "end W2")) -> pd.DataFrame:
    g = df.groupby(["d", "arm"])
    out = pd.DataFrame({c: g[c].mean().map("{:.4f}".format) + " ± "
                        + g[c].std(ddof=0).map("{:.4f}".format) for c in cols})
    out["n"] = g.size()
    return out


def frames(root: str = ROOT, reference: str = os.path.join(REFERENCE, "original")
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuned = tuned_points()
    pts = pd.DataFrame({d: {a: tuned[d].get(a) for a in ARMS} for d in tuned}).T
    df = pd.concat([records(reference, arms=tr.ARMS), records(root, arms=ARMS)])
    df["arm"] = df["arm"].map(tr.ARM_LABELS)
    return summary(df), pts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dim", type=int, required=True)
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--force", action="store_true")
    sub.add_parser("collect")
    t = sub.add_parser("test")
    t.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    t.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    sub.add_parser("count")
    a = ap.parse_args(argv)
    cfg = Run.from_env()
    if a.cmd == "run":
        run_cell(a.dim, a.index, cfg, a.root, a.force)
    elif a.cmd == "collect":
        print(json.dumps({str(k): v for k, v in collect(a.root).items()}))
    elif a.cmd == "test":
        test(a.dims, a.seeds, cfg, a.root)
    else:
        print(len(WAVE))


if __name__ == "__main__":
    main()
