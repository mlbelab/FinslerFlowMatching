from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from ffm.core.arms import run_arm
from ffm.core.metrics import wasserstein
from ffm.core.paths import PUBLIC_ROOT, experiment_root
from ffm.method import Run

from .datasets import TRAIN_BINS, build_training_set, load_cloud
from .reference_release import release_wasserstein
from .train import (HARNESS_BUDGET_NAME, N_STEPS_ERYTHROID, TEST_DIR, TRAJ_DIR, _jsonable,
                    engine_arm, evaluate, hparams_for, scored_target, transport_field)

KEY = "erythroid_defaults"
BENCH = "erythroid_bench"
REVIEW = "erythroid_review"
DATASET = "erythroid"
DIMS = (2, 20, 50)
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("ffm", "pathb")
BASELINES = ("cfm", "mfm_land", "curly")
CONFIGS = ("library", "lodo")
LIBRARY = {"ffm": {"rho_mult": 1.0, "lam_mult": 0.3}, "pathb": {"width_mult": 0.3}}
DATASETS = ("pancreas", "erythroid", "itracer")
FFM_OF = {"pancreas": ("ffm",), "erythroid": ("ffm",),
          "itracer": ("ffm_expr", "ffm_velo", "ffm", "ffm_lin_velo")}
LODO_RULE = ("10**median(log10) per coordinate over every tuned point of the other "
             "datasets (all dims; iTracer all four kernels), 4 significant digits")
CONFIG_LABELS = {"tuned": "tuned", "library": "D0 library", "lodo": "D1 LODO"}
BASE_LABELS = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly": "Curly-FM"}
KEEP = ("w2_true", "w2", "cos_dist", "l2")
END_MAX_N = 10_000
REF = "2941 test cells (Table E1)"


def tuned_points(name: str = DATASET) -> dict:
    path = os.path.join(PUBLIC_ROOT, "scripts", name, "tuned.json")
    with open(path) as fh:
        return {int(d): v for d, v in json.load(fh).items()}


def _log_median(values) -> float:
    return float(f"{10.0 ** np.median(np.log10(np.asarray(values, dtype=float))):.4g}")


def lodo_inputs(held: str = DATASET) -> dict:
    ffm, width = [], []
    for name in DATASETS:
        if name == held:
            continue
        for d, at in sorted(tuned_points(name).items()):
            ffm += [{"dataset": name, "dim": d, "arm": a, **at[a]} for a in FFM_OF[name]]
            if "pathb" in at:
                width.append({"dataset": name, "dim": d, **at["pathb"]})
    return {"ffm": ffm, "pathb": width}


def lodo_point(held: str = DATASET) -> dict:
    inp = lodo_inputs(held)
    return {"ffm": {k: _log_median([p[k] for p in inp["ffm"]])
                    for k in ("rho_mult", "lam_mult")},
            "pathb": {"width_mult": _log_median([p["width_mult"] for p in inp["pathb"]])}}


def config_point(config: str, dim: int) -> dict:
    if config == "tuned":
        return tuned_points()[dim]
    return LIBRARY if config == "library" else lodo_point()


def point_of(config: str, dim: int, arm: str) -> dict:
    at = config_point(config, dim)
    return {**at["ffm"], **at["pathb"]} if arm == "pathb" else dict(at[arm])


def budget_of(cfg: Run) -> dict:
    return {"n_steps": N_STEPS_ERYTHROID, "dtype": str(cfg.dtype),
            "harness": HARNESS_BUDGET_NAME, "end_w2": "exact"}


def root_dir(smoke: bool = False) -> str:
    return experiment_root(KEY, smoke)


def seeds_of(smoke: bool) -> tuple:
    return (0,) if smoke else SEEDS


def rec_path(root: str, c) -> str:
    config, dim, arm, seed = c
    return os.path.join(root, config, f"d{dim}_{arm}_s{seed}.json")


def cells(seeds=SEEDS) -> list:
    return ([("tuned", d, a, s) for d in DIMS for a in ARMS + BASELINES for s in seeds]
            + [(c, d, a, s) for c in CONFIGS for d in DIMS for a in ARMS for s in seeds])


def _fresh(path: str, point: dict, budget: dict) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        prior = json.load(fh)
    return prior.get("point") == point and prior.get("budget") == budget


def _write(rec: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(rec), fh, indent=1, sort_keys=True)


def _end_w2(end: np.ndarray, cloud) -> float:
    return float(wasserstein(np.asarray(end, dtype=np.float64),
                             cloud.X[cloud.marginal(TRAIN_BINS[1])], max_n=END_MAX_N))


def run_job(config: str, dim: int, seed: int, cfg: Run, root: str,
            force: bool = False) -> int:
    budget = budget_of(cfg)
    todo = [a for a in ARMS if force or not _fresh(
        rec_path(root, (config, dim, a, seed)), point_of(config, dim, a), budget)]
    if not todo:
        return 0
    cloud = load_cloud(dim)
    ts = build_training_set(cloud)
    for arm in todo:
        point = point_of(config, dim, arm)
        t0 = time.time()
        hp, scales = hparams_for(dim, arm, point, ts, cfg.device, cfg.dtype, cfg.smoke)
        res = run_arm(engine_arm(arm), ts, hp, seed, cfg.device, cfg.dtype,
                      n_steps=N_STEPS_ERYTHROID)
        seconds = time.time() - t0
        field, tag = transport_field(res, hp)
        metrics = evaluate(res.traj, field, cloud, scored_target(cloud), cfg.device,
                           cfg.dtype, seed=seed)
        traj = np.asarray(res.traj, dtype=np.float64)
        rec = {"config": config, "dim": dim, "arm": arm, "seed": seed, "point": point,
               "budget": budget, **{k: float(metrics[k]) for k in KEEP},
               "end_w2": _end_w2(traj[-1], cloud), "scored_field": tag,
               "scales": scales, "train_seconds": seconds,
               "device": (torch.cuda.get_device_name(0) if cfg.device.type == "cuda"
                          else "cpu"),
               "torch": torch.__version__, "smoke": bool(cfg.smoke)}
        if config == "lodo":
            rec["lodo"] = {"rule": LODO_RULE, "inputs": lodo_inputs()}
        _write(rec, rec_path(root, (config, dim, arm, seed)))
        print(f"[defaults] {config} d{dim} {arm} s{seed}  W2 {rec['w2_true']:.4f}  "
              f"end W2 {rec['end_w2']:.4f}  ({seconds:.0f}s)", flush=True)
    return len(todo)


def backfill(dim: int, root: str, seeds=SEEDS, force: bool = False) -> int:
    bench = experiment_root(BENCH)
    cloud, n = None, 0
    for arm in ARMS + BASELINES:
        for seed in seeds:
            path = rec_path(root, ("tuned", dim, arm, seed))
            src = os.path.join(bench, TEST_DIR, f"d{dim}_{arm}_s{seed}.json")
            slab = os.path.join(bench, TRAJ_DIR, f"d{dim}_{arm}_s{seed}.npz")
            if (os.path.exists(path) and not force) or not (
                    os.path.exists(src) and os.path.exists(slab)):
                continue
            cloud = load_cloud(dim) if cloud is None else cloud
            with open(src) as fh:
                shipped = json.load(fh)
            point = {**(shipped.get("inherited") or {}), **shipped["override"]}
            assert point == point_of("tuned", dim, arm), (
                f"{src} was run at {point}, tuned.json says {point_of('tuned', dim, arm)}")
            with np.load(slab) as z:
                mid, end = np.asarray(z["mid"]), np.asarray(z["end"])
            check = release_wasserstein(
                torch.as_tensor(mid, dtype=torch.float32),
                torch.as_tensor(cloud.X[scored_target(cloud)], dtype=torch.float32),
                power=2)
            m = shipped["metrics"]
            assert abs(check - m["w2_true"]) <= 1e-3 * max(1.0, m["w2_true"]), (
                f"{slab} gives W2 {check}, the record says {m['w2_true']}")
            rec = {"config": "tuned", "dim": dim, "arm": arm, "seed": seed, "point": point,
                   "source": f"{BENCH}/{TEST_DIR}", **{k: float(m[k]) for k in KEEP},
                   "end_w2": _end_w2(end, cloud), "w2_true_from_slab": float(check),
                   "train_seconds": shipped.get("train_seconds"),
                   "smoke": bool(shipped.get("smoke"))}
            _write(rec, path)
            n += 1
            print(f"[defaults] tuned d{dim} {arm} s{seed}  W2 {rec['w2_true']:.4f}  "
                  f"end W2 {rec['end_w2']:.4f}  (from shipped slab)", flush=True)
    return n


def ensure(cfg: Run, configs=CONFIGS, dims=DIMS, root: str | None = None) -> dict:
    root = root_dir(cfg.smoke) if root is None else root
    for d in dims:
        backfill(d, root, seeds_of(cfg.smoke))
        for c in configs:
            for s in seeds_of(cfg.smoke):
                run_job(c, d, s, cfg, root)
    return load_records(root, cfg.smoke)


def load_records(root: str | None = None, smoke: bool = False) -> dict:
    root = root_dir(smoke) if root is None else root
    out = {}
    for c in cells(seeds_of(smoke)):
        p = rec_path(root, c)
        if os.path.exists(p):
            with open(p) as fh:
                out[c] = json.load(fh)
    return out


def _welch(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((b.mean() - a.mean()) / se) if se > 0 else np.nan


def compare(dim, ref: str, ffm: dict, base: dict) -> list:
    stat = lambda mid, end: {"n": len(mid), "mid W2": mid.mean(), "sd": mid.std(),
                             "end W2": end.mean(), "end sd": end.std()}
    base = {k: v for k, v in base.items() if len(v[0])}
    best = min(base, key=lambda k: base[k][0].mean()) if base else None
    rows = []
    for (variant, config), (mid, end) in ffm.items():
        if not len(mid):
            continue
        tuned = ffm.get((variant, "tuned"), (np.array([]),))[0]
        row = {"d": dim, "method": f"FFM {variant}", "config": config, "reference": ref,
               **stat(mid, end),
               "regret %": (100.0 * (mid.mean() / tuned.mean() - 1.0)
                            if len(tuned) and config != "tuned" else np.nan)}
        if best is not None:
            bm = base[best][0]
            row.update({"best tuned baseline": best,
                        "vs best %": 100.0 * (mid.mean() / bm.mean() - 1.0),
                        "t (best - FFM)": _welch(mid, bm),
                        "beats all tuned baselines": bool(mid.mean() < bm.mean())})
        rows.append(row)
    rows += [{"d": dim, "method": k, "config": "tuned", "reference": ref, **stat(*v)}
             for k, v in base.items()]
    return rows


def _indexed(rows: list, keys=("d", "method", "config")) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index(list(keys)) if rows else pd.DataFrame()


def _vals(recs: dict, config: str, dim: int, arm: str, key: str) -> np.ndarray:
    return np.array([r[key] for (c, d, a, _), r in sorted(recs.items())
                     if (c, d, a) == (config, dim, arm)], dtype=float)


def frame_main(recs: dict) -> pd.DataFrame:
    rows = []
    for d in DIMS:
        pair = lambda c, a: (_vals(recs, c, d, a, "w2_true"),
                             _vals(recs, c, d, a, "end_w2"))
        ffm = {(v, CONFIG_LABELS[c]): pair(c, a)
               for v, a in (("det", "ffm"), ("noise", "pathb"))
               for c in ("tuned",) + CONFIGS}
        base = {BASE_LABELS[a]: pair("tuned", a) for a in BASELINES}
        rows += compare(d, REF, ffm, base)
    return _indexed(rows)


def frame_summary(main: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for d in main.index.get_level_values(0).unique() if len(main) else ():
        blk = main.loc[d]
        is_ffm = np.asarray(blk.index.get_level_values(0).str.startswith("FFM"))
        ffm, base = blk[is_ffm], blk[~is_ffm]
        row = {f"{m} {c}": r["mid W2"] for (m, c), r in ffm.iterrows()}
        if len(base):
            best = base["mid W2"].idxmin()
            row.update({"best tuned baseline": best[0],
                        "best baseline W2": base.loc[best, "mid W2"]})
            if ("Curly-FM", "tuned") in base.index:
                row["Curly-FM"] = base.loc[("Curly-FM", "tuned"), "mid W2"]
        defaults = [(m, c, r) for (m, c), r in ffm.iterrows() if c != "tuned"]
        row.update({f"{m} {c} regret %": r["regret %"] for m, c, r in defaults})
        row.update({f"{m} {c} beats all": r.get("beats all tuned baselines", np.nan)
                    for m, c, r in defaults})
        out[d] = row
    df = pd.DataFrame(out).T
    df.index.name = "d"
    return df


def frame_points() -> pd.DataFrame:
    rows = {"D0 library (Sheet-tuned)": {"rho_mult": LIBRARY["ffm"]["rho_mult"],
                                         "lam_mult": LIBRARY["ffm"]["lam_mult"],
                                         "width_mult": LIBRARY["pathb"]["width_mult"],
                                         "n FFM inputs": 0, "n width inputs": 0}}
    for held in DATASETS:
        p, inp = lodo_point(held), lodo_inputs(held)
        rows[f"D1 LODO for {held}"] = {**p["ffm"], **p["pathb"],
                                       "n FFM inputs": len(inp["ffm"]),
                                       "n width inputs": len(inp["pathb"])}
    df = pd.DataFrame(rows).T
    df["c_eff"] = 1.0 / np.sqrt(1.0 + df["lam_mult"].astype(float) ** 2)
    return df


def frame_review() -> pd.DataFrame:
    root = experiment_root(REVIEW)

    def pair(a, seeds):
        got = []
        for s in seeds:
            p = os.path.join(root, "cells", f"{a}_s{s}_none.json")
            if os.path.exists(p):
                with open(p) as fh:
                    got.append(json.load(fh))
        return (np.array([r["w2_true"] for r in got], dtype=float),
                np.array([r["end_w2"] for r in got], dtype=float))

    rows = []
    for sample, seeds in (("seeds 0-4", range(5)), ("20 seeds", range(20))):
        ffm = {("det", "tuned"): pair("ffm", seeds),
               ("det", "D0 library"): pair("ffm_default", seeds)}
        base = {BASE_LABELS[a]: pair(a, seeds) for a in BASELINES}
        rows += [{**r, "sample": sample} for r in compare(2, REF, ffm, base)]
    return _indexed(rows, ("sample", "d", "method", "config"))


def frames(recs: dict | None = None, smoke: bool = False) -> tuple:
    recs = load_records(smoke=smoke) if recs is None else recs
    main = frame_main(recs)
    out = (main, frame_summary(main), frame_points(), frame_review())
    return tuple(f for f in out if len(f))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Erythroid FFM at pre-specified default points")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", choices=CONFIGS, required=True)
    r.add_argument("--dim", type=int, choices=DIMS, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--force", action="store_true")
    b = sub.add_parser("backfill")
    b.add_argument("--dim", type=int, choices=DIMS, required=True)
    b.add_argument("--force", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    cfg = Run.from_env()
    root = root_dir(cfg.smoke)
    if a.cmd == "run":
        run_job(a.config, a.dim, a.seed, cfg, root, a.force)
    elif a.cmd == "backfill":
        backfill(a.dim, root, seeds_of(cfg.smoke), a.force)
    else:
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            for f in frames(smoke=cfg.smoke):
                print(f.round(4).to_string())
                print()


if __name__ == "__main__":
    main()
