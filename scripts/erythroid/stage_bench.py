from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from ffm.core.metrics import traj_slice, wasserstein
from ffm.core.paths import experiment_root
from ffm.core.protocol import TrainingSet
from ffm.core.transition import (asymmetry_scale, build_uniform_knn_transition,
                                     build_velocity_kernel)
from scripts.larry import bench as larry
from ffm.method import Run

from .datasets import _flux_alignment
from .stage_data import LOO, SPLITS, cache_path, stage_time

NAME = "erythroid_stage_bench"
DIMS = (20, 50)
SEEDS = (0, 1, 2, 3, 4)
KERNELS = ("velocity", "expression")
UNITS = ("cfm", "mfm_land", "curly", "ffm_velocity", "ffm_expression", "ffm_pancreas")
METHODS = ("cfm", "mfm_land", "curly", "curly_unit", "ffm", "ffm_noise", "ffm_expr", "ffm_expr_noise",
           "ffm_pp")
LABELS = {**{m: larry.LABELS[m] for m in METHODS if m in larry.LABELS},
          "curly": "Curly-FM, transferred alpha", "curly_unit": "Curly-FM, v rescaled to |x1 - x0|, alpha = 1"}
SOURCES = {k: v for k, v in larry.SOURCES.items() if k != "erythroid"}
N_EVAL = 800
EVAL_REPS = 5
KNN = larry.KNN
CURVE = tuple(round(0.05 * k, 2) for k in range(1, 20))
WINDOW_STEPS = 7


def root_dir(smoke: bool = False) -> str:
    return experiment_root(NAME, smoke)


def rec_path(root: str, split: str, d: int, method: str, seed: int) -> str:
    return os.path.join(root, "records", f"{split}_d{d}_{method}_s{seed}.json")


def window_path(root: str, split: str, d: int, method: str, seed: int) -> str:
    return os.path.join(root, "window", f"{split}_d{d}_{method}_s{seed}.json")


def window_of(held: tuple) -> tuple[float, float]:
    return 1.0 / (len(held) + 1), len(held) / (len(held) + 1)


def cloud(split: str, d: int) -> dict:
    with np.load(cache_path(split)) as z:
        c = {k: z[k] for k in z.files}
    src, held, tgt = SPLITS[split]
    st = stage_time(c["stage"])
    t0, t1 = st[np.isin(c["stage"], src)].mean(), st[np.isin(c["stage"], tgt)].mean()
    t = np.where(np.isin(c["stage"], held), 0.5, np.nan) if split in LOO else (st - t0) / (t1 - t0)
    return {"split": split, "X": c["X"][:, :d].astype(np.float64), "V": c["velocity"][:, :d].astype(np.float64),
            "stage": c["stage"], "celltype": c["celltype"], "t": t, "latent": c["latent_time"]}


def n_eval(c: dict) -> int:
    return int(min(N_EVAL, np.isin(c["stage"], SPLITS[c["split"]][0]).sum()))


def training_set(c: dict, kernel: str) -> TrainingSet:
    src, _, tgt = SPLITS[c["split"]]
    idx = np.flatnonzero(np.isin(c["stage"], (*src, *tgt)))
    X, V = c["X"][idx], c["V"][idx]
    P = build_uniform_knn_transition(X) if kernel == "expression" else build_velocity_kernel(X, V)
    st = c["stage"][idx]
    return TrainingSet(name=f"Erythroid-{c['split']}-d{X.shape[1]}-{kernel}", idx=idx, X=X,
                       P=sp.csr_matrix(P), p0=np.isin(st, src), p1=np.isin(st, tgt), rho=0.0,
                       holdout_band=None, p_variant=kernel, diag={"n_train": int(len(idx))}, velocity=V)


def velocity_ratio(ts: TrainingSet) -> float:
    rng = np.random.default_rng(0)
    X0, X1 = ts.X[ts.p0], ts.X[ts.p1]
    dx = X1[rng.integers(0, len(X1), 4000)] - X0[rng.integers(0, len(X0), 4000)]
    return float(np.median(np.linalg.norm(ts.velocity, axis=1)) / np.median(np.linalg.norm(dx, axis=1)))


def train_curly_unit(ts: TrainingSet, pts: dict, seed: int, cfg: Run) -> dict:
    r = velocity_ratio(ts)
    traj, extra = larry.train_unit("curly", replace(ts, velocity=ts.velocity / r),
                                   {**pts, "curly": {"curly_alpha": 1.0}}, seed, cfg)["curly"]
    return {"curly_unit": (traj, {**extra, "velocity_scale": 1.0 / r})}


def _halves(H: np.ndarray, k: int, n: int) -> float:
    perm = np.random.default_rng(k).permutation(len(H))
    h = len(H) // 2
    return wasserstein(H[perm[:h]], H[perm[h:]], p=2, max_n=n, seed=k)


def _refs(H: np.ndarray, S: np.ndarray, T: np.ndarray, n: int) -> dict:
    rows = {"floor": [_halves(H, k, n) for k in range(EVAL_REPS)],
            "copy_source": [wasserstein(S, H, p=2, max_n=n, seed=k) for k in range(EVAL_REPS)],
            "copy_target": [wasserstein(T, H, p=2, max_n=n, seed=k) for k in range(EVAL_REPS)]}
    return {**{k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in rows.items()},
            "n": int(len(H)), "n_eval": n}


def references(c: dict) -> dict:
    src, held, tgt = SPLITS[c["split"]]
    S, T = c["X"][np.isin(c["stage"], src)], c["X"][np.isin(c["stage"], tgt)]
    n = n_eval(c)
    out = {s: {**_refs(c["X"][c["stage"] == s], S, T, n), "t": float(c["t"][c["stage"] == s][0])} for s in held}
    if len(held) > 1:
        out["+".join(held)] = {**_refs(c["X"][np.isin(c["stage"], held)], S, T, n), "t": 0.5,
                               "window": list(window_of(held))}
    return out


def diagnostics(c: dict, root: str) -> dict:
    out = {}
    for kernel in KERNELS:
        ts = training_set(c, kernel)
        larry.save_kernel(ts, os.path.join(root, "kernels", f"{c['split']}_{kernel}_d{ts.X.shape[1]}.npz"))
        X, P = ts.X, ts.P
        m = np.asarray(P @ X) - X
        axis = X[ts.p1].mean(0) - X[ts.p0].mean(0)
        off = (P - sp.diags(P.diagonal())).tocsr()
        tot = np.asarray(off.sum(1)).ravel()
        fwd = np.asarray(off[:, np.flatnonzero(ts.p1)].sum(1)).ravel()
        bwd = np.asarray(off[:, np.flatnonzero(ts.p0)].sum(1)).ravel()
        cm = (m @ axis) / np.maximum(np.linalg.norm(m, axis=1) * np.linalg.norm(axis), 1e-300)
        out[kernel] = {"P_asym": float(asymmetry_scale(P)), "cos_drift_axis_source": float(cm[ts.p0].mean()),
                       "cos_drift_axis_target": float(cm[ts.p1].mean()),
                       "mass_source_to_target": float((fwd[ts.p0] / tot[ts.p0]).mean()),
                       "mass_target_to_source": float((bwd[ts.p1] / tot[ts.p1]).mean()),
                       "cos_flux_velocity": float(_flux_alignment(P, X, ts.velocity))}
    ts = training_set(c, "velocity")
    X, V = ts.X, ts.velocity
    axis = X[ts.p1].mean(0) - X[ts.p0].mean(0)
    cv = (V @ axis) / np.maximum(np.linalg.norm(V, axis=1) * np.linalg.norm(axis), 1e-300)
    lt = c["latent"][ts.idx]
    a, b = lt[ts.p0], lt[ts.p1]
    out["velocity"].update({"cos_velocity_axis_source": float(cv[ts.p0].mean()),
                            "cos_velocity_axis_target": float(cv[ts.p1].mean()),
                            "frac_velocity_forward_source": float((cv[ts.p0] > 0).mean()),
                            "latent_time_auroc_target_vs_source": float(np.mean(b[:, None] > a[None, :])
                                                                        + 0.5 * np.mean(b[:, None] == a[None, :]))})
    return out


def evaluate(c: dict, traj: np.ndarray) -> dict:
    src, held, tgt = SPLITS[c["split"]]
    n = n_eval(c)
    out = {}
    for s in held:
        H = c["X"][c["stage"] == s]
        t = float(c["t"][c["stage"] == s][0])
        gen = traj_slice(traj, t)
        ok = np.isfinite(gen).all(1)
        w = [wasserstein(gen[ok], H, p=2, max_n=n, seed=k) for k in range(EVAL_REPS)]
        curve = {}
        for u in CURVE:
            g = traj_slice(traj, u)
            curve[str(u)] = wasserstein(g[np.isfinite(g).all(1)], H, p=2, max_n=n, seed=0)
        names = sorted(np.unique(c["celltype"][c["stage"] == s]))
        labels = np.array([names.index(x) for x in c["celltype"][c["stage"] == s]])
        nn = cKDTree(H).query(gen[ok], k=KNN)[1]
        votes = np.stack([(labels[nn] == j).sum(1) for j in range(len(names))], axis=1)
        gtype = np.bincount(votes.argmax(1), minlength=len(names)) / len(nn)
        otype = np.bincount(labels, minlength=len(names)) / len(labels)
        out[s] = {"t": t, "W2": float(np.mean(w)), "W2_reps": w, "curve": curve,
                  "celltype_tv": float(0.5 * np.abs(gtype - otype).sum()),
                  "celltype_generated": dict(zip(names, gtype.tolist())),
                  "celltype_observed": dict(zip(names, otype.tolist())), "nonfinite": float(1 - ok.mean())}
    end = traj_slice(traj, 1.0)
    T = c["X"][np.isin(c["stage"], tgt)]
    w_end = [wasserstein(end[np.isfinite(end).all(1)], T, p=2, max_n=n, seed=k) for k in range(EVAL_REPS)]
    return {"held": out, "W2_end": float(np.mean(w_end)), "W2_end_reps": w_end, "n_eval": n}


def evaluate_window(c: dict, traj: np.ndarray) -> dict:
    held = SPLITS[c["split"]][1]
    n = n_eval(c)
    H = c["X"][np.isin(c["stage"], held)]
    lo, hi = window_of(held)
    pool = np.concatenate([traj_slice(traj, u) for u in np.linspace(lo, hi, WINDOW_STEPS)])
    half = traj_slice(traj, 0.5)
    pool, half = pool[np.isfinite(pool).all(1)], half[np.isfinite(half).all(1)]
    w_pool = [wasserstein(pool, H, p=2, max_n=n, seed=k) for k in range(EVAL_REPS)]
    w_half = [wasserstein(half, H, p=2, max_n=n, seed=k) for k in range(EVAL_REPS)]
    return {"window": [lo, hi], "W2_window": float(np.mean(w_pool)), "W2_window_reps": w_pool,
            "W2_half": float(np.mean(w_half)), "W2_half_reps": w_half, "n_eval": n}


def run_unit(split: str, d: int, unit: str, seed: int, cfg: Run, root: str, force: bool = False,
             window: bool = False) -> None:
    path_of = window_path if window else rec_path
    if not force and all(os.path.exists(path_of(root, split, d, m, seed)) for m in larry.unit_methods(unit)):
        print(f"[ery_stage] skip {split} d{d} {unit} s{seed}")
        return
    c = cloud(split, d)
    pts = larry.transfer_points(d, SOURCES)
    if unit == "ffm_pancreas":
        pts["ffm"] = dict(pts["sources"]["pancreas"]["ffm"])
    kernel = larry.UNIT_KERNEL.get(unit, "velocity")
    ts = training_set(c, kernel)
    if unit == "curly_unit":
        out = train_curly_unit(ts, pts, seed, cfg)
    else:
        out = larry.train_unit(unit, ts, pts, seed, cfg)
    for method, (traj, extra) in out.items():
        rec = {"split": split, "dim": d, "method": method, "seed": seed, "kernel": kernel,
               "n_source": int(ts.p0.sum()), "n_target": int(ts.p1.sum()),
               **(evaluate_window(c, traj) if window else evaluate(c, traj)), **extra}
        path = path_of(root, split, d, method, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(path + ".tmp", path)
        msg = (f"window {rec['W2_window']:.4f}  t=1/2 {rec['W2_half']:.4f}" if window else
               "  ".join(f"{s} {v['W2']:.4f}" for s, v in rec["held"].items()) + f"  end {rec['W2_end']:.4f}")
        print(f"[ery_stage] {split} d{d} {method:9s} s{seed}  {msg}  ({rec['seconds']:.0f}s)", flush=True)


def tasks() -> list[tuple[str, int, str, int]]:
    return ([(s, d, u, k) for s in SPLITS for d in DIMS for u in UNITS for k in SEEDS]
            + [(s, d, "curly_unit", k) for s in SPLITS for d in DIMS for k in SEEDS])


def window_tasks() -> list[tuple[str, int, str, int]]:
    return [(s, d, u, k) for s in ("long", "longer") for d in DIMS for u in (*UNITS, "curly_unit") for k in SEEDS]


def _paired(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for b in ("ffm", "ffm_expr"):
        base = df[df["method"] == LABELS[b]].set_index(["split", "d", "held", "seed"])["W2"]
        for m in ("cfm", "mfm_land", "curly_unit"):
            o = df[df["method"] == LABELS[m]].set_index(["split", "d", "held", "seed"])["W2"]
            diff = (base - o).dropna()
            for k, g in diff.groupby(level=[0, 1, 2]):
                out.append({"split": k[0], "d": k[1], "held": k[2], "FFM": LABELS[b], "vs": LABELS[m],
                            "FFM minus other": round(float(g.mean()), 3),
                            "FFM lower (seeds)": f"{int((g < 0).sum())}/{len(g)}"})
    return pd.DataFrame(out).set_index(["split", "d", "held", "FFM", "vs"])


def frames(root: str | None = None, smoke: bool = False) -> dict:
    root = root_dir(smoke) if root is None else root
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "records", "*.json")))]
    refs = json.load(open(os.path.join(root, "references.json")))
    pm = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    rows = [{"split": r["split"], "d": r["dim"], "held": s, "method": LABELS[r["method"]], "seed": r["seed"],
             "W2": v["W2"], "celltype_tv": v["celltype_tv"], "W2_end": r["W2_end"],
             "best_t": float(min(v["curve"], key=v["curve"].get))}
            for r in recs for s, v in r["held"].items()]
    df = pd.DataFrame(rows)
    key = ["split", "d", "held", "method"]
    summary = df.groupby(key)[["W2", "W2_end", "celltype_tv"]].agg(pm)
    summary["best t"] = df.groupby(key)["best_t"].median()
    summary["n"] = df.groupby(key).size()
    ref = pd.DataFrame([{"split": sp_, "d": int(d), "held": s, "t": round(v["t"], 3), "n cells": v["n"],
                         "n eval": v["n_eval"],
                         **{k: f"{v[k]['mean']:.3f} ± {v[k]['sd']:.3f}" for k in ("floor", "copy_source", "copy_target")}}
                        for sp_, kv in refs.items() for d, hv in kv.items() for s, v in hv.items()]).set_index(["split", "d", "held"])
    loo = df[df["split"].isin(LOO)]
    loo_tab = loo.groupby(["d", "method", "held"])["W2"].agg(pm).unstack("held")
    loo_tab["average over held-out stages"] = loo.groupby(["d", "method"])["W2"].agg(pm)
    wrecs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "window", "*.json")))]
    wdf = pd.DataFrame([{"split": r["split"], "d": r["dim"], "held": h, "method": LABELS[r["method"]],
                         "seed": r["seed"], "W2": r[k]}
                        for r in wrecs for h, k in (("t = 1/2", "W2_half"), ("pooled window", "W2_window"))])
    win = wdf.groupby(["split", "d", "method", "held"])["W2"].agg(pm).unstack("held")
    diag = json.load(open(os.path.join(root, "diagnostics.json")))
    diag = pd.DataFrame({(sp_, int(d), k): v for sp_, dv in diag.items() for d, kv in dv.items()
                         for k, v in kv.items()}).T
    info = pd.DataFrame({s: json.load(open(cache_path(s).replace(".npz", ".json"))) for s in SPLITS
                         if os.path.exists(cache_path(s).replace(".npz", ".json"))}).T
    points = pd.DataFrame({d: {k: v for k, v in larry.transfer_points(d, SOURCES).items() if k != "sources"}
                           for d in DIMS}).T
    return {"summary": summary, "references": ref, "paired": _paired(df), "diagnostics": diag,
            "counts": info[["counts", "n_genes", "n_velocity_genes", "projection_check"]], "points": points,
            "loo": loo_tab, "window": win, "window_paired": _paired(wdf)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=NAME)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--force", action="store_true")
    r.add_argument("--window", action="store_true")
    sub.add_parser("prep")
    sub.add_parser("count")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    cfg = Run.from_env().but(ot_max_pts=larry.OT_MAX_PTS)
    root = root_dir(cfg.smoke)
    if a.cmd == "count":
        print(len(tasks()), len(window_tasks()))
    elif a.cmd == "prep":
        os.makedirs(root, exist_ok=True)
        refs, diag = {}, {}
        for s in SPLITS:
            if not os.path.exists(cache_path(s)):
                continue
            refs[s] = {d: references(cloud(s, d)) for d in DIMS}
            diag[s] = {d: diagnostics(cloud(s, d), root) for d in DIMS}
        for name, obj in (("references", refs), ("diagnostics", diag)):
            with open(os.path.join(root, f"{name}.json"), "w") as fh:
                json.dump(obj, fh, indent=1)
        print(json.dumps(refs, indent=1))
        print(json.dumps(diag, indent=1))
    elif a.cmd == "run":
        split, d, unit, seed = (window_tasks() if a.window else tasks())[a.index]
        run_unit(split, d, unit, seed, cfg, root, a.force, a.window)
    else:
        for k, v in frames(smoke=cfg.smoke).items():
            print(v.to_string())


if __name__ == "__main__":
    main()
