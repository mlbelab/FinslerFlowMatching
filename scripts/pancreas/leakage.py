from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ffm.core.paths import data_dir, experiment_root
from ffm.core.scvelo_data import _rescale01
from ffm.core.transition import P_KNN_K, P_SELF_WEIGHT
from ffm.method import FWFactory, Run, cuda_warm_up, moments_from, sigma_for

from . import datasets
from .ambient import MIN_SHARED_COUNTS, N_TOP_GENES, PCA_MIN_ABS_CORR, RAW
from .datasets import (N_PCS, NUM_TIMES, VEL_WEIGHT, VELO_MODE, build_training_set,
                       load_cloud)
from .noise import tuned_point
from .train import rec_path, stage_test, state_path

NAME = "pancreas_leakage"
SHIPPED = "original"
CACHES = ("allcells", "trainonly")
EXTRA_CACHES = ("band5", "band35", "fate")
TREES = (SHIPPED,) + CACHES
VELO_ARMS = ("ffm", "pathb", "curly")
REF_ARMS = ("cfm", "mfm_land")
ROW_ORDER = VELO_ARMS + REF_ARMS
LABELS = {"ffm": "FFM det", "pathb": "FFM noise", "curly": "Curly-FM", "cfm": "OT-CFM",
          "mfm_land": "MFM/LAND"}
DIMS = (2, 10, 20, 50)
SPACES = {2: "UMAP (z-scored)", 10: "PCA 10", 20: "PCA 20", 50: "PCA 50"}
SEEDS = (0, 1, 2, 3, 4)
JUDGE_SEED = SEEDS[0]
BASE_N_STEPS = 102
N_NEIGHBORS = 30
METRICS = {"mid W2": ("intermediate", "W2"), "end W2": ("endpoint", "W2")}
PAIRS = (("allcells", SHIPPED), ("trainonly", SHIPPED), ("trainonly", "allcells"))
MAIN_PAIR = ("trainonly", SHIPPED)
CACHE_FILE = f"pancreas_d{N_PCS}_vw{VEL_WEIGHT}_{VELO_MODE}.npz"
DIAG_FILE = "diagnostics.json"
EDGE_FRAC = 0.2
VERSIONS = ("scvelo", "scanpy", "anndata", "numpy", "scipy", "scikit-learn", "umap-learn",
            "numba", "pandas", "matplotlib")


def root(smoke: bool = False) -> str:
    return experiment_root(NAME, smoke)


def tree_root(tree: str, smoke: bool = False) -> str:
    return os.path.join(root(smoke), tree)


def cache_dir(tree: str) -> str:
    return data_dir("pancreas_cache") if tree == SHIPPED else os.path.join(
        data_dir("pancreas_cache_refit"), tree)


def arms_of(tree: str) -> tuple:
    return ROW_ORDER if tree in (SHIPPED, "trainonly") else VELO_ARMS


def pair_key(a: str, b: str) -> str:
    return f"{a} - {b}"


def tuned_points() -> dict:
    return {d: tuned_point(d) for d in DIMS}


def tuned_mismatches() -> list:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")
    with open(path) as fh:
        js = {int(k): v for k, v in json.load(fh).items()}
    nb = tuned_points()
    return [(d, arm, nb[d].get(arm), js.get(d, {}).get(arm)) for d in DIMS
            for arm in sorted(set(nb[d]) | set(js.get(d, {})))
            if nb[d].get(arm) != js.get(d, {}).get(arm)]


def problem(cache: str, dims, cfg, tuned: dict) -> dict:
    datasets._FULL_P_CACHE.clear()
    clouds = {d: load_cloud(d, cache) for d in dims}
    tsets = {d: build_training_set(clouds[d]) for d in dims}
    tens, metric, sigma = {}, {}, {}
    for d in dims:
        ts = tsets[d]
        X_t = cfg.tensor(np.asarray(ts.X, dtype=np.float64))
        tens[d] = (X_t[torch.as_tensor(ts.p0, device=cfg.device)],
                   X_t[torch.as_tensor(ts.p1, device=cfg.device)])
    for d in dims:
        X = np.asarray(tsets[d].X, dtype=np.float64)
        b0_pts, D_pts = moments_from(tsets[d].P, X)
        pt = tuned[d]["ffm"]
        metric[d] = FWFactory(X, D_pts, b0_pts, cfg, rho_mult=pt["rho_mult"],
                              lam_mult=pt["lam_mult"]).base
        sigma[d] = sigma_for(metric[d], tuned[d]["pathb"]["width_mult"])
    return {"clouds": clouds, "tsets": tsets, "tens": tens, "metric": metric, "sigma": sigma}


def device_path(out: str, dim: int, arms, seeds) -> str:
    return os.path.join(out, "device",
                        f"d{dim}_{'-'.join(arms)}_s{'-'.join(map(str, seeds))}.json")


def run(tree: str, dim: int, cfg=None, force: bool = False, arms=None,
        seed: int | None = None) -> None:
    cfg = Run.from_env() if cfg is None else cfg
    arms = arms_of(tree) if arms is None else tuple(arms)
    assert set(arms) <= set(arms_of(tree)), f"{arms} not all in {arms_of(tree)}"
    out = tree_root(tree, cfg.smoke)
    if seed is None:
        seeds = cfg.seeds if cfg.smoke else SEEDS
    else:
        seeds = (JUDGE_SEED,) if seed == JUDGE_SEED else (JUDGE_SEED, seed)
    if len(seeds) > 1 and seed is not None:
        gone = [a for a in arms if not os.path.exists(rec_path(out, dim, a, JUDGE_SEED))]
        assert not gone and os.path.exists(state_path(out, dim, JUDGE_SEED)), (
            f"{tree} d{dim}: seed {JUDGE_SEED} incomplete {gone}; run --seed {JUDGE_SEED}")
    tuned = tuned_points()
    new = [f"d{dim}_{a}_s{s}" for a in arms for s in seeds
           if force or not os.path.exists(rec_path(out, dim, a, s))]
    prob = problem(cache_dir(tree), (dim,), cfg, tuned)
    cuda_warm_up(prob["metric"][dim], *prob["tens"][dim], cfg)
    stage_test(out, (dim,), tuned, seeds, cfg, **prob, n_steps=BASE_N_STEPS, force=force,
               arms=arms)
    if new:
        path = device_path(out, dim, arms, seeds)
        prior = []
        if os.path.exists(path):
            with open(path) as fh:
                prior = json.load(fh)
        dev = (torch.cuda.get_device_name(cfg.device) if cfg.device.type == "cuda"
               else "cpu")
        prior.append({"host": socket.gethostname(), "device": dev,
                      "torch": torch.__version__, "records": new})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(prior, fh, indent=1)


def _preprocess(raw: str):
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 1
    sc.settings.verbosity = 1
    adata = sc.read_h5ad(raw)
    umap = np.asarray(adata.obsm["X_umap"], dtype=np.float64).copy()
    clusters = np.asarray(adata.obs["clusters"].astype(str))
    scv.pp.filter_genes(adata, min_shared_counts=MIN_SHARED_COUNTS)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.pca(adata, n_comps=N_PCS)
    return adata, umap, clusters


def _fit(adata, cells: np.ndarray, latent: bool) -> dict:
    import scvelo as scv

    n_jobs = len(os.sched_getaffinity(0))
    sub = adata[cells].copy()
    scv.pp.moments(sub, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS)
    scv.tl.recover_dynamics(sub, n_jobs=n_jobs, show_progress_bar=False)
    scv.tl.velocity(sub, mode="dynamical")
    scv.tl.velocity_graph(sub, n_jobs=n_jobs, show_progress_bar=False)
    scv.tl.velocity_embedding(sub, basis="umap")
    out = {"vgene": np.nan_to_num(np.asarray(sub.layers["velocity"]), nan=0.0),
           "velocity_umap": np.asarray(sub.obsm["velocity_umap"], dtype=np.float64).copy(),
           "info": {"n_cells": int(sub.n_obs),
                    "n_velocity_genes": int(np.sum(sub.var["velocity_genes"])),
                    "n_fit_genes": int(np.isfinite(np.asarray(sub.var["fit_alpha"])).sum()),
                    "n_neighbors": int(sub.uns["neighbors"]["params"]["n_neighbors"])}}
    if latent:
        try:
            scv.tl.latent_time(sub)
            out["latent_time"] = _rescale01(np.asarray(sub.obs["latent_time"],
                                                       dtype=np.float64))
        except Exception as e:
            out["info"]["latent_time_error"] = repr(e)
    return out


def _align(X_new: np.ndarray, X_ref: np.ndarray) -> tuple[np.ndarray, dict]:
    corr = np.array([np.corrcoef(X_new[:, j], X_ref[:, j])[0, 1]
                     for j in range(X_ref.shape[1])])
    A_new, A_ref = X_new - X_new.mean(axis=0), X_ref - X_ref.mean(axis=0)
    if np.abs(corr).min() >= PCA_MIN_ABS_CORR:
        A, how = np.diag(np.sign(corr)), "sign"
    else:
        A, how = np.linalg.lstsq(A_new, A_ref, rcond=None)[0], "lstsq"
    resid = np.linalg.norm(A_ref - A_new @ A) / np.linalg.norm(A_ref)
    off = np.linalg.norm(A - np.diag(np.diag(A))) / np.linalg.norm(A)
    return A, {"how": how, "abs_corr": np.abs(corr).tolist(),
               "min_abs_corr": float(np.abs(corr).min()),
               "n_sign_flips": int((corr < 0).sum()), "rel_resid": float(resid),
               "offdiag_frac": float(off)}


def _latent_drift(lt: np.ndarray, cloud) -> dict:
    edges = np.quantile(lt, np.linspace(0, 1, NUM_TIMES + 1))
    bins = np.digitize(lt, edges[1:-1], right=False)
    ref = cloud.latent_time
    return {"pearson": float(np.corrcoef(lt, ref)[0, 1]),
            "spearman": float(pd.Series(lt).corr(pd.Series(ref), method="spearman")),
            "bin_agreement": float(np.mean(bins == cloud.bin_id)),
            "withheld_recovered": float(np.mean(bins[cloud.target] == cloud.bin_id[cloud.target]))}


def _versions() -> dict:
    import importlib.metadata as md

    return {p: md.version(p) for p in VERSIONS}


def _uniform_like(P: sp.csr_matrix) -> sp.csr_matrix:
    C = P.tocoo()
    vals = np.where(C.row == C.col, P_SELF_WEIGHT, (1.0 - P_SELF_WEIGHT) / P_KNN_K)
    U = sp.csr_matrix((vals, (C.row, C.col)), shape=P.shape)
    assert np.allclose(np.asarray(U.sum(axis=1)).ravel(), 1.0)
    return U


def _tv(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    return 0.5 * np.asarray(abs(A - B).sum(axis=1)).ravel()


def _groups(cloud, shown: np.ndarray) -> dict:
    lt, b = cloud.latent_time[shown], cloud.bin_id[shown]
    lo, hi = b == datasets.TRAIN_BINS[0], b == datasets.TRAIN_BINS[1]
    edge = ((lo & (lt >= np.quantile(lt[lo], 1.0 - EDGE_FRAC)))
            | (hi & (lt <= np.quantile(lt[hi], EDGE_FRAC))))
    return {"bin0": lo, "bin2": hi, "gap_edge": edge, "interior": ~edge}


def _spaces() -> dict:
    out = {}
    for d in DIMS:
        clouds = {t: load_cloud(d, cache_dir(t)) for t in TREES}
        shown = clouds[SHIPPED].shown
        assert all(np.array_equal(c.shown, shown) for c in clouds.values())
        assert all(np.array_equal(c.X, clouds[SHIPPED].X) for c in clouds.values())
        ts = {t: build_training_set(c) for t, c in clouds.items()}
        P = {t: s.P.tocsr() for t, s in ts.items()}
        rec = {"cos": {}, "tv": {}, "P": {}}
        groups = _groups(clouds[SHIPPED], shown)
        for a, b in PAIRS:
            Va, Vb = clouds[a].velocity[shown], clouds[b].velocity[shown]
            na, nb = np.linalg.norm(Va, axis=1), np.linalg.norm(Vb, axis=1)
            cos = np.sum(Va * Vb, axis=1) / np.maximum(na * nb, 1e-300)
            q25, q50, q75 = np.quantile(cos, [0.25, 0.5, 0.75])
            tv = _tv(P[a], P[b])
            rec["cos"][pair_key(a, b)] = {
                "median": float(q50), "q25": float(q25), "q75": float(q75),
                "mean": float(cos.mean()), "frac_neg": float(np.mean(cos < 0)),
                "norm_ratio_median": float(np.median(na / np.maximum(nb, 1e-300))),
                **{f"median_{g}": float(np.median(cos[m])) for g, m in groups.items()},
                **{f"frac_neg_{g}": float(np.mean(cos[m] < 0)) for g, m in groups.items()}}
            rec["tv"][pair_key(a, b)] = {"mean": float(tv.mean()),
                                         "median": float(np.median(tv)),
                                         "q90": float(np.quantile(tv, 0.9)),
                                         **{f"mean_{g}": float(tv[m].mean())
                                            for g, m in groups.items()}}
        for t in TREES:
            rec["P"][t] = {"asymmetry": ts[t].diag["P_asymmetry"],
                           "cos_J_velocity": ts[t].diag["cos_J_velocity"],
                           "tv_vs_uniform_knn": float(_tv(P[t], _uniform_like(P[t])).mean()),
                           "velocity_norm_median_shown": float(np.median(
                               np.linalg.norm(clouds[t].velocity[shown], axis=1)))}
        out[str(d)] = rec
    return out


def fit_cells(name: str, cloud) -> np.ndarray:
    if name == "allcells":
        return np.arange(len(cloud.X))
    if name == "trainonly":
        return cloud.shown
    if name == "fate":
        return np.flatnonzero(np.isin(cloud.bin_id, (datasets.HOLDOUT_BIN, datasets.TRAIN_BINS[1])))
    from .band import band_split
    return np.union1d(cloud.shown, band_split(cloud, int(name[4:]) / 100.0)[0])


def build(raw: str = RAW, force: bool = False, names=CACHES) -> dict:
    path = os.path.join(root(), DIAG_FILE)
    todo = [c for c in names if force or not os.path.exists(os.path.join(cache_dir(c), CACHE_FILE))]
    prior = {}
    if os.path.exists(path):
        with open(path) as fh:
            prior = json.load(fh)
    if not todo and prior:
        return prior
    with np.load(os.path.join(cache_dir(SHIPPED), CACHE_FILE), allow_pickle=True) as z:
        shipped = {k: z[k] for k in z.files}
    adata, umap, clusters = _preprocess(raw)
    order = {"n_cells": int(len(clusters)), "n_cells_cache": int(len(shipped["clusters"])),
             "clusters_equal": bool(np.array_equal(clusters,
                                                   shipped["clusters"].astype(str))),
             "umap_max_abs_diff": float(np.abs(umap - shipped["umap"]).max())}
    assert order["clusters_equal"] and order["umap_max_abs_diff"] == 0.0, order
    X_ref = np.asarray(shipped["X"], dtype=np.float64)
    A, pca = _align(np.asarray(adata.obsm["X_pca"][:, :N_PCS], dtype=np.float64), X_ref)
    loadings = np.asarray(adata.varm["PCs"], dtype=np.float64)[:, :N_PCS] @ A
    cloud = load_cloud(N_PCS, cache_dir(SHIPPED))
    assert not np.isin(cloud.bin_id[fit_cells("trainonly", cloud)], datasets.HOLDOUT_BIN).any()
    fits = dict(prior.get("fits", {}))
    for name in todo:
        cells = fit_cells(name, cloud)
        fit = _fit(adata, cells, latent=name == "allcells")
        vpca = np.array(shipped["velocity_pca"], dtype=np.float64)
        vumap = np.array(shipped["velocity_umap"], dtype=np.float64)
        vpca[cells] = fit["vgene"] @ loadings
        vumap[cells] = fit["velocity_umap"]
        os.makedirs(cache_dir(name), exist_ok=True)
        np.savez_compressed(os.path.join(cache_dir(name), CACHE_FILE),
                            **{**shipped, "velocity_pca": vpca, "velocity_umap": vumap})
        fits[name] = dict(fit["info"])
        fits[name]["withheld_velocity"] = ("refit" if name == "allcells"
                                           else "shipped (evaluation-only readers)")
        if "latent_time" in fit:
            fits[name]["latent_time_vs_cache"] = _latent_drift(fit["latent_time"], cloud)
    with open(raw, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    diag = {**prior, "raw": {"file": os.path.basename(raw), "md5": md5}, "order": order, "pca": pca,
            "fits": fits, "n_shown": int(len(cloud.shown)), "versions": _versions(),
            "spaces": _spaces()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(diag, fh, indent=1, sort_keys=True)
    return diag


def refresh_spaces() -> dict:
    path = os.path.join(root(), DIAG_FILE)
    with open(path) as fh:
        diag = json.load(fh)
    diag["spaces"] = _spaces()
    with open(path, "w") as fh:
        json.dump(diag, fh, indent=1, sort_keys=True)
    return diag


def records(smoke: bool = False) -> pd.DataFrame:
    rows = []
    for tree in TREES:
        for dim in DIMS:
            for arm in arms_of(tree):
                for seed in SEEDS:
                    p = rec_path(tree_root(tree, smoke), dim, arm, seed)
                    if not os.path.exists(p):
                        continue
                    with open(p) as fh:
                        m = json.load(fh)["metrics"]
                    rows.append({"tree": tree, "dim": dim, "arm": arm, "seed": seed,
                                 **{k: m[a][b] for k, (a, b) in METRICS.items()}})
    return pd.DataFrame(rows, columns=["tree", "dim", "arm", "seed", *METRICS])


def _pm(v) -> str:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return f"{v.mean():.4f} ± {v.std(ddof=0):.4f}" if len(v) else ""


def _cell(frame: pd.DataFrame, key: tuple, col) -> np.ndarray:
    if col not in frame.columns or key not in frame.index:
        return np.zeros(0)
    v = frame.loc[key, col].to_numpy(dtype=np.float64)
    return v[np.isfinite(v)]


def frames(smoke: bool = False) -> dict:
    df = records(smoke)
    wide = df.set_index(["dim", "arm", "seed", "tree"])[list(METRICS)].unstack("tree")
    paired = pd.DataFrame({(m, pair_key(a, b)): wide[(m, a)] - wide[(m, b)]
                           for m in METRICS for a, b in PAIRS
                           if (m, a) in wide.columns and (m, b) in wide.columns},
                          index=wide.index)
    main = pair_key(*MAIN_PAIR)
    rows, pairs, stats = {}, {}, {}
    for dim in DIMS:
        for arm in ROW_ORDER:
            key = (dim, LABELS[arm])
            row, pr, st = {}, {}, {}
            for m in METRICS:
                for t in TREES:
                    v = _cell(wide, (dim, arm), (m, t))
                    row[(m, t)] = _pm(v)
                    st[(m, t, "mean")] = float(v.mean()) if len(v) else np.nan
                    st[(m, t, "sd")] = float(v.std(ddof=0)) if len(v) else np.nan
                    st[(m, t, "n")] = len(v)
                for a, b in PAIRS:
                    dv = _cell(paired, (dim, arm), (m, pair_key(a, b)))
                    pr[(m, pair_key(a, b))] = _pm(dv)
                    pr[(m, pair_key(a, b) + " lower")] = (f"{int((dv < 0).sum())}/{len(dv)}"
                                                         if len(dv) else "")
                    st[(m, pair_key(a, b), "mean")] = float(dv.mean()) if len(dv) else np.nan
                    st[(m, pair_key(a, b), "sd")] = float(dv.std(ddof=0)) if len(dv) else np.nan
                row[(m, main)] = pr[(m, main)]
                row[(m, "lower")] = pr[(m, main + " lower")]
            rows[key], pairs[key], stats[key] = row, pr, st
    out = {"records": df, "paired": paired}
    for name, tab in (("summary", rows), ("pairs", pairs), ("numeric", stats)):
        f = pd.DataFrame.from_dict(tab, orient="index")
        f.columns = pd.MultiIndex.from_tuples(f.columns)
        f.index = pd.MultiIndex.from_tuples(f.index, names=["d", "method"])
        out[name] = f
    return out


def paper_frames(tree: str = "trainonly", smoke: bool = False) -> dict:
    rows, curves, times, floors = {}, {}, {}, {}
    for dim in DIMS:
        for arm in arms_of(tree):
            recs = [json.load(open(p)) for p in (rec_path(tree_root(tree, smoke), dim, arm, s)
                                                 for s in SEEDS) if os.path.exists(p)]
            if not recs:
                continue
            m = [r["metrics"] for r in recs]
            win = {t: [x["window"][t] for x in m] for t in m[0].get("window", {})}
            key = (dim, LABELS[arm])
            rows[key] = {"W2(1/2)": _pm([x["intermediate"]["W2"] for x in m]),
                         "W2[1/3,2/3]": _pm([np.mean(list(x["window"].values())) for x in m])
                         if win else "",
                         "W2(1)": _pm([x["endpoint"]["W2"] for x in m]),
                         "W2,F(1/2)": _pm([x["intermediate"].get("W2F", np.nan) for x in m]),
                         "n": len(m)}
            curves[key] = {t: _pm(v) for t, v in win.items()}
            times[key] = float(np.mean([r["train_seconds"] for r in recs]))
            f = m[0]["floors"]
            floors[(dim, "sampling floor")] = {"W2(1/2)": f"{f['intermediate']['W2']:.4f}",
                                               "W2(1)": f"{f['endpoint']['W2']:.4f}",
                                               "W2,F(1/2)": f"{f['intermediate'].get('W2F', np.nan):.4f}"}
    idx = lambda d: pd.MultiIndex.from_tuples(list(d), names=["d", "method"])
    table = pd.DataFrame(list(rows.values()), index=idx(rows))
    fl = pd.DataFrame(list(floors.values()), index=idx(floors))
    return {"fig3": pd.concat([table, fl]).sort_index(level=0, sort_remaining=False),
            "fig5a": pd.DataFrame(list(curves.values()), index=idx(curves)),
            "fig5b": table["W2,F(1/2)"].unstack("d"),
            "table2": pd.Series(times, name="seconds").rename_axis(["d", "method"]).unstack("d")}


def diagnostics() -> dict:
    with open(os.path.join(root(), DIAG_FILE)) as fh:
        dg = json.load(fh)
    out = {"raw": dg}
    for part, level in (("cos", "pair"), ("tv", "pair"), ("P", "tree")):
        f = pd.DataFrame({(SPACES[int(d)], k): v for d, s in dg["spaces"].items()
                          for k, v in s[part].items()}).T
        f.index.names = ["space", level]
        out["cosine" if part == "cos" else part] = f
    return out


def render(f: dict, smoke: bool = False) -> str:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    out = []
    for dim in DIMS:
        out.append(f"===== d = {dim} ({SPACES[dim]}): W2 at t = 1/2 vs the 1112 test cells "
                   f"(mid) and at t = 1 vs bin 2 (end); mean ± population sd over seeds "
                   f"{SEEDS}; trainonly - original is the per-seed paired difference")
        out.append(f["summary"].xs(dim, level="d").to_string())
    out.append("\n===== all paired differences (mean ± population sd over seeds; "
               "k/n = seeds where the first tree is lower)")
    out.append(f["pairs"].dropna(how="all").to_string())
    per_seed = f["paired"].get(("mid W2", pair_key(*MAIN_PAIR)))
    if per_seed is not None:
        out.append(f"\n===== mid W2, {pair_key(*MAIN_PAIR)}, per seed")
        out.append(per_seed.unstack("seed").round(4).to_string())
    mism = tuned_mismatches()
    out.append(f"\ntuned.json vs notebook TUNED: {'agree' if not mism else mism}")
    if not smoke and os.path.exists(os.path.join(root(), DIAG_FILE)):
        dg = diagnostics()
        out.append("\n===== velocity of the 2464 shown cells, first vs second tree: cosine "
                   "(in the chart each model reads)")
        out.append(dg["cosine"].round(4).to_string())
        out.append("\n===== rows of the training P on the shown cells: total-variation "
                   "distance")
        out.append(dg["tv"].round(4).to_string())
        out.append("\n===== training P per tree (tv_vs_uniform_knn: TV between P and a "
                   "uniform walk on the same 15-NN graph)")
        out.append(dg["P"].round(4).to_string())
        raw = dg["raw"]
        pca = {k: v for k, v in raw["pca"].items() if k != "abs_corr"}
        out.append(f"\norder {raw['order']}\npca {pca}\nfits {raw['fits']}\n"
                   f"versions {raw['versions']}")
    return "\n".join(out)


def cmd_build(args) -> None:
    dg = build(args.raw, args.force, tuple(args.names.split(",")) if args.names else CACHES)
    print(f"[leakage] caches + {DIAG_FILE} in {root()}  pca {dg['pca']['how']} "
          f"min|corr| {dg['pca']['min_abs_corr']:.6f}")


def cmd_diag(args) -> None:
    refresh_spaces()
    print(f"[leakage] spaces refreshed in {os.path.join(root(), DIAG_FILE)}")


def cmd_run(args) -> None:
    run(args.tree, args.dim, force=args.force,
        arms=tuple(args.arms.split(",")) if args.arms else None, seed=args.seed)


def cmd_report(args) -> None:
    smoke = Run.from_env().smoke
    f = frames(smoke)
    text = render(f, smoke)
    os.makedirs(root(smoke), exist_ok=True)
    with open(os.path.join(root(smoke), "report.txt"), "w") as fh:
        fh.write(text + "\n")
    for name in ("summary", "pairs", "paired", "numeric"):
        f[name].to_csv(os.path.join(root(smoke), f"{name}.csv"))
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(prog="leakage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", default=RAW)
    b.add_argument("--force", action="store_true")
    b.add_argument("--names", default="")
    b.set_defaults(fn=cmd_build)
    g = sub.add_parser("diag")
    g.set_defaults(fn=cmd_diag)
    r = sub.add_parser("run")
    r.add_argument("--tree", choices=TREES, required=True)
    r.add_argument("--dim", type=int, choices=DIMS, required=True)
    r.add_argument("--arms", default="")
    r.add_argument("--seed", type=int, choices=SEEDS, default=None)
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
