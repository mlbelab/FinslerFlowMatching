from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np

from ffm.core.paths import data_dir, experiment_root

from .datasets import DIMS, HOLDOUT_BIN, P_K, TRAIN_BINS, load_cloud

KEY = "erythroid_leakage"
RAW_URL = "https://ndownloader.figshare.com/files/27686871"
RAW = os.environ.get("ERYTHROID_RAW", os.path.join(data_dir("raw"), "erythroid_lineage.h5ad"))
SHIPPED = data_dir("erythroid_cache")
FITS = {"trainonly": ("shown", False), "allcells": ("all", False), "ownpca": ("shown", True)}
MATCHED = "matched"
VARIANTS = ("shipped", "trainonly", "allcells", "ownpca", MATCHED)
VARIANT_LABELS = {"record": "shipped record", "shipped": "shipped-velocity rerun",
                  "trainonly": "train-only velocity", "allcells": "all-cells refit",
                  "ownpca": "train-only, own-PCA kNN", MATCHED: "train-only, norm-matched"}
FIELD_LABELS = {"shipped": "shipped", "trainonly": "train-only", "allcells": "all-cells refit",
                "ownpca": "train-only own-PCA kNN"}
ARMS = ("ffm", "pathb", "curly")
REUSED = ("cfm", "mfm_land")
SWEEPABLE = ("curly",)
CURLY_K = 30
LABELS = {"ffm": "FFM det", "pathb": "FFM noise", "curly": "Curly-FM",
          "cfm": "OT-CFM", "mfm_land": "MFM"}
SEEDS = (0, 1, 2, 3, 4)
N_KEEP = max(DIMS)
MIN_PC_CORR = 0.999
END_MAX_N = 10_000
ALL_CELL_STATE = {"obsm": ("X_pca",), "varm": ("PCs",), "uns": ("pca", "neighbors"),
                  "obsp": ("distances", "connectivities")}


def root(smoke: bool = False) -> str:
    return experiment_root(KEY, smoke)


def cache_dir(variant: str) -> str:
    return SHIPPED if variant == "shipped" else os.path.join(root(), f"cache_{variant}")


def fit_dir(tag: str, smoke: bool = False) -> str:
    return os.path.join(root(smoke), f"fit_{tag}")


def _write(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def check_rows(adata) -> dict:
    z = np.load(os.path.join(SHIPPED, "erythroid_d2.npz"), allow_pickle=True)
    out = {"n_obs": int(adata.n_obs), "n_cache": int(len(z["X"]))}
    if out["n_obs"] != out["n_cache"]:
        return {**out, "ok": False}
    ct = np.asarray(adata.obs["celltype"].astype(str)).astype(str)
    st = np.asarray(adata.obs["stage"].astype(str)).astype(str)
    um = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
    x2 = (um - um.mean(axis=0)) / um.std(axis=0)
    out.update({
        "celltype_equal": bool(np.array_equal(ct, z["celltype"].astype(str))),
        "stage_equal": bool(np.array_equal(st, z["stage"].astype(str))),
        "umap_equal": bool(np.array_equal(um, z["umap"])),
        "umap_max_abs_diff": float(np.abs(um - z["umap"]).max()),
        "x_d2_max_abs_diff": float(np.abs(x2 - z["X"]).max()),
    })
    out["ok"] = out["celltype_equal"] and out["stage_equal"] and out["umap_equal"]
    return out


def _all_cell_state(adata) -> dict:
    found = {}
    for slot, keys in ALL_CELL_STATE.items():
        store = getattr(adata, slot)
        hit = [k for k in keys if k in store]
        if hit:
            found[slot] = hit
    return found


def _strip_all_cell_state(adata) -> dict:
    found = _all_cell_state(adata)
    for slot, keys in found.items():
        store = getattr(adata, slot)
        for k in keys:
            del store[k]
    return found


def verify() -> dict:
    import anndata as ad

    adata = ad.read_h5ad(RAW)
    chk = check_rows(adata)
    rec = {"raw": RAW, "url": RAW_URL, "bytes": os.path.getsize(RAW), "sha256": _sha256(RAW),
           "n_vars": int(adata.n_vars), "obs": list(adata.obs.columns),
           "var": list(adata.var.columns), "var_head": [str(v) for v in adata.var_names[:3]],
           "layers": list(adata.layers), "obsm": {k: list(np.shape(v)) for k, v in adata.obsm.items()},
           "obsp": list(adata.obsp), "uns": list(adata.uns), "all_cell_state": _all_cell_state(adata),
           "neighbors_params": dict(adata.uns.get("neighbors", {}).get("params", {})),
           "rows": chk}
    if chk["ok"]:
        cloud = load_cloud(2, SHIPPED)
        st = np.asarray(adata.obs["stage"].astype(str)).astype(str)
        rec["stage_by_bin"] = {str(b): {str(k): int(n) for k, n in zip(
            *np.unique(st[cloud.bin_id == b], return_counts=True))} for b in range(3)}
    _write(rec, os.path.join(root(), "raw_check.json"))
    print(f"[leakage] raw {rec['bytes']} B sha256 {rec['sha256'][:12]} rows {chk}")
    assert chk["ok"], f"raw h5ad does not match the shipped cache by row order: {chk}"
    return rec


def pca_step() -> None:
    from .preprocess import N_PCS

    import scanpy as sc
    import scvelo as scv
    import unitvelo as utv

    out = os.path.join(root(), "pca_allcells.npz")
    if os.path.exists(out):
        return
    adata = sc.read_h5ad(RAW)
    chk = check_rows(adata)
    assert chk["ok"], chk
    adata.var["raw_index"] = np.arange(adata.n_vars)
    cfg = utv.config.Configuration()
    scv.pp.filter_and_normalize(adata, min_shared_counts=cfg.MIN_SHARED_COUNTS,
                                n_top_genes=cfg.N_TOP_GENES)
    sc.pp.pca(adata, n_comps=N_PCS)
    Xp = np.asarray(adata.obsm["X_pca"], dtype=np.float64)[:, :N_KEEP]
    PCs = np.asarray(adata.varm["PCs"], dtype=np.float64)[:, :N_KEEP]
    ref = load_cloud(N_KEEP, SHIPPED).X
    assert np.array_equal(load_cloud(20, SHIPPED).X, ref[:, :20])
    corr = np.array([np.corrcoef(Xp[:, j], ref[:, j])[0, 1] for j in range(N_KEEP)])
    sign = np.where(corr < 0, -1.0, 1.0)
    diff = np.abs(Xp * sign - ref).max(axis=0)
    np.savez_compressed(out, loadings=PCs * sign, raw_index=adata.var["raw_index"].to_numpy(),
                        corr=corr, max_abs_diff=diff, n_genes=int(adata.n_vars))
    print(f"[leakage] pca: {adata.n_vars} genes, min |corr| {np.abs(corr).min():.6f}, "
          f"max |dX| {diff.max():.3g}")
    assert (np.abs(corr) >= MIN_PC_CORR).all(), np.abs(corr)


def refit(tag: str, gpu: int = -1, smoke: bool = False) -> None:
    from .preprocess import LABEL, _patch_scvelo, finish_from_checkpoint

    import scanpy as sc
    import scvelo as scv
    import unitvelo as utv

    work = fit_dir(tag, smoke)
    out = os.path.join(work, "fit.npz")
    if os.path.exists(out):
        return
    os.makedirs(work, exist_ok=True)
    cloud = load_cloud(2, SHIPPED)
    cells, strip = FITS[tag]
    rows = cloud.shown if cells == "shown" else np.arange(len(cloud.X))
    ckpt = os.path.join(work, "post_fit_adata.h5ad")
    note = os.path.join(work, "stripped.json")
    os.chdir(work)
    _patch_scvelo(ckpt)
    if os.path.exists(ckpt):
        adata = finish_from_checkpoint(ckpt)
    else:
        adata = sc.read_h5ad(RAW)
        chk = check_rows(adata)
        assert chk["ok"], chk
        adata = adata[rows].copy()
        _write(_strip_all_cell_state(adata) if strip else {}, note)
        adata.var["raw_index"] = np.arange(adata.n_vars)
        cfg = utv.config.Configuration()
        cfg.R2_ADJUST = True
        cfg.IROOT = None
        cfg.FIT_OPTION = "1"
        cfg.GPU = gpu
        cfg.AGENES_R2 = 1
        if smoke:
            cfg.MAX_ITER = 100
        adata = utv.run_model(adata, LABEL, config_file=cfg)
        if "velocity_umap" not in adata.obsm:
            scv.tl.velocity_graph(adata)
            scv.tl.velocity_embedding(adata, basis="umap")
    um = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
    assert np.array_equal(um, cloud.umap[rows]), "refit lost the row order"
    np.savez_compressed(
        out, rows=rows, raw_index=np.asarray(adata.var["raw_index"], dtype=np.int64),
        velocity=np.asarray(adata.layers["velocity"], dtype=np.float32),
        velocity_genes=np.asarray(adata.var["velocity_genes"], dtype=bool),
        velocity_umap=np.asarray(adata.obsm["velocity_umap"], dtype=np.float64),
        latent_time=np.asarray(adata.obs["latent_time"], dtype=np.float64),
        n_neighbors=int(adata.uns["neighbors"]["params"]["n_neighbors"]),
        stripped=open(note).read(), smoke=smoke)
    for p in (ckpt, os.path.join(work, "res")):
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    print(f"[leakage] refit {tag}: {adata.n_obs} cells, {adata.n_vars} genes, "
          f"{int(adata.var['velocity_genes'].sum())} velocity genes -> {out}")


def project(fit, pca) -> np.ndarray:
    pos = {int(g): i for i, g in enumerate(pca["raw_index"])}
    genes = np.asarray(fit["raw_index"])
    src = np.array([j for j, g in enumerate(genes) if int(g) in pos], dtype=np.int64)
    dst = np.array([pos[int(genes[j])] for j in src], dtype=np.int64)
    vel = np.asarray(fit["velocity"], dtype=np.float64)[:, src]
    vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    return vel @ np.asarray(pca["loadings"], dtype=np.float64)[dst]


def build(tag: str) -> None:
    pca = np.load(os.path.join(root(), "pca_allcells.npz"))
    assert (np.abs(pca["corr"]) >= MIN_PC_CORR).all(), np.abs(pca["corr"])
    fit = np.load(os.path.join(fit_dir(tag), "fit.npz"))
    assert not bool(fit["smoke"]), "a smoke fit cannot become a cache"
    rows = np.asarray(fit["rows"])
    keep = np.isin(rows, load_cloud(2, SHIPPED).shown)
    Vg = project(fit, pca)
    out = cache_dir(tag)
    os.makedirs(out, exist_ok=True)
    for d in DIMS:
        z = np.load(os.path.join(SHIPPED, f"erythroid_d{d}.npz"), allow_pickle=True)
        V = np.array(z["velocity"], dtype=np.float64)
        new = np.asarray(fit["velocity_umap"], dtype=np.float64) if d == 2 else Vg[:, :d]
        V[rows[keep]] = new[keep]
        assert np.isfinite(V).all(), f"d={d}: non-finite refit velocity"
        np.savez_compressed(os.path.join(out, f"erythroid_d{d}.npz"), velocity=V,
                            **{k: z[k] for k in z.files if k != "velocity"})
    print(f"[leakage] cache {tag}: {int(keep.sum())} shown cells replaced -> {out}")


def build_matched(src: str = "trainonly") -> dict:
    out = cache_dir(MATCHED)
    os.makedirs(out, exist_ok=True)
    scale = {}
    for d in DIMS:
        a, b = load_cloud(d, SHIPPED), load_cloud(d, cache_dir(src))
        idx = a.shown
        scale[d] = float(np.median(np.linalg.norm(a.velocity[idx], axis=1))
                         / np.median(np.linalg.norm(b.velocity[idx], axis=1)))
        z = np.load(os.path.join(cache_dir(src), f"erythroid_d{d}.npz"), allow_pickle=True)
        V = np.array(z["velocity"], dtype=np.float64)
        V[idx] *= scale[d]
        np.savez_compressed(os.path.join(out, f"erythroid_d{d}.npz"), velocity=V,
                            **{k: z[k] for k in z.files if k != "velocity"})
    _write({"source": src, "scale": scale}, os.path.join(out, "scale.json"))
    print(f"[leakage] cache {MATCHED}: {src} x {scale} -> {out}")
    return scale


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(n, 1e-300)


def _near_withheld(cloud, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    idx = cloud.shown
    _, nn = cKDTree(cloud.X).query(cloud.X[idx], k=k + 1)
    return (cloud.bin_id[nn[:, 1:]] == HOLDOUT_BIN).any(axis=1)


def _tv_rows(P, Q) -> np.ndarray:
    return 0.5 * np.asarray(abs(P - Q).sum(axis=1)).ravel()


def _curly_field(X: np.ndarray, V: np.ndarray, Q: np.ndarray, k: int = CURLY_K) -> np.ndarray:
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(X).query(Q, k=k)
    h = np.maximum(dist[:, -1:], 1e-12)
    w = np.exp(-dist ** 2 / (2 * h ** 2))
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    return np.einsum("qk,qkd->qd", w, V[idx])


def _gap_field(X: np.ndarray, va: np.ndarray, vb: np.ndarray, Q: np.ndarray,
               vQ: np.ndarray) -> dict:
    ua, ub = _curly_field(X, va, Q), _curly_field(X, vb, Q)
    na, nb = np.linalg.norm(ua, axis=1), np.linalg.norm(ub, axis=1)
    return {"gap_cos_median": float(np.median(_cos(ua, ub))),
            "gap_norm_ratio_median": float(np.median(nb / np.maximum(na, 1e-300))),
            "gap_align_a": float(np.median(_cos(ua, vQ))),
            "gap_align_b": float(np.median(_cos(ub, vQ)))}


def _agreement(X: np.ndarray, va: np.ndarray, vb: np.ndarray, near: np.ndarray) -> dict:
    from ffm.core.transition import build_velocity_kernel

    from .datasets import _flux_alignment

    c = _cos(va, vb)
    na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    ok = na > 0
    Pa, Pb = build_velocity_kernel(X, va), build_velocity_kernel(X, vb)
    Pp = build_velocity_kernel(X, va[np.random.default_rng(0).permutation(len(va))])
    tv, tvp = _tv_rows(Pa, Pb), _tv_rows(Pa, Pp)
    return {"cos_median": float(np.median(c)), "cos_q25": float(np.percentile(c, 25)),
            "cos_q75": float(np.percentile(c, 75)), "cos_mean": float(c.mean()),
            "frac_cos_negative": float((c < 0).mean()),
            "norm_ratio_median": float(np.median(nb[ok] / na[ok])),
            "tv_mean": float(tv.mean()), "tv_shuffled_reference": float(tvp.mean()),
            "n_near_withheld": int(near.sum()),
            "cos_median_near": float(np.median(c[near])),
            "cos_median_far": float(np.median(c[~near])),
            "tv_mean_near": float(tv[near].mean()), "tv_mean_far": float(tv[~near].mean()),
            "cos_J_v_a": float(_flux_alignment(Pa, X, va)),
            "cos_J_v_b": float(_flux_alignment(Pb, X, vb)),
            "n_zero_b": int((nb == 0).sum()), "n_cells": int(len(X))}


def diagnostics() -> dict:
    from scipy.stats import spearmanr

    out = {"pca": {}, "fits": {}, "agreement": {}}
    pca = np.load(os.path.join(root(), "pca_allcells.npz"))
    out["pca"] = {"min_abs_corr": float(np.abs(pca["corr"]).min()),
                  "max_abs_dX": float(np.max(pca["max_abs_diff"])),
                  "n_genes": int(pca["n_genes"])}
    tags = [t for t in FITS if os.path.exists(os.path.join(cache_dir(t), "erythroid_d2.npz"))]
    clouds = {d: load_cloud(d, SHIPPED) for d in DIMS}
    pca_genes = set(int(g) for g in pca["raw_index"])
    for t in tags:
        fit = np.load(os.path.join(fit_dir(t), "fit.npz"))
        rows = np.asarray(fit["rows"])
        finite = np.isfinite(np.asarray(fit["velocity"])).all(axis=0)
        genes = np.asarray(fit["raw_index"])
        out["fits"][t] = {
            "n_cells": int(len(rows)), "n_genes": int(len(genes)),
            "n_velocity_genes": int(np.asarray(fit["velocity_genes"]).sum()),
            "n_finite_velocity_genes": int(finite.sum()),
            "pca_genes_in_fit": int(len(pca_genes & set(int(g) for g in genes))),
            "pca_genes_with_finite_velocity": int(len(pca_genes & set(int(g) for g in genes[finite]))),
            "latent_spearman_vs_cached": float(spearmanr(
                fit["latent_time"], clouds[2].latent_time[rows]).correlation),
            "n_neighbors": int(fit["n_neighbors"]), "stripped": json.loads(str(fit["stripped"]))}
    pairs = [("shipped", t) for t in tags] + [(a, b) for a, b in (
        ("allcells", "trainonly"), ("trainonly", "ownpca")) if a in tags and b in tags]
    scale_file = os.path.join(cache_dir(MATCHED), "scale.json")
    if os.path.exists(scale_file):
        with open(scale_file) as fh:
            out["matched"] = json.load(fh)
    for a, b in pairs:
        for d in DIMS:
            ca, cb = load_cloud(d, cache_dir(a)), load_cloud(d, cache_dir(b))
            assert np.array_equal(ca.X, cb.X) and np.array_equal(ca.bin_id, cb.bin_id)
            assert np.array_equal(ca.velocity[ca.target], cb.velocity[cb.target])
            idx, tgt = ca.shown, ca.target
            out["agreement"][f"{b}_vs_{a}_d{d}"] = {
                "a": a, "b": b, "dim": d,
                **_agreement(ca.X[idx], ca.velocity[idx], cb.velocity[idx],
                             _near_withheld(ca, P_K)),
                **_gap_field(ca.X[idx], ca.velocity[idx], cb.velocity[idx], ca.X[tgt],
                             ca.velocity[tgt])}
    _write(out, os.path.join(root(), "diag.json"))
    print(f"[leakage] diagnostics for {tags} -> {os.path.join(root(), 'diag.json')}")
    return out


def tuned_points() -> dict:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")) as fh:
        return {int(d): v for d, v in json.load(fh).items()}


def rec_path(variant: str, dim: int, arm: str, seed: int, smoke: bool = False) -> str:
    return os.path.join(root(smoke), "runs", variant, f"d{dim}_{arm}_s{seed}.json")


def _fresh(path: str, override: dict, inherit, smoke: bool) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        r = json.load(fh)
    return (r.get("override") == override and r.get("inherited") == inherit
            and r.get("smoke") == smoke)


def sweep_path(variant: str, dim: int, arm: str, point: dict, seed: int) -> str:
    from .train import _override_tag

    return os.path.join(root(), "sweep", variant, f"d{dim}_{arm}_{_override_tag(point)}_s{seed}.json")


def sweep_one(variant: str, dim: int, arm: str, point: dict, seed: int, cfg) -> dict:
    import torch

    from ffm.core.arms import run_arm

    from .datasets import build_training_set, split_shown
    from .train import (N_STEPS_ERYTHROID, SPLIT_SEED, VAL_FRACTION, engine_arm, hparams_for,
                        sweep_terms)

    cloud = load_cloud(dim, cache_dir(variant))
    i_train, i_val = split_shown(cloud, VAL_FRACTION, SPLIT_SEED)
    ts = build_training_set(cloud, subset=i_train)
    t0 = time.time()
    hp, _ = hparams_for(dim, arm, point, ts, cfg.device, cfg.dtype, cfg.smoke)
    res = run_arm(engine_arm(arm), ts, hp, seed, cfg.device, cfg.dtype,
                  n_steps=N_STEPS_ERYTHROID)
    return {"variant": variant, "dim": dim, "arm": arm, "seed": seed, "point": point,
            "metrics": sweep_terms(res.traj, cloud, i_val), "train_seconds": time.time() - t0,
            "smoke": bool(cfg.smoke),
            "device": (torch.cuda.get_device_name(0) if cfg.device.type == "cuda" else "cpu")}


def sweep_cells(variant: str, dim: int, arm: str, cfg, index: int | None = None) -> None:
    from .train import grid_points
    from .tune import SWEEP_SEEDS

    points = grid_points(arm, dim)
    for point in points if index is None else [points[index]]:
        for seed in SWEEP_SEEDS:
            path = sweep_path(variant, dim, arm, point, seed)
            if os.path.exists(path):
                continue
            rec = sweep_one(variant, dim, arm, point, seed, cfg)
            _write(rec, path)
            print(f"[leakage] sweep {variant} d{dim} {arm} {point} s{seed}  objective "
                  f"{rec['metrics']['objective']:.4f}  ({rec['train_seconds']:.0f}s)", flush=True)


def sweep_frame(variant: str, dim: int, arm: str):
    import pandas as pd

    d = os.path.join(root(), "sweep", variant)
    rows = []
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        with open(os.path.join(d, name)) as fh:
            r = json.load(fh)
        if r["dim"] == dim and r["arm"] == arm:
            rows.append({**r["point"], "seed": r["seed"], **{k: r["metrics"][k] for k in (
                "objective", "W2_intermediate", "W2_endpoint")}})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keys = [c for c in df.columns if c not in ("seed", "objective", "W2_intermediate",
                                               "W2_endpoint")]
    out = df.groupby(keys)[["objective", "W2_intermediate", "W2_endpoint"]].mean()
    out["n_seeds"] = df.groupby(keys)["seed"].count()
    return out


def run_one(variant: str, dim: int, arm: str, seed: int, cfg) -> dict:
    import torch

    from ffm.core.arms import run_arm
    from ffm.core.metrics import wasserstein

    from .datasets import build_training_set
    from .train import (N_STEPS_ERYTHROID, _jsonable, engine_arm, evaluate, hparams_for,
                        inherited_metric, scored_target, transport_field)

    cloud = load_cloud(dim, cache_dir(variant))
    ts = build_training_set(cloud)
    tuned = tuned_points()[dim]
    override, inherit = tuned[arm], inherited_metric(arm, tuned)
    t0 = time.time()
    hp, scales = hparams_for(dim, arm, {**(inherit or {}), **override}, ts, cfg.device,
                             cfg.dtype, cfg.smoke)
    res = run_arm(engine_arm(arm), ts, hp, seed, cfg.device, cfg.dtype,
                  n_steps=N_STEPS_ERYTHROID)
    train_s = time.time() - t0
    field, tag = transport_field(res, hp)
    m = evaluate(res.traj, field, cloud, scored_target(cloud), cfg.device, cfg.dtype, seed=seed)
    end = wasserstein(np.asarray(res.traj[-1], dtype=np.float64),
                      cloud.X[cloud.marginal(TRAIN_BINS[1])], max_n=END_MAX_N)
    keep = ("w2_true", "w2", "cos_dist", "l2", "u_snr")
    return {"variant": variant, "dim": dim, "arm": arm, "seed": seed,
            "override": override, "inherited": inherit,
            "metrics": {**{k: float(m[k]) for k in keep}, "end_w2": float(end)},
            "scored_field": tag, "hparams": hp.as_dict(), "scales": _jsonable(scales),
            "ts_diag": ts.diag, "cloud_diag": cloud.diag, "train_seconds": train_s,
            "n_steps": N_STEPS_ERYTHROID, "dtype": str(cfg.dtype), "smoke": bool(cfg.smoke),
            "device": (torch.cuda.get_device_name(0) if cfg.device.type == "cuda" else "cpu")}


def run_cells(dim: int, arm: str, seed: int, variants, cfg) -> None:
    tuned = tuned_points()[dim]
    from .train import inherited_metric

    inherit = inherited_metric(arm, tuned)
    for v in variants:
        if not os.path.exists(os.path.join(cache_dir(v), f"erythroid_d{dim}.npz")):
            continue
        path = rec_path(v, dim, arm, seed, cfg.smoke)
        if _fresh(path, tuned[arm], inherit, bool(cfg.smoke)):
            continue
        rec = run_one(v, dim, arm, seed, cfg)
        _write(rec, path)
        print(f"[leakage] {v} d{dim} {arm} s{seed}  W2 {rec['metrics']['w2_true']:.4f}  "
              f"end {rec['metrics']['end_w2']:.4f}  ({rec['train_seconds']:.0f}s "
              f"{rec['device']})", flush=True)


def reference() -> None:
    from ffm.core.metrics import wasserstein

    bench = experiment_root("erythroid_bench")
    for d in DIMS:
        cloud = load_cloud(d, SHIPPED)
        tgt = cloud.X[cloud.marginal(TRAIN_BINS[1])]
        for arm in ARMS + REUSED:
            for s in SEEDS:
                path = rec_path("record", d, arm, s)
                if os.path.exists(path):
                    continue
                with open(os.path.join(bench, "test", f"d{d}_{arm}_s{s}.json")) as fh:
                    r = json.load(fh)
                end = np.load(os.path.join(bench, "trajectories", f"d{d}_{arm}_s{s}.npz"))["end"]
                e = wasserstein(np.asarray(end, dtype=np.float64), tgt, max_n=END_MAX_N)
                _write({"variant": "record", "dim": d, "arm": arm, "seed": s,
                        "override": r["override"], "inherited": r["inherited"],
                        "metrics": {"w2_true": float(r["metrics"]["w2_true"]),
                                    "end_w2": float(e)},
                        "source": f"erythroid_bench/test/d{d}_{arm}_s{s}.json + "
                                  f"trajectories/d{d}_{arm}_s{s}.npz[end]"}, path)
        print(f"[leakage] shipped records d{d}: end W2 from saved slabs")


def load_runs(smoke: bool = False):
    import pandas as pd

    rows = []
    for v in ("record",) + VARIANTS:
        d = os.path.join(root(smoke), "runs", v)
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            with open(os.path.join(d, name)) as fh:
                r = json.load(fh)
            rows.append({"variant": v, "dim": r["dim"], "arm": r["arm"], "seed": r["seed"],
                         "mid W2": r["metrics"]["w2_true"], "end W2": r["metrics"]["end_w2"],
                         "device": r.get("device", "")})
    return pd.DataFrame(rows, columns=["variant", "dim", "arm", "seed", "mid W2", "end W2",
                                       "device"])


def _pm(x) -> str:
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return ""
    n = "" if len(x) == len(SEEDS) else f" (n={len(x)})"
    return f"{x.mean():.3f} ± {x.std():.3f}{n}"


def frame_metric(df, metric: str):
    import pandas as pd

    present = [v for v in ("record",) + VARIANTS if (df["variant"] == v).any()]
    out = {}
    for d in DIMS:
        for arm in ARMS + REUSED:
            row = {}
            by = {v: df[(df.variant == v) & (df.dim == d) & (df.arm == arm)].set_index("seed")[metric]
                  for v in present}
            for v in present:
                row[VARIANT_LABELS[v]] = _pm(by[v].values)
            for v in VARIANTS[1:]:
                if v not in present or "shipped" not in present:
                    continue
                common = by["shipped"].index.intersection(by[v].index)
                diff = (by[v][common] - by["shipped"][common]).values
                row[f"Δ {VARIANT_LABELS[v]}"] = (
                    f"{_pm(diff)}  ({int((diff > 0).sum())}/{len(diff)} up)" if len(diff) else "")
            if any(row.values()):
                out[(d, LABELS[arm])] = row
    frame = pd.DataFrame(out).T
    frame.index.names = ["d", f"{metric} (mean ± pop. sd, 5 seeds)"]
    return frame


def frame_gap(df):
    import pandas as pd

    out = {}
    for d in DIMS:
        for v in VARIANTS:
            sub = df[(df.variant == v) & (df.dim == d)]
            if not len(sub):
                continue
            m = sub.groupby("arm")[["mid W2", "end W2"]].mean()
            if not {"curly", "ffm", "pathb"} <= set(m.index):
                continue
            out[(d, VARIANT_LABELS[v])] = {
                "Curly − FFM det, mid": m.loc["curly", "mid W2"] - m.loc["ffm", "mid W2"],
                "Curly − FFM noise, mid": m.loc["curly", "mid W2"] - m.loc["pathb", "mid W2"],
                "Curly − FFM det, end": m.loc["curly", "end W2"] - m.loc["ffm", "end W2"],
                "Curly − FFM noise, end": m.loc["curly", "end W2"] - m.loc["pathb", "end W2"]}
    frame = pd.DataFrame(out).T
    if len(frame):
        frame.index.names = ["d", "velocity"]
    return frame


def frame_diag():
    import pandas as pd

    path = os.path.join(root(), "diag.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as fh:
        diag = json.load(fh)
    cols = ("cos_median", "cos_q25", "cos_q75", "frac_cos_negative", "norm_ratio_median",
            "tv_mean", "tv_shuffled_reference", "n_near_withheld", "cos_median_near",
            "cos_median_far", "tv_mean_near", "tv_mean_far", "cos_J_v_a", "cos_J_v_b",
            "gap_cos_median", "gap_norm_ratio_median", "gap_align_a", "gap_align_b")
    out = {(r["dim"], f"{FIELD_LABELS[r['b']]} vs {FIELD_LABELS[r['a']]}"):
           {c: r[c] for c in cols} for r in diag["agreement"].values()}
    frame = pd.DataFrame(out).T.sort_index()
    frame.index.names = ["d", "velocity pair (shown cells)"]
    return frame


def frame_retune(dim: int = 2, arm: str = "curly"):
    import pandas as pd

    shipped = data_dir("erythroid_sweep")
    rows = []
    for name in sorted(os.listdir(shipped)) if os.path.isdir(shipped) else []:
        if name.startswith(f"w1_d{dim}_{arm}_"):
            with open(os.path.join(shipped, name)) as fh:
                r = json.load(fh)
            rows.append({**r["override"], "objective": r["metrics"]["objective"]})
    cols = {}
    if rows:
        df = pd.DataFrame(rows)
        cols["shipped"] = df.groupby([c for c in df.columns if c != "objective"])["objective"].mean()
    for v in ("trainonly", "ownpca"):
        f = sweep_frame(v, dim, arm)
        if len(f):
            cols[v] = f["objective"]
    frame = pd.DataFrame(cols)
    if len(frame):
        frame.columns = [f"objective, {FIELD_LABELS[c]} velocity" for c in frame.columns]
    return frame


def frames(smoke: bool = False):
    df = load_runs(smoke)
    return (frame_metric(df, "mid W2"), frame_metric(df, "end W2"), frame_gap(df),
            frame_diag(), frame_retune())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="leakage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("pca")
    r = sub.add_parser("refit")
    r.add_argument("--cells", choices=sorted(FITS), required=True)
    r.add_argument("--gpu", type=int, default=-1)
    r.add_argument("--smoke", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--cells", choices=sorted(FITS), nargs="*")
    sub.add_parser("matched")
    sub.add_parser("diag")
    sub.add_parser("reference")
    w = sub.add_parser("sweep")
    w.add_argument("--variant", required=True)
    w.add_argument("--dim", type=int, required=True)
    w.add_argument("--arm", choices=SWEEPABLE, required=True)
    w.add_argument("--index", type=int)
    u = sub.add_parser("run")
    u.add_argument("--dim", type=int, required=True)
    u.add_argument("--arm", choices=ARMS, required=True)
    u.add_argument("--seed", type=int, required=True)
    u.add_argument("--variants", default=",".join(VARIANTS))
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        verify()
    elif a.cmd == "pca":
        pca_step()
    elif a.cmd == "refit":
        refit(a.cells, a.gpu, a.smoke)
    elif a.cmd == "build":
        for t in a.cells or [t for t in FITS if os.path.exists(os.path.join(fit_dir(t), "fit.npz"))]:
            build(t)
    elif a.cmd == "matched":
        build_matched()
    elif a.cmd == "diag":
        diagnostics()
    elif a.cmd == "reference":
        reference()
    elif a.cmd == "sweep":
        from ffm.method import Run

        sweep_cells(a.variant, a.dim, a.arm, Run.from_env(), a.index)
    elif a.cmd == "run":
        from ffm.method import Run

        variants = tuple(v for v in a.variants.split(",") if v)
        assert set(variants) <= set(VARIANTS), variants
        run_cells(a.dim, a.arm, a.seed, variants, Run.from_env())
    else:
        for f in frames():
            print(f.to_string())
            print()


if __name__ == "__main__":
    main()
