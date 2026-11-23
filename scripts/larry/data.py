from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp

from ffm.core.paths import data_dir

RAW = os.environ.get("LARRY_H5AD", os.path.join(data_dir("larry_raw"),
                                                "larry_invitro_adata_sub_raw.h5ad"))
CACHE_DIR = os.environ.get("LARRY_CACHE", data_dir("larry_cache"))
CACHE = os.path.join(CACHE_DIR, "larry_d50.npz")
SUBSAMPLE = os.path.join(CACHE_DIR, "subsample.json")
DAYS = (2.0, 4.0, 6.0)
SOURCE, HELD, TARGET = DAYS
COUNTS = {2.0: 8000, 4.0: 4000, 6.0: 8000}
SEED = 42
N_PCS = 50
N_TOP_GENES = 2000
N_NEIGHBORS = 30
MIN_SHARED_COUNTS = 20


def _dense(M) -> np.ndarray:
    return np.asarray(M.toarray() if sp.issparse(M) else M, dtype=np.float64)


def subsample(time, state, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for day in DAYS:
        pool = np.flatnonzero(time == day)
        want = min(COUNTS[day], len(pool))
        if want == len(pool):
            out[day] = np.sort(pool)
            continue
        types, n = np.unique(state[pool], return_counts=True)
        quota = n * want / len(pool)
        take = np.floor(quota).astype(int)
        take[np.argsort(-(quota - take))[:want - take.sum()]] += 1
        out[day] = np.sort(np.concatenate([rng.choice(pool[state[pool] == t], k, replace=False)
                                           for t, k in zip(types, take)]))
    return out


def _project(counts, keep, target, hvg, mean, PCs) -> np.ndarray:
    total = np.asarray(counts.sum(1), dtype=np.float64).reshape(-1, 1)
    C = _dense(counts[:, keep]) * (target / np.maximum(total, 1.0))
    return (np.log1p(C)[:, hvg] - mean) @ PCs


def preprocess(raw: str = RAW, n_jobs: int = 8) -> dict:
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 1
    adata = sc.read_h5ad(raw)
    time = np.asarray(adata.obs["time_info"], dtype=np.float64)
    state = np.asarray(adata.obs["state_info"].astype(str))
    picks = subsample(time, state)
    order = np.concatenate([picks[d] for d in DAYS])
    sub = adata[order].copy()
    seen = np.isin(np.asarray(sub.obs["time_info"], dtype=np.float64), (SOURCE, TARGET))
    A, B = sub[seen].copy(), sub[~seen].copy()
    raw_counts = A.X.copy()

    target = float(np.median(np.asarray(raw_counts.sum(1), dtype=np.float64)))
    scv.pp.filter_genes(A, min_shared_counts=MIN_SHARED_COUNTS)
    keep = np.flatnonzero(np.isin(sub.var_names, A.var_names))
    scv.pp.normalize_per_cell(A)
    sc.pp.log1p(A)
    sc.pp.highly_variable_genes(A, n_top_genes=N_TOP_GENES)
    hvg = np.asarray(A.var["highly_variable"])
    A = A[:, hvg].copy()
    sc.pp.pca(A, n_comps=N_PCS)
    mean = _dense(A.X).mean(0)
    PCs = np.asarray(A.varm["PCs"])[:, :N_PCS]
    check = float(np.abs(_project(raw_counts, keep, target, hvg, mean, PCs)
                         - np.asarray(A.obsm["X_pca"])[:, :N_PCS]).max())

    scv.pp.moments(A, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS)
    scv.tl.recover_dynamics(A, n_jobs=n_jobs)
    scv.tl.velocity(A, mode="dynamical")
    scv.tl.velocity_graph(A, n_jobs=n_jobs)
    scv.tl.velocity_confidence(A)
    scv.tl.latent_time(A)

    n = sub.n_obs
    X = np.zeros((n, N_PCS))
    V = np.zeros((n, N_PCS))
    X[seen] = np.asarray(A.obsm["X_pca"])[:, :N_PCS]
    X[~seen] = _project(B.X, keep, target, hvg, mean, PCs)
    V[seen] = np.nan_to_num(np.asarray(A.layers["velocity"])) @ PCs
    latent, conf = np.full(n, np.nan), np.full(n, np.nan)
    latent[seen] = np.asarray(A.obs["latent_time"])
    conf[seen] = np.asarray(A.obs["velocity_confidence"])
    clones = sp.csr_matrix(sub.obsm["X_clone"])
    has = np.diff(clones.indptr) > 0
    clone = np.full(n, -1)
    clone[has] = clones.indices[clones.indptr[:-1][has]]

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        CACHE, X=X.astype(np.float32), velocity=V.astype(np.float32),
        day=np.asarray(sub.obs["time_info"], dtype=np.float64),
        state=np.asarray(sub.obs["state_info"].astype(str), dtype=str), clone=clone.astype(np.int64),
        cell_id=np.asarray(sub.obs_names, dtype=str), library=np.asarray(sub.obs["Library"].astype(str), dtype=str),
        raw_index=order.astype(np.int64), latent_time=latent, velocity_confidence=conf)
    info = {"seed": SEED, "requested": {str(k): v for k, v in COUNTS.items()},
            "available": {str(d): int((time == d).sum()) for d in DAYS},
            "selected": {str(d): int(len(picks[d])) for d in DAYS},
            "raw_index": {str(d): picks[d].tolist() for d in DAYS},
            "n_genes_filtered": int(len(keep)), "n_hvg": int(hvg.sum()),
            "normalise_target": target, "day4_projection_check": check,
            "velocity_fit_cells": int(seen.sum()), "n_jobs": n_jobs}
    with open(SUBSAMPLE, "w") as fh:
        json.dump(info, fh)
    return info


def load() -> dict:
    with np.load(CACHE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="larry_data")
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args(argv)
    info = preprocess(a.raw, a.n_jobs)
    print(json.dumps({k: v for k, v in info.items() if k != "raw_index"}, indent=1))


if __name__ == "__main__":
    main()
