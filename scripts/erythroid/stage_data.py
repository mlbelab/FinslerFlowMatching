from __future__ import annotations

import argparse
import json
import os
import shutil
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import scipy.sparse as sp

from ffm.core.paths import data_dir, experiment_root

from .leakage import RAW, _strip_all_cell_state, check_rows, project
from .preprocess import LABEL, N_PCS, _patch_scvelo, finish_from_checkpoint

CACHE_DIR = os.environ.get("ERYTHROID_STAGE_CACHE", data_dir("erythroid_stage_cache"))
STAGES = ("E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5")
EARLY = ("E7.0", "E7.25", "E7.5")
SPLITS = {"long": ((*EARLY, "E7.75"), ("E8.0", "E8.25"), ("E8.5",)),
          "longer": (EARLY, ("E7.75", "E8.0", "E8.25"), ("E8.5",)),
          "early": (("E7.75",), ("E8.0",), ("E8.25",)),
          "late": (("E8.0",), ("E8.25",), ("E8.5",)),
          "loo_e775": (EARLY, ("E7.75",), ("E8.0",)),
          "loo_e80": (("E7.75",), ("E8.0",), ("E8.25",)),
          "loo_e825": (("E8.0",), ("E8.25",), ("E8.5",))}
LOO = ("loo_e775", "loo_e80", "loo_e825")
N_KEEP = 50


def fit_stages(split: str) -> tuple:
    src, held, tgt = SPLITS[split]
    return tuple(s for s in STAGES if s not in held) if split in LOO else (*src, *tgt)


def stage_time(s) -> np.ndarray:
    return np.asarray([float(str(v)[1:]) for v in np.atleast_1d(s)])


def cache_path(split: str) -> str:
    return os.path.join(CACHE_DIR, f"{split}.npz")


def _normalised(counts, genes: np.ndarray, size: np.ndarray, target: float) -> np.ndarray:
    C = counts[:, genes]
    C = np.asarray(C.toarray() if sp.issparse(C) else C, dtype=np.float64)
    return np.log1p(C * (target / np.maximum(size, 1.0))[:, None])


def fit(split: str, gpu: int = -1) -> dict:
    import scanpy as sc
    import scvelo as scv
    import unitvelo as utv
    from scvelo.core import set_initial_size

    t0 = time.time()
    src, held, tgt = SPLITS[split]
    work = os.path.join(experiment_root("erythroid_stage_bench"), f"fit_{split}")
    os.makedirs(work, exist_ok=True)
    adata = sc.read_h5ad(RAW)
    chk = check_rows(adata)
    assert chk["ok"], chk
    stage = np.asarray(adata.obs["stage"].astype(str)).astype(str)
    rows = np.flatnonzero(np.isin(stage, (*fit_stages(split), *held)))
    seen = np.flatnonzero(np.isin(stage, fit_stages(split)))
    adata.var["raw_index"] = np.arange(adata.n_vars)
    set_initial_size(adata)
    size = np.asarray(adata.obs["initial_size"], dtype=np.float64)
    A = adata[seen].copy()
    stripped = _strip_all_cell_state(A)
    cfg = utv.config.Configuration()

    P = A.copy()
    scv.pp.filter_and_normalize(P, min_shared_counts=cfg.MIN_SHARED_COUNTS, n_top_genes=cfg.N_TOP_GENES)
    sc.pp.pca(P, n_comps=N_PCS)
    genes = np.asarray(P.var["raw_index"], dtype=np.int64)
    mean = np.asarray(P.X.mean(axis=0), dtype=np.float64).ravel()
    PCs = np.asarray(P.varm["PCs"], dtype=np.float64)[:, :N_KEEP]
    target = float(np.median(size[seen]))
    X = (_normalised(adata.X[rows], genes, size[rows], target) - mean) @ PCs
    pos = np.searchsorted(rows, seen)
    check = float(np.abs(X[pos] - np.asarray(P.obsm["X_pca"])[:, :N_KEEP]).max())
    print(f"[stage_data] {split}: {len(seen)} seen, {len(rows)} cells, {len(genes)} genes, "
          f"projection check {check:.2e}", flush=True)
    assert check < 1e-3, check
    X[pos] = np.asarray(P.obsm["X_pca"])[:, :N_KEEP]

    ckpt = os.path.join(work, "post_fit_adata.h5ad")
    os.chdir(work)
    _patch_scvelo(ckpt)
    if os.path.exists(ckpt):
        U = finish_from_checkpoint(ckpt)
    else:
        U = A.copy()
        cfg.R2_ADJUST = True
        cfg.IROOT = None
        cfg.FIT_OPTION = "1"
        cfg.GPU = gpu
        cfg.AGENES_R2 = 1
        U = utv.run_model(U, LABEL, config_file=cfg)
    assert np.array_equal(np.asarray(U.obs_names), np.asarray(A.obs_names)), "refit lost the row order"
    vg = np.asarray(U.var["raw_index"], dtype=np.int64)
    V = np.zeros_like(X)
    V[pos] = project({"raw_index": vg, "velocity": np.asarray(U.layers["velocity"])},
                     {"raw_index": genes, "loadings": PCs})
    latent = np.full(len(rows), np.nan)
    latent[pos] = np.asarray(U.obs["latent_time"], dtype=np.float64)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path(split), X=X.astype(np.float32), velocity=V.astype(np.float32), rows=rows,
        stage=stage[rows].astype(str), celltype=np.asarray(adata.obs[LABEL].astype(str))[rows].astype(str),
        seen=np.isin(rows, seen), latent_time=latent, genes=genes)
    info = {"split": split, "source": list(src), "held": list(held), "target": list(tgt),
            "fit_stages": list(fit_stages(split)),
            "counts": {s: int((stage == s).sum()) for s in STAGES if np.isin(s, stage[rows])},
            "n_genes": int(len(genes)), "n_velocity_genes": int(np.asarray(U.var["velocity_genes"]).sum()),
            "gene_overlap": int(np.isin(vg, genes).sum()), "projection_check": check,
            "normalise_target": target, "stripped": stripped, "seconds": time.time() - t0}
    with open(cache_path(split).replace(".npz", ".json"), "w") as fh:
        json.dump(info, fh, indent=1)
    for p in (ckpt, os.path.join(work, "res")):
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="erythroid_stage_data")
    ap.add_argument("split", choices=tuple(SPLITS))
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args(argv)
    print(json.dumps(fit(a.split, a.gpu), indent=1))


if __name__ == "__main__":
    main()
