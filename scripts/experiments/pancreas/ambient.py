"""The pancreas cloud in **gene space** — the ambient matrix the cached PCA prefixes came from.

:mod:`scripts.experiments.pancreas.datasets` hands out ``d = 2 / 10 / 20 / 50``, and the
last three are prefixes of one 50-component PCA.  That PCA was computed inside
:func:`scripts.core.scvelo_data._build_from_h5ad` and only its *output* was cached, so the
space it is a projection **of** is not in the shipped ``.npz``.  Anything that wants to ask
a question about the representation rather than about the model — how many dimensions the
cloud locally occupies, say — needs that space back.

This module rebuilds it by replaying exactly the gene-space half of that function::

    scv.pp.filter_genes(min_shared_counts=20)
    scv.pp.normalize_per_cell()
    sc.pp.log1p()
    sc.pp.highly_variable_genes(n_top_genes=2000)   ->  subset

and stops there, before ``sc.pp.pca`` and long before ``recover_dynamics``.  So it costs
seconds rather than the minutes the dynamical fit costs, and it needs scanpy and scVelo but
no CellRank.

The rebuild is **checked, not asserted by construction**: :func:`load_ambient` runs the same
``sc.pp.pca(n_comps=50)`` on what it just built and requires it to reproduce the cached
``X`` component-for-component (up to the sign each eigenvector is defined only up to).  If
the scanpy version, the gene filter or the HVG flavour has drifted since the cache was
written, that check fails loudly rather than handing back a matrix which is *a* gene space
but not *the* one the columns come from.

The raw h5ad is a 52 MB download rather than something the repository ships; scVelo will
fetch it on first use.  The rebuilt matrix is cached beside the other pancreas arrays.
"""
from __future__ import annotations

import os

import numpy as np

from scripts.core.paths import data_dir, rel
from scripts.core.scvelo_data import load_pancreas
from scripts.experiments.pancreas.datasets import CACHE_DIR, N_PCS, VEL_WEIGHT, VELO_MODE

#: highly-variable genes retained — ``_build_from_h5ad``'s ``n_top_genes`` default, which
#: is what the cached PCA was computed on.  Changing it invalidates the check below, which
#: is the point of naming it here rather than inlining the literal.
N_TOP_GENES = 2000

#: minimum shared spliced/unspliced counts, ``scv.pp.filter_genes``'s argument there.
MIN_SHARED_COUNTS = 20

#: the raw AnnData.  Not shipped (52 MB); ``scv.datasets.pancreas`` downloads it here on
#: first use, and the env var lets a cluster job point at a copy it already has.
RAW = os.environ.get(
    "PANCREAS_RAW", os.path.join(data_dir("raw"), "endocrinogenesis_day15.h5ad"))

#: name of the rebuilt cache, inside the same directory as the PCA cache it belongs to.
AMBIENT_FILE = f"pancreas_ambient_hvg{N_TOP_GENES}.npz"

#: the PCA check's tolerance.  Loose because scanpy's ARPACK solver is not bit-stable
#: across runs and the trailing components of a 50-component fit are genuinely noisy;
#: tight enough that a different gene set or a different normalisation cannot pass.
PCA_MIN_ABS_CORR = 0.999


def ambient_path(cache_dir: str | None = None) -> str:
    return os.path.join(cache_dir or CACHE_DIR, AMBIENT_FILE)


def load_ambient(cache_dir: str | None = None, raw: str = RAW,
                 rebuild: bool = False, download: bool = True
                 ) -> tuple[np.ndarray, dict]:
    """``(X, info)`` — the ``(3696, 2000)`` log-normalised HVG matrix, cached.

    ``info`` carries the gene names, the sparsity, and the worst per-component correlation
    the PCA check saw, so a caller can print how well the rebuild matched rather than
    taking the assert's word for it.
    """
    path = ambient_path(cache_dir)
    if os.path.exists(path) and not rebuild:
        z = np.load(path, allow_pickle=True)
        X = np.asarray(z["X"], dtype=np.float64)
        return X, {"genes": z["genes"], "source": path,
                   "min_abs_corr": float(z["min_abs_corr"]),
                   "density": float((X != 0).mean())}

    X, genes, min_corr = _rebuild(raw, cache_dir, download)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), genes=genes,
                        min_abs_corr=np.float64(min_corr))
    print(f"[pancreas.ambient] cached -> {rel(path)}")
    return X, {"genes": genes, "source": path, "min_abs_corr": min_corr,
               "density": float((X != 0).mean())}


def _rebuild(raw: str, cache_dir: str | None, download: bool):
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 1
    sc.settings.verbosity = 1

    if not os.path.exists(raw):
        assert download, (
            f"no raw h5ad at {raw} and download=False.\n"
            f"Fetch it once (52 MB) with:\n"
            f"    python -c \"import scvelo as scv; scv.datasets.pancreas('{raw}')\"\n"
            f"or point $PANCREAS_RAW at an existing copy.")
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        print(f"[pancreas.ambient] downloading the raw h5ad (52 MB) -> {raw}")
        scv.datasets.pancreas(raw)

    print(f"[pancreas.ambient] replaying the gene-space preprocessing on {raw}")
    adata = sc.read_h5ad(raw)
    scv.pp.filter_genes(adata, min_shared_counts=MIN_SHARED_COUNTS)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES)
    adata = adata[:, adata.var["highly_variable"]].copy()

    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    genes = np.asarray(adata.var_names, dtype=object)
    assert X.shape[1] == N_TOP_GENES, f"HVG selection returned {X.shape[1]} genes"

    # --- the check: does this space's own PCA reproduce the cached prefixes? -------- #
    ds, _meta = load_pancreas("", n_pcs=N_PCS, vel_weight=VEL_WEIGHT, mode=VELO_MODE,
                              cache_dir=cache_dir or CACHE_DIR)
    cached = np.asarray(ds.X, dtype=np.float64)
    assert cached.shape[0] == X.shape[0], (
        f"cell count drifted: cache has {cached.shape[0]}, rebuild has {X.shape[0]}")

    sc.pp.pca(adata, n_comps=N_PCS)
    fresh = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    # each eigenvector is defined up to a sign, so the comparison is on |correlation|
    corr = np.array([abs(np.corrcoef(fresh[:, j], cached[:, j])[0, 1])
                     for j in range(N_PCS)])
    min_corr = float(np.nanmin(corr))
    assert min_corr >= PCA_MIN_ABS_CORR, (
        f"the rebuilt gene space does not reproduce the cached PCA: worst component "
        f"|corr| = {min_corr:.6f} < {PCA_MIN_ABS_CORR} (component "
        f"{int(np.nanargmin(corr))}).  The preprocessing in scripts.core.scvelo_data or "
        f"the installed scanpy/scVelo has drifted from what wrote the cache.")
    print(f"[pancreas.ambient] PCA check ok: worst of {N_PCS} components "
          f"|corr| = {min_corr:.6f}")
    return X, genes, min_corr
