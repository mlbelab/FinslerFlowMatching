"""UniTVelo preprocessing of the mouse gastrulation erythroid lineage.

This is step 0 of :mod:`scripts.experiments.erythroid`, and it is the one module in
this repository that does **not** run in the repository's environment: UniTVelo pins
TensorFlow, which does not coexist with the main stack, so it gets its own conda env and
hands its result over as a plain ``.npz`` that everything downstream reads.

    conda run -n unitvelo python -m scripts.experiments.erythroid.preprocess

It is also the one step a fresh clone cannot run, because the raw ``.h5ad`` is a 1.4 GB
download rather than something the repository ships.  Point ``--raw`` at your copy (see
Pijuan-Sala et al., 2019); the ``.npz`` files it writes *are* shipped, under
``data/erythroid_cache/``, so the rest of the tree runs without ever executing this.

What it reproduces
------------------
Petrović et al. (2025), *Curly Flow Matching*, §E.2: scVelo + UniTVelo imputation and
velocity graph, with ``utv.run_model`` supplying the unified cell latent time.  Their two
notebooks then build the model spaces, and the two do *not* agree with the paper's prose,
so we follow the **release**:

* ``notebooks/2d_mouse_erythroid.ipynb`` — d = 2 is ``X_umap`` **z-scored per axis**,
  with ``velocity_umap`` as the reference field.
* ``notebooks/nd_mouse_erythroid.ipynb`` — d > 2 is ``sc.pp.pca(adata, 100)`` and the
  first ``d`` **principal components, unstandardised** (their cell 7 computes a
  ``coords`` z-score and then never uses it), with the reference field pushed through the
  same loadings, ``adata.layers["velocity"] @ adata.varm["PCs"]``.  The paper text
  instead says ``highly_variable_genes(n_top_genes=d)``, which is not what the released
  notebook runs; unstandardised PCA is why their d = 20 / 50 L2 column carries a ×10³.

Outputs one ``.npz`` per requested dimension under ``data/erythroid_cache/`` (override
with ``ERYTHROID_CACHE``).  Everything downstream is built in the main environment from
these arrays.

It also writes :data:`AMBIENT_FILE` — the gene space ``adata.X`` those PCA prefixes are a
projection *of*, which the per-dimension files do not carry and which nothing can
reconstruct from them.  That one is **not** shipped (tens of MB, see the repository's
``.gitignore``), so a clone that wants to ask a question about the ambient space rather
than about the model has to re-run this step; :mod:`scripts.experiments.erythroid.ambient`
is what reads it and what says so when it is absent.
"""
from __future__ import annotations

import argparse
import os

# UniTVelo 0.2.5.2 calls ``tf.keras.optimizers.legacy.Adam``, which Keras 3 removed, and
# TensorFlow reads this only at import time — so it must be set before ``unitvelo`` is
# imported anywhere in the process.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np

from scripts.core.paths import rel

#: the raw AnnData this step reads.  Not shipped, so it is an env override over a
#: repo-relative default rather than a path baked into the source.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
RAW = os.environ.get("ERYTHROID_RAW",
                     os.path.join(_REPO, "data", "raw", "erythroid_lineage.h5ad"))
#: where the ``.npz`` files land, and where ``datasets.py`` looks for them.  Same env
#: variable the rest of the tree reads, so the two cannot point at different caches.
CACHE = os.environ.get("ERYTHROID_CACHE", os.path.join(_REPO, "data", "erythroid_cache"))

#: the cluster column UniTVelo groups by; the five erythroid lineage stages.
LABEL = "celltype"
#: PCA rank their nd notebook computes before slicing the first ``d`` columns.
N_PCS = 100
#: dimensions we export.  2 is the UMAP special case; the rest are PCA prefixes.
DIMS = (2, 20, 50)


#: written just before the step that crashed on the first attempt, so a failure after
#: the hour-long TensorFlow fit costs minutes rather than the hour.
CHECKPOINT = "post_fit_adata.h5ad"

#: the gene-space export.  Written by default rather than behind an opt-in flag: it is
#: seconds of work at the end of an hour-long fit, and nobody re-runs that fit to pick up
#: a flag they forgot.  ``--skip-ambient`` is there for the disk-constrained case.
AMBIENT_FILE = "erythroid_ambient.npz"


def _patch_scvelo(ckpt_path: str | None) -> None:
    """Two patches to scVelo 0.2.5, applied from here rather than to site-packages.

    1. ``scvelo.core.parallelize`` ends with ``np.array(res)`` over a *ragged* tuple, on
       which NumPy ≥ 1.24 raises, so scVelo 0.2.5 cannot compute a velocity graph under
       the NumPy TensorFlow 2.21 requires.  The conversion is pointless — the result is
       only ``zip(*res)``-ed — so the call site is rebound to ``as_array=False``, in
       ``scvelo.tools.velocity_graph``'s namespace alone.
    2. ``velocity_graph`` is wrapped to dump the AnnData first, so a crash inside it does
       not cost the 12000-iteration unified-time fit that precedes it.
    """
    import importlib

    import scvelo as scv

    vg = importlib.import_module("scvelo.tools.velocity_graph")
    _orig_parallelize = vg.parallelize

    def parallelize_listwise(*args, **kwargs):
        kwargs["as_array"] = False
        return _orig_parallelize(*args, **kwargs)

    vg.parallelize = parallelize_listwise

    if ckpt_path is None:
        return

    _orig_graph = scv.tl.velocity_graph

    def velocity_graph_checkpointed(adata, *args, **kwargs):
        if not os.path.exists(ckpt_path):
            try:
                adata.write_h5ad(ckpt_path)
                print(f"[preprocess] checkpoint written: {rel(ckpt_path)}")
            except Exception as exc:                      # noqa: BLE001
                print(f"[preprocess] checkpoint FAILED ({type(exc).__name__}: {exc}); "
                      "continuing without one")
        return _orig_graph(adata, *args, **kwargs)

    scv.tl.velocity_graph = velocity_graph_checkpointed


def finish_from_checkpoint(path: str):
    """The tail of ``fit_velo_genes`` re-run on a checkpointed AnnData.

    Replays the four calls UniTVelo makes after the fit, in its order and with its
    arguments.  ``scv.tl.latent_time`` is kept even though ``FIT_OPTION = '1'``
    immediately overwrites its output with the unified ``latent_time_gm``.
    """
    import scanpy as sc
    import scvelo as scv

    adata = sc.read_h5ad(path)
    print(f"[preprocess] resumed from {rel(path)}: {adata.n_obs} cells, "
          f"{adata.n_vars} genes")
    assert "velocity" in adata.layers, "checkpoint predates the velocity layer"
    assert "latent_time_gm" in adata.obs, "checkpoint predates the unified time"

    scv.tl.velocity_graph(adata, sqrt_transform=True)
    scv.tl.velocity_embedding(adata, basis="umap")
    scv.tl.latent_time(adata, min_likelihood=None)
    adata.obs["latent_time"] = adata.obs["latent_time_gm"]
    del adata.obs["latent_time_gm"]
    return adata


def run_unitvelo(raw: str, gpu: int):
    """``utv.run_model`` on the raw h5ad, following UniTVelo's documented workflow.

    ``R2_ADJUST``/``FIT_OPTION='1'`` are UniTVelo's unified-time mode, which produces the
    single transcriptome-wide ``latent_time`` the reference bins into three marginals;
    ``IROOT = None`` leaves the time direction to the model.  These are the package
    defaults for that mode and the reference names no override.

    The AnnData is passed as an *object* rather than as a path, because UniTVelo
    0.2.5.2's path branch calls ``scv.read``, which scVelo dropped in 0.3.  ``run_model``
    runs its own ``filter_and_normalize`` + ``moments`` on whatever it is handed.
    """
    import scanpy as sc
    import scvelo as scv
    import unitvelo as utv

    velo_config = utv.config.Configuration()
    velo_config.R2_ADJUST = True
    velo_config.IROOT = None
    velo_config.FIT_OPTION = "1"
    velo_config.GPU = gpu
    velo_config.AGENES_R2 = 1

    adata = utv.run_model(sc.read_h5ad(raw), LABEL, config_file=velo_config)

    if "velocity_umap" not in adata.obsm:
        scv.tl.velocity_graph(adata)
        scv.tl.velocity_embedding(adata, basis="umap")
    return adata


def build_spaces(adata, dims) -> dict:
    """The (X, velocity) pair for each requested dimension, exactly as they slice it."""
    import scanpy as sc

    out = {}

    if 2 in dims:
        X = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        V = np.asarray(adata.obsm["velocity_umap"], dtype=np.float64)
        out[2] = (X, V)

    hi = [d for d in dims if d > 2]
    if hi:
        sc.pp.pca(adata, n_comps=N_PCS)
        vel = np.asarray(adata.layers["velocity"], dtype=np.float64)
        n_nan = int(np.isnan(vel).any(axis=0).sum())
        vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        PCs = np.asarray(adata.varm["PCs"], dtype=np.float64)
        Xp = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        Vp = vel @ PCs
        print(f"[preprocess] velocity layer: {n_nan}/{vel.shape[1]} genes with NaN "
              f"(zeroed before the PCA projection)")
        for d in hi:
            assert d <= Xp.shape[1], f"asked for d={d} but only {Xp.shape[1]} PCs"
            out[d] = (Xp[:, :d].copy(), Vp[:, :d].copy())
    return out


def export_ambient(adata, out: str) -> str:
    """``adata.X`` — UniTVelo's post-``filter_and_normalize`` gene space — as a CSR ``.npz``.

    This is the space :func:`build_spaces` runs ``sc.pp.pca`` on, so it is the ambient the
    ``d > 2`` columns are prefixes of, and it is the only representation in this pipeline
    that is not already a low-dimensional summary of something else.

    ``celltype`` and ``stage`` ride along so a reader can assert row alignment against any
    of the per-dimension files rather than trusting that two ``.npz`` written by the same
    call are in the same order.  Stored sparse because it is ~75 % zeros and dense float64
    would be an order of magnitude larger for no gain.
    """
    import scipy.sparse as sp

    X = adata.X
    X = sp.csr_matrix(X, dtype=np.float32) if sp.issparse(X) else \
        sp.csr_matrix(np.asarray(X, dtype=np.float32))
    assert np.isfinite(X.data).all(), "non-finite entries in the ambient gene matrix"

    path = os.path.join(out, AMBIENT_FILE)
    np.savez_compressed(
        path, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.asarray(X.shape),
        genes=np.asarray(adata.var_names, dtype=object),
        celltype=np.asarray(adata.obs[LABEL].astype(str)),
        stage=np.asarray(adata.obs["stage"].astype(str)))
    print(f"[preprocess] ambient  X {X.shape}  density {X.nnz / np.prod(X.shape):.3f}  "
          f"-> {path}")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=CACHE)
    ap.add_argument("--dims", nargs="+", type=int, default=list(DIMS))
    ap.add_argument("--gpu", type=int, default=-1,
                    help="UniTVelo GPU index; -1 runs TensorFlow on CPU")
    ap.add_argument("--resume", action="store_true",
                    help="skip the fit and continue from the post-fit checkpoint")
    ap.add_argument("--skip-ambient", action="store_true",
                    help=f"do not write {AMBIENT_FILE} (tens of MB of gene space)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)
    ckpt = os.path.join(args.out, CHECKPOINT)
    _patch_scvelo(ckpt)
    if args.resume:
        assert os.path.exists(ckpt), f"no checkpoint at {ckpt}"
        adata = finish_from_checkpoint(ckpt)
    else:
        adata = run_unitvelo(args.raw, args.gpu)

    latent = np.asarray(adata.obs["latent_time"], dtype=np.float64)
    assert np.isfinite(latent).all(), "UniTVelo returned a non-finite latent_time"

    if not args.skip_ambient:
        export_ambient(adata, args.out)

    spaces = build_spaces(adata, tuple(args.dims))
    for d, (X, V) in sorted(spaces.items()):
        assert X.shape == V.shape, f"d={d}: X {X.shape} vs V {V.shape}"
        assert np.isfinite(X).all() and np.isfinite(V).all(), f"d={d}: non-finite"
        path = os.path.join(args.out, f"erythroid_d{d}.npz")
        np.savez_compressed(path, X=X, velocity=V, latent_time=latent,
                            celltype=np.asarray(adata.obs[LABEL].astype(str)),
                            stage=np.asarray(adata.obs["stage"].astype(str)),
                            umap=np.asarray(adata.obsm["X_umap"], dtype=np.float64))
        print(f"[preprocess] d={d:2d}  X {X.shape}  |v| median "
              f"{np.median(np.linalg.norm(V, axis=1)):.4g}  -> {path}")

    print(f"[preprocess] {adata.n_obs} cells, latent_time in "
          f"[{latent.min():.3f}, {latent.max():.3f}]")


if __name__ == "__main__":
    main()
