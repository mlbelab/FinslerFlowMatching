"""Load the scVelo Pancreas endocrinogenesis dataset into a Finsler ``Dataset``.

This is the real-data counterpart of the synthetic generators in
``scripts.core.datasets``.  It turns the raw h5ad
(``endocrinogenesis_day15.h5ad``: 3696 cells, raw spliced/unspliced counts,
clusters ``Ductal -> Ngn3 low EP -> Ngn3 high EP -> Pre-endocrine ->
{Alpha, Beta, Delta, Epsilon}``) into the seven fields the pipeline consumes:

    X    : (N, 50)  top-50 PCA coordinates (geometry + network space)
    P    : (N, N)   scipy-sparse row-stochastic CellRank transition matrix,
                    a MIXTURE of a velocity kernel and a similarity kernel:
                        P = vel_weight · VelocityKernel + (1-vel_weight) · ConnectivityKernel
    tau  : (N,)     scVelo dynamical ``latent_time`` rescaled to [0, 1]
    v    : (N, 50)  unit P-implied displacement (drift proxy; anchors/shape only)
    p0   : (N,) bool  source marginal  = Ductal (root progenitor)
    p1   : (N,) bool  target marginal  = the 4 terminal fates {Alpha,Beta,Delta,Epsilon}

Alongside the ``Dataset`` we return a :class:`PancreasMeta` sidecar with the UMAP
embedding, cluster labels, and the CellRank absorption-probability matrix ``B``
(N x C, fate probabilities to the terminal macrostates).  Nothing in the published
benchmarks reads ``B`` — it was the input to a fate-concordance evaluation that ran only
in the exploratory half of the repository — but it is cached alongside the rest because
recomputing it means re-running GPCCA, and the pancreas tree's celltype-TV statistic is
the surviving descendant of that question.

The scVelo dynamical fit (``recover_dynamics``) is slow (minutes), so all derived
arrays are cached to ``cache_dir`` and reloaded on subsequent runs.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .datasets import Dataset, _unit
from .paths import data_dir, rel

# The canonical differentiation endpoints of the Pancreas set.
ROOT_CLUSTER = "Ductal"
TERMINAL_FATES = ("Alpha", "Beta", "Delta", "Epsilon")


@dataclass
class PancreasMeta:
    """Non-``Dataset`` side information used for visualisation + fate concordance."""
    umap: np.ndarray            # (N, 2)  precomputed UMAP embedding
    clusters: np.ndarray        # (N,)    cluster label strings
    latent_time: np.ndarray     # (N,)    scVelo latent time in [0, 1]
    B: np.ndarray               # (N, C)  absorption / fate probabilities
    fate_names: list[str]       # (C,)    column names of B (terminal macrostates)
    terminal_centroids: np.ndarray   # (C, d)  PCA centroid of each fate's cells
    velocity_pca: np.ndarray    # (N, d)  RNA velocity projected to PCA (Curly-FM reference)
    velocity_umap: np.ndarray   # (N, 2)  RNA velocity embedded into UMAP (Curly-FM 2-D reference)
    root_cluster: str = ROOT_CLUSTER


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
def load_pancreas(
    path: str,
    n_pcs: int = 50,
    n_top_genes: int = 2000,
    n_neighbors: int = 30,
    vel_weight: float = 0.8,
    mode: str = "dynamical",
    cache_dir: str | None = None,
    rebuild: bool = False,
    seed: int = 0,
) -> tuple[Dataset, PancreasMeta]:
    """Build (or load from cache) the Pancreas ``Dataset`` + ``PancreasMeta``.

    ``cache_dir`` defaults to the shipped ``<public>/data/pancreas_cache``, resolved
    against this file rather than the working directory -- a relative default would send
    a notebook and a batch cell to two different caches.
    """
    cache_dir = cache_dir or data_dir("pancreas_cache")
    cache = os.path.join(cache_dir, f"pancreas_d{n_pcs}_vw{vel_weight}_{mode}.npz")
    if os.path.exists(cache) and not rebuild:
        print(f"[scvelo_data] loading cache {rel(cache)}")
        return _load_cache(cache)

    ds, meta = _build_from_h5ad(
        path, n_pcs, n_top_genes, n_neighbors, vel_weight, mode, seed)
    os.makedirs(cache_dir, exist_ok=True)
    _save_cache(cache, ds, meta)
    print(f"[scvelo_data] cached -> {rel(cache)}")
    return ds, meta


def project_to_dim(ds: Dataset, meta: PancreasMeta, dim: int
                   ) -> tuple[Dataset, PancreasMeta]:
    """Restrict the Pancreas dataset to a ``dim``-dimensional coordinate space.

    We run the *same* model in three data spaces to compare how dimensionality
    affects the transport, and — crucially — to plot honestly:

      * ``dim == 2``  -> the precomputed 2-D **UMAP embedding** becomes the data /
        geometry / plotting space.  Because the flow now lives directly in UMAP
        coordinates, pushed-forward points ARE genuine UMAP coordinates and can be
        scattered as-is (no non-invertible "nearest real cell" projection needed).
      * ``dim <= n_pcs`` -> the top-``dim`` **PCA columns**.  PCA axes are ordered
        by explained variance, so slicing ``X[:, :dim]`` is the natural
        lower-dimensional embedding (top-10 PCs ⊂ top-50 PCs).

    The transition matrix ``P``, progress coordinate ``tau`` and endpoint marginals
    ``p0``/``p1`` are defined per cell and are therefore dimension-independent — they
    are carried over unchanged.  Only the geometry-relevant fields are rebuilt in
    the new coordinate space: the drift proxy ``v = unit(P·X − X)`` and the
    terminal-fate centroids used by the nearest-centroid fate assignment.
    """
    d_full = ds.X.shape[1]
    if dim == 2:
        X = np.asarray(meta.umap, dtype=np.float64).copy()
    else:
        assert 1 <= dim <= d_full, (
            f"dim must be 2 (UMAP) or in [1, {d_full}] (PCA), got {dim}")
        X = np.asarray(ds.X[:, :dim], dtype=np.float64).copy()

    P = ds.P.tocsr() if sp.issparse(ds.P) else ds.P
    v = _unit(np.asarray(P @ X - X))                      # one-step P displacement
    ds_d = Dataset(ds.name, X=X, P=ds.P, tau=ds.tau, v=v, p0=ds.p0, p1=ds.p1)

    # terminal-fate centroids must live in the SAME space the endpoints land in
    centroids = np.stack(
        [X[meta.clusters == f].mean(0) for f in meta.fate_names], axis=0)
    # Curly-FM's reference velocity field must live in the current space:
    #   * PCA (dim<=50): the RNA velocity linearly projected onto the top-`dim` PCs
    #     (velocity_pca is already `Vgene @ PCs`, so slicing is the top-`dim` view).
    #   * UMAP (dim==2): the RNA velocity *embedded* into the UMAP by scVelo's
    #     ``velocity_embedding`` (``meta.velocity_umap``) — Curly-FM's authentic 2-D
    #     reference field (the non-linear projection RNA velocity has no linear analogue for).
    if dim == 2:
        velocity = np.asarray(meta.velocity_umap, dtype=np.float64).copy()
    else:
        velocity = np.asarray(meta.velocity_pca[:, :dim], dtype=np.float64).copy()
    meta_d = dataclasses.replace(
        meta, terminal_centroids=centroids, velocity_pca=velocity)
    return ds_d, meta_d


# --------------------------------------------------------------------------- #
#  Heavy path: scVelo + CellRank
# --------------------------------------------------------------------------- #
def _build_from_h5ad(path, n_pcs, n_top_genes, n_neighbors, vel_weight, mode, seed):
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 1
    sc.settings.verbosity = 1

    print(f"[scvelo_data] reading {rel(path)}")
    adata = sc.read_h5ad(path)
    umap = np.asarray(adata.obsm["X_umap"], dtype=np.float64).copy()  # keep original UMAP
    clusters = np.asarray(adata.obs["clusters"].astype(str))

    # --- scVelo preprocessing (gene filtering only; all 3696 cells retained) ---
    # This scvelo version's ``filter_and_normalize`` neither accepts ``n_top_genes``
    # nor exposes ``filter_genes_dispersion``, so we filter + normalise with scvelo
    # and select highly-variable genes with scanpy.
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    # explicit PCA so the loadings (varm['PCs']) exist and X_pca is consistent with
    # the velocity projection below (mirrors the Curly-FM notebooks' sc.pp.pca).
    sc.pp.pca(adata, n_comps=n_pcs)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    assert adata.n_obs == umap.shape[0], "cell count changed during preprocessing"

    # --- velocity (dynamical: recover_dynamics enables true latent_time) ---
    if mode == "dynamical":
        print("[scvelo_data] recover_dynamics (slow) ...")
        scv.tl.recover_dynamics(adata, n_jobs=max(1, os.cpu_count() // 2))
        scv.tl.velocity(adata, mode="dynamical")
    else:
        scv.tl.velocity(adata, mode=mode)
    scv.tl.velocity_graph(adata, n_jobs=max(1, os.cpu_count() // 2))

    # --- RNA velocity embedded into the 2-D UMAP (Curly-FM's true 2-D reference field) ---
    # scv.tl.velocity_embedding projects the high-D RNA velocity onto the UMAP via the
    # velocity graph (the non-linear analogue of the linear PCA projection below).  This
    # is exactly ``adata.obsm['velocity_umap']`` that the Curly-FM repo consumes in its
    # 2-D / embedding setting (``get_cell_velocities`` -> ``velocity_umap``).
    scv.tl.velocity_embedding(adata, basis="umap")
    velocity_umap = np.asarray(adata.obsm["velocity_umap"], dtype=np.float64).copy()

    # --- progress coordinate tau = latent_time (dynamical) or velocity_pseudotime ---
    try:
        scv.tl.latent_time(adata)
        tau_raw = np.asarray(adata.obs["latent_time"], dtype=np.float64)
    except Exception as e:  # stochastic mode / degenerate fit -> fall back
        print(f"[scvelo_data] latent_time failed ({e}); using velocity_pseudotime")
        scv.tl.velocity_pseudotime(adata)
        tau_raw = np.asarray(adata.obs["velocity_pseudotime"], dtype=np.float64)
    tau = _rescale01(tau_raw)

    # --- X = top-n_pcs PCA (recomputed by moments on the processed data) ---
    X = np.asarray(adata.obsm["X_pca"][:, :n_pcs], dtype=np.float64).copy()
    d = X.shape[1]

    # --- RNA velocity projected into the same PCA space (Curly-FM reference field) ---
    # velocity_pca = velocity_genes @ PCs  (linear projection; NaN velocities -> 0),
    # exactly as the Curly-FM notebooks build ``adata.obsm['X_velocity']``.
    Vgene = np.nan_to_num(np.asarray(adata.layers["velocity"]), nan=0.0)
    PCs = np.asarray(adata.varm["PCs"])[:, :n_pcs]
    velocity_pca = (Vgene @ PCs).astype(np.float64)                # (N, n_pcs)

    # --- P = mixture of a velocity kernel and a similarity kernel (CellRank) ---
    P, combined = _build_mixture_P(adata, vel_weight)

    # --- fate / absorption probabilities B to the terminal macrostates ---
    B, fate_names = _fate_probabilities(combined, adata, clusters)

    # --- endpoint marginals ---
    p0 = clusters == ROOT_CLUSTER
    p1 = np.isin(clusters, TERMINAL_FATES)
    assert p0.any() and p1.any(), "empty p0/p1 — check cluster names"

    # --- v: unit P-implied one-step displacement in PCA space (drift proxy) ---
    v = _unit(np.asarray(P @ X - X))

    # terminal-fate PCA centroids (for fate concordance nearest-centroid assignment)
    centroids = np.stack([X[clusters == f].mean(0) for f in fate_names], axis=0)

    ds = Dataset("Pancreas", X=X, P=P.tocsr(), tau=tau, v=v, p0=p0, p1=p1)
    meta = PancreasMeta(
        umap=umap, clusters=clusters, latent_time=tau, B=B,
        fate_names=list(fate_names), terminal_centroids=centroids,
        velocity_pca=velocity_pca, velocity_umap=velocity_umap)
    return ds, meta


def _build_mixture_P(adata, vel_weight: float):
    """P = vel_weight·VelocityKernel + (1-vel_weight)·ConnectivityKernel  (row-stochastic)."""
    from cellrank.kernels import ConnectivityKernel, VelocityKernel

    print(f"[scvelo_data] CellRank kernels: {vel_weight} velocity + "
          f"{1 - vel_weight} similarity")
    vk = VelocityKernel(adata).compute_transition_matrix()
    ck = ConnectivityKernel(adata).compute_transition_matrix()
    combined = vel_weight * vk + (1.0 - vel_weight) * ck
    combined.compute_transition_matrix()
    P = sp.csr_matrix(combined.transition_matrix)
    # guarantee exact row-stochasticity for the Dataset assertion
    rs = np.asarray(P.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    P = sp.diags(1.0 / rs) @ P
    return P.tocsr(), combined


def _fate_probabilities(combined, adata, clusters):
    """Absorption probabilities B (N x C) to the terminal fates via CellRank GPCCA.

    Terminal states are pinned to the four known endocrine fate clusters so that
    the columns of ``B`` line up with :data:`TERMINAL_FATES`.
    """
    import cellrank as cr
    import pandas as pd

    g = cr.estimators.GPCCA(combined)
    # pin terminal states to the known fate clusters (cell -> fate name, else NaN)
    labels = pd.Series(np.where(np.isin(clusters, TERMINAL_FATES), clusters, np.nan),
                       index=adata.obs_names, dtype="object").astype("category")
    try:
        g.set_terminal_states(labels)
        g.compute_fate_probabilities()
    except Exception as e:
        print(f"[scvelo_data] pinned terminal states failed ({e}); "
              f"falling back to GPCCA auto-detection")
        g.compute_schur(n_components=20)
        g.compute_macrostates(n_states=len(TERMINAL_FATES) + 2, cluster_key="clusters")
        g.predict_terminal_states()
        g.compute_fate_probabilities()

    fp = g.fate_probabilities
    B = np.asarray(fp)                       # (N, C)
    fate_names = [str(x) for x in fp.names]
    return B, fate_names


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _rescale01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.all():                     # NaNs -> median (keeps monotonicity intact)
        a = a.copy()
        a[~finite] = np.nanmedian(a[finite])
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)


def _save_cache(path: str, ds: Dataset, meta: PancreasMeta) -> None:
    P = ds.P.tocsr()
    np.savez_compressed(
        path,
        X=ds.X, tau=ds.tau, v=ds.v, p0=ds.p0, p1=ds.p1,
        P_data=P.data, P_indices=P.indices, P_indptr=P.indptr, P_shape=P.shape,
        umap=meta.umap, clusters=meta.clusters, latent_time=meta.latent_time,
        B=meta.B, fate_names=np.array(meta.fate_names, dtype=object),
        terminal_centroids=meta.terminal_centroids,
        velocity_pca=meta.velocity_pca, velocity_umap=meta.velocity_umap,
    )


def _load_cache(path: str) -> tuple[Dataset, PancreasMeta]:
    z = np.load(path, allow_pickle=True)
    P = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]),
                      shape=tuple(z["P_shape"]))
    ds = Dataset("Pancreas", X=z["X"], P=P, tau=z["tau"], v=z["v"],
                 p0=z["p0"], p1=z["p1"])
    # velocity_umap was added later; fall back to zeros for pre-existing caches so an
    # un-rebuilt cache still loads (Curly-FM 2-D would then be degenerate — rebuild to fix).
    velocity_umap = (z["velocity_umap"] if "velocity_umap" in z.files
                     else np.zeros_like(z["umap"]))
    meta = PancreasMeta(
        umap=z["umap"], clusters=z["clusters"], latent_time=z["latent_time"],
        B=z["B"], fate_names=list(z["fate_names"]),
        terminal_centroids=z["terminal_centroids"],
        velocity_pca=z["velocity_pca"], velocity_umap=velocity_umap)
    return ds, meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    # The h5ad is not shipped: ``scvelo.datasets.pancreas()`` downloads it, and the
    # derived cache under ``<public>/data/`` is what every other module reads.  Rebuilding
    # is an author's operation, so the raw file's location is given rather than defaulted.
    ap.add_argument("--path", default=os.environ.get("PANCREAS_H5AD", ""),
                    help="raw endocrinogenesis_day15.h5ad; $PANCREAS_H5AD by default")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    # fail before scVelo is imported, not four minutes into a fit
    assert a.path or not a.rebuild, (
        "--rebuild needs the raw h5ad: pass --path, set $PANCREAS_H5AD, or fetch it with "
        "scvelo.datasets.pancreas()")
    ds, meta = load_pancreas(a.path, rebuild=a.rebuild)
    print("Dataset:", ds.name, "X", ds.X.shape, "P nnz/row",
          round(ds.P.nnz / ds.X.shape[0], 1))
    print("p0", int(ds.p0.sum()), "p1", int(ds.p1.sum()),
          "tau", (round(float(ds.tau.min()), 3), round(float(ds.tau.max()), 3)))
    print("fates", meta.fate_names, "B", meta.B.shape,
          "B rowsum~1:", bool(np.allclose(meta.B.sum(1), 1.0, atol=1e-3)))
