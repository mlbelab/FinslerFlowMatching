"""The erythroid cloud in **gene space**, if step 0 was re-run to export it.

The shipped ``erythroid_d{2,20,50}.npz`` carry ``X``, ``velocity``, ``latent_time`` and the
labels — and nothing else.  The ``d > 2`` columns are prefixes of a 100-component PCA of
UniTVelo's post-``filter_and_normalize`` ``adata.X``, but neither that matrix nor its
loadings (``varm["PCs"]``) were written, so the ambient space **cannot be recovered from
what the repository ships**.  It is not a matter of inverting a projection: a 50-column
prefix of a rank-100 PCA of a 2000-gene matrix has thrown away both the discarded
components and the basis.

Unlike the pancreas — whose ambient :mod:`scripts.experiments.pancreas.ambient` rebuilds in
twelve seconds from a 52 MB download — this one needs the 1.4 GB Pijuan-Sala h5ad *and* the
TensorFlow-pinned UniTVelo environment, because the gene space is defined by what
``utv.run_model`` filtered and normalised.  So :func:`load_ambient` is a reader, not a
builder: it returns the matrix if
:data:`scripts.experiments.erythroid.preprocess.AMBIENT_FILE` is on disk and otherwise
returns ``None`` together with the exact command that produces it.

Returning ``None`` rather than raising is deliberate.  A notebook cell that compares
representations should print three PCA columns and a line saying why the fourth is missing;
it should not fail to execute on a clone, and it should not quietly drop the column as
though ambient space had never been part of the comparison.
"""
from __future__ import annotations

import os

import numpy as np

from scripts.experiments.erythroid.datasets import CACHE_DIR
from scripts.experiments.erythroid.preprocess import AMBIENT_FILE

#: the command that writes the file, quoted verbatim in the not-found message.
BUILD_HINT = (
    "ERYTHROID_RAW=/path/to/erythroid_lineage.h5ad \\\n"
    "    conda run -n unitvelo python -m scripts.experiments.erythroid.preprocess")


def ambient_path(cache_dir: str | None = None) -> str:
    return os.path.join(cache_dir or CACHE_DIR, AMBIENT_FILE)


def load_ambient(cache_dir: str | None = None, check_against: np.ndarray | None = None
                 ) -> tuple[np.ndarray | None, dict]:
    """``(X, info)`` with ``X`` dense ``(n_cells, n_genes)``, or ``(None, info)``.

    ``info["reason"]`` is set exactly when ``X`` is ``None`` and is written to be printed
    to a reader as-is.  ``check_against`` takes the ``celltype`` array of a per-dimension
    cache and asserts the two are row-aligned, which is the only thing that makes an
    ambient number comparable with the PCA numbers beside it.
    """
    path = ambient_path(cache_dir)
    if not os.path.exists(path):
        return None, {"path": path, "reason": (
            f"no ambient gene matrix at {path}.\n"
            f"The erythroid PCA caches do not carry the space they project, and it cannot "
            f"be reconstructed from them (no loadings, no gene matrix).  Re-run step 0 "
            f"with the raw 1.4 GB h5ad to export it:\n\n    {BUILD_HINT}\n")}

    import scipy.sparse as sp

    z = np.load(path, allow_pickle=True)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                      shape=tuple(int(s) for s in z["shape"]))
    dense = np.asarray(X.todense(), dtype=np.float64)

    if check_against is not None:
        got = np.asarray(z["celltype"], dtype=str)
        ref = np.asarray(check_against, dtype=str)
        assert got.shape == ref.shape and (got == ref).all(), (
            f"the ambient export is not row-aligned with the PCA cache "
            f"({got.shape} vs {ref.shape}); they came from different preprocess runs")

    return dense, {"path": path, "genes": z["genes"],
                   "density": float(X.nnz / np.prod(X.shape))}
