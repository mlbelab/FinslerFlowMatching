"""The iTracer microdissected cerebral organoid, its region R2, and its scar families.

He et al. (2022), *Lineage recording in human cerebral organoids*, Nature Methods 19,
90-99 — ``10.1038/s41592-021-01344-8``.  iTracer writes two things into the same cells
that are read out with the transcriptome: a **reporter barcode**, integrated once per
founder and inherited by everything downstream of it, and an **inducible CRISPR scar**,
written at day 15 into that barcode's target site.  Barcode nests scar, so a cell carries
a two-level address — barcode family, then scar family within it — and the pair is a
*recorded* ancestry rather than a transcriptional guess.

Their Fig. 5 is the deep-sampling experiment: one 200 um vibratome section of a single
day-60 organoid (Org13, scarred at day 15), two spatially distant regions microdissected
off it and captured separately, two 10x reactions each.  Region 1 is
diencephalon-mesencephalon, **region 2 the rhombencephalon (hindbrain)**, and the two
regions' lineages are entirely disjoint.  This module keeps R2.

Why R2 is the right cloud for a lineage-informed transition matrix
------------------------------------------------------------------
* **The lineage is recorded and it is deep.**  R2 holds nine scar families of at least 50
  cells each, and the tenth-largest has 13 — a genuine gap, not a threshold artefact.
  Every one of those nine contains both progenitors and neurons, so the barcode says
  something about *transport* and not merely about which cells are alike.  All nine carry
  a real edit; see :func:`scar_is_edited` for the distinction that matters here.
* **The trajectory is one axis.**  The authors' own analysis (their Fig. 5f-i) runs a
  diffusion pseudotime on exactly these cells and reads it as NPC -> neuron; the cells are
  their cluster group CG3-1, the NPC/neuron half of the hindbrain region.
* **It is a *reconstructed* lineage.**  Unlike *C. elegans* the shared-scar relation is
  probabilistic and coarse -- three barcode families, nine scars -- which is the realistic
  case for a lineage-informed kernel.

What is kept
------------
:func:`load_cloud` reproduces the authors' selection, in their order:

1. R2 only — capture reactions ``Region2.1`` and ``Region2.2`` (16 274 cells).
2. Cluster group **CG3-1**, :data:`CG3_1`, the neurogenic clusters of that region, taken
   from the authors' own ``RNA_snn_res.0.6`` labels in the released metadata rather than
   reclustered (13 053 cells).
3. Cells whose barcode family has more than :data:`MIN_BARCODE_CELLS` cells there *and*
   whose scar family has more than :data:`MIN_SCAR_CELLS` (2901 cells, 3 barcode families,
   9 scar families — the same nine the paper reports).

The caches
----------
Two files under ``<public>/data/itracer/``.  The first is the released matrix rebuilt as
one ``.h5ad`` (29 274 cells x 25 800 genes of raw counts, plus the released per-cell
metadata); building it pulls ~190 MB from Mendeley Data.  The second is a few-MB ``.npz``
of everything derived — the 50 PCs, both UMAPs, the pseudotime, the NPC/neuron score —
so a notebook can plot without redoing the embedding.  Neither touches the network once
written.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from scripts.core.paths import data_dir

#: where both caches and the downloaded release files live.
CACHE_DIR = os.environ.get("ITRACER_CACHE", data_dir("itracer"))

#: Mendeley Data 10.17632/nj3p3pxv6p, the paper's processed-data deposit.  Keyed by the
#: local filename, valued by the file's UUID in the ``public-files`` endpoint.  The three
#: ``iTracer (microdissected)`` members plus the NPC-vs-neuron differential-expression
#: table from ``ext`` that the authors' own score is defined against.  Their ``seurat.rds``
#: is the same matrix again inside a 1.3 GB R object and is not fetched.
MENDELEY = "https://data.mendeley.com/public-files/datasets/nj3p3pxv6p/files"
RELEASE = {
    "counts.mtx.gz": "6fec78ed-e686-4b24-bb93-6beb599e9c7e",
    "features.tsv.gz": "00bdc8a3-ee77-481b-ae1c-aec6fc8d0d1c",
    "meta.tsv.gz": "407c2850-82c3-4acc-8067-cccc871bbb04",
    "Kanton_DE_NPC_neurons.tsv": "06433194-a5bd-4285-b154-ef17853acccb",
}

#: the two 10x reactions the hindbrain region was split across.
R2_SAMPLES = ("Region2.1", "Region2.2")

#: R1, the diencephalon-mesencephalon region, kept only to count what is being dropped.
R1_SAMPLES = ("Region1.1", "Region1.2")

#: cluster group CG3-1 in the authors' ``RNA_snn_res.0.6`` labelling — the NPC-to-neuron
#: clusters of R2, and the exact set their Fig. 5f pseudotime was run on.  Seurat writes
#: this column 0-based, so these are the integers in the released ``meta.tsv.gz``.
CG3_1 = (1, 4, 5, 7, 8, 10, 12, 17)

#: a barcode family must exceed this many cells in CG3-1 n R2, and a scar family this
#: many, to enter the cloud.  The authors' thresholds; the second is what makes it "the
#: nine scar families with at least 50 cells" of their Fig. 5f.
MIN_BARCODE_CELLS, MIN_SCAR_CELLS = 100, 50

#: the embedding, as the paper built it: 5000 vst highly variable genes, scaled, 50 PCs,
#: and the first 20 of those carried into UMAP, the diffusion map and the pseudotime.
#:
#: All 50 are cached, because a benchmark space of dimension ``d`` is the first ``d`` of
#: them and the widest column is ``d = 50``.  :data:`PT_PCS` does **not** follow: the
#: diffusion map, ``s``, the tertile cuts and hence the whole task are pinned to the first
#: 20 components and are the same numbers they were when 20 was all that was stored.
N_HVG, N_PCS, PT_PCS = 5000, 50, 20

#: the kernel's three constants: the directed kNN support, the self-tuning bandwidth's
#: neighbour rank, and the lineage temperature.  See :func:`lineage_kernel`.
K_SUPPORT, K_SIGMA, BETA = 15, 7, 3.0

#: the **velocity** temperature, the third factor's counterpart to :data:`BETA`.  Both
#: factors are ``exp(temperature x score)`` over a score the data supplies — ``R`` in
#: ``[0, 1]`` for the lineage, ``cos(v_i, g_j - g_i)`` in ``[-1, 1]`` for the velocity — so
#: setting the two temperatures equal is what stops the comparison between them from being
#: decided by a constant chosen here.  It is deliberately *not* searched: a velocity factor
#: given its own tuned weight while the lineage factor keeps a fixed one would answer
#: "which factor tunes better", not "which factor carries more information".
NU = 3.0

#: the scVelo settings behind the velocity factor, and the artifact they are read from.
#: :mod:`scripts.experiments.itracer.velocity` builds the file from the ENA FASTQs; nothing
#: here can produce it, so its absence is an error naming that pipeline and never a silent
#: fallback to no velocity.
VELO_H5AD = "itracer_r2_velocity.h5ad"
VELO_HVG, VELO_PCS, VELO_KNN = 2000, 30, 30

#: the cut that turns the pseudotime into three equal marginals.
N_THIRDS = 3

#: an *unedited* scar target site: one integration, all CIGAR match, no indel.  A cell whose
#: every integration looks like this carries the barcode but was never cut, so its "scar"
#: names the wild-type sequence and is not a lineage mark.  See :func:`scar_is_edited`.
UNEDITED = re.compile(r"^\d+:\d+M$")

#: the authors' NPC/neuron score: mean log-expression of the neuron-side genes minus the
#: NPC-side genes of the Kanton et al. differential-expression table, filtered exactly as
#: their ``hindbrain_pt_lineage_comparison.r`` filters it.
DE_FILTER = dict(padj=0.01, logFC=np.log(1.2), pct_in=50.0, pct_out=20.0, auc=0.6)


# --------------------------------------------------------------------------- #
#  The release
# --------------------------------------------------------------------------- #
def load_itracer(cache_dir: str = CACHE_DIR, rebuild: bool = False):
    """The 29 274 microdissected cells as an ``AnnData`` of raw counts.

    Both regions and every cluster: the subsetting is :func:`load_cloud`'s job, and the
    counts of what R2 is being separated *from* are part of what the notebook reports.
    """
    import anndata as ad

    cache = os.path.join(cache_dir, "itracer_microdissected.h5ad")
    if rebuild or not os.path.exists(cache):
        _build_h5ad(cache, cache_dir)
    return ad.read_h5ad(cache)


def npc_neuron_genes(var_names, cache_dir: str = CACHE_DIR):
    """``(neuron_genes, npc_genes)`` — the two marker sets behind the NPC/neuron score."""
    import pandas as pd

    _fetch(cache_dir, ["Kanton_DE_NPC_neurons.tsv"])
    de = pd.read_csv(os.path.join(cache_dir, "Kanton_DE_NPC_neurons.tsv"), sep="\t")
    keep = ((de["padj"] < DE_FILTER["padj"]) & (de["logFC"] > DE_FILTER["logFC"])
            & (de["pct_in"] > DE_FILTER["pct_in"]) & (de["pct_out"] < DE_FILTER["pct_out"])
            & (de["auc"] > DE_FILTER["auc"]))
    present = np.asarray(var_names, dtype=str)
    return tuple(np.intersect1d(present, de.feature[keep & (de.group == g)].astype(str))
                 for g in ("neuron", "NPC"))


# --------------------------------------------------------------------------- #
#  The lineage address
# --------------------------------------------------------------------------- #
def scar_family(barcode, scar) -> np.ndarray:
    """``"<barcode>|<scar>"`` per cell, or ``""`` where either is missing.

    The scar string is only an address *within* its barcode — the same repair outcome
    arises independently on different barcodes — so the family is the pair, never the
    scar alone.
    """
    import pandas as pd

    b, s = pd.Series(barcode).astype(str), pd.Series(scar).astype(str)
    ok = pd.notna(barcode) & pd.notna(scar) & (b != "nan") & (s != "nan")
    return np.where(ok, b + "|" + s, "").astype(str)


def scar_is_edited(scar) -> np.ndarray:
    """Per cell: does the recorded scar actually carry an edit?

    ``ScarMerge`` is one CIGAR-like call per reporter integration, joined by ``_`` — so
    ``0:47M3D44M`` is a 3 bp deletion 47 bases in, ``0:42M14D49M_0:47M3D44M`` is a cell
    with two integrations each independently cut, and ``0:91M`` is a *read* target site
    that was never cut.  The last case is the one to watch: it is a scar *call*, it gets a
    family label like any other, and it is not lineage information — every uncut cell in
    the organoid carries the same wild-type sequence, so grouping on it groups strangers.
    A cell counts as edited here if **any** of its integrations departs from all-match.

    Returns a boolean array; cells with no scar call at all are ``False``.
    """
    s = np.asarray(scar, dtype=str)
    return np.array([bool(x) and x != "nan"
                     and not all(UNEDITED.match(p) for p in x.split("_"))
                     for x in s], dtype=bool)


def lineage_relatedness(barcode, scar) -> np.ndarray:
    """``R[i, j]`` in ``{0, 1/2, 1}`` — how close two cells are on the recorded tree.

    The authors' own two-level lineage distance (their
    ``hindbrain_pt_lineage_comparison.r``, ``dists_lineages``) is
    :math:`\\mathbf 1[\\mathrm{bf}_i \\ne \\mathrm{bf}_j] +
    \\mathbf 1[\\mathrm{sf}_i \\ne \\mathrm{sf}_j] \\in \\{0, 1, 2\\}`, and this is that
    distance read as a similarity, :math:`R = 1 - d/2`: same scar family 1, same barcode
    family but a different scar 1/2, unrelated 0.  Cells with no barcode call score 0
    against everything including each other — a missing barcode is missing evidence, not
    evidence of being unrelated to the barcoded cells, but it cannot be turned into an
    edge weight either.
    """
    b = np.asarray(barcode, dtype=str)
    f = scar_family(barcode, scar)
    known = f != ""
    same_b = (b[:, None] == b[None, :]) & known[:, None] & known[None, :]
    same_f = (f[:, None] == f[None, :]) & known[:, None] & known[None, :]
    return 0.5 * same_b.astype(np.float64) + 0.5 * same_f.astype(np.float64)


# --------------------------------------------------------------------------- #
#  The kernel
# --------------------------------------------------------------------------- #
def lineage_kernel(X, barcode, scar, k: int = K_SUPPORT, k_sigma: int = K_SIGMA,
                   beta: float = BETA, symmetric_support: bool = False,
                   C=None, nu: float = 0.0) -> tuple:
    """``(P, K_expr, K_lineage, K_velocity)`` — the lineage-informed transition matrix.

    .. math::

        P_{ij}\\ \\propto\\ \\underbrace{K_{\\mathrm{expr}}(x_i,x_j)}_{\\text{local}}\\cdot
        \\underbrace{K_{\\mathrm{lineage}}(i,j)}_{\\text{barcode + scar}}\\cdot
        \\underbrace{K_{\\mathrm{velocity}}(i,j)}_{\\text{RNA velocity}}

    Three factors, each a *temperature* on a score the data supplies, and every one of them
    switched off by setting its temperature to zero — at which point its factor is
    identically 1 and drops out of the product exactly.  That is what makes the ablations
    ablations: two ``P``s that differ in one temperature cannot differ in their support,
    their bandwidth, their directedness or their row normalisation.

    There is still **no pseudotime and no direction indicator** anywhere in this function —
    ``s`` is not an argument, so nothing ``P`` does can be an artefact of the axis the
    marginals were cut on.  The velocity factor is not that axis: it is read off spliced
    and unspliced counts, which the pseudotime never enters.

    ``K_expr`` is a self-tuning Gaussian, :math:`\\exp(-d_{ij}^2/\\sigma_i\\sigma_j)` with
    :math:`\\sigma_i` the distance from *i* to its ``k_sigma``-th neighbour, so the
    bandwidth widens where the cloud is sparse instead of one global :math:`\\varepsilon`
    having to fit both the ventricular zone and the neuron end.  Its support is
    **directed**: row *i* keeps *i*'s own ``k`` nearest neighbours and is not symmetrised,
    because symmetrising lets a late cell that happens to list an early one among its
    neighbours hand a long-range edge back — the exact shortcut a
    withhold-the-middle experiment is testing for.  ``symmetric_support=True`` restores
    the ``OR`` symmetrisation, and exists so that the notebook can measure what the
    asymmetry is worth rather than assert it.

    ``K_lineage`` is :math:`\\exp(\\beta R_{ij})` over :func:`lineage_relatedness`, so a
    same-scar pair is up-weighted :math:`e^{\\beta}` and an unrelated pair is left at 1.
    The term reweights edges; it never creates or vetoes one, and an unbarcoded cell is
    simply never up-weighted.

    ``K_velocity`` is :math:`\\exp(\\nu C_{ij})` over the velocity alignment ``C`` of
    :func:`velocity_alignment` — the cosine between *i*'s RNA-velocity vector and the
    displacement from *i* to *j* in expression space, in ``[-1, 1]``.  ``C = None`` (the
    default) means no velocity was supplied and the factor is exactly 1; so does
    ``nu = 0`` with a ``C`` present, which is the ablation.  Like the lineage term it
    reweights the existing support and cannot create an edge, but unlike it the term is
    signed: an edge pointing *against* the velocity is down-weighted rather than merely
    not up-weighted, which is the only asymmetry in the product.

    Rows are normalised.  Every row of ``K`` carries its own ``k`` neighbours and
    ``K_lineage`` is strictly positive, so no row can be empty and there are no absorbing
    cells to patch — the sink case only ever arose from the hard indicator this kernel no
    longer has.

    This is the scGESTALT granule kernel with one factor swapped and
    one dropped — the allele-Jaccard of a scGESTALT ``HMID`` replaced by the two-level
    barcode/scar address iTracer actually records, and the direction factor removed.
    """
    from scipy.spatial.distance import cdist

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    assert n == len(barcode) == len(scar), "X, barcode and scar must agree"
    assert k_sigma <= k < n, f"need k_sigma <= k < n, got {k_sigma}, {k}, {n}"

    d = cdist(X, X)
    np.fill_diagonal(d, np.inf)
    o = np.argsort(d, axis=1)
    sigma = d[np.arange(n), o[:, k_sigma - 1]]
    support = np.zeros((n, n), dtype=bool)
    support[np.arange(n)[:, None], o[:, :k]] = True
    if symmetric_support:
        support |= support.T
    K = np.exp(-d ** 2 / (sigma[:, None] * sigma[None, :]))
    K[~support] = 0.0
    np.fill_diagonal(K, 0.0)

    L = np.exp(beta * lineage_relatedness(barcode, scar))
    if C is None:
        V = np.ones_like(L)
    else:
        C = np.asarray(C, dtype=np.float64)
        assert C.shape == (n, n), f"velocity alignment is {C.shape}, the cloud is {n} cells"
        V = np.exp(nu * C)
    W = K * L * V
    row = W.sum(axis=1)
    assert (row > 0).all(), f"{int((row == 0).sum())} empty rows in W"
    return W / row[:, None], K, L, V


# --------------------------------------------------------------------------- #
#  The velocity factor
# --------------------------------------------------------------------------- #
def patch_scvelo_for_numpy2() -> None:
    """scvelo 0.3.4 is the last release and predates NumPy 2, which no longer coerces a
    size-1 array into a float slot.  The stochastic estimator does exactly that: with no
    offset the design matrix has one column, so ``inv(A'A)A'y`` comes back shaped ``(1,)``
    and ``gamma[i] = ...`` raises.  Every mode of ``scv.tl.velocity`` hits it, downgrading
    NumPy would break torch, and there is no newer scvelo to upgrade to — so the single
    branch we use is replaced by its closed form, which for one column is just
    ``<x, y> / <x, x>`` (and 0 where the column is empty, as ``pinv`` of 0 gives 0).  The
    other three branches unpack two or three values and are unaffected; they stay upstream's.
    """
    import warnings

    from scvelo.tools import optimization as _opt

    if getattr(_opt.leastsq_generalized, "_np2_patched", False):
        return
    _orig = _opt.leastsq_generalized

    def leastsq_generalized(x, y, x2, y2, res_std=None, res2_std=None,
                            fit_offset=False, fit_offset2=False, perc=None):
        if fit_offset or fit_offset2:
            return _orig(x, y, x2, y2, res_std, res2_std, fit_offset, fit_offset2, perc)

        # the preamble is upstream's, verbatim in effect: percentile weighting, then the
        # two moment systems stacked and scaled by their residual standard deviations.
        if perc is not None:
            if isinstance(perc, (list, tuple)):
                perc = perc[1]
            w = _opt.csr_matrix(
                _opt.get_weight(x, y, perc=perc) | _opt.get_weight(x, perc=perc)
            ).astype(bool)
            x, y = w.multiply(x).tocsr(), w.multiply(y).tocsr()

        n_var = x.shape[1]
        if res_std is None or res2_std is None:
            res_std = res2_std = np.ones(n_var)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = np.vstack((_opt.make_dense(x) / res_std, x2 / res2_std))
            Y = np.vstack((_opt.make_dense(y) / res_std, y2 / res2_std))

        xx = np.einsum("ij,ij->j", X, X)
        gamma = np.divide(np.einsum("ij,ij->j", X, Y), xx,
                          out=np.zeros(n_var), where=xx != 0).astype("float32")
        gamma[np.isnan(gamma)] = 0.0
        zero = np.zeros(n_var, dtype="float32")
        return zero, zero.copy(), gamma

    leastsq_generalized._np2_patched = True
    _opt.leastsq_generalized = leastsq_generalized
    # velocity.py did `from .optimization import leastsq_generalized`, so rebinding the
    # definition module alone would leave the caller pointing at the broken one.  Reach it
    # through sys.modules and not `from scvelo.tools import velocity`: the package's
    # __init__ rebinds that name to the *function* it re-exports, so the obvious import
    # hands back a callable and the patch would silently decorate it and do nothing.
    import sys
    _vel = sys.modules["scvelo.tools.velocity"]
    _vel.leastsq_generalized = leastsq_generalized
    assert _vel.leastsq_generalized is leastsq_generalized and callable(_vel.velocity)


def normalize_for_velocity(A, n_top_genes: int = VELO_HVG) -> None:
    """Gene filtering, count normalisation, log and HVG selection, in place.

    This is the *cloud-level* half of the velocity pipeline and it runs once, on all 2901
    cells — the same standing as the 50-component PCA every benchmark space is a prefix of
    (:func:`_build_cloud`), and for the same reason: it is an unsupervised choice of
    representation, it reads no marginal and no pseudotime, and refitting it per split
    would make the three columns three different gene sets.  Everything that depends on a
    *neighbour graph* is in :func:`fit_velocity` instead, and that half is rebuilt on
    whatever cells a training set is allowed to see.

    scvelo 0.3.4 stripped HVG selection and the log out of ``filter_and_normalize`` — it
    now only filters genes and normalises counts, and forwards anything else it does not
    recognise straight to ``normalize_per_cell``, where ``n_top_genes`` is a ``TypeError``.
    So the two steps it used to fold in are spelled out.  ``log1p`` touches ``X`` only, not
    the splice layers: scVelo's moments are computed from normalised-but-unlogged counts.
    """
    import scanpy as sc
    import scvelo as scv

    scv.pp.filter_and_normalize(A, min_shared_counts=20)
    sc.pp.log1p(A)
    sc.pp.highly_variable_genes(A, n_top_genes=n_top_genes, flavor="seurat", subset=True)


def fit_velocity(A, n_pcs: int = VELO_PCS, n_neighbors: int = VELO_KNN) -> None:
    """Moments and the stochastic velocity estimate on ``A`` alone, in place.

    The half of the pipeline that reads a neighbour graph: the PCA and kNN behind
    ``Ms``/``Mu``, and the per-gene :math:`\\gamma` fitted across the cells present.  Both
    are refitted on exactly the cells handed in, which is what lets
    :func:`velocity_alignment` be called on a training split without the withheld cells
    having smoothed the velocity of the cells that were kept.
    """
    import scvelo as scv

    patch_scvelo_for_numpy2()
    scv.pp.moments(A, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.tl.velocity(A, mode="stochastic")


def cloud_cell_ids(cache_dir: str = CACHE_DIR) -> np.ndarray:
    """``<reaction>_<barcode>`` for the candidate cells, in cloud row order."""
    cloud = load_cloud(cache_dir)
    A = load_itracer(cache_dir)
    return np.asarray(A.obs_names)[cloud.idx]


def load_velocity(cache_dir: str = CACHE_DIR):
    """The quantified cloud: 2901 cells in cloud row order, spliced/unspliced layers."""
    import anndata as ad

    path = os.path.join(cache_dir, VELO_H5AD)
    assert os.path.exists(path), (
        f"{path} is missing.  He et al. never released spliced/unspliced counts; the "
        f"velocity arms need them requantified from the ENA FASTQs first "
        f"(`bash scripts/sbatch/itracer_velocity_submit.sh`)")
    return ad.read_h5ad(path)


def velocity_alignment(idx=None, cache_dir: str = CACHE_DIR, rebuild: bool = False):
    """``C[i, j] = cos(v_i, g_j - g_i)`` over the cells ``idx``, in ``[-1, 1]``.

    The velocity factor's score, and CellRank's: how well the displacement from *i* to *j*
    in expression space agrees with the direction *i* is actually moving in, as read off
    its spliced/unspliced ratio.  ``+1`` is straight ahead of *i*, ``-1`` straight behind
    it, ``0`` sideways.  ``g`` is the smoothed spliced expression ``Ms`` and ``v`` the
    velocity, both in the HVG space of :func:`normalize_for_velocity`.

    **The refit is the point.**  scVelo is rerun on ``idx`` alone — its PCA, its kNN, the
    ``Ms``/``Mu`` smoothing and the per-gene :math:`\\gamma` — so when a sweep cell trains
    on 85 % of the shown cells, the velocities it steers by were not smoothed over the
    15 % it is scored against, and never over the withheld middle third under any
    protocol.  Reusing the whole-cloud fit would have handed the velocity factor a
    neighbourhood the lineage factor is explicitly denied, and the ablation between them
    would have measured that instead.

    Genes whose :math:`\\gamma` did not fit are dropped, because a NaN velocity coordinate
    would make every cosine involving that cell NaN.  Cached per cell subset: the sweep and
    the benchmark each use one, so the two fits are built by ``--prebuild`` and every array
    task reads them.
    """
    idx = np.arange(len(load_cloud(cache_dir).third)) if idx is None else \
        np.sort(np.asarray(idx, dtype=int))
    key = hashlib.sha1(idx.tobytes()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"itracer_velocity_align_{len(idx)}_{key}.npz")
    if os.path.exists(path) and not rebuild:
        return np.load(path)["C"].astype(np.float64)

    A = load_velocity(cache_dir)
    ids = cloud_cell_ids(cache_dir)
    assert A.n_obs == len(ids) and (np.asarray(A.obs_names) == ids).all(), (
        f"{VELO_H5AD} is not in cloud row order ({A.n_obs} cells against {len(ids)}); "
        f"rerun the assemble stage of scripts.experiments.itracer.velocity")
    A = A[idx].copy()
    fit_velocity(A)

    V = np.asarray(A.layers["velocity"], dtype=np.float64)
    G = np.asarray(A.layers["Ms"], dtype=np.float64)
    keep = np.isfinite(V).all(axis=0) & (V.std(axis=0) > 0)
    assert keep.sum() >= 100, f"only {int(keep.sum())} genes carry a velocity; refusing"
    V, G = V[:, keep], G[:, keep]

    # cos(v_i, g_j - g_i) = (v_i.g_j - v_i.g_i) / (||v_i|| ||g_j - g_i||), all n^2 at once.
    vg = V @ G.T
    num = vg - np.diag(vg)[:, None]
    gg = G @ G.T
    q = np.diag(gg).copy()
    d2 = np.maximum(q[:, None] + q[None, :] - 2.0 * gg, 0.0)
    den = np.sqrt(np.einsum("ij,ij->i", V, V))[:, None] * np.sqrt(d2)
    C = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    np.fill_diagonal(C, 0.0)
    np.clip(C, -1.0, 1.0, out=C)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path + ".tmp.npz", C=C.astype(np.float32),
                        idx=idx, n_genes=int(keep.sum()))
    os.replace(path + ".tmp.npz", path)
    print(f"[itracer] velocity alignment {C.shape} on {int(keep.sum())} genes "
          f"-> {os.path.basename(path)}")
    return C


# --------------------------------------------------------------------------- #
#  The derived cloud
# --------------------------------------------------------------------------- #
class Cloud:
    """The R2 neurogenic cells and everything the notebook plots them with.

    Attributes are per-*candidate*-cell unless the name says ``all``: ``X`` the
    :data:`N_PCS` principal components a benchmark space is a prefix of, ``umap`` their
    own 2-D embedding, ``s`` the
    diffusion pseudotime rescaled to ``(0, 1]``, ``third`` the marginal a cell falls in
    (0 = ``p_0``, 1 = withheld, 2 = ``p_1``), and ``barcode`` / ``scar`` / ``family`` the
    recorded address.  ``idx`` indexes back into the 29 274-cell ``AnnData``.

    ``s`` cuts the three marginals and places the withheld one on a model's time axis.  It
    is not an input to :func:`lineage_kernel` and never enters ``P``.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def marginal(self, b: int) -> np.ndarray:
        """Row indices of one third of the pseudotime, in cloud coordinates."""
        return np.flatnonzero(self.third == b)

    @property
    def shown(self) -> np.ndarray:
        """Row indices of ``p_0 u p_1`` — everything the withheld-middle kernel may see."""
        return np.flatnonzero(self.third != 1)


def load_cloud(cache_dir: str = CACHE_DIR, rebuild: bool = False) -> Cloud:
    """The candidate cells with their embedding, pseudotime and thirds, from cache."""
    # the name carries both widths, so the 20-PC cache the benchmark was built on cannot
    # be read as though it were the 50-PC one d = 50 needs.
    cache = os.path.join(cache_dir, f"itracer_r2_cg31_p{N_PCS}_t{PT_PCS}_g{N_HVG}.npz")
    if rebuild or not os.path.exists(cache):
        _build_cloud(cache, cache_dir)
    z = np.load(cache, allow_pickle=True)
    assert z["X"].shape[1] == N_PCS, (
        f"{cache} holds {z['X'].shape[1]} components, not N_PCS = {N_PCS}; "
        f"delete it and let load_cloud rebuild")
    return Cloud(**{k: z[k] for k in z.files},
                 diag={"n_all": int(z["score_all"].shape[0]), "cache": cache})


def _build_cloud(cache: str, cache_dir: str) -> None:
    import pandas as pd
    import scanpy as sc

    print(f"[itracer] building {os.path.basename(cache)} — a few minutes, once")
    A = load_itracer(cache_dir)
    obs = A.obs

    # 1. the embedding, on all 29 274 cells and in the paper's order: vst HVGs off the raw
    #    counts, then log-normalise, scale, PCA.  The candidate cells inherit these PCs
    #    rather than getting their own, which is what makes the pseudotime the authors'.
    sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, flavor="seurat_v3")
    hv = A.var.highly_variable.to_numpy().copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)

    neuron_genes, npc_genes = npc_neuron_genes(A.var_names, cache_dir)
    score_all = (np.asarray(A[:, neuron_genes].X.mean(axis=1)).ravel()
                 - np.asarray(A[:, npc_genes].X.mean(axis=1)).ravel())

    B = A[:, hv].copy()
    sc.pp.scale(B, max_value=10)
    sc.tl.pca(B, n_comps=N_PCS, svd_solver="arpack")
    pca_all = np.asarray(B.obsm["X_pca"], dtype=np.float64)
    sc.pp.neighbors(B, n_neighbors=15, n_pcs=PT_PCS)
    sc.tl.umap(B, random_state=0)
    umap_all = np.asarray(B.obsm["X_umap"], dtype=np.float64)

    # 2. the authors' three-step selection.
    region = obs["orig.ident"].to_numpy().astype(str)
    cluster = obs["RNA_snn_res.0.6"].to_numpy().astype(int)
    barcode = obs["GeneBarcodeMerge"].to_numpy(dtype=object)
    scar = obs["ScarMerge"].to_numpy(dtype=object)
    family = scar_family(barcode, scar)

    base = np.isin(region, R2_SAMPLES) & np.isin(cluster, CG3_1) & (family != "")
    big_b = pd.Series(np.asarray(barcode, dtype=str)[base]).value_counts()
    big_f = pd.Series(family[base]).value_counts()
    cand = (base
            & np.isin(np.asarray(barcode, dtype=str),
                      big_b.index[big_b > MIN_BARCODE_CELLS].to_numpy())
            & np.isin(family, big_f.index[big_f > MIN_SCAR_CELLS].to_numpy()))
    idx = np.flatnonzero(cand)

    # 3. the pseudotime, on the candidate cells' own diffusion map over the same 20 PCs,
    #    as the ranked first component.  destiny's sign is arbitrary and scanpy's is too,
    #    so it is fixed by the NPC/neuron score rather than by inspection.
    import anndata as ad
    C = ad.AnnData(X=np.zeros((len(idx), 1), dtype=np.float32))
    C.obsm["X_pca"] = pca_all[idx, :PT_PCS]
    sc.pp.neighbors(C, n_neighbors=15, use_rep="X_pca")
    sc.tl.diffmap(C, n_comps=15)
    sc.tl.umap(C, random_state=0)
    dc1 = np.asarray(C.obsm["X_diffmap"][:, 1], dtype=np.float64)
    s = pd.Series(dc1).rank().to_numpy() / len(idx)
    if np.corrcoef(s, score_all[idx])[0, 1] < 0:
        s = 1.0 + 1.0 / len(idx) - s
    edges = np.quantile(s, [1 / N_THIRDS, 2 / N_THIRDS])
    third = np.digitize(s, edges)

    np.savez_compressed(
        cache, idx=idx, X=pca_all[idx, :N_PCS], umap=np.asarray(C.obsm["X_umap"]),
        s=s, third=third, edges=edges, score=score_all[idx],
        barcode=np.asarray(barcode, dtype=str)[idx], scar=np.asarray(scar, dtype=str)[idx],
        family=family[idx], region=region[idx], cluster=cluster[idx],
        umap_all=umap_all, score_all=score_all, region_all=region, cluster_all=cluster,
        family_all=family, barcode_all=np.asarray(barcode, dtype=str),
        neuron_genes=neuron_genes, npc_genes=npc_genes)
    print(f"[itracer] {len(idx)} candidate cells, "
          f"{len(np.unique(family[idx]))} scar families, cut at s = "
          f"{edges[0]:.3f} / {edges[1]:.3f}")


# --------------------------------------------------------------------------- #
#  Fetching and assembling the release
# --------------------------------------------------------------------------- #
def _fetch(cache_dir: str, names=None) -> None:
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    for name in names or RELEASE:
        path = os.path.join(cache_dir, name)
        if os.path.exists(path) and os.path.getsize(path):
            continue
        print(f"[itracer] downloading {name} from Mendeley Data 10.17632/nj3p3pxv6p")
        urllib.request.urlretrieve(f"{MENDELEY}/{RELEASE[name]}/file_downloaded", path)


def _build_h5ad(cache: str, cache_dir: str) -> None:
    import anndata as ad
    import pandas as pd
    import scipy.io as sio
    import scipy.sparse as sp

    _fetch(cache_dir, ["counts.mtx.gz", "features.tsv.gz", "meta.tsv.gz"])
    print(f"[itracer] building {os.path.basename(cache)}")
    M = sio.mmread(os.path.join(cache_dir, "counts.mtx.gz")).tocsr()
    genes = pd.read_csv(os.path.join(cache_dir, "features.tsv.gz"),
                        header=None)[0].astype(str).to_numpy()
    meta = pd.read_csv(os.path.join(cache_dir, "meta.tsv.gz"), sep="\t", index_col=0)
    assert M.shape == (len(genes), len(meta)), (
        f"release disagrees with itself: {M.shape} vs {len(genes)} genes, {len(meta)} cells")

    adata = ad.AnnData(X=sp.csr_matrix(M.T, dtype=np.float32), obs=meta,
                       var=pd.DataFrame(index=pd.Index(genes)))
    adata.uns["source"] = "Mendeley Data 10.17632/nj3p3pxv6p, He et al. 2022"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    adata.write_h5ad(cache, compression="gzip")
