"""Verbatim transcription of the Curly-FM release's erythroid reference field + metrics.

The three numbers this benchmark reports — cosine distance, L2, and the column their
Table 4 heads "W2" — are whatever ``notebooks/2d_mouse_erythroid.ipynb`` cell 12
computes, so this module transcribes that cell rather than paraphrasing it and
the verification suite in the research repository diffs our vectorised evaluator against it on
random input.  Everything here is *their* code with their constants; the only edits are
ones the transcription forces.

Provenance, all in the Curly-FM release (https://github.com/kpetrovicc/curly-flow-matching, MIT):

* :func:`get_ut_knn_gaussian` — ``notebooks/2d_mouse_erythroid.ipynb`` cell 7, identical
  in ``notebooks/nd_mouse_erythroid.ipynb``.
* :func:`release_metrics` — cell 12's ``CurlyWrapperWithMetrics`` reduction plus the
  ``optimal_transport.wasserstein(pred, x1, power=1)`` call.
* :func:`release_wasserstein` — ``src/models/components/optimal_transport.py``'s
  ``wasserstein``, *both* of its branches, so the true W2 we report beside their column
  is computed by their code.

Three properties of their code are load-bearing and pinned by checks:

1. **The ×100.**  ``v_xt`` is multiplied by 100 before it is returned, which sets what L2
   means; it does not affect the cosine.
2. **The floor at 0.5.**  ``distance_factor = sigmoid(...)/2 + 0.5`` lies in [0.5, 1], so
   the returned field is at most half the kNN estimate even on a cell.  The appendix reads
   as if the weight were in [0, 1]; the released code halves it, and the released code is
   what the published numbers were measured on.
3. **``dist_thresh`` is an absolute length, 0.2**, not rescaled with the space.  In the
   unstandardised PCA of d = 20 / 50 nearest-neighbour distances are far larger, the
   sigmoid saturates, and the reference field degrades towards pure
   :math:`\\mathcal{N}(0, 0.01^2)` noise.  :func:`reference_field_diagnostics` measures
   that saturation per dimension.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

#: their ``k`` for the erythroid reference field (appendix §E.3: "k = 30 neighbors for
#: mouse erythroid experiments to compute ground-truth velocities").
REFERENCE_K = 30
#: cell 7 constants, unrenamed.
VELOCITY_SCALE = 100.0
#: ...except that the two notebooks disagree on it: ``2d_mouse_erythroid`` ends the kNN
#: estimate with ``* 100``, ``nd_mouse_erythroid`` with ``* 1``, and nothing else differs.
#: UMAP velocities are O(0.04) and PCA ones O(800), so the 2-D scale applied at d = 20 /
#: 50 would put L2 four orders of magnitude off their published column.
VELOCITY_SCALE_BY_DIM = {2: 100.0, 20: 1.0, 50: 1.0}
DIST_THRESH = 0.2
SIGMOID_SHARPNESS = 100.0
NOISE_STD = 0.01


def velocity_scale_for_dim(dim: int) -> float:
    """Which notebook's ``v_xt`` scale applies in a given space."""
    assert dim in VELOCITY_SCALE_BY_DIM, (
        f"d = {dim} matches neither released notebook; add its scale explicitly rather "
        f"than defaulting, since the two differ by 100x")
    return VELOCITY_SCALE_BY_DIM[dim]


# --------------------------------------------------------------------------- #
#  Their cell 7, transcribed
# --------------------------------------------------------------------------- #
def get_ut_knn_gaussian(
    xt: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    k: int = 100,
    eps: float = 1e-12,
    velocity_scale: float = VELOCITY_SCALE,
):
    """``notebooks/2d_mouse_erythroid.ipynb`` cell 7, unmodified.

    Do not tidy this.  The default ``k = 100`` is theirs and is always overridden by the
    caller; ``w`` is rebound from the softmax weights to the sigmoid sharpness
    mid-function, which is theirs too.  ``velocity_scale`` is the single line on which the
    2-D and n-D copies differ, so it is a parameter rather than a constant.
    """
    x = torch.cat([x0, x1], dim=0)
    v = torch.cat([v0, v1], dim=0)

    dists = torch.cdist(xt, x)

    knn_dists, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)

    h = knn_dists[:, -1:].clamp_min(eps)
    w = torch.exp(-(knn_dists**2) / (2 * h**2))
    w = w / (w.sum(dim=1, keepdim=True) + eps)

    v_knn = v[knn_idx]
    v_xt = (w.unsqueeze(-1) * v_knn).sum(dim=1) * velocity_scale

    distance_factor = knn_dists[:, :1]
    w = 100
    dist_thresh = 0.2
    distance_factor = (torch.nn.functional.sigmoid((distance_factor - dist_thresh) * w) / 2) + 0.5
    noise_vector = torch.randn_like(v_xt) * 0.01
    v_xt = (1 - distance_factor) * v_xt + distance_factor * noise_vector

    return v_xt, knn_dists


# --------------------------------------------------------------------------- #
#  Their OT utility, transcribed with both branches
# --------------------------------------------------------------------------- #
def release_wasserstein(x0: torch.Tensor, x1: torch.Tensor, power: int = 2) -> float:
    """``optimal_transport.wasserstein`` from the release, exact-EMD branch.

    At ``power=1`` the cost is the plain Euclidean distance and the returned value is the
    EMD itself; at ``power=2`` the cost is squared and the square root is taken.  The
    erythroid notebooks call the first while heading the column "W2".  Both branches are
    kept so the true W2 we report alongside their column is the same code path they would
    have taken.  ``numItermax=1e7`` and the uniform marginals are theirs.
    """
    import ot as pot

    assert power in (1, 2), f"their assert allows power 1 or 2, got {power}"
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = float(pot.emd2(a, b, M.detach().cpu().numpy(), numItermax=int(1e7)))
    if power == 2:
        ret = math.sqrt(ret)
    return ret


# --------------------------------------------------------------------------- #
#  Their cell 12 reduction, transcribed
# --------------------------------------------------------------------------- #
def release_metrics(pred: torch.Tensor, v_pred: torch.Tensor, u_t: torch.Tensor,
                    target: torch.Tensor) -> dict[str, float]:
    """The three reported statistics, reduced exactly as cell 12 reduces them.

    ``cos_dist`` and ``L2`` come out of ``CurlyWrapperWithMetrics.forward``:

        cos_dist   = 1 - cosine_similarity(u_t, x_dot, dim=1)
        L2_squared = sum((u_t - x_dot) ** 2, dim=1)

    and are the *per-step* quantities.  What their table reports is **not** this: see
    :func:`release_path_metrics`.  This function is the pointwise reduction, used by the
    path version at each step and by the checks.

    The third is ``optimal_transport.wasserstein(pred, x1, power=1)``: an exact EMD
    against the unsquared Euclidean cost, reported here as ``w2`` under their heading
    because renaming it would make our column incomparable with their block.  ``w2_true``
    is the same distance at ``power=2``, i.e. the quantity their heading names, and it
    exists only for the rows we ran.
    """
    assert pred.shape == v_pred.shape == u_t.shape, (
        f"pred {tuple(pred.shape)}, v_pred {tuple(v_pred.shape)}, "
        f"u_t {tuple(u_t.shape)} must agree")
    assert target.shape[1] == pred.shape[1], "target dimension must match pred"

    cos_dist = (1 - F.cosine_similarity(u_t, v_pred, dim=1)).mean().item()
    l2 = torch.sum((u_t - v_pred) ** 2, dim=1).mean().item()

    return {"cos_dist": cos_dist, "l2": l2,
            "w2": release_wasserstein(pred, target, power=1),
            "w2_true": release_wasserstein(pred, target, power=2)}


def release_path_metrics(states, field, x0, x1, v0, v1, k: int, dt: float,
                         velocity_scale: float, target: torch.Tensor,
                         seed: int = 0) -> dict[str, float]:
    """Their Table 4 quantities: ``cos_dist`` and ``L2`` **integrated along the path**.

    ``CurlyWrapperWithMetrics.forward`` returns ``cat([x_dot, cos_dist, L2_squared])`` as
    the value of the vector field, and the wrapper is handed to ``NeuralODE.trajectory``
    with ``z0 = cat([x0, zeros(n, 2)])``, so the solver treats the last two channels as
    derivatives of extra state components starting at zero.  What ``cosine_traj[-1]``
    holds is ``∫₀^{1/2} (1 - cos(u_t, v_θ)) dt``, not the cosine distance at t = 1/2, and
    reading it as an endpoint value would report roughly twice their number.

    Their solver is ``euler`` over ``linspace(0, 1/2, 100)``, a left-endpoint Riemann sum;
    this reproduces that sum over our own integrator's step grid.  ``field`` is
    ``f(t, x)`` — for Path B the full SDE drift, since that is what moved the samples.
    ``target`` is scored once, at the endpoint.
    """
    torch.manual_seed(seed)
    cos_acc, l2_acc = 0.0, 0.0
    u_norms = []
    for i, x in enumerate(states[:-1]):
        t = torch.full((x.shape[0], 1), i * dt, dtype=x.dtype, device=x.device)
        with torch.no_grad():
            u_t, _ = get_ut_knn_gaussian(x, x0, x1, v0, v1, k=k,
                                         velocity_scale=velocity_scale)
            v_pred = field(t, x)
            cos_acc += float((1 - F.cosine_similarity(u_t, v_pred, dim=1)).mean()) * dt
            l2_acc += float(torch.sum((u_t - v_pred) ** 2, dim=1).mean()) * dt
        u_norms.append(float(u_t.norm(dim=1).mean()))

    pred = states[-1]
    return {"cos_dist": cos_acc, "l2": l2_acc,
            "w2": release_wasserstein(pred, target, power=1),
            "w2_true": release_wasserstein(pred, target, power=2),
            "u_norm_mean": float(np.mean(u_norms))}


# --------------------------------------------------------------------------- #
#  How much of the reference field survives in each space
# --------------------------------------------------------------------------- #
def reference_field_diagnostics(xt: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor,
                                v0: torch.Tensor, v1: torch.Tensor,
                                k: int = REFERENCE_K) -> dict[str, float]:
    """How far ``get_ut_knn_gaussian`` has degraded to noise at these query points.

    ``distance_factor`` is the weight the released code puts on
    :math:`\\mathcal{N}(0, 0.01^2)` rather than on the kNN velocity estimate.  It is
    bounded below by 0.5 and driven to 1 by the fixed absolute ``dist_thresh = 0.2``, so
    in a space whose nearest-neighbour distances exceed 0.2 the field every method is
    scored against is mostly noise.  Returns the mean weight, the saturated fraction, and
    the median first-neighbour distance that drives it.
    """
    x = torch.cat([x0, x1], dim=0)
    d1 = torch.cdist(xt, x).topk(k=k, dim=1, largest=False).values[:, 0]
    factor = torch.sigmoid((d1 - DIST_THRESH) * SIGMOID_SHARPNESS) / 2 + 0.5
    return {
        "noise_weight_mean": float(factor.mean()),
        "noise_weight_saturated_frac": float((factor > 0.99).float().mean()),
        "nn_dist_median": float(d1.median()),
        "signal_weight_mean": float(1.0 - factor.mean()),
    }


def signal_to_noise(u_t: torch.Tensor) -> float:
    """Crude check that a reference field carries anything: ‖u‖ against the noise floor.

    A pure-noise field has expected norm ``NOISE_STD * sqrt(d)``, so a ratio near 1 means
    the cosine distance in that column is measuring alignment against nothing.
    """
    d = u_t.shape[1]
    floor = NOISE_STD * math.sqrt(d)
    return float(u_t.norm(dim=1).mean() / floor)


def as_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
