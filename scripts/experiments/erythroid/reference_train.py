"""Verbatim transcription of the Curly-FM release's erythroid **training** cells.

:mod:`~scripts.experiments.erythroid.reference_release` transcribes how the reference *scores* a
model; this module transcribes how it *fits* one.  Together they are the reference
pipeline end to end, and running it on our cache is what separates three things E1's two
blocks otherwise conflate — the preprocessing, the trainer, and the method.

What this buys, stated as the three-line decomposition it exists to measure::

    published (their code, their data)  ->  their code, OUR data     =  data effect
    their code, our data                ->  shared harness, their hp =  pipeline effect
    shared harness, matched search      ->  ours                     =  the method

Only the last line is a claim about methods.  Without this module the whole gap sits in
one number and every part of it can be attributed to whatever the reader prefers.

Provenance
----------
Everything below is their release's ``notebooks/{2d,nd}_mouse_erythroid.ipynb``, cell
numbers given per function.  The two notebooks are the same code with different constants
**except** for the geodesic loss, where they also differ in structure.  Their ``MLP`` and
their ``ExactOptimalTransportConditionalFlowMatcher`` are imported from ``torchcfm``, not
re-implemented.

Edits the transcription forces, and nothing else: ``device``, ``k``, ``dim`` and
``velocity_scale`` become arguments; ``X``/``V`` are lists of two arrays rather than a
``np.stack``, which would require the marginals to be equal-length and they are not under
the sweep's 85 % split; the seeds are set per call rather than once; the ``tqdm`` /
``plt`` / timing lines are dropped.

Kept exactly as written, because a tidy-up here would silently change the numbers:

1. **The double noise in** :func:`get_batch_geo` — :func:`get_xt` already applied
   ``sqrt(t(1-t)) sigma eps`` and the caller adds the same term again.
2. **The cosine weight is 50 at d = 2 and 0 at d = 20/50.**  The nd notebook writes the
   term and multiplies it by zero, so at d > 2 the geodesic net is a pure magnitude
   regression onto ``alpha * ut``.  The paper describes one loss.
3. **``alpha`` multiplies ``ut``, not ``mu_t_dot``.**  Our engine's adapter
   (:mod:`scripts.core.curly_fm`) puts it on the other factor, so the two are not the
   same parameter and their grids are not interchangeable.
4. **``knn_loss`` is computed and discarded** (``+ 0 * knn_loss``), hinge included.
5. **The reference field at training time is built from the minibatch**, so their pool is
   512 points wide and resampled every iteration — a different estimator from the same
   function called on the full marginals at evaluation time.

None of these arms sees ``P``, the Finsler geometry, or the withheld marginal: the entry
points take positions and velocities of the two *kept* marginals and nothing else, which
the verification suite in the research repository pins by signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize  # their call site reads ``scipy.optimize.linear_sum_assignment``
import torch
from torch.func import vmap

from scripts.core.arms import ArmResult
from scripts.core.train import integrate_flow

from .reference_release import REFERENCE_K, get_ut_knn_gaussian, velocity_scale_for_dim

# their MLP, via torchcfm (the notebooks' own import), falling back to our vendored copy
# of the release's byte-identical ``MLP`` so this module imports without either package.
try:
    from torchcfm.conditional_flow_matching import \
        ExactOptimalTransportConditionalFlowMatcher
    from torchcfm.models import MLP
except ImportError:  # pragma: no cover - exercised only without torchcfm installed
    from scripts.core.vendor.curly_release import MLP  # noqa: F401
    ExactOptimalTransportConditionalFlowMatcher = None


#: The two arms this module provides.  They are *not* engine arms and never enter
#: :data:`scripts.core.arms.ARMS`.
RELEASE_ARMS = ("cfm_release", "curly_release")

#: their ``train_ts``.  Two kept marginals, so every ``for t_start in range(len(ts)-1)``
#: below runs exactly once; the loops are kept because they are theirs.
TRAIN_TS = [0, 2]


@dataclass(frozen=True)
class ReleaseSettings:
    """One notebook's hyper-parameters, transcribed.

    Every field is a literal from a numbered cell; nothing is inferred or carried over
    from the paper's Table 12, which disagrees with the release on the widths and on the
    geodesic iteration count.  The release is what the published numbers were measured on.
    """

    notebook: str
    cfm_width: int
    cfm_iters: int
    geo_width: int
    geo_iters: int
    vel_width: int
    vel_iters: int
    batch_size: int
    lr: float
    sigma: float
    geo_alpha: float
    vel_alpha: float
    cosine_weight: float
    hinge_value: float
    k: int
    velocity_scale: float


#: ``2d_mouse_erythroid.ipynb`` cells 12/18/19/21/22 and ``nd_mouse_erythroid.ipynb``
#: cells 15/16/21/22/24/25.  d = 20 and d = 50 are the same notebook run twice.
RELEASE_SETTINGS: dict[int, ReleaseSettings] = {
    2: ReleaseSettings(
        notebook="2d_mouse_erythroid.ipynb",
        cfm_width=64, cfm_iters=3000,
        geo_width=64, geo_iters=2000,
        vel_width=64, vel_iters=3000,
        batch_size=256, lr=1e-4, sigma=0.01,
        geo_alpha=0.1, vel_alpha=1.0,
        cosine_weight=50.0,          # cell 19: "+ 50*cosine_loss"
        hinge_value=0.1, k=REFERENCE_K, velocity_scale=100.0),
    20: ReleaseSettings(
        notebook="nd_mouse_erythroid.ipynb",
        cfm_width=128, cfm_iters=3000,
        geo_width=256, geo_iters=3000,
        vel_width=256, vel_iters=3000,
        batch_size=256, lr=1e-4, sigma=0.01,
        geo_alpha=0.1, vel_alpha=1.0,
        cosine_weight=0.0,           # cell 22: "+ 0*cosine_loss"
        hinge_value=0.1, k=REFERENCE_K, velocity_scale=1.0),
}
RELEASE_SETTINGS[50] = RELEASE_SETTINGS[20]


def release_settings(dim: int) -> ReleaseSettings:
    assert dim in RELEASE_SETTINGS, (
        f"d = {dim} matches no released notebook; add its cells explicitly rather than "
        f"reusing another dimension's, since the two notebooks differ in the loss")
    s = RELEASE_SETTINGS[dim]
    assert s.velocity_scale == velocity_scale_for_dim(dim), (
        "the training-time field scale must be the evaluation-time one: both are the "
        "same notebook's get_ut_knn_gaussian")
    return s


class ReleaseVelAdapter(torch.nn.Module):
    """Their ``model(cat([x, t], -1))`` behind our ``v_net(t, x)`` calling convention.

    The only wrapper in this module: ``integrate_flow`` and ``evaluate`` both call
    ``f(t, x)``, and the tensor handed to the MLP is byte-for-byte what their
    ``torch.cat([xt, t[:, None]], dim=-1)`` produces.
    """

    def __init__(self, mlp: torch.nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], 1)
        return self.mlp(torch.cat([x, t], dim=-1))


# --------------------------------------------------------------------------- #
#  OT-CFM — their cells 9/12/13 (2d), 12/15/16 (nd)
# --------------------------------------------------------------------------- #
def get_batch(FM_module, X, batch_size, n_times, ts_train, device, return_noise=False):
    """``nd_mouse_erythroid.ipynb`` cell 12, unmodified but for ``device``.

    ``n_times`` is passed and never read; that is theirs.  With ``ts_train = [0, 2]`` the
    loop body runs once, ``t_start = 0``, so ``t`` comes back on [0, 1].
    """
    ts = []
    xts = []
    uts = []
    noises = []
    for t_start in range(len(ts_train) - 1):
        t_end = t_start + 1

        x0 = (
            torch.from_numpy(
                X[t_start][np.random.randint(X[t_start].shape[0], size=batch_size)]
            )
            .float()
            .to(device)
        )
        x1 = (
            torch.from_numpy(
                X[t_end][np.random.randint(X[t_end].shape[0], size=batch_size)]
            )
            .float()
            .to(device)
        )

        if return_noise:
            t, xt, ut, eps = FM_module.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
        else:
            t, xt, ut = FM_module.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )

        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)

    t = torch.cat(ts)

    xt = torch.cat(xts)
    ut = torch.cat(uts)
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut


def train_ot_cfm_release(x_train, s: ReleaseSettings, dim: int, seed: int,
                         device, dtype, log_every: int = 500):
    """Their cells 15 + 16: minibatch exact-OT CFM, 3000 iterations, Adam at 1e-4.

    ``sigma`` here is torchcfm's conditional-path noise, not a bridge width.  The coupling
    is an exact EMD over the 256-point minibatch, recomputed every iteration — a different
    object from our engine's ``cfm`` arm, which solves one entropic Sinkhorn problem over
    up to 600 endpoints and samples from it.
    """
    assert ExactOptimalTransportConditionalFlowMatcher is not None, (
        "torchcfm is required for the cfm_release arm: pip install torchcfm")
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = s.batch_size
    sigma = s.sigma
    ot_cfm_model = MLP(dim=dim, time_varying=True, w=s.cfm_width).to(device)
    ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), s.lr)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    loss_otcfm = []
    for i in range(s.cfm_iters):
        ot_cfm_optimizer.zero_grad()
        t, xt, ut = get_batch(FM, x_train, batch_size, None, TRAIN_TS, device)
        vt = ot_cfm_model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        ot_cfm_optimizer.step()

        if i % 100 == 0:
            loss_otcfm.append(loss.cpu().item())
        if log_every and i % log_every == 0:
            print(f"    [release-otcfm] iter {i:5d}  L={float(loss):.5f}")

    diag = {"iters": s.cfm_iters, "width": s.cfm_width, "lr": s.lr, "sigma": sigma,
            "batch_size": batch_size, "loss_tail": float(np.mean(loss_otcfm[-5:])),
            "coupling": "torchcfm ExactOptimalTransportConditionalFlowMatcher",
            "notebook": s.notebook}
    return ReleaseVelAdapter(ot_cfm_model).eval(), diag


# --------------------------------------------------------------------------- #
#  Curly-FM — their cells 16/17/19/20/22 (2d), 19/20/22/23/25 (nd)
# --------------------------------------------------------------------------- #
def get_xt(t, t_start, x0, x1, geodesic_model, sigma=0.0):
    """``nd_mouse_erythroid.ipynb`` cell 19, unmodified."""
    mu_t = (1 - t) * x0 + t * x1 + t * (1-t) * (geodesic_model(torch.cat([x0, x1, t+t_start], dim=-1)))
    epsilon = torch.randn_like(x0)
    x_t = mu_t + torch.sqrt(t*(1-t))*sigma * epsilon
    return mu_t, x_t, epsilon


def get_xt_xt_dot(t, t_start, t_end, x0, x1, geodesic_model, sigma=0.0):
    """``nd_mouse_erythroid.ipynb`` cell 19, unmodified.

    ``t_end`` is accepted and unused, as in theirs.  The per-coordinate ``autograd.grad``
    loop is their way of getting ``d mu_t / dt``.
    """
    with torch.enable_grad():
        t = t[..., None]
        t.requires_grad_(True)
        mu_t, xt, eps = get_xt(t, t_start, x0, x1, geodesic_model, sigma=sigma)
        mu_t_dot_list = []
        for i in range(xt.shape[-1]):
            mu_t_dot_list.append(
                torch.autograd.grad(torch.sum(mu_t[..., i]), t, create_graph=True)[0]
            )
        mu_t_dot = torch.cat(mu_t_dot_list, -1)
    return xt, mu_t_dot, eps


def coupling_geo_new(t_start, t_end, x0, x1, x0s, x1s, v0s, v1s, geodesic_model, k,
                     sigma, velocity_scale):
    """``nd_mouse_erythroid.ipynb`` cell 19, unmodified but for ``velocity_scale``.

    Curly-FM's transport plan: pair the batch so that the interpolant's own velocity
    disagrees least with the reference field, by exact linear assignment on the all-pairs
    cost.  This is the coupling that *is* the method, so the harness does not equalise it.
    """
    batch_size, d = x0.shape

    t = torch.rand(1).type_as(x0) * torch.ones((batch_size, batch_size), device=x0.device)
    x0_r = x0.repeat(batch_size, 1, 1)
    x1_r = x1.repeat(batch_size, 1, 1).transpose(0, 1)
    xt, mu_t_dot, eps = get_xt_xt_dot(
        t, t_start, t_end, x0_r, x1_r, geodesic_model, sigma
    )

    ut = vmap(lambda x: get_ut_knn_gaussian(x, x0s, x1s, v0s, v1s, k=k,
                                            velocity_scale=velocity_scale)[0],
              randomness="different")(xt)

    L2_cost = 0.5 * ((mu_t_dot.detach() - ut) ** 2).sum(-1)
    _, j = scipy.optimize.linear_sum_assignment(L2_cost.detach().cpu().numpy())

    pi_x0 = x0[j]
    pi_x1 = x1

    return pi_x0, pi_x1, eps


def get_batch_geo(geo, X, V, batch_size, sigma, ts_train, k, device, velocity_scale):
    """``nd_mouse_erythroid.ipynb`` cell 20, unmodified but for ``k``/``device``/scale.

    Note the ``xt = xt + sqrt(t(1-t)) sigma eps`` line: :func:`get_xt` already applied
    that displacement, so the returned ``xt`` carries it twice.  Theirs.
    """
    ts = []
    t_orig_list = []
    mu_t_dots = []
    eps_list = []
    xts = []
    uts = []
    dists = []
    for t_start in range(len(ts_train) - 1):
        t_end = t_start + 1

        idcs_0 = np.random.randint(X[t_start].shape[0], size=batch_size)
        idcs_1 = np.random.randint(X[t_end].shape[0], size=batch_size)

        x0 = torch.from_numpy(X[t_start][idcs_0]).float().to(device)
        x1 = torch.from_numpy(X[t_end][idcs_1]).float().to(device)

        v0 = torch.from_numpy(V[t_start][idcs_0]).float().to(device)
        v1 = torch.from_numpy(V[t_end][idcs_1]).float().to(device)

        t = torch.rand(x0.shape[0]).type_as(x0)
        t_o = t

        xt, mu_t_dot, eps = get_xt_xt_dot(t, t_start, t_end, x0, x1, geo, sigma=sigma)
        ut, dist = get_ut_knn_gaussian(
            xt,
            x0,
            x1,
            v0,
            v1,
            k=k,
            velocity_scale=velocity_scale,
        )
        xt = xt + torch.sqrt(t * (1-t)).unsqueeze(1) * sigma * eps

        t_orig_list.append(t_o)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
        mu_t_dots.append(mu_t_dot)
        eps_list.append(eps)
        dists.append(dist)

    t_orig = torch.cat(t_orig_list)
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    mu_t_dot = torch.cat(mu_t_dots)
    eps = torch.cat(eps_list)
    dist = torch.cat(dists)

    return t_orig, t, xt, ut, mu_t_dot, eps, dist


def get_batch_vel(geo, X, V, batch_size, sigma, ts_train, k, device, velocity_scale):
    """``nd_mouse_erythroid.ipynb`` cell 23, unmodified but for ``k``/``device``/scale.

    The reference pool handed to the coupling is the batch itself
    (``coupling_geo_new(..., x0, x1, v0, v1, ...)``), i.e. 512 kept cells redrawn every
    iteration — never the full cloud and never the withheld marginal.
    """
    ts = []
    t_orig_list = []
    mu_t_dots = []
    eps_list = []
    xts = []
    for t_start in range(len(ts_train) - 1):
        t_end = t_start + 1

        idcs_0 = np.random.randint(X[t_start].shape[0], size=batch_size)
        idcs_1 = np.random.randint(X[t_end].shape[0], size=batch_size)

        x0 = torch.from_numpy(X[t_start][idcs_0]).float().to(device)
        x1 = torch.from_numpy(X[t_end][idcs_1]).float().to(device)

        v0 = torch.from_numpy(V[t_start][idcs_0]).float().to(device)
        v1 = torch.from_numpy(V[t_end][idcs_1]).float().to(device)

        t = torch.rand(x0.shape[0]).type_as(x0)
        t_o = t

        x0, x1, _ = coupling_geo_new(t_start, t_end, x0, x1, x0, x1, v0, v1, geo, k,
                                     sigma=sigma, velocity_scale=velocity_scale)
        xt, mu_t_dot, eps = get_xt_xt_dot(t, t_start, t_end, x0, x1, geo, sigma=sigma)

        t_orig_list.append(t_o)
        ts.append(t + t_start)
        xts.append(xt)
        mu_t_dots.append(mu_t_dot)
        eps_list.append(eps)

    t_orig = torch.cat(t_orig_list)
    t = torch.cat(ts)
    xt = torch.cat(xts)
    mu_t_dot = torch.cat(mu_t_dots)
    eps = torch.cat(eps_list)

    return t_orig, t, xt, mu_t_dot, eps


def train_curly_release(x_train, v_train, s: ReleaseSettings, dim: int, seed: int,
                        device, dtype, log_every: int = 500):
    """Their cells 21 + 22 (geodesic) and 24 + 25 (flow), in that order.

    Stage 1 fits ``phi`` so the interpolant's velocity matches ``alpha * u``; stage 2
    freezes nothing, as theirs does not — ``mu_t_dot.detach()`` is what stops the
    gradient.  The geodesic loss is the one place the two notebooks differ structurally,
    so it is written once with ``s.cosine_weight`` carrying the difference.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = s.batch_size
    sigma = s.sigma
    alpha = s.geo_alpha
    k = s.k
    vs = s.velocity_scale

    geo_model = MLP(dim=dim*2, out_dim=dim, time_varying=True, w=s.geo_width).to(device)
    geo_optimizer = torch.optim.AdamW(geo_model.parameters(), s.lr)

    train_loss = []
    for i in range(s.geo_iters):
        geo_optimizer.zero_grad()
        t_orig, t, xt, ut, mu_t_dot, eps, knn_dist = get_batch_geo(
            geo_model, x_train, v_train, batch_size, sigma, TRAIN_TS, k, device, vs)

        hinge_value = s.hinge_value
        knn_dist[knn_dist < hinge_value] = hinge_value
        knn_loss = torch.mean(knn_dist)

        cosine_loss = 1 - torch.nn.functional.cosine_similarity(ut, mu_t_dot).mean()
        loss = (1 * torch.mean((alpha*ut - mu_t_dot) ** 2)
                + s.cosine_weight * cosine_loss + 0 * knn_loss)

        loss.backward()
        geo_optimizer.step()

        if i % 100 == 0:
            train_loss.append(loss.cpu().item())
        if log_every and i % log_every == 0:
            print(f"    [release-curly-geo] iter {i:5d}  L={float(loss):.5f}")

    vel_model = MLP(dim=dim, time_varying=True, w=s.vel_width).to(device).to(device)
    vel_optimizer = torch.optim.Adam(vel_model.parameters(), s.lr)

    vel_loss = []
    for i in range(s.vel_iters):
        vel_optimizer.zero_grad()

        t_orig, t, xt, mu_t_dot, eps = get_batch_vel(
            geo_model, x_train, v_train, batch_size, sigma, TRAIN_TS, k, device, vs)
        vt = vel_model(torch.cat([xt.detach(), t[:, None]], dim=-1))

        loss = torch.mean((vt - mu_t_dot.detach()) ** 2)

        loss.backward()
        vel_optimizer.step()

        if i % 100 == 0:
            vel_loss.append(loss.cpu().item())
        if log_every and i % log_every == 0:
            print(f"    [release-curly-vel] iter {i:5d}  L={float(loss):.6f}")

    diag = {"geo_iters": s.geo_iters, "vel_iters": s.vel_iters,
            "geo_width": s.geo_width, "vel_width": s.vel_width, "lr": s.lr,
            "sigma": sigma, "alpha": alpha, "cosine_weight": s.cosine_weight,
            "batch_size": batch_size, "k": k, "velocity_scale": vs,
            "loss_geo_tail": float(np.mean(train_loss[-5:])),
            "loss_vel_tail": float(np.mean(vel_loss[-5:])),
            "coupling": "coupling_geo_new (drift-discrepancy, exact assignment)",
            "notebook": s.notebook}
    return ReleaseVelAdapter(vel_model).eval(), diag


# --------------------------------------------------------------------------- #
#  Entry point used by the driver
# --------------------------------------------------------------------------- #
def run_release_arm(arm: str, X0: np.ndarray, X1: np.ndarray, V0: np.ndarray,
                    V1: np.ndarray, dim: int, seed: int, device, dtype,
                    n_steps: int, smoke: bool = False,
                    log_every: int = 500) -> ArmResult:
    """Train one release arm and push the source marginal forward, as an ``ArmResult``.

    The signature takes **four arrays** — the positions and velocities of the two kept
    marginals — rather than a ``TrainingSet``, so an arm that cannot reach ``P``, the
    geometry or the withheld marginal cannot accidentally read them.
    the verification suite in the research repository asserts the signature, not the body.

    ``smoke`` cuts every loop to 40 iterations for a wiring check.
    """
    assert arm in RELEASE_ARMS, f"unknown release arm {arm!r}"
    assert X0.shape[1] == X1.shape[1] == dim, (X0.shape, X1.shape, dim)
    assert V0.shape == X0.shape and V1.shape == X1.shape

    s = release_settings(dim)
    if smoke:
        from dataclasses import replace as _replace
        s = _replace(s, cfm_iters=40, geo_iters=40, vel_iters=40, batch_size=64)

    x_train = [np.ascontiguousarray(X0, dtype=np.float32),
               np.ascontiguousarray(X1, dtype=np.float32)]
    v_train = [np.ascontiguousarray(V0, dtype=np.float32),
               np.ascontiguousarray(V1, dtype=np.float32)]

    if arm == "cfm_release":
        v_net, diag = train_ot_cfm_release(x_train, s, dim, seed, device, dtype,
                                           log_every=log_every)
    else:
        v_net, diag = train_curly_release(x_train, v_train, s, dim, seed, device, dtype,
                                          log_every=log_every)

    src = torch.as_tensor(x_train[0], dtype=dtype, device=device)
    with torch.no_grad():
        traj = integrate_flow(v_net, src, n_steps=n_steps)

    diag.update({"provenance": "release code, our UniTVelo cache",
                 "reads_P": False, "reads_geometry": False, "n_source": int(len(src))})
    return ArmResult(arm, v_net, traj.cpu().numpy(), diag, {})
