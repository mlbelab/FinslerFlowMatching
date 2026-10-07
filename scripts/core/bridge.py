"""Path B — Noisy Schrödinger Bridge Matching.

Two networks are trained over the optimal coupling: a drift field ``v_θ(t, x)`` and
a score field ``s_φ(t, x)``.  The bridge conditionals are *time-locked* to the
frozen Finsler geodesic backbone φ_{η*} produced by Path-A Phase 1: the metric
tensor is evaluated strictly as a function of time along the deterministic mean, so
the conditional Fokker–Planck geometry is a pure function of t (per endpoint pair).

Metric vs mobility
------------------
Two distinct tensors are built from the same Σ(x), and keeping them apart is the
whole point of this module:

    G_t = ( ρ I_d + Σ(x_{t,η*}) )^{-1}     the *metric* / cost tensor  (Path A)
    M_t = G_t^{-1} = ρ I_d + Σ(x_{t,η*})   the *mobility* / diffusion tensor

G is what a geodesic pays per unit length, so it must be **large where motion is
expensive** — hence the inversion.  M is what spreads probability mass, so it must be
**large where the data actually spreads**, and it is (up to the floor ρ) exactly the
empirical diffusion coefficient Σ_0 of Hypothesis 1.  Everything below —
the conditional covariance, both matching targets, and the injected Brownian
increment — uses M.  Using G there would inject noise hardest across the manifold,
i.e. into the data voids the metric was constructed to forbid.

Conditional path  p_t(x | x0, x1) = N(x; μ_t, A_t)
--------------------------------------------------
    μ_t = (1-t)x0 + t x1 + t(1-t) φ_{η*}(t, x0, x1)          (geodesic mean)
    A_t = σ² t(1-t) M_t                                       (conditional covariance)

Matching targets (simulation-free, evaluated at a sampled x ~ p_t)
------------------------------------------------------------------
    v_t(x|x0,x1) = ∂_t μ_t
                   + (1-2t)/(2 t(1-t)) · (x - μ_t)
                   + ½ · ∂_t M_t · M_t^{-1} · (x - μ_t)               (CFM target)

    ∇_x log p_t(x|x0,x1) = -1/(σ² t(1-t)) · M_t^{-1} · (x - μ_t)      (CSM target)

    L_CFM(θ) = E ‖ v_θ(t,x) - v_t(x|x0,x1) ‖²
    L_CSM(φ) = E λ(t)² ‖ s_φ(t,x) - ∇_x log p_t(x|x0,x1) ‖²

λ(t) is the score-matching weight.  We default to λ(t) = σ·√(t(1-t)) (the local
noise std), so λ(t)² exactly cancels the 1/(σ²t(1-t)) blow-up of the score target
near the temporal boundaries and the CSM loss stays O(1) — the standard
denoising-score-matching normalisation.

Note M_t^{-1} is obtained as ``geom.mobility_inv(...)`` — the geometry object already
has to build the base tensor G_0 = (ρI + Σ)^{-1} for its metric, so no *extra* batched
inversion enters the training loop.  It is deliberately not ``geom.metric_tensor(...)``:
under the Freidlin–Wentzell metric form the cost tensor carries an extra conformal
factor a(x)² that reparameterises time and must not touch the diffusion.

Inference — anisotropic Euler–Maruyama on the finite-noise SDE
--------------------------------------------------------------
    dX_t = [ v_θ(t,X_t) + σ² M(X_t) s_φ(t,X_t) + σ² ∇·M(X_t) ] dt
           + √2 · σ · M(X_t)^{1/2} dW_t

At inference there is no conditioning pair, so the mobility is realised
state-dependently as M(X_t) = ρI + Σ(X_t); this keeps the injected noise tangent to
the manifold (Σ large along data) and ~isotropically small (→ ρI) in voids.
σ = 0 recovers the deterministic probability-flow ODE.

The third drift term is the Itô correction that state-dependence forces.  With
diffusion matrix D(x) = σ²M(x) the Fokker–Planck operator is ∂_i∂_j(D_ij p), which
splits as ∇·((∇·D) p) + ∇·(D ∇p): a process whose diffusion varies in space needs the
spurious drift ∇·D = σ²∇·M on top of the score term to transport the law the training
targets describe.  (∇·M)_i = Σ_j ∂_j M_ij is taken by forward-mode AD on the same
accessor the sampler shapes its noise with — see :func:`mobility_divergence`.  It is
**not** a training term: the conditional covariance A_t is time-locked to the geodesic
mean, so under the conditional law D is a function of t alone and ∇·D vanishes there.
What training does need for the same reason is ∂_t M_t, which the CFM target carries.

Both terms disappear identically under a mobility that is constant in x and t — see
:class:`ConstantMobility`, which is the control that isolates them.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
from torch import Tensor

from scripts.core.geometry import FinslerGeometry
from scripts.core.models import PhiNet, VelocityNet
from scripts.core.train import EndpointCoupler, interpolant_and_velocity


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class BridgeConfig:
    """Hyper-parameters for Path-B (Schrödinger-bridge) training and generation."""

    sbm_iters: int = 1500          # optimisation steps for (v_θ, s_φ)
    batch_size: int = 512
    lr: float = 2e-3
    grad_clip: float = 5.0
    t_eps: float = 1e-2            # clamp t to [t_eps, 1-t_eps] (t(1-t) in denominators)
    dt_fd: float = 2e-3            # finite-difference step for ∂_t M_t
    log_every: int = 300
    width: int = 128
    depth: int = 4
    seed: int = 0


# --------------------------------------------------------------------------- #
#  Geometry helpers evaluated at the (time-locked) geodesic mean
# --------------------------------------------------------------------------- #
def metric_sqrt(M: Tensor) -> Tensor:
    """A noise-shaping factor L with L Lᵀ = M  ->  (B, d, d).

    M is the mobility tensor ρI + Σ, safely PD for ρ > 0.  We prefer its **Cholesky**
    factor to the symmetric principal square root, with an eigh fallback for a batch
    element that is not numerically PD in float32.  Both the
    conditional bridge sample ``x = μ + σ√(t(1-t)) L ξ`` (training) and the SDE
    increment ``√(2Δt) σ L ζ`` (inference) drive an *isotropic* Gaussian ξ/ζ, so only
    the covariance ``L Lᵀ = M`` enters the law — the choice of factorisation is
    immaterial to the resulting distribution.  Cholesky is ~1000× faster than a
    batched ``eigh`` for many small d×d matrices (the per-step cost dominated SDE
    integration on real single-cell N), with a symmetric-eigh fallback if a batch
    element is not numerically PD.
    """
    try:
        return torch.linalg.cholesky(M)
    except Exception:
        # Scale the fallback jitter by M's own trace rather than fixing it at 1e-6
        # absolute, so it is neither negligible nor dominant at any target trace.
        d = M.shape[-1]
        scale = (M.diagonal(dim1=-2, dim2=-1).sum(-1) / d).clamp(min=1e-12)
        jitter = 1e-6 * scale[..., None, None] * torch.eye(
            d, device=M.device, dtype=M.dtype)
        evals, evecs = torch.linalg.eigh(M + jitter)
        sqrt_evals = torch.sqrt(torch.clamp(evals, min=0.0))
        return evecs @ torch.diag_embed(sqrt_evals) @ evecs.transpose(-1, -2)


def mobility_divergence(geom, x: Tensor) -> Tensor:
    """``(∇·M)_i = Σ_j ∂_j M_ij`` at the current state  ->  (B, d).

    The Itô correction the inference SDE needs because it reads the mobility at X_t
    rather than at the geodesic mean (see the module docstring).  Three paths, in the
    order they are tried:

    1. ``geom.constant_mobility`` -> exactly zero, without touching the geometry.
    2. ``geom.mobility_divergence`` -> the geometry's own closed form.  Both metric
       classes here smooth fixed matrices with a Gaussian kernel, for which the
       divergence is analytic and costs about what reading M costs; they implement it,
       and the sampler calls it 300 times a run, so this is the path that actually runs.
    3. otherwise, forward-mode AD — the definition, and what any new geometry gets for
       free.  **Forward, not reverse:** reverse mode needs one backward pass per (i, j)
       entry, d² of them, because the contraction ties the derivative index to the
       column index and no single cotangent expresses that; a JVP along e_j returns the
       whole ∂M/∂x_j block at once, so d passes suffice.  Still d passes over the
       smoother, which is ~150x a plain ``mobility`` call at d = 50 — hence (2).

    the verification suite in the research repository pins (2) against (3) against central differences.
    """
    if getattr(geom, "constant_mobility", False):
        return torch.zeros_like(x)
    closed_form = getattr(geom, "mobility_divergence", None)
    if closed_form is not None:
        with torch.no_grad():
            return closed_form(x)
    return _mobility_divergence_ad(geom, x)


def _mobility_divergence_ad(geom, x: Tensor) -> Tensor:
    """``∇·M`` by forward-mode AD — the definition :func:`mobility_divergence` falls back
    to, and the reference its closed-form fast paths are checked against."""
    out = torch.zeros_like(x)
    # enable_grad because the sampler runs under no_grad and forward-mode is gated by the
    # same flag; nothing is retained, the JVP output is detached on the way out
    with torch.enable_grad():
        for j in range(x.shape[1]):
            e_j = torch.zeros_like(x)
            e_j[:, j] = 1.0
            _, dM_dxj = torch.func.jvp(geom.metric_tensor_inv, (x.detach(),), (e_j,))
            out = out + dM_dxj[:, :, j].detach()
    return out


class ConstantMobility:
    """A bridge geometry whose mobility is one fixed SPD matrix — constant in x *and* t.

    Wraps any object satisfying the three-accessor mobility API the bridge duck-types
    (``mobility`` / ``mobility_inv`` / ``metric_tensor_inv``) and replaces it with a
    single matrix M₀; every other attribute is delegated, so the wrapped geometry still
    supplies ``d``, the device and whatever a caller reads off it.

    This is the control for the two geometry-variation terms at once.  With M constant,
    ∂_t M_t ≡ 0 kills the CFM target's ½·∂_tM·M⁻¹·δ, and ∇·M ≡ 0 kills the sampler's
    Itô correction, so the pair reduces to the textbook constant-diffusion bridge —
    conditional covariance σ²t(1-t)M₀, drift v_θ + σ²M₀s_φ, noise √2σM₀^{1/2}dW — while
    the interpolant and the coupling stay exactly the ones the Finsler geometry built.
    Distinct from ``pathb_iso``'s M ≡ I: the anisotropy of the data's mobility is kept,
    only its *variation* is removed, which is the thing the two terms exist for.

    :meth:`from_points` takes the data average, the one constant that leaves the
    bridge's overall noise scale (and hence ``width_mult``) where it was.
    """

    constant_mobility = True

    def __init__(self, geom, M0: Tensor):
        assert M0.ndim == 2 and M0.shape[0] == M0.shape[1], "M0 must be one d x d matrix"
        self._geom = geom
        self.M0 = M0
        self.M0_inv = torch.linalg.inv(M0)

    @classmethod
    def from_points(cls, geom, X: Tensor, chunk: int = 512) -> "ConstantMobility":
        """M₀ = mean_i M(x_i) over the given cloud, symmetrised."""
        with torch.no_grad():
            tot = sum(geom.metric_tensor_inv(X[s:s + chunk]).sum(0)
                      for s in range(0, len(X), chunk))
        M0 = tot / len(X)
        return cls(geom, 0.5 * (M0 + M0.T))

    def mobility(self, x: Tensor) -> Tensor:
        return self.M0[None].expand(x.shape[0], *self.M0.shape)

    def mobility_inv(self, x: Tensor) -> Tensor:
        return self.M0_inv[None].expand(x.shape[0], *self.M0_inv.shape)

    #: the bridge's name for :meth:`mobility`, kept an alias here for the same reason
    #: :class:`~scripts.core.geometry.FinslerGeometry` keeps it (see ``geometry.py``)
    metric_tensor_inv = mobility

    def mobility_divergence(self, x: Tensor) -> Tensor:
        """Zero, by construction — spelled out so ``__getattr__`` cannot delegate the
        wrapped geometry's closed form and reintroduce the term this class removes."""
        return torch.zeros_like(x)

    def mean_mobility_eig(self, chunk: int = 512) -> float:
        """``tr M₀ / d`` — the scale a bridge width is quoted against.

        Spelled out rather than delegated, though for an M₀ built by :meth:`from_points`
        the two agree exactly (the trace is linear, so the trace of the mean is the mean
        of the traces).  That identity is the point: ``sigma_for(·, width_mult)`` returns
        the *same* σ here as on the geometry this wraps, so the constant-mobility arm and
        ``pathb`` run at one noise level and the comparison is about the mobility's
        variation, not about how much noise each got.
        """
        return float(self.M0.diagonal().sum() / self.M0.shape[0])

    def __getattr__(self, name: str):
        return getattr(self._geom, name)


def _mean_at(phi: PhiNet, geom: FinslerGeometry, x0: Tensor, x1: Tensor,
             t: Tensor) -> Tensor:
    """Geodesic mean μ_t = (1-t)x0 + t x1 + t(1-t) φ(t) at scalar-per-row time t."""
    with torch.enable_grad():
        m, _ = interpolant_and_velocity(phi, t, x0, x1, create_graph=False)
    return m.detach()


# --------------------------------------------------------------------------- #
#  Conditional bridge sample x ~ p_t and its CFM / CSM targets
# --------------------------------------------------------------------------- #
def sample_bridge_targets(
    phi: PhiNet,
    geom: FinslerGeometry,
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
    sigma: float,
    cfg: BridgeConfig,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Draw x ~ N(μ_t, A_t) and return (x, v_target, s_target, lambda_t).

    All geometry is time-locked: M_t and ∂_t M_t are evaluated at the deterministic
    geodesic mean μ_t (a pure function of t for a fixed endpoint pair), never at the
    noisy sample x — this matches the conditional Fokker–Planck derivation for
    Path B.
    """
    assert t.shape[1] == 1
    t = torch.clamp(t, cfg.t_eps, 1.0 - cfg.t_eps)             # keep t(1-t) > 0
    tt = t * (1.0 - t)                                         # (B,1)

    # geodesic mean μ_t and its time-derivative ∂_t μ_t = ẋ_{t,η}
    with torch.enable_grad():
        mu, mu_dot = interpolant_and_velocity(phi, t, x0, x1, create_graph=False)
    mu, mu_dot = mu.detach(), mu_dot.detach()

    # Time-locked mobility M_t = ρI + Σ(μ_t) and its inverse M_t^{-1} = G_t.  Both come
    # from the geometry object rather than being hand-rolled, so the G≡I ablation stays
    # self-consistent (under use_metric=False both accessors return I, and a hand-rolled
    # ρI+Σ(μ) would silently disagree).  Note M^{-1} costs nothing: it *is* the metric.
    M = geom.mobility(mu)                                      # (B, d, d)  mobility
    Minv = geom.mobility_inv(mu)                               # (B, d, d)  = G_0(μ_t)

    # ∂_t M_t by central finite differences along the geodesic backbone.  Under a mobility
    # that does not vary in x the mean moves but M does not, so the derivative is exactly
    # zero and the two extra geometry reads are pure waste — skip them.
    if getattr(geom, "constant_mobility", False):
        dM_dt = torch.zeros_like(M)
    else:
        h = cfg.dt_fd
        t_plus = torch.clamp(t + h, cfg.t_eps, 1.0 - cfg.t_eps)
        t_minus = torch.clamp(t - h, cfg.t_eps, 1.0 - cfg.t_eps)
        M_plus = geom.metric_tensor_inv(_mean_at(phi, geom, x0, x1, t_plus))
        M_minus = geom.metric_tensor_inv(_mean_at(phi, geom, x0, x1, t_minus))
        dM_dt = (M_plus - M_minus) / (t_plus - t_minus)[:, :, None]   # (B, d, d)

    # sample x ~ N(μ_t, A_t),  A_t = σ² t(1-t) M_t  =>  x = μ + σ√(t(1-t)) M^{1/2} ξ
    if sigma > 0.0:
        L = metric_sqrt(M)                                    # M_t^{1/2}
        xi = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        std = sigma * torch.sqrt(tt)                          # (B,1)
        delta = std * torch.einsum("bij,bj->bi", L, xi)      # x - μ_t
    else:
        delta = torch.zeros_like(mu)                         # σ = 0 -> collapse to mean
    x = mu + delta

    # CFM target  v_t = ∂_tμ + (1-2t)/(2 t(1-t)) δ + ½ ∂_tM_t M_t^{-1} δ
    #
    # The *probability-flow* velocity of the conditional path — ẋ = ∂_tμ +
    # ½ Ȧ_t A_t^{-1} δ with A_t = σ² t(1-t) M_t, whose isotropic part is
    # ½ d/dt log(t(1-t)) = (1-2t)/(2t(1-t)).  This is the object Prop. 21 states and the
    # one :func:`integrate_sde` assumes: that sampler adds σ²M s_φ itself, so a target
    # already carrying the score drift would apply it twice.
    coef = (1.0 - 2.0 * t) / (2.0 * tt)                       # (B,1)
    dM_Minv_delta = torch.einsum(
        "bij,bjk,bk->bi", dM_dt, Minv, delta)                # ∂_tM_t M_t^{-1} δ
    v_target = mu_dot + coef * delta + 0.5 * dM_Minv_delta

    # CSM target  ∇log p_t = -1/(σ² t(1-t)) M_t^{-1} δ
    if sigma > 0.0:
        Minv_delta = torch.einsum("bij,bj->bi", Minv, delta)
        s_target = -Minv_delta / (sigma ** 2 * tt)
        lambda_t = sigma * torch.sqrt(tt)                    # λ(t) = local noise std
    else:
        s_target = torch.zeros_like(mu)
        lambda_t = torch.zeros_like(t)

    return x, v_target, s_target, lambda_t


# --------------------------------------------------------------------------- #
#  Train the drift v_θ and score s_φ networks jointly
# --------------------------------------------------------------------------- #
def train_sbm(
    phi: PhiNet,
    geom: FinslerGeometry,
    coupler: EndpointCoupler,
    cfg: BridgeConfig,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[VelocityNet, VelocityNet, list[float]]:
    """Minimise L_CFM(θ) + L_CSM(φ) for a fixed noise level σ.

    φ_η is frozen (pretrained with the Finsler loss); v_θ and s_φ are learned.  For
    σ = 0 the path is deterministic, the score target vanishes, and only v_θ is
    trained (s_φ is returned untrained but unused by the ODE-limit integrator).

    Returns (v_net, s_net, history) where history is the combined per-iter loss.
    """
    torch.manual_seed(cfg.seed)
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)

    v_net = VelocityNet(geom.d, width=cfg.width, depth=cfg.depth).to(device=device, dtype=dtype)
    s_net = VelocityNet(geom.d, width=cfg.width, depth=cfg.depth).to(device=device, dtype=dtype)
    params = chain(v_net.parameters(), s_net.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 17)
    history: list[float] = []

    for it in range(cfg.sbm_iters):
        x0, x1 = coupler.sample(cfg.batch_size)
        t = torch.rand(cfg.batch_size, 1, device=device, dtype=dtype)
        x, v_tgt, s_tgt, lam = sample_bridge_targets(
            phi, geom, x0, x1, t, sigma, cfg, generator=gen)
        t = torch.clamp(t, cfg.t_eps, 1.0 - cfg.t_eps)

        loss_cfm = ((v_net(t.detach(), x.detach()) - v_tgt.detach()) ** 2).mean()
        if sigma > 0.0:
            resid = s_net(t.detach(), x.detach()) - s_tgt.detach()
            loss_csm = ((lam.detach() ** 2) * (resid ** 2)).mean()
        else:
            loss_csm = torch.zeros((), device=device, dtype=dtype)
        loss = loss_cfm + loss_csm

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            chain(v_net.parameters(), s_net.parameters()), cfg.grad_clip)
        opt.step()

        history.append(float(loss))
        if it % cfg.log_every == 0 or it == cfg.sbm_iters - 1:
            print(f"      [sbm σ={sigma:g}] iter {it:5d}  "
                  f"L_CFM={float(loss_cfm):.5f}  L_CSM={float(loss_csm):.5f}")
    return v_net, s_net, history


# --------------------------------------------------------------------------- #
#  Generate stochastic paths — anisotropic Euler–Maruyama on the SDE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def integrate_sde(
    v_net: VelocityNet,
    s_net: VelocityNet,
    geom: FinslerGeometry,
    x0: Tensor,
    sigma: float,
    n_steps: int = 200,
    seed: int = 0,
    noise_off_after: float = 1.0,
    t_eps: float = BridgeConfig.t_eps,
) -> Tensor:
    """Integrate dX = (v_θ + σ² M_t s_φ + σ² ∇·M_t) dt + √2 σ M_t^{1/2} dW, t=0 -> 1.

    M_t = ρI + Σ(X_t) is the data-derived *mobility* evaluated at the current state
    (there is no conditioning pair at inference), so the injected noise is anisotropic
    and manifold-tangent: largest along the directions in which P spreads mass, and
    shrinking to the isotropic floor ρI in voids.  σ = 0 gives the deterministic
    probability-flow ODE.

    Reading M at the state is what makes the diffusion matrix D = σ²M space-dependent,
    and a space-dependent D carries the Itô spurious drift ∇·D = σ²∇·M — the third term,
    from :func:`mobility_divergence` (forward-mode AD, d passes).  It is not a free
    parameter or a correction one may drop: without it the Fokker–Planck generator of
    the integrated process is not the one the conditional targets were derived for.
    Under :class:`ConstantMobility` it is identically zero and costs nothing.

    ``t_eps`` clamps the time *argument fed to the networks* to [t_eps, 1-t_eps] —
    the same window they were trained on (:attr:`BridgeConfig.t_eps`, applied at
    ``bridge.py`` line ~127).  The integration grid itself is untouched, so the band
    times t = k/3 still land exactly on a step.  This is a train/inference
    consistency fix, not a change to the SDE: both conditional targets carry a
    1/(t(1-t)) factor, so the networks are *only ever* fit on [t_eps, 1-t_eps], and
    evaluating them at t = 0 asks for an extrapolation of a function whose true
    value there is infinite.  With n_steps = 300 the unclamped loop evaluated three
    steps below the trained window and two above it, and an explicit Euler step of a
    drift with stiffness k(t) = 1/(t(1-t)) is stable only while dt·k < 2 — satisfied
    comfortably at t_eps = 0.01 (dt·k = 0.34) and violated outright at t = 0.  Left
    unclamped this produced rare but total divergence: 2/18 Cycle Path-B LOTO cells
    reached |X| ~ 1e5 by t = 1/3 and 1e10 by t = 1.

    ``noise_off_after`` implements a **deterministic terminal tail**: once
    t ≥ noise_off_after the stochastic increment is suppressed and only the
    (drift + score) push is applied.  Rationale — near t=1 the score drift alone,
    σ²M s_φ ≈ -δ/(t(1-t)) with δ = X_t-μ_t, provides an arbitrarily strong pull
    onto the endpoint, so a single Euler step lands the sample on the backbone;
    the last noise kick √(2Δt)σ M^{1/2}ζ then only adds O(√Δt) spread that no
    remaining step can correct, inflating the terminal marginal.  This is also
    where the train/inference geometry mismatch (M(μ_t) vs M(X_t)) is largest.
    The default 1.0 keeps the original all-the-way-to-t=1 noisy behaviour, since
    the last integrated time is (n_steps-1)/n_steps < 1.0.

    The stochastic increment is shaped by M_t, the same tensor the drift's score term
    uses and the one the networks were fit under: this is the SDE whose marginals the
    conditional targets were derived for.

    Returns the full trajectory tensor of shape (n_steps + 1, B, d).
    """
    device, dtype = x0.device, x0.dtype
    dt = 1.0 / n_steps
    gen = torch.Generator(device=device).manual_seed(seed)
    x = x0.clone()
    traj = [x.clone()]
    sqrt_2dt_sigma = float(np.sqrt(2.0 * dt) * sigma) if sigma > 0.0 else 0.0

    for n in range(n_steps):
        t_scalar = n * dt
        # networks see the clamped time; the integration grid keeps the true t_scalar
        t_net = min(max(t_scalar, t_eps), 1.0 - t_eps)
        t0 = torch.full((x.shape[0], 1), t_net, device=device, dtype=dtype)
        M = geom.metric_tensor_inv(x)                         # M_t = ρI + Σ(X_t)
        drift = v_net(t0, x)
        if sigma > 0.0:
            score = s_net(t0, x)
            drift = drift + (sigma ** 2) * torch.einsum("bij,bj->bi", M, score)
            # Itô correction for the state-dependence of D = σ²M  (zero if M is constant)
            drift = drift + (sigma ** 2) * mobility_divergence(geom, x)
        x = x + dt * drift
        # deterministic terminal tail: no noise once t ≥ noise_off_after
        if sqrt_2dt_sigma > 0.0 and t_scalar < noise_off_after:
            L = metric_sqrt(M)                               # M_t^{1/2}
            dw = torch.randn(x.shape, generator=gen, device=device, dtype=dtype)
            x = x + sqrt_2dt_sigma * torch.einsum("bij,bj->bi", L, dw)
        traj.append(x.clone())
    return torch.stack(traj, dim=0)
