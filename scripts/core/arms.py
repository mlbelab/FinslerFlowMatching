"""The six method arms, behind one uniform ``(TrainingSet, HParams) -> ArmResult`` call.

Every arm sees exactly the same ``TrainingSet`` (X_train, P_train, p_0, p_1) and
returns the same object: a drift network plus the trajectory obtained by pushing the
**training** source cloud (band T0 ∩ pool) forward to t = 1.  Nothing downstream
needs to know which arm produced it, which is what makes the comparison fair.

The metric ladder (the user's ablation) is four settings of one geometry object:

    ffm_eucl    G ≡ I,   β = 0      F(x,v) = ‖v‖            no P at all
    ffm_riem    G = G_P, β = 0      F = √(vᵀGv)             second moment only
    ffm_drift   G ≡ I,   β = β_P    F = ‖v‖ + βᵀv           first moment only
    ffm         G = G_P, β = β_P    F = √(vᵀGv) + βᵀv       + the first moment

``ffm_drift`` is the mirror of ``ffm_riem``: the two rungs each keep one moment of P, so
between them they say which of the two the geometry is actually living off.  It is not in
:data:`MAIN_ARMS` — the Sheet's ablation table is the published three-rung one — and is
run where a benchmark asks for it by name (the circles panel does).

where the last row's ``(G_P, β_P)`` is whichever metric form ``HParams.metric_form`` /
``HParams.drift_mode`` / ``HParams.sigma_mode`` select — the original direction-only
Randers 1-form off the P^asym flux over a P^sym diffusion, the Freidlin–Wentzell action
off the full-P first moment, or either metric form over the full-P *second* moment.  The
ladder is the same three switches whichever is chosen, which is what makes the variants
comparable.  Note that ``ffm_riem`` is blind to ``drift_mode`` (its β is zero, and the
``randers`` form's G does not read b) but *not* to ``sigma_mode``, which makes it the
rung that isolates the diffusion estimator on its own.

and the remaining three arms are the external comparisons:

    cfm         straight-line CFM over a Euclidean-OT coupling (geometry-free)
    mfm_land    Metric Flow Matching under the authors' LAND metric (point cloud only)
    curly       Curly Flow Matching (Petrović et al. 2025) over a measured velocity
                field when the cloud has one, else over the b̂ read out of P
    pathb       noisy Schrödinger-bridge matching on the Randers backbone (σ > 0)

plus one arm reported only in the appendix:

    pathb_iso   Path B with mobility M_t ≡ I — same Randers interpolant and
                Finsler-OT coupling, isotropic instead of geometry-shaped noise

Which arms actually depend on P
-------------------------------
Only ``ffm``, ``ffm_riem``, ``curly``, ``pathb`` and ``pathb_iso`` read the
transition matrix (``pathb_iso`` through its backbone, not its noise).
``ffm_eucl`` neutralises the metric, and ``cfm``/``mfm_land`` never touch P by
construction.  Curly-FM is P-dependent because its reference velocity field is the
kernel-smoothed flux b̂ that P defines — it is the analogue of the RNA velocity the
authors feed it on real data, and the only P-derived object it ever sees.  The
P-variant experiment (directional / stagnant / hybrid) therefore only runs the
P-dependent arms — running the others three times would burn compute reproducing
identical numbers.  :data:`P_DEPENDENT_ARMS` is the single source of that truth.

Training-budget warning
-----------------------
``phase1_iters`` / ``phase2_iters`` are deliberately *not* swept.  The Cycle's FFM W2
collapses when they drop below 2500 / 1500 (the interpolant has not yet learnt to
follow the ring and starts chording across it), so a sweep that was allowed to
shorten training would "discover" a setting that silently breaks one dataset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .coupling import build_ot_coupler, build_standard_ot_coupler
from .curly_fm import CurlyConfig, train_curly_from_field
from .geometry import FinslerGeometry
from .mfm_metric import make_mfm_metric
from .train import (
    EndpointCoupler,
    TrainConfig,
    integrate_flow,
    train_cfm,
    train_phase1,
    train_phase2,
)
from .bridge import BridgeConfig, ConstantMobility, integrate_sde, train_sbm

from .protocol import TrainingSet

#: every arm the suite knows about, in table order
ARMS = ("ffm", "ffm_riem", "ffm_eucl", "ffm_drift", "cfm", "mfm_land", "curly",
        "pathb", "pathb_iso", "pathb_const")
#: the arms of the headline grid.  ``pathb_const`` is deliberately **not** here: it is
#: an opt-in diagnostic for what the mobility's variation in x and t contributes, asked
#: for by name, not a column every default grid should pay for.
#: ``pathb_iso`` is *run* everywhere ``pathb`` is —
#: it is the control for the score-Jacobian second-moment metric (trained on M ≡ I
#: rather than M = ρI+Σ, so its Bures distance to the data mobility is the scale
#: against which Path B's is read) and it needs the same ρ ladder and P variants to
#: be comparable.  It stays out of the main-body *reporting* groups below.
#: ``ffm_drift`` is spelled out of this list rather than inherited from :data:`ARMS`:
#: the published ladder is the three-rung one, and adding a fourth here would silently
#: widen every default grid and every table that takes ``MAIN_ARMS`` as its column set.
MAIN_ARMS = ("ffm", "ffm_riem", "ffm_eucl", "cfm", "mfm_land", "curly",
             "pathb", "pathb_iso")
#: the metric-ladder ablation, bottom rung first
ABLATION_ARMS = ("ffm_eucl", "ffm_riem", "ffm")
#: the two arms of the σ sweep — Path B's geometry-shaped bridge vs its isotropic
#: ablation.  Distinct from :data:`NOISY_ARMS`, which is a *reporting* group.
SIGMA_SWEEP_ARMS = ("pathb", "pathb_iso")
#: main-body reporting groups.  ``FLOW_ARMS`` are the deterministic transport models;
#: ``NOISY_ARMS`` carry an explicit noise level and so are compared with each other
#: rather than against the deterministic arms.  The three metric-ladder rungs are
#: deliberately absent from both: they are an ablation, not a baseline.
#:
#: Curly-FM sits with the *deterministic* arms because :attr:`HParams.curly_sigma`
#: defaults to zero — its authors' setting, and the only one they report a result at
#: (Petrović et al. 2025, App. G).  A tree that has a reference noise level overrides it
#: (circles and erythroid both run at their published 0.01), and such a tree must not use
#: these groups to decide a block heading: they read the *default*, not the point an arm
#: was actually run at.  The Sheet notebook prints Curly-FM's effective σ beside its table
#: for exactly that reason.
FLOW_ARMS = ("cfm", "mfm_land", "curly", "ffm")
NOISY_ARMS = ("pathb",)
#: arms whose output can change when P is rebuilt differently.  ``pathb_iso`` and
#: ``pathb_const`` are here too: only their *noise* is flattened, while their geodesic
#: backbone and OT coupling are still built from the Finsler geometry of P — and
#: ``pathb_const``'s M̄ is itself an average of P's mobility.
P_DEPENDENT_ARMS = ("ffm", "ffm_riem", "curly", "pathb", "pathb_iso", "pathb_const")
#: default integration resolution (must be divisible by 3 so t = k/3 lands on a step)
N_STEPS = 300


# --------------------------------------------------------------------------- #
#  Hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class HParams:
    """One flat hyper-parameter record shared by all arms (unused fields ignored).

    Flat rather than per-arm nested so a sweep point, a W&B config and a result JSON
    are all the same dictionary — no arm-specific plumbing to keep in sync.
    """

    # --- Randers geometry (ffm / ffm_riem / pathb) --------------------------- #
    rho: float = 0.03            # isotropic floor of M = ρI+Σ; small => strong void penalty
    c: float = 0.9               # Randers 1-form strength, must stay < 1
    target_trace: float = 2.0    # sets the Σ temperature ε adaptively
    # metric_form / drift_mode select the two geometry variants compared in the paper:
    #   ("randers", "asym")  the original -- direction-only 1-form off the P^asym flux
    #   ("fw",      "full")  the Freidlin-Wentzell action off the full-P first moment
    # See :mod:`scripts.core.geometry` for the derivation.  Kept as two independent
    # switches, not one enum, so the intermediate ("randers", "full") can be run as the
    # control that separates "which drift estimator" from "which metric form".
    metric_form: str = "randers"
    drift_mode: str = "asym"
    # which second moment of P estimates the diffusion Σ, and hence both the metric
    # G_0 = (rho I + Sigma)^{-1} and Path B's mobility M_t.  "sym" is the published
    # split (Hypothesis 1: the reversible half of P carries the diffusion); "full" is
    # the forward conditional second moment, contaminated at O(eps^2) by the drift but
    # never mixing a row with its reverse edges.  Trace-matched to "sym" by an exact
    # cancellation, so the switch moves anisotropy only -- see scripts.core.geometry.
    sigma_mode: str = "sym"
    a_floor: float = 0.1         # floor on a/ā; only metric_form="fw" reads it
    # λ of metric_form="fw_lambda": the single regulariser that replaces *both* c and
    # a_floor in the quadrature form a = sqrt(||b||²_{G_0} + λ²).  Absolute, but only
    # meaningful against the data's own ||b||_{G_0} (~0.033 on the Sheet at rho=0.1),
    # so grids for it are placed in units of that mean and the run record logs both it
    # and the resulting c_eff.  Requires c = 1.0; see scripts.core.geometry.
    fw_lambda: float = 0.03
    eps_kernel_scale: float = 1.0  # multiplies the median-kNN smoothing bandwidth
    # --- LAND metric (mfm_land) --------------------------------------------- #
    land_gamma: float = 0.2      # local-covariance bandwidth
    land_rho: float = 1e-3       # metric floor
    land_alpha: float = 1.0      # metric exponent
    # "auto" applies the reference's dimension rule (LAND in 2-D, conformal RBF above);
    # "land"/"rbf" force one.  The rule is the reference's *default*, and their per-cloud
    # configs name the metric explicitly, so a caller reproducing one of those configs
    # must be able to say which family it asks for rather than have it inferred.
    land_kind: str = "auto"
    # --- Phase-2 optimal transport ------------------------------------------ #
    ot_K: int = 8                # Monte-Carlo quadrature nodes for C_ij
    ot_max_pts: int = 200        # endpoints per side entering the Sinkhorn solve
    blur_frac: float = 0.1       # entropic reg = blur_frac · median(C)
    # --- optimisation -------------------------------------------------------- #
    lr: float = 2e-3
    width: int = 128
    depth: int = 4
    batch_size: int = 512
    phase1_iters: int = 2500     # NOT swept — see the module docstring
    phase2_iters: int = 1500     # NOT swept — see the module docstring
    grad_clip: float = 5.0
    # --- Curly-FM (curly) ---------------------------------------------------- #
    curly_alpha: float = 0.01    # magnitude weight of the L2 term ‖u − α·μ̇_t‖²
    # Noise in Curly-FM's own interpolant, x_t = mu_t + sqrt(t(1-t))*sigma*eps.  Zero is
    # *their* value: the release's `get_xt` defaults to `sigma=0.0`, and the paper is
    # explicit -- "we recommend setting sigma to zero unless some reference sigma value
    # is known.  Therefore, all of our experiments are performed under sigma = gt = 0"
    # (Petrovic et al. 2025, App. G).  So a cloud with no reference noise level runs the
    # baseline at the setting its authors run it at.  The two trees that *do* have a
    # reference value override this: the circles port takes 0.01 from their own
    # `circles_toy.ipynb`, and the erythroid port takes 0.01 from their appendix Table 12.
    curly_sigma: float = 0.0
    curly_k: int = 20            # kNN width of their reference-field estimator
    curly_coupling_batch: int = 128   # assignment cost is O(b²·N) — keep it modest
    # "release" trains their own MLP, "shared" trains the engine's PhiNet/VelocityNet.
    # The default is theirs, so every record already in the tree stays measured on the
    # architecture it was measured on; the notebooks pass "shared", because there the
    # claim is that five arms differ by method alone.  See scripts.core.curly_fm.
    curly_net: str = "release"
    # --- Path B -------------------------------------------------------------- #
    sigma: float = 0.15          # bridge noise level
    sbm_iters: int = 1500
    noise_off_after: float = 1.0  # deterministic terminal tail (1.0 = never off)

    def train_config(self, seed: int) -> TrainConfig:
        return TrainConfig(
            phase1_iters=self.phase1_iters, phase2_iters=self.phase2_iters,
            batch_size=self.batch_size, lr=self.lr, grad_clip=self.grad_clip,
            seed=seed, width=self.width, depth=self.depth,
        )

    def bridge_config(self, seed: int) -> BridgeConfig:
        return BridgeConfig(
            sbm_iters=self.sbm_iters, batch_size=self.batch_size, lr=self.lr,
            grad_clip=self.grad_clip, width=self.width, depth=self.depth, seed=seed,
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArmResult:
    """What one trained arm hands to the metrics layer."""

    arm: str
    v_net: torch.nn.Module
    traj: np.ndarray                     # (N_STEPS+1, B, d), pushed-forward source
    diag: dict                           # OT / loss diagnostics, JSON-serialisable
    aux: dict                            # phi, s_net, geom — kept for the figures


# --------------------------------------------------------------------------- #
#  Geometry construction
# --------------------------------------------------------------------------- #
def build_geometry(kind: str, ts: TrainingSet, hp: HParams,
                   device: torch.device, dtype: torch.dtype):
    """The metric object for one arm.

    ``kind`` is one of ``randers`` / ``riemannian`` / ``euclidean`` (all three are the
    same :class:`FinslerGeometry` with different switches, so the ablation isolates
    the metric and nothing else) or ``land`` (the MFM comparison).
    """
    if kind == "land":
        # ``make_mfm_metric`` picks LAND in 2-D and the conformal RBF metric above it,
        # following the reference implementation.  ``rho``/``alpha`` mean the same thing
        # in both; ``gamma`` is LAND's bandwidth and is dropped on a >2-D cloud, where
        # RBF's own KMeans bandwidths take its place.
        return make_mfm_metric(
            ts.X, device=device, dtype=dtype, kind=hp.land_kind,
            params={"gamma": hp.land_gamma, "rho": hp.land_rho, "alpha": hp.land_alpha},
        )
    switches = {
        "randers": dict(use_metric=True, use_one_form=True),
        "riemannian": dict(use_metric=True, use_one_form=False),
        "euclidean": dict(use_metric=False, use_one_form=False),
        # G ≡ I but the 1-form kept: β is then read off the *identity* background,
        # β = −c·b/‖b‖, so F(x,v) = ‖v‖ + βᵀv is a flat Randers metric that discounts
        # travel along the drift and nothing else.
        "drift": dict(use_metric=False, use_one_form=True),
    }
    assert kind in switches, f"unknown geometry kind {kind!r}"
    return FinslerGeometry(
        ts.X, ts.P, rho=hp.rho, c=hp.c, target_trace=hp.target_trace,
        metric_form=hp.metric_form, drift_mode=hp.drift_mode,
        sigma_mode=hp.sigma_mode, a_floor=hp.a_floor,
        fw_lambda=hp.fw_lambda, eps_kernel_scale=hp.eps_kernel_scale,
        device=device, dtype=dtype, **switches[kind],
    )


# --------------------------------------------------------------------------- #
#  Shared Path-A pipeline (Phase 1 -> Phase 2 -> Phase 3)
# --------------------------------------------------------------------------- #
def _path_a(geom, ts: TrainingSet, hp: HParams, seed: int,
            device: torch.device, dtype: torch.dtype) -> tuple:
    """Geodesic interpolant, geometry-aware OT coupling, velocity distillation."""
    cfg = hp.train_config(seed)
    product = EndpointCoupler(ts.X, ts.p0, ts.p1, device, dtype, seed=seed)

    phi, hist1 = train_phase1(geom, product, cfg, device, dtype)
    ot_coupler, ot_diag = build_ot_coupler(
        phi, geom, ts.X, ts.p0, ts.p1, device, dtype,
        K=hp.ot_K, blur_frac=hp.blur_frac, max_pts=hp.ot_max_pts, seed=seed)
    v_net, hist2 = train_phase2(phi, ot_coupler, cfg, geom.d, device, dtype)
    return phi, v_net, ot_diag, hist1, hist2


def _source_cloud(ts: TrainingSet, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
    """The cells the flow is integrated from: the *training* source pool (T0 ∩ pool).

    Deliberately not the eval third of T0 — pushing the evaluation cells themselves
    forward would make the T0 column identically zero and would leak the eval sample
    into every downstream marginal.
    """
    return torch.as_tensor(ts.X[ts.p0], dtype=dtype, device=device)


def _loss_tail(hist: list[float], n: int = 100) -> float:
    """Mean of the last ``n`` iterations — a less jumpy convergence report."""
    return float(np.mean(hist[-n:])) if hist else float("nan")


# --------------------------------------------------------------------------- #
#  The arms
# --------------------------------------------------------------------------- #
def _run_ffm_family(kind: str, arm: str, ts: TrainingSet, hp: HParams, seed: int,
                    device: torch.device, dtype: torch.dtype,
                    n_steps: int) -> ArmResult:
    geom = build_geometry(kind, ts, hp, device, dtype)
    phi, v_net, ot_diag, h1, h2 = _path_a(geom, ts, hp, seed, device, dtype)
    traj = integrate_flow(v_net, _source_cloud(ts, device, dtype), n_steps=n_steps)
    diag = {"ot": ot_diag, "loss_phase1": _loss_tail(h1), "loss_phase2": _loss_tail(h2)}
    if isinstance(geom, FinslerGeometry):
        diag["geometry"] = geom.verify()
    return ArmResult(arm, v_net, traj.cpu().numpy(), diag,
                     {"phi": phi, "geom": geom, "hist1": h1, "hist2": h2})


def run_ffm(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Ours: full Randers/Finsler metric — G from P's second moment, β from its first.

    Which moments, and how the first one enters, is ``hp.metric_form`` /
    ``hp.drift_mode`` / ``hp.sigma_mode``.
    """
    return _run_ffm_family("randers", "ffm", ts, hp, seed, device, dtype, n_steps)


def run_ffm_riem(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Ablation rung 2: symmetric Riemannian metric (β = 0) — P's second moment only.

    Blind to ``hp.drift_mode`` by construction, so it is the rung that reads
    ``hp.sigma_mode`` and nothing else.
    """
    return _run_ffm_family("riemannian", "ffm_riem", ts, hp, seed, device, dtype, n_steps)


def run_ffm_eucl(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Ablation rung 1: G ≡ I and β = 0, i.e. F(x,v) = ‖v‖ — the pipeline without geometry.

    Not redundant with ``cfm``: this keeps the *whole* Path-A machinery (a learned
    interpolant φ, a geodesic-cost Sinkhorn coupling, distillation) and removes only
    the metric.  Agreement between the two is the control that says the ladder's
    bottom rung really is the geometry-free baseline.
    """
    return _run_ffm_family("euclidean", "ffm_eucl", ts, hp, seed, device, dtype, n_steps)


def run_ffm_drift(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Rung 1', the mirror of ``ffm_riem``: G ≡ I with β kept — the drift alone.

    F(x,v) = ‖v‖ + βᵀv over a flat background, so the only thing P contributes is the
    direction its first moment points in: no anisotropy, no density term, nothing from
    the second moment.  Against ``ffm`` it says how much of the result the metric buys
    on top of the 1-form, and against ``ffm_riem`` which of the two moments carries it.
    """
    return _run_ffm_family("drift", "ffm_drift", ts, hp, seed, device, dtype, n_steps)


def run_mfm_land(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Metric Flow Matching under the authors' LAND metric — point cloud, no P."""
    return _run_ffm_family("land", "mfm_land", ts, hp, seed, device, dtype, n_steps)


def run_cfm(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Straight-line CFM over an entropic **Euclidean**-OT coupling (OT-CFM).

    The standard baseline: same Sinkhorn solver and blur as the geometric arms, but
    the cost is ‖x0 − x1‖² and the interpolant is the straight segment, so neither
    the point cloud's shape nor P enters anywhere.
    """
    cfg = hp.train_config(seed)
    coupler, ot_diag = build_standard_ot_coupler(
        ts.X, ts.p0, ts.p1, device, dtype,
        blur_frac=hp.blur_frac, max_pts=hp.ot_max_pts, seed=seed)
    d = int(ts.X.shape[1])
    v_net, hist = train_cfm(coupler, cfg, d, device, dtype)
    traj = integrate_flow(v_net, _source_cloud(ts, device, dtype), n_steps=n_steps)
    return ArmResult("cfm", v_net, traj.cpu().numpy(),
                     {"ot": ot_diag, "loss_cfm": _loss_tail(hist)},
                     {"hist": hist})


def run_pathb(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Path B: noisy Schrödinger-bridge matching on the frozen Randers backbone.

    Phase 1 and the Finsler-OT coupling are shared with ``ffm`` — the difference is
    Phase 3: instead of distilling ẋ, we fit (v_θ, s_φ) to the σ > 0 bridge
    conditionals and sample by anisotropic Euler–Maruyama.  The reported trajectory
    is therefore *stochastic*; its randomness is seeded from ``seed`` so a rerun
    reproduces it exactly.
    """
    geom = build_geometry("randers", ts, hp, device, dtype)
    cfg = hp.train_config(seed)
    bcfg = hp.bridge_config(seed)
    product = EndpointCoupler(ts.X, ts.p0, ts.p1, device, dtype, seed=seed)

    phi, hist1 = train_phase1(geom, product, cfg, device, dtype)
    ot_coupler, ot_diag = build_ot_coupler(
        phi, geom, ts.X, ts.p0, ts.p1, device, dtype,
        K=hp.ot_K, blur_frac=hp.blur_frac, max_pts=hp.ot_max_pts, seed=seed)
    v_net, s_net, hist = train_sbm(phi, geom, ot_coupler, bcfg, hp.sigma, device, dtype)

    traj = integrate_sde(v_net, s_net, geom, _source_cloud(ts, device, dtype),
                         sigma=hp.sigma, n_steps=n_steps, seed=seed,
                         noise_off_after=hp.noise_off_after)
    diag = {"ot": ot_diag, "loss_phase1": _loss_tail(hist1), "loss_sbm": _loss_tail(hist),
            "sigma": hp.sigma, "geometry": geom.verify()}
    return ArmResult("pathb", v_net, traj.cpu().numpy(), diag,
                     {"phi": phi, "s_net": s_net, "geom": geom, "hist1": hist1,
                      "hist": hist})


def run_curly(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Curly Flow Matching (Petrović et al., NeurIPS 2025) over a measured velocity.

    Curly-FM expects a measured per-cell velocity (RNA velocity in their paper).  When
    the benchmark supplies one — ``ts.velocity``, as the Curly circles do — the arm is
    handed it **raw**, because that is its published input: giving it the smoothed flux
    instead would be scoring the baseline on a coarsened version of its own data while
    we read the same field.  Our arms never touch ``ts.velocity``; they see it only
    after :func:`~scripts.core.transition.build_velocity_kernel` has turned it into P,
    which is strictly less information.

    On the clouds with no arrows — the Sheet, the MFM Arch — no arm is ever shown the
    generator's tangent, so we fall back to the one velocity the data does define: the
    kernel-smoothed empirical flux b̂(x_i) that our Randers geometry is itself built
    from.  Both methods therefore
    read the same P and nothing else, which is the only information-matched way to
    run this baseline here.  That coupling is deliberate and survives the metric
    variants: ``hp.drift_mode`` changes ``geom.drift``, so switching to the full-P
    first moment hands Curly-FM the *same* new reference field our metric uses, and
    the comparison stays information-matched instead of quietly favouring us.

    Everything downstream is the authors' own code (``get_xt_xt_dot``, ``get_u_xt``,
    their drift-discrepancy ``coupling``, their MLP), driven at *our* iteration budget,
    learning rate and width so capacity and compute match the other arms.  Their MLP
    is fixed at three hidden layers, so ``hp.depth`` does not apply to this arm at the
    default ``hp.curly_net = "release"``; setting it to ``"shared"`` trains the engine's
    own ``PhiNet``/``VelocityNet`` instead — same losses, same coupling, same smoother —
    and is what the notebooks pass, so that there every arm has the identical network.

    Scoring caveat, repeated from :mod:`scripts.core.curly_fm`: on the fallback path
    the velocity metric ``cos(v_θ, b̂)`` uses this arm's own training target as the
    reference, so its cosine is near-ceiling by construction and carries no
    information.  Its W2 does.  (Where ``ts.velocity`` exists the reference field is
    an oracle none of the arms fit directly, and the cosine is informative again.)
    """
    geom = build_geometry("randers", ts, hp, device, dtype)
    if ts.velocity is not None:
        b_ref_np = np.asarray(ts.velocity, dtype=np.float64)
    else:
        with torch.no_grad():
            # b̂ sampled at the training cells; Curly-FM's kNN smoother interpolates it
            # to the off-lattice x_t its interpolant visits.
            b_ref_np = geom.drift(
                torch.as_tensor(ts.X, dtype=dtype, device=device)).cpu().numpy()
    cfg = CurlyConfig(
        geo_iters=hp.phase1_iters, vel_iters=hp.phase2_iters,
        batch_size=hp.batch_size, coupling_batch=hp.curly_coupling_batch,
        width=hp.width, k=hp.curly_k, sigma=hp.curly_sigma, alpha=hp.curly_alpha,
        lr=hp.lr, log_every=0, seed=seed, net=hp.curly_net, depth=hp.depth,
    )
    v_net, cdiag = train_curly_from_field(
        ts.X, b_ref_np, np.where(ts.p0)[0], np.where(ts.p1)[0],
        device, dtype, cfg)
    traj = integrate_flow(v_net, _source_cloud(ts, device, dtype), n_steps=n_steps)
    return ArmResult("curly", v_net, traj.cpu().numpy(),
                     {"curly": cdiag, "loss_phase1": cdiag["loss_geo"],
                      "loss_phase2": cdiag["loss_vel"],
                      # which field the baseline was actually trained on — a run
                      # record should never leave this to be inferred from the cloud
                      "velocity_source": ("measured" if ts.velocity is not None
                                          else "flux_from_P")},
                     {"geom": geom})


def run_pathb_iso(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Path B with **isotropic** noise: identical to ``pathb`` except M_t ≡ I.

    The Randers geometry still shapes the interpolant φ and the Finsler-OT coupling,
    so the *backbone* the bridge is built around is unchanged; only the noise the
    bridge injects around it loses its anisotropy, in both the conditional targets
    (A_t = σ²t(1−t)I, and the score target loses its M_t^{-1}) and the
    Euler–Maruyama step.  This isolates one question: does shaping the diffusion by
    the data's own mobility M = ρI + Σ buy anything over shaping only the mean path?
    With M ≡ I the sampler reduces to a bridge with the standard isotropic diffusion,
    i.e. SF²M with a learned interpolant.

    It is also the control for :func:`~scripts.core.metrics.score_second_moment`:
    its score network is *trained* on the identity, so the Bures distance between the
    mobility read out of its Jacobian and the data's M_P measures the anisotropy the
    isotropic bridge is blind to, and sets the scale for reading ``pathb``'s number.
    """
    geom = build_geometry("randers", ts, hp, device, dtype)
    # A second geometry object with the metric switched off.  Passing this to the
    # bridge (and only to the bridge) is what makes the ablation exact: metric_tensor
    # and metric_tensor_inv both return I, so M_t, ∂_tM_t, M_t^{-1} and M_t^{1/2} stay
    # mutually consistent instead of being patched at four separate call sites.
    geom_iso = build_geometry("euclidean", ts, hp, device, dtype)
    cfg = hp.train_config(seed)
    bcfg = hp.bridge_config(seed)
    product = EndpointCoupler(ts.X, ts.p0, ts.p1, device, dtype, seed=seed)

    phi, hist1 = train_phase1(geom, product, cfg, device, dtype)
    ot_coupler, ot_diag = build_ot_coupler(
        phi, geom, ts.X, ts.p0, ts.p1, device, dtype,
        K=hp.ot_K, blur_frac=hp.blur_frac, max_pts=hp.ot_max_pts, seed=seed)
    v_net, s_net, hist = train_sbm(phi, geom_iso, ot_coupler, bcfg, hp.sigma,
                                   device, dtype)
    traj = integrate_sde(v_net, s_net, geom_iso, _source_cloud(ts, device, dtype),
                         sigma=hp.sigma, n_steps=n_steps, seed=seed,
                         noise_off_after=hp.noise_off_after)
    diag = {"ot": ot_diag, "loss_phase1": _loss_tail(hist1), "loss_sbm": _loss_tail(hist),
            "sigma": hp.sigma, "isotropic_noise": True, "geometry": geom.verify()}
    return ArmResult("pathb_iso", v_net, traj.cpu().numpy(), diag,
                     {"phi": phi, "s_net": s_net, "geom": geom, "hist1": hist1,
                      "hist": hist})


def run_pathb_const(ts, hp, seed, device, dtype, n_steps=N_STEPS) -> ArmResult:
    """Path B with a **constant** mobility: M_t ≡ M̄, the data average of ρI + Σ(x).

    The control that isolates the two terms state- and time-dependence of the mobility
    forces into the bridge, and nothing else.  ``pathb`` carries both: the CFM target's
    ½·∂_tM_t·M_t^{-1}·δ, because M is read along the moving geodesic mean, and the
    sampler's σ²∇·M, because M is read at the wandering state.  Freeze M at one matrix
    and both are *identically* zero — not approximated away, not switched off by a flag:
    ∂_tM_t = 0 and ∇·M = 0 are properties of the tensor, and
    :class:`~scripts.core.bridge.ConstantMobility` is what makes the bridge see that.

    M̄ is the mean over the training cloud rather than I, which is what separates this
    arm from ``pathb_iso``.  The isotropic control removes the *anisotropy* of the noise
    as well (and rescales it: tr M̄/d is generally not 1, so its bridge runs at a
    different effective width at the same σ).  This one keeps both the average
    anisotropy and the average scale, so a gap against ``pathb`` reads as the value of
    letting the mobility *vary* over the manifold, which is the claim the geometry
    makes.  Backbone and coupling remain the full Randers ones in both.
    """
    geom = build_geometry("randers", ts, hp, device, dtype)
    # As in run_pathb_iso: one wrapper handed to the bridge and nothing else, so M_t,
    # ∂_tM_t, M_t^{-1} and M_t^{1/2} stay mutually consistent by construction.
    X_t = torch.as_tensor(ts.X, dtype=dtype, device=device)
    geom_const = ConstantMobility.from_points(geom, X_t)
    cfg = hp.train_config(seed)
    bcfg = hp.bridge_config(seed)
    product = EndpointCoupler(ts.X, ts.p0, ts.p1, device, dtype, seed=seed)

    phi, hist1 = train_phase1(geom, product, cfg, device, dtype)
    ot_coupler, ot_diag = build_ot_coupler(
        phi, geom, ts.X, ts.p0, ts.p1, device, dtype,
        K=hp.ot_K, blur_frac=hp.blur_frac, max_pts=hp.ot_max_pts, seed=seed)
    v_net, s_net, hist = train_sbm(phi, geom_const, ot_coupler, bcfg, hp.sigma,
                                   device, dtype)
    traj = integrate_sde(v_net, s_net, geom_const, _source_cloud(ts, device, dtype),
                         sigma=hp.sigma, n_steps=n_steps, seed=seed,
                         noise_off_after=hp.noise_off_after)
    diag = {"ot": ot_diag, "loss_phase1": _loss_tail(hist1), "loss_sbm": _loss_tail(hist),
            "sigma": hp.sigma, "constant_mobility": True,
            "mean_mobility_trace_over_d": float(
                geom_const.M0.diagonal().sum() / geom_const.M0.shape[0]),
            "geometry": geom.verify()}
    return ArmResult("pathb_const", v_net, traj.cpu().numpy(), diag,
                     {"phi": phi, "s_net": s_net, "geom": geom,
                      "geom_bridge": geom_const, "hist1": hist1, "hist": hist})


_RUNNERS = {
    "ffm": run_ffm,
    "ffm_riem": run_ffm_riem,
    "ffm_eucl": run_ffm_eucl,
    "ffm_drift": run_ffm_drift,
    "cfm": run_cfm,
    "mfm_land": run_mfm_land,
    "curly": run_curly,
    "pathb": run_pathb,
    "pathb_iso": run_pathb_iso,
    "pathb_const": run_pathb_const,
}
assert set(_RUNNERS) == set(ARMS), "ARMS and the runner table disagree"


def run_arm(arm: str, ts: TrainingSet, hp: HParams, seed: int,
            device: torch.device, dtype: torch.dtype = torch.float32,
            n_steps: int = N_STEPS) -> ArmResult:
    """Train one arm on one training set and push the source cloud forward."""
    assert arm in _RUNNERS, f"unknown arm {arm!r}; expected one of {ARMS}"
    assert n_steps % 3 == 0, (
        f"n_steps must be divisible by 3 so the band times t=k/3 land exactly on an "
        f"integration step, got {n_steps}")
    torch.manual_seed(seed)
    return _RUNNERS[arm](ts, hp, seed, device, dtype, n_steps=n_steps)
