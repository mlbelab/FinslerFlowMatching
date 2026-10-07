r"""Data-derived Randers/Finsler geometry (Path A).

Given points {x_i} and a transition matrix P, decompose

    P_sym  = (P + P^T) / 2 ,     P_asym = (P - P^T) / 2 ,

and define the *per-point* diffusion tensor and drift-current

    Σ_ab(x_i)      = 1/(2ε) Σ_j P_sym_ij (x_i - x_j)_a (x_i - x_j)_b       (d×d, PSD)
    Σ_ab^full(x_i) = 1/(2ε) Σ_j P_ij     (x_i - x_j)_a (x_i - x_j)_b       (d×d, PSD)
    J_i^a          =        Σ_j P_asym_ij (x_j - x_i)_a                     (d-vector)
    J_i^full       =        Σ_j P_ij      (x_j - x_i)_a                     (d-vector)

``drift_mode`` picks which first moment feeds b(x) and ``sigma_mode`` which second
moment feeds Σ(x) — see below.  All four are always accumulated (they cost two extra
``index_add_``\ s over the same edge list) so a run can report the ones it did not use
as diagnostics.

Which moment of P estimates which coefficient
---------------------------------------------
Hypothesis 1 reads the SDE off P by *splitting* it: the reversible half
carries the diffusion, the irreversible half carries the drift.  ``sigma_mode="sym"``
and ``drift_mode="asym"`` are that split, and they are the defaults.

The full-P moments are the other natural estimator, and they are not obviously worse.
For a chain approximating dX = b dt + √(2Σ_0) dW over a step of size ε, the *forward*
conditional moments are

    Σ_j P_ij (x_j - x_i)          = ε b + O(ε²)
    Σ_j P_ij (x_i-x_j)(x_i-x_j)ᵀ  = 2ε Σ_0 + ε² b bᵀ + O(ε³) ,

so both full-P moments are consistent as ε → 0; the second one is contaminated only at
second order, by the rank-one drift outer product.  What the symmetrisation buys is that
it cancels that contamination — and the density-gradient term with it — at the price of
mixing each row with its *reverse* edges, which is only a conditional distribution under
detailed balance.  So "which moment" is an empirical question about a finite sample at
finite ε, not a settled one, which is why both are switches rather than constants.

Two exact facts about the pair, both asserted at construction:

* **Σ^full is PSD too.**  Its weights are P_ij ≥ 0, so it is a non-negative combination
  of outer products exactly as Σ^sym is.  The difference Σ^full − Σ^sym = 1/(2ε) Σ_j
  P_asym_ij Δ_ij Δ_ijᵀ is symmetric but *indefinite*, so the two tensors differ in shape
  and not merely in scale.
* **Their mean traces are identical.**  ‖Δ_ij‖² is symmetric in (i, j) while P_asym is
  antisymmetric, so Σ_ij P_asym_ij ‖Δ_ij‖² = 0 identically.  The adaptive temperature ε
  (chosen so that mean_i tr Σ(x_i) ≈ ``target_trace``) is therefore the *same number*
  under either mode, and a comparison between them is a comparison of anisotropy and of
  the per-point redistribution of trace — never of overall scale.  That is what makes
  the switch a clean control rather than a rescaling in disguise.

These are extended to continuous space by kernel smoothing with
``k_ε(x, x_i) = exp(-||x - x_i||^2 / (4 ε^2))``:

    Σ_ab(x) = Σ_i k_ε Σ_ab(x_i) / (Σ_i k_ε + ε)
    b^a(x)  = Σ_i k_ε J_i^a     /  Σ_i k_ε        + ε_b

Two metric forms
----------------
Write ``G_0(x) = (ρ·I_d + Σ(x))^{-1}`` for the base (Riemannian) tensor.  Both forms
below are Randers metrics ``F(x, v) = sqrt(v^T G v) + β^T v``; they differ in how the
drift enters.

``metric_form="randers"`` (the original, and still the default)

    G(x) = G_0(x) ,   β(x) = -c · G_0(x) b(x) / (||b||_{G_0}(x) + ε_0) .

    The 1-form is the drift *direction* only: its ``G^{-1}``-norm is c by construction
    (up to ε_0), so the directional discount is the same everywhere and the magnitude
    of b never reaches F.

``metric_form="fw"`` — the Freidlin–Wentzell geometric action

    For dX = b dt + sqrt(2ε) σ dW with mobility A = σσ^T, the FW rate functional is
    S_T[φ] = (1/4) ∫_0^T (φ̇ - b)^T A^{-1} (φ̇ - b) dt.  Minimising over the travel
    time T (the Maupertuis reduction) leaves the reparameterisation-invariant

        S_geo[φ] = ∫ [ ||φ'||_{A^{-1}} · ||b||_{A^{-1}} - <φ', b>_{A^{-1}} ] ds ,

    which is exactly a Randers metric.  Identifying A with the empirical mobility
    M = ρI + Σ read off P_sym, so that A^{-1} = G_0, and relaxing the 1-form by c < 1:

        a(x) = ||b(x)||_{G_0(x)} ,   G(x) = a(x)^2 G_0(x) ,   β(x) = -c G_0(x) b(x) .

    Two properties make this the better-behaved form.  (i) Admissibility is exact:
    β^T G^{-1} β = c^2 identically, so c < 1 is the *whole* strong-convexity condition
    and no ε_0 regulariser is needed in the denominator.  (ii) At c = 1 Cauchy–Schwarz
    gives F >= 0 with equality iff v is parallel to b, i.e. flowing with the drift is
    free and every deviation is charged — the quasipotential, rather than a fixed
    percentage discount for heading the right way.  Because F now carries the *scale*
    of b as well as its direction, ``a`` is reported in units of its own data-set mean
    ā (a pure global rescale of F: it leaves the Phase-1 argmin and the entropic OT
    coupling, whose blur is a fraction of median C, exactly invariant) and floored by
    ``a_floor`` so that G stays positive definite where the drift vanishes.

    ``drift_mode="full"`` is the natural partner: the FW action wants the *whole*
    drift b_0, and Σ_j P_ij (x_j - x_i) is precisely the one-step conditional mean
    displacement.  The P_asym moment estimates only the non-gradient part, which is
    what the direction-only 1-form above needs but is not what b_0 is.

``metric_form="fw_lambda"`` — the same action, regularised in quadrature

    The ``fw`` form above reaches a valid metric by two separate devices: a relaxation
    c < 1 on the 1-form, and an additive ``a_floor`` keeping G positive definite where
    b vanishes.  Both are extra knobs, and neither falls out of the theory.  Writing
    D_ρ(x) = ρI + D(x) for the regularised second moment, the quadrature form

        F(x, v) = sqrt(v^T D_ρ^{-1} v) · sqrt(||b_0||²_{D_ρ^{-1}} + λ²)
                  - v^T D_ρ^{-1} b_0

    replaces both with the single λ.  In the a/G/β notation above that is

        a(x) = sqrt(||b||²_{G_0} + λ²) ,   G = a² G_0 ,   β = -G_0 b     (c ≡ 1)

    and the strong-convexity margin is then *exact and analytic*:

        β^T G^{-1} β = ||b||²_{G_0} / a²  =  1 - λ²/a²  <  1   for every λ > 0.

    So λ alone certifies the metric, no ε_0 and no a_floor; at λ → 0 it becomes the
    true FW quasipotential (F ≥ 0, zero iff v ∥ b), and as λ → ∞ the 1-form's relative
    strength a/λ → 0 and F degenerates to the pure Riemannian λ·sqrt(v^T G_0 v).  λ is
    therefore the FW-native version of c, related by

        c_eff(x) = ||β||_{G^{-1}} = ||b||_{G_0} / sqrt(||b||²_{G_0} + λ²) ,

    which is how its sweep grid is placed: λ is chosen in units of the measured
    data-set mean of ||b||_{G_0}, and that mean is exactly the ā already computed
    below.  λ also subsumes ``a_floor``'s job — in a data void b → 0, so a → λ and
    G → λ²G_0 stays positive definite on its own.

    Both moments come from the *full* P here (``drift_mode="full"``,
    ``sigma_mode="full"``): D̂_ρ(x_i) = (1/Δt) Σ_j P_ij ΔΔ^T and
    b̂_0(x_i) = (1/Δt) Σ_j P_ij Δ are the forward conditional moments the FW rate
    functional is written in terms of, not the P_sym / P_asym split.  The 1/Δt is the
    engine's ε temperature and cancels: F is reported in units of ā, a pure global
    rescale (see below), so only the *ratio* λ / ||b||_{G_0} is physically meaningful.

The isotropic floor ρ controls the on-/off-manifold cost contrast: where data lies
Σ has a large along-manifold eigenvalue, so G_0 is small (cheap); in data voids Σ→0
and G_0→(1/ρ)·I (cost 1/√ρ per unit length). Taking ρ ≪ min data-density scale
therefore imposes a *huge* penalty on cutting through voids relative to travelling
along the manifold, forcing geodesics onto the data.  Under ``fw`` that penalty is
modulated by a(x)^2, which is itself ≈ a_floor^2 in a void (no data, no drift), so ρ
and ``a_floor`` trade off against each other and are swept together.

**Metric vs mobility.**  G is the *cost* tensor: it must be large where motion is
expensive, hence the inversion.  The *mobility* (diffusion) tensor is

    M(x) = ρ·I_d + Σ(x) ,

which is (up to the floor ρ) the empirical diffusion coefficient Σ_0 that Hypothesis 1
identifies with the reversible part of P.  Anything that spreads mass —
Path B's conditional covariance A_t and its injected Brownian increment — must use M,
not G, so that noise runs *along* high-probability directions rather than across them.
Under ``randers`` M is literally G^{-1}.  Under ``fw`` it is **not**: the conformal
factor a^2 comes from reparameterising *time* in the FW action, and a change of clock
must not change the diffusion tensor of the SDE.  So the two are separate accessors —
:meth:`mobility` / :meth:`mobility_inv` are the diffusion pair (= G_0^{-1}, G_0) and
:meth:`metric_tensor` / :meth:`cost_tensor_inv` are the cost pair — and they coincide
only in the legacy form.  ``metric_tensor_inv`` is kept as an alias of
:meth:`mobility`, which is the sense every existing caller used it in.  All of them
stay consistent under the ``use_metric`` ablation (all collapse to I).

Everything is implemented in torch and is differentiable in the query point ``x``
so that F(x_t, ẋ_t) can be back-propagated through the interpolant network.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor


#: the three shapes the Randers metric can take — see the module docstring
METRIC_FORMS = ("randers", "fw", "fw_lambda")

#: the two that carry the FW conformal factor G = a²G_0, and so need ā
FW_FORMS = ("fw", "fw_lambda")
#: which first moment of P estimates the drift b
DRIFT_MODES = ("asym", "full")
#: which second moment of P estimates the diffusion Σ
SIGMA_MODES = ("sym", "full")

class FinslerGeometry:
    """Randers metric F(x, v) derived from a point cloud + transition matrix."""

    def __init__(
        self,
        X: np.ndarray,
        P: np.ndarray,
        eps_kernel: float | None = None,   # ε   : smoothing bandwidth
        eps_kernel_scale: float = 1.0,      # multiplies the auto-chosen bandwidth
        target_trace: float = 2.0,          # sets ε (Σ temperature) adaptively
        rho: float = 0.03,                  # ρ   : isotropic floor of M = ρI+Σ (small!)
        c: float = 0.9,                     # Randers 1-form strength, must be < 1
        metric_form: str = "randers",       # "randers"|"fw"|"fw_lambda" (module docstring)
        drift_mode: str = "asym",           # "asym" | "full"   first moment for b
        sigma_mode: str = "sym",            # "sym"  | "full"   second moment for Σ
        a_floor: float = 0.1,               # floor on a/ā, keeps G_fw pos. def. at b=0
        fw_lambda: float = 0.03,            # λ: the quadrature regulariser of "fw_lambda"
        use_one_form: bool = True,          # False -> symmetric Riemannian (MFM ablation)
        use_metric: bool = True,            # False -> G ≡ I_d (Euclidean ablation)
        eps0: float = 1e-3,                 # ε_0 : β denominator regulariser (randers)
        eps_b: float = 1e-6,                # ε_b : drift regulariser
        eps_denom: float = 1e-6,            # ε   : Σ(x) denominator regulariser
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert metric_form in METRIC_FORMS, (
            f"unknown metric_form {metric_form!r}, expected {METRIC_FORMS}")
        if metric_form == "fw_lambda":
            # The quadrature form carries no c: λ is the whole strong-convexity
            # certificate (β^T G^{-1} β = 1 - λ²/a² < 1), and multiplying β by a second
            # relaxation would be two knobs for one job.  Demanded explicitly rather
            # than ignored, so a c left over in an HParams grid fails loudly.
            assert c == 1.0, (
                f"metric_form='fw_lambda' is the unrelaxed FW action and requires "
                f"c = 1.0 exactly (got {c}); admissibility comes from fw_lambda={fw_lambda}")
            assert fw_lambda > 0.0, (
                "fw_lambda must be positive: at λ = 0 the metric is the exact "
                "quasipotential, F vanishes along b, and no geodesic problem is posed")
        else:
            assert 0.0 < c < 1.0, (
                "Randers strength c must lie in (0, 1) for a valid metric")
        assert rho > 0.0, "rho must be positive to keep G positive-definite"
        assert drift_mode in DRIFT_MODES, (
            f"unknown drift_mode {drift_mode!r}, expected {DRIFT_MODES}")
        assert sigma_mode in SIGMA_MODES, (
            f"unknown sigma_mode {sigma_mode!r}, expected {SIGMA_MODES}")
        assert a_floor > 0.0 or metric_form != "fw", (
            "the FW form needs a_floor > 0: where the drift vanishes a -> 0 and "
            "G = a^2 G_0 would be singular")
        assert eps_kernel_scale > 0.0, "the bandwidth scale must be positive"
        self.device = torch.device(device)
        self.dtype = dtype
        self.rho = float(rho)
        self.c = float(c)
        self.metric_form = str(metric_form)
        self.drift_mode = str(drift_mode)
        self.sigma_mode = str(sigma_mode)
        self.target_trace = float(target_trace)
        self.a_floor = float(a_floor)
        self.fw_lambda = float(fw_lambda)
        self.use_one_form = bool(use_one_form)
        # G ≡ I_d ablation.  Read at *query* time (not baked into any buffer), so it
        # may also be flipped after construction.  Σ_pts / J_pts are still built, so
        # ``drift`` — and hence β and the empirical-flux diagnostics — remain available;
        # only the Riemannian part of F is neutralised.  Setting both this and
        # ``use_one_form`` to False makes F(x, v) = ‖v‖ exactly (pure Euclidean).
        self.use_metric = bool(use_metric)
        self.eps0 = float(eps0)
        self.eps_b = float(eps_b)
        self.eps_denom = float(eps_denom)

        Xt = torch.as_tensor(X, dtype=dtype, device=self.device)
        N, d = Xt.shape
        self.N, self.d = N, d
        self.X = Xt

        # ------------------------------------------------------------------ #
        #  Sparse-edge construction of the per-point Σ_i and J_i.
        #
        #  Σ_i and J_i only sum over j with P^sym_ij ≠ 0 / P^asym_ij ≠ 0.  For a
        #  k-NN transition matrix (e.g. a CellRank velocity+similarity kernel)
        #  each row has O(k) nonzeros, so the dense (N,N,d) and (N,N,d,d) tensors
        #  the naive form would build (2.7 GB / 137 GB at N=3696, d=50) are almost
        #  entirely zeros.  We instead iterate over the *edge list* of the union
        #  nonzero pattern of P and its transpose, accumulating into the small
        #  (N,d,d) and (N,d) buffers with ``index_add_``.  For a dense P with
        #  exact zeros off the k-NN graph (the synthetic benchmarks) this yields
        #  Σ_pts / J_pts numerically identical to the old einsum form, since the
        #  omitted terms contribute exactly zero.
        # ------------------------------------------------------------------ #
        P_csr = self._to_csr(P)                             # (N, N) row-stochastic
        # P^sym = (P+Pᵀ)/2 and P^asym = (P-Pᵀ)/2 share the union sparsity pattern.
        Psym = (P_csr + P_csr.T) * 0.5
        Pasym = (P_csr - P_csr.T) * 0.5
        Psym = Psym.tocoo()
        Pasym_csr = Pasym.tocsr()
        rows = torch.as_tensor(Psym.row.astype(np.int64), device=self.device)
        cols = torch.as_tensor(Psym.col.astype(np.int64), device=self.device)
        w_sym = torch.as_tensor(Psym.data, dtype=dtype, device=self.device)
        # align P^asym values to the same (row, col) ordering as P^sym
        w_asym = torch.as_tensor(
            np.asarray(Pasym_csr[Psym.row, Psym.col]).ravel(),
            dtype=dtype, device=self.device,
        )
        E = rows.numel()
        chunk = 8192

        # --- pass 1: adaptive Σ temperature ε so that mean_i tr Σ(x_i)≈target_trace
        # tr Σ_i = 1/(2ε) Σ_j P_sym_ij ||Δ_ij||^2  ->  choose ε from the raw scale.
        # ``raw_full`` is the same sum under the full-P weights.  It is *not* used to set
        # ε — the two are equal in exact arithmetic (see the module docstring: P_asym is
        # antisymmetric and ‖Δ‖² is symmetric, so their contraction vanishes), so setting
        # the temperature from the symmetric sum in both modes both keeps every legacy
        # number bit-identical and makes the modes trace-matched by construction.  It is
        # accumulated only so the cancellation can be asserted rather than assumed.
        raw = torch.zeros((), dtype=dtype, device=self.device)
        raw_full = torch.zeros((), dtype=dtype, device=self.device)
        for s in range(0, E, chunk):
            sl = slice(s, min(s + chunk, E))
            de = Xt[rows[sl]] - Xt[cols[sl]]                # (c, d)  Δ_ij = x_i - x_j
            sq = (de ** 2).sum(-1)                          # (c,)   ‖Δ_ij‖², symmetric
            raw = raw + (w_sym[sl] * sq).sum()
            raw_full = raw_full + ((w_sym[sl] + w_asym[sl]) * sq).sum()
        # matches the old ``(P_sym * ||Δ||^2).sum(1).mean()`` (mean over the N rows)
        self.eps_sigma = float(raw / N / (2.0 * target_trace) + 1e-12)
        trace_gap = float((raw_full - raw).abs() / (raw.abs() + 1e-30))
        # float32 accumulation over E edges, so this is a round-off tolerance, not a
        # modelling one; anything larger means the edge alignment of w_asym is wrong.
        assert trace_gap < 1e-3, (
            f"Σ_j P_asym_ij ‖Δ_ij‖² should vanish identically, but the full-P and "
            f"symmetric total traces differ by {trace_gap:.2e} — w_asym is misaligned "
            f"with the P^sym edge ordering")
        self.trace_gap = trace_gap

        # --- pass 2: accumulate both second moments (N,d,d) and both first moments
        #     (N,d) over the same edge list.  The full-P buffers cost one extra
        #     ``index_add_`` each and (for Σ) one extra (N,d,d) allocation; that is
        #     37 MB at the largest cloud in the repo (N=3696, d=50) and 140 kB on the
        #     synthetic benchmarks, which is the price of being able to report the
        #     estimator a run did *not* use.
        #
        #     The sums are taken on the CPU.  ``index_add_`` on a CUDA tensor accumulates
        #     with atomics, so its summation order -- and the last bits of every moment,
        #     ~1e-15 at float64 -- changed from one build to the next, and training is
        #     sensitive enough to turn that into a different run (the same iTracer cell
        #     re-run gave W2 21.0 and 1.44).  On the CPU the accumulation is sequential, so
        #     the geometry is bit-reproducible; the products are still formed on device.
        acc = torch.device("cpu")
        self.Sigma_sym_pts = torch.zeros((N, d, d), dtype=dtype, device=acc)
        self.Sigma_full_pts = torch.zeros((N, d, d), dtype=dtype, device=acc)
        self.J_asym_pts = torch.zeros((N, d), dtype=dtype, device=acc)
        self.J_full_pts = torch.zeros((N, d), dtype=dtype, device=acc)
        inv_two_eps = 1.0 / (2.0 * self.eps_sigma)
        for s in range(0, E, chunk):
            sl = slice(s, min(s + chunk, E))
            r = rows[sl]
            de = Xt[r] - Xt[cols[sl]]                       # (c, d)  Δ_ij = x_i - x_j
            outer = de[:, :, None] * de[:, None, :]         # (c, d, d)
            # neither the full first nor the full second moment needs a second sparse
            # lookup: on the union pattern P = P_sym + P_asym holds edge-by-edge, so the
            # full-P weight is the sum of the two already-aligned edge weights.
            w_full = w_sym[sl] + w_asym[sl]
            r_acc = r.to(acc)
            self.Sigma_sym_pts.index_add_(
                0, r_acc, ((w_sym[sl, None, None] * outer) * inv_two_eps).to(acc))
            self.Sigma_full_pts.index_add_(
                0, r_acc, ((w_full[:, None, None] * outer) * inv_two_eps).to(acc))
            # J_i^a = Σ_j P_asym_ij (x_j - x_i) = -Σ_j P_asym_ij Δ_ij
            self.J_asym_pts.index_add_(0, r_acc, (w_asym[sl, None] * (-de)).to(acc))
            self.J_full_pts.index_add_(0, r_acc, (w_full[:, None] * (-de)).to(acc))
        self.Sigma_sym_pts = self.Sigma_sym_pts.to(self.device)
        self.Sigma_full_pts = self.Sigma_full_pts.to(self.device)
        self.J_asym_pts = self.J_asym_pts.to(self.device)
        self.J_full_pts = self.J_full_pts.to(self.device)

        #: the moments b(x) and Σ(x) are smoothed from.  Selecting here rather than
        #: branching in :meth:`drift` / :meth:`sigma` keeps every downstream caller
        #: (Path B's mobility, Curly-FM's reference field, the β diagnostics)
        #: automatically consistent with the geometry it was handed.
        self.J_pts = (self.J_full_pts if self.drift_mode == "full"
                      else self.J_asym_pts)
        self.Sigma_pts = (self.Sigma_full_pts if self.sigma_mode == "full"
                          else self.Sigma_sym_pts)

        # kernel bandwidth ε (default: median nearest-neighbour distance)
        if eps_kernel is None:
            with torch.no_grad():
                dmat = torch.cdist(Xt, Xt)
                dmat.fill_diagonal_(float("inf"))
                nn_dist = dmat.min(dim=1).values
                # over the *positive* nearest-neighbour distances.  A cloud drawn with
                # replacement -- which the Curly-FM release's circles are, and which real
                # data with tied measurements can be -- carries coincident rows whose
                # nearest neighbour is at distance 0.  Those pairs report no length
                # scale, so including them drags the median toward zero and, past a
                # duplicate fraction of a half, makes it exactly zero.  Dropping them
                # leaves the bandwidth as "the typical spacing between *distinct*
                # points", which is what the smoothing kernel wants in either case.
                positive = nn_dist[nn_dist > 0]
                assert positive.numel() > 0, (
                    "every sample coincides with another; the cloud carries no length "
                    "scale and no smoothing bandwidth exists")
                eps_kernel = float(positive.median())
        self.eps_kernel = float(eps_kernel) * float(eps_kernel_scale)
        assert self.eps_kernel > 0.0, "degenerate smoothing bandwidth"

        self.I = torch.eye(d, dtype=dtype, device=self.device)

        # ------------------------------------------------------------------ #
        #  ā: the data-set mean of a(x) = ||b||_{G_0}, used to report the FW
        #  conformal factor in its own units.  Dividing F by ā is a pure global
        #  rescale — it changes neither the Phase-1 minimiser nor the entropic OT
        #  coupling (whose blur is a fraction of median C) — but it keeps the
        #  Phase-1 loss at O(1) whatever the units of X are, so one learning rate
        #  works for both metric forms.  Computed on the data points themselves,
        #  in chunks, because the kernel is (chunk, N).
        # ------------------------------------------------------------------ #
        self.a_bar = 1.0
        #: mean ||b||_{G_0} on the data, *before* the λ quadrature.  Under ``fw`` this is
        #: ā itself; under ``fw_lambda`` it is the scale λ has to be read against, and
        #: it is what the sweep grid for λ is placed in units of, so it is reported
        #: separately rather than folded into ā.
        self.a_raw_bar = 1.0
        #: the *largest* ||b||_{G_0} on the data.  ā sets what λ is quoted against; this
        #: sets whether λ is representable at all.  The admissibility margin under
        #: ``fw_lambda`` is 1 - β^T G^-1 β = λ²/(a_raw² + λ²), so it is smallest exactly
        #: where a_raw is largest, and a ladder whose bottom rung puts that minimum
        #: below the working dtype's resolution is a rung on which β^T G^-1 β rounds to
        #: 1 and the metric stops being Finsler.  :func:`scripts.core.scales.resolve`
        #: refuses such a rung, and needs the max rather than the mean to do it.
        self.a_raw_max = 1.0
        if self.metric_form in FW_FORMS:
            with torch.no_grad():
                tot, tot_raw, hi, seen = 0.0, 0.0, 0.0, 0
                for s in range(0, N, 1024):
                    xs = Xt[s:s + 1024]
                    G0s, bs = self._g0(xs), self.drift(xs)
                    a_raw_s = self._a_raw(xs, G0s, bs)
                    tot += float(self._a_unnorm(xs, G0s, bs).sum())
                    tot_raw += float(a_raw_s.sum())
                    hi = max(hi, float(a_raw_s.max()))
                    seen += int(xs.shape[0])
                self.a_bar = tot / max(seen, 1)
                self.a_raw_bar = tot_raw / max(seen, 1)
                self.a_raw_max = hi
            assert self.a_bar > 0.0, (
                "mean ||b||_{G_0} vanished on the data — P has no first moment at all, "
                "so the FW metric is undefined")

        # diagnostics
        with torch.no_grad():
            evals = torch.linalg.eigvalsh(self.Sigma_pts)   # (N, d)
            self.sigma_eig_range = (float(evals.min()), float(evals.max()))
            self.J_norm_mean = float(self.J_pts.norm(dim=1).mean())
            self.J_asym_norm_mean = float(self.J_asym_pts.norm(dim=1).mean())
            self.J_full_norm_mean = float(self.J_full_pts.norm(dim=1).mean())
            # mean traces: equal by the exact cancellation above, so a divergence here
            # is the single number that would say the switch had become a rescaling
            tr_sym = self.Sigma_sym_pts.diagonal(dim1=1, dim2=2).sum(1)
            tr_full = self.Sigma_full_pts.diagonal(dim1=1, dim2=2).sum(1)
            self.sigma_trace_mean = {"sym": float(tr_sym.mean()),
                                     "full": float(tr_full.mean())}
            # how far apart the two estimators are *per point*, in units of the
            # symmetric one.  Zero would mean P is reversible on this cloud and the
            # switch cannot matter; the Sheet's velocity kernel puts it well above.
            diff = (self.Sigma_full_pts - self.Sigma_sym_pts).flatten(1).norm(dim=1)
            base = self.Sigma_sym_pts.flatten(1).norm(dim=1)
            self.sigma_rel_diff_mean = float((diff / (base + 1e-30)).mean())

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_csr(P) -> "sp.csr_matrix":
        """Coerce a dense ndarray / torch tensor / scipy-sparse P to CSR float64."""
        if sp.issparse(P):
            return P.tocsr().astype(np.float64)
        if isinstance(P, torch.Tensor):
            P = P.detach().cpu().numpy()
        return sp.csr_matrix(np.asarray(P, dtype=np.float64))

    # ------------------------------------------------------------------ #
    #  Continuous fields
    # ------------------------------------------------------------------ #
    def _kernel(self, x: Tensor) -> Tensor:
        """k_ε(x, x_i) = exp(-||x - x_i||^2 / (4 ε^2))   ->  (B, N).

        ||x - x_i||^2 = |x|^2 + |x_i|^2 - 2 x·x_i  keeps this a (B,N) matmul with no
        (B,N,d) intermediate (378 MB at B=512,N=3696,d=50).
        """
        x2 = (x ** 2).sum(1, keepdim=True)                  # (B, 1)
        xi2 = (self.X ** 2).sum(1)[None, :]                 # (1, N)
        sqd = torch.clamp(x2 + xi2 - 2.0 * (x @ self.X.T), min=0.0)   # (B, N)
        return torch.exp(-sqd / (4.0 * self.eps_kernel ** 2))

    def sigma(self, x: Tensor) -> Tensor:
        """Smoothed diffusion tensor Σ(x)  ->  (B, d, d).

        Implemented as a single (B,N)@(N,d*d) matmul rather than an einsum so that
        no (B,N,d,d) intermediate is ever materialised (~14 GB at B=512,N=3696,d=50).
        """
        k = self._kernel(x)                                 # (B, N)
        num = (k @ self.Sigma_pts.reshape(self.N, self.d * self.d)).reshape(
            -1, self.d, self.d)
        den = k.sum(1)[:, None, None] + self.eps_denom
        return num / den

    def mobility_divergence(self, x: Tensor) -> Tensor:
        """(∇·M)_i = Σ_j ∂_j M_ij in closed form  ->  (B, d).

        Path B's sampler needs this at every step (see
        :func:`scripts.core.bridge.mobility_divergence` for why).  Taken by AD it costs d
        passes over the smoother, which at d = 50 is ~150x a plain ``mobility`` call and
        turns a Path-B run from seconds into tens of minutes.  It does not have to be: M
        is a Nadaraya–Watson average of *fixed* matrices under a Gaussian kernel, so its
        divergence is available in closed form at the price of two more (B,N)@(N,d)
        matmuls — the same order as reading M itself.

        With k_i = exp(-||x-x_i||²/4ε²), S = Σ_i k_i + ε_den and Σ = (Σ_i k_i D_i)/S,

            ∂_j k_i = -k_i (x - x_i)_j / 2ε²
            ∂_j Σ_ab = Σ_i ∂_j k_i (D_i,ab - Σ_ab) / S

        and contracting b = j and telescoping the sum over i leaves

            ∇·M = -[ (ε_den/S)·(Σ_i k_i D_i)x - Σ_i k_i D_i x_i + Σ·(Σ_i k_i x_i) ] / (2ε²S)

        exact, regulariser included.  ρI contributes nothing, and neither does the ablation
        M ≡ I — both are constant.  the verification suite in the research repository checks this against AD and
        against finite differences, which is the only reason a hand-derived gradient
        belongs in a hot loop.
        """
        if not self.use_metric:
            return torch.zeros_like(x)
        if getattr(self, "_sigma_x_pts", None) is None:
            # (Σ_i x_i) per point, the one extra (N, d) table the closed form needs
            self._sigma_x_pts = torch.einsum(
                "nab,nb->na", self.Sigma_pts, self.X).contiguous()
        k = self._kernel(x)                                       # (B, N)
        S = k.sum(1)[:, None] + self.eps_denom                    # (B, 1)
        A = (k @ self.Sigma_pts.reshape(self.N, self.d * self.d)).reshape(
            -1, self.d, self.d)                                   # Σ_i k_i D_i
        sig = A / S[:, :, None]                                   # Σ(x)
        u = k @ self._sigma_x_pts                                 # Σ_i k_i D_i x_i
        m = k @ self.X                                            # Σ_i k_i x_i
        bracket = (self.eps_denom / S) * torch.einsum("bij,bj->bi", A, x) \
            - u + torch.einsum("bij,bj->bi", sig, m)
        return -bracket / (2.0 * self.eps_kernel ** 2 * S)

    def drift(self, x: Tensor) -> Tensor:
        """Smoothed non-gradient drift b(x)  ->  (B, d)."""
        k = self._kernel(x)
        num = k @ self.J_pts                                 # (B, N)@(N, d) -> (B, d)
        # NOTE: the kernel underflows to exactly 0 in data voids (e.g. the centre
        # of the Cycle ring). Regularise the denominator so b(x) -> 0 there rather
        # than 0/0 = NaN. This makes the metric gracefully Euclidean in voids.
        den = k.sum(1)[:, None] + self.eps_denom
        return num / den + self.eps_b

    def _g0(self, x: Tensor) -> Tensor:
        """Base Riemannian tensor G_0(x) = (ρ·I + Σ(x))^{-1}, ablation switch ignored.

        Private because every *public* accessor has to honour ``use_metric``; this one
        deliberately does not, so that ā can be measured from the real geometry even on
        an object whose metric is switched off.
        """
        return torch.linalg.inv(self.rho * self.I[None] + self.sigma(x))

    def _a_raw(self, x: Tensor, G0: Tensor, b: Tensor) -> Tensor:
        """a(x) = ||b(x)||_{G_0(x)}  ->  (B,).  Unnormalised, unfloored, λ-free."""
        q = torch.einsum("bi,bij,bj->b", b, G0, b)
        return torch.sqrt(torch.clamp(q, min=0.0))

    def _a_unnorm(self, x: Tensor, G0: Tensor, b: Tensor) -> Tensor:
        """The FW conformal factor before the ā rescale  ->  (B,).

        ``fw``         ||b||_{G_0}                    — floored later, additively
        ``fw_lambda``  sqrt(||b||²_{G_0} + λ²)        — already floored, in quadrature

        ā is the data-set mean of *this*, so ``conformal_factor`` has mean ≈ 1 under
        either form and one Phase-1 learning rate serves both.
        """
        a2 = torch.einsum("bi,bij,bj->b", b, G0, b).clamp(min=0.0)
        if self.metric_form == "fw_lambda":
            return torch.sqrt(a2 + self.fw_lambda ** 2)
        return torch.sqrt(a2)

    def conformal_factor(self, x: Tensor, G0: Tensor | None = None,
                         b: Tensor | None = None) -> Tensor:
        """FW conformal factor a_n(x)  ->  (B,), normalised to mean ≈ 1 on the data.

        ``fw``         a_n = ||b||_{G_0} / ā + a_floor
        ``fw_lambda``  a_n = sqrt(||b||²_{G_0} + λ²) / ā     — λ *is* the floor

        Identically 1 under ``metric_form="randers"`` and under the ``use_metric``
        ablation, so ``G = a_n^2 G_0`` is the single expression for all three forms and
        the Euclidean rung of the ladder really is F(x,v) = ||v|| in every one.
        """
        if self.metric_form not in FW_FORMS or not self.use_metric:
            return torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        if G0 is None:
            G0 = self._g0(x)
        if b is None:
            b = self.drift(x)
        a_n = self._a_unnorm(x, G0, b) / self.a_bar
        # the quadrature form needs no additive floor; adding one would break the exact
        # β^T G^{-1} β = 1 - λ²/a² identity that certifies it
        return a_n if self.metric_form == "fw_lambda" else a_n + self.a_floor

    # ------------------------------------------------------------------ #
    #  The two measured scales a ladder is quoted against
    # ------------------------------------------------------------------ #
    @property
    def rho_star(self) -> float:
        """``mean_i tr Σ(x_i) / d`` — the scale the ρ ladder is placed in units of.

        A *measured* quantity read back off the accumulated moments, not a restatement
        of the input: the temperature ε pins ``mean_i tr Σ`` to ``target_trace``
        (see the ε calibration above), so this comes out at ``target_trace / d`` and the
        equality is the check that the pinning happened.  ρ is a floor on a tensor of
        squared displacements, so its natural unit is that tensor's own mean eigenvalue —
        ρ far below this regularises nothing and ρ far above it drowns the data's
        anisotropy — and quoting it as ``rho_mult · rho_star`` is what makes one rung
        mean the same thing at d = 2 and at d = 50, where the absolute values differ by
        a factor of 25.

        The companion scale for λ is :attr:`a_raw_bar`, which unlike this one cannot be
        known before the geometry is built: it depends on ρ through G_0 = (ρI + Σ)^-1.
        """
        return float(self.Sigma_pts.diagonal(dim1=1, dim2=2).sum(1).mean()) / self.d

    def mean_mobility_eig(self, chunk: int = 512) -> float:
        """``m_bar = mean_i tr M(x_i) / d``: the scale a bridge width is quoted against.

        A conditional bridge has per-direction std ``σ·sqrt(t(1-t)·m_bar)``, so quoting σ
        as a multiple of the smoothing bandwidth — see
        :func:`scripts.core.scales.sigma_from_width` — needs this and nothing else.  The
        same object as :meth:`scripts.method.metric.Metric.mean_mobility_eig`, computed
        the same way, so the two trees agree on what a ``width_mult`` buys.

        This is ``ρ + rho_star`` in closed form; it is summed rather than asserted
        because the ``use_metric`` ablation makes it something else.
        """
        with torch.no_grad():
            tot = sum(float(self.mobility(self.X[s:s + chunk])
                            .diagonal(dim1=-2, dim2=-1).sum())
                      for s in range(0, self.N, chunk))
        return tot / (self.N * self.d)

    def mobility(self, x: Tensor) -> Tensor:
        """Mobility (diffusion) tensor M(x)  ->  (B, d, d).

        ``M = ρI + Σ(x)``, the published form.

        Computed without an inversion, and kept in sync with :meth:`mobility_inv`
        under the ``use_metric`` ablation so callers that need both (Path B,
        :meth:`verify`) can never disagree about which geometry is in force.

        This is the tensor that shapes *diffusion*: Path B's conditional covariance
        A_t = σ²t(1-t)·M_t and its SDE increment √2·σ·M_t^{1/2}dW both use it, so the
        injected noise is largest along the directions in which P actually spreads
        mass.  It is also the reference M_P against which
        :func:`~scripts.core.metrics.score_second_moment` compares the
        second moment read off the learned score network.

        Under ``metric_form="fw"`` this is *not* the inverse of :meth:`metric_tensor`:
        the FW conformal factor rescales the clock, not the noise.  See the module
        docstring.
        """
        if not self.use_metric:
            return self.I[None].expand(x.shape[0], self.d, self.d)
        return self.rho * self.I[None] + self.sigma(x)

    #: historical name for :meth:`mobility`.  Every existing call site used it in the
    #: mobility sense (Path B's M_t, the score-Jacobian reference), which is exactly
    #: what it still returns; only the FW form makes "the inverse of the metric" and
    #: "the mobility" two different tensors, and those callers want this one.
    metric_tensor_inv = mobility

    @property
    def constant_mobility(self) -> bool:
        """Is :meth:`mobility` the same matrix at every x (and hence at every t)?

        True exactly under the G ≡ I ablation, where M(x) = I identically.  Path B reads
        this to skip two things that are then provably zero: the CFM target's ∂_tM_t and
        the sampler's Itô drift σ²∇·M.  See :func:`scripts.core.bridge.mobility_divergence`
        and :class:`scripts.core.bridge.ConstantMobility`.
        """
        return not self.use_metric

    def base_metric(self, x: Tensor) -> Tensor:
        """G_0(x) = (ρI + Σ(x))^{-1}, honouring the G ≡ I ablation  ->  (B, d, d).

        Every caller that wants "the metric" rather than "the inverse diffusion" — the
        Randers 1-form, above all — asks for this one.
        """
        if not self.use_metric:
            return self.I[None].expand(x.shape[0], self.d, self.d)
        return self._g0(x)

    def mobility_inv(self, x: Tensor) -> Tensor:
        """M(x)^{-1}  ->  (B, d, d).

        This *is* the base metric G_0 and costs nothing extra.
        """
        return self.base_metric(x)

    def metric_tensor(self, x: Tensor) -> Tensor:
        """Cost tensor G(x) = a_n(x)^2 · G_0(x)  ->  (B, d, d).

        This is what enters F(x,v) = sqrt(vᵀGv) + βᵀv: large where motion is expensive.
        Under ``metric_form="randers"`` a_n ≡ 1 and G = G_0 = (ρI + Σ)^{-1}; under
        ``"fw"`` the conformal factor carries the magnitude of the drift.

        With ``use_metric=False`` this collapses to the identity, i.e. the Euclidean
        ablation of the data-derived geometry.
        """
        return self._cost_fields(x, want_beta=False)[0]

    def cost_tensor_inv(self, x: Tensor) -> Tensor:
        """G(x)^{-1} = a_n(x)^{-2} · (ρI + Σ)  ->  (B, d, d).

        Only the strong-convexity check needs this; the diffusion wants
        :meth:`mobility`, which is the same tensor *without* the conformal factor.
        """
        if not self.use_metric:
            return self.I[None].expand(x.shape[0], self.d, self.d)
        a_n = self.conformal_factor(x)
        return self.mobility(x) / (a_n ** 2)[:, None, None]

    def one_form(self, x: Tensor, G: Tensor | None = None) -> Tensor:
        """Admissible Randers 1-form β(x)  ->  (B, d).

        ``randers``  β = -c G_0 b / (||b||_{G_0} + ε_0)   — direction only, ||β|| ≈ c
        ``fw``       β = -c G_0 b / ā                     — carries the drift magnitude

        ``G`` is accepted (and ignored under ``fw``, which needs G_0 rather than the
        conformally rescaled cost tensor) only so that the legacy call
        ``one_form(x, G=metric_tensor(x))`` still avoids recomputing Σ.
        """
        if self.metric_form == "randers" and G is not None:
            G0 = G
        else:
            G0 = self.base_metric(x)
        return self._beta(x, G0, self.drift(x))

    def _beta(self, x: Tensor, G0: Tensor, b: Tensor) -> Tensor:
        """β from an already-computed (G_0, b) pair — see :meth:`one_form`."""
        G0b = torch.einsum("bij,bj->bi", G0, b)             # (B, d)
        if self.metric_form in FW_FORMS:
            # Both FW forms take β = -c G_0 b, divided by ā only because G carries the
            # same 1/ā (so F/ā is a *pure* global rescale and admissibility is untouched).
            #   fw         ||β||_{G^{-1}} = c·(a_raw/ā)/a_n <= c exactly — no ε_0 needed
            #   fw_lambda  c ≡ 1, so it is a_raw/sqrt(a_raw²+λ²) = sqrt(1 - λ²/a²) < 1
            return -self.c * G0b / self.a_bar
        norm_G_b = torch.sqrt(torch.clamp((b * G0b).sum(1), min=0.0))
        return -self.c * G0b / (norm_G_b[:, None] + self.eps0)

    def _cost_fields(self, x: Tensor, want_beta: bool = True):
        """``(G_cost, β or None)`` with Σ(x) and b(x) each evaluated exactly once.

        F needs both tensors and, under ``fw``, so does the conformal factor; going
        through the public accessors would smooth the kernel four times per call.
        """
        if not self.use_metric:
            G0 = self.I[None].expand(x.shape[0], self.d, self.d)
        else:
            G0 = self._g0(x)
        need_b = want_beta and self.use_one_form
        if self.metric_form in FW_FORMS and self.use_metric:
            b = self.drift(x)
            a_n = self.conformal_factor(x, G0, b)
            G = (a_n ** 2)[:, None, None] * G0
        else:
            b = self.drift(x) if need_b else None
            G = G0
        if not need_b:
            return G, None
        if b is None:
            b = self.drift(x)
        return G, self._beta(x, G0, b)

    # ------------------------------------------------------------------ #
    #  The metric
    # ------------------------------------------------------------------ #
    def F(self, x: Tensor, v: Tensor, eps_sqrt: float = 1e-9) -> Tensor:
        """Randers norm F(x, v) = sqrt(v^T G v) + β^T v   ->  (B,).

        With ``use_one_form=False`` the β term is dropped and F reduces to the
        symmetric Riemannian norm sqrt(v^T G v) — the Metric-Flow-Matching (MFM)
        ablation of the full (Finsler/Randers) metric.  Adding ``use_metric=False``
        drops G as well, leaving F(x, v) = ‖v‖ — the Euclidean ablation.  The three
        settings give the metric ladder used in the ablation table:

            (G=I,   β=0)    Euclidean
            (G=G_P, β=0)    Riemannian   (from P^sym only)
            (G=G_P, β=β_P)  Randers      (adds the P^asym flux)

        Under ``metric_form="fw"`` the middle rung keeps the conformal factor, so it is
        F = a_n·sqrt(vᵀG_0v): the drift still sets *how expensive* a region is, and only
        the direction discount is removed.  That is the right knock-out for the FW form —
        dropping a_n as well would delete two things at once.
        """
        assert x.shape == v.shape and x.dim() == 2
        G, beta = self._cost_fields(x)
        vGv = torch.einsum("bi,bij,bj->b", v, G, v)
        riemann = torch.sqrt(torch.clamp(vGv, min=eps_sqrt))
        if beta is None:
            return riemann
        return riemann + (beta * v).sum(1)

    def F2(self, x: Tensor, v: Tensor) -> Tensor:
        """Squared metric F(x, v)^2 (the geodesic-energy integrand)."""
        return self.F(x, v) ** 2

    # ------------------------------------------------------------------ #
    #  Self-verification (theory sanity checks)
    # ------------------------------------------------------------------ #
    def verify(self, n_probe: int = 256, seed: int = 0) -> dict:
        """Assert the derived structure matches the theory; return diagnostics."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        idx = torch.randint(0, self.N, (n_probe,), generator=g, device=self.device)
        x = self.X[idx] + 0.05 * torch.randn(n_probe, self.d, generator=g,
                                             device=self.device, dtype=self.dtype)

        # 1) *both* per-point second moments are symmetric PSD.  Σ^full is checked even
        #    when it is not in force: its weights are the row-stochastic P_ij >= 0, so
        #    PSD is a property of the accumulation and not of which mode was selected,
        #    and a negative eigenvalue would mean the edge weights had gone wrong.
        for nm, Sig in (("Σ^sym", self.Sigma_sym_pts), ("Σ^full", self.Sigma_full_pts)):
            assert torch.allclose(Sig, Sig.transpose(1, 2), atol=1e-5), (
                f"{nm} not symmetric")
            assert torch.linalg.eigvalsh(Sig).min() >= -1e-5, f"{nm} not PSD"

        # 2) strong-convexity condition for a genuine Randers metric: β^T G^{-1} β < 1.
        #    Note ``cost_tensor_inv``, not ``mobility``: under the FW form the two are
        #    no longer the same tensor and only the former is the metric's inverse.
        G, beta = self._cost_fields(x)
        if beta is None:                                    # β = 0 rungs are trivially fine
            beta = torch.zeros_like(x)
        Ginv = self.cost_tensor_inv(x)
        bGb = torch.einsum("bi,bij,bj->b", beta, Ginv, beta)
        assert bGb.max() < 1.0, f"β not admissible: max β^T G^-1 β = {bGb.max():.4f}"
        if self.metric_form == "fw" and self.use_metric and self.use_one_form:
            # the FW 1-form's norm is c·(a_raw/ā)/a_n, i.e. exactly c in the limit
            # a_floor -> 0.  Anything above c^2 means the two halves of the metric were
            # built from different G_0 and the closed form no longer holds.
            assert bGb.max() <= self.c ** 2 + 1e-4, (
                f"FW admissibility is analytic: expected <= c^2 = {self.c ** 2:.4f}, "
                f"got {bGb.max():.4f}")
        if self.metric_form == "fw_lambda" and self.use_metric and self.use_one_form:
            # Here the margin is not merely bounded, it is *known*: β^T G^{-1} β must
            # equal 1 - λ²/a² pointwise.  Checking the identity rather than the bound
            # catches a G and a β built from different G_0, which a bound would not.
            a_raw = self._a_raw(x, self._g0(x), self.drift(x))
            predicted = 1.0 - self.fw_lambda ** 2 / (a_raw ** 2 + self.fw_lambda ** 2)
            err = (bGb - predicted).abs().max()
            # Same 1e-4 as the ``fw`` bound above, and for the same reason: bGb costs a
            # d x d inverse and two quadratic forms, and the arms run in float32 (eps
            # 1.2e-7), so a few hundred eps is round-off.  A G and a β built from
            # different G_0 miss by O(0.1-1), not by O(1e-6), so the looser bound gives
            # up nothing — a 1e-6 tolerance here only rejected float32.
            assert err < 1e-4, (
                f"fw_lambda admissibility identity β^T G^-1 β = 1 - λ²/a² is off by "
                f"{err:.2e}; G and β were not built from the same G_0")

        # 3) F strictly positive on random non-zero directions
        v = torch.randn(n_probe, self.d, generator=g, device=self.device,
                        dtype=self.dtype)
        Fv = self.F(x, v)
        assert Fv.min() > 0.0, f"F not positive: min F = {Fv.min():.4e}"

        # per-unit-length cost of cutting through a void vs moving along the
        # cheapest (along-manifold) direction:  sqrt(1 + max_eig(Σ)/ρ).  Under the
        # G≡I ablation there is no such contrast, so the ratio is exactly 1.
        penalty_ratio = (
            1.0 if not self.use_metric
            else float((1.0 + self.sigma_eig_range[1] / self.rho) ** 0.5)
        )

        with torch.no_grad():
            a_n = self.conformal_factor(x)

        return {
            "rho": self.rho,
            "c": self.c,
            "metric_form": self.metric_form,
            "drift_mode": self.drift_mode,
            "sigma_mode": self.sigma_mode,
            # the diffusion tensor's conditioning, which Path B divides by twice.
            # Reported from the global Σ eigenvalue range, so it is an envelope.
            "mobility_cond": float(
                (self.sigma_eig_range[1] + self.rho)
                / (self.sigma_eig_range[0] + self.rho)),
            "a_floor": self.a_floor,
            "use_metric": self.use_metric,
            "use_one_form": self.use_one_form,
            "eps_sigma": self.eps_sigma,
            "eps_kernel": self.eps_kernel,
            "sigma_eig_range": self.sigma_eig_range,
            "J_norm_mean": self.J_norm_mean,
            # both first moments, whichever one is in force: the ratio is the single
            # number that says how much of P's drift the P^asym estimator throws away
            "J_asym_norm_mean": self.J_asym_norm_mean,
            "J_full_norm_mean": self.J_full_norm_mean,
            # the second-moment counterparts: the mean traces must agree (the switch is
            # trace-preserving by construction) while the per-point Frobenius gap says
            # how much anisotropy the choice of estimator actually moves
            "sigma_trace_mean": dict(self.sigma_trace_mean),
            "sigma_rel_diff_mean": self.sigma_rel_diff_mean,
            "sigma_trace_gap": self.trace_gap,
            "a_bar": self.a_bar,
            # λ is meaningless as an absolute — only λ / ||b||_{G_0} is — so both the
            # scale it is read against and the resulting effective Randers strength are
            # logged per run.  ``c_eff`` is directly comparable to the published ``c``.
            "fw_lambda": self.fw_lambda,
            "a_raw_bar": self.a_raw_bar,
            "a_raw_max": self.a_raw_max,
            "fw_lambda_over_a": (self.fw_lambda / self.a_raw_bar
                                 if self.metric_form == "fw_lambda" else None),
            "c_eff": (self.a_raw_bar / float(np.hypot(self.a_raw_bar, self.fw_lambda))
                      if self.metric_form == "fw_lambda" else None),
            "a_n_range": (float(a_n.min()), float(a_n.max())),
            "max_beta_norm_Ginv": float(bGb.max()),
            "F_min": float(Fv.min()),
            "void_vs_manifold_cost_ratio": penalty_ratio,
        }
