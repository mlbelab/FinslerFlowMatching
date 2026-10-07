"""The metric: the regularised Freidlin–Wentzell action of the SDE that ``P`` estimates.

$$F(x,v)=\\sqrt{v^{\\!\\top}C_\\rho^{-1}v}\\;\\sqrt{\\|b_0\\|^2_{C_\\rho^{-1}}+\\lambda^2}
        \\;-\\;v^{\\!\\top}C_\\rho^{-1}b_0,
\\qquad C_\\rho=\\tilde D+\\rho I,$$

a Randers metric with $G=a^2C_\\rho^{-1}$ and $\\beta=-C_\\rho^{-1}b_0$.  Both fields are
extended off the samples by a Gaussian kernel smoother whose bandwidth is the median
*positive* nearest-neighbour distance — positive because a coincident pair of rows
reports no length scale, and some ported clouds have many of them.

Two knobs, both swept.  ``rho`` floors the diffusion; ``lam`` regularises the 1-form and
is reported through the convention-free strength

    c_eff = a_bar / sqrt(a_bar^2 + lam^2)  in (0, 1),

``c_eff -> 1`` being a pure quasipotential and ``c_eff -> 0`` Riemannian.

**The mobility is $C_\\rho$ itself, not $G^{-1}$.**  The conformal factor $a^2$ rescales
the clock, not the noise, so it is deliberately absent from :meth:`Metric.mobility` —
which is the tensor Path B diffuses with.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.method.config import Run
from scripts.method.moments import rho_star


class Metric:
    """Kernel smoothing, ``F``, one global calibration, and the Path-B mobility.

    Everything is differentiable in ``x`` so that ``F(x_t, xdot_t)`` back-propagates
    through the interpolant.  ``mobility`` / ``mobility_inv`` / ``metric_tensor_inv`` are
    the API :mod:`scripts.core.bridge` duck-types, so Path B's conditional targets are
    *imported* from the engine rather than transcribed here.
    """

    name = "metric"

    def __init__(self, X, cfg: Run):
        self.cfg = cfg
        self.X = torch.as_tensor(X, **cfg.torch_kw)
        self.N, self.d = self.X.shape
        self.device, self.dtype = cfg.device, cfg.dtype
        self.eps_den, self.eps_b = float(cfg.eps_den), float(cfg.eps_b)
        self.I = torch.eye(self.d, **cfg.torch_kw)
        with torch.no_grad():
            dm = torch.cdist(self.X, self.X)
            dm.fill_diagonal_(float("inf"))
            nn_d = dm.min(dim=1).values
            pos = nn_d[nn_d > 0]              # coincident rows report no length scale
            self.eps_kernel = float(pos.median()) * float(cfg.eps_kernel_scale)
        self.scale = 1.0                      # set by ``calibrate``

    # -- kernel smoothing ---------------------------------------------------------#
    def _kernel(self, x):
        """``k_eps(x, x_i) -> (B, N)``, as a matmul with no ``(B, N, d)`` intermediate."""
        sqd = torch.clamp((x ** 2).sum(1, keepdim=True) + (self.X ** 2).sum(1)[None, :]
                          - 2.0 * (x @ self.X.T), min=0.0)
        return torch.exp(-sqd / (4.0 * self.eps_kernel ** 2))

    def _smooth(self, k, vals):
        tail = vals.shape[1:]
        num = k @ vals.reshape(self.N, -1)
        den = k.sum(1, keepdim=True) + self.eps_den
        return (num / den).reshape(-1, *tail)

    # -- the norm -----------------------------------------------------------------#
    def tensors(self, x):
        """``(G, beta)`` of the Randers form at ``x``."""
        raise NotImplementedError

    def F(self, x, v, eps_sqrt=1e-9):
        G, beta = self.tensors(x)
        vGv = torch.einsum("bi,bij,bj->b", v, G, v)
        return self.scale * (torch.sqrt(torch.clamp(vGv, min=eps_sqrt))
                             + (beta * v).sum(1))

    def F2(self, x, v):
        return self.F(x, v) ** 2

    def calibrate(self, n=512, seed=0):
        """One constant so that ``E[F(x_i, v)] = 1`` over unit-norm random ``v``.

        It rescales the *cost* only: the Phase-1 minimiser and the entropic coupling
        (reg = ``blur_frac * median C``) are invariant under it, and it never touches the
        mobility, so Path B's noise level is unaffected.  Re-run at every grid point,
        because ``1/E[F]`` moves with rho and lambda.
        """
        if not self.cfg.normalise_f:
            return self
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(self.N, generator=g)[:n].to(self.device)
        v = torch.randn(len(idx), self.d, generator=g).to(self.device, self.dtype)
        v = v / v.norm(dim=1, keepdim=True)
        with torch.no_grad():
            self.scale = 1.0
            mean_F = float(self.F(self.X[idx], v).mean())
        assert mean_F > 0, "F is not positive on average; the metric is degenerate"
        self.scale = 1.0 / mean_F
        return self

    # -- the Path-B mobility ------------------------------------------------------#

    #: every metric in this module builds M(x) out of the local covariance, so none of
    #: them is x-independent; Path B reads this to decide whether ∂_tM_t and σ²∇·M are
    #: structurally zero.  ``scripts.core.bridge.ConstantMobility`` is the wrapper that
    #: sets it True.
    constant_mobility = False

    def mobility(self, x):
        """``M(x)``: the DIFFUSION tensor, never the cost."""
        raise NotImplementedError

    def mobility_inv(self, x):
        return torch.linalg.inv(self.mobility(x))

    def metric_tensor_inv(self, x):
        """``bridge.py``'s name for the same object — see ``geometry.py:602``."""
        return self.mobility(x)

    def mean_mobility_eig(self, chunk=512):
        """``m_bar = mean_i tr M(x_i)/d``: the scale a bridge width is quoted against."""
        with torch.no_grad():
            tot = sum(float(self.mobility(self.X[s:s + chunk])
                            .diagonal(dim1=-2, dim2=-1).sum())
                      for s in range(0, self.N, chunk))
        return tot / (self.N * self.d)


class TrueFW(Metric):
    """``F = sqrt(v' C_rho^-1 v) * sqrt(||b_0||^2_{C_rho^-1} + lam^2) - v' C_rho^-1 b_0``."""

    name = "FFM"

    def __init__(self, X, D_pts, b0_pts, cfg: Run, rho: float, lam: float | None = None,
                 lam_mult: float = 0.3):
        super().__init__(X, cfg)
        self.D_pts = torch.as_tensor(D_pts, **cfg.torch_kw)
        self.b0_pts = torch.as_tensor(b0_pts, **cfg.torch_kw)
        self.rho = float(rho)
        # a_bar = the measured mean ||b_0||_{C_rho^-1}, computed at lam = 0 so it is a
        # property of the data and rho alone.  It sets the default lambda and gives
        # c_eff = a_bar/sqrt(a_bar^2+lam^2), the convention-free reading of any lambda.
        self.lam = 0.0
        with torch.no_grad():
            tot = 0.0
            for s in range(0, self.N, 1024):
                Ci, b = self._fields(self.X[s:s + 1024])
                tot += float(torch.sqrt(torch.clamp(
                    torch.einsum("bi,bij,bj->b", b, Ci, b), min=0.0)).sum())
            self.a_bar = tot / self.N
        assert self.a_bar > 0, "the first moment vanished everywhere; FW is undefined"
        self.lam = float(lam_mult) * self.a_bar if lam is None else float(lam)

    def mobility(self, x):
        """``C_rho = D~ + rho I``.  The FW conformal factor ``a^2`` is deliberately absent."""
        return self._smooth(self._kernel(x), self.D_pts) + self.rho * self.I[None]

    def mobility_divergence(self, x):
        """``(div M)_i = sum_j d_j M_ij`` in closed form -> ``(B, d)``.

        Path B's sampler needs this every step (``scripts.core.bridge``).  ``C_rho`` is a
        Nadaraya-Watson average of the fixed matrices ``D_i`` under a Gaussian kernel, so
        its divergence is analytic and costs two extra ``(B,N)@(N,d)`` matmuls instead of
        the ``d`` passes AD would take.  With ``k_i = exp(-|x-x_i|^2/4 eps^2)``,
        ``S = sum_i k_i + eps_den`` and ``D~ = (sum_i k_i D_i)/S``,

            d_j k_i  = -k_i (x - x_i)_j / (2 eps^2)
            d_j D~_ab = sum_i d_j k_i (D_i,ab - D~_ab) / S

        and contracting ``b = j``, then telescoping the sum over ``i``, gives

            div M = -[ (eps_den/S)(sum_i k_i D_i)x - sum_i k_i D_i x_i + D~ (sum_i k_i x_i)
                     ] / (2 eps^2 S)

        exact, regulariser included; ``rho I`` is constant and drops out.  Checked against
        AD and finite differences in the verification suite in the research repository — the only thing that makes a
        hand-derived gradient in a hot loop safe.  Mirrors
        ``scripts.core.geometry.FinslerGeometry.mobility_divergence``, which smooths the
        same way.
        """
        if getattr(self, "_D_x_pts", None) is None:
            self._D_x_pts = torch.einsum("nab,nb->na", self.D_pts, self.X).contiguous()
        k = self._kernel(x)                                      # (B, N)
        S = k.sum(1, keepdim=True) + self.eps_den                # (B, 1)
        A = (k @ self.D_pts.reshape(self.N, self.d * self.d)).reshape(-1, self.d, self.d)
        Dt = A / S[:, :, None]                                   # D~(x)
        u = k @ self._D_x_pts                                    # sum_i k_i D_i x_i
        m = k @ self.X                                           # sum_i k_i x_i
        bracket = (self.eps_den / S) * torch.einsum("bij,bj->bi", A, x) \
            - u + torch.einsum("bij,bj->bi", Dt, m)
        return -bracket / (2.0 * self.eps_kernel ** 2 * S)

    def _fields(self, x):
        k = self._kernel(x)
        D = self._smooth(k, self.D_pts)
        b = self._smooth(k, self.b0_pts) + self.eps_b
        return torch.linalg.inv(D + self.rho * self.I[None]), b

    def tensors(self, x):
        Ci, b = self._fields(x)
        a2 = torch.einsum("bi,bij,bj->b", b, Ci, b).clamp(min=0.0) + self.lam ** 2
        return a2[:, None, None] * Ci, -torch.einsum("bij,bj->bi", Ci, b)


class ConformalFW(Metric):
    """``F(x,v) = c(x)^gamma * F_FW(x,v)`` — the Randers metric rescaled pointwise.

    The Freidlin–Wentzell metric reads ``P`` only through ``b_0`` and ``D~``, so it knows
    where the dynamics *point* but nothing about where the data *is*: outside the occupied
    region the kernel-smoothed moments simply decay to their neighbours' values and ``F``
    stays finite.  MFM's metric is the complementary object — a scalar built from the point
    cloud alone that blows up off the manifold and knows nothing of the dynamics.  This
    multiplies one onto the other.

    Multiplying a Randers norm by a positive function of position is again a Randers norm,

    .. math::

        c^\\gamma F = \\sqrt{v^{\\!\\top}(c^{2\\gamma}G)v} + (c^{\\gamma}\\beta)^{\\!\\top}v,

    and the admissibility number is exactly invariant:
    :math:`\\|c^\\gamma\\beta\\|_{(c^{2\\gamma}G)^{-1}} = \\|\\beta\\|_{G^{-1}}`.  So the
    drift asymmetry FFM gets from ``P`` — the whole content of the first moment — survives
    untouched, and all the factor can do is make *length* expensive where ``c`` is large.
    That is what makes this an addition of information rather than a different method:
    ``gamma = 0`` is FFM exactly, and the ladder in between is a dose.

    ``c(x)^2 = tr M_mfm(x) / d``, the mean squared MFM length of a unit direction, read off
    ``factor.F2`` on the ``d`` basis vectors.  For the conformal RBF metric used above 2-D
    that is exactly its ``M(x)``; for the diagonal LAND metric at ``d = 2`` it is the mean
    of the diagonal, i.e. the isotropic part.  Only the isotropic part is taken because the
    anisotropy of the cost is FFM's to set — ``C_rho`` is already a directional object, and
    two anisotropies do not compose canonically.

    The mobility is **not** rescaled.  ``a^2`` is absent from :meth:`TrueFW.mobility` for
    the same reason: a conformal factor rescales the clock, not the noise.
    """

    name = "FFMxMFM"

    def __init__(self, base: TrueFW, factor, gamma: float = 1.0):
        # deliberately not Metric.__init__: every field it would build (the cloud, the
        # kernel bandwidth, the identity) is already on ``base`` and is unchanged by a
        # conformal rescaling, and rebuilding it means a second N x N cdist.
        self.cfg, self.base, self.factor = base.cfg, base, factor
        self.gamma = float(gamma)
        self.X, self.N, self.d = base.X, base.N, base.d
        self.device, self.dtype = base.device, base.dtype
        self.eps_den, self.eps_b, self.I = base.eps_den, base.eps_b, base.I
        self.eps_kernel = base.eps_kernel
        self.rho, self.lam, self.a_bar = base.rho, base.lam, base.a_bar
        self.scale = 1.0                      # set by ``calibrate``
        self._basis = torch.eye(self.d, **base.cfg.torch_kw)

    def conformal(self, x):
        """``c(x)^2 = tr M_mfm(x)/d``, differentiable in ``x``."""
        c2 = sum(self.factor.F2(x, self._basis[k][None].expand(x.shape[0], self.d))
                 for k in range(self.d)) / self.d
        return torch.clamp(c2, min=1e-12)

    def tensors(self, x):
        G, beta = self.base.tensors(x)
        s = self.conformal(x) ** (0.5 * self.gamma)          # c^gamma
        return (s ** 2)[:, None, None] * G, s[:, None] * beta

    def mobility(self, x):
        return self.base.mobility(x)

    def mobility_divergence(self, x):
        return self.base.mobility_divergence(x)

    def factor_spread(self, chunk: int = 1024) -> dict:
        """What the factor actually does on the cloud, in length units.

        ``c`` is quoted as a ratio to its own median, because the overall level is absorbed
        by :meth:`calibrate` and only the *variation* carries information.  A spread near
        1 means the factor is constant on the data and the arm has collapsed back to FFM.
        """
        with torch.no_grad():
            c = torch.cat([self.conformal(self.X[s:s + chunk]).sqrt()
                           for s in range(0, self.N, chunk)])
        q = torch.quantile(c, torch.tensor([0.05, 0.5, 0.95], **self.cfg.torch_kw))
        med = float(q[1])
        return {"c_median": med, "c_p05_ratio": float(q[0]) / med,
                "c_p95_ratio": float(q[2]) / med, "c_max_ratio": float(c.max()) / med}


class FWFactory:
    """The moments, the cloud and the config bound once; a grid point is :meth:`at`.

    Two scales can only be *measured*, and both are measured here so that no ladder is
    ever quoted as a bare number: ``rho_star`` from the second moment alone, and
    ``a_bar`` from a first instantiation at the default rho.  Every rung below is a
    multiple of one of them.
    """

    def __init__(self, X, D_pts, b0_pts, cfg: Run,
                 rho_mult: float = 1.0, lam_mult: float = 0.3):
        self.X, self.D_pts, self.b0_pts, self.cfg = X, D_pts, b0_pts, cfg
        self.rho_star = rho_star(D_pts)
        self.rho = float(rho_mult) * self.rho_star
        # a_bar is only measurable from a fitted metric, so the default lambda -- and
        # with it the ladder ``at`` defaults to -- comes from this first instantiation.
        self.base = TrueFW(X, D_pts, b0_pts, cfg, rho=self.rho,
                           lam=None, lam_mult=lam_mult).calibrate()
        self.a_bar = self.base.a_bar
        self.lam = self.base.lam

    def at(self, rho: float | None = None, lam: float | None = None) -> TrueFW:
        """One metric at an overridden grid point, calibrated.

        The moments never move, so a grid point costs a calibration and nothing else.
        """
        return TrueFW(self.X, self.D_pts, self.b0_pts, self.cfg,
                      rho=self.rho if rho is None else rho,
                      lam=self.lam if lam is None else lam).calibrate()

    def rho_rungs(self, mults) -> tuple[float, ...]:
        """Multiples of the measured ``rho_star``."""
        return tuple(float(m) * self.rho_star for m in mults)

    def lam_rungs(self, mults) -> tuple[float, ...]:
        """Multiples of the measured ``a_bar``."""
        return tuple(float(m) * self.a_bar for m in mults)


def sigma_for(metric: Metric, mult: float) -> float:
    """The sigma whose bridge half-width at ``t = 1/2`` is ``mult`` kernel bandwidths.

    A conditional bridge has per-direction std ``sigma*sqrt(t(1-t)*m_bar)``, so at
    ``t = 1/2`` it is ``sigma*sqrt(m_bar)/2``.  Quoting sigma this way makes the injected
    displacement a physical object rather than a bare number: with raw moments the
    mobility carries the data's own scale, so one shared sigma across two clouds — or two
    dimensions of the same cloud — would mean two arbitrary and different noise levels.
    """
    return 2.0 * float(mult) * metric.eps_kernel / np.sqrt(metric.mean_mobility_eig())


def c_eff_of(mt: TrueFW) -> float:
    """``a_bar/sqrt(a_bar^2 + lam^2)``.  A report, never an input."""
    return float(mt.a_bar / np.hypot(mt.a_bar, mt.lam))
