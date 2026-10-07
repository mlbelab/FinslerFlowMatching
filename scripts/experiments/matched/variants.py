"""Five path costs over one set of moments, and the two-coupling Path A.

The whole point of this module is that the *fields* are shared and only the *pricing*
differs.  Every class here derives from :class:`scripts.method.metric.TrueFW`, so the
first moment and the local covariance come out of the same two lines --
:meth:`~scripts.method.metric.TrueFW._fields` -- that the proposed method uses.  A variant
that built its own smoother, or its own :math:`\\rho`, would be a different experiment.

The fixed-time costs are not Finsler norms
------------------------------------------
:math:`\\tfrac1a(\\dot x-a m)^\\top C_\\rho^{-1}(\\dot x-a m)` is an *affine* quadratic in
:math:`\\dot x`, not a 1-homogeneous norm, so it has no ``(G, beta)`` and cannot go
through :meth:`Metric.tensors`.  It does not need to: Phase 1, the Phase-2 cost matrix and
Phase 3 read a metric only through ``.d`` and ``.F2``, so a cost object that supplies
those is a drop-in.  :class:`FixedTimeCost` supplies them from a single ``_raw2`` hook and
keeps :meth:`Metric.calibrate`'s convention (:math:`\\mathbb E[F]=1` over unit directions),
so every arm's Phase-1 loss enters Adam and ``clip_grad_norm_`` at the same scale.  That
calibration is not cosmetic here -- gradient clipping is the one part of the shared
trainer that is *not* invariant to a global rescaling of the loss.

What ``a`` is, and why it is searched
-------------------------------------
:math:`m` is a **raw** one-step conditional displacement; :math:`\\dot x_t` is a velocity
on the normalised interval :math:`[0,1]`.  The two do not share a scale, and Remark 20
distinguishes them.  ``a`` is the global effective transition duration in graph-step
units that reconciles them, and fixing it at 1 would be a statement about the graph's
step size rather than about the objective.  It is therefore *measured* and then *searched*
around the measurement:

.. math:: a_\\star = \\frac{\\mathbb E_{x_0\\sim p_0,\\,x_1\\sim p_1}\\|x_1-x_0\\|}
                           {\\mathbb E_i \\|m(x_i)\\|}

-- how many graph steps it takes to cover the transport distance at the drift's own speed
-- and the ladder is a multiple of it, on the same half-decade spacing and with the same
number of rungs as the :math:`\\lambda` ladder the proposed method gets.

Two rescalings that are provably inert, and are asserted to be
--------------------------------------------------------------
* The ``1/a`` prefactor is a positive constant at fixed ``a``.  It cannot move the
  Phase-1 argmin, it is absorbed exactly by :meth:`Metric.calibrate`, and the Phase-2
  regulariser is ``blur_frac * median(C)`` so it cannot move the coupling either.  It is
  written anyway, because it is what the fixed-time objective says.  The search over
  ``a`` is therefore a search over the *drift scale inside the residual*, which is live.
* :math:`\\lambda` in :math:`F_{\\rm second}=\\lambda\\sqrt{v^\\top C_\\rho^{-1}v}` is a
  global factor for the same three reasons.  ``ffm_second`` accordingly has **one** real
  knob, :math:`\\rho`, and is given a ladder twice as fine on it rather than a second axis
  that would return identical rows.

Both are checked in the verification suite in the research repository, not merely argued here.
"""
from __future__ import annotations

import time

import numpy as np
import ot
import torch

from scripts.method.config import Run
from scripts.method.metric import TrueFW
from scripts.method.moments import rho_star
from scripts.method.nets import Coupler
from scripts.method.phases import (build_ot, integrate, train_cfm, train_geodesic)

#: the two coupling settings, reported side by side.  ``common`` is one fixed Euclidean
#: entropic-OT plan shared verbatim by every arm, so a difference between arms under it is
#: a difference in the path objective alone; ``specific`` lets each arm's own learned path
#: cost build its plan, which is the complete pipeline.
COUPLINGS = ("common", "specific")

#: the arms, in the order a table prints them: the two fixed-time controls, then the two
#: metric variants, then the existing ablation that is carried for the caption alone.
ARMS = ("graph_drift", "aniso_fixed", "ffm_second", "ffm_full", "ffm_riem")

#: which arms are *controlled comparison variants* rather than the proposed method or one
#: of its ablations.  A label that loses this distinction is the mistake this tree exists
#: to avoid -- neither is a published baseline and neither is Curly-FM.
CONTROLS = ("graph_drift", "aniso_fixed")

ARM_LABELS = {
    "graph_drift": "Graph-drift FM (control)",
    "aniso_fixed": "Anisotropic fixed-time FM (control)",
    "ffm_second": "Second-moment-only FFM (m = 0)",
    "ffm_full": "FFM, Eq. (12) (ours)",
    "ffm_riem": "FFM, one-form deleted (existing ablation)",
}


# --------------------------------------------------------------------------- #
#  The five costs
# --------------------------------------------------------------------------- #
class FullFFM(TrueFW):
    """Eq. (12) unchanged -- inherited whole, so "ours" is not a re-implementation."""

    name = "ffm_full"


class RiemannOnly(TrueFW):
    """The **existing** metric ablation: :math:`\\beta^\\top v` deleted, :math:`a^2` kept.

    :math:`F = a(x)\\sqrt{v^\\top C_\\rho^{-1}v}` with
    :math:`a^2=\\|m\\|^2_{C_\\rho^{-1}}+\\lambda^2`.  The directional term is gone but the
    conformal factor still reads the first moment, so this removes *direction* and not
    *first-moment information*.  Carried here only so the difference from
    :class:`SecondOnly` is a measured number rather than a claim in a caption.
    """

    name = "ffm_riem"

    def tensors(self, x):
        G, _ = super().tensors(x)
        return G, torch.zeros_like(x)


class SecondOnly(TrueFW):
    """:math:`F_{\\rm second}(x,v)=\\lambda\\sqrt{v^\\top C_\\rho^{-1}v}`: :math:`m=0`.

    Setting :math:`m=0` throughout Eq. (12) sends :math:`\\beta=-C_\\rho^{-1}m` to zero
    *and* collapses :math:`a^2` to :math:`\\lambda^2`, which is the whole difference from
    :class:`RiemannOnly`.  Goes through :meth:`Metric.F` unchanged, so it is priced by the
    same code as ``ffm_full``.
    """

    name = "ffm_second"

    def tensors(self, x):
        Ci, _ = self._fields(x)
        return (self.lam ** 2) * Ci, torch.zeros_like(x)


class FixedTimeCost(TrueFW):
    """A fixed-time quadratic cost that duck-types a metric for the three phases.

    ``_raw2(x, v)`` is the *uncalibrated* cost density; ``F`` and ``F2`` wrap it in the
    one global constant :meth:`Metric.calibrate` sets, exactly as they do for a norm.
    ``F2`` is written as ``scale**2 * clamp(_raw2, 0)`` rather than ``F**2`` so that a
    cost which is genuinely zero -- an interpolant that matched the drift exactly -- is
    reported as zero instead of as the square of the square-root floor.
    """

    def __init__(self, X, D_pts, b0_pts, cfg: Run, rho: float, a: float,
                 lam: float | None = None, lam_mult: float = 0.3):
        super().__init__(X, D_pts, b0_pts, cfg, rho=rho, lam=lam, lam_mult=lam_mult)
        assert a > 0, f"the transition duration must be positive, got a={a}"
        self.a = float(a)

    def _m(self, x):
        """The smoothed first moment -- the same expression :meth:`_fields` returns."""
        return self._smooth(self._kernel(x), self.b0_pts) + self.eps_b

    def _raw2(self, x, v):
        raise NotImplementedError

    def F(self, x, v, eps_sqrt=1e-9):
        return self.scale * torch.sqrt(torch.clamp(self._raw2(x, v), min=eps_sqrt))

    def F2(self, x, v):
        return self.scale ** 2 * torch.clamp(self._raw2(x, v), min=0.0)


class GraphDrift(FixedTimeCost):
    """:math:`\\tfrac1a\\|\\dot x - a\\,m(x)\\|^2` -- graph aggregation, no geometry.

    Asks whether turning velocities into a neighbourhood-averaged drift is by itself what
    beats the per-cell velocity baseline.  :math:`C_\\rho` never enters the cost, so
    :math:`\\rho` is not a knob of this arm; it is still constructed at
    :math:`\\rho_\\star` because the shared object carries the fields, and ``verify``
    asserts the cost is numerically independent of it.
    """

    name = "graph_drift"

    def _raw2(self, x, v):
        r = v - self.a * self._m(x)
        return (r * r).sum(1) / self.a


class AnisoFixed(FixedTimeCost):
    """:math:`\\tfrac1a(\\dot x-a m)^\\top C_\\rho^{-1}(\\dot x-a m)` -- both moments.

    The matched competitor: it receives the first moment as a drift target and the second
    as the norm the residual is measured in, which is every statistic of ``P`` the
    proposed action reads.  What it does not have is the 1-homogeneity that makes Eq. (12)
    a *geometry* -- its cost depends on how fast the path is traversed, so the comparison
    is fixed-time against time-free and nothing else.
    """

    name = "aniso_fixed"

    def _raw2(self, x, v):
        Ci, m = self._fields(x)
        r = v - self.a * m
        return torch.einsum("bi,bij,bj->b", r, Ci, r) / self.a


# --------------------------------------------------------------------------- #
#  The measured scales, and the factory that is the single source of the fields
# --------------------------------------------------------------------------- #
def measure_a_star(cost: FixedTimeCost, X0_t, X1_t, chunk: int = 1024) -> float:
    """:math:`a_\\star`: graph steps per unit transport distance, in this split.

    Deterministic -- the chord mean is over the *whole* product of the two marginals, not
    a sample -- and independent of :math:`\\rho`, since :meth:`FixedTimeCost._m` does not
    read it.  So one split has one :math:`a_\\star`, shared by both fixed-time arms and by
    every rung of their ladders.
    """
    with torch.no_grad():
        m = torch.cat([cost._m(cost.X[s:s + chunk]) for s in range(0, cost.N, chunk)])
        m_bar = float(m.norm(dim=1).mean())
        chord = float(torch.cdist(X0_t.double(), X1_t.double()).mean())
    assert m_bar > 0, "the first moment vanished everywhere; a_star is undefined"
    return chord / m_bar


class MatchedFactory:
    """``P``'s two moments, the cloud and the config bound once; a cell is :meth:`at`.

    This is the object the "same quantities, supplied to every variant" claim rests on:
    ``b0_pts`` and ``D_pts`` are passed in once, stored once, and handed unchanged to
    every cost below.  The three scales a ladder is quoted against -- :math:`\\rho_\\star`
    from the second moment, :math:`\\bar a` from the first, :math:`a_\\star` from the
    first and the two marginals -- are all measured here, so no rung is ever a bare number.
    """

    def __init__(self, X, D_pts, b0_pts, cfg: Run, X0_t, X1_t):
        self.X, self.D_pts, self.b0_pts, self.cfg = X, D_pts, b0_pts, cfg
        self.rho_star = rho_star(D_pts)
        probe = GraphDrift(X, D_pts, b0_pts, cfg, rho=self.rho_star, a=1.0)
        self.a_bar = probe.a_bar                      # mean ||m||_{C^-1} at rho_star
        self.a_star = measure_a_star(probe, X0_t, X1_t)

    # -- the ladders, as multiples of a measured scale ---------------------------- #
    def rho_of(self, mult: float) -> float:
        return float(mult) * self.rho_star

    def lam_of(self, mult: float) -> float:
        return float(mult) * self.a_bar

    def a_of(self, mult: float) -> float:
        return float(mult) * self.a_star

    def scales(self) -> dict:
        return {"rho_star": float(self.rho_star), "a_bar": float(self.a_bar),
                "a_star": float(self.a_star)}

    def at(self, arm: str, point: dict):
        """One calibrated cost object at one grid point.

        The moments never move, so a grid point costs a construction and a calibration.
        ``point`` carries only the knobs that arm actually has; asking for one it does not
        have is an error rather than a silently ignored keyword, because a ladder that is
        quietly inert is the failure mode a matched comparison cannot afford.
        """
        assert arm in ARMS, f"unknown arm {arm!r}; this tree has {ARMS}"
        want = KNOBS[arm]
        assert set(point) == set(want), (
            f"{arm} takes exactly {sorted(want)}, got {sorted(point)}")
        common = (self.X, self.D_pts, self.b0_pts, self.cfg)
        if arm == "graph_drift":
            return GraphDrift(*common, rho=self.rho_star,
                              a=self.a_of(point["a_mult"])).calibrate()
        if arm == "aniso_fixed":
            return AnisoFixed(*common, rho=self.rho_of(point["rho_mult"]),
                              a=self.a_of(point["a_mult"])).calibrate()
        if arm == "ffm_second":
            # lam is a global factor here and is provably inert (see the module
            # docstring); it is pinned at the a_bar default so the object is well-formed.
            return SecondOnly(*common, rho=self.rho_of(point["rho_mult"]),
                              lam=self.a_bar).calibrate()
        cls = FullFFM if arm == "ffm_full" else RiemannOnly
        return cls(*common, rho=self.rho_of(point["rho_mult"]),
                   lam=self.lam_of(point["lam_mult"])).calibrate()


#: the knobs each arm has, and therefore the exact key set of its grid point.  Declared
#: once so the tuner, the factory and ``verify`` cannot disagree about it.
KNOBS: dict[str, tuple[str, ...]] = {
    "graph_drift": ("a_mult",),
    "aniso_fixed": ("rho_mult", "a_mult"),
    "ffm_second": ("rho_mult",),
    "ffm_full": ("rho_mult", "lam_mult"),
    "ffm_riem": ("rho_mult", "lam_mult"),
}


# --------------------------------------------------------------------------- #
#  The common coupling
# --------------------------------------------------------------------------- #
def build_ot_euclidean(X0_t, X1_t, seed: int, cfg: Run, max_pts: int | None = None):
    """Entropic OT under :math:`\\|x_0-x_1\\|^2`, everything else as in ``build_ot``.

    Deliberately a line-for-line twin of :func:`scripts.method.phases.build_ot` with the
    Finsler cost swapped out, including the subsample generator (``seed + 91``) -- so at a
    given seed **every arm sees the identical pair set and the identical plan**, which is
    what makes the common-coupling column isolate the path objective.

    The regulariser follows the repository's convention, ``blur_frac * median(C)``, rather
    than a shared numerical :math:`\\varepsilon`: the Euclidean cost and the five Finsler
    costs are on five different scales, and a fixed :math:`\\varepsilon` would hand them
    five different amounts of entropy.
    """
    max_pts = cfg.ot_max_pts if max_pts is None else max_pts
    g = torch.Generator(device="cpu").manual_seed(seed + 91)
    i0 = torch.randperm(len(X0_t), generator=g)[:max_pts].to(cfg.device)
    i1 = torch.randperm(len(X1_t), generator=g)[:max_pts].to(cfg.device)
    A, B = X0_t[i0], X1_t[i1]
    C = torch.cdist(A, B) ** 2
    Cn = C.detach().double().cpu().numpy()
    reg = float(cfg.ot_blur_frac * np.median(Cn) + 1e-12)
    pi = ot.sinkhorn(np.full(len(A), 1 / len(A)), np.full(len(B), 1 / len(B)), Cn, reg,
                     method="sinkhorn_log", numItermax=cfg.ot_n_iter)
    pi = torch.as_tensor(pi, **cfg.torch_kw)
    p = pi / pi.sum()
    ent = float(-(p * torch.log(p.clamp_min(1e-30))).sum() / np.log(p.numel()))
    return Coupler(A, B, pi=pi, seed=seed), {"C_median": float(C.median()), "reg": reg,
                                             "pi_entropy_frac": ent, "cost": "euclidean"}


# --------------------------------------------------------------------------- #
#  Path A, once, under both couplings
# --------------------------------------------------------------------------- #
def matched_path_a(cost, X0_t, X1_t, seed: int, cfg: Run, verbose: bool = False) -> dict:
    """Phase 1 once; Phases 2 and 3 twice, under the two coupling settings.

    Phase 1 is coupling-free by construction -- :func:`scripts.method.phases.path_a` fits
    the interpolant against the *independent product* of the two marginals, and only
    Phase 2 introduces a plan -- so the same :math:`\\varphi` serves both columns and the
    second column costs a Sinkhorn plus a distillation rather than a run.  That is the
    reuse the review asks for, and it also makes the two columns paired at the level of
    the interpolant and not merely at the level of the seed.

    Timings are per column and *include* the shared Phase 1, because that is what running
    the pipeline in that setting alone would cost.
    """
    log = 250 if verbose else 0
    t0 = time.perf_counter()
    phi = train_geodesic(cost, Coupler(X0_t, X1_t, seed=seed), seed, cfg, log_every=log)
    t_phase1 = time.perf_counter() - t0

    out = {"phi": phi, "t_phase1": t_phase1}
    for mode in COUPLINGS:
        t1 = time.perf_counter()
        if mode == "common":
            coupler, diag = build_ot_euclidean(X0_t, X1_t, seed, cfg)
        else:
            coupler, diag = build_ot(phi, cost, X0_t, X1_t, seed, cfg)
        t_ot = time.perf_counter() - t1

        t2 = time.perf_counter()
        v_net = train_cfm(phi, coupler, cost.d, seed, cfg, log_every=log)
        traj = integrate(v_net, X0_t, cfg).cpu().numpy().astype(np.float64)
        t_tail = time.perf_counter() - t2

        out[mode] = {"coupler": coupler, "v_net": v_net, "traj": traj, "ot": diag,
                     "t_ot": t_ot, "t_phase3": t_tail,
                     "total_s": t_phase1 + t_ot + t_tail}
    return out
