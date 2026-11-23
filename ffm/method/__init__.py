"""Finsler Flow Matching as plain PyTorch modules.

``config``   :class:`Run`, the fixed budgets of a run (Appendix C)
``moments``  first and second transition moments ``m``, ``C`` of ``P`` (Eq. 8)
``metric``   the regularised Freidlin-Wentzell Randers metric ``F_P`` (Eq. 9)
``nets``     the geodesic interpolant ``x_t = (1-t)x0 + t x1 + t(1-t) phi(t, x0, x1)`` (Eq. 11)
``phases``   geodesic training (Eq. 12), Finsler OT (Eq. 13), flow matching and integration
``bridge``   anisotropic Gaussian conditional paths, score and flow matching, the SDE sampler
``rulers``   Euclidean ``W2`` and the Finsler ``W2,F``
"""
from __future__ import annotations

from ffm.core.bridge import ConstantMobility
from ffm.method.bridge import SdeDrift, path_b, train_sbm
from ffm.method.config import Run
from ffm.method.metric import (ConformalFW, FWFactory, Metric, TrueFW,
                                   c_eff_of, sigma_for)
from ffm.method.moments import moments_from, rho_star
from ffm.method.nets import (Coupler, GeoPathNet, PhiNet, VelocityNet, interpolant,
                        timestep_embedding)
from ffm.method.phases import (build_ot, cuda_warm_up, finsler_cost, integrate, path_a,
                          train_cfm, train_geodesic)
from ffm.method.rulers import euclidean_w2, finsler_w2
from ffm.method.tuning import (Fanout, Grid, cell_tag, format_tuned, on_edge,
                                   preflight_cell, print_ladder, rank, require_complete,
                                   smoke_root)


__all__ = [
    "Run",
    "moments_from", "rho_star",
    "Metric", "TrueFW", "ConformalFW", "FWFactory", "sigma_for", "c_eff_of",
    "GeoPathNet", "PhiNet", "VelocityNet", "timestep_embedding", "interpolant",
    "Coupler",
    "train_geodesic", "finsler_cost", "build_ot", "train_cfm", "integrate", "path_a",
    "cuda_warm_up",
    "train_sbm", "SdeDrift", "path_b", "ConstantMobility",
    "finsler_w2", "euclidean_w2",
    "Grid", "Fanout", "rank", "on_edge", "format_tuned", "preflight_cell", "cell_tag", "smoke_root",
    "print_ladder",
    "require_complete",
]
