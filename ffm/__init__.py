"""Finsler Flow Matching (FFM): dynamics-aware geodesic interpolation for single-snapshot trajectory inference.

``ffm.method``  transition moments, the P-derived Randers metric, the geodesic interpolant,
                Finsler optimal transport, flow and score matching, as plain PyTorch modules.
``ffm.core``    the experiment engine: transition kernels, arms, bridges, couplings, metrics
                and the vendored baselines (MFM, Curly-FM).
"""
from importlib import import_module

__version__ = "1.0.0"

_API = ("Run", "moments_from", "rho_star", "TrueFW", "FWFactory", "sigma_for", "c_eff_of",
        "train_geodesic", "finsler_cost", "build_ot", "train_cfm", "integrate", "path_a",
        "train_sbm", "path_b", "finsler_w2", "euclidean_w2")


def __getattr__(name):
    if name in _API:
        return getattr(import_module("ffm.method"), name)
    raise AttributeError(f"module 'ffm' has no attribute {name!r}")


__all__ = list(_API)
