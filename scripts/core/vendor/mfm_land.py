"""LAND metric tensor — Kapusniak et al. (2024), *Metric Flow Matching*, verbatim.

Source: ``mfm/geo_metrics/land.py`` of https://github.com/kksniak/metric-flow-matching
(MIT, Copyright (c) 2024 Kacper Kapusniak).  Copied in full and unedited; see
:mod:`scripts.core.vendor` for why it is copied rather than imported.

``scripts.core.mfm_metric.LandMetric`` calls :func:`land_metric_tensor` only.
:func:`weighting_function` is its own dependency and :func:`weighting_function_dt` is
unused here — both are kept so the module is the upstream file and not a subset of it,
which is what lets ``verify`` diff the two as wholes.
"""
import torch


def weighting_function(x, samples, gamma):
    pairwise_sq_diff = (x[:, None, :] - samples[None, :, :]) ** 2
    pairwise_sq_dist = pairwise_sq_diff.sum(-1)
    weights = torch.exp(-pairwise_sq_dist / (2 * gamma**2))
    return weights


def land_metric_tensor(x, samples, gamma, rho):
    weights = weighting_function(x, samples, gamma)  # Shape [B, N]
    differences = samples[None, :, :] - x[:, None, :]  # Shape [B, N, D]
    squared_differences = differences**2  # Shape [B, N, D]

    # Compute the sum of weighted squared differences for each dimension
    M_dd_diag = torch.einsum("bn,bnd->bd", weights, squared_differences) + rho

    # Invert the metric tensor diagonal for each x_t
    M_dd_inv_diag = 1.0 / M_dd_diag  # Shape [B, D] since it's diagonal
    return M_dd_inv_diag


def weighting_function_dt(x, dx_dt, samples, gamma, weights):
    pairwise_sq_diff_dt = (x[:, None, :] - samples[None, :, :]) * dx_dt[:, None, :]
    return -pairwise_sq_diff_dt.sum(-1) * weights / (gamma**2)
