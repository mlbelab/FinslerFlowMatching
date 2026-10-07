"""The two raw spatial moments of the transition matrix.

$$b_0(x_i)=\\sum_j P_{ij}\\,\\Delta_{ij},\\qquad
  \\tilde D(x_i)=\\sum_j P_{ij}\\,\\Delta_{ij}\\Delta_{ij}^{\\!\\top},\\qquad
  \\Delta_{ij}=x_j-x_i .$$

The first carries the *sense* of the dynamics, the second its local spread.  They are the
only statistics of ``P`` that the metric — and therefore anything downstream of it —
reads.  Both are raw sums over the row: **no time step and no trace normalisation**, so
neither carries units of its own, which is why every ladder built on them is quoted as a
multiple of a measured scale (:func:`rho_star`, and ``a_bar`` in :mod:`scripts.method.metric`).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def moments_from(Q, X):
    """``(m_i, C_i)``: the first and second spatial moments of row *i* of ``Q``.

    Expanded so nothing is materialised per ``(i, j)``:

        C_i = sum_j Q_ij x_j x_j^T - x_i mu_i^T - mu_i x_i^T + s_i x_i x_i^T

    with ``s_i = sum_j Q_ij`` and ``mu_i = sum_j Q_ij x_j``.  The whole thing is two
    sparse products, so it costs the same on 4000 cells as on 40 000.
    """
    Q = sp.csr_matrix(Q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, dim = X.shape
    s = np.asarray(Q.sum(axis=1)).ravel()
    mu = Q @ X
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, dim * dim)
    Qxx = (Q @ XX).reshape(n, dim, dim)
    m = mu - s[:, None] * X
    C = (Qxx - X[:, :, None] * mu[:, None, :] - mu[:, :, None] * X[:, None, :]
         + s[:, None, None] * X[:, :, None] * X[:, None, :])
    return m, 0.5 * (C + C.transpose(0, 2, 1))       # symmetrise away the round-off


def rho_star(D_pts) -> float:
    """``mean_i tr D~(x_i) / d`` — the measured scale the rho ladder is quoted in.

    With raw moments this is also the effective time step the transition matrix implies,
    which is why it is the natural floor for a metric of squared displacements: rho below
    it regularises nothing, rho far above it drowns the data's own anisotropy.
    """
    D = np.asarray(D_pts, dtype=np.float64)
    n, dim = D.shape[0], D.shape[1]
    return float(np.trace(D, axis1=1, axis2=2).sum()) / n / dim
