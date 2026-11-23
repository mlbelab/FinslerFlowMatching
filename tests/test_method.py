import numpy as np
import pytest
import scipy.sparse as sp

import ffm


def toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-1.0, 1.0, n), 0.1 * rng.normal(size=n)]
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    W = np.exp(-D / 0.05) * (X[None, :, 0] > X[:, None, 0] - 0.05)
    np.fill_diagonal(W, 0.0)
    return X, sp.csr_matrix(W / W.sum(1, keepdims=True))


def test_api():
    assert ffm.__version__
    assert callable(ffm.path_a) and callable(ffm.moments_from)
    with pytest.raises(AttributeError):
        ffm.not_an_attribute


def test_moments():
    X, P = toy(40)
    m, C = ffm.moments_from(P, X)
    Pd = P.toarray()
    delta = X[None, :, :] - X[:, None, :]
    assert np.allclose(m, (Pd[:, :, None] * delta).sum(1))
    assert np.allclose(C, np.einsum("ij,ijk,ijl->ikl", Pd, delta, delta))


def test_pipeline():
    X, P = toy()
    cfg = ffm.Run.from_env(geo_iters=10, cfm_iters=10, sb_iters=10, n_steps=5, ot_max_pts=30)
    m, C = ffm.moments_from(P, X)
    metric = ffm.FWFactory(X, C, m, cfg, rho_mult=1.0, lam_mult=0.3).base
    source, target = X[:, 0] < -0.5, X[:, 0] > 0.5
    X0, X1 = cfg.tensor(X[source]), cfg.tensor(X[target])
    det = ffm.path_a(metric, X0, X1, seed=0, cfg=cfg)
    sde = ffm.path_b(metric, det["phi"], det["coupler"], X0, ffm.sigma_for(metric, 0.3), seed=0, cfg=cfg)
    assert det["traj"].shape == (6, int(source.sum()), 2)
    assert np.isfinite(det["traj"]).all() and np.isfinite(sde["traj"]).all()
    assert 0.0 < ffm.c_eff_of(metric) < 1.0
