"""Fitted extensions of the moment fields off the samples.

:mod:`scripts.method.extend` holds the variants that keep the shipped Nadaraya-Watson
smoother and change one of its properties (window, far field, the moments it averages).
The objects here replace or reweight the smoother with something *fitted*: the network
and the SVR the paper says "would do as well", the SPD-aware average, and the learned
off-support barrier -- rebuilt on the repository's own pipeline so each can be run
against the shipped control.

Each inherits the rest from :class:`~scripts.method.metric.TrueFW` (the Randers form,
``a_bar``, ``lam``, the calibration) and takes ``rho`` and ``lam`` as absolute values, so
an arm built on it differs from the shipped metric by the extension alone:

``LogEuclidFW``       the shipped weights averaging ``log(D_i + rho I)`` instead of
                      ``D_i``; same window, and the far field is exactly ``rho I``
``SvrFW``             an epsilon-SVR (RBF) per component of ``b_0`` and of the Cholesky
                      factor of ``D_i + rho I``; far from the data it is its intercept
``MlpFW``             one MLP for ``b_0`` and a factor of ``D``, plain least squares,
                      optionally with Gaussian input noise (:func:`noise_for_neff`)
``LearnedBarrierFW``  the shipped fields times a conformal factor read off an MLP
                      support classifier: near 1 on the cloud, ``phi_max`` off it

The three that change ``C_rho`` have no closed-form divergence, so Path B is refused on
them; the barrier leaves the mobility alone and keeps it.  Every fit is a function of its
seed on a fixed device, and every network is frozen before the metric is used, so
Phase 1 differentiates through ``x`` only.

Nothing in the shipped pipeline imports this module.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as tF

from scripts.method.extend import kish_neff
from scripts.method.metric import TrueFW


def _spd_log(M: np.ndarray, floor: float) -> np.ndarray:
    """``logm`` of each symmetric ``M_i``, eigenvalues clipped at ``floor`` (float64)."""
    w, V = np.linalg.eigh(M)
    return np.einsum("nij,nj,nkj->nik", V, np.log(np.maximum(w, floor)), V)


def _lower(flat: torch.Tensor, d: int, rows, cols) -> torch.Tensor:
    """``(B, d(d+1)/2) -> (B, d, d)`` lower-triangular, differentiable in ``flat``."""
    L = flat.new_zeros(flat.shape[0], d, d)
    L[:, rows, cols] = flat
    return L


def _sym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.mT)


class _Mlp(torch.nn.Module):
    """SiLU layers on the coordinates centred and divided by ONE global scale, so the net
    sees the cloud's own geometry.  SiLU, not ReLU/SELU: Phase 1 differentiates in ``x``."""

    def __init__(self, X: torch.Tensor, d_out: int, width: int, depth: int):
        super().__init__()
        self.register_buffer("mu", X.mean(0))
        self.register_buffer("sd", X.var(0).mean().sqrt().clamp_min(1e-12))
        dims = [X.shape[1]] + [int(width)] * int(depth)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.SiLU()]
        self.body = torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], d_out))

    def forward(self, x):
        return self.body((x - self.mu) / self.sd)


def _seeded_net(make, seed: int, like: torch.Tensor) -> torch.nn.Module:
    """``make()`` initialised from ``seed`` on the CPU generator, which is restored after,
    so a fit neither depends on nor disturbs the caller's global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        net = make()
    return net.to(like.device, like.dtype)


def _frozen(net: torch.nn.Module) -> torch.nn.Module:
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class _MomentNet(torch.nn.Module):
    """``x -> (b(x), L(x))``: an MLP read as a first moment and a lower-triangular factor
    with a softplus diagonal, each on the data's own scale (``b_0``'s per-axis mean and
    std; ``sqrt(rho_star)`` for the factor), so the raw outputs are O(1)."""

    def __init__(self, X, b0, D, width: int, depth: int):
        super().__init__()
        self.d = X.shape[1]
        rows, cols = torch.tril_indices(self.d, self.d)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.register_buffer("on_diag", rows == cols)
        self.register_buffer("b_mu", b0.mean(0))
        self.register_buffer("b_sd", b0.std(0).clamp_min(1e-12))
        rho_star = float(torch.diagonal(D, dim1=1, dim2=2).sum(1).mean()) / self.d
        self.l_scale = rho_star ** 0.5
        self.net = _Mlp(X, self.d + len(rows), width, depth)

    def forward(self, x):
        o = self.net(x)
        b = self.b_mu + self.b_sd * o[:, :self.d]
        l = o[:, self.d:]
        l = self.l_scale * torch.where(self.on_diag, tF.softplus(l), l)
        return b, _lower(l, self.d, self.rows, self.cols)


def noise_for_neff(X, cfg, n_eff: float = 100.0, iters: int = 40) -> float:
    """The input-noise std that makes :class:`MlpFW` a Kish-``n_eff`` kernel smoother.

    ``eps`` is the one global bandwidth at which the shipped kernel
    ``exp(-|x - x_i|^2 / 4 eps^2)`` gives the samples a median Kish size of ``n_eff``
    (bisection in ``log eps``); the returned ``sqrt(2) eps`` is the std of the Gaussian
    with that kernel, ``exp(-|x - x_i|^2 / 2 s^2)``.
    """
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), device=cfg.device)
    assert 1.0 < n_eff < len(Xt), f"n_eff must lie in (1, N = {len(Xt)}), got {n_eff}"
    sq = torch.cdist(Xt, Xt) ** 2
    nn = sq.clone().fill_diagonal_(float("inf")).min(dim=1).values.sqrt()
    eps0 = float(nn[nn > 0].median())
    lo, hi = np.log(1e-3 * eps0), np.log(1e3 * eps0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        k = torch.exp(-sq / (4.0 * np.exp(2.0 * mid)))
        lo, hi = (mid, hi) if float(kish_neff(k).median()) < n_eff else (lo, mid)
    return float(np.sqrt(2.0) * np.exp(0.5 * (lo + hi)))


def _no_divergence(name: str):
    raise NotImplementedError(f"{name} has no closed-form div C_rho; Path B is not "
                              f"defined on it (only Path A is)")


class LogEuclidFW(TrueFW):
    """The FW metric with ``C_rho`` averaged in the matrix logarithm.

    ``L_i = log(D_i + rho I)`` (eigenvalues of ``D_i`` clipped at 0 first) is averaged
    with the shipped weights and the ``eps_den`` floor carrying ``log(rho) I``,

        L(x) = (sum_i k_i L_i + eps_den log(rho) I) / (sum_i k_i + eps_den),

    and ``C_rho = expm(L)``, ``C_rho^-1 = expm(-L)``.  Where the kernel is empty
    ``C_rho -> rho I`` exactly -- the shipped far field, not ``I``.  The geometric mean
    removes the arithmetic mean's swelling of the determinant; window, ``b`` and far field
    are the shipped ones.  ``matrix_exp`` rather than ``eigh``: its gradient stays finite
    at the repeated eigenvalues of the far field.
    """

    name = "FFM-logeuclid"

    def __init__(self, X, D_pts, b0_pts, cfg, rho: float, lam: float | None = None,
                 lam_mult: float = 0.3):
        assert rho > 0, f"the log-Euclidean mean needs rho > 0, got {rho}"
        D = np.asarray(D_pts, dtype=np.float64)
        self.L_pts = torch.as_tensor(_spd_log(D + rho * np.eye(D.shape[1]), rho),
                                     **cfg.torch_kw)
        self.log_rho = float(np.log(rho))
        super().__init__(X, D_pts, b0_pts, cfg, rho=rho, lam=lam, lam_mult=lam_mult)

    def _log_mean(self, k):
        num = k @ self.L_pts.reshape(self.N, -1) \
            + (self.eps_den * self.log_rho) * self.I.reshape(1, -1)
        L = (num / (k.sum(1, keepdim=True) + self.eps_den)).reshape(-1, self.d, self.d)
        return _sym(L)

    def _fields(self, x):
        k = self._kernel(x)
        Ci = _sym(torch.linalg.matrix_exp(-self._log_mean(k)))
        return Ci, self._smooth(k, self.b0_pts) + self.eps_b

    def mobility(self, x):
        return _sym(torch.linalg.matrix_exp(self._log_mean(self._kernel(x))))

    def mobility_divergence(self, x):
        _no_divergence("LogEuclidFW")


class SvrFW(TrueFW):
    """The FW metric with both fields read off epsilon-SVRs (RBF kernel).

    One ``sklearn.svm.SVR`` per component, fitted on z-scored targets: the ``d``
    components of ``b_0`` and the ``d(d+1)/2`` lower-triangular entries of the Cholesky
    factor of ``D_i`` (jittered by ``1e-3 rho I`` so it exists), and ``C_rho = L L^T +
    rho I`` exactly as the shipped metric floors it -- an SVR can undershoot its targets
    between and beyond the samples, and a floor carried inside the fitted factor would
    let ``C`` fall below ``rho`` there, an accidental barrier.  All share
    ``gamma = 1 / (d var X)`` (sklearn's ``'scale'``).  The prediction is re-evaluated in
    torch so it is differentiable in ``x``:

        z(x) = exp(-gamma |x - X_sv|^2) @ A + intercept,

    ``A`` holding each component's dual coefficients on the union of support vectors.
    An RBF expansion decays to its intercept, so **far from the data the fields are the
    intercepts** -- a typical drift and spread, not the shipped ``(eps_b, rho I)``.

    The fit count grows as ``d^2``, so this is restricted to ``d <= 10``.
    """

    name = "FFM-svr"

    def __init__(self, X, D_pts, b0_pts, cfg, rho: float, lam: float | None = None,
                 lam_mult: float = 0.3, C: float = 1.0, epsilon: float = 0.1):
        from sklearn.svm import SVR

        X64 = np.asarray(X, dtype=np.float64)
        N, d = X64.shape
        assert d <= 10, (f"SvrFW fits d + d(d+1)/2 = {d + d * (d + 1) // 2} separate SVRs "
                         f"and is restricted to d <= 10, got d = {d}")
        self.svr_C, self.svr_epsilon = float(C), float(epsilon)
        self.svr_gamma = 1.0 / (d * X64.var())
        rows, cols = np.tril_indices(d)
        chol = np.linalg.cholesky(np.asarray(D_pts, dtype=np.float64)
                                  + 1e-3 * rho * np.eye(d))
        Y = np.concatenate([np.asarray(b0_pts, dtype=np.float64), chol[:, rows, cols]], 1)
        mu, sd = Y.mean(0), Y.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        A, icpt = np.zeros((N, Y.shape[1])), np.zeros(Y.shape[1])
        for j in range(Y.shape[1]):
            m = SVR(kernel="rbf", C=self.svr_C, epsilon=self.svr_epsilon,
                    gamma=self.svr_gamma).fit(X64, (Y[:, j] - mu[j]) / sd[j])
            A[m.support_, j] = m.dual_coef_[0]
            icpt[j] = m.intercept_[0]
        sv = np.flatnonzero(np.abs(A).sum(1) > 0)
        self.n_sv = len(sv)
        self.sv_X, self.sv_A = cfg.tensor(X64[sv]), cfg.tensor(A[sv])
        self.y_icpt, self.y_mu, self.y_sd = cfg.tensor(icpt), cfg.tensor(mu), cfg.tensor(sd)
        self.tril = (torch.as_tensor(rows, device=cfg.device),
                     torch.as_tensor(cols, device=cfg.device))
        self.on_diag = self.tril[0] == self.tril[1]
        super().__init__(X, D_pts, b0_pts, cfg, rho=rho, lam=lam, lam_mult=lam_mult)

    def predict(self, x):
        """The un-standardised SVR outputs ``(B, d + d(d+1)/2)``."""
        sqd = torch.clamp((x ** 2).sum(1, keepdim=True) + (self.sv_X ** 2).sum(1)[None, :]
                          - 2.0 * (x @ self.sv_X.T), min=0.0)
        z = torch.exp(-self.svr_gamma * sqd) @ self.sv_A + self.y_icpt
        return self.y_mu + self.y_sd * z

    def _moments(self, x):
        y = self.predict(x)
        L = _lower(y[:, self.d:], self.d, *self.tril)
        return _sym(L @ L.mT) + self.rho * self.I[None], y[:, :self.d] + self.eps_b

    def _fields(self, x):
        C, b = self._moments(x)
        return torch.linalg.inv(C), b

    def mobility(self, x):
        return self._moments(x)[0]

    def mobility_divergence(self, x):
        _no_divergence("SvrFW")


class MlpFW(TrueFW):
    """The FW metric with both fields read off one fitted MLP.

    ``x -> (b(x), L(x))``, ``L`` lower-triangular with a softplus diagonal, and
    ``C_rho = L L^T + rho I``, fitted by plain least squares,

        mean |b - b0_i|^2 / var(b0) + mean ||L L^T - D_i||_F^2 / mean ||D_i||_F^2

    (each term 1 for the null predictor), Adam at lr 1e-3, batches of 512.  With
    ``noise_scale = s > 0`` every batch's inputs get ``N(0, s^2 I)`` added, and the
    population minimiser of that loss is the Nadaraya-Watson average under the Gaussian
    kernel ``exp(-|x - x_i|^2 / 2 s^2)`` -- the shipped smoother at ``eps = s/sqrt(2)``,
    PSD, so the factor can reach it; :func:`noise_for_neff` gives the ``s`` of a chosen
    Kish size.  At ``s = 0`` it is an unregularised fit, and off the data it returns
    whatever the net extrapolates (asymptotically affine in ``x``), floored at ``rho I``.
    The extrapolated drift is capped at the largest ``|b|_{C^-1}`` the fit reaches on the
    samples: uncapped, ``|b|`` grows without bound off the data, the admissibility number
    ``|b|/sqrt(|b|^2 + lam^2)`` rounds to 1 in float32 and ``F`` turns non-positive along
    the drift.

    The fit is a function of ``fit_seed`` (initialisation, batches, noise); the net is
    frozen afterwards.  ``fit_loss`` is the loss on the clean samples.
    """

    name = "FFM-mlp"

    def __init__(self, X, D_pts, b0_pts, cfg, rho: float, lam: float | None = None,
                 lam_mult: float = 0.3, noise_scale: float = 0.0, iters: int = 3000,
                 fit_seed: int = 0, width: int = 64, depth: int = 3):
        self.noise_scale, self.fit_seed = float(noise_scale), int(fit_seed)
        Xt, b0, D = cfg.tensor(X), cfg.tensor(b0_pts), cfg.tensor(D_pts)
        net = _seeded_net(lambda: _MomentNet(Xt.cpu().float(), b0.cpu().float(),
                                             D.cpu().float(), width, depth),
                          fit_seed, Xt)
        var_b = ((b0 - b0.mean(0)) ** 2).sum(1).mean()
        ms_D = (D ** 2).sum((1, 2)).mean()

        def loss(x, i):
            b, L = net(x)
            return (((b - b0[i]) ** 2).sum(1).mean() / var_b
                    + ((L @ L.mT - D[i]) ** 2).sum((1, 2)).mean() / ms_D)

        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        g = torch.Generator(device="cpu").manual_seed(self.fit_seed)
        N, d = Xt.shape
        for _ in range(int(iters)):
            i = torch.randint(N, (512,), generator=g).to(cfg.device)
            x = Xt[i]
            if self.noise_scale > 0:
                xi = torch.randn(512, d, generator=g).to(**cfg.torch_kw)
                x = x + self.noise_scale * xi
            opt.zero_grad(set_to_none=True)
            loss(x, i).backward()
            opt.step()
        self.field = _frozen(net)
        with torch.no_grad():
            self.fit_loss = float(loss(Xt, torch.arange(N, device=cfg.device)))
            b, L = self.field(Xt)
            C = _sym(L @ L.mT) + float(rho) * torch.eye(d, **cfg.torch_kw)[None]
            self.b_cap = float(torch.sqrt(torch.einsum(
                "bi,bi->b", b, torch.linalg.solve(C, b)).clamp(min=0.0)).max())
        super().__init__(X, D_pts, b0_pts, cfg, rho=rho, lam=lam, lam_mult=lam_mult)

    def _moments(self, x):
        b, L = self.field(x)
        return _sym(L @ L.mT) + self.rho * self.I[None], b + self.eps_b

    def _fields(self, x):
        C, b = self._moments(x)
        Ci = torch.linalg.inv(C)
        nb = torch.sqrt(torch.einsum("bi,bij,bj->b", b, Ci, b).clamp(min=1e-30))
        return Ci, b * torch.clamp(self.b_cap / nb, max=1.0)[:, None]

    def mobility(self, x):
        return self._moments(x)[0]

    def mobility_divergence(self, x):
        _no_divergence("MlpFW")


class LearnedBarrierFW(TrueFW):
    """``Phi(x) F_FW(x, v)``, ``Phi = 1 + (phi_max - 1)(1 - s(x))``, ``s`` learned support.

    An MLP classifier (BCE, Adam at lr 1e-3) is trained to tell the samples ``X`` (label
    1) from as many points per batch drawn uniformly from ``X``'s per-axis bounding box
    widened by 25 % of its range on each side (label 0), then frozen.  Its odds
    ``exp(logit)`` estimate the density ratio of the cloud to that box, and
    ``s = min(1, odds / odds_q)`` with ``odds_q`` the ``support_q`` quantile of the odds on
    the samples: ``Phi = 1`` wherever the cloud is at least as dense as at its sparsest
    5 % of samples, and ``-> phi_max`` where it has no density at all.  Reading the
    sigmoid itself instead would not work at d = 2, where the cloud fills so much of its
    box that the posterior saturates near 0.87 and ``Phi`` would sit at ~4.6 on the data.
    The support it knows is exactly the cloud the metric is built on -- under the band
    protocol, the shown cells plus the visible band cells.  Moments and window are the
    shipped ones; as in
    :class:`~scripts.method.extend.BarrierFW` the conformal factor leaves the
    admissibility number and the mobility untouched, so Path B keeps the closed-form
    divergence.  ``fit_loss`` is the BCE on the samples against as many fresh box points.
    """

    name = "FFM-learned-barrier"

    def __init__(self, X, D_pts, b0_pts, cfg, rho: float, lam: float | None = None,
                 lam_mult: float = 0.3, phi_max: float = 30.0, iters: int = 3000,
                 fit_seed: int = 0, width: int = 64, depth: int = 3,
                 support_q: float = 0.05):
        self.phi_max, self.fit_seed = float(phi_max), int(fit_seed)
        super().__init__(X, D_pts, b0_pts, cfg, rho=rho, lam=lam, lam_mult=lam_mult)
        lo, hi = self.X.min(0).values, self.X.max(0).values
        lo, hi = lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo)
        net = _seeded_net(lambda: _Mlp(self.X.cpu().float(), 1, width, depth),
                          fit_seed, self.X)
        g = torch.Generator(device="cpu").manual_seed(self.fit_seed)

        def batch(n_pos, idx=None):
            i = torch.randint(self.N, (n_pos,), generator=g).to(self.device) \
                if idx is None else idx
            neg = lo + (hi - lo) * torch.rand(n_pos, self.d, generator=g).to(**cfg.torch_kw)
            y = torch.cat([torch.ones(n_pos), torch.zeros(n_pos)]).to(**cfg.torch_kw)
            return torch.cat([self.X[i], neg]), y

        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(int(iters)):
            x, y = batch(512)
            opt.zero_grad(set_to_none=True)
            tF.binary_cross_entropy_with_logits(net(x)[:, 0], y).backward()
            opt.step()
        self.support = _frozen(net)
        with torch.no_grad():
            x, y = batch(self.N, torch.arange(self.N, device=self.device))
            self.fit_loss = float(tF.binary_cross_entropy_with_logits(net(x)[:, 0], y))
            z = torch.cat([self.support(self.X[a:a + 4096])[:, 0]
                           for a in range(0, self.N, 4096)])
            self.logit_q = float(torch.quantile(z.double().cpu(), float(support_q)))

    def barrier(self, x):
        """``Phi(x)`` from the clipped odds ratio, in log space so it never overflows."""
        s = torch.exp(torch.clamp(self.support(x)[:, 0] - self.logit_q, max=0.0))
        return 1.0 + (self.phi_max - 1.0) * (1.0 - s)

    def tensors(self, x):
        G, beta = super().tensors(x)
        p = self.barrier(x)
        return (p ** 2)[:, None, None] * G, p[:, None] * beta
