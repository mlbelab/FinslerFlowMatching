"""Path B: the same geometry with the SDE on.

``p_t(x | x_0, x_1) = N(mu_t, sigma^2 t(1-t) M_t)`` with ``mu_t`` the frozen geodesic
interpolant and ``M_t = C_rho(mu_t)``, trained by joint CFM + CSM matching and sampled by
anisotropic Euler–Maruyama,

    dX = (v_theta + sigma^2 M s_phi + sigma^2 div M) dt + sqrt(2) sigma M^{1/2} dW,

the last drift term being the Itô correction for reading M at the state rather than at
the conditioning mean (see :mod:`scripts.core.bridge`; it vanishes for a constant M).

The conditional targets are :mod:`scripts.core.bridge`'s — **imported, not transcribed**.
Only the training loop is local, so the noisy arm uses the same SELU stack and the same
Phase-3 budget as the deterministic one and the pair differs by the SDE alone.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.core.bridge import (BridgeConfig, integrate_sde, mobility_divergence,
                                 sample_bridge_targets)
from scripts.method.config import Run
from scripts.method.metric import Metric
from scripts.method.nets import Coupler, VelocityNet


def train_sbm(phi, metric: Metric, coupler: Coupler, sigma: float, seed: int, cfg: Run,
              iters: int | None = None, log_every: int = 0):
    """Phase 3', Schrödinger-bridge matching: ``L_CFM(theta) + L_CSM(phi)`` at fixed sigma.

    The two parameter sets are disjoint and Adam is per-parameter scale-adaptive, so the
    plain sum is not a weighting problem: neither term rescales the other's step.
    """
    iters = (cfg.sb_iters if iters is None else iters)
    bcfg = BridgeConfig(sbm_iters=iters, batch_size=cfg.batch, lr=cfg.cfm_lr,
                        grad_clip=cfg.grad_clip, t_eps=cfg.sb_t_eps, dt_fd=cfg.sb_dt_fd,
                        width=cfg.net_width, depth=cfg.net_depth, seed=seed + 2)
    torch.manual_seed(bcfg.seed)
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    v_net = VelocityNet(metric.d, **cfg.net_kw).to(cfg.device, cfg.dtype)
    s_net = VelocityNet(metric.d, **cfg.net_kw).to(cfg.device, cfg.dtype)
    params = list(v_net.parameters()) + list(s_net.parameters())
    opt = torch.optim.Adam(params, lr=bcfg.lr)
    gen = torch.Generator(device=cfg.device).manual_seed(seed + 17)
    for it in range(iters):
        x0, x1 = coupler.sample(cfg.batch)
        t = torch.rand(cfg.batch, 1, **cfg.torch_kw)
        x, v_tgt, s_tgt, lam = sample_bridge_targets(phi, metric, x0, x1, t, sigma, bcfg,
                                                     generator=gen)
        t = torch.clamp(t, bcfg.t_eps, 1.0 - bcfg.t_eps)
        l_cfm = ((v_net(t, x.detach()) - v_tgt.detach()) ** 2).mean()
        l_csm = ((lam.detach() ** 2) * (s_net(t, x.detach()) - s_tgt.detach()) ** 2).mean()
        loss = l_cfm + l_csm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, bcfg.grad_clip)
        opt.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"    iter {it:5d}  L_CFM {float(l_cfm):.5f}  L_CSM {float(l_csm):.5f}")
    for p in phi.parameters():
        p.requires_grad_(True)
    return v_net, s_net


class SdeDrift:
    """``v_theta + sigma^2 M(x) s_phi + sigma^2 div M(x)`` — the field that moved the cloud.

    Same ``(t, x)`` signature as a :class:`~scripts.method.nets.VelocityNet`, so anything
    downstream that reads a run's ``v_net`` scores Path B on its full drift and not on
    the drift network alone.  The score and Itô terms are part of the dynamics that
    produced the cloud the distance columns measure, so leaving them out would score a
    different model from the one that was run — this is the same total drift
    :func:`scripts.core.bridge.integrate_sde` steps with, term for term.
    """

    def __init__(self, v_net, s_net, metric: Metric, sigma: float):
        self.v_net, self.s_net, self.metric, self.sigma = v_net, s_net, metric, sigma

    def __call__(self, t, x):
        out = self.v_net(t, x)
        if self.sigma > 0.0:
            M = self.metric.mobility(x)
            out = out + self.sigma ** 2 * torch.einsum("bij,bj->bi", M, self.s_net(t, x))
            out = out + self.sigma ** 2 * mobility_divergence(self.metric, x)
        return out


def path_b(metric: Metric, phi, coupler: Coupler, X0_t, sigma: float, seed: int,
           cfg: Run, verbose: bool = False):
    """Phase 3' + the anisotropic Euler–Maruyama pushforward, on a frozen ``phi`` / ``pi*``.

    ``phi`` and ``coupler`` come from the deterministic arm at the same seed, which is
    what makes the two columns one method sampled two ways rather than two methods.
    """
    v_net, s_net = train_sbm(phi, metric, coupler, sigma, seed, cfg,
                             log_every=250 if verbose else 0)
    traj = integrate_sde(v_net, s_net, metric, X0_t, sigma, n_steps=cfg.n_steps,
                         seed=seed, t_eps=cfg.sb_t_eps)
    return {"phi": phi, "coupler": coupler,
            "v_net": SdeDrift(v_net, s_net, metric, sigma),
            "drift_net": v_net, "score_net": s_net, "sigma": sigma,
            "traj": traj.cpu().numpy().astype(np.float64)}
