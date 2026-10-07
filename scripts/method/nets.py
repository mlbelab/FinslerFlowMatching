"""The networks, the interpolant and the coupler.

**Every arm in every notebook uses the same two networks**, and they are the engine's:
:class:`scripts.core.models.PhiNet` for the geodesic correction and
:class:`scripts.core.models.VelocityNet` for the flow and score fields.  There is one
definition of them in the repository and it lives in the backbone, so a difference
between our arm and a baseline is a difference of geometry, coupling or loss, and never
of capacity.

That is a change from an earlier version of this module, which carried its own three
SELU layers of width 64 with ``t`` concatenated raw — Kapusniak et al.'s ``GeoPathMLP`` /
``VelocityNet``, matched to MFM so *our* arm ran on *their* architecture.  It had the
opposite problem: the two engine baselines (OT-CFM, MFM/LAND) go through
:mod:`scripts.core.train`, which builds the engine nets, so the notebook was comparing
9 027 parameters against 15 171 at ``d = 3, w = 64``, with the extra capacity on the
baselines' side.  Matching one pair meant mismatching the other; the only way to match
all of them is for all of them to share one class.

``GeoPathNet`` is kept as an alias of ``PhiNet`` because it is the name the interpolant
literature uses and the notebooks read better with it.
"""
from __future__ import annotations

import torch

from scripts.core.models import PhiNet, VelocityNet, timestep_embedding

#: The geodesic correction ``phi_eta(t, x_0, x_1) -> R^d``, under its interpolant name.
GeoPathNet = PhiNet

__all__ = ["GeoPathNet", "PhiNet", "VelocityNet", "timestep_embedding",
           "interpolant", "Coupler"]


def interpolant(phi, t, x0, x1, create_graph: bool):
    """``x_t`` and its EXACT time derivative.

    ``x_t = (1-t)x_0 + t x_1 + t(1-t) phi(t, x_0, x_1)``, so the boundary conditions hold
    by construction and are never a penalty term.  ``d_t phi`` is taken by autograd one
    output component at a time — a finite difference here would put the interpolant's
    discretisation error straight into the Finsler energy that Phase 1 minimises.
    """
    t = t.clone().requires_grad_(True)
    p = phi(t, x0, x1)
    dp = torch.stack([torch.autograd.grad(p[:, k].sum(), t, create_graph=create_graph,
                                          retain_graph=True)[0][:, 0]
                      for k in range(p.shape[1])], dim=1)
    x_t = (1.0 - t) * x0 + t * x1 + t * (1.0 - t) * p
    x_dot = (x1 - x0) + (1.0 - 2.0 * t) * p + t * (1.0 - t) * dp
    return x_t, x_dot


class Coupler:
    """Draws ``(x_0, x_1)`` pairs.  ``pi=None`` is the independent product coupling.

    The generator is explicit and CPU-side so that a coupling draw is reproducible
    independently of what the training loop has drawn from the global RNG.
    """

    def __init__(self, X0, X1, pi=None, seed: int = 0):
        self.X0, self.X1, self.pi = X0, X1, pi
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        if pi is not None:
            self.flat = (pi / pi.sum()).reshape(-1).cpu()

    def sample(self, batch: int):
        if self.pi is None:
            i = torch.randint(len(self.X0), (batch,), generator=self.g)
            j = torch.randint(len(self.X1), (batch,), generator=self.g)
        else:
            f = torch.multinomial(self.flat, batch, replacement=True, generator=self.g)
            i, j = f // self.pi.shape[1], f % self.pi.shape[1]
        return self.X0[i.to(self.X0.device)], self.X1[j.to(self.X1.device)]
