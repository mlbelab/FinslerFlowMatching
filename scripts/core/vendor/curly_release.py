"""Curly-FM single-marginal primitives — Petrović et al. (2025), verbatim.

Sources, both from https://github.com/kpetrovicc/curly-flow-matching (MIT, Copyright (c)
2025 Katarina Petrovic):

* ``src/models/components/mlp.py``                    -> :class:`MLP`
* ``src/models/components/single_marginal_utils.py``  -> :func:`get_xt`,
  :func:`get_xt_xt_dot`, :func:`get_u_xt`, :func:`coupling`

Copied rather than imported so that ``public/`` runs without their checkout; see
:mod:`scripts.core.vendor`.  The four functions are the ones
:mod:`scripts.core.curly_fm` and :mod:`scripts.experiments.erythroid.reference_train` call.  The
rest of ``single_marginal_utils`` is plotting, a torchcfm-derived ``sample_ot``, and a
Lightning wrapper we never touch, so it is not copied — this module is a *subset* of that
file, unlike :mod:`scripts.core.vendor.mfm_land`, which is a whole one.

**Two deviations, both in the imports and neither in a function body.**

* Their file opens ``import scipy`` and then calls ``scipy.optimize.linear_sum_assignment``
  in :func:`coupling`.  That works in their environment only because some other import
  (``seaborn`` -> ``scipy.optimize``) has already bound the submodule.  With their
  plotting imports dropped, the same line raises ``AttributeError``, so this copy imports
  ``scipy.optimize`` explicitly.  The call site is unchanged.
* ``matplotlib``, ``seaborn``, ``os``, ``pot`` and ``gaussian_kde`` are dropped with the
  functions that used them.
"""
import numpy as np  # noqa: F401  - kept: their file's namespace, used by the dropped half
import scipy.optimize  # see the module docstring: theirs is a bare ``import scipy``
import torch
import torch.nn.functional as F  # noqa: F401  - as above
from torch.func import vmap


# --- src/models/components/mlp.py ------------------------------------------------- #
class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# --- src/models/components/single_marginal_utils.py -------------------------------- #
def get_xt(t, x0, x1, geodesic_model, sigma=0.0):
    mu_t = (1 - t) * x0 + t * x1 +  t * (1-t) * (geodesic_model(torch.cat([x0, x1, t], dim=-1)))
    epsilon = torch.randn_like(x0)
    x_t = mu_t + torch.sqrt(t*(1-t))*sigma * epsilon
    return mu_t, x_t, epsilon

def get_xt_xt_dot(t, x0, x1, geodesic_model, sigma=0.0):
    with torch.enable_grad():
        t = t[..., None]
        t.requires_grad_(True)
        mu_t, xt, eps = get_xt(t, x0, x1, geodesic_model, sigma=sigma)
        mu_t_dot_list = []
        for i in range(xt.shape[-1]):
            mu_t_dot_list.append(
                torch.autograd.grad(torch.sum(mu_t[..., i]), t, create_graph=True)[0]
            )
        mu_t_dot = torch.cat(mu_t_dot_list, -1)
    return xt, mu_t_dot, eps

def get_u_xt(xt, x, v, k=20):
    dists = torch.cdist(xt, x)
    knn_dists, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)
    h = knn_dists[:, -1:].clamp_min(1e-12)
    w = torch.exp(-(knn_dists**2) / (2 * h**2))
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    v_knn = v[knn_idx]
    v_xt = (w.unsqueeze(-1) * v_knn).sum(dim=1)

    return v_xt

def coupling(x0, x1, batch_size, xs, vs, geodesic_model, sigma):
    t = torch.rand(1).type_as(x0) * torch.ones(batch_size, batch_size, device=x0.device)
    x0_r = x0.repeat(batch_size, 1, 1)
    x1_r = x1.repeat(batch_size, 1, 1).transpose(0, 1)
    xt, mu_t_dot, eps = get_xt_xt_dot(t, x0_r, x1_r, geodesic_model, sigma=sigma)
    ut = vmap(lambda x: get_u_xt(x, xs, vs))(xt)
    L2_cost = 0.5*((mu_t_dot.detach() - ut)**2).sum(-1)
    _, col_ind = scipy.optimize.linear_sum_assignment(L2_cost.detach().cpu().numpy())
    pi_x0 = x0[col_ind]
    pi_x1 = x1
    return pi_x0, pi_x1
