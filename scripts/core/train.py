"""Training for Path A: Deterministic Finsler Flow Matching.

Phase 1 (Geodesic Interpolant)
    Train φ_η to minimise the Finsler path energy of the boundary-enforced curve
        x_{t,η} = (1-t)x0 + t x1 + t(1-t) φ_η(t, x0, x1)
        L_geo(η) = E_{(x0,x1)~q, t~U[0,1]} [ F(x_{t,η}, ẋ_{t,η})^2 ].

Phase 3 (Velocity Distillation)  [``train_phase2`` below]
    Freeze φ_η and distil the interpolant velocities into a flow field v_θ
        L_CFM(θ) = E_{(x0,x1)~π*} [ || v_θ(t, x_{t,η}) - ẋ_{t,η} ||^2 ].

Endpoint coupling: Phase 1 trains φ on the independent product coupling
(:class:`EndpointCoupler`); Phase 2 (see ``scripts.core.coupling``) upgrades this to
the geometry-aware entropic-OT coupling π*, which Phase 3 distils over.  Because
:class:`~scripts.core.coupling.OTCoupler` shares the ``.sample`` interface, the
Phase-3 loop (``train_phase2``) is agnostic to which coupling it is handed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .geometry import FinslerGeometry
from .models import PhiNet, VelocityNet


@dataclass
class TrainConfig:
    phase1_iters: int = 2500
    phase2_iters: int = 1500
    batch_size: int = 512
    lr: float = 2e-3
    grad_clip: float = 5.0
    log_every: int = 250
    seed: int = 0
    width: int = 128
    depth: int = 4


# --------------------------------------------------------------------------- #
#  Endpoint coupling  q(x0, x1) = p_0(x0) · p_1(x1)
# --------------------------------------------------------------------------- #
class EndpointCoupler:
    """Independent (product) coupling of the source/target endpoint marginals.

    x0 is drawn uniformly from the labelled source set p_0 and x1 uniformly from
    the target set p_1 (see the per-dataset definitions in ``datasets.py``):
        Arch         : p_0 = left end,  p_1 = right end
        Cycle        : p_0 = top,       p_1 = bottom
        Bifurcation  : p_0 = stem start, p_1 = both branch tips together
    """

    def __init__(self, X: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                 device: torch.device, dtype: torch.dtype, seed: int = 0):
        self.X = torch.as_tensor(X, dtype=dtype, device=device)
        self.src = torch.as_tensor(np.where(p0)[0], device=device)
        self.tgt = torch.as_tensor(np.where(p1)[0], device=device)
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def sample(self, batch: int) -> tuple[Tensor, Tensor]:
        i0 = self.src[torch.randint(0, len(self.src), (batch,),
                                    generator=self.gen, device=self.device)]
        i1 = self.tgt[torch.randint(0, len(self.tgt), (batch,),
                                    generator=self.gen, device=self.device)]
        return self.X[i0], self.X[i1]


# --------------------------------------------------------------------------- #
#  Interpolant + its exact time-derivative
# --------------------------------------------------------------------------- #
def interpolant_and_velocity(
    phi: PhiNet, t: Tensor, x0: Tensor, x1: Tensor, create_graph: bool
) -> tuple[Tensor, Tensor]:
    """Return x_{t,η} and ẋ_{t,η} = d/dt x_{t,η}.

        x_t = (1-t)x0 + t x1 + t(1-t) φ(t)
        ẋ_t = (x1 - x0) + (1-2t) φ(t) + t(1-t) ∂_t φ(t)

    ∂_t φ is obtained by autograd (create_graph=True during Phase 1 so the energy
    can be back-propagated to η).
    """
    t = t.clone().requires_grad_(True)
    phi_val = phi(t, x0, x1)                                     # (B, d)
    d = phi_val.shape[1]

    # ∂_t φ, component by component (each output depends only on its own t row)
    dphi_dt = torch.zeros_like(phi_val)
    for k in range(d):
        grad_k = torch.autograd.grad(
            phi_val[:, k].sum(), t, create_graph=create_graph, retain_graph=True
        )[0]
        dphi_dt[:, k] = grad_k[:, 0]

    x_t = (1 - t) * x0 + t * x1 + t * (1 - t) * phi_val
    x_dot = (x1 - x0) + (1 - 2 * t) * phi_val + t * (1 - t) * dphi_dt
    return x_t, x_dot


# --------------------------------------------------------------------------- #
#  Phase 1 — geodesic interpolant
# --------------------------------------------------------------------------- #
def train_phase1(
    geom: FinslerGeometry, coupler: EndpointCoupler, cfg: TrainConfig,
    device: torch.device, dtype: torch.dtype,
) -> tuple[PhiNet, list[float]]:
    torch.manual_seed(cfg.seed)
    phi = PhiNet(geom.d, width=cfg.width, depth=cfg.depth).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(phi.parameters(), lr=cfg.lr)
    history: list[float] = []

    for it in range(cfg.phase1_iters):
        x0, x1 = coupler.sample(cfg.batch_size)
        t = torch.rand(cfg.batch_size, 1, device=device, dtype=dtype)
        x_t, x_dot = interpolant_and_velocity(phi, t, x0, x1, create_graph=True)
        loss = geom.F2(x_t, x_dot).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(phi.parameters(), cfg.grad_clip)
        opt.step()

        history.append(float(loss))
        if it % cfg.log_every == 0 or it == cfg.phase1_iters - 1:
            print(f"    [phase1] iter {it:5d}  L_geo = {float(loss):.5f}")
    return phi, history


# --------------------------------------------------------------------------- #
#  Phase 2 — velocity distillation (CFM)
# --------------------------------------------------------------------------- #
def train_phase2(
    phi: PhiNet, coupler: EndpointCoupler, cfg: TrainConfig, d: int,
    device: torch.device, dtype: torch.dtype,
) -> tuple[VelocityNet, list[float]]:
    torch.manual_seed(cfg.seed + 1)
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)

    v_net = VelocityNet(d, width=cfg.width, depth=cfg.depth).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(v_net.parameters(), lr=cfg.lr)
    history: list[float] = []

    for it in range(cfg.phase2_iters):
        x0, x1 = coupler.sample(cfg.batch_size)
        t = torch.rand(cfg.batch_size, 1, device=device, dtype=dtype)
        # frozen φ, but we still need ∂_t φ to form the target velocity
        with torch.enable_grad():
            x_t, x_dot = interpolant_and_velocity(phi, t, x0, x1, create_graph=False)
        x_t, x_dot = x_t.detach(), x_dot.detach()

        v_pred = v_net(t.detach(), x_t)
        loss = ((v_pred - x_dot) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), cfg.grad_clip)
        opt.step()

        history.append(float(loss))
        if it % cfg.log_every == 0 or it == cfg.phase2_iters - 1:
            print(f"    [phase2] iter {it:5d}  L_CFM = {float(loss):.6f}")
    return v_net, history


# --------------------------------------------------------------------------- #
#  CFM baseline — straight-line conditional flow matching (no geometry)
# --------------------------------------------------------------------------- #
def train_cfm(
    coupler: EndpointCoupler, cfg: TrainConfig, d: int,
    device: torch.device, dtype: torch.dtype,
) -> tuple[VelocityNet, list[float]]:
    """Vanilla Conditional Flow Matching (Lipman/Tong): straight interpolant.

        x_t = (1-t) x0 + t x1 ,   target velocity = x1 - x0
        L(θ) = E [ || v_θ(t, x_t) - (x1 - x0) ||^2 ]

    This is the geometry-free baseline: it draws straight Euclidean lines between
    coupled endpoints (cf. Phase 2, but with φ ≡ 0 and no metric).
    """
    torch.manual_seed(cfg.seed + 2)
    v_net = VelocityNet(d, width=cfg.width, depth=cfg.depth).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(v_net.parameters(), lr=cfg.lr)
    history: list[float] = []

    for it in range(cfg.phase2_iters):
        x0, x1 = coupler.sample(cfg.batch_size)
        t = torch.rand(cfg.batch_size, 1, device=device, dtype=dtype)
        x_t = (1 - t) * x0 + t * x1
        target = x1 - x0
        loss = ((v_net(t, x_t) - target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), cfg.grad_clip)
        opt.step()

        history.append(float(loss))
        if it % cfg.log_every == 0 or it == cfg.phase2_iters - 1:
            print(f"    [cfm]    iter {it:5d}  L_CFM = {float(loss):.6f}")
    return v_net, history


# --------------------------------------------------------------------------- #
#  Trajectory integration under the learned flow field
# --------------------------------------------------------------------------- #
@torch.no_grad()
def integrate_flow(
    v_net: VelocityNet, x0: Tensor, n_steps: int = 100,
) -> Tensor:
    """Integrate dx/dt = v_θ(t, x) from t=0 to t=1 with RK4.

    Returns the full trajectory tensor of shape (n_steps + 1, B, d).
    """
    device, dtype = x0.device, x0.dtype
    dt = 1.0 / n_steps
    x = x0.clone()
    traj = [x.clone()]
    for n in range(n_steps):
        t0 = torch.full((x.shape[0], 1), n * dt, device=device, dtype=dtype)
        k1 = v_net(t0, x)
        k2 = v_net(t0 + dt / 2, x + dt / 2 * k1)
        k3 = v_net(t0 + dt / 2, x + dt / 2 * k2)
        k4 = v_net(t0 + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.clone())
    return torch.stack(traj, dim=0)


@dataclass
class PathAResult:
    name: str
    phi: PhiNet
    v_net: VelocityNet
    geom: FinslerGeometry
    hist1: list[float] = field(default_factory=list)
    hist2: list[float] = field(default_factory=list)
