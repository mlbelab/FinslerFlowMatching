"""Neural networks for Path A (Deterministic Finsler Flow Matching).

PhiNet       : the boundary-vanishing correction φ_η(t, x0, x1) that bends the
               straight-line interpolant into an energy-minimising geodesic.
VelocityNet  : the distilled continuous flow field v_θ(t, x).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def timestep_embedding(t: Tensor, dim: int = 32, max_period: float = 100.0) -> Tensor:
    """Sinusoidal embedding of scalar t ∈ [0, 1]  ->  (B, dim)."""
    assert t.dim() == 2 and t.shape[1] == 1, "t must be (B, 1)"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:  # pad to requested width if odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 128, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PhiNet(nn.Module):
    """φ_η(t, x0, x1) — the correction term of the boundary-enforced interpolant

        x_{t,η} = (1-t) x0 + t x1 + t(1-t) φ_η(t, x0, x1).

    Because it is multiplied by t(1-t), the boundary conditions
    x_{0}=x0 and x_{1}=x1 hold for *any* φ_η.
    """

    def __init__(self, d: int, t_emb_dim: int = 32, width: int = 128, depth: int = 4):
        super().__init__()
        self.d = d
        self.t_emb_dim = t_emb_dim
        self.mlp = _MLP(t_emb_dim + 2 * d, d, width=width, depth=depth)

    def forward(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        assert t.shape[1] == 1 and x0.shape[1] == self.d and x1.shape[1] == self.d
        te = timestep_embedding(t, self.t_emb_dim)
        return self.mlp(torch.cat([te, x0, x1], dim=1))


class VelocityNet(nn.Module):
    """v_θ(t, x) — the distilled continuous flow-matching vector field."""

    def __init__(self, d: int, t_emb_dim: int = 32, width: int = 128, depth: int = 4):
        super().__init__()
        self.d = d
        self.t_emb_dim = t_emb_dim
        self.mlp = _MLP(t_emb_dim + d, d, width=width, depth=depth)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        assert t.shape[1] == 1 and x.shape[1] == self.d
        te = timestep_embedding(t, self.t_emb_dim)
        return self.mlp(torch.cat([te, x], dim=1))
