"""One dataclass holding everything a notebook run fixes, in place of module globals.

Every constant here was a bare name in the notebook's constants cell, read implicitly by
a dozen functions.  Bundling them means a function's inputs are visible in its signature,
a second experiment can vary one of them without editing the shared code, and a smoke run
is a different :class:`Run` rather than an ``if SMOKE`` scattered through the file.

Nothing selectable lives here.  Every ladder — rho, lambda, sigma, the baselines' own
knobs — is swept, and what the sweep picks is passed explicitly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import torch


@dataclass
class Run:
    """The fixed part of a notebook run.  See :func:`Run.from_env` for the usual entry."""

    # ---- run control ----------------------------------------------------------- #
    smoke: bool = False
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32
    seeds: tuple[int, ...] = (0, 1, 2)          # model seeds; the split is held fixed
    split_seed: int = 0
    tune_seed: int = 0

    # ---- moments of P, and the smoothing kernel -------------------------------- #
    eps_kernel_scale: float = 1.0               # eps = median positive nn distance
    eps_den: float = 1e-6                       # kernel-smoothing denominator floor
    eps_b: float = 1e-6                         # floor on the smoothed first moment

    # ---- Phase 1: the geodesic interpolant ------------------------------------- #
    geo_iters: int = 2500
    geo_lr: float = 2e-3
    # One architecture for every arm — ours and all three baselines.  These two go both
    # into ``scripts.method``'s own nets and into the engine's ``HParams``, so a notebook cannot
    # give one arm more capacity than another.  See :mod:`scripts.method.nets`.
    net_width: int = 64
    net_depth: int = 4                          # hidden layers of the shared MLP
    batch: int = 256
    grad_clip: float = 5.0

    # ---- Phase 2: the Finsler-cost entropic OT --------------------------------- #
    ot_k: int = 8                               # quadrature nodes for C_ij
    ot_max_pts: int = 200                       # endpoints per side entering Sinkhorn
    ot_blur_frac: float = 0.1                   # reg = ot_blur_frac * median(C)
    ot_n_iter: int = 20_000
    ot_chunk: int = 4096

    # ---- Phase 3: distillation, and the pushforward ---------------------------- #
    cfm_iters: int = 1500
    cfm_lr: float = 2e-3
    n_steps: int = 100                          # integration steps for p_0's pushforward

    # ---- Path B: the Schrodinger bridge ---------------------------------------- #
    sb_iters: int | None = None                 # None -> cfm_iters, so the pair matches
    sb_t_eps: float = 1e-2                      # t clamp; both targets carry 1/(t(1-t))
    sb_dt_fd: float = 2e-3                      # central difference for d_t M

    # ---- misc -------------------------------------------------------------------#
    normalise_f: bool = True                    # one constant per metric so E[F] = 1
    fs_heading: int = 8                         # figure typography: every header
    fs_legend: int = 5

    def __post_init__(self) -> None:
        if self.sb_iters is None:
            self.sb_iters = self.cfm_iters

    # -- constructors -------------------------------------------------------------#
    @classmethod
    def from_env(cls, **over) -> "Run":
        """The usual entry: honour ``NB_SMOKE`` / ``NB_CPU``, then apply overrides.

        ``NB_SMOKE=1`` cuts every iteration budget and drops to one seed, so a notebook
        checks its wiring in minutes; the numbers it then prints are meaningless and the
        notebooks say so.  ``NB_CPU=1`` forces CPU.
        """
        smoke = bool(int(os.environ.get("NB_SMOKE", "0")))
        device = torch.device("cpu" if os.environ.get("NB_CPU") else
                              ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = cls(smoke=smoke, device=device)
        if smoke:
            cfg = replace(cfg, seeds=(0,), geo_iters=200, cfm_iters=200, sb_iters=None)
        return replace(cfg, **over) if over else cfg

    def but(self, **over) -> "Run":
        """A copy with fields replaced — for an ablation that changes one budget."""
        return replace(self, **over)

    # -- convenience --------------------------------------------------------------#
    @property
    def arm_kw(self) -> dict:
        """The :class:`scripts.core.arms.HParams` fields a baseline must share with us.

        Every knob a notebook fixes rather than selects — the budgets, the optimiser, the
        architecture, the OT quadrature — in one place, so an engine arm and our own
        trainer cannot drift apart by an edit to only one of them.  ``curly_net`` is here
        for the same reason: it is the switch that puts the one remaining baseline with
        its own network class onto the shared one.  A caller adds the *selected* knobs
        (``land_rho``, ``curly_alpha``, ``sigma``) on top.
        """
        return {"lr": self.geo_lr, "batch_size": self.batch, "grad_clip": self.grad_clip,
                "phase1_iters": self.geo_iters, "phase2_iters": self.cfm_iters,
                "ot_K": self.ot_k, "ot_max_pts": self.ot_max_pts,
                "blur_frac": self.ot_blur_frac, "curly_net": "shared", **self.net_kw}

    @property
    def net_kw(self) -> dict:
        """``dict(width=..., depth=...)`` — the shared architecture, as keywords.

        Keywords and not positional, because :class:`scripts.core.models.PhiNet`'s
        second positional argument is ``t_emb_dim`` and passing a width there would
        silently build a different net.
        """
        return {"width": self.net_width, "depth": self.net_depth}

    @property
    def torch_kw(self) -> dict:
        """``dict(device=..., dtype=...)``, the pair half the calls below need."""
        return {"device": self.device, "dtype": self.dtype}

    def tensor(self, a) -> torch.Tensor:
        return torch.as_tensor(a, **self.torch_kw)

    def __str__(self) -> str:
        return (f"device {self.device}   seeds {self.seeds}   smoke {self.smoke}   "
                f"iters {self.geo_iters}/{self.cfm_iters}/{self.sb_iters}")
