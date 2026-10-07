"""The fitted extensions of the moment fields, for ``notebooks/pancreas_extensions.ipynb``.

The paper extends the per-point moments with a Nadaraya-Watson smoother and names a
network and an SVR as alternatives.  :mod:`scripts.method.extend_fit` builds those -- the
log-Euclidean (exp-log) mean of the second moments on the shipped window, an RBF
epsilon-SVR, and an MLP fitted by least squares, plain and with the input noise of an
``n_eff = 100`` window -- each at the shipped absolute ``(rho, lambda)``, so an arm differs
from the shipped metric by the extension alone.  The runs are cells of the robustness grid
(:mod:`scripts.experiments.pancreas.review`) and the off-support numbers are its support
diagnostics, so the notebook and that grid read one set of records; this module adds the
fit on the samples and the figures.

The cost maps show ``c(x) = E_v F(x, v)`` over unit ``v``, relative to its mean on the
samples: below 1 the metric makes moving there cheaper than moving through the data, and
that is where a geodesic between the two marginals goes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from scripts.core.present import FAINT, GAP, INK, PAPER_REGIONS, SOURCE, TARGET
from scripts.method import Run, moments_from

from . import review as rv
from .datasets import build_training_set, load_cloud

ARMS = ("ffm", "ffm_logeuc", "ffm_svr", "ffm_mlp", "ffm_mlp_noise")
REFERENCE = ("cfm", "mfm_land", "curly")
DIMS = (2, 10)
#: the colour scale of the cost maps: log10 of the cost over its mean on the samples
COST_RANGE = (-1.0, 1.0)
SUPPORT_COLS = ("void chord", "void scored", "cost void / data", "cost scored / data",
                "max |beta|")
REGION_LABELS = {SOURCE: "$p_0$ (shown)", TARGET: "$p_1$ (shown)", GAP: "withheld (scored)"}


@dataclass
class Fit:
    """One space's training set, per-point moments and the metric of every arm."""
    cloud: object
    ts: object
    b0: np.ndarray
    D: np.ndarray
    metrics: dict
    seconds: dict


def cells(smoke: bool = False) -> list:
    """The grid cells these arms and the baselines occupy at d = 2 and 10: the seed pool,
    seeds 0-2 and the 1e-7 perturbation draws -- no search, band or hardware cells."""
    return [c for c in rv.cells(DIMS, smoke)
            if c[1] in ARMS + REFERENCE and c[3] in rv.PERTS]


def ensure(cfg: Run, tuned: dict) -> dict:
    """The runs and the support records, training or measuring only what disk lacks."""
    recs = rv.ensure_cells(cfg, tuned, cells(cfg.smoke))
    rv.ensure_support(cfg, tuned, DIMS, recs["root"], arms=ARMS)
    for d in DIMS:
        for a in ARMS:
            with open(rv.support_path(recs["root"], d, a)) as fh:
                recs["support"][(d, a)] = json.load(fh)
    return recs


def header(recs: dict) -> str:
    devs = sorted({r.get("device", "?") for r in recs["runs"].values()})
    return (f"{len(recs['runs'])}/{len(cells(recs['smoke']))} runs, "
            f"{len(recs['support'])} support records, {', '.join(devs)}")


def fit(dim: int, cfg: Run, tuned: dict) -> Fit:
    """Every arm's metric on the shipped training set of ``dim``, timed."""
    cloud = load_cloud(dim)
    ts = build_training_set(cloud)
    X = np.asarray(ts.X, dtype=np.float64)
    b0, D = moments_from(ts.P, X)
    metrics, seconds = {}, {}
    for a in ARMS:
        t0 = time.perf_counter()
        metrics[a] = rv.build_metric(a, X, b0, D, cfg, tuned[dim])
        seconds[a] = time.perf_counter() - t0
    return Fit(cloud, ts, b0, D, metrics, seconds)


@torch.no_grad()
def frame_fit(fits: dict, chunk: int = 2048) -> pd.DataFrame:
    """On the samples: how far each arm's ``C_rho`` and ``b`` sit from the per-point
    moments (median error over median size), the SVR's support vectors per output as a
    fraction of the samples, and the build time."""
    rows = {}
    for d, f in fits.items():
        for arm, m in f.metrics.items():
            C0 = torch.as_tensor(f.D, device=m.device, dtype=m.dtype) + m.rho * m.I[None]
            b0 = torch.as_tensor(f.b0, device=m.device, dtype=m.dtype)
            eC, eb = [], []
            for a in range(0, m.N, chunk):
                x = m.X[a:a + chunk]
                eC.append(torch.linalg.matrix_norm(m.mobility(x) - C0[a:a + chunk]))
                eb.append((m._fields(x)[1] - b0[a:a + chunk]).norm(dim=1))
            rows[(d, rv.ARM_LABELS[arm])] = {
                "C error / size": float(torch.cat(eC).median()
                                        / torch.linalg.matrix_norm(C0).median()),
                "b error / size": float(torch.cat(eb).median() / b0.norm(dim=1).median()),
                "SV fraction": (float((m.sv_A != 0).sum(0).double().mean()) / m.N
                                if hasattr(m, "sv_A") else np.nan),
                "fit s": f.seconds[arm]}
    df = pd.DataFrame(rows).T
    df.index.names = ["d", "on the samples"]
    return df


def frame_support(recs: dict) -> pd.DataFrame:
    """The grid's support diagnostics for these arms, without the kernel's ``n_eff``
    columns (a property of the shipped window, not of a fitted extension)."""
    keep = {(d, a): r for (d, a), r in recs["support"].items() if a in ARMS}
    df = rv.frame_support({**recs, "support": keep})
    if not len(df):
        return df
    order = [(d, rv.ARM_LABELS[a]) for d in DIMS for a in ARMS if (d, a) in keep]
    return df.loc[order, list(SUPPORT_COLS)]


@torch.no_grad()
def travel_cost(metric, P: torch.Tensor, n_dirs: int = 32, seed: int = 0,
                chunk: int = 2048) -> np.ndarray:
    """``E_v F(x, v)`` over ``n_dirs`` fixed unit directions, per row of ``P``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n_dirs, metric.d, generator=g)
    v = (v / v.norm(dim=1, keepdim=True)).to(metric.device, metric.dtype)
    out = []
    for a in range(0, len(P), chunk):
        G, beta = metric.tensors(P[a:a + chunk])
        vGv = torch.einsum("kd,bde,ke->bk", v, G, v)
        out.append((metric.scale * (torch.sqrt(vGv.clamp(min=1e-9)) + beta @ v.T)).mean(1))
    return torch.cat(out).double().cpu().numpy()


def _grid(X: np.ndarray, n: int, pad: float = 0.08):
    lo, hi = X.min(0), X.max(0)
    lo, hi = lo - pad * (hi - lo), hi + pad * (hi - lo)
    U, W = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    return U, W, np.stack([U.ravel(), W.ravel()], 1)


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _regions(ax, f: Fit, s: float = 0.6, alpha: float = 0.45) -> None:
    X = np.asarray(f.ts.X)
    test = f.cloud.X[f.cloud.target_test]
    for pts, r in ((X[f.ts.p0], SOURCE), (X[f.ts.p1], TARGET), (test, GAP)):
        ax.scatter(pts[:, 0], pts[:, 1], s=s, lw=0, c=PAPER_REGIONS[r], alpha=alpha,
                   rasterized=True)


def _legend(fig, cfg: Run, entries) -> None:
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=3, color=c, label=lab)
                        for c, lab in entries],
               fontsize=cfg.fs_legend + 1, loc="outside lower center", ncol=len(entries),
               frameon=False, handletextpad=0.1, columnspacing=1.5)


def figure_cost(f: Fit, cfg: Run, stem: str, n: int = 160):
    """log10 travel cost over the d = 2 chart, one panel per arm, a shared scale."""
    U, W, P = _grid(np.asarray(f.cloud.X, dtype=np.float64), n)
    P = cfg.tensor(P)
    fig, axes = plt.subplots(1, len(f.metrics), figsize=(1.9 * len(f.metrics) + 0.5, 2.5),
                             layout="constrained", squeeze=False)
    for ax, (arm, m) in zip(axes[0], f.metrics.items()):
        c = np.log10(travel_cost(m, P) / travel_cost(m, m.X).mean()).reshape(U.shape)
        im = ax.pcolormesh(U, W, c, cmap="Greys", vmin=COST_RANGE[0], vmax=COST_RANGE[1],
                           shading="auto", rasterized=True)
        _regions(ax, f, s=0.4, alpha=0.35)
        ax.set_title(rv.ARM_LABELS[arm], fontsize=cfg.fs_heading)
        _bare(ax)
    bar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, pad=0.01, extend="both")
    bar.set_label("log10 cost / mean on the samples", fontsize=cfg.fs_legend + 1)
    bar.ax.tick_params(labelsize=cfg.fs_legend, width=0.5, length=2)
    bar.outline.set_linewidth(0.5)
    _legend(fig, cfg, [(PAPER_REGIONS[r], lab) for r, lab in REGION_LABELS.items()])
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf")
    return fig


def figure_pool(recs: dict, cfg: Run, stem: str):
    """W2 at t = 1/2, one dot per seed (the d = 2 pool, seeds 0-2 at d = 10), median bar."""
    arms = ARMS + REFERENCE
    fig, axes = plt.subplots(1, len(DIMS), figsize=(6.4, 2.4), layout="constrained",
                             sharey=True, squeeze=False)
    jit = np.random.default_rng(0)
    for ax, d in zip(axes[0], DIMS):
        for i, a in enumerate(arms):
            seeds = range(rv.POOL.get(a, 0)) if d == rv.POOL_DIM else rv.BASE_SEEDS
            v = np.array(list(rv._w2(recs["runs"], d, a, seeds).values()))
            if not len(v):
                continue
            y = i + jit.uniform(-0.2, 0.2, len(v))
            ax.scatter(v, y, s=7, lw=0, c=INK if a in ARMS else FAINT, alpha=0.75)
            ax.plot([np.median(v)] * 2, [i - 0.32, i + 0.32], c=INK, lw=1.0)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([rv.ARM_LABELS[a] for a in arms], fontsize=cfg.fs_legend + 1)
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize=cfg.fs_legend + 1, width=0.5, length=2)
        ax.tick_params(axis="y", width=0, length=0)
        ax.grid(axis="x", lw=0.4, c=FAINT, alpha=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        n = "pool" if d == rv.POOL_DIM else "seeds 0-2"
        ax.set_title(f"d = {d}, {n}", fontsize=cfg.fs_heading)
        ax.set_xlabel("W2 at t = 1/2", fontsize=cfg.fs_legend + 1)
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf")
    return fig


def figure_mid(recs: dict, f: Fit, cfg: Run, stem: str, seed: int = 0):
    """Where each arm puts ``p_0`` at ``t = 1/2`` (the scored slice), d = 2, one seed."""
    shown = f.cloud.X[f.cloud.shown]
    test = f.cloud.X[f.cloud.target_test]
    fig, axes = plt.subplots(1, len(ARMS), figsize=(1.9 * len(ARMS), 2.5),
                             layout="constrained", sharex=True, sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], ARMS):
        c = (rv.POOL_DIM, arm, seed, "none")
        mid = np.load(rv.mid_path(recs["root"], c))["mid"]
        ax.scatter(shown[:, 0], shown[:, 1], s=0.4, lw=0, c=FAINT, alpha=0.5, rasterized=True)
        ax.scatter(test[:, 0], test[:, 1], s=0.6, lw=0, c=PAPER_REGIONS[GAP], alpha=0.5,
                   rasterized=True)
        ax.scatter(mid[:, 0], mid[:, 1], s=0.6, lw=0, c=INK, alpha=0.7, rasterized=True)
        ax.set_title(f"{rv.ARM_LABELS[arm]}\nW2 {recs['runs'][c]['W2_mid']:.3f}",
                     fontsize=cfg.fs_heading, linespacing=1.1)
        _bare(ax)
    _legend(fig, cfg, [(FAINT, "shown"), (PAPER_REGIONS[GAP], REGION_LABELS[GAP]),
                       (INK, "model at t = 1/2")])
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf")
    return fig
