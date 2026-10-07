"""The paper's pancreas path figure, drawn from the saved slabs rather than from memory.

Every arm pushes the same source cloud forward, so the columns
:func:`~scripts.experiments.pancreas.train.save_trajectory` keeps are the same particles
in every slab and one choice of start points serves all the panels.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.experiments.sheet.datasets import GAP, SOURCE, TARGET
from scripts.experiments.sheet.paper_focus import QUAL_DET, QUAL_NOISY, two_line
from scripts.method import figures as nbfig

from .datasets import TRAIN_BINS
from .report import load_slab
from .train import ARM_LABELS

N_PATHS = 5
NCOL = 2
BLOCKS = (("Data", ("__data__",)),
          (r"Deterministic ($\sigma = 0$)", ("cfm", "mfm_land", "curly", "ffm")),
          (r"Stochastic ($\sigma > 0$)", ("pathb",)))
PANEL_LABELS = {"ffm": two_line(QUAL_DET, ours=True),
                "pathb": two_line(QUAL_NOISY, ours=True)}


def _label(arm: str) -> str:
    if arm == "__data__":
        return "the withheld marginal"
    return PANEL_LABELS.get(arm, ARM_LABELS[arm])


def figure_paths(root: str, dim: int, cloud, out_dir: str, cfg, seed: int = 0,
                 stem: str = "pancreas_paths") -> str:
    pf = nbfig.sheet_primitives()
    reg = nbfig.PAPER_REGIONS
    U = cloud.umap
    mu, sd = U.mean(axis=0), U.std(axis=0)
    lo, hi = U.min(axis=0), U.max(axis=0)
    pad = 0.04 * (hi - lo)

    cells = [a for _h, arms in BLOCKS for a in arms]
    slabs = {a: load_slab(root, dim, a, seed) for a in cells if a != "__data__"}
    cols = pf._lateral_starts(slabs["ffm"]["paths"][0], N_PATHS)

    nrow = -(-len(cells) // NCOL)
    fig, axes = plt.subplots(nrow, NCOL, layout="constrained",
                             figsize=(2.95 * NCOL, 2.35 * nrow))
    axes = np.asarray(axes).reshape(-1)

    for k, arm in enumerate(cells):
        ax = axes[k]
        for b, colour in ((TRAIN_BINS[0], reg[SOURCE]), (TRAIN_BINS[1], reg[TARGET])):
            m = cloud.marginal(b)
            ax.scatter(U[m, 0], U[m, 1], s=1.4, lw=0, c=colour, alpha=0.42, zorder=1)
        m = cloud.target_test
        ax.scatter(U[m, 0], U[m, 1], s=2.4, lw=0, c=reg[GAP], alpha=1.0, zorder=1)
        if arm != "__data__":
            pf._draw_paths(ax, slabs[arm]["paths"][:, cols] * sd + mu, three_d=False)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        label = _label(arm)
        ax.set_title(label, fontsize=cfg.fs_heading, linespacing=1.1,
                     pad=2 + cfg.fs_heading * label.count("\n"))
        if k == 0:
            ax.set_ylabel(f"$d = {dim}$", fontsize=cfg.fs_heading)
    for ax in axes[len(cells):]:
        ax.set_axis_off()

    dot = dict(marker="o", ls="", ms=3)
    fig.legend(handles=[
        plt.Line2D([], [], **dot, color=reg[SOURCE], label="$p_0$ (shown)"),
        plt.Line2D([], [], **dot, color=reg[TARGET], label="$p_1$ (shown)"),
        plt.Line2D([], [], **dot, color=reg[GAP],
                   label="withheld $t = 1/2$ marginal (scored)"),
        plt.Line2D([], [], marker="o", ls="", ms=4, color="#ffffff",
                   markeredgecolor=nbfig.INK, label="$t{=}0$"),
        plt.Line2D([], [], marker="*", ls="", ms=7, color="#ffffff",
                   markeredgecolor=nbfig.INK, label="$t{=}1$")],
        loc="outside lower center", ncol=5, frameon=False,
        fontsize=cfg.fs_legend, handletextpad=0.3, columnspacing=1.5)

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.show()
    return written[0]
