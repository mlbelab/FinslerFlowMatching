import contextlib
import io
import json
import os
import pathlib
import time
import zlib

import numpy as np
import scipy.sparse as sp
import torch

from ffm.core.arms import HParams, run_arm
from ffm.core.bridge import integrate_sde
from ffm.core.present import QUAL_DET, QUAL_NOISY, two_line
from ffm.core.train import integrate_flow
from ffm.method import Coupler, Run, cuda_warm_up, integrate, sigma_for, train_sbm
from scripts import records as rec
from scripts.sheet.protocol import corridor_mask
from scripts.sheet.split import make_split, split_training_set

from . import calibration as cal
from .datasets import FATE_NAMES, make_dataset

ALLOCATIONS = {"50": "forksheet_calibration", "70": "forksheet_calibration", "40": "forksheet_allocation",
               "20": "forksheet_allocation", "10": "forksheet_allocation"}
SEEDS = tuple(range(10))
ROWS = (("OT-CFM", "cfm", 0.0, "cfm"), ("MFM / LAND", "mfm_land", 0.0, "mfm_land"),
        ("Curly-FM", "curly", 0.0, "curly"), ("FFM (ours), deterministic", "ffm_det", 0.0, "ffm_det"),
        ("FFM (ours), with noise", "ffm_CC", cal.TUNED_WIDTH, "ffm_C"))
BASELINES = ("cfm", "mfm_land", "curly")
FIG_TAG, FIG_SEED, N_PATHS = "50", 0, 5
BLOCKS = ((r"Deterministic ($\sigma = 0$)", ("cfm", "mfm_land", "curly", "ffm_det")),
          (r"Stochastic ($\sigma > 0$)", ("ffm_CC",)))
FIG_LABELS = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly": "Curly-FM",
              "ffm_det": two_line(QUAL_DET, ours=True), "ffm_CC": two_line(QUAL_NOISY, ours=True)}


def _record(tag, seed, arm, width):
    return cal.record_path(rec.path(ALLOCATIONS[tag]), tag, seed, arm, width)


def _paths_file():
    return rec.path(ALLOCATIONS[FIG_TAG], f"sh{FIG_TAG}_s{FIG_SEED}_trajectories.npz")


def expected():
    """The 250 Table 9 records and the Figure 5(b) trajectory file."""
    return [_record(t, s, arm, w) for t in ALLOCATIONS for _, arm, w, _ in ROWS for s in SEEDS] + [_paths_file()]


def ensure(cfg=None):
    """Re-create missing records and the trajectory file; nothing runs when all exist."""
    def build(missing):
        cfg_ = Run.from_env() if cfg is None else cfg
        gone = set(missing) - {_paths_file()}
        if gone:
            _rebuild(gone, cfg_)
        if _paths_file() in missing:
            run_trajectories(cfg_)
    rec.ensure(expected(), build)


def _rebuild(gone, cfg):
    budget, warm = cal.budget_of(cfg), False
    for tag, name in ALLOCATIONS.items():
        root = rec.path(name)
        todo = {}
        for seed in SEEDS:
            for _, arm, width, unit in ROWS:
                if _record(tag, seed, arm, width) in gone:
                    todo.setdefault(seed, {}).setdefault((unit, width), []).append(arm)
        if not todo:
            continue
        pb = cal.problem(tag, cfg)
        if not warm:
            warm = cuda_warm_up(pb.metric, pb.X0, pb.X1, cfg) or True
        for seed, units in todo.items():
            bb = cal.backbone(pb, seed, cfg, root) if any(u.startswith("ffm") for u, _ in units) else None
            for (unit, width), arms in units.items():
                t0 = time.time()
                out = cal.run_unit(pb, unit, width, seed, bb, cfg)
                secs = time.time() - t0
                for arm in arms:
                    phat, extra = out[arm]
                    cal._dump(cal.make_record(pb, phat, seed, arm, width, {**extra, "seconds": secs}, budget),
                              _record(tag, seed, arm, width))


def _sde_paths(pb, v_net, s_net, sigma, salt, cols, cfg):
    rep = torch.arange(len(pb.start), device=cfg.device).repeat_interleave(cal.K)
    base, keep = zlib.crc32(salt.encode()), {}
    for c in sorted({int(i) * cal.K // cal.CHUNK for i in cols}):
        a = c * cal.CHUNK
        traj = integrate_sde(v_net, s_net, pb.metric, pb.start[rep[a:a + cal.CHUNK]], sigma,
                             n_steps=cfg.n_steps, seed=base + c, t_eps=cfg.sb_t_eps)
        for i in cols:
            if int(i) * cal.K // cal.CHUNK == c:
                keep[int(i)] = cal._np(traj[:, int(i) * cal.K - a])
    return np.stack([keep[int(i)] for i in cols], axis=1)


def run_trajectories(cfg=None):
    """Re-run the seed-0, 50/50 Table 9 units; save the Figure 5(b) paths and return their recall."""
    cfg = Run.from_env() if cfg is None else cfg
    pb = cal.problem(FIG_TAG, cfg)
    cuda_warm_up(pb.metric, pb.X0, pb.X1, cfg)
    bb = cal.backbone(pb, FIG_SEED, cfg, rec.path(ALLOCATIONS[FIG_TAG]))
    cols = np.sort(np.random.default_rng(FIG_SEED).choice(pb.n_src, N_PATHS, replace=False))
    paths, phat = {}, {}
    for unit in BASELINES:
        hp = HParams(**{**cfg.arm_kw, **pb.point[unit]})
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_arm(unit, pb.ts_base, hp, FIG_SEED, cfg.device, cfg.dtype, n_steps=cal.BASE_N_STEPS)
        traj = cal._np(integrate_flow(res.v_net, pb.start, n_steps=cal.BASE_N_STEPS))
        paths[unit], phat[unit] = traj[:, cols], cal.hard(pb, traj[-1])
    traj = cal._np(integrate(bb[1], pb.start, cfg))
    paths["ffm_det"], phat["ffm_det"] = traj[:, cols], cal.hard(pb, traj[-1])
    sigma = sigma_for(pb.metric, cal.TUNED_WIDTH)
    v_net, s_net = train_sbm(bb[0], pb.metric, Coupler(bb[2][0], bb[2][1], pi=bb[2][2], seed=FIG_SEED),
                             sigma, FIG_SEED, cfg)
    salt = f"{FIG_TAG}|{FIG_SEED}|ffm_CC|{cal.TUNED_WIDTH:g}"
    phat["ffm_CC"], _ = cal.sample(pb, v_net, s_net, pb.metric, sigma, salt, cfg)
    paths["ffm_CC"] = _sde_paths(pb, v_net, s_net, sigma, salt, cols, cfg)
    os.makedirs(os.path.dirname(_paths_file()), exist_ok=True)
    np.savez_compressed(_paths_file(), cols=cols, **{a: p.astype(np.float32) for a, p in paths.items()})
    res_mask = pb.masks["res"]
    return {a: cal.recall(p[res_mask], pb.y[res_mask]) for a, p in phat.items()}


def _recall(tag, arm, width, side):
    values = []
    for s in SEEDS:
        with open(_record(tag, s, arm, width)) as fh:
            values.append(json.load(fh)["recall"][side][0])
    return values


def table9():
    """Table 9: lower- and upper-arm recall (%) under five nominal lower/upper allocations."""
    rows = [{"Method": label, "Arm": side,
             **{cal.SHARE_LABEL[t]: rec.pm(_recall(t, arm, width, side), 1) for t in ALLOCATIONS}}
            for label, arm, width, _ in ROWS for side in FATE_NAMES.values()]
    return rec.frame(rows, index=["Method", "Arm"])


def _primitives():
    from scripts import figures as nbfig

    from .figures import fork_primitives
    return nbfig, fork_primitives()


def figure5a():
    """Figure 5(a): the ForkSheet cloud with the antisymmetric flux of the training kernel."""
    import matplotlib.pyplot as plt
    nbfig, pf = _primitives()
    cfg = Run()
    ds = make_dataset("ForkSheet", shares=cal.SHARES[FIG_TAG])
    ts = split_training_set(ds, make_split(ds, seed=cfg.split_seed))
    X, P = np.asarray(ts.X, dtype=np.float64), sp.csr_matrix(ts.P)
    corr = corridor_mask(ds)
    lo, hi = pf._limits(ds)
    fig = plt.figure(figsize=(5.0, 3.4), layout="constrained")
    ax3 = fig.add_subplot(111, projection="3d")
    pf._surface(ax3, ds, alpha=0.5, lw=0.35)
    pf._scatter_cloud(ax3, ds, corr, three_d=True, s=1.5, alpha_scale=1.35)
    pasym = (P - P.T) * 0.5
    J = np.asarray(pasym @ X) - np.asarray(pasym.sum(axis=1)) * X
    g = nbfig.bin_flux(ds.uv[ts.idx], J[:, :2], nu=15, nw=13, min_count=6)
    zu, zw = pf._fork_dz(g[:, 0], g[:, 1])
    vec = np.c_[g[:, 2], g[:, 3], zu * g[:, 2] + zw * g[:, 3]]
    vec *= 0.10 / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    xyz = np.c_[g[:, :2], pf._fork_z(g[:, 0], g[:, 1])] - vec / 2
    ax3.quiver(*xyz.T, *vec.T, color="#3a3a3a", linewidth=0.75, arrow_length_ratio=0.4, alpha=0.9, zorder=3)
    pf._style_3d(ax3, lo, hi, zoom=1.4)
    ax3.set_title(r"$\mathcal{M} \subset \mathbb{R}^3$", fontsize=cfg.fs_heading + 1, pad=-4)
    fig.legend(handles=pf._legend_handles(ds, corr, extra_paths=False), loc="outside lower center", ncol=3,
               frameon=False, fontsize=cfg.fs_legend + 1, handletextpad=0.3, columnspacing=1.6)
    stem = pathlib.Path(rec.figure_path("forksheet_dataset"))
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.show()


def figure5b():
    """Figure 5(b): seed-0 learned trajectories of five source cells at the 50/50 allocation."""
    import matplotlib.pyplot as plt
    nbfig, pf = _primitives()
    cfg = Run()
    ds = make_dataset("ForkSheet", shares=cal.SHARES[FIG_TAG])
    corr = corridor_mask(ds)
    lo, hi = pf._limits(ds)
    ratio = float((hi[1] - lo[1]) / (hi[0] - lo[0])) + 0.14
    tr = np.load(_paths_file())
    bg = nbfig.BlockGrid([(h, [(FIG_LABELS[a], tr[a]) for a in arms]) for h, arms in BLOCKS],
                         row_ratios=(0.84, ratio), height=3.05, heading_fs=cfg.fs_heading)
    for bi, (_h, cols) in enumerate(bg.blocks):
        for k, (label, paths) in enumerate(cols):
            ax3 = bg.ax(1, bi, k, projection="3d")
            pf._surface(ax3, ds, alpha=0.5, lw=0.35)
            pf._scatter_cloud(ax3, ds, corr, three_d=True, s=1.5, alpha_scale=1.35)
            pf._draw_paths(ax3, paths, three_d=True)
            pf._style_3d(ax3, lo, hi, zoom=pf.PANEL_ZOOM)
            ax3.set_title(label, fontsize=cfg.fs_heading, linespacing=1.1,
                          pad=-6 - cfg.fs_heading * label.count("\n"))
            ax2 = bg.ax(2, bi, k)
            pf._scatter_cloud(ax2, ds, corr, three_d=False, s=1.4)
            pf._draw_paths(ax2, paths, three_d=False)
            pf._style_2d(ax2, lo, hi)
    bg.legend(list(pf._legend_handles(ds, corr, extra_paths=True)), fontsize=cfg.fs_legend)
    bg.save(pathlib.Path(rec.figure_path("forksheet_paths")))
    plt.show()
