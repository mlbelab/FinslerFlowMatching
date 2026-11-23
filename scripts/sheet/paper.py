import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

from ffm.core.arms import HParams, run_arm
from ffm.core.metrics import mean_floor
from ffm.core.present import QUAL_DET, QUAL_NOISY, two_line
from ffm.method import (FWFactory, Run, TrueFW, c_eff_of, cuda_warm_up, finsler_w2, moments_from, path_a,
                        path_b, sigma_for)
from scripts import records as rec

from .datasets import make_dataset
from .evaluate import evaluate_route
from .protocol import corridor_mask, pooled_slices
from .split import make_split, split_route_eval, split_training_set

NAME = "sheet_paper"
SEEDS = (0, 1, 2)
BASE_N_STEPS = 102
N_COLS = 5
BASELINES = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly": "Curly-FM"}
MAIN = ("cfm", "mfm_land", "curly", "ffm", "pathb")
ABLATION = ("eucl_off", "eucl_on", "riem", "ffm")
ABL_LAM_OFF = 1000.0
LABELS = {"cfm": "OT-CFM", "mfm_land": "MFM (LAND)", "curly": "Curly-FM", "ffm": "FFM (ours)",
          "pathb": "FFM (ours), with noise"}
ABL_ROWS = {"eucl_off": ("I", "no"), "eucl_on": ("I", "yes"), "riem": ("C_ρ⁻¹", "no"), "ffm": ("C_ρ⁻¹", "yes")}
HAMILTONIAN = {"ffm": "FFM (ours)", "wlf_finsler": "Finsler-WLF", "gsbm_finsler": "Finsler-GSBM",
               "wlf_euclid": "Euclidean WLF", "gsbm_euclid": "Euclidean GSBM"}
TUNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


class EuclideanFW(TrueFW):
    def mobility(self, x):
        return self.I[None].expand(x.shape[0], self.d, self.d)

    def _fields(self, x):
        b = self._smooth(self._kernel(x), self.b0_pts) + self.eps_b
        return self.I[None].expand(x.shape[0], self.d, self.d), b


def record_path(arm, seed):
    return rec.path(NAME, "records", f"{arm}_s{seed}.json")


def expected():
    arms = sorted(set(MAIN) | set(ABLATION))
    return [record_path(a, s) for a in arms for s in SEEDS] + [rec.path(NAME, "floors.json"),
                                                              rec.path(NAME, "trajectories.npz")]


def solver_paths():
    return {rec.path("matched_bench", "hamiltonian", f"sheet_{m}_s{s}.json"): (m, s)
            for m in HAMILTONIAN for s in range(5)}


def problem(cfg):
    ds = make_dataset("Sheet")
    split = make_split(ds, seed=cfg.split_seed)
    ts = split_training_set(ds, split)
    rev = split_route_eval(ds, split, "test")
    X, P = np.asarray(ts.X, dtype=np.float64), sp.csr_matrix(ts.P)
    X_t = cfg.tensor(X)
    X0_t = X_t[torch.as_tensor(ts.p0, device=cfg.device)]
    X1_t = X_t[torch.as_tensor(ts.p1, device=cfg.device)]
    b0_pts, D_pts = moments_from(P, X)
    with open(TUNED_FILE) as fh:
        tuned = json.load(fh)
    fw = FWFactory(X, D_pts, b0_pts, cfg, rho_mult=tuned["ffm"]["rho_mult"], lam_mult=tuned["ffm"]["lam_mult"])
    return {"ds": ds, "split": split, "ts": ts, "rev": rev, "X": X, "P": P, "X0_t": X0_t, "X1_t": X1_t,
            "b0": b0_pts, "D": D_pts, "fw": fw, "tuned": tuned}


def _metrics(out, pb, cfg):
    return evaluate_route(out["traj"], pb["ds"], pb["rev"], v_net=out["v_net"], device=cfg.device,
                          dtype=cfg.dtype, seed=0)


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(path + ".tmp", path)


def run(cfg=None):
    cfg = Run.from_env() if cfg is None else cfg
    pb = problem(cfg)
    fw, tuned, X0_t, X1_t = pb["fw"], pb["tuned"], pb["X0_t"], pb["X1_t"]
    from . import tune as sheet_tune
    rho_rungs = fw.rho_rungs(sheet_tune.RHO_MULTS)
    lam_rungs = fw.lam_rungs(sheet_tune.LAM_MULTS)
    cuda_warm_up(fw.at(rho=rho_rungs[0], lam=lam_rungs[0]), X0_t, X1_t, cfg)
    metric_star = fw.at(rho=fw.rho_rungs([tuned["ffm"]["rho_mult"]])[0],
                        lam=fw.lam_rungs([tuned["ffm"]["lam_mult"]])[0])
    runs = {}
    for seed in SEEDS:
        t0 = time.time()
        out = path_a(metric_star, X0_t, X1_t, seed, cfg)
        out["metrics"], out["seconds"] = _metrics(out, pb, cfg), time.time() - t0
        runs[("ffm", seed)] = out
    import dataclasses
    ts_base = dataclasses.replace(pb["ts"], velocity=pb["b0"])
    base_hp = cfg.arm_kw
    for arm in BASELINES:
        for seed in SEEDS:
            t0 = time.time()
            res = run_arm(arm, ts_base, HParams(**{**base_hp, **tuned[arm]}), seed, cfg.device, cfg.dtype,
                          n_steps=BASE_N_STEPS)
            out = {"v_net": res.v_net, "traj": np.asarray(res.traj, dtype=np.float64)}
            out["metrics"], out["seconds"] = _metrics(out, pb, cfg), time.time() - t0
            runs[(arm, seed)] = out
    sigma = sigma_for(metric_star, tuned["pathb"]["width_mult"])
    for seed in SEEDS:
        t0 = time.time()
        ref = runs[("ffm", seed)]
        out = path_b(metric_star, ref["phi"], ref["coupler"], X0_t, sigma, seed, cfg)
        out["metrics"], out["seconds"] = _metrics(out, pb, cfg), time.time() - t0
        runs[("pathb", seed)] = out
    rev = pb["rev"]
    judge = (runs[("ffm", SEEDS[0])]["phi"], metric_star)
    corr_ref = np.asarray(rev.corridor_ref, dtype=np.float64)
    n_pool = len(pooled_slices(runs[("ffm", SEEDS[0])]["traj"], rev.t_lo, rev.t_hi))
    wf_floor = mean_floor(corr_ref, n_pool, lambda a, b, k: {"WF": finsler_w2(a, b, judge, cfg, seed=k)})
    wf = {}
    for arm in ("ffm", "pathb", *BASELINES):
        for seed in SEEDS:
            cloud = pooled_slices(runs[(arm, seed)]["traj"], rev.t_lo, rev.t_hi)
            wf[(arm, seed)] = finsler_w2(cloud, corr_ref, judge, cfg, seed=seed)
    abl_rho = fw.rho_rungs([tuned["ffm"]["rho_mult"]])[0]
    abl_metric = {
        "eucl_off": EuclideanFW(pb["X"], pb["D"], pb["b0"], cfg, rho=abl_rho, lam=None,
                                lam_mult=ABL_LAM_OFF).calibrate(),
        "eucl_on": EuclideanFW(pb["X"], pb["D"], pb["b0"], cfg, rho=abl_rho, lam=None,
                               lam_mult=float(tuned["ffm"]["lam_mult"])).calibrate(),
        "riem": TrueFW(pb["X"], pb["D"], pb["b0"], cfg, rho=abl_rho, lam=None, lam_mult=ABL_LAM_OFF).calibrate(),
        "ffm": metric_star,
    }
    for arm in ("eucl_off", "eucl_on", "riem"):
        for seed in SEEDS:
            t0 = time.time()
            out = path_a(abl_metric[arm], X0_t, X1_t, seed, cfg)
            out["metrics"], out["seconds"] = _metrics(out, pb, cfg), time.time() - t0
            runs[(arm, seed)] = out
    abl_floor = mean_floor(corr_ref, n_pool, lambda a, b, k: {"WF": finsler_w2(a, b, judge, cfg, seed=k)})
    for arm in ("eucl_off", "eucl_on", "riem"):
        for seed in SEEDS:
            cloud = pooled_slices(runs[(arm, seed)]["traj"], rev.t_lo, rev.t_hi)
            wf[(arm, seed)] = finsler_w2(cloud, corr_ref, judge, cfg, seed=seed)
    for (arm, seed), out in runs.items():
        m = out["metrics"]
        _write(record_path(arm, seed), {
            "arm": arm, "seed": seed, "mid_W2": m["intermediate"]["W2"], "end_W2": m["endpoint"]["W2"],
            "W2F": wf[(arm, seed)], "c_eff": c_eff_of(abl_metric[arm]) if arm in abl_metric else None,
            "seconds": out["seconds"], "window": [rev.t_lo, rev.t_hi], "device": str(cfg.device),
            "gpu": torch.cuda.get_device_name() if cfg.device.type == "cuda" else None})
    f0 = runs[("ffm", SEEDS[0])]["metrics"]["floors"]
    _write(rec.path(NAME, "floors.json"), {"mid_W2": f0["intermediate"]["W2"], "end_W2": f0["endpoint"]["W2"],
                                           "W2F": wf_floor["WF"], "W2F_ablation": abl_floor["WF"]})
    from scripts.figures import sheet_primitives
    pf = sheet_primitives()
    n_src = min(runs[(a, SEEDS[0])]["traj"].shape[1] for a in set(MAIN) | set(ABLATION))
    cols = pf._lateral_starts(runs[("ffm", SEEDS[0])]["traj"][0, :n_src], N_COLS)
    arrays = {f"{a}": runs[(a, SEEDS[0])]["traj"][:, cols] for a in set(MAIN) | set(ABLATION)}
    os.makedirs(rec.path(NAME), exist_ok=True)
    np.savez_compressed(rec.path(NAME, "trajectories.npz"), cols=np.asarray(cols), **arrays)


def ensure(cfg=None):
    rec.ensure(expected(), lambda missing: run(cfg))
    rec.ensure(list(solver_paths()), lambda missing: _run_solvers(missing, cfg))


def _run_solvers(missing, cfg):
    from scripts.matched import hamiltonian
    cfg = Run.from_env() if cfg is None else cfg
    for p in missing:
        m, s = solver_paths()[p]
        hamiltonian.run("sheet", m, s, cfg, hamiltonian.root_dir(cfg.smoke))


def _cell(arm, key, digits=4):
    return rec.pm([rec.load_one(NAME, "records", f"{arm}_s{s}.json")[key] for s in SEEDS], digits)


def mid_end():
    fl = rec.load_one(NAME, "floors.json")
    rows = [{"Method": LABELS[a], "mid W2": _cell(a, "mid_W2"), "end W2": _cell(a, "end_W2"),
             "mid W2,F": _cell(a, "W2F")} for a in MAIN]
    rows.append({"Method": "sampling floor", "mid W2": f"{fl['mid_W2']:.4f}", "end W2": f"{fl['end_W2']:.4f}",
                 "mid W2,F": f"{fl['W2F']:.4f}"})
    return rec.frame(rows, index="Method")


def ablation():
    fl = rec.load_one(NAME, "floors.json")
    rows = []
    for arm in ABLATION:
        g, drift = ABL_ROWS[arm]
        c = rec.load_one(NAME, "records", f"{arm}_s{SEEDS[0]}.json")["c_eff"]
        rows.append({"G": g, "Drift": drift, "c_eff": f"{c:.3f}", "mid W2": _cell(arm, "mid_W2"),
                     "end W2": _cell(arm, "end_W2"), "W2,F": _cell(arm, "W2F")})
    rows.append({"G": "sampling floor", "Drift": "", "c_eff": "", "mid W2": f"{fl['mid_W2']:.4f}",
                 "end W2": f"{fl['end_W2']:.4f}", "W2,F": f"{fl['W2F_ablation']:.4f}"})
    return rec.frame(rows, index="G")


def solvers():
    rows = []
    for m, label in HAMILTONIAN.items():
        rs = rec.load("matched_bench", f"hamiltonian/sheet_{m}_s*.json")
        rows.append({"Method": label, "Mid W2": rec.pm([r["W2_mid"] for r in rs], 3),
                     "End W2": rec.pm([r["W2_end"] for r in rs], 3),
                     "Time (s)": f"{rec.mean([r['seconds'] for r in rs]):.0f}"})
    return rec.frame(rows, index="Method")


def _figure_setup():
    import matplotlib
    backend = matplotlib.get_backend()
    from scripts import figures as nbfig
    pf = nbfig.sheet_primitives()
    matplotlib.use(backend)
    return nbfig, pf


def _paths_grid(blocks, labels, stem):
    import pathlib

    import matplotlib.pyplot as plt
    nbfig, pf = _figure_setup()
    cfg = Run()
    ds = make_dataset("Sheet")
    corr = corridor_mask(ds)
    lo, hi = pf._limits(ds)
    ratio = float((hi[1] - lo[1]) / (hi[0] - lo[0])) + 0.14
    tr = np.load(rec.path(NAME, "trajectories.npz"))
    bg = nbfig.BlockGrid([(h, [(labels[a], tr[a]) for a in arms]) for h, arms in blocks],
                         row_ratios=(0.84, ratio), height=3.05, heading_fs=cfg.fs_heading)
    for bi, (_h, cols) in enumerate(bg.blocks):
        for k, (label, paths) in enumerate(cols):
            ax3 = bg.ax(1, bi, k, projection="3d")
            pf._surface(ax3, alpha=0.5, lw=0.35)
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
    bg.save(pathlib.Path(rec.figure_path(stem)))
    plt.show()


def figure_paths():
    blocks = [(r"Deterministic ($\sigma = 0$)", ["cfm", "mfm_land", "curly", "ffm"]),
              (r"Stochastic ($\sigma > 0$)", ["pathb"])]
    labels = {**BASELINES, "ffm": two_line(QUAL_DET, ours=True), "pathb": two_line(QUAL_NOISY, ours=True)}
    _paths_grid(blocks, labels, "sheet_paths")


def figure_ablation():
    labels = {"eucl_off": "$G = I$\n$\\beta = 0$", "eucl_on": "$G = I$\n$\\beta \\neq 0$",
              "riem": "$G = a^2 C_\\rho^{-1}$\n$\\beta = 0$", "ffm": "$G = a^2 C_\\rho^{-1}$\n$\\beta \\neq 0$"}
    _paths_grid([("Metric ablation", list(ABLATION))], labels, "sheet_metric_ablation")


def figure_dataset():
    import pathlib

    import matplotlib.pyplot as plt
    nbfig, pf = _figure_setup()
    cfg = Run()
    ds = make_dataset("Sheet")
    split = make_split(ds, seed=cfg.split_seed)
    ts = split_training_set(ds, split)
    X, P = np.asarray(ts.X, dtype=np.float64), sp.csr_matrix(ts.P)
    u, w = ds.uv[ts.idx, 0], ds.uv[ts.idx, 1]
    corr = corridor_mask(ds)
    lo, hi = pf._limits(ds)
    fig = plt.figure(figsize=(5.0, 3.4), layout="constrained")
    ax3 = fig.add_subplot(111, projection="3d")
    U, W = np.meshgrid(np.linspace(-pf.SHEET_L, pf.SHEET_L, 25), np.linspace(-pf.SHEET_W, pf.SHEET_W, 9))
    ax3.plot_wireframe(U, W, pf._sheet_z(U, W), color=nbfig.FAINT, linewidth=0.35, alpha=0.5, zorder=0)
    pf._scatter_cloud(ax3, ds, corr, three_d=True, s=1.5, alpha_scale=1.35)
    a_len, a_rad = 0.09, 0.16
    pasym = (P - P.T) * 0.5
    J = np.asarray(pasym @ X) - np.asarray(pasym.sum(axis=1)) * X
    pts = np.array([[uu, ww] for ww in (-0.36, -0.21, 0.21, 0.36) for uu in np.linspace(-1.15, 1.15, 8)])
    vec = np.asarray([J[np.linalg.norm(np.c_[u, w] - p, axis=1) < a_rad].mean(0) for p in pts])
    h = 1e-3
    zu = (pf._sheet_z(pts[:, 0] + h, pts[:, 1]) - pf._sheet_z(pts[:, 0] - h, pts[:, 1])) / (2 * h)
    zw = (pf._sheet_z(pts[:, 0], pts[:, 1] + h) - pf._sheet_z(pts[:, 0], pts[:, 1] - h)) / (2 * h)
    vec = np.c_[vec[:, 0], vec[:, 1], zu * vec[:, 0] + zw * vec[:, 1]]
    vec *= a_len / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    xyz = np.c_[pts, pf._sheet_z(pts[:, 0], pts[:, 1])] - vec / 2
    ax3.quiver(*xyz.T, *vec.T, color="#3a3a3a", linewidth=0.75, arrow_length_ratio=0.4, alpha=0.9, zorder=3)
    pf._style_3d(ax3, lo, hi, zoom=2.6)
    ax3.view_init(elev=26.0, azim=-40.0)
    ax3.set_title(r"$\mathcal{M} \subset \mathbb{R}^3$", fontsize=cfg.fs_heading + 1, pad=-4)
    fig.legend(handles=pf._legend_handles(ds, corr, extra_paths=False), loc="outside lower center", ncol=3,
               frameon=False, fontsize=cfg.fs_legend + 1, handletextpad=0.3, columnspacing=1.6)
    stem = pathlib.Path(rec.figure_path("sheet_dataset"))
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.show()
