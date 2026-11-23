from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from ffm.core.arms import HParams, run_arm
from ffm.core.bridge import integrate_sde
from ffm.core.itracer_data import BETA
from ffm.core.metrics import traj_slice, wasserstein
from ffm.core.paths import PUBLIC_ROOT, experiment_root
from ffm.core.protocol import TrainingSet
from ffm.core.transition import asymmetry_scale, build_uniform_knn_transition, build_velocity_kernel
from ffm.method import Coupler, FWFactory, Run, moments_from, path_a, sigma_for, train_sbm
from scripts.pancreas.datasets import _flux_alignment

from .data import HELD, SOURCE, TARGET, load

NAME = "larry_bench"
DIMS = (20, 50)
SEEDS = (0, 1, 2, 3, 4)
KERNELS = ("velocity", "expression", "lineage")
UNITS = ("cfm", "mfm_land", "curly", "ffm_velocity", "ffm_expression", "ffm_lineage")
METHODS = ("cfm", "mfm_land", "curly", "ffm", "ffm_noise", "ffm_expr", "ffm_expr_noise", "ffm_lin")
LABELS = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly": "Curly-FM", "ffm": "FFM, velocity kernel",
          "ffm_noise": "FFM with noise, velocity kernel", "ffm_expr": "FFM, expression kernel",
          "ffm_expr_noise": "FFM with noise, expression kernel",
          "ffm_lin": "FFM, velocity + lineage kernel",
          "ffm_pp": "FFM, velocity kernel, Pancreas point (sensitivity)"}
UNIT_KERNEL = {"ffm_velocity": "velocity", "ffm_expression": "expression", "ffm_lineage": "lineage",
               "ffm_pancreas": "velocity"}
UNIT_METHOD = {"ffm_velocity": "ffm", "ffm_expression": "ffm_expr", "ffm_lineage": "ffm_lin",
               "ffm_pancreas": "ffm_pp"}
N_EVAL = 2000
EVAL_REPS = 5
KNN = 15
OT_MAX_PTS = 600
BASE_N_STEPS = 102
DAY_COLORS = {SOURCE: "#2a78d6", HELD: "#eb6834", TARGET: "#1baf7a"}
DAY_ROLE = {SOURCE: "source", HELD: "withheld", TARGET: "target"}
SOURCES = {"pancreas": ("pancreas", "ffm"), "erythroid": ("erythroid", "ffm"),
           "itracer": ("itracer", "ffm_velo")}


def root_dir(smoke: bool = False) -> str:
    return experiment_root(NAME, smoke)


def rec_path(root: str, d: int, method: str, seed: int) -> str:
    return os.path.join(root, "records", f"d{d}_{method}_s{seed}.json")


def _logmedian(v) -> float:
    return float(np.exp(np.median(np.log(np.asarray(v, dtype=np.float64)))))


def transfer_points(d: int, sources: dict = SOURCES) -> dict:
    tuned = {}
    for name, (pkg, ffm_key) in sources.items():
        with open(os.path.join(PUBLIC_ROOT, "scripts", pkg, "tuned.json")) as fh:
            t = json.load(fh)[str(d)]
        tuned[name] = {"ffm": t[ffm_key], "pathb": t.get("pathb"), "mfm_land": t["mfm_land"],
                       "curly": t["curly"], "cfm": {"blur_frac": t["cfm"].get("blur_frac",
                                                                           HParams.blur_frac)}}
    pick = lambda arm, knob: _logmedian([v[arm][knob] for v in tuned.values() if v[arm]])
    return {"ffm": {"rho_mult": pick("ffm", "rho_mult"), "lam_mult": pick("ffm", "lam_mult")},
            "pathb": {"width_mult": pick("pathb", "width_mult")},
            "mfm_land": {"land_rho": pick("mfm_land", "land_rho")},
            "curly": {"curly_alpha": pick("curly", "curly_alpha")},
            "cfm": {"blur_frac": pick("cfm", "blur_frac")}, "sources": tuned}


def lineage_reweight(P: sp.csr_matrix, clone: np.ndarray, beta: float = BETA) -> sp.csr_matrix:
    P = sp.csr_matrix(P, dtype=np.float64)
    diag = P.diagonal()
    off = (P - sp.diags(diag)).tocsr()
    off.eliminate_zeros()
    rows = np.repeat(np.arange(P.shape[0]), np.diff(off.indptr))
    same = (clone[rows] == clone[off.indices]) & (clone[rows] >= 0)
    off.data = off.data * np.where(same, np.exp(beta), 1.0)
    s = np.asarray(off.sum(1)).ravel()
    off = sp.diags((1.0 - diag) / np.maximum(s, 1e-300)) @ off
    return (off + sp.diags(diag)).tocsr()


def cloud(d: int) -> dict:
    z = load()
    return {"X": z["X"][:, :d].astype(np.float64), "V": z["velocity"][:, :d].astype(np.float64),
            "day": z["day"], "state": z["state"], "clone": z["clone"], "z": z}


def training_set(c: dict, kernel: str) -> TrainingSet:
    idx = np.flatnonzero(np.isin(c["day"], (SOURCE, TARGET)))
    X, V = c["X"][idx], c["V"][idx]
    if kernel == "expression":
        P = build_uniform_knn_transition(X)
    else:
        P = build_velocity_kernel(X, V)
        if kernel == "lineage":
            P = lineage_reweight(P, c["clone"][idx])
    day = c["day"][idx]
    return TrainingSet(name=f"LARRY-d{X.shape[1]}-{kernel}", idx=idx, X=X, P=sp.csr_matrix(P),
                       p0=day == SOURCE, p1=day == TARGET, rho=0.0, holdout_band=None,
                       p_variant=kernel, diag={"n_train": int(len(idx))}, velocity=V)


def save_kernel(ts: TrainingSet, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    P = sp.csr_matrix(ts.P)
    np.savez_compressed(path, data=P.data.astype(np.float32), indices=P.indices,
                        indptr=P.indptr, shape=np.asarray(P.shape), idx=ts.idx)


def diagnostics(d: int, root: str) -> dict:
    c = cloud(d)
    out = {}
    for kernel in KERNELS:
        ts = training_set(c, kernel)
        save_kernel(ts, os.path.join(root, "kernels", f"{kernel}_d{d}.npz"))
        X, P = ts.X, ts.P
        day = c["day"][ts.idx]
        m = np.asarray(P @ X) - X
        axis = X[ts.p1].mean(0) - X[ts.p0].mean(0)
        off = (P - sp.diags(P.diagonal())).tocsr()
        tot = np.asarray(off.sum(1)).ravel()
        to6 = np.asarray(off[:, np.flatnonzero(day == TARGET)].sum(1)).ravel()
        to2 = np.asarray(off[:, np.flatnonzero(day == SOURCE)].sum(1)).ravel()
        cm = (m @ axis) / np.maximum(np.linalg.norm(m, axis=1) * np.linalg.norm(axis), 1e-300)
        out[kernel] = {"P_asym": float(asymmetry_scale(P)),
                       "cos_drift_axis_day2": float(cm[ts.p0].mean()),
                       "cos_drift_axis_day6": float(cm[ts.p1].mean()),
                       "mass_day2_to_day6": float((to6[ts.p0] / tot[ts.p0]).mean()),
                       "mass_day6_to_day2": float((to2[ts.p1] / tot[ts.p1]).mean()),
                       "cos_flux_velocity": float(_flux_alignment(P, X, ts.velocity))}
    ts = training_set(c, "velocity")
    X, V = ts.X, ts.velocity
    axis = X[ts.p1].mean(0) - X[ts.p0].mean(0)
    cv = (V @ axis) / np.maximum(np.linalg.norm(V, axis=1) * np.linalg.norm(axis), 1e-300)
    toward = X[ts.p1].mean(0)[None] - X[ts.p0]
    ct = np.sum(V[ts.p0] * toward, 1) / np.maximum(
        np.linalg.norm(V[ts.p0], axis=1) * np.linalg.norm(toward, axis=1), 1e-300)
    lt = c["z"]["latent_time"][ts.idx]
    a, b = lt[ts.p0], lt[ts.p1]
    auroc = float((np.mean(b[:, None] > a[None, :]) + 0.5 * np.mean(b[:, None] == a[None, :])))
    out["velocity"].update({
        "cos_velocity_axis_day2": float(cv[ts.p0].mean()),
        "cos_velocity_axis_day6": float(cv[ts.p1].mean()),
        "frac_velocity_forward_day2": float((cv[ts.p0] > 0).mean()),
        "cos_velocity_to_day6_mean_day2": float(ct.mean()),
        "latent_time_mean_day2": float(np.nanmean(a)), "latent_time_mean_day6": float(np.nanmean(b)),
        "latent_time_auroc_day6_vs_day2": auroc,
        "velocity_confidence_mean": float(np.nanmean(c["z"]["velocity_confidence"][ts.idx]))})
    return out


def floor(c: dict) -> dict:
    H = c["X"][c["day"] == HELD]
    vals = []
    for k in range(EVAL_REPS):
        perm = np.random.default_rng(k).permutation(len(H))
        h = len(H) // 2
        vals.append(wasserstein(H[perm[:h]], H[perm[h:]], p=2, max_n=N_EVAL, seed=k))
    return {"W2_mean": float(np.mean(vals)), "W2_sd": float(np.std(vals)), "n_each": int(min(len(H) // 2, N_EVAL))}


def evaluate(c: dict, traj: np.ndarray, tree: cKDTree, labels: np.ndarray, names) -> dict:
    mid, end = traj_slice(traj, 0.5), traj_slice(traj, 1.0)
    H, T = c["X"][c["day"] == HELD], c["X"][c["day"] == TARGET]
    ok = np.isfinite(mid).all(1)
    w_mid = [wasserstein(mid[ok], H, p=2, max_n=N_EVAL, seed=k) for k in range(EVAL_REPS)]
    w_end = [wasserstein(end[np.isfinite(end).all(1)], T, p=2, max_n=N_EVAL, seed=k)
             for k in range(EVAL_REPS)]
    nn = tree.query(mid[ok], k=KNN)[1]
    votes = np.stack([(labels[nn] == j).sum(1) for j in range(len(names))], axis=1)
    gen = np.bincount(votes.argmax(1), minlength=len(names)) / len(nn)
    obs = np.bincount(labels, minlength=len(names)) / len(labels)
    return {"W2_mid": float(np.mean(w_mid)), "W2_mid_reps": w_mid, "W2_end": float(np.mean(w_end)),
            "W2_end_reps": w_end, "celltype_tv": float(0.5 * np.abs(gen - obs).sum()),
            "celltype_generated": dict(zip(names, gen.tolist())),
            "celltype_observed": dict(zip(names, obs.tolist())), "nonfinite": float(1 - ok.mean())}


def time_distances(d: int) -> dict:
    c = cloud(d)
    X = {day: c["X"][c["day"] == day] for day in (SOURCE, HELD, TARGET)}
    rows = {}
    for day in (SOURCE, HELD, TARGET):
        vals = []
        for k in range(EVAL_REPS):
            perm = np.random.default_rng(k).permutation(len(X[day]))
            h = len(X[day]) // 2
            vals.append(wasserstein(X[day][perm[:h]], X[day][perm[h:]], p=2, max_n=N_EVAL, seed=k))
        rows[f"within Day {int(day)} (two halves)"] = vals
    for a, b in ((SOURCE, HELD), (HELD, TARGET), (SOURCE, TARGET)):
        rows[f"Day {int(a)} vs Day {int(b)}"] = [
            wasserstein(X[a], X[b], p=2, max_n=N_EVAL, seed=k) for k in range(EVAL_REPS)]
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "reps": v} for k, v in rows.items()}


def compute_umap(root: str) -> np.ndarray:
    import umap
    c = cloud(50)
    Y = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(c["X"])
    np.savez_compressed(os.path.join(root, "umap.npz"), Y=Y.astype(np.float32), day=c["day"])
    return Y


def umap_figure(root: str | None = None, out: str | None = None):
    import matplotlib.pyplot as plt
    root = root_dir() if root is None else root
    with np.load(os.path.join(root, "umap.npz")) as z:
        Y, day = z["Y"], z["day"]
    ink, muted = "#1a1a19", "#d6d5d0"
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7), constrained_layout=True)
    for t in (TARGET, SOURCE, HELD):
        m = day == t
        axes[0].scatter(Y[m, 0], Y[m, 1], s=1.5, c=DAY_COLORS[t], alpha=0.5, linewidths=0,
                        rasterized=True, label=f"Day {int(t)} ({DAY_ROLE[t]})")
    hs, ls = axes[0].get_legend_handles_labels()
    order = [ls.index(f"Day {int(t)} ({DAY_ROLE[t]})") for t in (SOURCE, HELD, TARGET)]
    leg = axes[0].legend([hs[i] for i in order], [ls[i] for i in order], markerscale=6,
                         frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    axes[0].set_title("All cells", fontsize=10, color=ink)
    for ax, t in zip(axes[1:], (SOURCE, HELD, TARGET)):
        m = day == t
        ax.scatter(Y[:, 0], Y[:, 1], s=1, c=muted, linewidths=0, rasterized=True)
        ax.scatter(Y[m, 0], Y[m, 1], s=1.5, c=DAY_COLORS[t], alpha=0.7, linewidths=0,
                   rasterized=True)
        ax.set_title(f"Day {int(t)}, {DAY_ROLE[t]} ({int(m.sum()):,} cells)", fontsize=10,
                     color=ink)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
    axes[0].set_xlabel("UMAP 1", fontsize=8, color=ink)
    axes[0].set_ylabel("UMAP 2", fontsize=8, color=ink)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    return fig


def umap_png(dpi: int = 100) -> bytes:
    import matplotlib.pyplot as plt
    fig = umap_figure()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _np(x) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float64)


def unit_methods(unit: str) -> tuple:
    return (("ffm", "ffm_noise") if unit == "ffm_velocity" else
            ("ffm_expr", "ffm_expr_noise") if unit == "ffm_expression" else
            (UNIT_METHOD[unit],) if unit in UNIT_METHOD else (unit,))


def train_unit(unit: str, ts: TrainingSet, pts: dict, seed: int, cfg: Run) -> dict:
    methods = unit_methods(unit)
    out = {}
    t0 = time.perf_counter()
    if unit in UNIT_KERNEL:
        X = np.asarray(ts.X, dtype=np.float64)
        b0, D = moments_from(ts.P, X)
        metric = FWFactory(X, D, b0, cfg, rho_mult=pts["ffm"]["rho_mult"],
                           lam_mult=pts["ffm"]["lam_mult"]).base
        Xt = cfg.tensor(X)
        X0 = Xt[torch.as_tensor(ts.p0, device=cfg.device)]
        X1 = Xt[torch.as_tensor(ts.p1, device=cfg.device)]
        det = path_a(metric, X0, X1, seed, cfg)
        out[methods[0]] = (_np(det["traj"]), {"point": pts["ffm"], "seconds": time.perf_counter() - t0})
        if len(methods) > 1:
            t1 = time.perf_counter()
            co = det["coupler"]
            sig = sigma_for(metric, pts["pathb"]["width_mult"])
            v_net, s_net = train_sbm(det["phi"], metric, Coupler(co.X0, co.X1, pi=co.pi, seed=seed),
                                     sig, seed, cfg)
            traj = integrate_sde(v_net, s_net, metric, X0, sig, n_steps=cfg.n_steps, seed=seed,
                                 t_eps=cfg.sb_t_eps)
            out[methods[1]] = (_np(traj), {"point": {**pts["ffm"], **pts["pathb"]}, "sigma": sig,
                                           "seconds": time.perf_counter() - t1
                                           + out[methods[0]][1]["seconds"]})
    else:
        knobs = pts[unit]
        hp = HParams(**{**cfg.arm_kw, **knobs, "ot_max_pts": OT_MAX_PTS})
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_arm(unit, ts, hp, seed, cfg.device, cfg.dtype, n_steps=BASE_N_STEPS)
        out[unit] = (_np(res.traj), {"point": knobs, "seconds": time.perf_counter() - t0})
    return out


def run_unit(d: int, unit: str, seed: int, cfg: Run, root: str, force: bool = False) -> None:
    if not force and all(os.path.exists(rec_path(root, d, m, seed)) for m in unit_methods(unit)):
        print(f"[larry] skip d{d} {unit} s{seed}")
        return
    c = cloud(d)
    pts = transfer_points(d)
    if unit == "ffm_pancreas":
        pts["ffm"] = dict(pts["sources"]["pancreas"]["ffm"])
    H = c["X"][c["day"] == HELD]
    names = sorted(np.unique(c["state"]))
    labels = np.array([names.index(s) for s in c["state"][c["day"] == HELD]])
    tree, fl = cKDTree(H), floor(c)
    kernel = UNIT_KERNEL.get(unit, "velocity")
    ts = training_set(c, kernel)
    out = train_unit(unit, ts, pts, seed, cfg)
    for method, (traj, extra) in out.items():
        rec = {"dim": d, "method": method, "seed": seed, "kernel": kernel,
               "n_source": int(ts.p0.sum()), "n_target": int(ts.p1.sum()), "n_held": int(len(H)),
               "floor": fl, **evaluate(c, traj, tree, labels, names), **extra}
        path = rec_path(root, d, method, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(path + ".tmp", path)
        print(f"[larry] d{d} {method:9s} s{seed}  mid W2 {rec['W2_mid']:.4f}  end W2 "
              f"{rec['W2_end']:.4f}  type TV {rec['celltype_tv']:.3f}  floor {fl['W2_mean']:.4f}  "
              f"({rec['seconds']:.0f}s)", flush=True)


def tasks() -> list[tuple[int, str, int]]:
    return ([(d, u, s) for d in DIMS for u in UNITS for s in SEEDS]
            + [(d, "ffm_pancreas", s) for d in DIMS for s in SEEDS])


def frames(root: str | None = None, smoke: bool = False) -> dict:
    root = root_dir(smoke) if root is None else root
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "records", "*.json")))]
    if not rows:
        return {}
    df = pd.DataFrame([{"d": r["dim"], "method": LABELS[r["method"]], "seed": r["seed"],
                        "W2_mid": r["W2_mid"], "W2_end": r["W2_end"], "celltype_tv": r["celltype_tv"],
                        "floor": r["floor"]["W2_mean"]} for r in rows])
    pm = lambda v: f"{np.mean(v):.4f} ± {np.std(v):.4f}"
    summary = df.groupby(["d", "method"])[["W2_mid", "W2_end", "celltype_tv"]].agg(pm)
    summary["n"] = df.groupby(["d", "method"]).size()
    floors = df.groupby("d")["floor"].first().rename("Day 4 floor W2").to_frame()
    types = pd.DataFrame({(r["dim"], LABELS[r["method"]]): r["celltype_generated"] for r in rows
                          if r["seed"] == 0}).T
    obs = pd.DataFrame({(r["dim"], "observed Day 4"): r["celltype_observed"] for r in rows
                        if r["seed"] == 0 and r["method"] == "cfm"}).T
    props = pd.concat([obs, types]).sort_index().round(3)
    diag_path = os.path.join(root, "diagnostics.json")
    diag = (pd.DataFrame({(int(d), k): v for d, kv in json.load(open(diag_path)).items()
                          for k, v in kv.items()}).T if os.path.exists(diag_path) else pd.DataFrame())
    sub = json.load(open(os.path.join(PUBLIC_ROOT, "data", "larry_cache", "subsample.json")))
    counts = pd.DataFrame({"requested": sub["requested"], "available": sub["available"],
                           "selected": sub["selected"]}).T
    dist_path = os.path.join(root, "distances.json")
    dist = (pd.DataFrame({f"d = {d}": {k: f"{v['mean']:.4f} ± {v['sd']:.4f}" for k, v in kv.items()}
                          for d, kv in json.load(open(dist_path)).items()})
            if os.path.exists(dist_path) else pd.DataFrame())
    return {"summary": summary, "floor": floors, "celltypes": props, "diagnostics": diag,
            "distances": dist,
            "counts": counts, "points": pd.DataFrame({d: {k: v for k, v in transfer_points(d).items()
                                                           if k != "sources"} for d in DIMS}).T}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="larry_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--force", action="store_true")
    sub.add_parser("diag")
    sub.add_parser("distances")
    sub.add_parser("umap")
    sub.add_parser("count")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    cfg = Run.from_env().but(ot_max_pts=OT_MAX_PTS)
    if a.cmd == "count":
        print(len(tasks()))
    elif a.cmd == "diag":
        out = {d: diagnostics(d, root_dir(cfg.smoke)) for d in DIMS}
        os.makedirs(root_dir(cfg.smoke), exist_ok=True)
        with open(os.path.join(root_dir(cfg.smoke), "diagnostics.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print(json.dumps(out, indent=1))
    elif a.cmd == "distances":
        out = {d: time_distances(d) for d in DIMS}
        with open(os.path.join(root_dir(cfg.smoke), "distances.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print(json.dumps({d: {k: round(v["mean"], 4) for k, v in kv.items()} for d, kv in out.items()},
                         indent=1))
    elif a.cmd == "umap":
        compute_umap(root_dir(cfg.smoke))
        umap_figure(root_dir(cfg.smoke), os.path.join(PUBLIC_ROOT, "notebooks", "out", "larry_umap"))
    elif a.cmd == "run":
        d, unit, seed = tasks()[a.index]
        run_unit(d, unit, seed, cfg, root_dir(cfg.smoke), a.force)
    else:
        f = frames(smoke=cfg.smoke)
        for k in ("counts", "points", "diagnostics", "floor", "summary", "celltypes"):
            print(f[k].to_string())


if __name__ == "__main__":
    main()
