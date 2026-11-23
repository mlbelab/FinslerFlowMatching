from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import io
import json
import os
import time
import zlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import binomtest

from ffm.core.arms import HParams, run_arm
from ffm.core.bridge import integrate_sde
from ffm.core.metrics import distribution_metrics, mean_floor
from ffm.core.paths import experiment_root
from ffm.core.scvelo_data import load_pancreas
from ffm.core.train import integrate_flow
from scripts.forksheet.calibration import (SHORT, _corr, _dump, _fresh, _np,
                                                       budget_of)
from ffm.method import (ConstantMobility, Coupler, FWFactory, GeoPathNet, Run,
                            VelocityNet, cuda_warm_up, integrate, moments_from, path_a,
                            sigma_for, train_sbm)

from .baselines import tuned_points
from .datasets import (CACHE_DIR, N_PCS, NUM_TIMES, VEL_WEIGHT, VELO_MODE,
                       build_training_set, load_cloud)

NAME = "pancreas_fate"
FATE_CACHE = os.path.join(os.path.dirname(CACHE_DIR), "fate")
DIMS = (20, 50)
SEEDS = (0, 1, 2, 3, 4)
FATES = ("Alpha", "Beta", "Delta", "Epsilon")
WIDTHS = (0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0, 3.0)
K = 64
CHUNK = 16384
BASE_N_STEPS = 102
KNN = 15
BASELINES = ("cfm", "mfm_land", "curly", "cdcfm")
UNIT_ARMS = {"ffm_det": ("ffm_det",), "ffm_C": ("ffm_CC", "ffm_CI", "ffm_Coff"),
             "ffm_I": ("ffm_IC", "ffm_II", "ffm_Ioff"), "sf2m": ("sf2m", "sf2m_off"),
             **{b: (b,) for b in BASELINES}}
STOCH_ARMS = ("ffm_CC", "ffm_CI", "ffm_IC", "ffm_II", "sf2m")
REFS = ("ref_mean", "ref_target")
LABELS = {**SHORT, "ref_mean": "constant: mean reference", "ref_target": "constant: target shares"}
COMPARE = (("ffm_CC", "ffm_Coff"), ("ffm_II", "ffm_Ioff"), ("ffm_CC", "ffm_II"),
           ("ffm_CC", "sf2m"), ("ffm_II", "sf2m"), ("sf2m", "sf2m_off"), ("ffm_CC", "ffm_det"))


def root_dir(smoke: bool = False) -> str:
    return experiment_root(NAME, smoke)


def record_path(root: str, dim: int, seed: int, arm: str, width: float) -> str:
    return os.path.join(root, "records", f"d{dim}_s{seed}_{arm}_w{width:g}.json")


def phat_path(record: str) -> str:
    head, name = os.path.split(record)
    return os.path.join(os.path.dirname(head), "phat", name[:-len(".json")] + ".npy")


def net_path(root: str, dim: int, seed: int) -> str:
    return os.path.join(root, "nets", f"d{dim}_s{seed}.pt")


def widths(dim: int) -> tuple:
    return tuple(sorted({float(tuned_points()[dim]["pathb"]["width_mult"]), *WIDTHS}))


def units(dim: int) -> list[tuple[str, float]]:
    out = [("ffm_det", 0.0)]
    for w in widths(dim):
        out += [("ffm_C", w), ("ffm_I", w), ("sf2m", w)]
    return out + [(b, 0.0) for b in BASELINES]


def reference() -> np.ndarray:
    _, meta = load_pancreas("", n_pcs=N_PCS, vel_weight=VEL_WEIGHT, mode=VELO_MODE,
                            cache_dir=CACHE_DIR)
    names = [str(f) for f in meta.fate_names]
    return np.asarray(meta.B, dtype=np.float64)[:, [names.index(f) for f in FATES]]


def spread(b: np.ndarray) -> float:
    return float(((b - b.mean(0)) ** 2).sum(1).mean())


def diversity() -> pd.DataFrame:
    B, cloud = reference(), load_cloud(DIMS[0])
    lt = cloud.latent_time
    dec = np.minimum((np.argsort(np.argsort(lt)) * 10) // len(lt), 9)
    parts = {**{f"tertile {b}": cloud.bin_id == b for b in range(NUM_TIMES)},
             **{f"decile {q}": dec == q for q in range(10)}}
    rows = {}
    for name, m in parts.items():
        b, cells = B[m], pd.Series(cloud.celltype[m]).value_counts()
        rows[name] = {"n": int(m.sum()), **dict(zip(FATES, b.mean(0).round(3))),
                      "spread": round(spread(b), 4),
                      "max>=0.8": round(float((b.max(1) >= 0.8).mean()), 3),
                      "main types": ", ".join(f"{k} {v}" for k, v in cells.head(2).items())}
    return pd.DataFrame(rows).T


def scaffold(cloud) -> SimpleNamespace:
    src = np.flatnonzero(cloud.bin_id == 1)
    lt = cloud.latent_time[src]
    late = lt >= np.median(lt)
    return SimpleNamespace(src=src, ref=reference()[src],
                           masks={"all": np.ones(len(src), dtype=bool), "late": late,
                                  "early": ~late})


def problem(dim: int, cfg: Run) -> SimpleNamespace:
    cloud = load_cloud(dim, FATE_CACHE)
    sc = scaffold(cloud)
    moved = dataclasses.replace(cloud, bin_id=np.array([1, 0, 2])[cloud.bin_id])
    ts = build_training_set(moved)
    X = np.asarray(ts.X, dtype=np.float64)
    point = tuned_points()[dim]
    b0, D = moments_from(ts.P, X)
    metric = FWFactory(X, D, b0, cfg, rho_mult=point["ffm"]["rho_mult"],
                       lam_mult=point["ffm"]["lam_mult"]).base
    src, tgt = ts.idx[ts.p0], ts.idx[ts.p1]
    assert np.array_equal(src, sc.src)
    fate_cells = tgt[np.isin(cloud.celltype[tgt], FATES)]
    lab = np.array([FATES.index(str(c)) for c in cloud.celltype[fate_cells]])
    tree = cKDTree(cloud.X[fate_cells])
    Xt = cfg.tensor(X)
    return SimpleNamespace(
        dim=dim, ts=ts, metric=metric, m_bar=float(metric.mean_mobility_eig()), point=point,
        X0=Xt[torch.as_tensor(ts.p0, device=cfg.device)],
        X1=Xt[torch.as_tensor(ts.p1, device=cfg.device)], start=cfg.tensor(cloud.X[src]),
        n_src=len(src), ref=sc.ref, tree=tree, lab=lab,
        far=float(np.quantile(tree.query(cloud.X[fate_cells], k=2)[0][:, 1], 0.99)),
        shares=np.bincount(lab, minlength=len(FATES)) / len(lab), masks=sc.masks,
        X_end=cloud.X[tgt], floor=mean_floor(cloud.X[tgt], len(src),
                                             lambda a, b, k: distribution_metrics(a, b, seed=k)))


def classify(pb, ends) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray(ends, dtype=np.float64)
    ok = np.isfinite(ends).all(axis=1)
    lab, far = np.full(len(ends), -1), np.zeros(len(ends))
    if ok.any():
        d, nn = pb.tree.query(ends[ok], k=KNN)
        votes = np.stack([(pb.lab[nn] == f).sum(axis=1) for f in range(len(FATES))], axis=1)
        lab[ok], far[ok] = votes.argmax(axis=1), d[:, 0] > pb.far
    return lab, far


def _normalise(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum(axis=1, keepdims=True)
    return np.where(tot > 0, counts / np.maximum(tot, 1.0), 1.0 / counts.shape[1])


def hard(pb, ends, seed: int) -> tuple[np.ndarray, dict]:
    lab, far = classify(pb, ends)
    counts = np.zeros((len(lab), len(FATES)))
    ok = lab >= 0
    counts[np.flatnonzero(ok), lab[ok]] = 1.0
    return _normalise(counts), {"outside": float(far.mean()), "nonfinite": float(1 - ok.mean()),
                                "endpoint": distribution_metrics(np.asarray(ends)[ok], pb.X_end,
                                                                 seed=seed)}


def sample(pb, v_net, s_net, geom, sigma: float, salt: str, cfg: Run, seed: int):
    rep = torch.arange(pb.n_src, device=cfg.device).repeat_interleave(K)
    base = zlib.crc32(salt.encode())
    counts, far, bad = np.zeros((pb.n_src, len(FATES))), 0.0, 0
    first = np.full((pb.n_src, pb.X_end.shape[1]), np.nan)
    for c, a in enumerate(range(0, len(rep), CHUNK)):
        j = rep[a:a + CHUNK]
        end = _np(integrate_sde(v_net, s_net, geom, pb.start[j], sigma, n_steps=cfg.n_steps,
                                seed=base + c, t_eps=cfg.sb_t_eps)[-1])
        lab, out = classify(pb, end)
        jj, ok = j.cpu().numpy(), lab >= 0
        bad += int((~ok).sum())
        far += float(out.sum())
        np.add.at(counts, (jj[ok], lab[ok]), 1.0)
        sel = np.arange(a, a + len(jj)) % K == 0
        first[jj[sel]] = end[sel]
    fin = np.isfinite(first).all(axis=1)
    return _normalise(counts), {"outside": far / len(rep), "nonfinite": bad / len(rep),
                                "endpoint": distribution_metrics(first[fin], pb.X_end, seed=seed)}


def _entropy(q: np.ndarray) -> np.ndarray:
    return -(np.clip(q, 1e-12, 1.0) * np.log(np.clip(q, 1e-12, 1.0))).sum(1) / np.log(q.shape[1])


def scores(q: np.ndarray, ref: np.ndarray) -> dict:
    corr = [c for c in (_corr(q[:, k], ref[:, k], False) for k in range(len(FATES))
                        if np.ptp(q[:, k]) > 1e-9) if c is not None]
    bias = float(((q.mean(0) - ref.mean(0)) ** 2).sum())
    return {"n": int(len(ref)), "se": float(((q - ref) ** 2).sum(1).mean()), "bias": bias,
            "cse": float((((q - q.mean(0)) - (ref - ref.mean(0))) ** 2).sum(1).mean()),
            "tv": float(0.5 * np.abs(q - ref).sum(1).mean()),
            "top1": float((q.argmax(1) == ref.argmax(1)).mean()),
            "pearson": float(np.mean(corr)) if corr else None,
            "entropy": float(_entropy(q).mean()), "spread": spread(ref),
            "q_mean": q.mean(0).tolist()}


def make_record(pb, q, seed: int, arm: str, width: float, extra: dict, budget) -> dict:
    return {"dim": pb.dim, "seed": int(seed), "arm": arm, "width": float(width),
            "K": K if arm in STOCH_ARMS else 1, "n_src": int(pb.n_src),
            "groups": {g: scores(q[m], pb.ref[m]) for g, m in pb.masks.items()},
            "eps_kernel": float(pb.metric.eps_kernel), "m_bar": pb.m_bar,
            "floor": pb.floor, "budget": budget, "point": pb.point, **extra}


def backbone(pb, seed: int, cfg: Run, root: str):
    path = net_path(root, pb.dim, seed)
    d = pb.metric.d
    if os.path.exists(path):
        blob = torch.load(path, map_location=cfg.device, weights_only=True)
        phi = GeoPathNet(d, **cfg.net_kw).to(cfg.device, cfg.dtype)
        v = VelocityNet(d, **cfg.net_kw).to(cfg.device, cfg.dtype)
        phi.load_state_dict(blob["phi"])
        v.load_state_dict(blob["v"])
        to = lambda a: a.to(cfg.device, cfg.dtype)
        return phi.eval(), v.eval(), (to(blob["A"]), to(blob["B"]), to(blob["pi"]))
    det = path_a(pb.metric, pb.X0, pb.X1, seed, cfg)
    c = det["coupler"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"phi": det["phi"].state_dict(), "v": det["v_net"].state_dict(),
                "A": c.X0.detach().cpu(), "B": c.X1.detach().cpu(),
                "pi": c.pi.detach().cpu()}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return det["phi"], det["v_net"], (c.X0, c.X1, c.pi)


def run_unit(pb, unit: str, width: float, seed: int, bb, cfg: Run) -> dict:
    geoms = {"C": pb.metric,
             "I": ConstantMobility(pb.metric, torch.eye(pb.metric.d, **cfg.torch_kw))}
    salt = lambda arm: f"{pb.dim}|{seed}|{arm}|{width:g}"
    if unit == "ffm_det":
        q, info = hard(pb, _np(integrate(bb[1], pb.start, cfg)[-1]), seed)
        return {"ffm_det": (q, info)}
    if unit in ("ffm_C", "ffm_I"):
        cov = unit[-1]
        sig_path = sigma_for(geoms[cov], width)
        t0 = time.time()
        v_net, s_net = train_sbm(bb[0], geoms[cov], Coupler(bb[2][0], bb[2][1], pi=bb[2][2],
                                                             seed=seed), sig_path, seed, cfg)
        info = {"path_cov": cov, "sigma_path": sig_path, "train_seconds": time.time() - t0}
        out = {}
        for diff in ("C", "I"):
            sig = sigma_for(geoms[diff], width)
            q, extra = sample(pb, v_net, s_net, geoms[diff], sig, salt(f"ffm_{cov}{diff}"), cfg,
                              seed)
            out[f"ffm_{cov}{diff}"] = (q, {**info, **extra, "diffusion": diff, "sigma": sig,
                                           "noise_var_rate": 2 * sig ** 2 * (
                                               pb.m_bar if diff == "C" else 1.0)})
        q, extra = hard(pb, _np(integrate_sde(v_net, s_net, geoms["I"], pb.start, 0.0,
                                              n_steps=cfg.n_steps, seed=seed,
                                              t_eps=cfg.sb_t_eps)[-1]), seed)
        out[f"ffm_{cov}off"] = (q, {**info, **extra, "diffusion": "off", "sigma": 0.0})
        return out
    if unit == "sf2m":
        sigma = 2.0 * width * pb.metric.eps_kernel
        hp = HParams(**{**cfg.arm_kw, "sigma": sigma, "sbm_iters": cfg.sb_iters})
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_arm("sf2m", pb.ts, hp, seed, cfg.device, cfg.dtype, n_steps=BASE_N_STEPS)
        info = {"sigma_path": sigma, "train_seconds": time.time() - t0}
        s_net, geom = res.aux["s_net"], res.aux["geom"]
        sig = sigma / np.sqrt(2.0)
        q, extra = sample(pb, res.v_net, s_net, geom, sig, salt("sf2m"), cfg, seed)
        q0, extra0 = hard(pb, _np(integrate_sde(res.v_net, s_net, geom, pb.start, 0.0,
                                                n_steps=cfg.n_steps, seed=seed,
                                                t_eps=cfg.sb_t_eps)[-1]), seed)
        return {"sf2m": (q, {**info, **extra, "diffusion": "I", "sigma": sig,
                             "noise_var_rate": 2 * sig ** 2}),
                "sf2m_off": (q0, {**info, **extra0, "diffusion": "off", "sigma": 0.0})}
    knobs = pb.point[unit]
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_arm(unit, pb.ts, HParams(**{**cfg.arm_kw, **knobs}), seed, cfg.device,
                      cfg.dtype, n_steps=BASE_N_STEPS)
    q, info = hard(pb, _np(integrate_flow(res.v_net, pb.start, n_steps=BASE_N_STEPS)[-1]), seed)
    return {unit: (q, {**info, "knobs": knobs})}


def write_refs(pb, root: str, budget) -> None:
    for arm, row in (("ref_mean", pb.ref.mean(0)), ("ref_target", pb.shares)):
        q = np.tile(row, (pb.n_src, 1))
        _dump(make_record(pb, q, -1, arm, 0.0, {}, budget), record_path(root, pb.dim, -1, arm, 0.0))


def ensure(cfg: Run, dims=DIMS, seeds=SEEDS, root: str | None = None,
           force: bool = False) -> None:
    root = root_dir(cfg.smoke) if root is None else root
    budget = budget_of(cfg)
    for dim in dims:
        point = tuned_points()[dim]
        todo = [(s, u, w) for s in seeds for u, w in units(dim)
                if force or any(not _fresh(record_path(root, dim, s, a, w), budget, point)
                                for a in UNIT_ARMS[u])]
        if not todo:
            continue
        pb = problem(dim, cfg)
        write_refs(pb, root, budget)
        cuda_warm_up(pb.metric, pb.X0, pb.X1, cfg)
        for seed in sorted({s for s, _, _ in todo}):
            mine = [(u, w) for s, u, w in todo if s == seed]
            bb = backbone(pb, seed, cfg, root) if any(u.startswith("ffm") for u, _ in mine) else None
            for u, w in mine:
                t0 = time.time()
                out = run_unit(pb, u, w, seed, bb, cfg)
                secs = time.time() - t0
                for arm, (q, extra) in out.items():
                    rec = make_record(pb, q, seed, arm, w, {**extra, "seconds": secs}, budget)
                    path = record_path(root, dim, seed, arm, w)
                    os.makedirs(os.path.dirname(phat_path(path)), exist_ok=True)
                    np.save(phat_path(path), q.astype(np.float32))
                    _dump(rec, path)
                    g = rec["groups"]["all"]
                    print(f"[fate] d{dim} s{seed} {arm:9s} w{w:<4g} se {g['se']:.4f} "
                          f"pearson {g['pearson'] or float('nan'):.3f} ({secs:.0f}s)", flush=True)


def rescore(root: str | None = None, smoke: bool = False) -> None:
    root = root_dir(smoke) if root is None else root
    cache = {}
    for path in sorted(glob.glob(os.path.join(root, "records", "*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        sc = cache.setdefault(rec["dim"], scaffold(load_cloud(rec["dim"])))
        q = (np.load(phat_path(path)).astype(np.float64) if os.path.exists(phat_path(path))
             else np.tile(rec["groups"]["all"]["q_mean"], (len(sc.ref), 1)))
        rec["groups"] = {g: scores(q[m], sc.ref[m]) for g, m in sc.masks.items()}
        _dump(rec, path)
    print(f"[fate] rescored {root}", flush=True)


def long_frame(root: str | None = None, smoke: bool = False) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(root_dir(smoke) if root is None else root,
                                           "records", "*.json"))):
        with open(p) as fh:
            r = json.load(fh)
        for g, s in r["groups"].items():
            rows.append({"d": r["dim"], "seed": r["seed"], "arm": r["arm"], "width": r["width"],
                         "group": g, **{k: s.get(k) for k in ("se", "bias", "cse", "tv", "top1",
                                                              "pearson", "entropy", "spread")},
                         "outside": r.get("outside"),
                         "end_W2": r.get("endpoint", {}).get("W2")})
    return pd.DataFrame(rows)


def _pm(v) -> str:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return f"{v.mean():.3f} ± {v.std(ddof=0):.3f}" if len(v) > 1 else (f"{v[0]:.3f}" if len(v) else "")


def frame_width(df: pd.DataFrame, metric: str = "se", groups=("all", "late")) -> pd.DataFrame:
    df = df[df["group"].isin(groups) & (df["arm"].isin(STOCH_ARMS) | df["arm"].str.endswith("off"))]
    t = df.groupby(["d", "group", "arm", "width"])[metric].apply(_pm).unstack("width")
    t.index = t.index.set_levels([LABELS.get(a, a) for a in t.index.levels[2]], level=2)
    return t


def frame_fixed(df: pd.DataFrame, groups=("all", "late")) -> pd.DataFrame:
    df = df[df["group"].isin(groups) & (df["width"] == 0.0)]
    t = df.groupby(["d", "group", "arm"])[["se", "bias", "cse", "pearson", "entropy",
                                          "end_W2"]].agg(_pm)
    t.index = t.index.set_levels([LABELS.get(a, a) for a in t.index.levels[2]], level=2)
    return t


def frame_wins(df: pd.DataFrame, metrics=("se", "cse", "pearson"),
               groups=("all", "late")) -> pd.DataFrame:
    rows = {}
    for (d, g), part in df[df["group"].isin(groups)].groupby(["d", "group"]):
        for metric in metrics:
            for w in sorted(part.loc[part["arm"].isin(STOCH_ARMS), "width"].unique()):
                for a, b in COMPARE:
                    wb = 0.0 if b == "ffm_det" else w
                    x = part[(part["arm"] == a) & (part["width"] == w)].set_index("seed")[metric]
                    y = part[(part["arm"] == b) & (part["width"] == wb)].set_index("seed")[metric]
                    x, y = x.align(y, join="inner")
                    if not len(x):
                        continue
                    k = int((x < y).sum())
                    rows[(d, g, metric, w, f"{LABELS[a]} vs {LABELS[b]}")] = {
                        "first lower": f"{k}/{len(x)}",
                        "mean diff": round(float((x - y).mean()), 4),
                        "sign p": round(binomtest(k, len(x)).pvalue, 4)}
    t = pd.DataFrame(rows).T
    t.index.names = ["d", "group", "metric", "width", "comparison"]
    return t


def floors(root: str | None = None, smoke: bool = False) -> pd.DataFrame:
    rows = {}
    for p in sorted(glob.glob(os.path.join(root_dir(smoke) if root is None else root,
                                           "records", "*.json"))):
        with open(p) as fh:
            r = json.load(fh)
        if r.get("floor"):
            rows[r["dim"]] = r["floor"]
    return pd.DataFrame(rows).T.rename_axis("d")


def frames(root: str | None = None, smoke: bool = False) -> dict:
    df = long_frame(root, smoke)
    return {"diversity": diversity(), "width": frame_width(df, "se"),
            "width_cse": frame_width(df, "cse"), "width_pearson": frame_width(df, "pearson"),
            "width_w2": frame_width(df, "end_W2", groups=("all",)),
            "width_outside": frame_width(df, "outside", groups=("all",)),
            "floor": floors(root, smoke), "fixed": frame_fixed(df), "wins": frame_wins(df),
            "long": df}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pancreas_fate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    r.add_argument("--force", action="store_true")
    sub.add_parser("diversity")
    sub.add_parser("rescore")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "diversity":
        print(diversity().to_string())
        return
    cfg = Run.from_env()
    if a.cmd == "run":
        ensure(cfg, a.dims, a.seeds, force=a.force)
    elif a.cmd == "rescore":
        rescore(smoke=cfg.smoke)
    else:
        f = frames(smoke=cfg.smoke)
        for k in ("diversity", "fixed", "width", "width_cse", "width_pearson", "width_w2",
                  "width_outside", "floor", "wins"):
            print(f[k].to_string())


if __name__ == "__main__":
    main()
