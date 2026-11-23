from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import time
import zlib

import numpy as np
import ot
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

from ffm.core.arms import HParams, run_arm
from ffm.core.bridge import integrate_sde
from ffm.core.metrics import wasserstein
from ffm.core.paths import experiment_root
from ffm.core.protocol import TrainingSet
from ffm.core.train import integrate_flow
from ffm.core.transition import build_uniform_knn_transition, build_velocity_kernel
from ffm.method import Coupler, FWFactory, Run, moments_from, path_a, sigma_for, train_sbm

from .bench import BASE_N_STEPS, OT_MAX_PTS, _np, cloud, transfer_points
from .data import CACHE_DIR, RAW, SOURCE, TARGET

NAME = "larry_fate"
DIMS = (20, 50)
SEEDS = (0, 1, 2, 3, 4)
KERNELS = ("expression", "velocity", "time", "lineage", "velocity_lineage", "time_lineage")
BASELINES = ("cfm", "mfm_land", "curly")
UNITS = tuple(f"ffm_{k}" for k in KERNELS) + BASELINES + ("sf2m",)
EXTRA_WIDTHS = (0.3, 1.0)
K = 64
KNN = 15
MIN_REL = 5
MIX = 0.5
OT_REG = 0.05
OT_TOP = 50
SAMPLE_CHUNK = 16384
CLONES = os.path.join(CACHE_DIR, "clones.npz")
NEU, MONO, UNDIFF = "Neutrophil", "Monocyte", "Undifferentiated"
GROUPS = ("nm", "any")


def root_dir(smoke: bool = False) -> str:
    return experiment_root(NAME, smoke)


def build_clones(raw: str = RAW) -> dict:
    import anndata as ad
    a = ad.read_h5ad(raw, backed="r")
    C = sp.csr_matrix(a.obsm["X_clone"])
    has = np.diff(C.indptr) > 0
    clone = np.full(a.n_obs, -1)
    clone[has] = C.indices[C.indptr[:-1][has]]
    state = np.asarray(a.obs["state_info"].astype(str))
    names = np.array(sorted(set(state)))
    np.savez_compressed(CLONES, clone=clone.astype(np.int64),
                        day=np.asarray(a.obs["time_info"], dtype=np.float64),
                        state=np.searchsorted(names, state).astype(np.int64), names=names,
                        n_clones=np.int64(C.shape[1]), multi=np.int64((np.diff(C.indptr) > 1).sum()))
    return {"n_cells": int(a.n_obs), "n_clones": int(C.shape[1]), "unlabelled": int((~has).sum()),
            "multi": int((np.diff(C.indptr) > 1).sum())}


def reference() -> tuple:
    with np.load(CLONES, allow_pickle=False) as z:
        clone, day, state, names, n = z["clone"], z["day"], z["state"], z["names"], int(z["n_clones"])
    fates = [str(s) for s in names if s != UNDIFF]
    code = {i: fates.index(str(s)) for i, s in enumerate(names) if s != UNDIFF}
    later = (day > SOURCE) & (clone >= 0) & (names[state] != UNDIFF)
    counts = np.zeros((n, len(fates)))
    np.add.at(counts, (clone[later], np.array([code[s] for s in state[later]], dtype=int)), 1.0)
    return fates, counts


def split(seed: int, n: int) -> np.ndarray:
    test = np.zeros(n, dtype=bool)
    test[np.random.default_rng(100 + seed).permutation(n)[: n // 2]] = True
    return test


def q_time(X: np.ndarray, p0: np.ndarray, p1: np.ndarray, d: int, root: str) -> sp.csr_matrix:
    path = os.path.join(root, "kernels", f"ot_d{d}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.as_tensor(X[p0], device=dev, dtype=torch.float64)
    B = torch.as_tensor(X[p1], device=dev, dtype=torch.float64)
    M = torch.cdist(A, B) ** 2
    a = torch.full((len(A),), 1.0 / len(A), device=dev, dtype=torch.float64)
    b = torch.full((len(B),), 1.0 / len(B), device=dev, dtype=torch.float64)
    pi = ot.sinkhorn(a, b, M, float(OT_REG * M.median()), method="sinkhorn_log", numItermax=5000)
    val, col = torch.topk(pi, OT_TOP, dim=1)
    val = (val / val.sum(1, keepdim=True)).cpu().numpy().ravel()
    rows = np.repeat(np.arange(len(A)), OT_TOP)
    Q = sp.csr_matrix((val, (rows, col.cpu().numpy().ravel())), shape=(len(A), len(B)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, data=Q.data, indices=Q.indices, indptr=Q.indptr, shape=np.asarray(Q.shape))
    os.replace(tmp, path)
    return Q


def q_lineage(X: np.ndarray, p0: np.ndarray, p1: np.ndarray, clone: np.ndarray,
              train_clone: np.ndarray) -> tuple:
    d2, d6 = np.flatnonzero(p0), np.flatnonzero(p1)
    col = {}
    for j, c in enumerate(clone[d6]):
        col.setdefault(int(c), []).append(j)
    own = np.array([clone[i] >= 0 and train_clone[clone[i]] and int(clone[i]) in col for i in d2])
    rows, cols, vals = [], [], []
    for r in np.flatnonzero(own):
        js = col[int(clone[d2[r]])]
        rows += [r] * len(js)
        cols += js
        vals += [1.0 / len(js)] * len(js)
    Q = sp.csr_matrix((vals, (rows, cols)), shape=(len(d2), len(d6)))
    src = np.flatnonzero(own)
    rest = np.flatnonzero(~own)
    nn = cKDTree(X[d2[src]]).query(X[d2[rest]], k=KNN)[1]
    T = sp.csr_matrix((np.full(nn.size, 1.0 / KNN), (np.repeat(np.arange(len(rest)), KNN),
                                                     src[nn.ravel()])), shape=(len(rest), len(d2)))
    stacked = sp.vstack([Q[src], T @ Q]).tocsr()
    return stacked[np.argsort(np.concatenate([src, rest]))], own


def mix(P: sp.csr_matrix, Q: sp.csr_matrix, p0: np.ndarray, p1: np.ndarray) -> sp.csr_matrix:
    n = P.shape[0]
    d2, d6 = np.flatnonzero(p0), np.flatnonzero(p1)
    q = Q.tocoo()
    Qf = sp.csr_matrix((q.data, (d2[q.row], d6[q.col])), shape=(n, n))
    w = np.where(p0, MIX, 0.0)
    return sp.csr_matrix(sp.diags(1.0 - w) @ P + sp.diags(w) @ Qf)


def problem(d: int, seed: int, cfg: Run, root: str) -> dict:
    c = cloud(d)
    idx = np.flatnonzero(np.isin(c["day"], (SOURCE, TARGET)))
    X, V = c["X"][idx], c["V"][idx]
    day, state, clone = c["day"][idx], c["state"][idx], c["clone"][idx]
    p0, p1 = day == SOURCE, day == TARGET
    fates, counts = reference()
    test = split(seed, len(counts))
    nm = counts[:, [fates.index(NEU), fates.index(MONO)]].sum(1)
    tot = counts.sum(1)
    in_test = (clone >= 0) & test[np.maximum(clone, 0)]
    elig = {"nm": nm[np.maximum(clone, 0)] >= MIN_REL, "any": tot[np.maximum(clone, 0)] >= MIN_REL}
    scored = {g: p0 & in_test & (clone >= 0) & e for g, e in elig.items()}
    train_ref = {g: p0 & ~in_test & (clone >= 0) & e for g, e in elig.items()}
    lab_cells = p1 & (state != UNDIFF)
    lab = np.array([fates.index(s) for s in state[lab_cells]])
    pts = transfer_points(d)
    return {"d": d, "seed": seed, "X": X, "V": V, "day": day, "state": state, "clone": clone,
            "p0": p0, "p1": p1, "fates": fates, "counts": counts, "test": test,
            "scored": scored, "train_ref": train_ref, "tree": cKDTree(X[lab_cells]), "lab": lab,
            "pts": pts, "src": np.flatnonzero(scored["any"] | scored["nm"]), "root": root}


def kernel(pb: dict, name: str) -> sp.csr_matrix:
    X, p0, p1 = pb["X"], pb["p0"], pb["p1"]
    base = build_velocity_kernel(X, pb["V"]) if name.startswith("velocity") else \
        build_uniform_knn_transition(X)
    if name in ("expression", "velocity"):
        return sp.csr_matrix(base)
    parts = []
    if "time" in name:
        parts.append(q_time(X, p0, p1, pb["d"], pb["root"]))
    if "lineage" in name:
        parts.append(q_lineage(X, p0, p1, pb["clone"], ~pb["test"])[0])
    return mix(base, sum(parts) / len(parts), p0, p1)


def training_set(pb: dict, P: sp.csr_matrix, name: str) -> TrainingSet:
    return TrainingSet(name=f"LARRY-fate-d{pb['d']}-{name}", idx=np.arange(len(pb["X"])),
                       X=pb["X"], P=sp.csr_matrix(P), p0=pb["p0"], p1=pb["p1"], rho=0.0,
                       holdout_band=None, p_variant=name, diag={}, velocity=pb["V"])


def soft(pb: dict, ends: np.ndarray) -> np.ndarray:
    ends = np.asarray(ends, dtype=np.float64)
    ok = np.isfinite(ends).all(1)
    q = np.full((len(ends), len(pb["fates"])), np.nan)
    if ok.any():
        nn = pb["tree"].query(ends[ok], k=KNN)[1]
        q[ok] = np.stack([(pb["lab"][nn] == f).mean(1) for f in range(len(pb["fates"]))], 1)
    return q


def hard(q: np.ndarray) -> np.ndarray:
    h = np.zeros_like(q)
    ok = np.isfinite(q).all(1)
    h[np.flatnonzero(ok), q[ok].argmax(1)] = 1.0
    h[~ok] = np.nan
    return h


def sample(pb: dict, v_net, s_net, geom, sigma: float, salt: str, cfg: Run) -> tuple:
    start = cfg.tensor(pb["X"][pb["src"]])
    rep = torch.arange(len(start), device=cfg.device).repeat_interleave(K)
    acc_s = np.zeros((len(start), len(pb["fates"])))
    acc_h = np.zeros_like(acc_s)
    n_ok = np.zeros(len(start))
    first = np.full((len(start), pb["X"].shape[1]), np.nan)
    base = zlib.crc32(salt.encode())
    for c, a in enumerate(range(0, len(rep), SAMPLE_CHUNK)):
        j = rep[a:a + SAMPLE_CHUNK]
        end = _np(integrate_sde(v_net, s_net, geom, start[j], sigma, n_steps=cfg.n_steps,
                                seed=base + c, t_eps=cfg.sb_t_eps)[-1])
        q = soft(pb, end)
        jj, ok = j.cpu().numpy(), np.isfinite(q).all(1)
        np.add.at(acc_s, jj[ok], q[ok])
        np.add.at(acc_h, jj[ok], hard(q[ok]))
        np.add.at(n_ok, jj[ok], 1.0)
        sel = (np.arange(a, a + len(jj)) % K == 0)
        first[jj[sel]] = end[sel]
    norm = np.maximum(n_ok, 1.0)[:, None]
    bad = n_ok == 0
    acc_s, acc_h = acc_s / norm, acc_h / norm
    acc_s[bad], acc_h[bad] = np.nan, np.nan
    return acc_s, acc_h, first, {"nonfinite": float(1.0 - n_ok.sum() / len(rep))}


def deterministic(pb: dict, ends: np.ndarray) -> tuple:
    q = soft(pb, ends)
    return q, hard(q), ends, {"nonfinite": float(1.0 - np.isfinite(q).all(1).mean())}


def bias_of(q: np.ndarray, fates: list) -> np.ndarray:
    a, b = q[:, fates.index(NEU)], q[:, fates.index(MONO)]
    s = a + b
    return np.where(s > 0, a / np.where(s > 0, s, 1.0), np.nan)


def _auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _clone_r(pred: np.ndarray, ref: np.ndarray, cl: np.ndarray) -> float | None:
    p = pd.Series(pred).groupby(cl).mean()
    r = pd.Series(ref).groupby(cl).first()
    return float(pearsonr(p, r)[0]) if np.ptp(p.to_numpy()) > 1e-12 else None


def scores(pb: dict, q: np.ndarray, group: str) -> dict:
    fates, counts = pb["fates"], pb["counts"]
    m = pb["scored"][group][pb["src"]]
    cl = pb["clone"][pb["src"]][m]
    ref = counts[cl] / counts[cl].sum(1, keepdims=True)
    trm = pb["train_ref"][group]
    tcl = pb["clone"][trm]
    const = (counts[tcl] / counts[tcl].sum(1, keepdims=True)).mean(0)
    qm = q[m]
    ok = np.isfinite(qm).all(1)
    qm = np.where(ok[:, None], qm, const[None])
    out = {"n": int(m.sum()), "n_clones": int(len(np.unique(cl))), "coverage": float(ok.mean()),
           "tv": float(0.5 * np.abs(qm - ref).sum(1).mean()),
           "se": float(((qm - ref) ** 2).sum(1).mean()),
           "cse": float((((qm - qm.mean(0)) - (ref - ref.mean(0))) ** 2).sum(1).mean()),
           "q_mean": qm.mean(0).tolist(), "ref_mean": ref.mean(0).tolist()}
    if group == "nm":
        rb = bias_of(ref, fates)
        pbias = bias_of(qm, fates)
        cb = float(bias_of(const[None], fates)[0])
        defined = np.isfinite(pbias)
        pbias = np.where(defined, pbias, cb)
        varies = np.ptp(pbias) > 1e-12
        out.update({"bias_defined": float(defined.mean()),
                    "pearson": float(pearsonr(pbias, rb)[0]) if varies else None,
                    "pearson_clone": _clone_r(pbias, rb, cl) if varies else None,
                    "spearman": float(spearmanr(pbias, rb)[0]) if varies else None,
                    "auroc": _auroc(rb > 0.5, pbias) if varies else 0.5,
                    "bias_mse": float(((pbias - rb) ** 2).mean()),
                    "bias_cmse": float((((pbias - pbias.mean()) - (rb - rb.mean())) ** 2).mean()),
                    "bias_pred": np.round(pbias, 4).tolist(), "bias_ref": np.round(rb, 4).tolist()})
    return out


def landing(pb: dict, first: np.ndarray, seed: int) -> dict:
    ok = np.isfinite(first).all(1)
    T = pb["X"][pb["p1"]]
    return {"W2_end": float(wasserstein(first[ok], T, p=2, max_n=2000, seed=seed)) if ok.any() else None}


def record(pb: dict, arm: str, width: float | None, out: tuple, extra: dict) -> dict:
    qs, qh, first, info = out
    return {"dim": pb["d"], "seed": pb["seed"], "arm": arm, "width": width,
            "K": 1 if width is None else K, "n_src": int(len(pb["src"])),
            "soft": {g: scores(pb, qs, g) for g in GROUPS},
            "hard": {g: scores(pb, qh, g) for g in GROUPS},
            **landing(pb, first, pb["seed"]), **info, **extra}


def widths(pb: dict) -> tuple:
    return tuple(sorted({float(pb["pts"]["pathb"]["width_mult"]), *EXTRA_WIDTHS}))


def rec_path(root: str, d: int, seed: int, arm: str, width) -> str:
    w = "det" if width is None else f"w{width:g}"
    return os.path.join(root, "records", f"d{d}_s{seed}_{arm}_{w}.json")


def _dump(rec: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh)
    os.replace(path + ".tmp", path)


def run_unit(d: int, seed: int, unit: str, cfg: Run, force: bool = False) -> None:
    root = root_dir(cfg.smoke)
    pb = problem(d, seed, cfg, root)
    ws = widths(pb)
    outs = ([(unit, None)] + [(f"{unit}_sde", w) for w in ws] if unit.startswith("ffm_") else
            [(unit, w) for w in ws] if unit == "sf2m" else [(unit, None)])
    if not force and all(os.path.exists(rec_path(root, d, seed, a, w)) for a, w in outs):
        return
    pts, t0 = pb["pts"], time.perf_counter()
    if unit.startswith("ffm_"):
        name = unit[4:]
        P = kernel(pb, name)
        X = pb["X"]
        b0, D = moments_from(P, X)
        fac = FWFactory(X, D, b0, cfg, rho_mult=pts["ffm"]["rho_mult"],
                        lam_mult=pts["ffm"]["lam_mult"])
        metric = fac.base
        Xt = cfg.tensor(X)
        X0, X1 = Xt[torch.as_tensor(pb["p0"], device=cfg.device)], Xt[torch.as_tensor(pb["p1"], device=cfg.device)]
        det = path_a(metric, X0, X1, seed, cfg)
        ends = det["traj"][-1][np.searchsorted(np.flatnonzero(pb["p0"]), pb["src"])]
        extra = {"point": pts["ffm"], "kernel": name, "seconds": time.perf_counter() - t0,
                 "scales": {"rho_star": fac.rho_star, "rho": fac.rho, "a_bar": fac.a_bar,
                            "lam": fac.lam}}
        _dump(record(pb, unit, None, deterministic(pb, ends), extra), rec_path(root, d, seed, unit, None))
        co = det["coupler"]
        for w in ws:
            t1 = time.perf_counter()
            sig = sigma_for(metric, w)
            v_net, s_net = train_sbm(det["phi"], metric, Coupler(co.X0, co.X1, pi=co.pi, seed=seed),
                                     sig, seed, cfg)
            res = sample(pb, v_net, s_net, metric, sig, f"{d}|{seed}|{unit}|{w:g}", cfg)
            _dump(record(pb, f"{unit}_sde", w, res, {**extra, "sigma": sig,
                                                     "seconds": time.perf_counter() - t1}),
                  rec_path(root, d, seed, f"{unit}_sde", w))
        print(f"[larry-fate] d{d} s{seed} {unit} {time.perf_counter() - t0:.0f}s", flush=True)
        return
    ts = training_set(pb, build_velocity_kernel(pb["X"], pb["V"]), "velocity")
    if unit == "sf2m":
        nn_d = cKDTree(pb["X"]).query(pb["X"], k=2)[0][:, 1]
        eps = float(np.median(nn_d[nn_d > 0])) * cfg.eps_kernel_scale
        for w in ws:
            t1 = time.perf_counter()
            sigma = 2.0 * w * eps
            hp = HParams(**{**cfg.arm_kw, "sigma": sigma, "sbm_iters": cfg.sb_iters,
                            "ot_max_pts": OT_MAX_PTS})
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_arm("sf2m", ts, hp, seed, cfg.device, cfg.dtype, n_steps=BASE_N_STEPS)
            out = sample(pb, res.v_net, res.aux["s_net"], res.aux["geom"], sigma / np.sqrt(2.0),
                         f"{d}|{seed}|sf2m|{w:g}", cfg)
            _dump(record(pb, "sf2m", w, out, {"sigma_path": sigma,
                                              "seconds": time.perf_counter() - t1}),
                  rec_path(root, d, seed, "sf2m", w))
        print(f"[larry-fate] d{d} s{seed} sf2m {time.perf_counter() - t0:.0f}s", flush=True)
        return
    knobs = pts[unit]
    hp = HParams(**{**cfg.arm_kw, **knobs, "ot_max_pts": OT_MAX_PTS})
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_arm(unit, ts, hp, seed, cfg.device, cfg.dtype, n_steps=BASE_N_STEPS)
    ends = _np(integrate_flow(res.v_net, cfg.tensor(pb["X"][pb["src"]]), n_steps=BASE_N_STEPS)[-1])
    _dump(record(pb, unit, None, deterministic(pb, ends),
                 {"point": knobs, "seconds": time.perf_counter() - t0}), rec_path(root, d, seed, unit, None))
    print(f"[larry-fate] d{d} s{seed} {unit} {time.perf_counter() - t0:.0f}s", flush=True)


def absorption(pb: dict, P: sp.csr_matrix) -> np.ndarray:
    absorb = pb["p1"] & (pb["state"] != UNDIFF)
    tr = np.flatnonzero(~absorb)
    ab = np.flatnonzero(absorb)
    Q = P[tr][:, tr]
    R = P[tr][:, ab] @ sp.csr_matrix((np.ones(len(ab)), (np.arange(len(ab)), pb["lab"])),
                                     shape=(len(ab), len(pb["fates"])))
    A = (sp.identity(len(tr)) - (1.0 - 1e-6) * Q).tocsc()
    B = spla.spsolve(A, R.toarray())
    out = np.full((len(pb["X"]), len(pb["fates"])), np.nan)
    out[tr] = B / np.maximum(B.sum(1, keepdims=True), 1e-300)
    return out[pb["src"]]


def references(d: int, seed: int, cfg: Run, force: bool = False) -> None:
    root = root_dir(cfg.smoke)
    path = rec_path(root, d, seed, "references", None)
    if os.path.exists(path) and not force:
        return
    pb = problem(d, seed, cfg, root)
    counts = pb["counts"]
    out = {}
    for g in GROUPS:
        trm = pb["train_ref"][g]
        tcl = pb["clone"][trm]
        tref = counts[tcl] / counts[tcl].sum(1, keepdims=True)
        nn = cKDTree(pb["X"][trm]).query(pb["X"][pb["src"]], k=KNN)[1]
        q = tref[nn].mean(1)
        out[f"knn_lineage_{g}"] = scores(pb, q, g)
        out[f"constant_{g}"] = scores(pb, np.tile(tref.mean(0), (len(pb["src"]), 1)), g)
    for name in KERNELS:
        q = absorption(pb, kernel(pb, name))
        out[f"absorption_{name}"] = {g: scores(pb, q, g) for g in GROUPS}
    m = pb["scored"]["nm"]
    cl = pb["clone"][m]
    with np.load(CLONES, allow_pickle=False) as z:
        rc, rd, rs, names = z["clone"], z["day"], z["state"], z["names"]
    sel = (rd > SOURCE) & np.isin(names[rs], (NEU, MONO))
    rel = {}
    for c_, s_ in zip(rc[sel], names[rs[sel]]):
        rel.setdefault(int(c_), []).append(s_ == NEU)
    rng = np.random.default_rng(seed)
    rs_ = []
    for _ in range(20):
        a, b = [], []
        for c_ in cl:
            v = np.asarray(rel[int(c_)])
            p = rng.permutation(len(v))
            h = len(v) // 2
            a.append(v[p[:h]].mean())
            b.append(v[p[h:]].mean())
        rs_.append(pearsonr(a, b)[0])
    r = float(np.mean(rs_))
    out["ceiling"] = {"split_half_r": r, "reliability": 2 * r / (1 + r),
                      "max_pearson": float(np.sqrt(max(2 * r / (1 + r), 0.0))),
                      "n": int(m.sum()), "n_clones": int(len(np.unique(cl)))}
    out["counts"] = {g: {"scored": int(pb["scored"][g].sum()), "train": int(pb["train_ref"][g].sum()),
                         "day2": int(pb["p0"].sum())} for g in GROUPS}
    _dump(out, path)


def tasks() -> list:
    return [(d, s, u) for d in DIMS for s in SEEDS for u in UNITS] + \
        [(d, s, "references") for d in DIMS for s in SEEDS]


def load_records(smoke: bool = False) -> list:
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root_dir(smoke), "records", "*.json")))
            if "references" not in p]


def _pm(v) -> str:
    v = [x for x in v if x is not None]
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "–"


def frames(smoke: bool = False, readout: str = "soft") -> dict:
    recs = load_records(smoke)
    if not recs:
        return {}
    rows = []
    for r in recs:
        nm, an = r[readout]["nm"], r[readout]["any"]
        rows.append({"d": r["dim"], "seed": r["seed"], "arm": r["arm"],
                     "width": "det" if r["width"] is None else r["width"],
                     "pearson": nm.get("pearson"), "spearman": nm.get("spearman"),
                     "auroc": nm.get("auroc"), "pearson_clone": nm.get("pearson_clone"),
                     "bias_cmse": nm["bias_cmse"], "bias_mse": nm["bias_mse"],
                     "tv": an["tv"], "W2_end": r.get("W2_end"), "n": nm["n"]})
    df = pd.DataFrame(rows)
    summary = df.groupby(["d", "arm", "width"]).agg(
        **{k: (k, _pm) for k in ("pearson", "pearson_clone", "spearman", "auroc", "bias_cmse",
                                 "tv", "W2_end")},
        n=("n", "mean"), seeds=("seed", "nunique"))
    refs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root_dir(smoke), "records",
                                                                      "*references*.json")))]
    return {"summary": summary, "long": df, "references": refs, "reference": reference_frame(smoke),
            "paired": paired(df)}


def reference_frame(smoke: bool = False) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(root_dir(smoke), "records", "*references*.json"))):
        d = int(os.path.basename(p).split("_")[0][1:])
        r = json.load(open(p))
        for k, v in r.items():
            if k.endswith("_nm") or k.startswith("absorption_"):
                nm = v.get("nm", v)
                rows.append({"d": d, "predictor": k.removesuffix("_nm"),
                             **{m: nm.get(m) for m in ("pearson", "pearson_clone", "auroc")}})
        rows.append({"d": d, "predictor": "ceiling", "pearson": r["ceiling"]["max_pearson"],
                     "split_half_r": r["ceiling"]["split_half_r"]})
    df = pd.DataFrame(rows)
    return df.groupby(["d", "predictor"]).agg(**{m: (m, _pm) for m in (
        "pearson", "pearson_clone", "auroc", "split_half_r")})


TWINS = (("lineage", "expression"), ("velocity_lineage", "velocity"), ("time_lineage", "time"),
         ("time", "expression"))


def paired(df: pd.DataFrame, metric: str = "pearson") -> pd.DataFrame:
    s = df.set_index(["d", "seed", "arm", "width"])[metric]
    at = lambda a, w: s.xs((a, w), level=("arm", "width"))
    sde = sorted(w for w in df["width"].unique() if w != "det")
    sfx = lambda w: "" if w == "det" else "_sde"
    jobs = [(f"{k}: SDE - deterministic", w, f"ffm_{k}_sde", f"ffm_{k}", "det")
            for k in KERNELS for w in sde]
    jobs += [(f"{a} - {b}", w, f"ffm_{a}{sfx(w)}", f"ffm_{b}{sfx(w)}", w)
             for a, b in TWINS for w in ["det"] + sde]
    rows = {}
    for name, w, a, b, wb in jobs:
        z = (at(a, w) - at(b, wb)).dropna()
        for d, v in z.groupby(level="d"):
            rows[(name, str(w), d)] = f"{v.mean():+.3f} ({int((v > 0).sum())}/{len(v)})"
    return pd.Series(rows).unstack(-1).rename_axis(["comparison", "width"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="larry.fate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("clones")
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--force", action="store_true")
    sub.add_parser("count")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    cfg = Run.from_env().but(ot_max_pts=OT_MAX_PTS)
    if a.cmd == "clones":
        print(json.dumps(build_clones()))
    elif a.cmd == "count":
        print(len(tasks()))
    elif a.cmd == "run":
        d, s, u = tasks()[a.index]
        if u == "references":
            references(d, s, cfg, a.force)
        else:
            run_unit(d, s, u, cfg, a.force)
    else:
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(frames(cfg.smoke)["summary"].to_string())


if __name__ == "__main__":
    main()
