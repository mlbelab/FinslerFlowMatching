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
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from scipy.stats import binomtest, rankdata

from ffm.core.arms import HParams, run_arm
from ffm.core.bridge import integrate_sde
from ffm.core.paths import experiment_root
from ffm.core.train import integrate_flow
from scripts.sheet.split import make_split, split_training_set
from ffm.method import (ConstantMobility, Coupler, FWFactory, GeoPathNet, Run,
                            VelocityNet, cuda_warm_up, integrate, moments_from, path_a,
                            sigma_for, train_sbm)

from .datasets import (FATE_NAMES, FORK_BAND, FORK_FATES, FORK_SPLIT, P_KNN_K, SOURCE,
                       TARGET, _arm, _shares, _trunk, fate_balls, fate_inside,
                       fate_resolved, make_dataset)

NAME = "forksheet_calibration"
SHARES = {"50": (0.5, 0.5), "70": (0.7, 0.3), "40": (0.4, 0.6), "20": (0.2, 0.8), "10": (0.1, 0.9)}
SHARE_LABEL = {"50": "50/50", "70": "70/30", "40": "40/60", "20": "20/80", "10": "10/90"}
SEEDS = tuple(range(10))
WIDTHS = (0.3, 1.0, 3.0)
K = 64
CHUNK = 32768
BASE_N_STEPS = 102
BINS = 10
BASELINES = ("cfm", "mfm_land", "curly")
CDC = {"cdcfm": {"cdc_gamma": 1.0, "cdc_rank": 0, "cdc_sigma_min": 0.0},
       "cdcfm_g4": {"cdc_gamma": 4.0, "cdc_rank": 0, "cdc_sigma_min": 0.0}}
DET_ENGINE = (*BASELINES, *CDC)
REFS = ("ref_bayes", "ref_q", "ref_qs", "ref_share")
FIXED = ("ffm_det", *DET_ENGINE, *REFS)
UNIT_ARMS = {"ffm_det": ("ffm_det",),
             "ffm_C": ("ffm_CC", "ffm_CI", "ffm_Coff"),
             "ffm_I": ("ffm_IC", "ffm_II", "ffm_Ioff"),
             "sf2m": ("sf2m", "sf2m_off"),
             **{b: (b,) for b in DET_ENGINE}}
GROUPS = ("unres", "res", "trunk", "all", "held", "held_unres", "held_res")
LABELS = {"ffm_det": "FFM ODE (Path A)",
          "ffm_CC": "FFM SDE, path C / diff C",
          "cfm": "OT-CFM",
          "mfm_land": "MFM / LAND",
          "curly": "Curly-FM",
          "cdcfm": "CDC-FM (gamma 1)",
          "cdcfm_g4": "CDC-FM (gamma 4)",
          "ffm_II": "FFM SDE, path I / diff I",
          "ffm_IC": "FFM SDE, path I / diff C",
          "ffm_CI": "FFM SDE, path C / diff I",
          "ffm_Coff": "FFM PF-ODE, path C",
          "ffm_Ioff": "FFM PF-ODE, path I",
          "sf2m": "[SF]2M SDE",
          "sf2m_off": "[SF]2M PF-ODE",
          "ref_bayes": "ref: Bayes p(label | position)",
          "ref_q": "ref: P_full absorption q",
          "ref_qs": "ref: position-smoothed q",
          "ref_share": "ref: constant share"}
SHORT = {"ffm_det": "ODE", "ffm_CC": "SDE(C,C)", "ffm_CI": "SDE(C,I)", "ffm_IC": "SDE(I,C)",
         "ffm_II": "SDE(I,I)", "ffm_Coff": "PF-ODE(C)", "ffm_Ioff": "PF-ODE(I)",
         "sf2m": "[SF]2M", "sf2m_off": "[SF]2M PF-ODE", "cfm": "OT-CFM", "mfm_land": "MFM",
         "curly": "Curly", "cdcfm": "CDC-FM", "cdcfm_g4": "CDC-FM g4"}
MODEL_ARMS = tuple(a for a in LABELS if not a.startswith("ref_"))
STOCH_ARMS = ("ffm_CC", "ffm_CI", "ffm_IC", "ffm_II", "ffm_Coff", "ffm_Ioff", "sf2m",
              "sf2m_off")
CAL_GROUPS = ("unres", "res", "all", "held")
CAL_METRICS = ("brier2", "brier3", "ece", "se_q", "se_qs", "se_bayes", "pearson_q",
               "spearman_q", "spearman_qs", "auroc", "entropy", "outside", "covered")
WIDTH_METRICS = ("brier2", "brier3", "ece", "se_qs", "spearman_qs", "entropy", "outside")
WIN_METRICS = ("brier2", "brier3", "ece", "se_qs")
COMPARE = (("ffm_CC", "ffm_II"), ("ffm_CC", "sf2m"), ("ffm_II", "sf2m"),
           ("ffm_CC", "ffm_IC"), ("ffm_CC", "ffm_CI"), ("ffm_CC", "ffm_Coff"),
           ("ffm_II", "ffm_Ioff"), ("sf2m", "sf2m_off"), ("ffm_CC", "ffm_det"),
           ("ffm_CC", "cfm"), ("ffm_CC", "mfm_land"), ("ffm_CC", "curly"),
           ("ffm_CC", "cdcfm"), ("ffm_CC", "cdcfm_g4"))


def tuned_point() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "sheet",
                        "tuned.json")
    with open(path) as fh:
        return json.load(fh)


TUNED_WIDTH = float(tuned_point()["pathb"]["width_mult"])
assert TUNED_WIDTH in WIDTHS


def root_dir(smoke: bool = False) -> str:
    return experiment_root(NAME, smoke)


def record_path(root: str, tag: str, seed: int, arm: str, width: float) -> str:
    return os.path.join(root, "records", f"sh{tag}_s{seed}_{arm}_w{width:g}.json")


def ref_path(root: str, tag: str, arm: str) -> str:
    return os.path.join(root, "records", f"sh{tag}_{arm}.json")


def net_path(root: str, tag: str, seed: int) -> str:
    return os.path.join(root, "nets", f"sh{tag}_s{seed}.pt")


def phat_path(record: str) -> str:
    head, name = os.path.split(record)
    return os.path.join(os.path.dirname(head), "phat", name[:-len(".json")] + ".npy")


def units() -> list[tuple[str, float]]:
    out = [("ffm_det", 0.0)]
    for w in WIDTHS:
        out += [("ffm_C", w), ("ffm_I", w), ("sf2m", w)]
    return out + [(b, 0.0) for b in DET_ENGINE]


def budget_of(cfg: Run) -> dict:
    return {"geo_iters": cfg.geo_iters, "cfm_iters": cfg.cfm_iters, "sb_iters": cfg.sb_iters,
            "n_steps": cfg.n_steps, "base_n_steps": BASE_N_STEPS, "K": K,
            "width": cfg.net_width, "depth": cfg.net_depth, "batch": cfg.batch}


def _fresh(path: str, budget: dict, point: dict) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        rec = json.load(fh)
    return rec.get("budget") == budget and rec.get("point") == point


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(_clean(obj), fh, sort_keys=True)
    os.replace(tmp, path)


def absorption(P, groups) -> np.ndarray:
    P = sp.csr_matrix(P, dtype=np.float64)
    G = np.stack([np.asarray(g, dtype=np.float64) for g in groups], axis=1)
    tgt = G.sum(axis=1) > 0
    E = sp.csr_matrix(P - sp.diags(P.diagonal()))
    E.eliminate_zeros()
    reach = tgt.copy()
    while True:
        nxt = reach | (np.asarray(E @ reach.astype(np.float64)).ravel() > 0)
        if (nxt == reach).all():
            break
        reach = nxt
    tr, ti = np.flatnonzero(reach & ~tgt), np.flatnonzero(tgt)
    Pt = P[tr]
    rhs = np.asarray(Pt[:, ti] @ G[ti])
    out = np.full(G.shape, np.nan)
    out[ti] = G[ti]
    out[tr] = splu(sp.csc_matrix(sp.identity(len(tr)) - Pt[:, tr])).solve(rhs)
    return out


def smooth(X: np.ndarray, vals: np.ndarray, k: int = P_KNN_K) -> np.ndarray:
    d, nn = cKDTree(X).query(X, k=k + 1)
    d, nn = d[:, 1:], nn[:, 1:]
    bw = float(np.median(d[:, -1]))
    w = np.exp(-d ** 2 / (2.0 * bw ** 2))
    return (w * vals[nn]).sum(axis=1) / w.sum(axis=1)


def bayes_lower(ds, shares) -> np.ndarray:
    p = _shares(shares)
    s = ds.prog
    tau = np.clip((s - FORK_SPLIT) / (1.0 - FORK_SPLIT), 0.0, 1.0)
    base, w = _trunk(s)[0], ds.uv[:, 1]
    lik = np.stack([pk * (np.abs(w - (base + d * a * _arm(tau, e, n)[0])) <= FORK_BAND + 1e-9)
                    for pk, (_, d, a, e, n) in zip(p, FORK_FATES)], axis=1)
    return lik[:, 0] / lik.sum(axis=1)


def references(ds, shares) -> dict:
    tgt = ds.region == TARGET
    q = absorption(ds.P_full, [tgt & (ds.fate == 0), tgt & (ds.fate == 1)])[:, 0]
    assert np.isfinite(q).all(), "some cell cannot reach the target under P_full"
    return {"q": q, "qs": smooth(ds.X, q), "bayes": bayes_lower(ds, shares),
            "share": np.full(len(q), float(_shares(shares)[0]))}


def scaffold(tag: str, cfg: Run) -> SimpleNamespace:
    ds = make_dataset("ForkSheet", shares=SHARES[tag])
    split = make_split(ds, seed=cfg.split_seed)
    ts = split_training_set(ds, split)
    src = ts.idx[ts.p0]
    hold = np.sort(np.concatenate([split.val_idx, split.test_idx]))
    hold = hold[ds.region[hold] == SOURCE]
    cells = np.concatenate([src, hold])
    train = np.arange(len(cells)) < len(src)
    res = fate_resolved(ds, cells)
    trunk = ds.prog[cells] < FORK_SPLIT
    refs = references(ds, SHARES[tag])
    return SimpleNamespace(
        tag=tag, ds=ds, ts=ts, point=tuned_point(), cells=cells, n_src=len(src),
        balls=fate_balls(ds), y=(ds.fate[cells] == 0).astype(np.float64),
        refs={k: v[cells] for k, v in refs.items()},
        masks={"unres": train & ~res, "res": train & res, "trunk": train & trunk,
               "all": train, "held": ~train, "held_unres": ~train & ~res,
               "held_res": ~train & res})


def problem(tag: str, cfg: Run) -> SimpleNamespace:
    pb = scaffold(tag, cfg)
    ts = pb.ts
    X = np.asarray(ts.X, dtype=np.float64)
    b0, D = moments_from(sp.csr_matrix(ts.P), X)
    ffm = pb.point["ffm"]
    fw = FWFactory(X, D, b0, cfg, rho_mult=ffm["rho_mult"], lam_mult=ffm["lam_mult"])
    pb.metric = fw.at(rho=fw.rho_rungs([ffm["rho_mult"]])[0],
                      lam=fw.lam_rungs([ffm["lam_mult"]])[0])
    pb.m_bar = float(pb.metric.mean_mobility_eig())
    pb.ts_base = dataclasses.replace(ts, velocity=b0)
    Xt = cfg.tensor(X)
    pb.X0 = Xt[torch.as_tensor(ts.p0, device=cfg.device)]
    pb.X1 = Xt[torch.as_tensor(ts.p1, device=cfg.device)]
    pb.start = cfg.tensor(pb.ds.X[pb.cells])
    return pb


def classify(pb, ends) -> np.ndarray:
    _, centres, axes, r = pb.balls
    ins = fate_inside(np.asarray(ends, dtype=np.float64), centres, axes, r)
    return np.where(ins.any(axis=1), ins.argmax(axis=1), 2)


def hard(pb, ends) -> np.ndarray:
    return np.eye(3)[classify(pb, ends)]


def _np(x) -> np.ndarray:
    return x.double().cpu().numpy()


def sample(pb, v_net, s_net, geom, sigma: float, salt: str, cfg: Run):
    n = len(pb.start)
    rep = torch.arange(n, device=cfg.device).repeat_interleave(K)
    base = zlib.crc32(salt.encode())
    counts = np.zeros((n, 3))
    bad = 0
    for c, a in enumerate(range(0, len(rep), CHUNK)):
        j = rep[a:a + CHUNK]
        end = _np(integrate_sde(v_net, s_net, geom, pb.start[j], sigma, n_steps=cfg.n_steps,
                                seed=base + c, t_eps=cfg.sb_t_eps)[-1])
        bad += int((~np.isfinite(end).all(axis=1)).sum())
        np.add.at(counts, (j.cpu().numpy(), classify(pb, end)), 1.0)
    return counts / K, bad / len(rep)


def _mean(a):
    return float(np.mean(a)) if len(a) else None


def _corr(a, b, rank: bool):
    if len(a) < 3:
        return None
    if rank:
        a, b = rankdata(a), rankdata(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _bins(p):
    return np.minimum((p * BINS).astype(np.int64), BINS - 1)


def _ece(p, y):
    if not len(p):
        return None
    b = _bins(p)
    return float(sum(abs((y[b == i] - p[b == i]).sum()) for i in range(BINS)) / len(p))


def _auroc(p, y):
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return None
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _entropy(p):
    c = np.clip(p, 1e-12, 1.0 - 1e-12)
    h = -(c * np.log2(c) + (1.0 - c) * np.log2(1.0 - c))
    return np.where((p <= 0.0) | (p >= 1.0), 0.0, h)


def _split2(phat):
    arm = phat[:, 0] + phat[:, 1]
    m = arm > 0
    return m, phat[m, 0] / arm[m]


def scores(phat, y, ref) -> dict:
    m, p2 = _split2(phat)
    y2 = y[m]
    b3 = (phat[:, 0] - y) ** 2 + (phat[:, 1] - (1.0 - y)) ** 2 + phat[:, 2] ** 2
    out = {"n": int(len(y)), "n2": int(m.sum()), "covered": _mean(m),
           "ybar": _mean(y), "ybar2": _mean(y2),
           "brier2": _mean((p2 - y2) ** 2), "brier3": _mean(b3),
           "ece": _ece(p2, y2), "auroc": _auroc(p2, y2),
           "entropy": _mean(_entropy(p2)), "outside": _mean(phat[:, 2]),
           "plow2": _mean(p2), "plow3": _mean(phat[:, 0])}
    for k in ("q", "qs", "bayes"):
        out[f"se_{k}"] = _mean((p2 - ref[k][m]) ** 2)
    for k in ("q", "qs"):
        out[f"pearson_{k}"] = _corr(p2, ref[k][m], False)
        out[f"spearman_{k}"] = _corr(p2, ref[k][m], True)
    return out


def reliability(phat, y) -> list:
    m, p2 = _split2(phat)
    y2, b = y[m], _bins(p2)
    return [[int((b == i).sum()), _mean(p2[b == i]), _mean(y2[b == i])] for i in range(BINS)]


def recall(phat, y) -> dict:
    out = {}
    for f, name in FATE_NAMES.items():
        m = (y > 0.5) == (f == 0)
        out[name] = ([100.0 * float(phat[m, f].mean()), 100.0 * float(phat[m, 1 - f].mean()),
                      100.0 * float(phat[m, 2].mean())] if m.any() else None)
    return out


def score_part(pb, phat) -> dict:
    sub = lambda m: {k: v[m] for k, v in pb.refs.items()}
    return {"groups": {g: scores(phat[m], pb.y[m], sub(m)) for g, m in pb.masks.items()},
            "reliability": {g: reliability(phat[pb.masks[g]], pb.y[pb.masks[g]])
                            for g in ("unres", "held")},
            "recall": recall(phat[pb.masks["res"]], pb.y[pb.masks["res"]]),
            "recall_held": recall(phat[pb.masks["held_res"]], pb.y[pb.masks["held_res"]])}


def make_record(pb, phat, seed: int, arm: str, width: float, extra: dict,
                budget: dict | None) -> dict:
    run = ({"eps_kernel": float(pb.metric.eps_kernel), "m_bar": pb.m_bar,
            "rho": float(pb.metric.rho), "lam": float(pb.metric.lam)}
           if hasattr(pb, "metric") else {})
    return {"tag": pb.tag, "shares": list(SHARES[pb.tag]), "seed": int(seed), "arm": arm,
            "width": float(width), "K": K if arm in STOCH_ARMS and "off" not in arm else 1,
            "n_src": int(pb.n_src), "n_held": int(len(pb.cells) - pb.n_src),
            **score_part(pb, phat), **run, "budget": budget, "point": pb.point, **extra}


def write_refs(pb, root: str) -> None:
    for a in REFS:
        p = pb.refs[a[4:]]
        phat = np.stack([p, 1.0 - p, np.zeros_like(p)], axis=1)
        _dump(make_record(pb, phat, -1, a, 0.0, {"family": "reference"}, None),
              ref_path(root, pb.tag, a))


def backbone(pb, seed: int, cfg: Run, root: str):
    path = net_path(root, pb.tag, seed)
    d = pb.metric.d
    if os.path.exists(path):
        blob = torch.load(path, map_location=cfg.device, weights_only=True)
        phi = GeoPathNet(d, **cfg.net_kw).to(cfg.device, cfg.dtype)
        v = VelocityNet(d, **cfg.net_kw).to(cfg.device, cfg.dtype)
        phi.load_state_dict(blob["phi"])
        v.load_state_dict(blob["v"])
        to = lambda a: a.to(cfg.device, cfg.dtype)
        return phi.eval(), v.eval(), (to(blob["A"]), to(blob["B"]), to(blob["pi"])), None
    det = path_a(pb.metric, pb.X0, pb.X1, seed, cfg)
    c = det["coupler"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"phi": det["phi"].state_dict(), "v": det["v_net"].state_dict(),
                "A": c.X0.detach().cpu(), "B": c.X1.detach().cpu(),
                "pi": c.pi.detach().cpu()}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return det["phi"], det["v_net"], (c.X0, c.X1, c.pi), det["traj"][-1]


def run_unit(pb, unit: str, width: float, seed: int, bb, cfg: Run) -> dict:
    geoms = {"C": pb.metric,
             "I": ConstantMobility(pb.metric, torch.eye(pb.metric.d, **cfg.torch_kw))}
    salt = lambda arm: f"{pb.tag}|{seed}|{arm}|{width:g}"
    if unit == "ffm_det":
        end = _np(integrate(bb[1], pb.start, cfg)[-1])
        extra = {"family": "ffm_ode"}
        if bb[3] is not None:
            extra["check"] = float(np.abs(end[:pb.n_src] - bb[3]).max())
        return {"ffm_det": (hard(pb, end), extra)}
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
            m_inf = pb.m_bar if diff == "C" else 1.0
            phat, bad = sample(pb, v_net, s_net, geoms[diff], sig, salt(f"ffm_{cov}{diff}"), cfg)
            out[f"ffm_{cov}{diff}"] = (phat, {**info, "family": "ffm_sde", "diffusion": diff,
                                              "sigma": sig, "noise_var_rate": 2 * sig ** 2 * m_inf,
                                              "nonfinite": bad})
        end = _np(integrate_sde(v_net, s_net, geoms["I"], pb.start, 0.0, n_steps=cfg.n_steps,
                                seed=seed, t_eps=cfg.sb_t_eps)[-1])
        out[f"ffm_{cov}off"] = (hard(pb, end), {**info, "family": "ffm_pfode",
                                                "diffusion": "off", "sigma": 0.0})
        return out
    if unit == "sf2m":
        sigma = 2.0 * width * pb.metric.eps_kernel
        hp = HParams(**{**cfg.arm_kw, "sigma": sigma, "sbm_iters": cfg.sb_iters})
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_arm("sf2m", pb.ts_base, hp, seed, cfg.device, cfg.dtype,
                          n_steps=BASE_N_STEPS)
        info = {"sigma_path": sigma, "train_seconds": time.time() - t0,
                "pi_entropy_frac": res.diag["ot"]["pi_entropy_frac"]}
        s_net, geom = res.aux["s_net"], res.aux["geom"]
        sig = sigma / np.sqrt(2.0)
        phat, bad = sample(pb, res.v_net, s_net, geom, sig, salt("sf2m"), cfg)
        end = _np(integrate_sde(res.v_net, s_net, geom, pb.start, 0.0, n_steps=cfg.n_steps,
                                seed=seed, t_eps=cfg.sb_t_eps)[-1])
        return {"sf2m": (phat, {**info, "family": "sf2m_sde", "diffusion": "I", "sigma": sig,
                                "noise_var_rate": 2 * sig ** 2, "nonfinite": bad}),
                "sf2m_off": (hard(pb, end), {**info, "family": "sf2m_pfode",
                                             "diffusion": "off", "sigma": 0.0})}
    knobs = CDC[unit] if unit in CDC else pb.point[unit]
    hp = HParams(**{**cfg.arm_kw, **knobs})
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_arm("cdcfm" if unit in CDC else unit, pb.ts_base, hp, seed, cfg.device,
                      cfg.dtype, n_steps=BASE_N_STEPS)
    end = _np(integrate_flow(res.v_net, pb.start, n_steps=BASE_N_STEPS)[-1])
    return {unit: (hard(pb, end), {"family": "baseline", "knobs": knobs,
                                   "check": float(np.abs(end[:pb.n_src] - res.traj[-1]).max())})}


def _fmt(x) -> str:
    return "nan" if x is None else f"{x:.3f}"


def ensure(cfg: Run, tags=None, seeds=None, root: str | None = None,
           force: bool = False) -> None:
    root = root_dir(cfg.smoke) if root is None else root
    tags = list(SHARES) if tags is None else list(tags)
    seeds = tuple((cfg.seeds if cfg.smoke else SEEDS) if seeds is None else seeds)
    budget, point = budget_of(cfg), tuned_point()
    warm = False
    for tag in tags:
        todo = [(s, u, w) for s in seeds for u, w in units()
                if force or any(not _fresh(record_path(root, tag, s, a, w), budget, point)
                                for a in UNIT_ARMS[u])]
        no_refs = any(not os.path.exists(ref_path(root, tag, a)) for a in REFS)
        if not todo and not no_refs:
            continue
        pb = problem(tag, cfg)
        if no_refs:
            write_refs(pb, root)
        if todo and not warm:
            warm = cuda_warm_up(pb.metric, pb.X0, pb.X1, cfg) or True
        for seed in sorted({s for s, _, _ in todo}):
            mine = [(u, w) for s, u, w in todo if s == seed]
            bb = (backbone(pb, seed, cfg, root) if any(u.startswith("ffm") for u, _ in mine)
                  else None)
            for u, w in mine:
                t0 = time.time()
                out = run_unit(pb, u, w, seed, bb, cfg)
                secs = time.time() - t0
                for arm, (phat, extra) in out.items():
                    rec = make_record(pb, phat, seed, arm, w, {**extra, "seconds": secs},
                                      budget)
                    path = record_path(root, tag, seed, arm, w)
                    os.makedirs(os.path.dirname(phat_path(path)), exist_ok=True)
                    np.save(phat_path(path), phat.astype(np.float32))
                    _dump(rec, path)
                    g = rec["groups"]["unres"]
                    print(f"[fate] sh{tag} s{seed} {arm:9s} w{w:<4g} unres brier2 "
                          f"{_fmt(g['brier2'])} p_low {_fmt(g['plow2'])} out "
                          f"{_fmt(g['outside'])} ({secs:.0f}s)", flush=True)


def rescore(cfg: Run, tags=None, root: str | None = None) -> None:
    root = root_dir(cfg.smoke) if root is None else root
    for tag in (list(SHARES) if tags is None else list(tags)):
        paths = [p for p in sorted(glob.glob(os.path.join(root, "records", f"sh{tag}_s*.json")))
                 if os.path.exists(phat_path(p))]
        if not paths:
            continue
        pb = scaffold(tag, cfg)
        write_refs(pb, root)
        for path in paths:
            with open(path) as fh:
                rec = json.load(fh)
            rec.update(score_part(pb, np.load(phat_path(path)).astype(np.float64)))
            _dump(rec, path)
        print(f"[fate] rescored sh{tag}: {len(paths)} records", flush=True)


def load(root: str | None = None, smoke: bool = False) -> list[dict]:
    root = root_dir(smoke) if root is None else root
    out = []
    for p in sorted(glob.glob(os.path.join(root, "records", "*.json"))):
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def long_frame(recs) -> pd.DataFrame:
    rows = [{"tag": r["tag"], "seed": r["seed"], "arm": r["arm"], "width": r["width"],
             "group": g, **m} for r in recs for g, m in r["groups"].items()]
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("tag", "arm", "group", "seed")]
    df[num] = df[num].astype(float)
    return df


def recall_frame(recs) -> pd.DataFrame:
    rows = []
    for r in recs:
        for name, v in (r.get("recall") or {}).items():
            if v is not None:
                rows.append({"tag": r["tag"], "seed": r["seed"], "arm": r["arm"],
                             "width": r["width"], "true arm": name, "recall": v[0],
                             "wrong arm": v[1], "outside": v[2]})
    return pd.DataFrame(rows)


def _pm(v, d: int = 3) -> str:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return f"{v.mean():.{d}f} ± {v.std():.{d}f}" if len(v) else "--"


def _arm_rows(df, arm: str, width: float):
    s = df[df["arm"] == arm]
    return s if arm in FIXED else s[np.isclose(s["width"], width)]


def frame_table4(recs, width: float = TUNED_WIDTH) -> pd.DataFrame:
    rf = recall_frame(recs)
    rows = {}
    for tag in sorted(set(rf["tag"])):
        for arm in MODEL_ARMS:
            s = _arm_rows(rf[rf["tag"] == tag], arm, width)
            for name in FATE_NAMES.values():
                t = s[s["true arm"] == name]
                if len(t):
                    rows[(SHARE_LABEL[tag], LABELS[arm], name)] = {
                        **{c: _pm(t[c], 1) for c in ("recall", "wrong arm", "outside")},
                        "seeds": len(t)}
    return pd.DataFrame(rows).T


def frame_calibration(recs, width: float = TUNED_WIDTH, groups=CAL_GROUPS,
                      metrics=CAL_METRICS) -> pd.DataFrame:
    lf = long_frame(recs)
    rows = {}
    for tag in sorted(set(lf["tag"])):
        for g in groups:
            sub = lf[(lf["tag"] == tag) & (lf["group"] == g)]
            for arm in LABELS:
                s = _arm_rows(sub, arm, width)
                if len(s):
                    rows[(SHARE_LABEL[tag], g, LABELS[arm])] = {m: _pm(s[m]) for m in metrics}
    return pd.DataFrame(rows).T


def frame_width(recs, groups=("unres", "held"), metrics=WIDTH_METRICS) -> pd.DataFrame:
    lf = long_frame(recs)
    rows = {}
    for tag in sorted(set(lf["tag"])):
        for g in groups:
            sub = lf[(lf["tag"] == tag) & (lf["group"] == g)]
            for m in metrics:
                for arm in STOCH_ARMS:
                    s = sub[sub["arm"] == arm]
                    if len(s):
                        rows[(SHARE_LABEL[tag], g, m, LABELS[arm])] = {
                            f"w={w:g}": _pm(s[np.isclose(s["width"], w)][m]) for w in WIDTHS}
    return pd.DataFrame(rows).T


def frame_proportion(recs, tag: str = "70",
                     groups=("unres", "trunk", "held_unres")) -> pd.DataFrame:
    lf = long_frame(recs)
    lf = lf[lf["tag"] == tag]
    rows = {}
    for g in groups:
        sub = lf[lf["group"] == g]
        for arm in LABELS:
            for w in ((0.0,) if arm in FIXED else WIDTHS):
                s = _arm_rows(sub, arm, w)
                if len(s):
                    rows[(g, LABELS[arm], "--" if arm in FIXED else f"{w:g}")] = {
                        "p(lower), 2-class": _pm(s["plow2"]),
                        "p(lower), 3-class": _pm(s["plow3"]),
                        "outside": _pm(s["outside"]),
                        "sq. dev. from Bayes": _pm(s["se_bayes"]),
                        "true lower fraction": f"{s['ybar'].mean():.3f}"}
    return pd.DataFrame(rows).T


def frame_wins(recs, groups=("unres", "held"), metrics=WIN_METRICS,
               pairs=COMPARE) -> pd.DataFrame:
    lf = long_frame(recs)
    lf = lf[lf["seed"] >= 0]
    rows = {}
    for tag in sorted(set(lf["tag"])):
        for w in WIDTHS:
            for g in groups:
                sub = lf[(lf["tag"] == tag) & (lf["group"] == g)]
                for m in metrics:
                    for a, b in pairs:
                        va = _arm_rows(sub, a, w).set_index("seed")[m]
                        vb = _arm_rows(sub, b, w).set_index("seed")[m]
                        dv = (va - vb).dropna()
                        if not len(dv):
                            continue
                        wa, wb = int((dv < -1e-12).sum()), int((dv > 1e-12).sum())
                        rows[(SHARE_LABEL[tag], f"{w:g}", g, m, f"{SHORT[a]} vs {SHORT[b]}")] = {
                            "A better": wa, "B better": wb, "ties": len(dv) - wa - wb,
                            "mean A-B": float(dv.mean()), "sd A-B": float(dv.std(ddof=0)),
                            "sign p": float(binomtest(wa, wa + wb).pvalue) if wa + wb else 1.0}
    return pd.DataFrame(rows).T


def frames(root: str | None = None, smoke: bool = False) -> dict:
    recs = load(root, smoke)
    return {"table4": frame_table4(recs), "calibration": frame_calibration(recs),
            "width": frame_width(recs), "proportion": frame_proportion(recs),
            "wins": frame_wins(recs)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.forksheet.calibration")
    ap.add_argument("--root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shares", nargs="*", default=list(SHARES), choices=list(SHARES))
    r.add_argument("--seeds", nargs="*", type=int, default=None)
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("rescore")
    s.add_argument("--shares", nargs="*", default=list(SHARES), choices=list(SHARES))
    p = sub.add_parser("report")
    p.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        ensure(Run.from_env(), args.shares, args.seeds, args.root, args.force)
        return
    if args.cmd == "rescore":
        rescore(Run.from_env(), args.shares, args.root)
        return
    with pd.option_context("display.width", 250, "display.max_rows", 2000,
                           "display.max_columns", 30):
        for name, df in frames(args.root, args.smoke).items():
            print(f"===== {name}")
            print(df.to_string())


if __name__ == "__main__":
    main()
