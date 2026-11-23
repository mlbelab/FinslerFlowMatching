from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ffm.core.arms import HParams
from ffm.core.metrics import traj_slice, wasserstein
from ffm.core.paths import PUBLIC_ROOT, data_dir, experiment_root
from ffm.core.protocol import TrainingSet
from ffm.core.transition import build_uniform_knn_transition, build_velocity_kernel
from scripts.erythroid.datasets import _flux_alignment
from scripts.erythroid.stage_bench import LABELS as STAGE_LABELS, train_curly_unit
from scripts.larry import bench as larry
from ffm.method import Run

NAME = "eb_bench"
URL = "https://raw.githubusercontent.com/KrishnaswamyLab/TrajectoryNet/master/data/eb_velocity_v5.npz"
SHA256 = "17623d16dd4fe7b679aad80916d130cb2863a4cd4ee742f139a3c2eed91ba230"
RAW = os.environ.get("EB_TNET", os.path.join(data_dir("eb_raw"), "eb_velocity_v5.npz"))
CACHE = os.path.join(data_dir("eb_cache"), "eb_tnet_d50.npz")
DIMS = (2, 5, 20, 50)
WHITEN = (5,)
LOW = {2: (2,), 5: (2, 20)}
HELD = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
UNITS = ("cfm", "mfm_land", "curly", "curly_unit", "ffm_velocity", "ffm_expression", "ffm_pancreas")
MFM_EB5 = {"land_kind": "land", "land_gamma": 0.125, "land_rho": 1e-3}
MFM_EB = {**MFM_EB5, "land_alpha": 2.0}
LABELS = {**STAGE_LABELS, "mfm_eb": "MFM, authors' EB setting (LAND, alpha = 2)"}
FULL = 100_000
REPS = 5


def root_dir(smoke: bool = False) -> str:
    return experiment_root(NAME, smoke)


def rec_path(root: str, d: int, k: int, method: str, seed: int) -> str:
    return os.path.join(root, "records", f"d{d}_k{k}_{method}_s{seed}.json")


def prepare() -> dict:
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    digest = hashlib.sha256(open(RAW, "rb").read()).hexdigest()
    assert digest == SHA256, digest
    with np.load(RAW, allow_pickle=False) as z:
        X, V, day = z["pcs"][:, :50], z["pcs_delta"][:, :50], z["sample_labels"]
        U, dU = z["phate"], z["delta_embedding"]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X.astype(np.float32), velocity=V.astype(np.float32), day=day.astype(np.int8),
                        phate=U.astype(np.float32), phate_velocity=dU.astype(np.float32))
    return {"cells": int(len(day)), "per_day": np.bincount(day).tolist(), "sha256": digest}


def cloud(d: int) -> dict:
    with np.load(CACHE) as z:
        X, V, day = z["X"][:, :d].astype(np.float64), z["velocity"][:, :d].astype(np.float64), z["day"]
        if d == 2:
            X, V = z["phate"].astype(np.float64), z["phate_velocity"].astype(np.float64)
    if d in WHITEN or d == 2:
        sd = X.std(0)
        X, V = (X - X.mean(0)) / sd, V / sd
    return {"d": d, "X": X, "V": V, "day": day}


def _tuned(d: int) -> dict:
    out = {}
    for name, (pkg, fk) in larry.SOURCES.items():
        with open(os.path.join(PUBLIC_ROOT, "scripts", pkg, "tuned.json")) as fh:
            t = json.load(fh)[str(d)]
        out[name] = {"ffm": t[fk], "pathb": t.get("pathb"), "curly": t["curly"],
                     "cfm": {"blur_frac": t["cfm"].get("blur_frac", HParams.blur_frac)}}
    return out


def points(d: int) -> dict:
    if d in (20, 50):
        pts = larry.transfer_points(d)
    else:
        tuned = [v for n in LOW[d] for v in _tuned(n).values()]
        pick = lambda arm, knob: larry._logmedian([v[arm][knob] for v in tuned if v[arm]])
        pan = [_tuned(n)["pancreas"]["ffm"] for n in LOW[d]]
        pts = {"ffm": {"rho_mult": pick("ffm", "rho_mult"), "lam_mult": pick("ffm", "lam_mult")},
               "pathb": {"width_mult": pick("pathb", "width_mult")},
               "curly": {"curly_alpha": pick("curly", "curly_alpha")},
               "cfm": {"blur_frac": pick("cfm", "blur_frac")},
               "sources": {"pancreas": {"ffm": {k: larry._logmedian([q[k] for q in pan]) for k in pan[0]}}}}
    if d in LOW:
        pts["mfm_land"] = dict(MFM_EB5)
    return pts


def training_set(c: dict, k: int, kernel: str) -> TrainingSet:
    idx = np.flatnonzero(np.isin(c["day"], (k - 1, k + 1)))
    X, V = c["X"][idx], c["V"][idx]
    P = build_uniform_knn_transition(X) if kernel == "expression" else build_velocity_kernel(X, V)
    day = c["day"][idx]
    return TrainingSet(name=f"EB-d{c['d']}-k{k}-{kernel}", idx=idx, X=X, P=sp.csr_matrix(P),
                       p0=day == k - 1, p1=day == k + 1, rho=0.0, holdout_band=None, p_variant=kernel,
                       diag={"n_train": int(len(idx))}, velocity=V)


def _w(A: np.ndarray, B: np.ndarray, seed: int = 0) -> dict:
    return {f"W{p}": wasserstein(A, B, p=p, max_n=FULL, seed=seed) for p in (1, 2)}


def references(c: dict, k: int) -> dict:
    H = c["X"][c["day"] == k]
    halves = []
    for s in range(REPS):
        perm = np.random.default_rng(s).permutation(len(H))
        halves.append(_w(H[perm[:len(H) // 2]], H[perm[len(H) // 2:]]))
    ts = training_set(c, k, "velocity")
    X, V = ts.X, ts.velocity
    axis = X[ts.p1].mean(0) - X[ts.p0].mean(0)
    cv = (V @ axis) / np.maximum(np.linalg.norm(V, axis=1) * np.linalg.norm(axis), 1e-300)
    return {"floor": {m: float(np.mean([h[m] for h in halves])) for m in ("W1", "W2")},
            "copy_source": _w(c["X"][c["day"] == k - 1], H), "copy_target": _w(c["X"][c["day"] == k + 1], H),
            "n_held": int(len(H)), "n_source": int(ts.p0.sum()),
            "cos_velocity_axis_source": float(cv[ts.p0].mean()), "cos_velocity_axis_target": float(cv[ts.p1].mean()),
            "frac_velocity_forward_source": float((cv[ts.p0] > 0).mean()),
            "cos_flux_velocity": float(_flux_alignment(ts.P, X, V))}


def evaluate(c: dict, k: int, traj: np.ndarray) -> dict:
    mid, end = traj_slice(traj, 0.5), traj_slice(traj, 1.0)
    out = _w(mid[np.isfinite(mid).all(1)], c["X"][c["day"] == k])
    out["W2_end"] = wasserstein(end[np.isfinite(end).all(1)], c["X"][c["day"] == k + 1], p=2, max_n=FULL)
    out["nonfinite"] = float(1 - np.isfinite(mid).all(1).mean())
    return out


def run_unit(d: int, k: int, unit: str, seed: int, cfg: Run, root: str, force: bool = False) -> None:
    if not force and all(os.path.exists(rec_path(root, d, k, m, seed)) for m in larry.unit_methods(unit)):
        print(f"[eb] skip d{d} k{k} {unit} s{seed}")
        return
    c = cloud(d)
    pts = points(d)
    if unit == "ffm_pancreas":
        pts["ffm"] = dict(pts["sources"]["pancreas"]["ffm"])
    kernel = larry.UNIT_KERNEL.get(unit, "velocity")
    ts = training_set(c, k, kernel)
    if unit == "curly_unit":
        out = train_curly_unit(ts, pts, seed, cfg)
    elif unit == "mfm_eb":
        out = {unit: larry.train_unit("mfm_land", ts, {**pts, "mfm_land": dict(MFM_EB)}, seed, cfg)["mfm_land"]}
    else:
        out = larry.train_unit(unit, ts, pts, seed, cfg)
    for method, (traj, extra) in out.items():
        rec = {"dim": d, "held": k, "method": method, "seed": seed, "kernel": kernel,
               "n_source": int(ts.p0.sum()), "n_target": int(ts.p1.sum()), **evaluate(c, k, traj), **extra}
        path = rec_path(root, d, k, method, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(path + ".tmp", path)
        print(f"[eb] d{d} k{k} {method:10s} s{seed}  W1 {rec['W1']:.4f}  W2 {rec['W2']:.4f}  "
              f"end {rec['W2_end']:.4f}  ({rec['seconds']:.0f}s)", flush=True)


def tasks() -> list[tuple[int, int, str, int]]:
    return ([(d, k, u, s) for d in DIMS for k in HELD for u in UNITS for s in SEEDS]
            + [(d, k, "mfm_eb", s) for d in LOW for k in HELD for s in SEEDS])


def frames(root: str | None = None, smoke: bool = False) -> dict:
    root = root_dir(smoke) if root is None else root
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "records", "*.json")))]
    refs = json.load(open(os.path.join(root, "references.json")))
    df = pd.DataFrame([{"d": r["dim"], "held": f"day {r['held']}", "method": LABELS[r["method"]], "seed": r["seed"],
                        "W1": r["W1"], "W2": r["W2"], "W2_end": r["W2_end"]} for r in recs])
    pm = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    out = {}
    for m in ("W1", "W2"):
        tab = df.groupby(["d", "method", "held"])[m].agg(pm).unstack("held")
        tab["average"] = df.groupby(["d", "method"])[m].agg(pm)
        out[m] = tab
    out["W2_end"] = df.groupby(["d", "method"])["W2_end"].agg(pm).to_frame()
    out["references"] = pd.DataFrame([{"d": int(d), "held": f"day {k}", "n held": v["n_held"], "n source": v["n_source"],
                                       **{f"{r} {m}": round(v[r][m], 3) for r in ("floor", "copy_source", "copy_target")
                                          for m in ("W1", "W2")},
                                       "cos v·axis source": round(v["cos_velocity_axis_source"], 3),
                                       "cos v·axis target": round(v["cos_velocity_axis_target"], 3),
                                       "cos flux·v": round(v["cos_flux_velocity"], 3)}
                                      for d, kv in refs.items() for k, v in kv.items()]).set_index(["d", "held"])
    paired = []
    for m in ("W1", "W2"):
        for b in ("ffm", "ffm_expr"):
            base = df[df["method"] == LABELS[b]].set_index(["d", "held", "seed"])[m]
            for o in ("cfm", "mfm_land", "mfm_eb", "curly_unit"):
                diff = (base - df[df["method"] == LABELS[o]].set_index(["d", "held", "seed"])[m]).dropna()
                for key, g in diff.groupby(level=[0]):
                    paired.append({"metric": m, "d": key, "FFM": LABELS[b], "vs": LABELS[o],
                                   "FFM minus other": round(float(g.mean()), 3),
                                   "FFM lower (days x seeds)": f"{int((g < 0).sum())}/{len(g)}"})
    out["paired"] = pd.DataFrame(paired).set_index(["metric", "d", "FFM", "vs"])
    out["points"] = pd.DataFrame({d: {k: v for k, v in points(d).items() if k != "sources"} for d in DIMS}).T
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=NAME)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--force", action="store_true")
    sub.add_parser("prep")
    sub.add_parser("count")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(), indent=1))
        return
    cfg = Run.from_env().but(ot_max_pts=larry.OT_MAX_PTS)
    root = root_dir(cfg.smoke)
    if a.cmd == "count":
        print(len(tasks()))
    elif a.cmd == "prep":
        os.makedirs(root, exist_ok=True)
        refs = {d: {k: references(cloud(d), k) for k in HELD} for d in DIMS}
        with open(os.path.join(root, "references.json"), "w") as fh:
            json.dump(refs, fh, indent=1)
        print(json.dumps(refs, indent=1))
    elif a.cmd == "run":
        d, k, unit, seed = tasks()[a.index]
        run_unit(d, k, unit, seed, cfg, root, a.force)
    else:
        for k, v in frames(smoke=cfg.smoke).items():
            print(v.to_string())


if __name__ == "__main__":
    main()
