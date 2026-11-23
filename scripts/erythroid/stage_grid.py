from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from ffm.core.metrics import traj_slice, wasserstein
from scripts.larry import bench as larry
from scripts.pancreas.train import TS_WINDOW
from ffm.method import Run

from .stage_bench import (DIMS, EVAL_REPS, SEEDS, SOURCES, cloud, evaluate_window, n_eval, root_dir,
                          train_curly_unit, training_set, window_path)
from .stage_data import SPLITS

SPLIT = "long"
UNITS = ("cfm", "mfm_land", "curly_unit", "ffm_velocity", "ffm_expression")
NAMES = {"cfm": "OT-CFM", "mfm_land": "MFM/LAND", "curly_unit": "Curly-FM", "ffm": "FFM velocity",
         "ffm_noise": "FFM velocity + noise", "ffm_expr": "FFM expression", "ffm_expr_noise": "FFM expression + noise"}
METRICS = (("W2_mid", "midpoint W2"), ("W2_mean", "mean W2 [1/3,2/3]"))


def grid_path(root: str, d: int, method: str, seed: int) -> str:
    return os.path.join(root, "grid", f"{SPLIT}_d{d}_{method}_s{seed}.json")


def tasks() -> list:
    return [(d, u, s) for d in DIMS for u in UNITS for s in SEEDS]


def evaluate_grid(c: dict, traj: np.ndarray, seed: int) -> dict:
    H = c["X"][np.isin(c["stage"], SPLITS[SPLIT][1])]
    S = {f"{t:g}": traj_slice(traj, t) for t in TS_WINDOW}
    S = {k: s[np.isfinite(s).all(1)] for k, s in S.items()}
    w = {k: wasserstein(s, H, seed=seed) for k, s in S.items()}
    n = n_eval(c)
    w800 = {k: float(np.mean([wasserstein(s, H, max_n=n, seed=r) for r in range(EVAL_REPS)])) for k, s in S.items()}
    return {"grid": w, "W2_mid": w["0.5"], "W2_mean": float(np.mean(list(w.values()))),
            "rows": [int(round(t * (traj.shape[0] - 1))) for t in TS_WINDOW], "n_ref": int(len(H)),
            "nonfinite": int(sum(len(traj[0]) - len(s) for s in S.values())), "grid_n800": w800}


def run_unit(d: int, unit: str, seed: int, cfg: Run, root: str, force: bool = False) -> None:
    if not force and all(os.path.exists(grid_path(root, d, m, seed)) for m in larry.unit_methods(unit)):
        return
    c = cloud(SPLIT, d)
    pts = larry.transfer_points(d, SOURCES)
    kernel = larry.UNIT_KERNEL.get(unit, "velocity")
    ts = training_set(c, kernel)
    out = train_curly_unit(ts, pts, seed, cfg) if unit == "curly_unit" else larry.train_unit(unit, ts, pts, seed, cfg)
    for method, (traj, extra) in out.items():
        rec = {"split": SPLIT, "dim": d, "method": method, "seed": seed, "kernel": kernel,
               "n_source": int(ts.p0.sum()), "n_target": int(ts.p1.sum()), **evaluate_grid(c, traj, seed),
               "pooled": evaluate_window(c, traj), **extra}
        path = grid_path(root, d, method, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(path + ".tmp", path)
        print(f"[ery_grid] d{d} {method:14s} s{seed}  W2(1/2) {rec['W2_mid']:.4f}  mean {rec['W2_mean']:.4f}  "
              f"({rec['seconds']:.0f}s)", flush=True)


def seeds_frame(root: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "grid", f"{SPLIT}_d*_s*.json"))):
        r = json.load(open(p))
        old = json.load(open(window_path(root, SPLIT, r["dim"], r["method"], r["seed"])))
        rows.append({"d": r["dim"], "method": r["method"], "seed": r["seed"], "W2_mid": r["W2_mid"],
                     "W2_mean": r["W2_mean"], **{f"W2({k})": v for k, v in r["grid"].items()},
                     "W2_mean_n800": float(np.mean(list(r["grid_n800"].values()))),
                     "W2_half": r["pooled"]["W2_half"], "W2_window": r["pooled"]["W2_window"],
                     "reproduces_stored": bool(r["pooled"]["W2_window_reps"] == old["W2_window_reps"]
                                               and r["pooled"]["W2_half_reps"] == old["W2_half_reps"]),
                     "nonfinite": r["nonfinite"]})
    df = pd.DataFrame(rows)
    df["order"] = df["method"].map({m: i for i, m in enumerate(NAMES)})
    return df.sort_values(["d", "order", "seed"]).drop(columns="order").reset_index(drop=True)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k, _ in METRICS] + ["W2_mean_n800", "W2_half", "W2_window"]
    g = df.groupby(["d", "method"])[keys]
    out = pd.concat({"mean": g.mean(), "sd": g.std(ddof=0)}, axis=1).swaplevel(0, 1, axis=1)
    return out[[(k, s) for k in keys for s in ("mean", "sd")]].assign(n=g.size())


def table(s: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({f"d={d} {name}": {NAMES[m]: f"{s.loc[(d, m), (k, 'mean')]:.3f} ± {s.loc[(d, m), (k, 'sd')]:.3f}"
                                        for m in NAMES if (d, m) in s.index}
                      for d in DIMS for k, name in METRICS})
    return t.rename_axis("Method")


def latex(s: pd.DataFrame) -> str:
    cols = [(d, k) for d in DIMS for k, _ in METRICS]
    best = {(d, k): s.loc[d][(k, "mean")].idxmin() for d, k in cols}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{$d=20$} & \multicolumn{2}{c}{$d=50$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & midpoint $W_2$ & mean $W_2$ $[1/3,2/3]$ & midpoint $W_2$ & mean $W_2$ $[1/3,2/3]$ \\",
             r"\midrule"]
    for m, name in NAMES.items():
        cells = [f"{s.loc[(d, m), (k, 'mean')]:.3f} $\\pm$ {s.loc[(d, m), (k, 'sd')]:.3f}" for d, k in cols]
        cells = [f"\\textbf{{{x}}}" if best[c] == m else x for x, c in zip(cells, cols)]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def frames(root: str | None = None, smoke: bool = False) -> dict:
    df = seeds_frame(root_dir(smoke) if root is None else root)
    s = summary(df)
    return {"table": table(s), "summary": s, "seeds": df, "latex": latex(s)}


def save(root: str) -> str:
    f = frames(root)
    out = os.path.join(root, "grid_summary")
    os.makedirs(out, exist_ok=True)
    f["seeds"].to_csv(os.path.join(out, "seeds.csv"), index=False)
    f["summary"].to_csv(os.path.join(out, "summary.csv"))
    f["table"].to_csv(os.path.join(out, "table.csv"))
    with open(os.path.join(out, "table.tex"), "w") as fh:
        fh.write(f["latex"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="erythroid.stage_grid")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--force", action="store_true")
    sub.add_parser("count")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    cfg = Run.from_env().but(ot_max_pts=larry.OT_MAX_PTS)
    root = root_dir(cfg.smoke)
    if a.cmd == "count":
        print(len(tasks()))
    elif a.cmd == "run":
        d, u, s = tasks()[a.index]
        run_unit(d, u, s, cfg, root, a.force)
    else:
        print(frames(root)["table"].to_string())
        print(f"wrote {save(root)}")


if __name__ == "__main__":
    main()
