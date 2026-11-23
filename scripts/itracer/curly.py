from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.distance import cdist

from ffm.core.itracer_data import K_SUPPORT, velocity_alignment
from ffm.method import moments_from

from .datasets import DIMS, UMAP_DIM

SCALE = 10.0
SEEDS = (0, 1, 2)
SEEDS_D2 = tuple(range(12))
BASELINES = ("cfm", "mfm_land", "curly", "curly_p")
FFM_ARMS = ("ffm_expr", "ffm_velo", "ffm", "ffm_lin_velo")
ROWS = BASELINES + FFM_ARMS
KEYS = (("mid W2", "interpolation"), ("end W2", "endpoint"))


def embed_velocity(X, C, k: int = K_SUPPORT, scale: float = SCALE) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    d = cdist(X, X)
    np.fill_diagonal(d, np.inf)
    nbr = np.argsort(d, axis=1)[:, :k]
    D = X[nbr] - X[:, None, :]
    n = np.linalg.norm(D, axis=2, keepdims=True)
    U = np.divide(D, n, out=np.zeros_like(D), where=n > 0)
    W = np.exp(scale * np.take_along_axis(np.asarray(C, dtype=np.float64), nbr, axis=1))
    T = W / W.sum(axis=1, keepdims=True)
    return np.einsum("ik,ikd->id", T, U) - U.mean(axis=1)


def scvelo_velocity(ts) -> np.ndarray:
    return embed_velocity(ts.X, velocity_alignment(ts.idx))


def moment_velocity(ts) -> np.ndarray:
    return moments_from(ts.P, ts.X)[0]


VELOCITY_OF = {"curly": scvelo_velocity, "curly_p": moment_velocity}


def with_velocity(ts, arm: str):
    if arm not in VELOCITY_OF:
        return ts
    V, b0 = VELOCITY_OF[arm](ts), moment_velocity(ts)
    nv, nb = np.linalg.norm(V, axis=1), np.linalg.norm(b0, axis=1)
    cos = np.sum(V * b0, axis=1) / np.maximum(nv * nb, 1e-300)
    return dataclasses.replace(ts, velocity=V, diag={
        **ts.diag, "velocity_input": VELOCITY_OF[arm].__name__,
        "velocity_norm_mean": float(nv.mean()), "b0_norm_mean": float(nb.mean()),
        "velocity_cos_b0_mean": float(cos.mean())})


def _vals(recs, point, dim, arm, key, seeds):
    return [r[key]["W2"] for r in recs if r["point"] == point and r["dim"] == dim
            and r["arm"] == arm and r["seed"] in seeds]


def _fmt(v, n):
    if not v:
        return ""
    s = f"{np.mean(v):.4f} ± {np.std(v):.4f}"
    return s if len(v) == n else f"{s} (n={len(v)})"


def _floor(recs, dim, key):
    r = next((r for r in recs if r["dim"] == dim), None)
    return "" if r is None else f"{r['floors'][key]['W2']:.4f}"


def _columns(dims):
    import pandas as pd
    return pd.MultiIndex.from_tuples([(f"d = {d}", lab) for d in dims for lab, _ in KEYS])


def table_main(recs, point: str = "tuned", dims=DIMS, seeds=SEEDS):
    import pandas as pd
    from .bench import ARM_LABEL
    rows = {ARM_LABEL[a]: [_fmt(_vals(recs, point, d, a, key, seeds), len(seeds))
                           for d in dims for _, key in KEYS] for a in ROWS}
    rows["sampling floor"] = [_floor(recs, d, key) for d in dims for _, key in KEYS]
    return pd.DataFrame.from_dict(rows, orient="index", columns=_columns(dims))


def table_points(recs, dims=DIMS, seeds=SEEDS):
    import pandas as pd
    from .bench import ARM_LABEL
    plan = ([(a, p) for a in BASELINES for p in ("tuned", "erythroid")]
            + [(a, p) for a in FFM_ARMS for p in ("tuned", "erythroid", "default", "lodo")])
    rows = {(ARM_LABEL[a], p): [_fmt(_vals(recs, p, d, a, key, seeds), len(seeds))
                                for d in dims for _, key in KEYS] for a, p in plan}
    out = pd.DataFrame.from_dict(rows, orient="index", columns=_columns(dims))
    out.index = pd.MultiIndex.from_tuples(out.index, names=["arm", "point"])
    return out


def summary(recs, seeds=SEEDS_D2):
    import pandas as pd
    df = pd.DataFrame([{"point": r["point"], "dim": r["dim"], "arm": r["arm"],
                        "seed": r["seed"], "mid": r["interpolation"]["W2"],
                        "end": r["endpoint"]["W2"], "train_s": r["train_seconds"]}
                       for r in recs if r["seed"] in seeds])
    g = df.groupby(["point", "dim", "arm"])
    return pd.concat({"n": g.seed.count(), "mid": g.mid.mean(), "mid_sd": g.mid.std(ddof=0),
                      "end": g.end.mean(), "end_sd": g.end.std(ddof=0),
                      "train_s": g.train_s.mean()}, axis=1)


def ladder(rows=None, arms=("curly", "curly_p")):
    import pandas as pd
    from .tune import ROOT, SELECTION_KEY, arm_ladder, sweep_rows
    rows = sweep_rows(ROOT) if rows is None else rows
    out = {}
    for d in DIMS:
        for a in arms:
            lad, _ = arm_ladder(rows, d, a)
            out[(f"d = {d}", a)] = {r["point"]["curly_alpha"]: r[SELECTION_KEY] for r in lad}
    return pd.DataFrame(out).sort_index()


def frames(smoke: bool = False) -> dict:
    from .bench import POINTS, load_records
    recs = load_records(seeds=SEEDS_D2, points=POINTS, smoke=smoke)
    return {"table1": table_main(recs),
            "d2_12_seeds": table_main(recs, dims=(UMAP_DIM,), seeds=SEEDS_D2),
            "points": table_points(recs), "summary": summary(recs)}
