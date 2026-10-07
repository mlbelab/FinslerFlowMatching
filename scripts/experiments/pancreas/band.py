"""Midpoint recovery along ``t``, as a function of how much of the band the geometry sees.

Two questions in one grid, both about the *reported* claim that an arm recovers the
withheld middle marginal at :math:`t = 1/2`.

**When does an arm actually pass the band?**  Every reported number is read at one model
time, and a cloud that sweeps past the transition slightly early or slightly late is
scored as if it had missed it.  So each cell records :math:`W_2` against the held band on
the :data:`TS_CURVE` grid, and the report reads the :data:`TS_REPORT` window,
:math:`t = 1/3` to :math:`t = 2/3`, printing the argmin beside the value at
:math:`t = 1/2`.  The argmin is an **oracle**
— it reads the answer to choose its time — and is printed as a diagnostic, never as a
score.

**What changes when the band stops being withheld?**  :data:`FRACS` shows the geometry
0 %, 5 % and 35 % of the middle marginal.  Those cells enter ``P`` and therefore the
metric, but they are in neither coupling marginal, so what the fraction buys is
neighbourhood structure across the gap and not a target to transport to.  Scoring is
always against the **rest** of the band, so the reference shrinks as the fraction grows
and the three columns of a row are not one number measured three times.  OT-CFM is the
control: it never reads ``P``, so its row is what the split alone does.

Runs at every dimension the caller asks for, one cached JSON per
``(dim, frac, arm, seed)``, so a re-run costs only the cells that are not on disk.

    python -m scripts.experiments.pancreas.band run
    python -m scripts.experiments.pancreas.band report
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scripts.core.arms import HParams, run_arm
from scripts.core.metrics import (distribution_metrics, floor_splits, mean_floor,
                                  traj_slice, wasserstein)
from scripts.core.paths import experiment_root, rel, table_file
from scripts.core.protocol import TrainingSet
from scripts.core.transition import (VELOCITY_SOFTMAX_SCALE, asymmetry_scale,
                                     build_velocity_kernel)
from scripts.method import FWFactory, Run, moments_from, path_a, path_b, sigma_for

from .datasets import (HOLDOUT_BIN, N_HOLDOUT_STRATA, P_K, TRAIN_BINS, load_cloud)

KEY = "pancreas_band"
CELL_DIR = "cells"

#: fraction of the withheld middle marginal handed to the geometry.
FRACS = (0.0, 0.05, 0.35)

#: model times every cell records.  Kept at the original quarter-to-three-quarter grid so
#: the cached records stay valid; the report reads the narrower :data:`TS_REPORT` window.
TS_CURVE = tuple(round(0.25 + 0.05 * k, 2) for k in range(11))

#: the reported window, t = 1/3 to t = 2/3 -- the stored times that fall inside it.
TS_REPORT = tuple(t for t in TS_CURVE if 1.0 / 3.0 - 1e-9 <= t <= 2.0 / 3.0 + 1e-9)

T_MID = 0.5

ARMS = ("ffm", "pathb", "cfm", "mfm_land", "curly")
ARM_LABELS = {"ffm": "FFM (ours), deterministic", "pathb": "FFM (ours), with noise",
              "cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly": "Curly-FM"}
ENGINE_ARMS = ("cfm", "mfm_land", "curly")

BASE_N_STEPS = 102

#: the band split is a property of the dataset, not of a model seed: two arms compared at
#: one fraction must be scored against the same held cells.
SPLIT_SEED = 0

TABLE_STEM = "pancreas_band"


def root_dir(smoke: bool = False) -> str:
    return experiment_root(KEY, smoke)


ROOT = root_dir()


def cell_path(root: str, dim: int, frac: float, arm: str, seed: int) -> str:
    return os.path.join(root, CELL_DIR, f"d{dim}_f{frac:g}_{arm}_s{seed}.json")


def cells(dims, seeds) -> list[tuple[int, float, str, int]]:
    return [(d, f, a, s) for d in dims for f in FRACS for a in ARMS for s in seeds]


# --------------------------------------------------------------------------- #
#  The split, and the training set it implies
# --------------------------------------------------------------------------- #
def band_split(cloud, frac: float, split_seed: int = SPLIT_SEED):
    """``(visible, held)`` global indices partitioning the withheld marginal.

    Stratified by latent-time decile within the band, the rule
    :func:`~scripts.experiments.pancreas.datasets.split_holdout` uses, so a 5 % slice is a
    thin sample of the whole transition rather than one edge of it.
    """
    assert 0.0 <= frac < 1.0, f"frac must be in [0, 1), got {frac}"
    tgt = cloud.target
    if frac == 0.0:
        return tgt[:0], tgt

    rng = np.random.default_rng(split_seed)
    order = np.argsort(cloud.latent_time[tgt], kind="stable")
    picks = []
    for stratum in np.array_split(order, N_HOLDOUT_STRATA):
        n_v = int(round(frac * len(stratum)))
        if n_v:
            picks.append(rng.choice(stratum, size=n_v, replace=False))
    seen = np.zeros(len(tgt), dtype=bool)
    if picks:
        seen[np.concatenate(picks)] = True
    return tgt[seen], tgt[~seen]


def training_set(cloud, visible: np.ndarray, k: int = P_K,
                 softmax_scale: float = VELOCITY_SOFTMAX_SCALE) -> TrainingSet:
    """``P`` rebuilt on the two shown marginals **plus** the visible band cells.

    Deliberately not :func:`~scripts.experiments.pancreas.datasets.build_training_set`,
    which asserts no middle-marginal cell is present -- that assert is the leak guard for
    the reported protocol and this module is the controlled violation of it.  The visible
    cells are nodes of ``P`` and nothing else: ``p0`` and ``p1`` are still the two training
    marginals, so no arm is given a point to transport to inside the band.
    """
    idx = np.sort(np.concatenate([cloud.shown, np.asarray(visible, dtype=np.int64)]))
    X, V = cloud.X[idx], cloud.velocity[idx]
    P = build_velocity_kernel(X, V, k=k, softmax_scale=softmax_scale)

    bins = cloud.bin_id[idx]
    p0, p1 = bins == TRAIN_BINS[0], bins == TRAIN_BINS[1]
    assert p0.sum() and p1.sum(), "a training marginal came out empty"

    diag = {"n_train": int(len(idx)), "n_visible_band": int((bins == HOLDOUT_BIN).sum()),
            "P_asymmetry": float(asymmetry_scale(P))}
    return TrainingSet(name=f"Pancreas-d{cloud.dim}-band", idx=idx, X=X,
                       P=sp.csr_matrix(P), p0=p0, p1=p1, rho=0.0,
                       holdout_band=HOLDOUT_BIN, p_variant="velocity_kernel",
                       diag=diag, velocity=V)


# --------------------------------------------------------------------------- #
#  Scoring
# --------------------------------------------------------------------------- #
def _nearest(Y, pool_X, chunk=1024):
    pool_sq = (pool_X ** 2).sum(axis=1)
    out = [(pool_sq - 2.0 * Y[s:s + chunk] @ pool_X.T).argmin(axis=1)
           for s in range(0, len(Y), chunk)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _composition(labels, order):
    v = pd.Series(labels).value_counts(normalize=True)
    return np.array([float(v.get(k, 0.0)) for k in order])


class CurveScorer:
    """:math:`W_2` and celltype TV against the held band, at every time in the curve."""

    def __init__(self, cloud, held: np.ndarray, n_draw: int, seed: int = SPLIT_SEED):
        self.pool_X, self.pool_y = cloud.X, cloud.celltype
        self.order = sorted(set(cloud.celltype))
        self.ref = cloud.X[held]
        self.end = cloud.X[cloud.marginal(TRAIN_BINS[1])]
        self.true_comp = _composition(cloud.celltype[held], self.order)

        # Size-matched to the n_draw particles each arm pushes and averaged over draws,
        # rather than read off one half-split -- a half of this band is smaller than
        # either side of the comparison it bounds, so a half-split floor reads high
        # (:func:`scripts.core.metrics.floor_splits`).  The
        # TV floor keeps its original construction: the scored draw is dropped from the
        # pool its labels are read off, so it never votes on its own composition.
        rows = []
        for k, (a_i, b_i) in enumerate(floor_splits(len(self.ref), n_draw, seed)):
            drop = np.zeros(len(cloud.X), dtype=bool)
            drop[held[a_i]] = True
            rows.append({
                "W2": float(wasserstein(self.ref[a_i], self.ref[b_i], seed=k)),
                "TV": self._tv(self.ref[a_i],
                               pool=(cloud.X[~drop], cloud.celltype[~drop]))})
        self.floors = {
            **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
            "endpoint_W2": float(mean_floor(
                self.end, n_draw,
                lambda a, b, k: distribution_metrics(a, b, seed=k), seed)["W2"]),
        }

    def _tv(self, pred, pool=None):
        pX, py = pool if pool is not None else (self.pool_X, self.pool_y)
        c = _composition(py[_nearest(pred, pX)], self.order)
        return 0.5 * float(np.abs(c - self.true_comp).sum())

    def __call__(self, traj, seed: int = 0) -> dict:
        curve = {f"{t:g}": {"W2": float(wasserstein(traj_slice(traj, t), self.ref,
                                                    seed=seed)),
                            "TV": self._tv(traj_slice(traj, t))}
                 for t in TS_CURVE}
        return {"curve": curve,
                "endpoint_W2": float(wasserstein(traj_slice(traj, 1.0), self.end,
                                                 seed=seed)),
                "floors": self.floors}


# --------------------------------------------------------------------------- #
#  Running
# --------------------------------------------------------------------------- #
def _budget(cfg: Run) -> dict:
    return {"geo_iters": cfg.geo_iters, "cfm_iters": cfg.cfm_iters,
            "sb_iters": cfg.sb_iters, "n_steps": cfg.n_steps,
            "base_n_steps": BASE_N_STEPS, "width": cfg.net_width,
            "depth": cfg.net_depth, "ts": list(TS_CURVE)}


def _fresh(path: str, point: dict, budget: dict) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        prior = json.load(fh)
    return prior.get("point") == point and prior.get("budget") == budget


def _point(tuned_dim: dict, arm: str) -> dict:
    if arm in ("ffm", "pathb"):
        return {"ffm": dict(tuned_dim["ffm"]), "pathb": dict(tuned_dim["pathb"])}
    return dict(tuned_dim[arm])


def missing(cfg: Run, tuned: dict, dims, root: str | None = None, seeds=None):
    root = root_dir(cfg.smoke) if root is None else root
    seeds = cfg.seeds if seeds is None else seeds
    budget = _budget(cfg)
    return [c for c in cells(dims, seeds)
            if not _fresh(cell_path(root, *c), _point(tuned[c[0]], c[2]), budget)]


def ensure(cfg: Run, tuned: dict, dims, root: str | None = None, seeds=None,
           force: bool = False, verbose: bool = False) -> dict:
    """Every ``(dim, frac, arm, seed)`` cell, training only what disk cannot answer."""
    root = root_dir(cfg.smoke) if root is None else root
    seeds = tuple(cfg.seeds if seeds is None else seeds)
    dims = tuple(dims)
    budget = _budget(cfg)
    todo = [c for c in cells(dims, seeds)
            if force or not _fresh(cell_path(root, *c), _point(tuned[c[0]], c[2]), budget)]
    if todo:
        _fill(todo, cfg, tuned, root, budget, verbose)
    return load_records(root, dims, seeds)


def _fill(todo, cfg: Run, tuned: dict, root: str, budget: dict, verbose: bool) -> None:
    by_split: dict[tuple[int, float], list] = {}
    for c in todo:
        by_split.setdefault((c[0], c[1]), []).append(c)

    for (dim, frac), group in sorted(by_split.items()):
        cloud = load_cloud(dim)
        visible, held = band_split(cloud, frac)
        ts = training_set(cloud, visible)
        X = np.asarray(ts.X, dtype=np.float64)
        score = CurveScorer(cloud, held, len(ts.p0))

        need_geom = any(a in ("ffm", "pathb") for _, _, a, _ in group)
        metric = sigma = X0_t = X1_t = None
        if need_geom:
            b0_pts, D_pts = moments_from(ts.P, X)
            metric = FWFactory(X, D_pts, b0_pts, cfg, **tuned[dim]["ffm"]).base
            sigma = sigma_for(metric, tuned[dim]["pathb"]["width_mult"])
            X0_t, X1_t = cfg.tensor(X[ts.p0]), cfg.tensor(X[ts.p1])

        backbone: dict[int, tuple] = {}
        for dim_, frac_, arm, seed in sorted(group, key=lambda c: (c[3], ARMS.index(c[2]))):
            t0 = time.time()
            if arm in ("ffm", "pathb"):
                if seed not in backbone:
                    det = path_a(metric, X0_t, X1_t, seed, cfg)
                    backbone[seed] = (det["phi"], det["coupler"], det["traj"],
                                      time.time() - t0)
                phi, coupler, det_traj, det_s = backbone[seed]
                if arm == "ffm":
                    traj, train_s = det_traj, det_s
                else:
                    t0 = time.time()
                    traj = path_b(metric, phi, coupler, X0_t, sigma, seed, cfg)["traj"]
                    train_s = time.time() - t0
            else:
                hp = HParams(**cfg.arm_kw, **tuned[dim][arm])
                traj = np.asarray(run_arm(arm, ts, hp, seed, cfg.device, cfg.dtype,
                                          n_steps=BASE_N_STEPS).traj, dtype=np.float64)
                train_s = time.time() - t0

            rec = {"dim": dim, "frac": frac, "arm": arm, "seed": seed,
                   "point": _point(tuned[dim], arm), "budget": budget,
                   "n_visible_band": int(len(visible)), "n_held": int(len(held)),
                   "n_train": int(ts.diag["n_train"]),
                   "P_asymmetry": ts.diag["P_asymmetry"],
                   "sigma": float(sigma) if arm == "pathb" else 0.0,
                   "train_seconds": float(train_s), **score(traj, seed=seed)}
            path = cell_path(root, dim, frac, arm, seed)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(rec, fh, indent=1, sort_keys=True)
            if verbose:
                print(f"[band] d{dim} f{frac:g} {arm:9s} s{seed}  "
                      f"mid W2 {rec['curve'][f'{T_MID:g}']['W2']:.4f}  "
                      f"({train_s:.0f}s)", flush=True)


def load_records(root: str | None = None, dims=(2, 10, 20, 50), seeds=(0, 1, 2)) -> dict:
    root = ROOT if root is None else root
    out = {}
    for c in cells(dims, seeds):
        p = cell_path(root, *c)
        if os.path.exists(p):
            with open(p) as fh:
                out[c] = json.load(fh)
    return out


#: where :func:`export_band_json` writes the per-(dim, frac) roll-up the notebook figure
#: reads.  One aggregate file per band split, derived from the cells above.
BAND_JSON_DIR = os.path.join("data", "band")


def export_band_json(recs: dict, out_dir: str = BAND_JSON_DIR) -> list[str]:
    """Roll the per-cell records up into one ``band_d{dim}_f{frac}.json`` per split.

    The figure wants ``(dim, frac) -> arm -> seed`` while the fan-out writes one file per
    cell, so something has to do this transpose.  It lives here, next to the cells it
    reads, so that ``data/band`` is always derived from run records rather than being a
    checked-in roll-up with no code that emits it.  The tuned point each cell was
    actually run at is copied into the file, and :func:`_check_export_tuned` asserts it
    against ``tuned.json``, so a roll-up cannot quietly outlive the point it came from.

    Only the fields the figure reads are emitted (``W2`` at the reported mid time, and
    the floor).  Per-time curves and TV stay in the cells; use :func:`load_records`.
    """
    tuned = _tuned()
    by_split: dict[tuple[int, float], dict] = {}
    for (dim, frac, arm, seed), r in recs.items():
        _check_export_tuned(dim, arm, r.get("point"), tuned)
        blk = by_split.setdefault((dim, frac), {"arms": {}, "dim": dim, "frac": frac})
        blk["arms"].setdefault(arm, {})[str(seed)] = {
            "held": {"W2": float(r["curve"][f"{T_MID:g}"]["W2"])},
            "endpoint": {"W2": float(r["endpoint_W2"])},
            "floors": {"held": {"W2": float(r["floors"]["W2"])},
                       "endpoint": {"W2": float(r["floors"]["endpoint_W2"])}},
            "train_seconds": float(r["train_seconds"]),
        }
        blk["n_held"] = int(r["n_held"])
        blk["seeds"] = sorted({int(s) for s in blk["arms"][arm]})
        blk["t_mid"] = T_MID
        blk["tuned"] = tuned[dim]

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for (dim, frac), blk in sorted(by_split.items()):
        p = os.path.join(out_dir, f"band_d{dim}_f{frac:g}.json")
        with open(p, "w") as fh:
            json.dump(blk, fh, indent=1, sort_keys=True)
        written.append(p)
    return written


def _check_export_tuned(dim: int, arm: str, point, tuned: dict) -> None:
    """Every exported cell must have been run at the point ``tuned.json`` selects."""
    want = _point(tuned[dim], arm)
    assert point == want, (
        f"band cell d={dim} {arm} was run at {point}, but tuned.json selects {want}; "
        "the roll-up would disagree with the reported hyper-parameters -- re-run the "
        "fan-out for this dimension rather than exporting a stale point")


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
def _dims_of(recs) -> list[int]:
    return sorted({d for d, _, _, _ in recs})


def _seeds_of(recs) -> list[int]:
    return sorted({s for _, _, _, s in recs})


def _across_seeds(recs, dim, frac, arm, t, field) -> list[float]:
    return [r["curve"][f"{t:g}"][field] for (d, f, a, _), r in recs.items()
            if (d, f, a) == (dim, frac, arm)]


def _mean(recs, dim, frac, arm, t, field) -> float:
    v = _across_seeds(recs, dim, frac, arm, t, field)
    return float(np.mean(v)) if v else float("nan")


def _sd(recs, dim, frac, arm, t, field) -> float:
    v = _across_seeds(recs, dim, frac, arm, t, field)
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _floor(recs, dim, frac, field) -> float:
    v = [r["floors"][field] for (d, f, _, _), r in recs.items() if (d, f) == (dim, frac)]
    return float(np.mean(v)) if v else float("nan")


def _n_held(recs, dim, frac) -> float:
    v = [r["n_held"] for (d, f, _, _), r in recs.items() if (d, f) == (dim, frac)]
    return float(np.mean(v)) if v else float("nan")


def frame_summary(recs: dict) -> pd.DataFrame:
    """One row per ``(dim, frac, arm)``: the reported time, the oracle time, the floor."""
    rows, index = [], []
    for dim in _dims_of(recs):
        for frac in FRACS:
            for arm in ARMS:
                w = np.array([_mean(recs, dim, frac, arm, t, "W2") for t in TS_REPORT])
                if np.all(np.isnan(w)):
                    continue
                j = TS_REPORT.index(T_MID)
                i = int(np.nanargmin(w))
                index.append((dim, f"{100 * frac:g}%", ARM_LABELS[arm]))
                rows.append({"mid W2": w[j],
                             "mid sd": _sd(recs, dim, frac, arm, T_MID, "W2"),
                             "argmin t": TS_REPORT[i], "W2 there": w[i],
                             "W2 floor": _floor(recs, dim, frac, "W2"),
                             "n held": _n_held(recs, dim, frac),
                             "n seeds": len(_across_seeds(recs, dim, frac, arm,
                                                          T_MID, "W2"))})
    idx = pd.MultiIndex.from_tuples(index, names=["d", "band shown", "arm"])
    return pd.DataFrame(rows, index=idx)


def frame_curve(recs: dict, field: str = "W2") -> pd.DataFrame:
    """``(dim, frac, arm)`` by model time."""
    rows, index = [], []
    for dim in _dims_of(recs):
        for frac in FRACS:
            for arm in ARMS:
                v = {t: _mean(recs, dim, frac, arm, t, field) for t in TS_REPORT}
                if np.all(np.isnan(list(v.values()))):
                    continue
                index.append((dim, f"{100 * frac:g}%", ARM_LABELS[arm]))
                rows.append(v)
            v = {t: _floor(recs, dim, frac, field) for t in TS_REPORT}
            if not np.isnan(v[TS_REPORT[0]]):
                index.append((dim, f"{100 * frac:g}%", "two-sample floor"))
                rows.append(v)
    idx = pd.MultiIndex.from_tuples(index, names=["d", "band shown", "arm"])
    return pd.DataFrame(rows, index=idx)


def frame_access(recs: dict) -> pd.DataFrame:
    """``mid W2`` against band access, and the same divided by OT-CFM's."""
    rows, index = [], []
    for dim in _dims_of(recs):
        for arm in ARMS:
            v = {f"{100 * f:g}%": _mean(recs, dim, f, arm, T_MID, "W2") for f in FRACS}
            if np.all(np.isnan(list(v.values()))):
                continue
            base = {f"{100 * f:g}%": _mean(recs, dim, f, "cfm", T_MID, "W2")
                    for f in FRACS}
            sd = {f"{100 * f:g}% sd": _sd(recs, dim, f, arm, T_MID, "W2") for f in FRACS}
            index.append((dim, ARM_LABELS[arm]))
            rows.append({**v, **sd, **{f"{k} / OT-CFM": v[k] / base[k] for k in v}})
        index.append((dim, "two-sample floor"))
        rows.append({f"{100 * f:g}%": _floor(recs, dim, f, "W2") for f in FRACS})
    idx = pd.MultiIndex.from_tuples(index, names=["d", "arm"])
    return pd.DataFrame(rows, index=idx)


FRAME_NAMES = ("summary", "access", "curve_W2")


def frames(recs: dict):
    return (frame_summary(recs), frame_access(recs), frame_curve(recs, "W2"))


def header(recs: dict) -> str:
    return (f"===== Pancreas band access, d = {_dims_of(recs)}, "
            f"seeds {tuple(_seeds_of(recs))}, "
            f"t in [{TS_REPORT[0]:g}, {TS_REPORT[-1]:g}]\n"
            f"the visible band cells enter P and the metric, never a coupling marginal; "
            f"scoring is against the rest of the band")


def render(recs: dict) -> str:
    out = [header(recs)]
    for f in frames(recs):
        out.append(f.round(4).to_string())
    return "\n\n".join(out) + "\n"


def save_tables(recs: dict, smoke: bool = False) -> str:
    stem = table_file(TABLE_STEM, smoke)[:-len(".json")]
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".txt", "w") as fh:
        fh.write(render(recs))
    for name, f in zip(FRAME_NAMES, frames(recs)):
        f.to_csv(f"{stem}_{name}.csv")
    return stem + ".txt"


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _tuned() -> dict:
    from .noise import tuned_point
    return {d: tuned_point(d) for d in (2, 10, 20, 50)}


def cmd_run(args) -> None:
    cfg = Run.from_env()
    dims = tuple(int(d) for d in args.dims.split(","))
    recs = ensure(cfg, _tuned(), dims, seeds=args.seeds, force=args.force, verbose=True)
    print(render(recs))
    print(f"wrote {rel(save_tables(recs, cfg.smoke))}")


def cmd_report(args) -> None:
    dims = tuple(int(d) for d in args.dims.split(","))
    recs = load_records(root_dir(args.smoke), dims, args.seeds)
    assert recs, "no records; run first"
    print(render(recs))
    if args.save:
        print(f"wrote {rel(save_tables(recs, args.smoke))}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dims", default="2,10,20,50")
    r.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")),
                   default=None)
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("report")
    q.add_argument("--dims", default="2,10,20,50")
    q.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")),
                   default=(0, 1, 2))
    q.add_argument("--smoke", action="store_true")
    q.add_argument("--save", action="store_true")
    q.set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
