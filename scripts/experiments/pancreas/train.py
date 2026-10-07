"""The reported pancreas grid, written to disk once and read back after that.

One JSON record and one trajectory slab per ``(dim, arm, seed)``, plus the geodesic net
and Sinkhorn plan of the FFM cell, so a second pass at the same tuned point reloads
instead of retraining.  A record carries the point it was run at and the FFM point whose
geodesic cost judged its :math:`W_{2,F}`; either moving is what makes it stale.

The moment fidelities are computed here rather than by the reader, because they need the
whole integration and the slab keeps only what the figure draws.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from scripts.core.arms import HParams, run_arm
from scripts.core.metrics import distribution_metrics, mean_floor, traj_slice
from scripts.core.paths import experiment_root
from scripts.experiments.sheet.paper_focus import PATHB_NAME, QUAL_DET, QUAL_NOISY
from scripts.method import euclidean_w2, finsler_w2, path_a, path_b
from scripts.method.nets import Coupler, GeoPathNet

from .datasets import TRAIN_BINS, holdout_moment_fidelity

ARMS = ("ffm", "pathb", "cfm", "mfm_land", "curly")
ENGINE_ARMS = ("cfm", "mfm_land", "curly")
ARM_LABELS = {
    "ffm": f"{PATHB_NAME} (ours), {QUAL_DET}",
    "pathb": f"{PATHB_NAME} (ours), {QUAL_NOISY}",
    "cfm": "OT-CFM",
    "mfm_land": "MFM / LAND",
    "curly": "Curly-FM",
}
ARM_OF_LABEL = {v: k for k, v in ARM_LABELS.items()}
REPORT_ORDER = ("ffm", "pathb", "cfm", "mfm_land", "curly")

TEST_DIR = "test"
TRAJ_DIR = "trajectories"
NETS_DIR = "nets"
N_TRAJ_SAVED = 64


def bench_root(smoke: bool = False) -> str:
    return experiment_root("pancreas_bench", smoke)


def rec_path(root: str, dim: int, arm: str, seed: int) -> str:
    return os.path.join(root, TEST_DIR, f"d{dim}_{arm}_s{seed}.json")


def traj_path(root: str, dim: int, arm: str, seed: int) -> str:
    return os.path.join(root, TRAJ_DIR, f"d{dim}_{arm}_s{seed}.npz")


def state_path(root: str, dim: int, seed: int) -> str:
    return os.path.join(root, NETS_DIR, f"d{dim}_ffm_s{seed}.pt")


def save_trajectory(traj: np.ndarray, path: str) -> None:
    """Persist what the figure draws and the endpoints a backfill would need."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = traj.shape[1]
    keep = np.linspace(0, n - 1, min(N_TRAJ_SAVED, n)).astype(int)
    np.savez_compressed(path,
                        mid=traj_slice(traj, 0.5).astype(np.float32),
                        paths=traj[:, keep].astype(np.float32),
                        path_idx=keep,
                        start=traj[0].astype(np.float32),
                        end=traj[-1].astype(np.float32))


def save_state(phi, coupler, path: str) -> None:
    """The frozen interpolant and the coupling Path B and the judge both reuse."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"phi": phi.state_dict(),
                "A": coupler.X0.detach().cpu(),
                "B": coupler.X1.detach().cpu(),
                "pi": coupler.pi.detach().cpu()}, path)


def load_state(path: str, d: int, seed: int, cfg):
    """Rebuild ``(phi, coupler)`` from :func:`save_state`, frozen as training left them."""
    blob = torch.load(path, map_location=cfg.device, weights_only=True)
    phi = GeoPathNet(d, **cfg.net_kw).to(cfg.device, cfg.dtype)
    phi.load_state_dict(blob["phi"])
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    to = lambda x: x.to(cfg.device, cfg.dtype)
    return phi, Coupler(to(blob["A"]), to(blob["B"]), pi=to(blob["pi"]), seed=seed)


class Scorer:
    """Everything a row of the table needs, per dimension, plus its sampling floors.

    ``judge = (phi, metric)`` is the tuned FFM cell of this dimension and is fixed for the
    whole column, so every arm's :math:`W_{2,F}` is read off one cost function.
    """

    def __init__(self, cloud, judge, cfg, n_draw: int, seed: int = 0, wf: bool = True):
        self.mid = cloud.X[cloud.target_test]
        self.end = cloud.X[cloud.marginal(TRAIN_BINS[1])]
        self.judge, self.cfg, self.wf = judge, cfg, wf

        # Floors are size-matched to the ``n_draw`` particles every arm pushes and
        # averaged over draws rather than read off a single half-split, which would be
        # smaller than either side of the comparison it bounds and so read high
        # (:func:`scripts.core.metrics.floor_splits`).
        self.floors = {
            "intermediate": mean_floor(
                self.mid, n_draw,
                lambda a, b, k: {**distribution_metrics(a, b, seed=k),
                                 **({"W2F": finsler_w2(a, b, judge, cfg),
                                     "W2E": euclidean_w2(a, b)} if wf else {})}, seed),
            "endpoint": mean_floor(
                self.end, n_draw,
                lambda a, b, k: distribution_metrics(a, b, seed=k), seed)}

    def __call__(self, traj, seed: int = 0, wf: bool | None = None):
        pred = traj_slice(traj, 0.5)
        wf = self.wf if wf is None else wf
        finsler = {"W2F": finsler_w2(pred, self.mid, self.judge, self.cfg, seed=seed),
                   "W2E": euclidean_w2(pred, self.mid, seed=seed)} if wf else {}
        return {"intermediate": {**distribution_metrics(pred, self.mid, seed=seed),
                                 **finsler},
                "endpoint": distribution_metrics(traj_slice(traj, 1.0), self.end,
                                                 seed=seed),
                "floors": self.floors}


def points_at(tuned_dim: dict) -> dict:
    """``arm -> the hyper-parameters it is run at`` for one dimension."""
    return {"ffm": tuned_dim["ffm"], "pathb": tuned_dim["pathb"],
            "cfm": tuned_dim.get("cfm", {}), "mfm_land": tuned_dim["mfm_land"],
            "curly": tuned_dim["curly"]}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write(rec: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(rec), fh, indent=1, sort_keys=True)


def _fresh(path: str, override: dict, judge: dict) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        prior = json.load(fh)
    return prior.get("override") == override and prior.get("judge") == judge


def _tag(point: dict) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in sorted(point.items())) or "(nothing searched)"


def stage_test(root: str, dims, tuned: dict, seeds, cfg, *, clouds, tsets, tens, metric,
               sigma, n_steps: int, force: bool = False) -> None:
    """Fill the reported grid, running only the cells disk cannot already answer.

    The FFM state of ``seeds[0]`` is loaded or trained first in every dimension: it is
    both Path B's frozen input and the judge of the :math:`W_{2,F}` column, so no other
    arm can be scored before it exists.
    """
    for dim in dims:
        points = points_at(tuned[dim])
        judge_pt = points["ffm"]
        d = metric[dim].d
        X0_t, X1_t = tens[dim]
        print(f"===== d = {dim} =====")

        todo = [(arm, seed) for arm in ARMS for seed in seeds
                if force or not _fresh(rec_path(root, dim, arm, seed), points[arm],
                                       judge_pt)
                or (arm == "ffm" and not os.path.exists(state_path(root, dim, seed)))]
        if not todo:
            print(f"[test] d = {dim} complete on disk, {len(ARMS) * len(seeds)} cells")
            continue

        states, pending = {}, {}
        for seed in seeds:
            sp = state_path(root, dim, seed)
            rp = rec_path(root, dim, "ffm", seed)
            if _fresh(rp, judge_pt, judge_pt) and os.path.exists(sp) and not force:
                states[seed] = load_state(sp, d, seed, cfg)
                print(f"[test] skip d{dim}_ffm_s{seed}")
                continue
            t0 = time.time()
            det = path_a(metric[dim], X0_t, X1_t, seed, cfg)
            states[seed] = (det["phi"], det["coupler"])
            pending[seed] = (det["traj"], det["ot"], time.time() - t0)

        judge = (states[seeds[0]][0], metric[dim])
        score = Scorer(clouds[dim], judge, cfg, len(X0_t))

        for seed in seeds:
            if seed in pending:
                traj, ot, seconds = pending.pop(seed)
                _emit(root, dim, "ffm", seed, traj, score, clouds[dim], judge_pt,
                      judge_pt, seconds, n_steps, cfg, sigma=0.0, ot=ot)
                save_state(*states[seed], state_path(root, dim, seed))

            rp = rec_path(root, dim, "pathb", seed)
            if _fresh(rp, points["pathb"], judge_pt) and not force:
                print(f"[test] skip d{dim}_pathb_s{seed}")
            else:
                phi, coupler = states[seed]
                t0 = time.time()
                noisy = path_b(metric[dim], phi, coupler, X0_t, sigma[dim], seed, cfg)
                _emit(root, dim, "pathb", seed, noisy["traj"], score, clouds[dim],
                      points["pathb"], judge_pt, time.time() - t0, n_steps, cfg,
                      sigma=float(sigma[dim]), ot=None)

        for arm in ENGINE_ARMS:
            for seed in seeds:
                rp = rec_path(root, dim, arm, seed)
                if _fresh(rp, points[arm], judge_pt) and not force:
                    print(f"[test] skip d{dim}_{arm}_s{seed}")
                    continue
                hp = HParams(**cfg.arm_kw, **points[arm])
                t0 = time.time()
                res = run_arm(arm, tsets[dim], hp, seed, cfg.device, cfg.dtype,
                              n_steps=n_steps)
                _emit(root, dim, arm, seed, np.asarray(res.traj, dtype=np.float64), score,
                      clouds[dim], points[arm], judge_pt, time.time() - t0, n_steps, cfg,
                      sigma=0.0, ot=res.diag)


def _emit(root, dim, arm, seed, traj, score, cloud, override, judge, seconds, n_steps,
          cfg, *, sigma, ot) -> None:
    traj = np.asarray(traj, dtype=np.float64)
    metrics = score(traj, seed=seed)
    rec = {"dim": dim, "arm": arm, "seed": seed, "override": override, "judge": judge,
           "metrics": metrics, "sigma": sigma, "ot": ot,
           "moments": holdout_moment_fidelity(traj, cloud, seed=seed),
           "n_ref_mid": len(score.mid), "train_seconds": seconds, "n_steps": n_steps,
           "smoke": bool(cfg.smoke)}
    _write(rec, rec_path(root, dim, arm, seed))
    save_trajectory(traj, traj_path(root, dim, arm, seed))
    m = metrics["intermediate"]
    print(f"[test] d{dim}_{arm}_s{seed}  {_tag(override)}  mid W2 {m['W2']:.4f}  "
          f"W2F {m['W2F']:.4f}  end W2 {metrics['endpoint']['W2']:.4f}  ({seconds:.0f}s)")
