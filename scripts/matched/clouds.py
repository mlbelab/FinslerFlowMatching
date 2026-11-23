from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from ffm.core.metrics import traj_slice
from ffm.core.selection import with_objective
from ffm.method import moments_from
from ffm.method.config import Run

from .problem import Problem

ERYTHROID_VAL_FRACTION = 0.15
ERYTHROID_SPLIT_SEED = 0


def build_erythroid(stage: str, cfg: Run, dim: int) -> Problem:
    from scripts.erythroid.datasets import (TRAIN_BINS, build_training_set,
                                                        load_cloud, split_shown)
    from scripts.erythroid.reference_release import release_wasserstein

    cloud = load_cloud(dim)
    i_train, i_val = split_shown(cloud, ERYTHROID_VAL_FRACTION, ERYTHROID_SPLIT_SEED)
    ts = build_training_set(cloud, subset=i_train if stage == "tune" else None)

    X = np.asarray(ts.X, dtype=np.float64)
    P = sp.csr_matrix(ts.P)
    b0_pts, D_pts = moments_from(P, X)

    X_t = cfg.tensor(X)
    X0_t = X_t[torch.as_tensor(ts.p0, device=cfg.device)]
    X1_t = X_t[torch.as_tensor(ts.p1, device=cfg.device)]

    i_end = i_val[cloud.bin_id[i_val] == TRAIN_BINS[1]]
    assert len(i_end) > 1, f"val slice of p1 is too small to score ({len(i_end)})"
    i_sel = cloud.target_val
    assert not np.intersect1d(i_sel, cloud.target_test).size
    mid_ref = cloud.X[cloud.target_test]
    end_ref = cloud.X[cloud.marginal(TRAIN_BINS[1])]

    def w(a, b, power=2):
        return release_wasserstein(torch.as_tensor(np.asarray(a), dtype=torch.float32),
                                   torch.as_tensor(np.asarray(b), dtype=torch.float32),
                                   power=power)

    def val(traj, seed=0):
        return with_objective({
            "W2_endpoint": w(traj[-1], cloud.X[i_end]),
            "W2_intermediate": w(traj[traj.shape[0] // 2], cloud.X[i_sel]),
        })

    def test(traj, seed=0):
        mid = traj_slice(traj, 0.5)
        return {"W2_mid": w(mid, mid_ref), "W1_mid": w(mid, mid_ref, power=1),
                "W2_end": w(traj_slice(traj, 1.0), end_ref)}

    return Problem(
        key=f"erythroid{dim}", stage=stage, X=X, P=P, b0_pts=b0_pts, D_pts=D_pts,
        X0_t=X0_t, X1_t=X1_t, val=val, test=test,
        refs={"W2_mid": mid_ref, "W2_end": end_ref},
        ruler=lambda a, b, k: w(a, b),
        diag={"n_train": int(len(ts.idx)), "n_p0": int(ts.p0.sum()),
              "n_p1": int(ts.p1.sum()), "n_mid_ref": int(len(mid_ref)),
              "n_end_ref": int(len(end_ref)), "d": int(X.shape[1]),
              "n_val_end": int(len(i_end)), "n_val_sel": int(len(i_sel)),
              "P_asymmetry": ts.diag["P_asymmetry"],
              "cos_J_velocity": ts.diag["cos_J_velocity"]})
