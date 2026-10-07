"""Artefact I/O shared by the benchmark drivers.

One *cell* is ``(dataset, arm, seed)``, and every cell produces three artefacts:

    <out>/runs/<key>.json           config + protocol + every metric
    <out>/weights/<key>.pt          v_theta, s_phi, phi state dicts
    <out>/trajectories/<key>.npz    the (S+1, B, d) pushforward, float32

The weights and the trajectory are kept deliberately: re-integrating a saved network is
cheap while retraining is not, so every later evaluation — the tables, the figures, the
geometry comparison — reads these rather than fitting again.  The ``.pt`` also carries
the architecture (``width``, ``depth``, ``d``) and the full hyper-parameter record, so a
checkpoint is loadable without consulting this file.

This module used to hold a standalone driver as well, which fitted every arm using the
hyper-parameters tuned for the synthetic suite.  That suite is gone and so is its
``tuned_hparams.json``; :mod:`scripts.experiments.sheet.paper_run` is now the only driver, and it runs
its own tuning stage against the benchmark's own train/val split.  What is left here is
the part both the driver and the verification suite need.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from scripts.core.arms import N_STEPS, HParams

#: iteration budget for a wiring check.  Never use these for a reported number: the
#: interpolant is known to collapse at short budgets (see ``scripts.core.arms``), so a
#: short run measures the budget, not the method.
SMOKE_HPARAMS = {"phase1_iters": 200, "phase2_iters": 200, "sbm_iters": 200}


def cell_key(dataset: str, arm: str, seed: int) -> str:
    return f"{dataset}_{arm}_s{seed}"


# --------------------------------------------------------------------------- #
#  Artefacts
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _atomic_write_json(record: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(record), f, indent=1)
    os.replace(tmp, path)          # a killed job can never leave half a JSON behind


def save_weights(result, hp: HParams, dataset: str, arm: str, seed: int, d: int,
                 path: str) -> None:
    """Every network the arm produced, plus enough metadata to rebuild it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "dataset": dataset, "arm": arm, "seed": seed, "d": d,
        "hparams": hp.as_dict(), "n_steps": N_STEPS,
        "v_net": result.v_net.state_dict(),
    }
    for name in ("s_net", "phi"):
        net = result.aux.get(name)
        if net is not None and hasattr(net, "state_dict"):
            payload[name] = net.state_dict()
    torch.save(payload, path)


def _ensure(root: str, *sub: str) -> str:
    path = os.path.join(root, *sub)
    os.makedirs(path, exist_ok=True)
    return path
