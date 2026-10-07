"""Readers over the records :mod:`~scripts.experiments.pancreas.train` leaves on disk.

:func:`collect` hands back the ``{(label, seed): record}`` mapping the shared table
helpers in :mod:`scripts.method.report` already speak, so the notebook prints from disk
and never from whatever happens to be in memory.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .train import ARM_LABELS, REPORT_ORDER, TEST_DIR, rec_path, traj_path

COLUMNS = {"mid W2": ("intermediate", "W2"),
           "end W2": ("endpoint", "W2"),
           "mid W2F": ("intermediate", "W2F")}

MOMENT_KEYS = ("R1", "R2", "frac_covered")


def labels(arms=REPORT_ORDER) -> list[str]:
    return [ARM_LABELS[a] for a in arms]


def collect(root: str, dims, seeds, arms=REPORT_ORDER) -> dict:
    """``{dim: {(label, seed): record}}``, asserting the grid is complete."""
    out: dict = {}
    for dim in dims:
        cells = {}
        for arm in arms:
            for seed in seeds:
                path = rec_path(root, dim, arm, seed)
                assert os.path.exists(path), (
                    f"missing {path}; the test stage writes one record per "
                    "(dim, arm, seed)")
                with open(path) as fh:
                    cells[(ARM_LABELS[arm], seed)] = json.load(fh)
        out[dim] = cells
    return out


def floors_of(runs: dict, seeds) -> dict:
    """The sampling floors, which every record in a dimension carries identically."""
    return next(iter(runs.values()))["metrics"]["floors"]


def n_ref_mid(runs: dict) -> int:
    return int(next(iter(runs.values()))["n_ref_mid"])


def table_moments(runs: dict, names, seeds) -> pd.DataFrame:
    """``R_1`` / ``R_2`` / coverage, mean over seeds, one row per arm."""
    rows = {}
    for nm in names:
        m = [runs[(nm, s)]["moments"] for s in seeds]
        rows[nm] = {k: float(np.mean([x[k] for x in m])) for k in MOMENT_KEYS}
    return pd.DataFrame(rows).T


def moment_diag(runs: dict, seeds) -> dict:
    d = next(iter(runs.values()))["moments"]
    return {k: d[k] for k in ("h", "n_nodes", "n_segments", "delta")}


def load_slab(root: str, dim: int, arm: str, seed: int) -> dict:
    with np.load(traj_path(root, dim, arm, seed)) as z:
        return {k: np.asarray(z[k], dtype=np.float64) if k != "path_idx"
                else np.asarray(z[k]) for k in z.files}
