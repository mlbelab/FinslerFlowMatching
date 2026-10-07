"""Driver for the erythroid benchmark: sweep our arms per dimension, then five seeds.

One *cell* is ``(dim, arm, seed)``.  Every cell rebuilds ``P`` on the two kept
latent-time marginals, trains one arm, integrates the source marginal forward, and is
scored at the midpoint of its time axis against the withheld middle marginal, using
:mod:`~scripts.experiments.erythroid.reference_release` — the reference's own three statistics,
computed by the reference's own code.

Which arms run here — three blocks, and what each one isolates
--------------------------------------------------------------
``published``  (:data:`PUBLISHED`)
    Their Table 4, transcribed.  Their code, their hyper-parameters, their data.

``release``  (:data:`RELEASE_ARMS` — ``cfm_release``, ``curly_release``)
    Their training code at their published hyper-parameters, on our UniTVelo cache, via
    :mod:`~scripts.experiments.erythroid.reference_train`.  Only the data changed against
    the row above, so ``published -> release`` measures the preprocessing effect.

``harness``  (``cfm``, ``mfm_land``, ``curly``) and ours (``ffm``, ``pathb``)
    Every arm through our trainer at their published budget (:func:`base_hparams`), on
    the same cloud, marginal cut, holdout split, rank key and five seeds.
    ``release -> harness`` is the trainer effect and ``harness baseline -> ours`` is the
    method difference, which is the only one of the three that is a claim about methods.

The harness holds the cloud, the cut, the splits, the evaluator and the whole optimiser
budget fixed, and leaves the loss, the geometry, **the coupling** and ODE-vs-SDE sampling
to the arm: OT-CFM *is* its exact-OT coupling and Curly-FM *is* its drift-discrepancy
assignment, so equalising those would delete the methods rather than compare them.

``mfm_land`` has no ``release`` cell and E1 prints ``--`` there: the MFM release ships no
erythroid config.  Two differences survive the harness — our nets use SiLU with a
sinusoidal time embedding where theirs use SELU with raw ``t``, and our ``cfm`` solves one
entropic Sinkhorn problem over up to ``ot_max_pts`` endpoints where theirs recomputes an
exact EMD per 256-point minibatch — and the ``release`` block is what bounds their size.

Selection
---------
They tune nothing on this dataset, so there is no published grid to match and ours is
chosen by :mod:`scripts.core.selection` — the mean of an endpoint ``W2`` at t = 1 against
the 15 % val split of the kept target marginal and a midpoint ``W2`` at t = 1/2 against
the 10 % selection slice of the withheld one.  Every reported number is measured on the
disjoint 90 %, with the reference field's neighbour pool restricted to the same
remainder.  The swept arm trains on the 85 % split of the kept cells (fixed
``SPLIT_SEED``) and is refit on all kept cells at the chosen point for the five seeds.

Our block is therefore not selected under the same conditions as the transcribed one,
which was measured blind; E1's legend says so and prints the subsampling shift.  Training
is untouched by all of this: the middle marginal is excluded from
:func:`~scripts.experiments.erythroid.datasets.build_training_set` regardless.

Per-dimension selection
-----------------------
The grid is searched independently at each of d = 2, 20, 50, since d = 2 is a
standardised UMAP and d = 20/50 are unstandardised PCA prefixes two orders of magnitude
larger.  That is necessary but not sufficient: a ladder of absolute values is re-searched
per dimension without being re-scaled, so our arms are addressed in measured multiples
(:mod:`scripts.core.scales`) — ``rho_mult`` in units of ``mean tr Σ/d``, ``lam_mult`` in
units of ``ā``, ``width_mult`` in kernel bandwidths — and one ladder is then valid at all
three dimensions.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from scripts.core.arms import N_STEPS, HParams, run_arm
from scripts.core.paths import experiment_root
from scripts.core.scales import MULTIPLE_OF
from scripts.core.scales import resolve as resolve_scales
from scripts.core.selection import OBJECTIVE_KEY, OBJECTIVE_TERMS, with_objective

from scripts.experiments.sheet.paper_focus import FFM_LABEL, PATHB_LABEL

from .datasets import (DIMS, HOLDOUT_BIN, TRAIN_BINS, build_training_set, load_cloud,
                       split_holdout, split_shown)
from .reference_release import (REFERENCE_K, get_ut_knn_gaussian,
                                reference_field_diagnostics, release_path_metrics,
                                release_wasserstein, signal_to_noise,
                                velocity_scale_for_dim)
from .reference_train import RELEASE_ARMS, release_settings, run_release_arm

#: Every arm this benchmark runs, in report order.  Three blocks — see the module
#: docstring; :data:`ARM_BLOCK` is the mapping the tables group by.
ARMS = ("cfm_release", "curly_release",
        "cfm", "mfm_land", "curly",
        "ffm", "pathb")
ARM_LABELS = {
    "cfm_release": "OT-CFM (their code)", "curly_release": "Curly-FM (their code)",
    "cfm": "OT-CFM (harness)", "mfm_land": "MFM/LAND (harness)",
    "curly": "Curly-FM (harness)",
    "ffm": f"{FFM_LABEL} (ours)", "pathb": f"{PATHB_LABEL} (ours)",
}
#: which of the three blocks each arm belongs to.  ``release`` rows are their code on our
#: cache, ``harness`` rows are baselines through our trainer, ``ours`` is Path A / Path B.
ARM_BLOCK = {"cfm_release": "release", "curly_release": "release",
             "cfm": "harness", "mfm_land": "harness", "curly": "harness",
             "ffm": "ours", "pathb": "ours"}
assert set(ARM_BLOCK) == set(ARMS)
#: the harness baseline each published/release method is compared against, i.e. the two
#: ends of the ``release -> harness`` delta E4 prints.  ``mfm_land`` has no release cell.
BLOCK_PAIRS = {"cfm": "cfm_release", "curly": "curly_release", "mfm_land": None}
#: arm -> the name its row carries in the transcribed :data:`PUBLISHED` block.
PUBLISHED_OF = {"cfm": "OT-CFM", "cfm_release": "OT-CFM",
                "curly": "CURLY-FM", "curly_release": "CURLY-FM",
                "mfm_land": "MFM"}

ARM_BASE: dict[str, str] = {}

OUR_METRIC: dict = {
    "metric_form": "fw_lambda", "c": 1.0,
    "drift_mode": "full", "sigma_mode": "full",
    "rho_mult": 0.3, "lam_mult": 0.03,
}
ARM_FIXED: dict[str, dict] = {
    "ffm": OUR_METRIC,
    "pathb": OUR_METRIC,
}

INHERIT_METRIC = {"pathb": "ffm"}
METRIC_AXES = ("rho_mult", "lam_mult")


def inherited_metric(arm: str, picks_at_dim: dict) -> dict | None:
    src = INHERIT_METRIC.get(arm)
    if src is None:
        return None
    parent = picks_at_dim.get(src)
    if parent is None:
        return None
    return {k: parent[k] for k in METRIC_AXES if k in parent}


def engine_arm(arm: str) -> str:
    """The :mod:`scripts.core.arms` arm behind a benchmark arm name."""
    assert arm not in RELEASE_ARMS, f"{arm} is not an engine arm; it is their code"
    return ARM_BASE.get(arm, arm)


#: their Table 4 block, transcribed from the reference paper, mean ± std over their runs.
#: Keyed (dim, method) -> (cos_dist, l2, w2).  The d>2 L2 column is printed in their
#: table as "L2 (x10^3)"; the values here are the *unscaled* numbers, so 1.885 x 10^3
#: is stored as 1885.0 and the report re-applies the scaling.
PUBLISHED: dict[tuple[int, str], dict] = {
    (2, "OT-CFM"):   {"cos_dist": (0.146, 0.001), "l2": (2.704, 0.019), "w2": (0.646, 0.006)},
    (2, "MFM"):      {"cos_dist": (0.014, 0.001), "l2": (1.999, 0.014), "w2": (0.269, 0.004)},
    (2, "CURLY-FM"): {"cos_dist": (0.009, 0.000), "l2": (1.663, 0.293), "w2": (0.369, 0.090)},
    (20, "OT-CFM"):   {"cos_dist": (0.489, 0.001), "l2": (1885.0, 20.0), "w2": (6.103, 0.074)},
    (20, "MFM"):      {"cos_dist": (0.495, 0.001), "l2": (1627.0, 40.0), "w2": (4.855, 0.052)},
    (20, "CURLY-FM"): {"cos_dist": (0.488, 0.001), "l2": (1721.0, 35.0), "w2": (6.124, 0.027)},
    (50, "OT-CFM"):   {"cos_dist": (0.490, 0.000), "l2": (2215.0, 22.0), "w2": (7.969, 0.029)},
    (50, "MFM"):      {"cos_dist": (0.494, 0.000), "l2": (1971.0, 23.0), "w2": (6.727, 0.022)},
    (50, "CURLY-FM"): {"cos_dist": (0.489, 0.000), "l2": (2045.0, 73.0), "w2": (7.729, 0.046)},
}
PUBLISHED_METHODS = ("OT-CFM", "MFM", "CURLY-FM")
#: which reported statistics the transcribed block actually has.  E1's fourth row, the
#: true 2-Wasserstein, exists only for the arms we ran.
PUBLISHED_METRICS = ("cos_dist", "l2", "w2")

SEEDS = (0, 1, 2, 3, 4)
#: fixed for the whole benchmark: the selection split must not move between grid points.
SPLIT_SEED = 0
VAL_FRACTION = 0.15

#: The run tree.  Defined here rather than in each caller so ``tune.py``, the notebook,
#: ``report`` and ``figures`` cannot end up pointed at four different directories.
ROOT = experiment_root("erythroid_bench")

TEST_DIR = "test"
TRAJ_DIR = "trajectories"
#: the sweep's own slabs, so a future selection term is a backfill and not a retrain.
SWEEP_TRAJ_DIR = "trajectories_sweep"

#: What ``stage_sweep --force`` audits a re-run cell against: the quantities upstream of
#: the optimiser, which are deterministic functions of (cache, dim, hyper-parameters) and
#: must come back identical.  The score must not, and is not audited.
AUDIT_DIAGS = ("cloud_diag", "ts_diag")
#: relative tolerance for the above: GPU reductions over ~10^4 cells reassociate.
AUDIT_RTOL = 1e-4
#: Diagnostics excluded from the audit because they are extremal and therefore amplify
#: exactly the last-digit noise the tolerance above is sized for.  Roughly thirty
#: non-extremal fields are still compared.
AUDIT_SKIP = ("mobility_cond", "sigma_eig_range", "a_n_range", "F_min",
              "max_beta_norm_Ginv", "void_vs_manifold_cost_ratio")
#: How many numeric fields the audit must find before it may pass, so a renamed
#: diagnostic block cannot turn the check into a no-op that reports success.
AUDIT_MIN_FIELDS = 8

#: ``run_arm`` requires divisibility by 3; 300 is also even, so ``traj[150]`` is exactly
#: the t = 1/2 slice their evaluation integrates to.
N_STEPS_ERYTHROID = N_STEPS

# --------------------------------------------------------------------------- #
#  The selection grid — matched budget
# --------------------------------------------------------------------------- #
#: **Every swept arm gets the same search freedom.**  Four clauses, all checked by
#: the verification suite in the research repository:
#:
#: 1. **Eight rungs per live axis**, for every arm.  Equal search freedom, not equal
#:    wall-clock.  A ladder widens by one rung of its own spacing whenever an argmin lands
#:    on an edge with the score still falling, **baselines first**; the count reached eight
#:    that way and every earlier record stays valid, since a cell is keyed by the point it
#:    measures and not by its position in the grid.
#: 2. **One knob per live axis.**  Each baseline has exactly one axis that does anything,
#:    provably so: ``run_cfm`` never reads ``hp.sigma`` and LAND's ``gamma`` is used only
#:    in the 2-D branch, so a second ladder for either would be eight identical runs.
#:    Ours turns two, ``rho_mult`` and ``lam_mult``, neither recoverable from the other,
#:    so it gets 8 x 8 = 64 cells against their 8.  This is the one place the budget is
#:    deliberately not equal, and E2's caption says so.
#: 3. **Each arm's published or default value is measured**, as a rung where the ladder's
#:    units can express it and as an appended point where they cannot (``pathb``'s anchor
#:    is the reference's absolute σ = 0.01; see :func:`grid_points`).
#: 4. **One rank key for everybody** — the objective of :mod:`scripts.core.selection`, so
#:    the baselines get exactly the holdout advantage ours has.
#:
#: Ours are quoted in measured multiples and the baselines' in their own absolute units,
#: because re-expressing a baseline's knob would make its published value unquotable.
GRID: dict[str, dict[str, list]] = {
    #: ``rho_mult`` — the isotropic floor of M = ρI + Σ in units of ``mean tr Σ/d``.
    #: ``lam_mult`` — the quadrature floor λ in units of ``ā = mean ‖b‖_{G_0}``, reported
    #: convention-free as ``c_eff = ā/sqrt(ā² + λ²)``.  Both are half-decade rungs over
    #: 3.5 decades, the spacing and extent :mod:`scripts.experiments.pancreas.tune` uses.
    #:
    #: The λ ladder sits two rungs above that tree's, and both ends are measurements.
    #: Below it, ``lam_mult = 0.001`` is under this cloud's float32 admissibility floor
    #: (:func:`scripts.core.scales.lam_mult_floor`): the margin λ²/(a_raw² + λ²) is a
    #: couple of ulp, so the metric is the degenerate Randers limit and F vanishes on a
    #: direction.  Above it, the widening rule fired on a +10 % top-rung gradient at d = 2.
    #: Large λ is the Riemannian limit and not a pathology: ``c_eff`` is 0.033 at the top
    #: rung, so the arm is asymptoting to the pure quasipotential G_0.
    "ffm":   {"rho_mult": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
              "lam_mult": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
    #: ``width_mult`` — the bridge half-width at t = ½ in kernel bandwidths
    #: (:func:`scripts.core.scales.sigma_from_width`), so it is a physical displacement.
    #: The same extent as :data:`scripts.experiments.pancreas.tune.SIGMA_MULTS` minus its
    #: UMAP-only top rung.
    "pathb": {"width_mult": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    #: ``blur_frac`` and not ``sigma``: ``run_cfm`` never reads ``hp.sigma``.  It sets the
    #: entropic regulariser to ``blur_frac · median(C)``, and ``blur_frac -> 0`` is the
    #: exact-OT coupling their minibatch solver uses, so this grid spans our arm towards
    #: theirs.  The engine default 0.1 is the middle rung.
    "cfm":   {"blur_frac": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
    #: ``curly_alpha`` — the magnitude weight of Curly-FM's geodesic loss, on decade
    #: rungs.  The engine writes that loss as ``mean((u − α·μ̇)²)`` and their notebook as
    #: ``mean((α·u − μ̇)²)``, so their published 0.1 is **10** here, four decades from the
    #: engine default 0.01, and the grid must contain both.  The vel stage stays at 1.0.
    "curly": {"curly_alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]},
}

#: ``mfm_land`` is the one arm whose knob depends on the dimension: ``make_mfm_metric``
#: uses LAND's ``gamma`` only in the 2-D branch, so a gamma grid at d = 20/50 would be
#: eight identical runs and ``rho`` is swept there instead.  The d = 2 rungs are centred
#: on MFM's published 0.125 and the d > 2 rungs on their published rho 1e-3.
GRID_BY_DIM: dict[str, dict[int, dict[str, list]]] = {
    "mfm_land": {
        2:  {"land_gamma": [0.0125, 0.04, 0.125, 0.4, 1.25, 4.0, 12.5, 40.0]},
        20: {"land_rho": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]},
        50: {"land_rho": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]},
    },
}

#: How many rungs every swept **axis** gets, per axis and not per arm (clause 2).
N_RUNGS = 8

#: **The no-search rung** — what each arm would run at if the grid did not exist, which
#: clause 3 requires to be in the grid and the assertion below enforces.
#:
#: A published value wins over the engine default.  ``pathb``'s anchor is therefore the
#: reference's σ = 0.01 and not the engine default 0.15, and ``curly``'s is their 0.1 in
#: the engine's parameterisation, i.e. 10.0.  ``ffm``'s is :data:`OUR_METRIC`'s own point,
#: so the arm's fixed edits and its anchor cannot disagree.
ANCHOR: dict[str, dict[int, dict] | dict] = {
    "ffm":      {"rho_mult": OUR_METRIC["rho_mult"],    # = the engine default rho at d=20
                 "lam_mult": OUR_METRIC["lam_mult"]},
    "pathb":    {"sigma": 0.01},            # theirs, per the Curly-FM release
    "cfm":      {"blur_frac": 0.1},         # engine default
    "curly":    {"curly_alpha": 10.0},      # their 0.1, on the engine's factor
    "mfm_land": {2: {"land_gamma": 0.125},  # their per-dataset yamls
                 20: {"land_rho": 1e-3}, 50: {"land_rho": 1e-3}},
}


def anchor_point(arm: str, dim: int) -> dict:
    """The no-search point for one arm at one dimension, or ``{}`` if it is not swept."""
    a = ANCHOR.get(arm)
    if a is None:
        return {}
    return a[dim] if arm in GRID_BY_DIM else a

#: Arms with no grid at all because their point is *theirs*: the release arms run at the
#: published hyper-parameters of :data:`scripts.experiments.erythroid.reference_train
#: .RELEASE_SETTINGS` and searching them would destroy the only thing they measure.
PINNED = RELEASE_ARMS

INHERIT: dict[str, str] = {}
assert set(GRID) | set(GRID_BY_DIM) | set(INHERIT) | set(PINNED) == set(ARMS), \
    "every arm must be swept, inherit a point, or be pinned to a published one"
assert not (set(GRID) & set(GRID_BY_DIM)), "an arm has one grid, not two"
for _arm, _spec in (list(GRID.items())
                    + [(a, s) for a, d in GRID_BY_DIM.items() for s in d.values()]):
    assert len(_spec) == 1 or ARM_BLOCK[_arm] == "ours", (
        f"{_arm} turns {sorted(_spec)} at once; only our arms may search more than one "
        f"axis, and only because their extra axes are live — see the matched-budget rule")
    for _axis, _rungs in _spec.items():
        assert len(_rungs) == N_RUNGS, (
            f"{_arm}.{_axis} has {len(_rungs)} rungs, not {N_RUNGS}; every swept axis "
            f"gets the same freedom along it (equal search freedom)")
assert set(ANCHOR) == set(GRID) | set(GRID_BY_DIM), \
    "every swept arm declares a no-search point, and only swept arms do"


#: **The harness budget: their published settings**, read out of
#: :data:`scripts.experiments.erythroid.reference_train.RELEASE_SETTINGS` so the two
#: cannot drift apart.  Running every arm at their budget is what makes the
#: ``release -> harness`` delta isolate the trainer rather than the compute; the price is
#: that our arms are no longer at the budget the engine defaults were chosen for, and the
#: archived ``erythroid_bench_prebudget/`` tree is what bounds that.  Widths take the
#: larger of their two nets at each dimension, so no arm is starved by another's choice.
HARNESS_BUDGET_NAME = "release-matched"


def base_hparams(dim: int, smoke: bool = False) -> HParams:
    """Shared starting point for every harness arm, before the grid override.

    :data:`~scripts.experiments.erythroid.reference_train.RELEASE_SETTINGS` for this
    dimension in engine names, with two exceptions: ``depth = 3`` is a translation of
    their three hidden layers, and ``ot_max_pts = 600`` is set from the data, since each
    marginal has ~3272 cells and the engine default of 200 would throw away most of the
    coupling problem.  600 is what fits at d = 50 with all three dimensions on one
    setting.  Their exact-OT solver has no analogue: it never sees more than a minibatch.
    """
    s = release_settings(dim)
    hp = HParams(
        width=max(s.cfm_width, s.geo_width, s.vel_width),
        depth=3,
        lr=s.lr,
        batch_size=s.batch_size,
        phase1_iters=s.geo_iters,     # geodesic / interpolant stage
        phase2_iters=s.vel_iters,     # flow-matching stage
        sbm_iters=s.vel_iters,        # Path B's bridge stage, the same allowance
        curly_k=s.k,                  # their reference-field width, k = 30
        curly_sigma=s.sigma,
        ot_max_pts=600,
    )
    if smoke:
        hp = replace(hp, phase1_iters=60, phase2_iters=60, sbm_iters=60, ot_max_pts=64)
    return hp


def hparams_for(dim: int, arm: str, override: dict, ts, device: torch.device,
                dtype: torch.dtype, smoke: bool = False) -> tuple[HParams, dict]:
    """``(the point this cell runs at, the scales its multiples were resolved against)``.

    The arm's fixed edits go on first and the swept override second, so a grid can never
    search over the switch that defines the arm.  :func:`scripts.core.scales.resolve` then
    turns every multiple into the absolute it names, measured off a geometry built on
    **this cell's** ``ts`` — the sweep stage trains on the 85 % split and the test stage
    on every kept cell, and ``ā`` and ``m̄`` are properties of whichever cloud that is.
    The measured scales come back with the HParams, so a record carries both the multiple
    it was addressed by and the absolute it ran at; neither alone is enough.
    """
    hp = base_hparams(dim, smoke)
    fixed = ARM_FIXED.get(arm, {})
    edits = {**fixed, **override}
    unknown = set(edits) - set(hp.as_dict()) - set(MULTIPLE_OF)
    assert not unknown, f"unknown hyper-parameter(s) {sorted(unknown)}"
    return resolve_scales(ts, hp, edits, device, dtype)


# --------------------------------------------------------------------------- #
#  Evaluation — their cell 12, with the neighbour pool made explicit
# --------------------------------------------------------------------------- #
def transport_field(res, hp: HParams) -> tuple[callable, str]:
    """The field that actually moves this arm's samples, as ``f(t, x)``.

    ``v_θ`` for a deterministic arm; for Path B the full SDE drift ``v_θ + σ² M(x) s_φ``,
    since scoring ``v_θ`` alone would compare Path B's marginal against a field that
    never transported it.  ``M`` is the spatial mobility ``ρI + Σ(x)``, which is the same
    object the training-time time-locked baseline evaluates.  Returns the callable and a
    tag recorded with the metrics, so a table never guesses which field a row was scored
    on.
    """
    v_net, aux = res.v_net, res.aux
    s_net, geom = aux.get("s_net"), aux.get("geom")
    if s_net is None or geom is None:
        return (lambda t, x: v_net(t, x)), "v_theta"

    sigma2 = float(hp.sigma) ** 2

    def drift(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        M = geom.mobility(x)                              # (n, d, d)
        s = s_net(t, x).unsqueeze(-1)                     # (n, d, 1)
        return v_net(t, x) + sigma2 * torch.bmm(M, s).squeeze(-1)

    return drift, "v_theta + sigma^2 M s_phi"


def evaluate(traj: np.ndarray, field, cloud, i_mid: np.ndarray, device: torch.device,
             dtype: torch.dtype, seed: int = 0) -> dict:
    """Score one arm's pushforward at t = 1/2 the way the release scores it.

    ``field`` is ``f(t, x)`` from :func:`transport_field`.  ``i_mid`` is the 90 % test
    slice of the withheld marginal, passed in rather than read off the cloud because it
    does two jobs — it is the W2 column's target *and* the ``x1`` half of the reference
    field's neighbour pool — and those must move together.

    ``traj`` is ``(n_steps+1, B, d)`` over a time axis mapping [0, 1] onto marginal 0 ->
    marginal 2, so their ``t_span=linspace(0, 1/2, 100)`` is our ``traj[:n_steps//2 + 1]``.
    The cos/L2 columns are path integrals over that half-interval and not endpoint values,
    because their two metric channels are extra ODE states; the W2 column is an endpoint
    quantity.  The reference field is
    :func:`~scripts.experiments.erythroid.reference_release.get_ut_knn_gaussian` with the
    neighbour pool their cell 12 passes, which is evaluation-only ground truth built
    partly from the holdout.  It adds a ``randn_like`` term, hence the seeding here.
    """
    n_steps = traj.shape[0] - 1
    assert n_steps % 2 == 0, f"need an even n_steps to land on t=1/2, got {n_steps}"
    n_half = n_steps // 2

    i_src = cloud.marginal(TRAIN_BINS[0])
    assert np.all(cloud.bin_id[i_mid] == HOLDOUT_BIN), \
        "the scored target must be drawn from the withheld marginal"
    x0 = torch.as_tensor(cloud.X[i_src], dtype=dtype, device=device)
    x1 = torch.as_tensor(cloud.X[i_mid], dtype=dtype, device=device)
    v0 = torch.as_tensor(cloud.velocity[i_src], dtype=dtype, device=device)
    v1 = torch.as_tensor(cloud.velocity[i_mid], dtype=dtype, device=device)

    states = [torch.as_tensor(traj[i], dtype=dtype, device=device)
              for i in range(n_half + 1)]
    scale = velocity_scale_for_dim(int(cloud.X.shape[1]))

    out = release_path_metrics(states, field, x0, x1, v0, v1, k=REFERENCE_K,
                               dt=1.0 / n_steps, velocity_scale=scale, target=x1,
                               seed=seed)

    pred = states[-1]
    torch.manual_seed(seed)
    with torch.no_grad():
        u_half, _ = get_ut_knn_gaussian(pred, x0, x1, v0, v1, k=REFERENCE_K,
                                        velocity_scale=scale)
    out["u_snr"] = signal_to_noise(u_half)
    out["velocity_scale"] = scale
    out.update({f"ref_{k}": v for k, v in
                reference_field_diagnostics(pred, x0, x1, v0, v1).items()})
    return out


#: metrics key the sweep ranks on: the repository's one objective, lower is better.  Its
#: two terms are stored separately by every cell, so a finished sweep can be re-ranked.
SELECTION_KEY = OBJECTIVE_KEY
SELECTION_LABEL = "objective (mean val W2)"


def selection_score(record: dict) -> float:
    """Rank key for the sweep, lower is better.

    Reads the stored scalar rather than recomputing it, and refuses a record that
    predates the rule instead of falling back to a different quantity.
    """
    m = record["metrics"]
    assert SELECTION_KEY in m, (
        f"sweep record has no {SELECTION_KEY!r} — it predates the single objective; "
        f"run `backfill --stage sweep` for d={record['dim']} {record['arm']}, or "
        "`sweep --force` if its slab is missing")
    return float(m[SELECTION_KEY])


# --------------------------------------------------------------------------- #
#  One cell
# --------------------------------------------------------------------------- #
def sweep_terms(traj: np.ndarray, cloud, i_val: np.ndarray) -> dict:
    """The objective's two terms for one swept cell, plus the averaged scalar.

    Factored out of :func:`run_cell` so :func:`stage_backfill` can recompute them from a
    saved slab.  Both terms are true ``W2`` (the release utility's ``power=2`` branch),
    even though the reported headline column is their ``power=1``.  ``i_val`` is the 15 %
    val split of the kept cells and the endpoint term uses its target-marginal half; the
    midpoint term uses ``cloud.target_val``, which nothing ever trains on.
    """
    i_end = i_val[cloud.bin_id[i_val] == TRAIN_BINS[1]]
    assert len(i_end) > 1, f"val slice of the target marginal is too small ({len(i_end)})"
    i_sel = cloud.target_val
    assert not set(i_sel.tolist()) & set(cloud.target_test.tolist()), \
        "the selection slice and the scored slice of the withheld marginal intersect"

    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    n_half = traj.shape[0] // 2
    terms = with_objective({
        "W2_endpoint": release_wasserstein(as_t(traj[-1]), as_t(cloud.X[i_end]), power=2),
        "W2_intermediate": release_wasserstein(as_t(traj[n_half]),
                                               as_t(cloud.X[i_sel]), power=2),
    })
    return {**terms, "n_val_cells": int(len(i_sel)), "n_val_end_cells": int(len(i_end))}


def scored_target(cloud) -> np.ndarray:
    """The 90 % test slice of the middle marginal, in one place so the W2 column and the
    reference field's neighbour pool cannot be given different slices."""
    return cloud.target_test


def run_cell(dim: int, arm: str, seed: int, override: dict, *, stage: str,
             device: torch.device, dtype: torch.dtype, n_steps: int,
             smoke: bool = False, traj_path: str | None = None,
             inherit: dict | None = None) -> dict:
    """Train one arm once and score it. ``stage`` is ``"sweep"`` or ``"test"``.

    In ``sweep`` the training set is the 85 % split and the metrics are the objective's
    two terms; in ``test`` it is every kept cell and the reference metrics too.  The two
    stages differ in what P is built on, which is why it is rebuilt here.
    """
    assert stage in ("sweep", "test"), stage
    assert not inherit or arm in INHERIT_METRIC, (
        f"{arm} inherits no metric, got {inherit}")
    cloud = load_cloud(dim)

    if stage == "sweep":
        i_train, i_val = split_shown(cloud, VAL_FRACTION, SPLIT_SEED)
        ts = build_training_set(cloud, subset=i_train)
    else:
        i_val = np.array([], dtype=np.int64)
        ts = build_training_set(cloud)

    t0 = time.time()
    if arm in RELEASE_ARMS:
        assert not override, f"{arm} is pinned to the published point, got {override}"
        hp, scales = None, {}
        res = run_release_arm(
            arm, ts.X[ts.p0], ts.X[ts.p1], ts.velocity[ts.p0], ts.velocity[ts.p1],
            dim=dim, seed=seed, device=device, dtype=dtype, n_steps=n_steps,
            smoke=smoke, log_every=0)
    else:
        hp, scales = hparams_for(dim, arm, {**(inherit or {}), **override},
                                 ts, device, dtype, smoke)
        res = run_arm(engine_arm(arm), ts, hp, seed, device, dtype, n_steps=n_steps)
    train_s = time.time() - t0

    n_half = res.traj.shape[0] // 2
    pred_half = res.traj[n_half]

    metrics: dict = {}
    if stage == "sweep":
        metrics.update(sweep_terms(res.traj, cloud, i_val))
    else:
        i_mid = scored_target(cloud)
        field, field_tag = transport_field(res, hp)
        metrics.update(evaluate(res.traj, field, cloud, i_mid, device, dtype, seed=seed))
        metrics["scored_field"] = field_tag
        metrics["n_target_cells"] = int(len(i_mid))

    if traj_path is not None:
        save_trajectory(res.traj, traj_path)

    if arm in RELEASE_ARMS:
        hparams, budget = asdict(release_settings(dim)), "release-verbatim"
    else:
        hparams, budget = hp.as_dict(), HARNESS_BUDGET_NAME

    return {
        "dim": dim, "arm": arm, "seed": seed, "stage": stage,
        "block": ARM_BLOCK[arm], "budget": budget,
        "override": override, "inherited": inherit,
        "hparams": hparams, "scales": _jsonable(scales),
        "metrics": metrics, "ts_diag": ts.diag, "arm_diag": _jsonable(res.diag),
        "cloud_diag": cloud.diag, "train_seconds": train_s,
        "n_steps": n_steps, "smoke": smoke,
    }


#: paths kept per test record for the figure.  The full slab is 200 MB at d = 50, so the
#: record carries the t = 1/2 slice in full (the scored object) and a fixed thinned set of
#: complete paths (the illustration).  Fixed, so two arms in one figure draw the same
#: cells and the comparison is between routes rather than between samples.
N_TRAJ_SAVED = 64


def save_trajectory(res_traj: np.ndarray, path: str) -> None:
    """Persist what the figure needs from one arm's integration."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_steps = res_traj.shape[0] - 1
    n = res_traj.shape[1]
    keep = np.linspace(0, n - 1, min(N_TRAJ_SAVED, n)).astype(int)
    np.savez_compressed(
        path,
        mid=res_traj[n_steps // 2].astype(np.float32),
        paths=res_traj[:, keep].astype(np.float32),
        path_idx=keep,
        start=res_traj[0].astype(np.float32),
        end=res_traj[-1].astype(np.float32),
    )


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


def _override_tag(override: dict) -> str:
    return "base" if not override else "_".join(
        f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}"
        for k, v in sorted(override.items()))


def _write_json(record: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(record), fh, indent=2)


def load_records(directory: str) -> list[dict]:
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            with open(os.path.join(directory, name)) as fh:
                out.append(json.load(fh))
    return out


# --------------------------------------------------------------------------- #
#  Stages
# --------------------------------------------------------------------------- #
def grid_spec(arm: str, dim: int) -> dict[str, list]:
    """The ``{knob: rungs}`` this arm is searched over at this dimension, empty for a
    pinned or inheriting arm.  ``mfm_land`` is the one entry that differs by dimension."""
    if arm in GRID_BY_DIM:
        by_dim = GRID_BY_DIM[arm]
        assert dim in by_dim, (
            f"{arm} has no grid at d={dim}; its swept knob is dimension-dependent and "
            f"must be chosen explicitly rather than inherited from another dimension")
        return by_dim[dim]
    return GRID.get(arm, {})


def grid_points(arm: str, dim: int) -> list[dict]:
    """The override dicts for one arm's grid, including the empty base point.

    The full product over the arm's axes, last axis fastest, matching
    :class:`scripts.method.tuning.Grid` so a flat cell index means the same thing here as
    to the scheduler that submitted it.  The anchor is appended when the ladder's units
    cannot express it (clause 3), which on this tree is ``pathb`` alone: its no-search
    point is the reference's absolute σ = 0.01 and the ladder is in kernel bandwidths.
    """
    spec = grid_spec(arm, dim)
    if not spec:
        return [{}]
    keys = list(spec)
    points = [dict(zip(keys, vals)) for vals in itertools.product(*spec.values())]
    anchor = anchor_point(arm, dim)
    if anchor and anchor not in points:
        points.append(anchor)
    return points


def _audit_inputs(tag: str, prior: dict, rec: dict) -> None:
    """Assert a re-run cell was handed the same data and the same geometry as before.

    Walks :data:`AUDIT_DIAGS` plus the geometry block over the fields present in *both*
    records, skipping :data:`AUDIT_SKIP`.  Floats at any nesting depth are compared to
    :data:`AUDIT_RTOL`; fewer than :data:`AUDIT_MIN_FIELDS` comparisons is a failure.
    """
    blocks = [(k, prior.get(k, {}), rec.get(k, {})) for k in AUDIT_DIAGS]
    blocks.append(("geometry",
                   prior.get("arm_diag", {}).get("geometry", {}),
                   rec.get("arm_diag", {}).get("geometry", {})))

    def _numeric(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def _agrees(a, b) -> bool:
        """Structural equality with a tolerance on every float anywhere inside."""
        if _numeric(a) and _numeric(b):
            return abs(a - b) <= AUDIT_RTOL * max(1.0, abs(a))
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(map(_agrees, a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(_agrees(a[k], b[k]) for k in a)
        return a == b

    n = 0
    for block, old, new in blocks:
        for key in sorted((set(old) & set(new)) - set(AUDIT_SKIP)):
            a, b = old[key], new[key]
            assert _agrees(a, b), (
                f"{tag}: {block}.{key} moved between runs, {a!r} -> {b!r}; that is "
                f"upstream of the optimiser, so it is a changed pipeline and not "
                f"training noise")
            n += 1
    assert n >= AUDIT_MIN_FIELDS, (
        f"{tag}: the re-run audit compared only {n} fields (< {AUDIT_MIN_FIELDS}); the "
        f"diagnostic blocks {AUDIT_DIAGS} have been renamed and the check is a no-op")


def arm_ladder(recs, dim: int, arm: str,
               inherited: dict | None = None) -> tuple[list[dict], int]:
    """One arm's rungs at one dimension, seed-averaged, plus how many the grid wants.

    These are the rows ``tune.py collect`` prints and the argmin of them is what
    :func:`tuned_points` returns, so the ladder shown and the point chosen are one
    ranking.  Three filters, each of which prints what it dropped: points not in the
    current grid, cells measured under a different :data:`ARM_FIXED` metric, and cells
    measured on a different inherited metric.  The second is not hypothetical —
    ``pathb``'s absolute ``sigma = 0.01`` anchor was also a rung of the pre-``fw_lambda``
    ladder, so the two records share a filename while measuring different arms.
    """
    source = INHERIT.get(arm, arm)
    live = {_override_tag(o) for o in grid_points(source, dim)}
    cells = [r for r in recs if r["dim"] == dim and r["arm"] == source]
    stale = {_override_tag(r["override"]) for r in cells} - live
    if stale:
        print(f"[tune] d={dim} {source}: ignoring {len(stale)} off-grid "
              f"point(s) {sorted(stale)}")
    cells = [r for r in cells if _override_tag(r["override"]) in live]

    fixed = {k: v for k, v in ARM_FIXED.get(source, {}).items() if k not in MULTIPLE_OF}
    if fixed:
        keep = [r for r in cells
                if all(r["hparams"].get(k) == v for k, v in fixed.items())]
        if len(keep) < len(cells):
            print(f"[tune] d={dim} {source}: ignoring {len(cells) - len(keep)} cell(s) "
                  f"measured under a different metric than {fixed}")
        cells = keep

    if source in INHERIT_METRIC:
        keep = [r for r in cells if r.get("inherited") == inherited]
        if len(keep) < len(cells):
            print(f"[tune] d={dim} {source}: ignoring {len(cells) - len(keep)} cell(s) "
                  f"measured on an inherited metric other than {inherited}")
        cells = keep

    by_point: dict[str, list] = {}
    for r in cells:
        by_point.setdefault(_override_tag(r["override"]), []).append(r)
    rows = []
    for rs in by_point.values():
        mean = lambda k: float(np.mean([r["metrics"][k] for r in rs]))
        rows.append({"point": rs[0]["override"], "n_seeds": len(rs),
                     **{t: mean(t) for t in OBJECTIVE_TERMS},
                     SELECTION_KEY: float(np.mean([selection_score(r) for r in rs]))})
    return rows, len(live)


def tuned_points(root: str, dims, arms) -> dict[tuple[int, str], dict]:
    """argmin of the selection score, per (dim, arm), averaged over the sweep seeds.

    An arm in :data:`INHERIT_METRIC` is ranked on its own cells but only those measured
    at its parent's chosen metric, which is resolved first.  An arm in :data:`PINNED`
    ranks on nothing and comes back with the empty override.
    """
    recs = load_records(os.path.join(root, "sweep"))
    picks: dict[tuple[int, str], dict] = {}
    best_of = lambda rows: min(rows, key=lambda r: r[SELECTION_KEY])["point"]
    for dim in dims:
        at_dim: dict[str, dict] = {}
        for arm in ([a for a in arms if a not in INHERIT_METRIC]
                    + [a for a in arms if a in INHERIT_METRIC]):
            if arm in PINNED:
                picks[(dim, arm)] = {}
                continue
            src = INHERIT_METRIC.get(arm)
            if src is not None and src not in at_dim:
                rows, _ = arm_ladder(recs, dim, src)
                if rows:
                    at_dim[src] = best_of(rows)
            rows, _ = arm_ladder(recs, dim, arm, inherited_metric(arm, at_dim))
            if rows:
                picks[(dim, arm)] = at_dim[arm] = best_of(rows)
    return picks


def stage_test(root: str, picks: dict, dims, arms, seeds, device, dtype, n_steps: int,
               smoke: bool, force: bool) -> None:
    """The reported grid: every (dim, arm, seed) at the point ``picks`` names.

    ``picks`` is passed in rather than derived, because the driver is
    ``notebooks/erythroid_ffm_paper.ipynb`` and what it holds is the pasted literal.  A
    caller that wants the sweep's own answer hands :func:`tuned_points` straight back.
    """
    out = os.path.join(root, TEST_DIR)
    for dim in dims:
        at_dim = {a: p for (d, a), p in picks.items() if d == dim}
        for arm in arms:
            override = picks.get((dim, arm))
            if override is None:
                print(f"[test] no tuned point for d={dim} {arm}; skipping")
                continue
            inherit = inherited_metric(arm, at_dim)
            assert arm not in INHERIT_METRIC or inherit, (
                f"d={dim} {arm} takes its metric from {INHERIT_METRIC[arm]}, which has "
                f"no tuned point here; run the sweep and `collect` first")
            for seed in seeds:
                tag = f"d{dim}_{arm}_s{seed}.json"
                path = os.path.join(out, tag)
                if os.path.exists(path) and not force:
                    with open(path) as fh:
                        prior = json.load(fh)
                    was, was_inherit = prior.get("override"), prior.get("inherited")
                    if was == override and was_inherit == inherit:
                        print(f"[test] skip {tag}")
                        continue
                    print(f"[test] {tag} was run at {_override_tag(was or {})} on "
                          f"{was_inherit}, sweep now picks {_override_tag(override)} on "
                          f"{inherit}; re-running")
                traj_path = os.path.join(root, TRAJ_DIR, f"d{dim}_{arm}_s{seed}.npz")
                rec = run_cell(dim, arm, seed, override, stage="test", device=device,
                               dtype=dtype, n_steps=n_steps,
                               smoke=smoke, traj_path=traj_path, inherit=inherit)
                _write_json(rec, path)
                m = rec["metrics"]
                print(f"[test] {tag}  cos={m['cos_dist']:.4f}  "
                      f"L2={m['l2']:.4g}  W1={m['w2']:.4f}  W2={m['w2_true']:.4f}  "
                      f"({rec['train_seconds']:.0f}s)")


# --------------------------------------------------------------------------- #
#  Backfill — endpoint statistics added after the fact, without a retrain
# --------------------------------------------------------------------------- #
#: record keys that are pure functions of the saved t = 1/2 cloud and the withheld
#: marginal, so they can be recomputed from ``trajectories/*.npz`` alone.
BACKFILLABLE = ("w2", "w2_true")


def stage_backfill(root: str, dims, arms, seeds, force: bool) -> None:
    """Recompute the cloud-distance metrics of existing records from the saved slab.

    Every statistic in :data:`BACKFILLABLE` is a deterministic function of the full
    t = 1/2 state :func:`save_trajectory` keeps and the withheld marginal, so adding one
    afterwards is arithmetic rather than a retrain.  Anything that depends on the *field*
    (``cos_dist``, ``L2``, the reference-field diagnostics) still needs ``test --force``.
    """
    out = os.path.join(root, TEST_DIR)
    clouds: dict[int, object] = {}
    n_done = 0
    for dim in dims:
        for arm in arms:
            for seed in seeds:
                tag = f"d{dim}_{arm}_s{seed}.json"
                path, traj = (os.path.join(out, tag),
                              os.path.join(root, TRAJ_DIR,
                                           f"d{dim}_{arm}_s{seed}.npz"))
                if not (os.path.exists(path) and os.path.exists(traj)):
                    continue
                with open(path) as fh:
                    rec = json.load(fh)
                if all(k in rec["metrics"] for k in BACKFILLABLE) and not force:
                    continue
                if dim not in clouds:
                    clouds[dim] = load_cloud(dim)
                cloud = clouds[dim]
                pred = torch.as_tensor(np.load(traj)["mid"], dtype=torch.float32)
                tgt = torch.as_tensor(cloud.X[scored_target(cloud)],
                                      dtype=torch.float32)
                assert pred.shape[1] == tgt.shape[1], (pred.shape, tgt.shape)
                fresh = {"w2": release_wasserstein(pred, tgt, power=1),
                         "w2_true": release_wasserstein(pred, tgt, power=2)}
                for k, v in fresh.items():
                    old = rec["metrics"].get(k)
                    assert old is None or abs(old - v) < 1e-4 * max(1.0, abs(old)), (
                        f"{tag}: stored {k}={old} but the saved t=1/2 cloud gives {v}; "
                        f"the slab is not the scored object")
                rec["metrics"].update(fresh)
                _write_json(rec, path)
                n_done += 1
                print(f"[backfill] {tag}  W1={rec['metrics']['w2']:.4f}  "
                      f"W2={rec['metrics']['w2_true']:.4f}")
    print(f"[backfill] updated {n_done} record(s)")


def stage_backfill_sweep(root: str, dims, arms, seeds, force: bool) -> None:
    """Recompute a finished sweep's objective terms from ``trajectories_sweep/``.

    The same bargain :func:`stage_backfill` strikes for the reported records: every cell
    saved its full t = 1/2 and t = 1 states and the objective is a pure function of those
    and the two val slices.  A cell whose slab is missing needs ``sweep --force``.
    """
    src = os.path.join(root, "sweep")
    clouds: dict[int, object] = {}
    n_done, missing = 0, []
    for name in sorted(os.listdir(src)) if os.path.isdir(src) else []:
        if not name.endswith(".json"):
            continue
        path = os.path.join(src, name)
        with open(path) as fh:
            rec = json.load(fh)
        if rec["dim"] not in dims or rec["arm"] not in arms or rec["seed"] not in seeds:
            continue
        if SELECTION_KEY in rec["metrics"] and not force:
            continue
        traj = os.path.join(root, SWEEP_TRAJ_DIR, name[:-5] + ".npz")
        if not os.path.exists(traj):
            missing.append(name)
            continue
        dim = rec["dim"]
        if dim not in clouds:
            clouds[dim] = load_cloud(dim)
        cloud = clouds[dim]
        _, i_val = split_shown(cloud, VAL_FRACTION, SPLIT_SEED)
        z = np.load(traj)
        stand_in = np.stack([z["mid"], z["mid"], z["end"]])
        rec["metrics"].update(sweep_terms(stand_in, cloud, i_val))
        _write_json(rec, path)
        n_done += 1
        print(f"[backfill/sweep] {name}  objective={rec['metrics'][SELECTION_KEY]:.4f}")
    if missing:
        print(f"[backfill/sweep] {len(missing)} cell(s) have no saved slab and need "
              f"`sweep --force`: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    print(f"[backfill/sweep] updated {n_done} record(s)")
