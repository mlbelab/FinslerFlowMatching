"""The submission protocol: quick tune on val, then one test pass, with wall clocks.

Two subcommands, run in this order.

``tune``
    A small grid per arm, trained on the **train** split and ranked by
    :func:`scripts.experiments.sheet.split.val_objective`.  The search is deliberately coarse: the
    brief asks for "good enough", not optimal, and every extra grid point is a full
    training run.

    The objective is the repository's one selection rule: endpoint W2 averaged with
    intermediate W2 against the **val half** of the withheld region, disjoint from test.
    Every grid point records both terms separately as well as their mean, so a finished
    sweep can be re-ranked — or given a new term — for free (``tune --rank-only``).

    Tuned points land in ``tuned/<cloud>.json``.

    The Sheet is searched on its own.  :data:`SWEEP_SOURCE` is the indirection that
    says so, and it is kept rather than inlined because the grouping it encodes is a
    modelling claim, not bookkeeping: ``rho`` floors ``G^{-1} = rho I + Sigma`` and
    ``Sigma`` carries units of (length)^2 per unit time, so its scale tracks the ambient
    dimension and the local cell spacing far more than it tracks the shape of the
    manifold.  Clouds of equal dimension may therefore share a sweep; a cloud of a
    different dimension may not.

``test``
    Train each arm on the train split with its inherited hyper-parameters, five model
    seeds, and score against the **test** split.  Val metrics are recorded alongside
    for completeness but are not what the paper reports.

One split, many model seeds
---------------------------
The 70/15/15 partition is drawn once per cloud, at ``--split-seed 0``, and held fixed
for every arm, every hyper-parameter point and every model seed.  It has to be:
hyper-parameters were chosen on *that* val set, so re-drawing the partition per model
seed would rotate those cells into the test set and leak the selection.  The error bars
therefore measure training stochasticity (initialisation and minibatching), which on
these clouds is the dominant term anyway.

Artefacts, all under ``outputs/route_bench_paper``::

    sweeps/<cloud>_<arm>_<i>.json     one grid point: config + every val term
    tuned/<cloud>.json                the chosen point per arm  (+ _report.json)
    runs/<cloud>_<arm>_s<seed>.json   test + val metrics, protocol, wall clock
    weights/…pt   trajectories/…npz   as in :mod:`scripts.experiments.sheet.run`

Usage
-----
    python -m scripts.experiments.sheet.paper_run tune                    # the Sheet grid
    python -m scripts.experiments.sheet.paper_run tune --clouds Sheet
    python -m scripts.experiments.sheet.paper_run tune --rank-only        # re-rank, trains nothing
    python -m scripts.experiments.sheet.paper_run test --seeds 0 1 2 3 4  # the reported grid
    python -m scripts.experiments.sheet.paper_run test --smoke            # wiring check only
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import time
import traceback

import numpy as np
import torch

from scripts.core.arms import ARMS, MAIN_ARMS, N_STEPS, HParams, run_arm
from scripts.core.paths import experiment_root

from .datasets import DATASET_NAMES, make_dataset
from .evaluate import evaluate_route
from .run import SMOKE_HPARAMS, _atomic_write_json, _ensure, _jsonable, save_weights
from .split import (
    OBJECTIVE_TERMS,
    make_split,
    objective_from,
    split_route_eval,
    split_training_set,
    val_objective,
)

DEFAULT_OUT = experiment_root("route_bench_paper")
#: the partition every reported number is computed on
SPLIT_SEED = 0
#: model seeds of the reported grid
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# --------------------------------------------------------------------------- #
#  Which cloud lends its hyper-parameters to which
# --------------------------------------------------------------------------- #
#: cloud -> the cloud whose sweep it inherits.  With only the Sheet left this is the
#: identity, but it is not dead: ``hparam_source`` is written into every run record, so
#: the provenance of a reported number stays explicit rather than implied.  See the
#: module docstring for when two clouds may share a point.
SWEEP_SOURCE = {"Sheet": "Sheet"}
assert set(SWEEP_SOURCE) == set(DATASET_NAMES), "SWEEP_SOURCE must cover every cloud"
#: the clouds a grid is actually run on
SWEEP_CLOUDS = tuple(dict.fromkeys(SWEEP_SOURCE.values()))

#: arms that get their own search.  The rest inherit, and must: an ablation that differs
#: from its parent in one switch must not also differ in optimiser and coupling
#: settings, or the row stops isolating the switch.
TUNED_ARMS = ("ffm", "pathb", "cfm", "mfm_land", "curly")
INHERIT = {"ffm_riem": "ffm", "ffm_eucl": "ffm", "ffm_drift": "ffm",
           "pathb_iso": "pathb", "pathb_const": "pathb"}
assert set(TUNED_ARMS) | set(INHERIT) == set(ARMS), "every arm must be tuned or inherit"

#: Coarse on purpose — one axis per question a reviewer would ask, two or three rungs
#: each.  ``phase1_iters``/``phase2_iters``/``sbm_iters`` are *not* swept: the geodesic
#: interpolant is known to collapse below 2500/1500 (see ``scripts.core.arms``), and a
#: sweep allowed to shorten training would happily buy a cheap val number by breaking
#: the very stage the method rests on.
SWEEP_GRIDS: dict[str, dict[str, list]] = {
    # rho: how hard the metric penalises leaving the manifold.  c: Randers 1-form
    # strength, i.e. how much of the P^asym flux reaches F — the directional term.
    "ffm": {"rho": [0.01, 0.03, 0.1], "c": [0.5, 0.9]},
    "pathb": {"sigma": [0.05, 0.15, 0.4], "rho": [0.03, 0.1]},
    "cfm": {"blur_frac": [0.03, 0.1, 0.3]},
    "mfm_land": {"land_gamma": [0.1, 0.2, 0.5], "land_rho": [1e-3, 1e-2]},
    # ``curly_sigma`` is not swept: their paper fixes it at zero for every experiment
    # it reports and recommends that setting "unless some reference sigma value is
    # known", which on this cloud it is not.  See :class:`scripts.core.arms.HParams`.
    "curly": {"curly_alpha": [0.003, 0.01, 0.1]},
}
assert set(SWEEP_GRIDS) == set(TUNED_ARMS)

# --------------------------------------------------------------------------- #
#  Geometry variants
# --------------------------------------------------------------------------- #
#: The metric the P-dependent arms are built on.  Each variant is a set of ``HParams``
#: defaults, its own sweep grid for the arms whose knobs change, and its own output
#: root — three parallel, self-contained result trees that :mod:`scripts.experiments.sheet.paper_geom`
#: reads side by side.  Nothing is shared, so a comparison row is two full pipelines
#: that differ only in the geometry, not two readings of one run.
#:
#:  ``asym``    the original: G = (ρI+Σ)^{-1}, β = -c G b / ‖b‖_G, b from P^asym.
#:  ``fw``      Freidlin–Wentzell: G = a²G_0, β = -c G_0 b, b from the full P.
#:  ``fullb``   the control that separates the two changes — original metric form,
#:              full-P drift.  Only ``ffm``/``pathb`` are run for it.
GEOMETRIES: dict[str, dict] = {
    "asym": {"metric_form": "randers", "drift_mode": "asym"},
    "fw": {"metric_form": "fw", "drift_mode": "full"},
    "fullb": {"metric_form": "randers", "drift_mode": "full"},
    # The quadrature FW action, and the only variant that takes *both* moments from the
    # full P — b and D_rho are then the two forward conditional moments of one P rather
    # than one estimator each from P^asym and P^sym.  ``c = 1.0`` is mandatory, not a
    # tuning choice: lambda is the whole admissibility certificate here
    # (β^T G^{-1} β = 1 - λ²/a²) and a second relaxation on top would be two knobs for
    # one job.  See ``scripts.core.geometry``'s ``fw_lambda`` docstring section.
    "fwlam": {"metric_form": "fw_lambda", "drift_mode": "full",
              "sigma_mode": "full", "c": 1.0},
}
DEFAULT_GEOMETRY = "asym"

#: where each variant's tree lives.  Separate roots rather than a subdirectory of one
#: root: ``runs/<cloud>_<arm>_s<seed>.json`` is already the natural key, and
#: every reporting entry point in the package takes ``--root``, so a variant is a whole
#: report for free instead of a new axis threaded through six modules.
GEOMETRY_OUT = {
    "asym": DEFAULT_OUT,
    "fw": DEFAULT_OUT + "_fw",
    "fullb": DEFAULT_OUT + "_fullb",
    "fwlam": DEFAULT_OUT + "_fwlam",
}
assert set(GEOMETRY_OUT) == set(GEOMETRIES)

#: per-variant sweep grids, merged over :data:`SWEEP_GRIDS`.  Only the arms whose knobs
#: actually change are overridden; ``cfm``/``mfm_land``/``curly`` keep theirs, so the
#: baselines are tuned identically in every tree.
GEOMETRY_GRIDS: dict[str, dict[str, dict[str, list]]] = {
    "asym": {},
    # The FW form has two more knobs and re-uses c and rho with different meanings, so
    # it gets its own grid.  ``rho`` now floors G_0 = (ρI+Σ)^{-1} *inside* a conformal
    # factor that is itself ≈ a_floor in a data void, so the void penalty is the product
    # of the two and they have to be searched together — 0.03 is the value the original
    # metric wants, 1.0 is the literal G_0 = (I+Σ)^{-1} of the FW derivation.
    # ``eps_kernel_scale`` matters far more here than it did before: the full-P first
    # moment is dominated by the local density gradient, which is exactly what the
    # bandwidth controls.
    #
    # Extended after the first pass, which selected ``rho``, ``eps_kernel_scale`` and
    # ``a_floor`` all at a grid *edge* — the optimum was not bracketed, so the reported
    # numbers were a lower bound on the form.  ``rho`` runs a decade past the FW-literal
    # 1.0 because the cost now scales as ρ^{-2} (through a² = bᵀG_0b) where the published
    # metric scaled as ρ^{-1}, so the useful range is both shifted and compressed;
    # ``a_floor`` gains 0.01 below and ``eps_kernel_scale`` 4.0 above for the same reason.
    # Points already on disk from the first pass are re-used, so only the new ones cost.
    "fw": {
        "ffm": {"c": [0.5, 0.9], "rho": [0.03, 1.0, 3.0, 10.0],
                "a_floor": [0.01, 0.05, 0.3],
                "eps_kernel_scale": [0.5, 1.0, 2.0, 4.0]},
        "pathb": {"sigma": [0.05, 0.15, 0.4], "rho": [0.03, 1.0, 3.0, 10.0]},
    },
    # the control changes only the drift estimator, so it keeps the original grid
    "fullb": {},
    # Two knobs, because the quadrature form *has* two: lambda absorbs both ``c`` and
    # ``a_floor``, and ``eps_kernel_scale`` is left at its default so this tree differs
    # from the published one in the metric and not in the smoother.
    #
    # The ranges were measured before they were set, on the Sheet's own training cloud
    # (``fw_lambda_ranges.py``), because neither knob is comparable to its ``fw``
    # namesake:
    #
    # ``fw_lambda`` — lambda is an absolute length but only ever appears against the
    #   data's own ||b||_{G_0}, whose mean ā is 0.033 here.  It is exactly the FW-native
    #   reading of the published ``c``: c_eff = ā/sqrt(ā²+λ²), so the published grid
    #   c ∈ [0.5, 0.9] is λ ∈ [0.057, 0.016] and this grid's four rungs are
    #   c_eff ≈ 0.996 / 0.957 / 0.741 / 0.314 — the published pair bracketed, plus a
    #   near-quasipotential rung below it and a nearly-Riemannian one above.
    # ``rho`` — *not* the ``fw`` tree's shifted-and-stretched [0.03, 10].  That range
    #   was forced by ``a_floor`` and by the ρ^{-2} scaling of an unnormalised cost;
    #   here lambda floors the conformal factor instead, and the global scale of F is
    #   irrelevant anyway (Phase 1's argmin is scale-free and ``sinkhorn_coupling`` sets
    #   reg = blur_frac * median(C)).  What is left is the published meaning of ρ, so
    #   the published range is kept and extended only to the FW-literal ρ = 1.  ā drifts
    #   ×2.8 across it, which reparameterises λ mildly — hence the two are swept jointly
    #   rather than one after the other.
    #
    # Both are widened by half-decades if the argmin lands on an edge, the same rule the
    # ``fw`` tree above was already re-run under -- and both needed it.  The first pass
    # over rho in [0.03, 1] x lambda in [0.003, 0.1] put ``ffm`` at (1.0, 0.1) and
    # ``pathb`` at (1.0, 0.03, sigma 0.05), i.e. on the top rho edge for both, the top
    # lambda edge for ``ffm`` and the bottom sigma edge for ``pathb``.  The estimate
    # above got lambda's *scale* right (0.1 is c_eff ~ 0.3, well inside the interval it
    # bracketed) and rho's ceiling wrong: what the measurement could not see is that the
    # objective would keep improving as the 1-form weakened, and a grid that stops where
    # the trend is still monotone reports a lower bound on the form rather than the form.
    # So rho runs two decades past the published range and lambda one, which reaches
    # c_eff ~ 0.03 -- the 1-form effectively off, F/lambda -> ||v||_{G_0}.  Bracketing
    # that limit is the point: if the argmin sits at it, the honest reading is that the
    # quadrature form's 1-form does not pay here, and that is only sayable with the rung
    # beyond it measured.  Points from the first pass are re-used by configuration, so a
    # widening costs only the new cells.
    "fwlam": {
        "ffm": {"rho": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "fw_lambda": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
        "pathb": {"sigma": [0.02, 0.05, 0.15, 0.4],
                  "rho": [0.03, 0.3, 1.0, 3.0, 10.0],
                  "fw_lambda": [0.01, 0.03, 0.1]},
    },
}
assert set(GEOMETRY_GRIDS) == set(GEOMETRIES)

#: the ``fullb`` control exists to attribute the difference, not to be a full benchmark,
#: so only the two arms whose metric it changes are run for it.  ``fwlam`` is restricted
#: for the same reason: the three arms that never read P would be bit-identical to the
#: published tree's, and the ablations would only re-ask a question ``fw`` already asked.
GEOMETRY_ARMS: dict[str, tuple[str, ...]] = {
    "asym": ARMS,
    "fw": ARMS,
    "fullb": ("ffm", "pathb"),
    "fwlam": ("ffm", "pathb"),
}
assert set(GEOMETRY_ARMS) == set(GEOMETRIES)


def geometry_hp(geometry: str = DEFAULT_GEOMETRY) -> dict:
    """The ``HParams`` overrides that define one geometry variant."""
    assert geometry in GEOMETRIES, (
        f"unknown geometry {geometry!r}, expected {tuple(GEOMETRIES)}")
    return dict(GEOMETRIES[geometry])


def sweep_grid(arm: str, geometry: str = DEFAULT_GEOMETRY) -> dict[str, list]:
    """``SWEEP_GRIDS[arm]`` with this variant's override applied, if it has one."""
    assert geometry in GEOMETRIES, (
        f"unknown geometry {geometry!r}, expected {tuple(GEOMETRIES)}")
    return GEOMETRY_GRIDS[geometry].get(arm, SWEEP_GRIDS[arm])


def grid_points(arm: str, geometry: str = DEFAULT_GEOMETRY) -> list[dict]:
    """Cartesian product of this variant's grid for ``arm``, as HParams overrides."""
    grid = sweep_grid(arm, geometry)
    keys = sorted(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def config_key(over: dict) -> str:
    """Canonical identity of one grid point, independent of its position in the grid."""
    return json.dumps(over, sort_keys=True)


def sweeps_by_config(out_dir: str, name: str, arm: str,
                     geometry: str = DEFAULT_GEOMETRY) -> dict[str, dict]:
    """Every sweep record on disk for one (cloud, arm), keyed by its *configuration*.

    Sweep records are filed under their index in the grid, but that index is a position
    in ``itertools.product`` and therefore moves whenever the grid is extended.  Keying
    the cache by content instead means a grid can grow without either silently reusing
    a record under the wrong hyper-parameters or throwing away work that is still valid.
    """
    found: dict[str, dict] = {}
    for path in glob.glob(os.path.join(out_dir, "sweeps", f"{name}_{arm}_*.json")):
        try:
            with open(path) as f:
                row = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        # a record from another variant is a different experiment, not a cache hit
        if row.get("geometry", DEFAULT_GEOMETRY) != geometry:
            continue
        found[config_key(row["hparams"])] = row
    return found


# --------------------------------------------------------------------------- #
#  Shared plumbing
# --------------------------------------------------------------------------- #
def _prepare(name: str, split_seed: int = SPLIT_SEED):
    """``(ds, split, training_set, {val, test} references)`` for one cloud."""
    ds = make_dataset(name)
    split = make_split(ds, seed=split_seed)
    ts = split_training_set(ds, split)
    refs = {w: split_route_eval(ds, split, w) for w in ("val", "test")}
    return ds, split, ts, refs


def device_name(device: torch.device) -> str:
    """The actual accelerator, not just ``"cuda"``.

    The wall-clock table is only meaningful if every cell ran on the same silicon, and
    ``str(device)`` reports only ``"cuda"``.  Recording the model here lets the report
    assert homogeneity instead of assuming it.
    """
    if device.type != "cuda":
        return str(device)
    return f"cuda:{torch.cuda.get_device_name(device.index or 0)}"


def _fit(arm: str, ts, hp: HParams, seed: int, device: torch.device,
         dtype: torch.dtype, n_steps: int) -> tuple:
    """Train one arm and return ``(result, wall_clock_seconds)``.

    CUDA is asynchronous, so the launch queue must be drained on both sides or the
    number reported is how fast Python got through the loop, not how long the GPU
    worked.  The measured span is the whole fit call: every training phase plus the one
    terminal pushforward integration the arm performs.  Scoring and I/O are outside it.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = run_arm(arm, ts, hp, seed, device, dtype, n_steps=n_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
#  tune
# --------------------------------------------------------------------------- #
def rank_cloud(name: str, arms: tuple[str, ...], out_dir: str,
               geometry: str = DEFAULT_GEOMETRY) -> dict:
    """Rank an already-swept cloud and write its tuned point.

    Pure bookkeeping over ``sweeps/*.json`` — no training.  Because every point stores
    its terms separately, re-ranking a finished sweep (``tune --rank-only``) costs
    nothing and never retrains, so a changed objective is a re-rank of the *same*
    trained models rather than a second, independently noisy search.
    """
    ranking: dict[str, list[dict]] = {}
    for arm in arms:
        rows = []
        for i, over in enumerate(grid_points(arm, geometry)):
            path = os.path.join(out_dir, "sweeps", f"{name}_{arm}_{i}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                row = json.load(f)
            # a record left behind by a smaller grid still sits at its old index; rank
            # only what the *current* grid asked for, or the winner could be a point
            # that is no longer in the search space
            if config_key(row["hparams"]) != config_key(over):
                print(f"  [stale] {name}_{arm}_{i}  {json.dumps(row['hparams'])} "
                      f"is not the current grid point {json.dumps(over)}", flush=True)
                continue
            row["objective"] = objective_from(row)
            rows.append(row)
        assert rows, f"no swept points for {arm} on {name} under {out_dir}/sweeps"
        rows.sort(key=lambda r: r["objective"])
        ranking[arm] = rows

    tuned = {arm: rows[0]["hparams"] for arm, rows in ranking.items()}
    for child, parent in INHERIT.items():
        if parent in tuned:
            tuned[child] = dict(tuned[parent])

    _atomic_write_json(tuned, os.path.join(out_dir, "tuned", f"{name}.json"))
    _atomic_write_json(ranking,
                       os.path.join(out_dir, "tuned", f"{name}_report.json"))
    print(f"[tuned] {name} ->  " + "  ".join(
        f"{a}={json.dumps(tuned[a])}" for a in sorted(ranking)), flush=True)
    return tuned


def tune_cloud(name: str, arms: tuple[str, ...], device: torch.device,
               out_dir: str, seed: int = 0, dtype: torch.dtype = torch.float32,
               extra_hp: dict | None = None, force: bool = False,
               geometry: str = DEFAULT_GEOMETRY) -> dict:
    """Run every grid point of every tuned arm on one cloud, then rank them on val."""
    ds, split, ts, _refs = _prepare(name)
    print(f"[tune] {ds.summary()}\n       split {json.dumps(split.counts()['train'])}"
          f"\n       geometry={geometry} {json.dumps(geometry_hp(geometry))}", flush=True)

    for arm in arms:
        # content-addressed view of whatever is already on disk, so that extending the
        # grid re-indexes the records instead of invalidating (or worse, mismatching) them
        prior = {} if force else sweeps_by_config(out_dir, name, arm, geometry)
        for i, over in enumerate(grid_points(arm, geometry)):
            key = f"{name}_{arm}_{i}"
            path = os.path.join(out_dir, "sweeps", f"{key}.json")
            done = prior.get(config_key(over))
            if done is not None:
                # a record written before a term existed cannot be ranked at all; say so
                # here rather than at ranking time
                if all(k in done for k in OBJECTIVE_TERMS):
                    # the point may have moved index when the grid grew; re-file it so
                    # rank_cloud, which walks indices, still sees it
                    if done.get("point") != i:
                        done["point"] = i
                        _atomic_write_json(done, path)
                        print(f"  [move] {key}  {json.dumps(over)}", flush=True)
                    else:
                        print(f"  [skip] {key}  {objective_from(done):.4f}", flush=True)
                    continue
                print(f"  [redo] {key}  predates {OBJECTIVE_TERMS}", flush=True)
            # geometry first: it defines the variant, the grid point refines it, and
            # --smoke overrides both
            hp = HParams(**{**geometry_hp(geometry), **over, **(extra_hp or {})})
            try:
                result, wall = _fit(arm, ts, hp, seed, device, dtype, N_STEPS)
                terms = val_objective(result.traj, ds, split, seed=seed)
            except Exception:
                traceback.print_exc()
                continue
            row = {"cloud": name, "arm": arm, "point": i, "hparams": over,
                   "geometry": geometry, "train_s": round(wall, 2), **terms}
            _atomic_write_json(row, path)
            print(f"  {arm:9s} {json.dumps(over):48s} obj={terms['objective']:.4f}"
                  f"  ({wall:.0f}s)", flush=True)

    return rank_cloud(name, arms, out_dir, geometry=geometry)


def load_tuned(cloud: str, out_dir: str = DEFAULT_OUT) -> dict[str, dict]:
    """Hyper-parameters for ``cloud``, from whichever cloud's sweep it inherits.

    Not geometry-aware on purpose: each variant writes to its own ``out_dir``
    (see :data:`GEOMETRY_OUT`), so the path already selects the right sweep and
    a variant can never silently read another variant's tuned values.
    """
    src = SWEEP_SOURCE[cloud]
    path = os.path.join(out_dir, "tuned", f"{src}.json")
    assert os.path.exists(path), (
        f"{cloud} inherits {src}'s sweep, but {path} does not exist — run "
        f"`python -m scripts.experiments.sheet.paper_run tune --clouds {src}` first")
    with open(path) as f:
        tuned = json.load(f)
    unknown = set(tuned) - set(ARMS)
    assert not unknown, f"{path} names arms that do not exist: {unknown}"
    return tuned


# --------------------------------------------------------------------------- #
#  test
# --------------------------------------------------------------------------- #
def test_cell(name: str, arm: str, seed: int, device: torch.device, out_dir: str,
              dtype: torch.dtype = torch.float32, n_steps: int = N_STEPS,
              extra_hp: dict | None = None, prep=None,
              geometry: str = DEFAULT_GEOMETRY) -> dict:
    """Train on the train split, score on the test split, keep every artefact."""
    ds, split, ts, refs = prep if prep is not None else _prepare(name)
    over = {**geometry_hp(geometry), **load_tuned(name, out_dir).get(arm, {})}
    if extra_hp:
        over.update(extra_hp)
    hp = HParams(**over)

    result, wall = _fit(arm, ts, hp, seed, device, dtype, n_steps)
    metrics = {w: evaluate_route(result.traj, ds, refs[w], v_net=result.v_net,
                                 device=device, dtype=dtype, seed=seed)
               for w in ("test", "val")}

    key = f"{name}_{arm}_s{seed}"
    save_weights(result, hp, name, arm, seed, ds.d,
                 os.path.join(out_dir, "weights", f"{key}.pt"))
    np.savez_compressed(
        os.path.join(_ensure(out_dir, "trajectories"), f"{key}.npz"),
        traj=result.traj.astype(np.float32))

    record = {
        "spec": {"dataset": name, "arm": arm, "seed": seed, "n_steps": n_steps,
                 "split_seed": SPLIT_SEED, "hparam_source": SWEEP_SOURCE[name],
                 "geometry": geometry},
        "hparams": hp.as_dict(),
        "protocol": ts.diag,
        "metrics": metrics["test"],            # the reported column
        "metrics_val": metrics["val"],
        "arm_diag": _jsonable(result.diag),
        "train_s": round(wall, 2),
        "device": device_name(device),
    }
    _atomic_write_json(record, os.path.join(out_dir, "runs", f"{key}.json"))
    return record


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --out defaults to None, not DEFAULT_OUT, so main() can tell "user said
    # nothing" (→ derive the root from --geometry) from "user named a root"
    p.add_argument("--out", default=None,
                   help=f"output root; defaults to the --geometry entry of "
                        f"{GEOMETRY_OUT}")
    p.add_argument("--geometry", default=DEFAULT_GEOMETRY, choices=list(GEOMETRIES),
                   help="which P-derived metric to build: 'asym' is the published "
                        "P^asym Randers form, 'fw' is the Freidlin-Wentzell form on "
                        "the full-P drift, 'fullb' is the estimator-only control")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="200-iteration wiring check; the numbers are not reportable")
    p.add_argument("--force", action="store_true", help="rerun cells that already exist")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tune", help="grid search, ranked on the val split")
    t.add_argument("--clouds", nargs="+", default=list(SWEEP_CLOUDS),
                   choices=list(DATASET_NAMES))
    t.add_argument("--arms", nargs="+", default=list(TUNED_ARMS), choices=list(TUNED_ARMS))
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--rank-only", action="store_true",
                   help="re-rank the existing sweep from its stored terms; trains nothing")

    r = sub.add_parser("test", help="the reported runs, scored on the test split")
    r.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES),
                   choices=list(DATASET_NAMES))
    r.add_argument("--arms", nargs="+", default=list(MAIN_ARMS), choices=list(ARMS))
    r.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    r.add_argument("--n-steps", type=int, default=N_STEPS)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = GEOMETRY_OUT[args.geometry]
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    extra = dict(SMOKE_HPARAMS) if args.smoke else None
    banner = "  [SMOKE — numbers are not reportable]" if args.smoke else ""

    if args.cmd == "tune":
        if args.rank_only:
            print(f"[paper_run tune] re-ranking {len(args.clouds)} cloud(s) "
                  f"from stored terms — no training")
            for cloud in args.clouds:
                rank_cloud(cloud, tuple(args.arms), args.out, geometry=args.geometry)
            return
        print(f"[paper_run tune] {len(args.clouds)} cloud(s) on {device}"
              f"  geometry={args.geometry}{banner}")
        for cloud in args.clouds:
            tune_cloud(cloud, tuple(args.arms), device, args.out, seed=args.seed,
                       extra_hp=extra, force=args.force, geometry=args.geometry)
        return

    # a variant may not implement every arm (the 'fullb' control is ffm/pathb
    # only); silently dropping is wrong, so refuse loudly instead
    allowed = GEOMETRY_ARMS[args.geometry]
    unsupported = [a for a in args.arms if a not in allowed]
    assert not unsupported, (
        f"geometry {args.geometry!r} does not define arms {unsupported}; "
        f"it covers {allowed}")

    cells = [(d, a, s) for d in args.datasets for a in args.arms for s in args.seeds]
    print(f"[paper_run test] {len(cells)} cell(s) on {device}"
          f"  geometry={args.geometry}{banner}")
    preps = {d: _prepare(d) for d in args.datasets}
    for d, (ds, split, _ts, _refs) in preps.items():
        print("  " + ds.summary() + "  |  " + "  ".join(
            f"{w}={split.counts()[w]['all']}" for w in ("train", "val", "test")))

    failed: list[str] = []
    for dataset, arm, seed in cells:
        key = f"{dataset}_{arm}_s{seed}"
        path = os.path.join(args.out, "runs", f"{key}.json")
        if os.path.exists(path) and not args.force:
            print(f"[skip] {key}")
            continue
        print(f"[run ] {key}", flush=True)
        try:
            rec = test_cell(dataset, arm, seed, device, args.out, n_steps=args.n_steps,
                            extra_hp=extra, prep=preps[dataset],
                            geometry=args.geometry)
        except Exception:
            traceback.print_exc()
            failed.append(key)
            continue
        m = rec["metrics"]
        print(f"[done] {key}  {rec['train_s']}s  "
              f"end W2={m['endpoint']['W2']:.4f}  mid W2={m['intermediate']['W2']:.4f}  "
              f"cos={m['drift']['cos']:+.3f}  "
              f"gap={m['route']['intermediate']['gap']:.2f}", flush=True)
    if failed:
        print(f"[paper_run] {len(failed)} cell(s) FAILED:\n  " + "\n  ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
