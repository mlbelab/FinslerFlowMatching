"""Path B's two noise knobs, ablated: the path noise sigma and the sampler's g(t).

    python -m scripts.experiments.pancreas.noise run
    python -m scripts.experiments.pancreas.noise report

The reported Path-B column fixes four things at once, and the table cannot tell them
apart: how much noise the conditional path carries, what *shape* that noise has, how much
the sampler injects at each time, and what shape *that* has.  This module varies the four
independently on Pancreas ``d = 20`` at the notebook's tuned FFM point, on a Path-A
backbone that is trained once per seed and then frozen -- so every cell below differs from
the shipped ``pathb`` by the noise alone and not by its interpolant or its coupling.

The four axes
-------------
``width_mult``   the conditional path noise, as :data:`~scripts.experiments.pancreas.tune
                 .SIGMA_MULTS` -- the same ladder the search ranked ``pathb`` on, so the
                 tuned rung is one of the rows rather than a point beside them.
``path_cov``     the conditional covariance ``A_t = sigma^2 t(1-t) C``, with
                 ``C = M_t = C_rho(mu_t)`` (ours) or ``C = I`` (the control).  This is a
                 *training* switch: it moves the CFM target's 1/2 d_tM M^-1 delta term and
                 the score target's M^-1, so each setting gets its own (v_theta, s_phi).
``diffusion``    the tensor the sampler shapes its step with, ``M(X_t)`` or ``I``.  This is
                 an *inference* switch over an already-trained pair, so the two readings
                 share a bridge and differ by the pushforward alone.
``schedule``     g(t), the time profile of the injected noise -- see :data:`SCHEDULES`.

Crossing ``path_cov`` with ``diffusion`` is the point of the middle table: ``pathb`` is
(M, M) and ``pathb_iso`` is (I, I), and the two off-diagonal cells are what say which of
the two the anisotropy is actually bought by.

Two conventions, so that a row varies one thing
-----------------------------------------------
**Amount.**  sigma is never a bare number: a rung is a *half-width at t = 1/2 in kernel
bandwidths*, resolved against whichever mobility is in force by
:func:`~scripts.method.metric.sigma_for`.  ``m_bar = tr M / d`` is not 1 on this cloud, so
quoting one sigma to both the M and the I arms would hand them different amounts of noise
and call the difference anisotropy.  Resolving per mobility injects the same half-width
either way and leaves only the *shape* to differ, which is the comparison being made.

**Schedule.**  Every schedule is rescaled on the integration grid so that
``mean_n g_n^2 = sigma^2`` exactly -- the same total injected variance, redistributed in
time.  Without that, "noise late is worse" and "more noise is worse" are the same column.
``off`` is the exception and the zero control: it is the probability-flow limit of a pair
that was *trained* noisy.

What is deliberately **not** varied: the drift the sampler steps.  It is
``v_theta + g^2 M s_phi + g^2 div M`` term for term as
:func:`scripts.core.bridge.integrate_sde` writes it, with g(t) in place of the constant
sigma, so the ``const`` / matched-diffusion cell reproduces the shipped sampler
bit-for-bit -- the verification suite in the research repository asserts exactly that, which is
what makes the other cells readable as departures from a known row rather than as output
of a second sampler.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from scripts.core.bridge import metric_sqrt, mobility_divergence
from scripts.core.paths import experiment_root, table_file
from scripts.method import (ConstantMobility, FWFactory, Run, moments_from, path_a,
                            sigma_for, train_sbm)

from .datasets import build_training_set, load_cloud
from .train import Scorer, load_state, save_state
from .tune import SIGMA_MULTS

#: the dimension the ablation runs at -- the reported real-data column, as in
#: :mod:`scripts.experiments.matched.pancreas`.  One dimension and not four: this is a
#: study of Path B's noise, and running it everywhere would quadruple a 50-minute job to
#: answer a question no column of it asks.
DIM = 20
KEY = f"pancreas{DIM}"

CELL_DIR = "cells"
NETS_DIR = "nets"

#: the path-noise ladder, taken from the search rather than restated, so the tuned rung is
#: a row of the table and the extent of the ablation is the extent the tuner saw.
WIDTH_MULTS = SIGMA_MULTS

#: the conditional covariance of the bridge (a training switch), and the tensor the
#: sampler shapes its step with (an inference switch).  ``M`` is the data's mobility
#: ``C_rho = D~ + rho I``; ``I`` is the identity, reached with the same
#: :class:`~scripts.core.bridge.ConstantMobility` wrapper the engine's frozen-M control
#: uses, so d_tM and div M are structurally zero in the I arms instead of numerically small.
PATH_COVS = ("M", "I")
DIFFUSIONS = ("M", "I")
COV_LABELS = {"M": "M_t = C_rho", "I": "I"}
DIFF_LABELS = {"M": "M(X_t)", "I": "I"}

TAIL_ON = 0.8

#: ``name -> (s(t), label)``.  ``s`` is the *shape*; :func:`schedule_grid` normalises it on
#: the integration grid so every schedule injects the same total variance.  ``off`` is not
#: normalised -- it is zero, the deterministic control.
SCHEDULES = {
    "const": (lambda t: np.ones_like(t), "g = sigma"),
    "down": (lambda t: 1.0 - t, "g ~ (1-t)"),
    "up": (lambda t: t, "g ~ t"),
    "bridge": (lambda t: np.sqrt(np.clip(t * (1.0 - t), 0.0, None)),
               "g ~ sqrt(t(1-t))"),
    "tail": (lambda t: (t < TAIL_ON).astype(float), f"g = 0 after t = {TAIL_ON:g}"),
    "off": (lambda t: np.zeros_like(t), "g = 0"),
}

#: the cell every other one is read against: the shipped Path-B sampler.
PRIMARY = ("const", "M")


def root_dir(smoke: bool = False) -> str:
    return experiment_root("pancreas_noise", smoke)


ROOT = root_dir()


def cell_path(root: str, seed: int, path_cov: str, mult: float) -> str:
    return os.path.join(root, CELL_DIR, f"d{DIM}_{path_cov}_w{mult:g}_s{seed}.json")


def state_path(root: str, seed: int) -> str:
    return os.path.join(root, NETS_DIR, f"d{DIM}_ffm_s{seed}.pt")


def variant_key(schedule: str, diffusion: str) -> str:
    return f"{schedule}|{diffusion}"


def cells(seeds) -> list[tuple[int, str, float]]:
    return [(s, c, m) for s in seeds for c in PATH_COVS for m in WIDTH_MULTS]


# --------------------------------------------------------------------------- #
#  g(t)
# --------------------------------------------------------------------------- #
def schedule_grid(name: str, n_steps: int) -> np.ndarray:
    """``g_n / sigma`` at the ``n_steps`` left endpoints, with ``mean_n g_n^2 = sigma^2``.

    Normalised on the grid the sampler actually steps on rather than by the continuous
    integral of ``s^2``: the two differ by an O(1/n) quadrature error, and an ablation
    whose rows are meant to hold the injected variance fixed should hold the *realised*
    one fixed, not the one a finer grid would have had.  ``const`` comes back exactly 1,
    which is what lets that row reproduce :func:`scripts.core.bridge.integrate_sde`.
    """
    assert name in SCHEDULES, f"unknown schedule {name!r}; expected one of {sorted(SCHEDULES)}"
    t = np.arange(n_steps, dtype=np.float64) / n_steps
    s = np.asarray(SCHEDULES[name][0](t), dtype=np.float64)
    assert s.shape == t.shape and np.all(s >= 0.0), f"{name} is not a non-negative shape"
    rms = float(np.sqrt((s ** 2).mean()))
    return s if rms == 0.0 else s / rms


@torch.no_grad()
def integrate_scheduled(v_net, s_net, geom, x0, sigma: float, g: np.ndarray,
                        n_steps: int, seed: int, t_eps: float):
    """:func:`scripts.core.bridge.integrate_sde` with a time-varying noise level.

    ``g`` is the per-step multiplier from :func:`schedule_grid`, so the level on the step
    out of ``t = n dt`` is ``g_n sigma`` and the whole drift moves with it:

        dX = [v_theta + g_n^2 M s_phi + g_n^2 div M] dt + sqrt(2 dt) g_n M^{1/2} z.

    The loop is the engine's, step for step and draw for draw -- same clamp of the network
    time to the trained window, same order of the three drift terms, same single
    ``randn`` per step -- so at ``g == 1`` it returns the engine's trajectory exactly.
    Transcribed rather than called because the engine's sampler takes one scalar sigma and
    a cut-off time, and threading a schedule through it would change the signature every
    benchmark in the repository integrates with.
    """
    assert len(g) == n_steps, f"g has {len(g)} rungs for {n_steps} steps"
    device, dtype = x0.device, x0.dtype
    dt = 1.0 / n_steps
    gen = torch.Generator(device=device).manual_seed(seed)
    x = x0.clone()
    traj = [x.clone()]
    for n in range(n_steps):
        g_n = float(sigma) * float(g[n])
        t_net = min(max(n * dt, t_eps), 1.0 - t_eps)
        t0 = torch.full((x.shape[0], 1), t_net, device=device, dtype=dtype)
        M = geom.metric_tensor_inv(x)
        drift = v_net(t0, x)
        if g_n > 0.0:
            score = s_net(t0, x)
            drift = drift + (g_n ** 2) * torch.einsum("bij,bj->bi", M, score)
            drift = drift + (g_n ** 2) * mobility_divergence(geom, x)
        x = x + dt * drift
        if g_n > 0.0:
            L = metric_sqrt(M)
            dw = torch.randn(x.shape, generator=gen, device=device, dtype=dtype)
            x = x + float(np.sqrt(2.0 * dt) * g_n) * torch.einsum("bij,bj->bi", L, dw)
        traj.append(x.clone())
    return torch.stack(traj, dim=0)


# --------------------------------------------------------------------------- #
#  One cell: one (seed, path covariance, sigma) bridge, every inference variant
# --------------------------------------------------------------------------- #
def mobilities(metric, cfg: Run) -> dict:
    """``{"M": the data's mobility, "I": the same object with M frozen at the identity}``.

    The identity arm is :class:`~scripts.core.bridge.ConstantMobility` and not a hand-
    written stub for the reason that class exists: ``mobility``, ``mobility_inv``,
    ``metric_tensor_inv`` and the two vanishing terms then agree by construction, and
    ``mean_mobility_eig`` returns 1, which is what makes :func:`sigma_for` quote the
    width-matched sigma for it.
    """
    return {"M": metric,
            "I": ConstantMobility(metric, torch.eye(metric.d, **cfg.torch_kw))}


def run_cell(seed: int, path_cov: str, mult: float, *, metric, phi, coupler, X0_t,
             score: Scorer, cfg: Run, point: dict, n_steps: int) -> dict:
    """Train one bridge and push the source cloud through every inference variant."""
    geoms = mobilities(metric, cfg)
    sigma_path = sigma_for(geoms[path_cov], mult)

    t0 = time.time()
    v_net, s_net = train_sbm(phi, geoms[path_cov], coupler, sigma_path, seed, cfg)
    train_s = time.time() - t0

    variants = {}
    for schedule in SCHEDULES:
        g = schedule_grid(schedule, n_steps)
        for diffusion in DIFFUSIONS:
            # the amount is re-quoted against the tensor the sampler shapes with, so a
            # (path, diffusion) pair differs in shape and never in half-width
            sigma_inf = sigma_for(geoms[diffusion], mult)
            t1 = time.time()
            traj = integrate_scheduled(v_net, s_net, geoms[diffusion], X0_t, sigma_inf, g,
                                       n_steps, seed, cfg.sb_t_eps)
            traj = traj.cpu().numpy().astype(np.float64)
            m = score(traj, seed=seed)
            variants[variant_key(schedule, diffusion)] = {
                "schedule": schedule, "diffusion": diffusion,
                "sigma": float(sigma_inf),
                "intermediate": m["intermediate"], "endpoint": m["endpoint"],
                "seconds": round(time.time() - t1, 2),
                "finite": bool(np.isfinite(traj[-1]).all()),
            }

    return {
        "key": KEY, "dim": DIM, "seed": seed, "path_cov": path_cov,
        "width_mult": float(mult), "sigma_path": float(sigma_path),
        "tuned_width_mult": float(point["pathb"]["width_mult"]),
        "point": point, "budget": _budget(cfg, n_steps),
        "m_bar": float(metric.mean_mobility_eig()),
        "eps_kernel": float(metric.eps_kernel),
        "variants": variants, "floors": score.floors,
        "train_seconds": round(train_s, 1), "smoke": bool(cfg.smoke),
    }


def _budget(cfg: Run, n_steps: int) -> dict:
    return {"geo_iters": cfg.geo_iters, "cfm_iters": cfg.cfm_iters,
            "sb_iters": cfg.sb_iters, "n_steps": n_steps,
            "width": cfg.net_width, "depth": cfg.net_depth}


def _fresh(path: str, point: dict, budget: dict) -> bool:
    """A cell is reusable only if it ran at this tuned point and this budget."""
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        prior = json.load(fh)
    return (prior.get("point") == point and prior.get("budget") == budget
            and set(prior.get("variants", {})) == {variant_key(s, d)
                                                   for s in SCHEDULES for d in DIFFUSIONS})


def missing(cfg: Run, point: dict, root: str | None = None, seeds=None,
            n_steps: int | None = None) -> list[tuple[int, str, float]]:
    """The cells disk cannot answer at this point and budget."""
    root = root_dir(cfg.smoke) if root is None else root
    seeds = cfg.seeds if seeds is None else seeds
    n_steps = cfg.n_steps if n_steps is None else n_steps
    budget = _budget(cfg, n_steps)
    return [c for c in cells(seeds) if not _fresh(cell_path(root, *c), point, budget)]


def ensure(cfg: Run, tuned: dict, root: str | None = None, seeds=None,
           n_steps: int | None = None, force: bool = False, verbose: bool = False) -> dict:
    """Every cell at every seed, training only what is not already on disk.

    ``tuned`` is one dimension's tuned block -- ``{"ffm": {...}, "pathb": {...}, ...}`` --
    passed in rather than read off :mod:`scripts.experiments.pancreas.tune` so the notebook
    ablates the point its own tables were produced at, and a drifted ``tuned.json`` cannot
    silently move the centre of the ladder out from under them.
    """
    root = root_dir(cfg.smoke) if root is None else root
    seeds = tuple(cfg.seeds if seeds is None else seeds)
    n_steps = cfg.n_steps if n_steps is None else n_steps
    point = {"ffm": dict(tuned["ffm"]), "pathb": dict(tuned["pathb"])}
    budget = _budget(cfg, n_steps)
    todo = [c for c in cells(seeds)
            if force or not _fresh(cell_path(root, *c), point, budget)]
    if todo:
        _fill(todo, cfg, point, root, seeds, n_steps, verbose)
    recs = load_records(root, seeds)
    save_tables(recs, cfg.smoke)
    return recs


def _fill(todo, cfg: Run, point: dict, root: str, seeds, n_steps: int,
          verbose: bool) -> None:
    cloud = load_cloud(DIM)
    ts = build_training_set(cloud)
    X = np.asarray(ts.X, dtype=np.float64)
    b0_pts, D_pts = moments_from(ts.P, X)
    metric = FWFactory(X, D_pts, b0_pts, cfg, **point["ffm"]).base

    X0_t, X1_t = cfg.tensor(X[ts.p0]), cfg.tensor(X[ts.p1])
    # wf=False: W_{2,F} needs a judge geodesic, and every row here shares one Path A per
    # seed, so the Finsler ruler would report the same cost function against itself
    score = Scorer(cloud, None, cfg, len(X0_t), wf=False)

    want = sorted({s for s, _, _ in todo})
    states = {}
    for seed in want:
        sp = state_path(root, seed)
        # the backbone is Path A at the tuned FFM point and is shared by every cell of
        # this seed, exactly as the reported pathb column shares the ffm column's
        if os.path.exists(sp):
            states[seed] = load_state(sp, metric.d, seed, cfg)
        else:
            det = path_a(metric, X0_t, X1_t, seed, cfg)
            states[seed] = (det["phi"], det["coupler"])
            save_state(*states[seed], sp)
        if verbose:
            print(f"[noise] backbone seed {seed} ready")

    for seed, path_cov, mult in todo:
        phi, coupler = states[seed]
        rec = run_cell(seed, path_cov, mult, metric=metric, phi=phi, coupler=coupler,
                       X0_t=X0_t, score=score, cfg=cfg, point=point, n_steps=n_steps)
        path = cell_path(root, seed, path_cov, mult)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
        if verbose:
            v = rec["variants"][variant_key(*PRIMARY)]
            print(f"[noise] s{seed} cov {path_cov} w {mult:<6g} sigma "
                  f"{rec['sigma_path']:.4f}  mid W2 {v['intermediate']['W2']:.4f}  "
                  f"({rec['train_seconds']:.0f}s)")


def load_records(root: str | None = None, seeds=(0, 1, 2)) -> dict:
    root = ROOT if root is None else root
    out = {}
    for c in cells(seeds):
        p = cell_path(root, *c)
        if os.path.exists(p):
            with open(p) as fh:
                out[c] = json.load(fh)
    return out


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
def _seeds_of(recs: dict) -> list[int]:
    return sorted({s for s, _, _ in recs})


def _tuned_mult(recs: dict) -> float:
    m = {r["tuned_width_mult"] for r in recs.values()}
    assert len(m) == 1, f"records disagree about the tuned rung: {sorted(m)}"
    return m.pop()


def _col(recs: dict, path_cov: str, mult: float, key: str, where: str,
         field: str) -> np.ndarray:
    """One metric over the seeds that carry this cell."""
    vals = [recs[(s, path_cov, mult)]["variants"][key][where][field]
            for s in _seeds_of(recs) if (s, path_cov, mult) in recs]
    return np.array(vals, dtype=float)


def _stats(recs, path_cov, mult, key) -> dict:
    mid = _col(recs, path_cov, mult, key, "intermediate", "W2")
    end = _col(recs, path_cov, mult, key, "endpoint", "W2")
    sd = lambda a: a.std(ddof=1) if len(a) > 1 else np.nan
    return {"mid W2": mid.mean(), "mid sd": sd(mid),
            "end W2": end.mean(), "end sd": sd(end), "n seeds": len(mid)}


def frame_sigma(recs: dict) -> pd.DataFrame:
    """The path-noise ladder, both conditional covariances side by side.

    One row per sigma rung; the sampler is the matched one (``const``, diffusion shaped
    like the covariance the bridge was trained under), so the pair of column blocks is
    ``pathb`` against ``pathb_iso`` at every noise level the tuner searched.
    """
    have = sorted({m for _, _, m in recs})
    rows = {}
    for mult in have:
        row = {}
        for cov in PATH_COVS:
            if not any((s, cov, mult) in recs for s in _seeds_of(recs)):
                continue
            st = _stats(recs, cov, mult, variant_key("const", cov))
            sig = np.mean([recs[(s, cov, mult)]["sigma_path"]
                           for s in _seeds_of(recs) if (s, cov, mult) in recs])
            row[(COV_LABELS[cov], "sigma")] = sig
            for c in ("mid W2", "mid sd", "end W2", "end sd"):
                row[(COV_LABELS[cov], c)] = st[c]
        rows[mult] = row
    out = pd.DataFrame(rows).T
    out.index.name = "width_mult"
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["path covariance", ""])
    return out


def frame_cov(recs: dict, mult: float | None = None) -> pd.DataFrame:
    """The 2 x 2: what the covariance buys in the path, and what it buys in the sampler.

    ``(M, M)`` is the shipped Path B and ``(I, I)`` is the isotropic control; the two
    off-diagonal cells separate the training switch from the inference one.  ``d mid W2``
    is against ``(M, M)`` at the same seeds -- negative favours the row.
    """
    mult = _tuned_mult(recs) if mult is None else mult
    base_mid = _col(recs, "M", mult, variant_key("const", "M"), "intermediate", "W2")
    base_end = _col(recs, "M", mult, variant_key("const", "M"), "endpoint", "W2")
    rows = []
    for cov in PATH_COVS:
        for diff in DIFFUSIONS:
            if not any((s, cov, mult) in recs for s in _seeds_of(recs)):
                continue
            st = _stats(recs, cov, mult, variant_key("const", diff))
            mid = _col(recs, cov, mult, variant_key("const", diff), "intermediate", "W2")
            end = _col(recs, cov, mult, variant_key("const", diff), "endpoint", "W2")
            rows.append({"path covariance": COV_LABELS[cov], "diffusion": DIFF_LABELS[diff],
                         **st,
                         "d mid W2": (mid - base_mid).mean(),
                         "d end W2": (end - base_end).mean()})
    return pd.DataFrame(rows).set_index(["path covariance", "diffusion"])


def frame_schedule(recs: dict, mult: float | None = None,
                   path_cov: str = "M") -> pd.DataFrame:
    """g(t) at the tuned rung, at equal injected variance, under both diffusion shapes.

    The bridge is the same trained pair in every row: only the sampler moves.  ``const``
    with the matched diffusion is the shipped Path B and the column ``d mid W2`` is read
    against it.
    """
    mult = _tuned_mult(recs) if mult is None else mult
    base_mid = _col(recs, path_cov, mult, variant_key(*PRIMARY), "intermediate", "W2")
    base_end = _col(recs, path_cov, mult, variant_key(*PRIMARY), "endpoint", "W2")
    rows = []
    for schedule, (_, label) in SCHEDULES.items():
        for diff in DIFFUSIONS:
            key = variant_key(schedule, diff)
            mid = _col(recs, path_cov, mult, key, "intermediate", "W2")
            end = _col(recs, path_cov, mult, key, "endpoint", "W2")
            if not len(mid):
                continue
            rows.append({"g(t)": label, "diffusion": DIFF_LABELS[diff],
                         **_stats(recs, path_cov, mult, key),
                         "d mid W2": (mid - base_mid).mean(),
                         "d end W2": (end - base_end).mean()})
    return pd.DataFrame(rows).set_index(["g(t)", "diffusion"])


FRAME_NAMES = ("sigma", "covariance", "schedule")

#: the rendered tables ship beside the notebook, not in the gitignored run tree -- see
#: :func:`~scripts.core.paths.table_file`, whose ``.json`` name this takes the stem of.
TABLE_STEM = "pancreas_noise"


def frames(recs: dict):
    return frame_sigma(recs), frame_cov(recs), frame_schedule(recs)


def render(recs: dict) -> str:
    out = [header(recs), ""]
    for f in frames(recs):
        out += [f.round(4).to_string(), ""]
    return "\n".join(out)


def save_tables(recs: dict, smoke: bool = False) -> str:
    """The three tables as printed, plus a CSV each so a reader need not re-parse them."""
    stem = table_file(TABLE_STEM, smoke)[:-len(".json")]
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".txt", "w") as fh:
        fh.write(render(recs) + "\n")
    for name, f in zip(FRAME_NAMES, frames(recs)):
        f.round(6).to_csv(f"{stem}_{name}.csv")
    return stem + ".txt"


def header(recs: dict) -> str:
    any_rec = next(iter(recs.values()))
    fl = any_rec["floors"]
    return (f"===== Pancreas, d = {any_rec['dim']}, Path-B noise; "
            f"tuned rung width_mult = {_tuned_mult(recs):g}, "
            f"m_bar = {any_rec['m_bar']:.4f}, seeds {tuple(_seeds_of(recs))}\n"
            "every schedule carries the same injected variance; sigma is width-matched "
            "to the tensor it is shaped by\n"
            "two-sample floors:  " + "   ".join(
                f"{k} W2 {v['W2']:.4f}" for k, v in fl.items()))


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def tuned_point(dim: int = DIM) -> dict:
    """The point the CLI ablates around: the **notebook's** ``TUNED`` literal.

    Read from the notebook and not from ``tuned.json`` beside the tuner, because the cell
    calls :func:`ensure` with the notebook's literal and a record is only reusable at the
    point it was run at.  Taking the tuner's copy here would mean a CLI pass and a notebook
    pass each retrain all 48 bridges and neither ever finds the other's records -- and
    silently, since both are "the tuned point".  The two copies are supposed to agree, and
    the verification suite in the research repository has a check that fails when they do not;
    while they disagree, the ablation follows the table it is explaining.
    """
    nb = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "../../../notebooks/pancreas_ffm_paper.ipynb")
    if os.path.exists(nb):
        with open(nb) as fh:
            src = "\n".join("".join(c["source"]) for c in json.load(fh)["cells"]
                            if c["cell_type"] == "code")
        m = re.search(r"^TUNED = \{.*?^\}", src, re.M | re.S)
        assert m, "the notebook has no TUNED literal"
        ns: dict = {}
        exec(m.group(0), ns)                                  # noqa: S102 - our own text
        return {int(k): v for k, v in ns["TUNED"].items()}[dim]

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")
    assert os.path.exists(path), f"no notebook and no tuned point at {path}"
    with open(path) as fh:
        return json.load(fh)[str(dim)]


def cmd_run(args) -> None:
    cfg = Run.from_env()
    if args.cpu:
        cfg = cfg.but(device=torch.device("cpu"))
    recs = ensure(cfg, tuned_point(), root=args.root, force=args.force, verbose=True)
    _print(recs)


def cmd_report(args) -> None:
    root = ROOT if args.root is None else args.root
    recs = load_records(root)
    if not recs:
        raise SystemExit(f"no noise-ablation records under {os.path.join(root, CELL_DIR)}")
    gone = [c for c in cells((0, 1, 2)) if c not in recs]
    if gone:
        print(f"warning: {len(gone)} of {len(cells((0, 1, 2)))} cells missing")
    _print(recs)


def _print(recs: dict) -> None:
    print(render(recs))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("report")
    c.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
