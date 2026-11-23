import contextlib
import io
import json
import os

import pandas as pd

from ffm.core.sensitivity import window
from scripts import records as rec
from scripts.matched import bench, hamiltonian

from . import band, fate, leakage, noise, train, tune
from .datasets import load_cloud

TREE = "trainonly"
PAPER_DIMS = (2, 20, 50)
SHORT_SEEDS = (0, 1, 2)
MAIN = ("cfm", "mfm_land", "curly", "ffm", "pathb")
LABELS = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly": "Curly-FM", "ffm": "FFM (ours)",
          "pathb": "FFM (ours), with noise"}
WINDOW_ORDER = ("ffm", "pathb", "cfm", "mfm_land", "curly")
WINDOW_LABELS = {**LABELS, "ffm": "FFM, deterministic", "pathb": "FFM, with noise"}
WINDOW_TIMES = {"0.3333": "W2(1/3)", "0.4333": "W2(0.433)", "0.5667": "W2(0.567)",
                "0.6667": "W2(2/3)"}
PATH_COV = {"M": "C_ρ(μ_t)", "I": "I"}
INFERENCE_COV = {"M": "C_ρ(X_t)", "I": "I"}
FATE_ROWS = (("Constant", "ref_mean", None), ("[SF]2M", "sf2m", "sf2m"), ("FFM", "ffm_det", None),
             ("FFM + velocity, with noise", "ffm_CC", "pathb"))
MATCHED = {"graph_drift": "Graph-drift FM", "aniso_fixed": "Anisotropic fixed-time FM",
           "ffm_second": "Second-moment FFM (m = 0)", "ffm_riem": "FFM, one-form deleted",
           "ffm_full": "FFM (ours)"}
COUPLINGS = {"common": "Shared", "specific": "Specific"}
SOLVER_KEY = "pancreas20"
SOLVERS = {"ffm": "FFM (ours)", "wlf_finsler": "Finsler-WLF", "gsbm_finsler": "Finsler-GSBM",
           "wlf_euclid": "Euclidean WLF", "gsbm_euclid": "Euclidean GSBM"}
LADDERS = {"rho_mult": tune.RHO_MULTS, "lam_mult": tune.LAM_MULTS}
SYMBOLS = {"rho_mult": "ρ", "lam_mult": "λ"}


def _leakage_job(cfg, d, s):
    judge = leakage.JUDGE_SEED
    if s != judge and not os.path.exists(train.state_path(leakage.tree_root(TREE), d, judge)):
        leakage.run(TREE, d, cfg, seed=judge)
    leakage.run(TREE, d, cfg, seed=s)


JOBS = {
    "leakage": _leakage_job,
    "trajectory": lambda cfg, arm: leakage.run(TREE, 2, cfg, force=True, arms=(arm,),
                                               seed=leakage.JUDGE_SEED),
    "band": lambda cfg, d, s: band.ensure(cfg, leakage.tuned_points(), (d,), seeds=(s,)),
    "noise": lambda cfg: noise.ensure(cfg, noise.tuned_point()),
    "fate": lambda cfg, d, s: fate.ensure(cfg, (d,), (s,)),
    "fate_ref": lambda cfg, d: fate.write_refs(fate.problem(d, cfg), fate.root_dir(cfg.smoke),
                                               fate.budget_of(cfg)),
    "bench": lambda cfg, key, arm, s: bench.run_one(key, arm, s, cfg, bench.load_tuned(),
                                                    root=bench.bench_root(cfg.smoke)),
    "hamiltonian": lambda cfg, key, m, s: hamiltonian.run(key, m, s, cfg,
                                                          hamiltonian.root_dir(cfg.smoke)),
    "sweep": lambda cfg, d, i: tune.run_cell(d, 1, i, cfg),
}


def _read(path):
    with open(path) as fh:
        return json.load(fh)


def _key(d):
    return f"pancreas{d}"


def _leak_path(d, arm, s):
    return train.rec_path(leakage.tree_root(TREE), d, arm, s)


def _fate_seeds(arm):
    return (-1,) if arm in fate.REFS else fate.SEEDS


FATE_WIDTH = {20: 0.3, 50: 0.03}


def _fate_width(d, noisy):
    """Noise width of the rows with noise in Table 2; 0 for the deterministic rows."""
    return FATE_WIDTH[d] if noisy else 0.0


def _fate_path(d, s, arm, tuned):
    return fate.record_path(fate.root_dir(), d, s, arm, _fate_width(d, tuned))


def _matched_path(d, arm, s):
    return bench.rec_path(bench.bench_root(), _key(d), arm, s)


def _solver_path(method, s):
    return hamiltonian.rec_path(hamiltonian.root_dir(), SOLVER_KEY, method, s)


def _sweep_path(d, point):
    return tune.cell_path(tune.ROOT, d, 1, "ffm", point)


def _sweep_index(point):
    return next(i for i, arm, p in tune.WAVE1.cells() if arm == "ffm" and p == point)


def _sensitivity(d):
    pt = noise.tuned_point(d)["ffm"]
    return [(knob, [{**pt, knob: v} for v in window(ladder, pt[knob])])
            for knob, ladder in LADDERS.items()]


def _catalogue():
    leak = leakage.tree_root(TREE)
    out = {_leak_path(d, a, s): ("leakage", d, s)
           for d in leakage.DIMS for s in leakage.SEEDS for a in MAIN}
    out.update({train.traj_path(leak, 2, a, leakage.JUDGE_SEED): ("trajectory", a) for a in MAIN})
    out.update({band.cell_path(band.root_dir(), *c): ("band", c[0], c[3])
                for c in band.cells(PAPER_DIMS, SHORT_SEEDS)})
    out.update({noise.cell_path(noise.root_dir(), *c): ("noise",) for c in noise.cells(SHORT_SEEDS)})
    out.update({_fate_path(d, s, arm, tuned): ("fate", d, s) if s >= 0 else ("fate_ref", d)
                for d in fate.DIMS for _, arm, tuned in FATE_ROWS for s in _fate_seeds(arm)})
    out.update({_matched_path(d, arm, s): ("bench", _key(d), arm, s)
                for d in PAPER_DIMS for arm in MATCHED for s in bench.seeds_for(_key(d))})
    out.update({_solver_path(m, s): ("hamiltonian", SOLVER_KEY, m, s)
                for m in SOLVERS for s in hamiltonian.SEEDS})
    out.update({_sweep_path(d, p): ("sweep", d, _sweep_index(p))
                for d in PAPER_DIMS for _, points in _sensitivity(d) for p in points})
    return out


def expected():
    """Every record the notebook reads."""
    return list(_catalogue())


def ensure(cfg=None):
    """Re-create missing records with the drivers that wrote them; nothing runs when all exist.

    Some drivers rebuild more than the missing record. The FFM states are not shipped, so a
    missing leakage record or d = 2 trajectory retrains the FFM interpolant of its seed and of
    seed 0 and rewrites their FFM records. A missing fate record reruns the whole width sweep of
    its seed, which rewrites that seed's shipped records with noise.
    """
    def build(missing):
        from ffm.method import Run
        cfg_ = Run.from_env() if cfg is None else cfg
        jobs, done = _catalogue(), set()
        for path in missing:
            if jobs[path] not in done and not os.path.exists(path):
                JOBS[jobs[path][0]](cfg_, *jobs[path][1:])
                done.add(jobs[path])
    return rec.ensure(expected(), build)


def _leak(d, arm):
    return [_read(_leak_path(d, arm, s)) for s in leakage.SEEDS]


def _w2(d, arm, where):
    return [r["metrics"][where]["W2"] for r in _leak(d, arm)]


def _matched(d, arm, mode):
    rs = [_read(_matched_path(d, arm, s)) for s in bench.seeds_for(_key(d))]
    return rec.pm([r[mode]["W2_mid"] for r in rs], 2)


def _cmse(d, arm, tuned):
    return rec.pm([_read(_fate_path(d, s, arm, tuned))["groups"]["late"]["cse"] for s in _fate_seeds(arm)])


def _noise(cov, width, variant):
    v = [_read(noise.cell_path(noise.root_dir(), s, cov, width))["variants"][variant]
         for s in SHORT_SEEDS]
    return {"W2(1/2)": rec.pm([x["intermediate"]["W2"] for x in v], 4),
            "W2(1)": rec.pm([x["endpoint"]["W2"] for x in v], 4)}


def _band(d, frac):
    return {(a, s): _read(band.cell_path(band.root_dir(), d, frac, a, s)) for a in MAIN for s in SHORT_SEEDS}


def _num(v):
    return f"{v:.1f}" if v >= 1 else f"{v:g}"


def mid_end():
    """Midpoint and endpoint W2 at d = 2, 20 and 50, with the sampling floors."""
    rows = []
    for d in PAPER_DIMS:
        rows += [{"d": d, "Method": LABELS[a], "mid W2": rec.pm(_w2(d, a, "intermediate")),
                  "end W2": rec.pm(_w2(d, a, "endpoint"))} for a in MAIN]
        fl = _leak(d, "ffm")[0]["metrics"]["floors"]
        rows.append({"d": d, "Method": "sampling floor", "mid W2": f"{fl['intermediate']['W2']:.3f}",
                     "end W2": f"{fl['endpoint']['W2']:.3f}"})
    return rec.frame(rows, index=["d", "Method"])


def table1():
    """Table 1, Pancreas column: midpoint W2 at d = 20 and 50."""
    rows = [{"d": d, "Method": LABELS[a], "Pancreas": rec.pm(_w2(d, a, "intermediate"))}
            for d in (20, 50) for a in MAIN]
    return rec.frame(rows, index=["d", "Method"])


def table1_bottom():
    """Table 1, bottom: matched-information objective ablation at d = 20."""
    labels = {**MATCHED, "graph_drift": "Graph-drift FM (control)"}
    rows = [{"Method": labels[arm], "Shared coupling": _matched(20, arm, "common"),
             "Method-specific coupling": _matched(20, arm, "specific")} for arm in MATCHED]
    return rec.frame(rows, index="Method")


def table2():
    """Table 2, Pancreas: fate cMSE against the CellRank reference on the later half of the
    source cells, with the noisy rows at their validation-tuned noise widths."""
    rows = [{"Method": label, **{f"d = {d}": _cmse(d, arm, tuned) for d in fate.DIMS}}
            for label, arm, tuned in FATE_ROWS]
    return rec.frame(rows, index="Method")


def table4():
    """Table 4: training seconds per seed, mean over five seeds."""
    rows = [{"Method": train.ARM_LABELS[a],
             **{f"d = {d}": f"{rec.mean(r['train_seconds'] for r in _leak(d, a)):.0f}" for d in leakage.DIMS}}
            for a in MAIN]
    return rec.frame(rows, index="Method")


def figure4a():
    """Figure 4(a): W2 at model times in [1/3, 2/3] and its mean over the 11-point grid."""
    rows = []
    for d in PAPER_DIMS:
        for a in WINDOW_ORDER:
            win = [r["metrics"]["window"] for r in _leak(d, a)]
            rows.append({"d": d, "Method": WINDOW_LABELS[a],
                         **{label: rec.pm([w[t] for w in win], 4) for t, label in WINDOW_TIMES.items()},
                         "W2[1/3,2/3]": rec.pm([rec.mean(w.values()) for w in win], 4)})
    return rec.frame(rows, index=["d", "Method"])


def figure4b():
    """Figure 4(b): conditional-path noise width at d = 20."""
    rows = [{"Width": f"{w:.3f}", **{(f"M_t = {PATH_COV[c]}", k): v for c in noise.PATH_COVS
                                     for k, v in _noise(c, w, noise.variant_key("const", c)).items()}}
            for w in noise.WIDTH_MULTS]
    df = rec.frame(rows, index="Width")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def figure4c():
    """Figure 4(c): path and inference covariance at d = 20 and the tuned noise width."""
    w = noise.tuned_point()["pathb"]["width_mult"]
    rows = [{"Path covariance": PATH_COV[c], "Inference covariance": INFERENCE_COV[g],
             **_noise(c, w, noise.variant_key("const", g))}
            for c in noise.PATH_COVS for g in noise.DIFFUSIONS]
    return rec.frame(rows, index=["Path covariance", "Inference covariance"])


def table5():
    """Table 5: one-at-a-time sensitivity of FFM; the selected value carries an asterisk."""
    rows = []
    for d in PAPER_DIMS:
        pt = noise.tuned_point(d)["ffm"]
        for knob, points in _sensitivity(d):
            other = next(k for k in LADDERS if k != knob)
            star = ["*" if p == pt else "" for p in points]
            obj = [_read(_sweep_path(d, p))[tune.SELECTION_KEY] for p in points]
            rows.append({"d": d, "Varied parameter": SYMBOLS[knob],
                         "Fixed parameter": f"{SYMBOLS[other]} = {_num(pt[other])}",
                         "Values": ", ".join(_num(p[knob]) + s for p, s in zip(points, star)),
                         "Validation objective": ", ".join(f"{o:.3f}{s}" for o, s in zip(obj, star))})
    return rec.frame(rows, index=["d", "Varied parameter"])


def table6():
    """Table 6: midpoint W2 with 0, 5 or 35 % of the intermediate population available."""
    mid, rows = f"{band.T_MID:g}", []
    for d in PAPER_DIMS:
        cells = {f"{100 * f:g}%": _band(d, f) for f in band.FRACS}
        rows += [{"d": d, "Method": LABELS[a],
                  **{k: rec.pm([c[(a, s)]["curve"][mid]["W2"] for s in SHORT_SEEDS], 2) for k, c in cells.items()}}
                 for a in MAIN]
        rows.append({"d": d, "Method": "Sampling floor",
                     **{k: f"{rec.mean(r['floors']['W2'] for r in c.values()):.2f}" for k, c in cells.items()}})
    return rec.frame(rows, index=["d", "Method"])


def table8():
    """Table 8: matched-information objective ablation at d = 2, 20 and 50."""
    rows = [{"Objective": label, **{(f"d = {d}", head): _matched(d, arm, mode)
                                    for d in PAPER_DIMS for mode, head in COUPLINGS.items()}}
            for arm, label in MATCHED.items()]
    df = rec.frame(rows, index="Objective")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def table10():
    """Table 10, Pancreas d = 20: FFM against WLF and GSBM with Euclidean and Finsler geometry."""
    rows = []
    for m, label in SOLVERS.items():
        rs = [_read(_solver_path(m, s)) for s in hamiltonian.SEEDS]
        rows.append({"Method": label, "Mid W2": rec.pm([r["W2_mid"] for r in rs], 2),
                     "End W2": rec.pm([r["W2_end"] for r in rs], 2),
                     "Time (s)": f"{rec.mean(r['seconds'] for r in rs):.0f}"})
    return rec.frame(rows, index="Method")


def figure_paths():
    """Learned trajectories at d = 2 (seed 0) over the shown and withheld cells."""
    from ffm.method import Run

    from . import figures
    with contextlib.redirect_stdout(io.StringIO()):
        cloud = load_cloud(2)
    stem = rec.figure_path("pancreas_paths")
    figures.figure_paths(leakage.tree_root(TREE), 2, cloud, os.path.dirname(stem), Run(),
                         seed=leakage.JUDGE_SEED, stem=os.path.basename(stem))
