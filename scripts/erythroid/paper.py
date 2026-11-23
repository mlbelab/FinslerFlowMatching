import json
import os

import pandas as pd

from scripts import records as rec

from .datasets import load_cloud
from .train import PUBLISHED, PUBLISHED_OF

BENCH = "erythroid_bench"
DEFAULTS = "erythroid_defaults"
STAGE = "erythroid_stage_bench"
SPLIT = "long"
HELD = "E8.0+E8.25"
SEEDS = (0, 1, 2, 3, 4)
PSEUDO_DIMS = (2, 20, 50)
REAL_DIMS = (20, 50)
FIG_DIM, FIG_SEED = 2, 0
DASH = "–"
PSEUDO = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly": "Curly-FM", "ffm": "FFM (ours)",
          "pathb": "FFM (ours), with noise"}
REAL = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly_unit": "Curly-FM", "ffm": "FFM (ours)",
        "ffm_noise": "FFM (ours), with noise", "ffm_expr": "FFM, expression",
        "ffm_expr_noise": "FFM, expression + noise"}
REAL_OF = {"cfm": "cfm", "mfm_land": "mfm_land", "curly": "curly_unit", "ffm": "ffm", "pathb": "ffm_noise"}
TUNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


def _test(d, arm, seed):
    return rec.path(BENCH, "test", f"d{d}_{arm}_s{seed}.json")


def _slab(d, arm, seed):
    return rec.path(BENCH, "trajectories", f"d{d}_{arm}_s{seed}.npz")


def _end(d, arm, seed):
    return rec.path(DEFAULTS, "tuned", f"d{d}_{arm}_s{seed}.json")


def _grid(d, method, seed):
    return rec.path(STAGE, "grid", f"{SPLIT}_d{d}_{method}_s{seed}.json")


def _stage(d, method, seed):
    return rec.path(STAGE, "records", f"{SPLIT}_d{d}_{method}_s{seed}.json")


def _refs():
    return rec.path(STAGE, "references.json")


def expected():
    pseudo = [p(d, a, s) for p in (_test, _end) for d in PSEUDO_DIMS for a in PSEUDO for s in SEEDS]
    real = [p(d, m, s) for p in (_grid, _stage) for d in REAL_DIMS for m in REAL for s in SEEDS]
    return pseudo + real + [_refs()] + [_slab(FIG_DIM, a, FIG_SEED) for a in PSEUDO]


def _read(path):
    with open(path) as fh:
        return json.load(fh)


def ensure(cfg=None):
    def build(missing):
        from ffm.core.paths import experiment_root
        from ffm.method import Run
        from scripts.larry import bench as larry

        from . import defaults, stage_bench, stage_grid, train
        cfg_ = Run.from_env() if cfg is None else cfg
        gone = set(missing)
        picks = {(int(d), a): p for d, v in _read(TUNED_FILE).items() for a, p in v.items()}
        for d in PSEUDO_DIMS:
            for arm in PSEUDO:
                for s in SEEDS:
                    redo = _slab(d, arm, s) in gone or (_end(d, arm, s) in gone
                                                        and not os.path.exists(_slab(d, arm, s)))
                    if redo or _test(d, arm, s) in gone:
                        train.stage_test(experiment_root(BENCH, cfg_.smoke), picks, (d,), (arm,), (s,),
                                         cfg_.device, cfg_.dtype, train.N_STEPS_ERYTHROID, cfg_.smoke, redo)
            if any(_end(d, arm, s) in gone for arm in PSEUDO for s in SEEDS):
                defaults.backfill(d, defaults.root_dir(cfg_.smoke), defaults.seeds_of(cfg_.smoke))
        rcfg = cfg_.but(ot_max_pts=larry.OT_MAX_PTS)
        root = stage_bench.root_dir(rcfg.smoke)
        for d, unit, s in stage_grid.tasks():
            methods = larry.unit_methods(unit)
            if any(_grid(d, m, s) in gone for m in methods):
                stage_grid.run_unit(d, unit, s, rcfg, root)
            if any(_stage(d, m, s) in gone for m in methods):
                stage_bench.run_unit(SPLIT, d, unit, s, rcfg, root)
        if _refs() in gone:
            refs = {SPLIT: {d: stage_bench.references(stage_bench.cloud(SPLIT, d)) for d in stage_bench.DIMS}}
            os.makedirs(os.path.dirname(_refs()), exist_ok=True)
            with open(_refs(), "w") as fh:
                json.dump(refs, fh, indent=1)
    return rec.ensure(expected(), build)


def _cell(path_of, d, arm, *keys):
    values = []
    for s in SEEDS:
        value = _read(path_of(d, arm, s))
        for k in keys:
            value = value[k]
        values.append(value)
    return rec.pm(values)


def mid_end():
    rows = [{"": f"Pseudotime, d = {d}", "Method": label,
             "mid W2": _cell(_test, d, arm, "metrics", "w2_true"), "end W2": _cell(_end, d, arm, "end_w2")}
            for d in PSEUDO_DIMS for arm, label in PSEUDO.items()]
    floors = _read(_refs())[SPLIT]
    for d in REAL_DIMS:
        rows += [{"": f"Real-time, d = {d}", "Method": label, "mid W2": _cell(_grid, d, m, "W2_mid"),
                  "end W2": _cell(_stage, d, m, "W2_end")} for m, label in REAL.items()]
        rows.append({"": f"Real-time, d = {d}", "Method": "sampling floor (800 points)",
                     "mid W2": f"{floors[str(d)][HELD]['floor']['mean']:.3f}", "end W2": DASH})
    return rec.frame(rows, index=["", "Method"])


def table1():
    rows = [{"": f"d = {d}", "Method": label,
             ("Pseudotime", "Erythroid"): _cell(_test, d, arm, "metrics", "w2_true"),
             ("Real-time", "Erythroid"): _cell(_grid, d, REAL_OF[arm], "W2_mid")}
            for d in REAL_DIMS for arm, label in PSEUDO.items()]
    df = rec.frame(rows, index=["", "Method"])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def table7_pseudotime():
    rows = []
    for d in PSEUDO_DIMS:
        for arm, label in PSEUDO.items():
            pub = PUBLISHED.get((d, PUBLISHED_OF.get(arm)))
            rows.append({"": f"Pseudotime, d = {d}", "Method": label,
                         "Published W1": f"{pub['w2'][0]:.3f} ± {pub['w2'][1]:.3f}" if pub else DASH,
                         "Reproduced W1": _cell(_test, d, arm, "metrics", "w2"),
                         "Reproduced W2": _cell(_test, d, arm, "metrics", "w2_true")})
    return rec.frame(rows, index=["", "Method"])


def table7_realtime():
    rows = []
    for m, label in REAL.items():
        row = {"Method": label}
        for d in REAL_DIMS:
            row[(f"d = {d}", "W2 (1/2)")] = _cell(_grid, d, m, "W2_mid")
            row[(f"d = {d}", "W2 [1/3, 2/3]")] = _cell(_grid, d, m, "W2_mean")
        rows.append(row)
    df = rec.frame(rows, index="Method")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _show(fig, name):
    import matplotlib.pyplot as plt
    stem = rec.figure_path(name)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.show()


def figure_dataset():
    from .figures import data_figure
    _show(data_figure(load_cloud(FIG_DIM)), "erythroid_dataset")


def figure_paths():
    from .figures import paths_figure
    _show(paths_figure(rec.path(BENCH), tuple(PSEUDO), seed=FIG_SEED, dim=FIG_DIM), "erythroid_paths")
