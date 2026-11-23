import pandas as pd

from scripts import records as rec

NAME = "eb_bench"
DIMS = (20, 50)
HELD = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
LABELS = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly_unit": "Curly-FM", "ffm_expr": "FFM (ours)",
          "ffm_expr_noise": "FFM (ours), with noise"}
UNITS = {"cfm": "cfm", "mfm_land": "mfm_land", "curly_unit": "curly_unit", "ffm_expr": "ffm_expression",
         "ffm_expr_noise": "ffm_expression"}
REFERENCES = "references.json"
DASH = "–"


def record_path(d, k, method, seed):
    return rec.path(NAME, "records", f"d{d}_k{k}_{method}_s{seed}.json")


def _runs():
    return [(d, k, m, s) for d in DIMS for k in HELD for m in LABELS for s in SEEDS]


def expected():
    return [record_path(*r) for r in _runs()] + [rec.path(NAME, REFERENCES)]


def ensure(cfg=None):
    def build(missing):
        from ffm.method import Run

        from . import bench
        cfg_ = (Run.from_env() if cfg is None else cfg).but(ot_max_pts=bench.larry.OT_MAX_PTS)
        if rec.path(NAME, REFERENCES) in missing:
            bench.main(["prep"])
        units = dict.fromkeys((d, k, UNITS[m], s) for d, k, m, s in _runs() if record_path(d, k, m, s) in missing)
        for d, k, unit, seed in units:
            bench.run_unit(d, k, unit, seed, cfg_, bench.root_dir(cfg_.smoke))
    return rec.ensure(expected(), build)


def _values(d, method, key):
    return [rec.load_one(NAME, "records", f"d{d}_k{k}_{method}_s{s}.json")[key] for k in HELD for s in SEEDS]


def _frame(rows):
    df = rec.frame(rows, index=["d", "Method"])
    df.index.names = [None, "Method"]
    return df


def mid_end():
    refs = rec.load_one(NAME, REFERENCES)
    rows = []
    for d in DIMS:
        rows += [{"d": f"d = {d}", "Method": label, "mid W2": rec.pm(_values(d, m, "W2")),
                  "end W2": rec.pm(_values(d, m, "W2_end"))} for m, label in LABELS.items()]
        floor = rec.mean(refs[str(d)][str(k)]["floor"]["W2"] for k in HELD)
        rows.append({"d": f"d = {d}", "Method": "sampling floor", "mid W2": f"{floor:.3f}", "end W2": DASH})
    return _frame(rows)


def table1():
    df = _frame([{"d": f"d = {d}", "Method": label, "Embryoid Body": rec.pm(_values(d, m, "W2"))}
                 for d in DIMS for m, label in LABELS.items()])
    df.columns = pd.MultiIndex.from_tuples([("LOTO", "Embryoid Body")])
    return df
