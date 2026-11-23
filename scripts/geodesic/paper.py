import pandas as pd

from scripts import records as rec

from . import validate as val

BUDGET = "2500"
TABLE_DIMS = (2, 20, 100)
LABELS = {"randers": "Constant Randers", "sphere": "Sphere", "wind": "Sphere + wind"}


def _tag(case, d, seed):
    return val.tag(case, d, seed)


def expected():
    return [rec.path(val.NAME, "records", _tag(c, d, s) + ".json")
            for c in val.CASES for d in TABLE_DIMS for s in val.SEEDS]


def ensure(cfg=None):
    def build(missing):
        from ffm.method import Run
        cfg_ = Run.from_env() if cfg is None else cfg
        index = {rec.path(val.NAME, "records", val.tag(c, d, s, k, e) + ".json"): i
                 for i, (c, d, s, k, e) in enumerate(val.tasks())}
        for p in missing:
            val.run_task(index[p], cfg_)
    rec.ensure(expected(), build)


def _mean(case, d, key):
    rs = [rec.load_one(val.NAME, "records", _tag(case, d, s) + ".json")["eval"][BUDGET][key] for s in val.SEEDS]
    return 100 * rec.mean(rs)


def table3():
    rows = []
    for case in val.CASES:
        row = {"Metric": LABELS[case]}
        for d in TABLE_DIMS:
            row[("Cost error (%)", f"d = {d}")] = f"{_mean(case, d, 'rel_K8'):.2f}"
        for d in TABLE_DIMS:
            row[("Path deviation (%)", f"d = {d}")] = f"{_mean(case, d, 'path_dev'):#.2g}"
        rows.append(row)
    df = rec.frame(rows, index="Metric")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df
