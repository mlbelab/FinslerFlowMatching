import glob
import json
import os
import statistics

import pandas as pd

from ffm.core.paths import experiment_root, figure_dir, flag

NO_TRAIN = "FFM_NO_TRAIN"


def path(experiment, *parts):
    return os.path.join(experiment_root(experiment), *parts)


def load(experiment, pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(path(experiment, pattern)))]


def load_one(experiment, *parts):
    with open(path(experiment, *parts)) as fh:
        return json.load(fh)


def pm(values, digits=3):
    values = [float(v) for v in values]
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.fmean(values):.{digits}f} ± {statistics.pstdev(values):.{digits}f}"


def mean(values):
    return statistics.fmean(float(v) for v in values)


def ensure(expected, build):
    missing = [p for p in expected if not os.path.exists(p)]
    if missing and flag(NO_TRAIN):
        raise FileNotFoundError(f"{len(missing)} of {len(expected)} records missing, e.g. {missing[0]}")
    if missing:
        build(missing)


def frame(rows, index=None, columns=None):
    df = pd.DataFrame(rows, columns=columns)
    if index is not None:
        df = df.set_index(index)
    return df


def figure_path(name):
    os.makedirs(figure_dir(), exist_ok=True)
    return os.path.join(figure_dir(), name)
