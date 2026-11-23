import pandas as pd

from scripts import records as rec

NAME = "larry_fate"
DIMS = (20, 50)
SEEDS = (0, 1, 2, 3, 4)
WIDTH = {20: 0.3, 50: 0.0547723}
ROWS = (("Constant", "references", False), ("[SF]2M", "sf2m", True), ("FFM", "ffm_velocity", False),
        ("FFM + velocity, with noise", "ffm_velocity_sde", True),
        ("FFM + velocity + lineage, with noise", "ffm_velocity_lineage_sde", True))


def _file(d, seed, arm, noisy):
    return f"d{d}_s{seed}_{arm}_{f'w{WIDTH[d]:g}' if noisy else 'det'}.json"


def _catalogue():
    return {rec.path(NAME, "records", _file(d, s, arm, noisy)): (d, s, arm.removesuffix("_sde"))
            for _, arm, noisy in ROWS for d in DIMS for s in SEEDS}


def expected():
    """The Table 2 records: five rows at d = 20 and 50, seeds 0-4.

    The rows with noise use ``WIDTH``: 0.3 at d = 20, the width chosen on the Pancreas fate
    cells, and the transferred width (``bench.transfer_points``) at d = 50.
    """
    return list(_catalogue())


def ensure(cfg=None):
    """Re-create missing records with the fate driver; nothing runs when all exist.

    The driver works per unit: a missing model record re-runs its whole unit (every width of
    ``fate.widths``, plus the deterministic model for FFM) and rewrites all of the unit's records.
    """
    def build(missing):
        from ffm.method import Run

        from . import fate
        cfg_ = (Run.from_env() if cfg is None else cfg).but(ot_max_pts=fate.OT_MAX_PTS)
        jobs = _catalogue()
        for d, s, unit in dict.fromkeys(jobs[p] for p in missing):
            if unit == "references":
                fate.references(d, s, cfg_)
            else:
                fate.run_unit(d, s, unit, cfg_)
    return rec.ensure(expected(), build)


def _records(d, arm, noisy):
    return [rec.load_one(NAME, "records", _file(d, s, arm, noisy)) for s in SEEDS]


def _cse(record, arm):
    return (record["constant_any"] if arm == "references" else record["hard"]["any"])["cse"]


def end_w2():
    """Endpoint W2 of the Table 2 fate models against the Day-6 cells."""
    rows = [{"Method": label, **{(f"d = {d}", "end W2"): rec.pm([r["W2_end"] for r in _records(d, arm, noisy)], 2)
                                 for d in DIMS}}
            for label, arm, noisy in ROWS if arm != "references"]
    df = rec.frame(rows, index="Method")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def table2():
    """Table 2, LARRY: fate prediction cMSE against the observed clonal fates."""
    rows = [{"Method": label, **{f"d = {d}": rec.pm([_cse(r, arm) for r in _records(d, arm, noisy)])
                                 for d in DIMS}}
            for label, arm, noisy in ROWS]
    return rec.frame(rows, index="Method")
