"""The matched-information ablations on Pancreas d = 20 only, as two frames.

    python -m scripts.experiments.matched.pancreas report
    python -m scripts.experiments.matched.pancreas run

``report`` reads what is on disk and prints; ``run`` trains whatever is missing first.
Both go through :func:`scripts.experiments.matched.bench.run_one`, so this module is a
pancreas-shaped view of the same records the two-dataset bench writes and never a second
implementation of the comparison.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.method import Run

from .bench import ROOT, SEEDS, PRIMARY, load_records, load_tuned, rec_path, run_one
from .tune import smoke_cfg
from .variants import ARM_LABELS, ARMS, COUPLINGS

KEY = "pancreas20"


def ensure(cfg: Run, tuned: dict | None = None, root: str = ROOT, seeds=SEEDS,
           force: bool = False) -> dict:
    """Every arm at every seed for Pancreas, training only what is not already recorded."""
    tuned = load_tuned() if tuned is None else tuned
    for arm in ARMS:
        for seed in seeds:
            run_one(KEY, arm, seed, cfg, tuned, root=root, force=force)
    return load_records(root, keys=(KEY,))


def missing(root: str = ROOT, seeds=SEEDS) -> list[tuple[str, int]]:
    import os
    return [(a, s) for a in ARMS for s in seeds
            if not os.path.exists(rec_path(root, KEY, a, s))]


def _col(recs, arm, mode, field, seeds):
    return np.array([recs[(KEY, arm, s)][mode][field]
                     for s in seeds if (KEY, arm, s) in recs], dtype=float)


def _seeds_present(recs, seeds=SEEDS):
    return [s for s in seeds if any((KEY, a, s) in recs for a in ARMS)]


def summary(recs: dict, seeds=SEEDS) -> pd.DataFrame:
    """One row per arm per coupling: mean +- sd over seeds of both metrics and runtime."""
    seeds = _seeds_present(recs, seeds)
    rows = []
    for mode in COUPLINGS:
        for arm in ARMS:
            if not any((KEY, arm, s) in recs for s in seeds):
                continue
            mid, end = _col(recs, arm, mode, "W2_mid", seeds), _col(recs, arm, mode, "W2_end", seeds)
            sec = _col(recs, arm, mode, "total_s", seeds)
            rows.append({"coupling": mode, "arm": ARM_LABELS[arm],
                         "mid W2": mid.mean(), "mid sd": mid.std(ddof=1),
                         "end W2": end.mean(), "end sd": end.std(ddof=1),
                         "runtime s": sec.mean(), "n seeds": len(mid)})
    return pd.DataFrame(rows).set_index(["coupling", "arm"])


DIFF_FIELDS = {"d mid W2": "W2_mid", "d end W2": "W2_end", "d runtime s": "total_s"}


def _diffs(recs, seeds, pair):
    a, b = pair
    seeds = [s for s in _seeds_present(recs, seeds)
             if (KEY, a, s) in recs and (KEY, b, s) in recs]
    return seeds, {m: {c: _col(recs, a, m, f, seeds) - _col(recs, b, m, f, seeds)
                       for c, f in DIFF_FIELDS.items()} for m in COUPLINGS}


def paired(recs: dict, seeds=SEEDS, pair=PRIMARY) -> pd.DataFrame:
    """One row per seed: the primary pair's difference, which a mean alone would hide."""
    seeds, d = _diffs(recs, seeds, pair)
    return pd.DataFrame([{"coupling": m, "seed": s,
                          **{c: d[m][c][i] for c in DIFF_FIELDS}}
                         for m in COUPLINGS for i, s in enumerate(seeds)]
                        ).set_index(["coupling", "seed"])


def paired_summary(recs: dict, seeds=SEEDS, pair=PRIMARY) -> pd.DataFrame:
    """The mean of those differences and how many of the seeds carry each sign."""
    seeds, d = _diffs(recs, seeds, pair)
    n = len(seeds)
    return pd.DataFrame([{"coupling": m, "stat": stat,
                          **{c: (f"{d[m][c].mean():.4f}" if stat == "mean"
                                 else f"{int((d[m][c] < 0).sum())}/{n}")
                             for c in DIFF_FIELDS}}
                         for m in COUPLINGS for stat in ("mean", "wins")]
                        ).set_index(["coupling", "stat"])


def header(recs: dict) -> str:
    any_rec = next(r for (k, _, _), r in recs.items() if k == KEY)
    d, fl, sc = any_rec["diag"], any_rec["floors"], any_rec["scales"]
    a, b = PRIMARY
    return (f"Pancreas, d = {d['d']}, n_train = {d['n_train']}, mid ref {d['n_mid_ref']} "
            f"cells, end ref {d['n_end_ref']}\n"
            "sampling floors (same reference, split in half):  "
            + "   ".join(f"{k} {v:.4f}" for k, v in fl.items()) + "\n"
            "measured scales:  " + "   ".join(f"{k} {v:.4g}" for k, v in sc.items()) + "\n"
            f"paired differences are {ARM_LABELS[a]} minus {ARM_LABELS[b]}; "
            "negative favours ours, no threshold applied")


def frames(recs: dict, seeds=SEEDS):
    return summary(recs, seeds), paired(recs, seeds), paired_summary(recs, seeds)


def cmd_report(args) -> None:
    recs = load_records(args.root, keys=(KEY,))
    if not recs:
        raise SystemExit(f"no Pancreas bench records under {args.root}; run `run` first")
    gone = missing(args.root)
    if gone:
        print(f"warning: {len(gone)} of {len(ARMS) * len(SEEDS)} records missing: {gone}")
    s, p, ps = frames(recs)
    print(header(recs), "\n")
    print(s.to_string(float_format="%.4f"), "\n")
    print(p.to_string(float_format="%.4f"), "\n")
    print(ps.to_string())


def cmd_run(args) -> None:
    cfg = smoke_cfg(args.smoke, args.cpu)
    ensure(cfg, root=args.root, force=args.force)
    cmd_report(args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("report")
    c.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
