"""The reported run: five paired seeds, two couplings, and the per-seed differences.

    python -m scripts.experiments.matched.bench plan
    python -m scripts.experiments.matched.bench run --index 7
    python -m scripts.experiments.matched.bench report

What "paired" means here, concretely
------------------------------------
At a fixed ``(dataset, seed)`` every arm gets the identical split, the identical source
and target tensors, the identical network initialisation (Phase 1 seeds
``torch.manual_seed(seed)``, Phase 3 ``seed + 1``), the identical minibatch sequence, the
identical architecture and the identical iteration budget.  Under the **common** coupling
they additionally get a bit-identical Sinkhorn plan, because
:func:`~scripts.experiments.matched.variants.build_ot_euclidean` draws its subsample from
``seed + 91`` and its cost does not read the arm.  The only thing that differs is the
cost density in Phase 1 and -- under the specific coupling -- the cost matrix in Phase 2.
So a per-seed difference is attributable to the objective and to nothing else, and
reporting the five differences individually rather than a bolded mean is what makes that
checkable.

What is reported, and against what
----------------------------------
Midpoint :math:`W_2`, endpoint :math:`W_2` and total runtime, per arm, per coupling, per
seed, on each tree's own **test** references -- disjoint from the validation cells the
search ranked on.  The sampling floor of each reference (that reference split in half and
scored against itself) is printed beside the table: a paired difference below the floor is
not a difference, and this tree states that rather than bolding it.

No threshold is applied anywhere.  ``report`` prints the five paired differences, their
mean, and how many of the five have each sign; it does not declare a winner and it does
not convert anything to a percentage improvement.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from scripts.core.paths import experiment_root
from scripts.method import Run

from . import problem as prob
from .tune import TUNED_FILE, smoke_cfg, warm_up
from .variants import ARM_LABELS, ARMS, COUPLINGS, MatchedFactory, matched_path_a

#: five paired model seeds, as the review asks.  The split is held fixed across all of
#: them; only the model seed moves.
SEEDS = (0, 1, 2, 3, 4)

#: the comparison the whole tree is built around.  Named so ``report`` cannot quietly
#: start differencing a different pair.
PRIMARY = ("ffm_full", "aniso_fixed")

ROOT = experiment_root("matched_bench", bool(int(os.environ.get("NB_SMOKE", "0"))))
TEST_DIR = "test"


def bench_root(smoke: bool = False) -> str:
    return experiment_root("matched_bench", smoke)


def rec_path(root: str, key: str, arm: str, seed: int) -> str:
    return os.path.join(root, TEST_DIR, f"{key}_{arm}_s{seed}.json")


def tasks() -> list[tuple[str, str, int]]:
    """``(dataset, arm, seed)`` in a fixed order -- the array index is this list."""
    return [(k, a, s) for k in prob.KEYS for a in ARMS for s in SEEDS]


def load_tuned(path: str = TUNED_FILE) -> dict:
    assert os.path.exists(path), (
        f"no tuned point at {path}; run `python -m scripts.experiments.matched.tune "
        f"collect` first")
    with open(path) as fh:
        return json.load(fh)


def _fresh(path: str, points: dict) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        prior = json.load(fh)
    return prior.get("points") == points


def run_one(key: str, arm: str, seed: int, cfg: Run, tuned: dict, root: str = ROOT,
            force: bool = False) -> dict:
    """One ``(dataset, arm, seed)`` at its selected point, under both couplings.

    The arm is run at the point that won **under each coupling**, which is why ``points``
    is a dict of two: tuning the common column on the specific column's argmin would have
    handed one of the two settings a point chosen for a different pipeline.  When the two
    argmins coincide -- which they often do -- Phase 1 is trained once and both columns
    come out of the same interpolant, exactly as in a sweep cell.
    """
    points = {m: tuned[key][m][arm] for m in COUPLINGS}
    path = rec_path(root, key, arm, seed)
    if _fresh(path, points) and not force:
        print(f"[bench] skip {key}_{arm}_s{seed}")
        with open(path) as fh:
            return json.load(fh)

    pb = prob.build(key, "bench", cfg)
    factory = MatchedFactory(pb.X, pb.D_pts, pb.b0_pts, cfg, pb.X0_t, pb.X1_t)
    warm_up(factory, pb.X0_t, pb.X1_t, cfg)

    rec = {"key": key, "arm": arm, "seed": seed, "points": points,
           "scales": factory.scales(), "diag": pb.diag, "smoke": bool(cfg.smoke)}

    # one Path A per *distinct* point; both couplings share it when the argmins agree.
    by_point: dict[str, dict] = {}
    for mode in COUPLINGS:
        tag = json.dumps(points[mode], sort_keys=True)
        if tag not in by_point:
            t0 = time.time()
            cost = factory.at(arm, points[mode])
            by_point[tag] = matched_path_a(cost, pb.X0_t, pb.X1_t, seed, cfg)
            by_point[tag]["wall_s"] = time.time() - t0
        out = by_point[tag][mode]
        rec[mode] = {**pb.test(out["traj"], seed=seed),
                     **{f"val_{k}": v for k, v in pb.val(out["traj"], seed=seed).items()},
                     "total_s": round(out["total_s"], 1),
                     "t_phase1": round(by_point[tag]["t_phase1"], 1),
                     "t_ot": round(out["t_ot"], 1),
                     "t_phase3": round(out["t_phase3"], 1),
                     "pi_entropy_frac": out["ot"]["pi_entropy_frac"]}

    rec["floors"] = pb.floors(seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    for mode in COUPLINGS:
        m = rec[mode]
        print(f"[bench] {key} {arm:<12} s{seed} {mode:<8}  mid W2 {m['W2_mid']:.4f}  "
              f"end W2 {m['W2_end']:.4f}  ({m['total_s']:.0f}s)")
    return rec


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
def load_records(root: str = ROOT, keys=prob.KEYS) -> dict:
    out = {}
    for key, arm, seed in tasks():
        if key not in keys:
            continue
        p = rec_path(root, key, arm, seed)
        if os.path.exists(p):
            with open(p) as fh:
                out[(key, arm, seed)] = json.load(fh)
    return out


def _col(recs, key, arm, mode, field):
    return np.array([recs[(key, arm, s)][mode][field]
                     for s in SEEDS if (key, arm, s) in recs], dtype=float)


def report(root: str = ROOT, keys=prob.KEYS) -> None:
    """Means with spreads, then the individual paired differences.

    The second half is the part the review asked for and the part a bolded mean hides: for
    the primary pair, one line per seed with both metrics, so a reader can see whether an
    advantage is five small consistent wins or one large one and four ties.
    """
    recs = load_records(root, keys)
    if not recs:
        raise SystemExit(f"no bench records under {os.path.join(root, TEST_DIR)}")

    for key in keys:
        have = [a for a in ARMS if any((key, a, s) in recs for s in SEEDS)]
        if not have:
            continue
        any_rec = next(r for (k, _, _), r in recs.items() if k == key)
        print(f"\n{'=' * 78}\n{key}   d = {any_rec['diag']['d']}   "
              f"n_train = {any_rec['diag']['n_train']}   "
              f"mid ref {any_rec['diag']['n_mid_ref']} cells, "
              f"end ref {any_rec['diag']['n_end_ref']}\n{'=' * 78}")
        fl = any_rec["floors"]
        print("sampling floors (same reference, split in half):  "
              + "   ".join(f"{k} {v:.4f}" for k, v in fl.items()))
        sc = any_rec["scales"]
        print("measured scales:  " + "   ".join(f"{k} {v:.4g}" for k, v in sc.items()))

        for mode in COUPLINGS:
            print(f"\n-- coupling: {mode} --")
            print(f"  {'arm':<40}{'mid W2':>16}{'end W2':>16}{'runtime s':>12}")
            for arm in have:
                mid, end = _col(recs, key, arm, mode, "W2_mid"), _col(recs, key, arm, mode, "W2_end")
                sec = _col(recs, key, arm, mode, "total_s")
                print(f"  {ARM_LABELS[arm]:<40}"
                      f"{mid.mean():>9.4f} ±{mid.std(ddof=1):<6.4f}"
                      f"{end.mean():>9.4f} ±{end.std(ddof=1):<6.4f}"
                      f"{sec.mean():>12.0f}")

        a, b = PRIMARY
        if not all(x in have for x in PRIMARY):
            continue
        print(f"\n-- paired differences, {ARM_LABELS[a]} minus {ARM_LABELS[b]} --")
        print("   negative favours ours; every seed shown, no threshold applied")
        for mode in COUPLINGS:
            print(f"\n   {mode} coupling")
            print(f"   {'seed':>6}{'d mid W2':>14}{'d end W2':>14}"
                  f"{'d runtime s':>14}")
            dmid = _col(recs, key, a, mode, "W2_mid") - _col(recs, key, b, mode, "W2_mid")
            dend = _col(recs, key, a, mode, "W2_end") - _col(recs, key, b, mode, "W2_end")
            dsec = _col(recs, key, a, mode, "total_s") - _col(recs, key, b, mode, "total_s")
            for i, s in enumerate([s for s in SEEDS if (key, a, s) in recs]):
                print(f"   {s:>6}{dmid[i]:>14.4f}{dend[i]:>14.4f}{dsec[i]:>14.0f}")
            print(f"   {'mean':>6}{dmid.mean():>14.4f}{dend.mean():>14.4f}"
                  f"{dsec.mean():>14.0f}")
            print(f"   {'wins':>6}{f'{int((dmid < 0).sum())}/{len(dmid)}':>14}"
                  f"{f'{int((dend < 0).sum())}/{len(dend)}':>14}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def cmd_plan(args) -> None:
    ts = tasks()
    if args.count:
        print(len(ts))
        return
    for i, (k, a, s) in enumerate(ts):
        print(f"{i:>3}  {k:<12} {a:<12} seed {s}")
    print(f"\n{len(ts)} tasks = {len(prob.KEYS)} datasets x {len(ARMS)} arms x "
          f"{len(SEEDS)} seeds.  Each writes both couplings.")


def cmd_run(args) -> None:
    ts = tasks()
    assert 0 <= args.index < len(ts), f"--index out of range 0..{len(ts) - 1}"
    key, arm, seed = ts[args.index]
    cfg = smoke_cfg(args.smoke, args.cpu)
    root = bench_root(cfg.smoke) if args.root is None else args.root
    run_one(key, arm, seed, cfg, load_tuned(args.tuned), root=root, force=args.force)


def cmd_report(args) -> None:
    report(args.root or ROOT)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--count", action="store_true")
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--tuned", default=TUNED_FILE)
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
