"""The one rule that ranks every hyper-parameter in this repository.

There used to be two selection protocols carried in parallel through every tree, every
filename and every table -- a leak-free proxy on the kept cells, and an endpoint/midpoint
average that read a validation slice of the withheld marginal.  Carrying both meant every
benchmark had two tuned points, two output trees and two sets of prose arguing about which
one the reader should believe.  There is now one, and it lives here so that a tuned point
means the same thing in the Sheet, the MFM Arch, the erythroid lineage and the Pancreas:

    objective = mean( W2(prediction at t = 1,   val cells of the target marginal),
                      W2(prediction at t = 1/2, val slice of the withheld marginal) )

Lower is better.  The two terms are stored separately by every sweep cell, so a finished
search can be re-ranked -- or a third term added -- without retraining anything, and that
is the only reason ``objective_from`` takes a dict rather than two floats.

**What this protocol is, and what it is not.**  Both val sets are disjoint from the test
cells every reported number is measured on, so this is an ordinary tuned-on-val /
reported-on-test split and not a leak into the score.  What it is *not* is blind: the
selected point has been shown where the route goes.  Each benchmark's prose says so.  The
guarantee the whole arrangement now rests on is therefore a single one, asserted by every
tree's ``verify`` suite -- **val and test never intersect, and no withheld cell reaches a
training set** (``P``, its bandwidth and the geometry are all rebuilt on the survivors).

Each experiment supplies its own two marginals; only the arithmetic is shared.
"""
from __future__ import annotations

import numpy as np

#: the two val terms the objective averages, in the order a table prints them.  Adding a
#: third here re-ranks every stored sweep in the repository and retrains nothing, which is
#: the property the split-terms-from-scalar design exists to buy.
OBJECTIVE_TERMS: tuple[str, ...] = ("W2_endpoint", "W2_intermediate")

#: the key the averaged scalar is stored under, so a record and a ranker cannot disagree
#: about the spelling.
OBJECTIVE_KEY = "objective"


def objective_from(terms: dict[str, float]) -> float:
    """Rank a recorded sweep point from its stored terms; no retraining needed.

    Raises rather than silently ranking a partial record: a cell written before a term
    existed would otherwise sort to the front of the grid on an incomplete average, and
    a tuned point chosen that way is not tuned for the published rule.
    """
    missing = [k for k in OBJECTIVE_TERMS if k not in terms]
    assert not missing, (
        f"sweep record is missing {missing} and cannot be ranked. Re-run the sweep "
        "with --force.")
    return float(np.mean([float(terms[k]) for k in OBJECTIVE_TERMS]))


def with_objective(terms: dict[str, float]) -> dict[str, float]:
    """``terms`` plus its averaged scalar, which is the shape every sweep record takes."""
    return {**terms, OBJECTIVE_KEY: objective_from(terms)}
