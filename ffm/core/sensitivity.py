"""One-at-a-time hyper-parameter sensitivity: one block per knob, five rungs each.

The question is how much of a reported number is the knob rather than the method.  Each
swept knob is walked over five rungs of the ladder it was searched on -- the tuned value
and two rungs either side -- while every other knob is held at its tuned value, and the
reported metric is tabulated against it.

Held is the operative word for an arm with two live axes.  Our FFM turns ``rho_mult`` and
``lam_mult`` together, so a single walk through that plane would confound them; the two
get a block each, the ``rho_mult`` block walking rho at the selected lambda and the
``lam_mult`` block walking lambda at the selected rho.  They are two orthogonal slices
through the same selected point, which is why their tuned rows agree.

The comparison that makes the table readable is against the seed spread of the same
metric: a rung whose deviation from the tuned rung stays inside it is indistinguishable
from re-running one configuration, which is the definition of a knob the result does not
rest on.  That test is a column rather than a sentence, so the table stands alone.
"""
from __future__ import annotations

import numpy as np

#: rungs per knob: the tuned value and two either side.
WINDOW = 5


def window(ladder, value, n: int = WINDOW, atol: float = 1e-9) -> list[float]:
    """The ``n`` rungs of ``ladder`` centred on ``value``, slid inwards at the ends.

    A tuned value at an end of the ladder still gets ``n`` rungs -- the window slides
    rather than truncating, so every block has the same number of rows and a short block
    always means a missing run and never an edge.  ``value`` must *be* a rung: a tuned
    point off the ladder means the ladder passed in is not the one that was searched, and
    snapping to the nearest rung would draw a window around a point nothing was run at.
    """
    rungs = [float(v) for v in ladder]
    i = int(np.argmin([abs(np.log(v) - np.log(value)) for v in rungs]))
    assert np.isclose(rungs[i], value, rtol=1e-6, atol=atol), (
        f"tuned value {value:g} is not a rung of the searched ladder {rungs}")
    lo = min(max(i - n // 2, 0), max(len(rungs) - n, 0))
    return rungs[lo:lo + n]


def table(panels, metric: str, *, noise: float = 0.0, fmt: str = "{:.4g}"):
    """One row per rung of each ``{"model", "knob", "tuned", "values", "scores"}`` entry.

    The tuned rung carries a star and is the reference for the ``delta %`` column;
    ``noise`` is the seed spread of ``metric`` in per cent, which turns into the last
    column -- set only where the knob moves the metric further than a re-run would.
    """
    import pandas as pd

    rows = []
    for p in (q for q in panels if len(q["values"]) > 1):
        x = np.asarray(p["values"], dtype=float)
        y = np.asarray(p["scores"], dtype=float)
        j = int(np.argmin(np.abs(np.log(x) - np.log(float(p["tuned"])))))
        for i in np.argsort(x):
            d = (y[i] - y[j]) / y[j] * 100.0
            rows.append({"model": p["model"], "knob": p["knob"],
                         "value": fmt.format(x[i]) + (" *" if i == j else ""),
                         metric: y[i], "delta %": d,
                         "> seed": "yes" if noise > 0 and abs(d) > noise else ""})
    assert rows, "nothing to tabulate: no knob has more than one rung"
    df = pd.DataFrame(rows).set_index(["model", "knob", "value"])
    if noise <= 0:
        df = df.drop(columns="> seed")
    return df.round({metric: 4, "delta %": 1})
