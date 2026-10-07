"""Route-selection benchmarks: bridge an unobserved gap and pick the right way round.

The usual leave-one-timepoint-out protocol deletes a random quartile of cells and asks
"how much does a missing timepoint cost?".  This benchmark asks a sharper question:

    p_0 and p_1 are two well-separated populations.  The region between them is
    **never shown to the model** -- not to the flow, not to the geometry, not to P.
    It exists only as the evaluation target.  Does the model route through it the way
    the dynamics say, or does it take the shortcut?

The cloud is built so that the *Euclidean* answer and the *dynamical* answer differ,
and so that the difference is visible by eye rather than only in a scalar.  There is
one: the :mod:`~scripts.experiments.sheet.datasets` **Sheet**.
"""
from .datasets import (  # noqa: F401
    DATASET_NAMES,
    REGION_COLOURS,
    REGION_LABELS,
    REGION_NAMES,
    DECOY,
    GAP,
    SOURCE,
    TARGET,
    RouteDataset,
    make_dataset,
)
