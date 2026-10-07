"""Per-experiment drivers.  One subpackage per benchmark, sharing one engine.

Everything reusable — the geometry, the two training paths, the arms, the metrics —
lives in :mod:`scripts.core` and is *identical* across experiments.  What lives here is
only what a given benchmark cannot share with another: its point cloud, its split, its
scoring target, its tables and its figure.

    :mod:`scripts.experiments.sheet`     the Sheet — route selection under a withheld region
    :mod:`scripts.experiments.forksheet` the ForkSheet — the same, with a bifurcation
    :mod:`scripts.experiments.pancreas`  scVelo endocrinogenesis, a withheld latent-time bin
    :mod:`scripts.experiments.erythroid` mouse gastrulation, the Curly-FM Table 4 port
    :mod:`scripts.experiments.itracer`   iTracer R2 hindbrain, a lineage-recorded section
    :mod:`scripts.experiments.matched`   the matched-information ablation, Sheet and Pancreas

The dependency direction is one-way: an experiment imports the engine, never the other
way round, and never another experiment's private helpers.  The one deliberate exception
is the *palette*: every experiment's ``figures`` module reads its colours from
:mod:`scripts.experiments.sheet.paper_figures`, and the reports reuse its table
primitives, so every figure and table in the paper shares one visual key rather than
drifting apart.
"""
