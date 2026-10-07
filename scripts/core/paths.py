"""Where run artefacts go — one function, so nothing writes next to its own source.

A run tree is hundreds of MB of weights and trajectory slabs and it is *not* part of the
repository: what ships is the code, the data caches under ``data/``, and the small
JSON record sets the notebooks cannot regenerate.  So artefacts land outside the source
tree by default and every driver asks here rather than deciding for itself.

Resolution order, first hit wins:

1. ``$FINSLER_OUT`` — an explicit override, for a cluster scratch disk or a one-off run.
2. ``<root>/outputs`` — the default, gitignored.

Two entries and not one because a cluster job usually wants its output on a scratch
filesystem while the notebook beside it wants the working copy, and neither should have
to edit the other's path to get it.
"""
from __future__ import annotations

import os

#: environment override, consulted before anything on disk
ENV_VAR = "FINSLER_OUT"

#: the source root — the directory holding ``scripts/`` and ``notebooks/``.  Three levels
#: up from this file, which sits at ``<root>/scripts/core/paths.py``.
PUBLIC_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rel(path) -> str:
    """A path as it reads *in the repository*, for anything that gets printed.

    Everything this module hands out is absolute, which is what makes a notebook and a
    batch cell agree on a file.  Printing one is a different matter: the stored output of
    a notebook ships with the repository, so an absolute path writes the machine it was
    executed on -- a username, a home directory, whatever the tree was called there --
    into a file a reader downloads.  The relative form says the same thing to that reader
    and nothing else besides.

    A path outside the source tree has no relative form worth printing (a ``$FINSLER_OUT``
    scratch disk is not reachable from ``<root>/``), so it comes back unchanged: better a
    caller prints a full path it meant to print than ``../../../scratch/...``.  For the
    same reason this only ever *shortens* -- a path that arrives relative is already in
    the form we want and is handed straight back, rather than being resolved against
    whatever directory the caller happens to be sitting in.
    """
    raw = os.fspath(path)
    if not os.path.isabs(raw):
        return raw
    absolute = os.path.abspath(raw)
    if absolute == PUBLIC_ROOT:
        return "."
    prefix = PUBLIC_ROOT + os.sep
    return absolute[len(prefix):] if absolute.startswith(prefix) else absolute


def output_root() -> str:
    """The directory under which every experiment writes its run tree."""
    override = os.environ.get(ENV_VAR)
    if override:
        return os.path.abspath(override)
    return os.path.join(PUBLIC_ROOT, "outputs")


def data_dir(name: str) -> str:
    """One versioned data cache, e.g. ``data_dir("pancreas_cache")``.

    Unlike run artefacts, the single-cell caches *are* shipped: they are a few MB each,
    they are what makes the real-data notebooks runnable from a fresh clone, and
    regenerating them needs a conda environment pinned to TensorFlow.  So they live under
    ``<public>/data/`` and are resolved relative to this file rather than to the working
    directory, which is what lets a notebook and a batch cell find the same file.
    """
    assert name and os.sep not in name and name not in (".", ".."), (
        f"cache name must be a single directory component, got {name!r}")
    return os.path.join(PUBLIC_ROOT, "data", name)


#: appended to an experiment name when the caller is running a wiring check
SMOKE_SUFFIX = "_smoke"


def experiment_root(name: str, smoke: bool = False) -> str:
    """One experiment's run tree, e.g. ``experiment_root("mfm_bench")``.

    ``name`` is a single path component by contract: these are sibling directories under
    one root, and a driver that could pass ``"../.."`` would silently write outside it.

    ``smoke=True`` returns a **separate sibling tree**, and that branch lives here rather
    than in each driver because forgetting it is not a cosmetic mistake.  A smoke pass
    runs every arm at a meaningless budget and writes records that are shaped exactly like
    reported ones; dropped into the reported tree they are picked up by the report, by the
    figures, and by the ``verify`` checks that read saved slabs — which is how a wiring
    check ends up silently redefining what the paper says.  One suffix, decided in the one
    place output locations come from, makes that impossible rather than merely discouraged.
    """
    assert name and os.sep not in name and name not in (".", ".."), (
        f"experiment name must be a single directory component, got {name!r}")
    return os.path.join(output_root(), f"{name}{SMOKE_SUFFIX}" if smoke else name)


def figure_dir(smoke: bool = False) -> str:
    """Where a notebook writes its figures: ``<public>/notebooks/out``.

    These are the exception to the rule at the top of this module -- they *are* part of
    the repository, because they are the figures the paper carries and they are a few MB
    in total, so unlike a run tree they belong beside the notebook that drew them.

    ``smoke=True`` gets ``out_smoke`` for exactly the reason :func:`experiment_root` gets
    its own tree: a wiring pass draws the same file names at a meaningless budget, and
    written into the shipped directory it silently replaces a paper figure with a picture
    of sixty iterations.  This is not hypothetical -- it happened here before the branch
    existed.  The path is absolute and rooted at this file, so a notebook executed from
    anywhere writes to the same place.
    """
    return os.path.join(PUBLIC_ROOT, "notebooks", f"out{SMOKE_SUFFIX}" if smoke else "out")


def table_file(name: str, smoke: bool = False) -> str:
    """Where a notebook stores the numbers behind its tables, e.g. ``"pancreas"``.

    A sibling of :func:`figure_dir` and for the same three reasons: these are results
    rather than run artefacts, they are kilobytes rather than hundreds of MB, and they
    are what the paper carries -- so they ship beside the notebook that printed them
    instead of in the gitignored run tree.

    The ``smoke`` branch matters more here than anywhere else in this module.  A wiring
    pass produces a table of exactly the right shape at a meaningless budget, and a table
    -- unlike a figure -- gives a reader no visual cue that the numbers came from sixty
    iterations.  Written into the shipped file it would silently become what the paper
    says.
    """
    assert name and os.sep not in name and name not in (".", ".."), (
        f"table name must be a single path component, got {name!r}")
    return os.path.join(PUBLIC_ROOT, "notebooks",
                        f"tables{SMOKE_SUFFIX}" if smoke else "tables", f"{name}.json")
