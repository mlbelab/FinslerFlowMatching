"""Third-party reference code, transcribed so that this tree stands on its own.

Two baselines in this repository are the authors' own code rather than our reading of
their papers, and until now they were *imported from a checkout on the author's machine*,
put on ``sys.path`` at import time.  That made the shipped half unrunnable anywhere else:
clone the repository, and ``scripts.core.curly_fm`` failed at its third import line.

So the handful of functions we actually call are copied in here, byte for byte, under
their own MIT licences and with the file and line they came from stated at the top of
each module.  Nothing is paraphrased, tidied or renamed; where a copy deviates from the
upstream text at all, the deviation is a comment saying so and why.

**The copies are checked against the originals rather than trusted.**  The upstream
provenance of each one is recorded in the ``RELEASES`` / ``VENDORED`` tables below, and
the diff-against-upstream check runs in the full research repository whenever a checkout
is present on the machine (``$MFM_RELEASE`` / ``$CURLY_RELEASE``).

This package is the *only* place in this tree that holds someone else's implementation.
Ports that are ours — :mod:`scripts.experiments.erythroid.reference_release` — are
transcriptions of a *published experiment* into our own idiom and live with the
experiment that runs them; these are the upstream *library*, unchanged, and live with the
engine that calls them.
"""
from __future__ import annotations

import os
import pathlib

# --------------------------------------------------------------------------- #
#  Provenance, machine-readable
# --------------------------------------------------------------------------- #
# Where each upstream release sits, and where it came from.  Nothing imports through
# these -- they exist so ``verify`` can diff the copies below against the originals, and
# they are **env-only** rather than carrying a default checkout path: a default would be
# one machine's directory layout baked into a published file, and the check already
# reports a clean skip when the variable is unset.
RELEASES: dict[str, tuple[str, str]] = {
    # key: (env var, upstream repository)
    "mfm":   ("MFM_RELEASE",   "https://github.com/kksniak/metric-flow-matching"),
    "curly": ("CURLY_RELEASE", "https://github.com/kpetrovicc/curly-flow-matching"),
}

# One entry per vendored module: which release it came from, which file in that release,
# and the top-level names copied out of it.  ``whole_file=True`` means the copy is the
# upstream module in full, so ``verify`` diffs every top-level definition rather than
# only the listed ones -- the stronger check, and the reason :mod:`mfm_land` keeps a
# function nothing here calls.
VENDORED: dict[str, dict] = {
    "mfm_land": {
        "release": "mfm",
        "path": "mfm/geo_metrics/land.py",
        "names": ("weighting_function", "land_metric_tensor", "weighting_function_dt"),
        "whole_file": True,
    },
    "curly_release": {
        "release": "curly",
        "path": "src/models/components/single_marginal_utils.py",
        "names": ("get_xt", "get_xt_xt_dot", "get_u_xt", "coupling"),
        "whole_file": False,
        # ``MLP`` comes from a second file in the same release, diffed alongside
        "extra": (("src/models/components/mlp.py", ("MLP",)),),
    },
}


def release_root(key: str) -> pathlib.Path | None:
    """Path to an upstream checkout, or ``None`` when this machine has none.

    Read from the release's environment variable and nowhere else: with no variable set
    there is no checkout to diff against, which is the ordinary case on any machine but
    the author's and is reported as a skip rather than a failure.
    """
    env, _upstream = RELEASES[key]
    raw = os.environ.get(env)
    if not raw:
        return None
    root = pathlib.Path(raw).expanduser()
    return root if root.is_dir() else None
