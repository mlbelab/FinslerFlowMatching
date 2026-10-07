"""Display name and colour per method arm.

A module of its own rather than a corner of a plotting file: the LaTeX tables, the
figures and the comparison reports all need these, and they must agree.

The paper calls ``ffm`` and ``pathb`` **one** method -- FFM, the SDE off and on -- and
tells them apart by a qualifier rather than by two acronyms.  Those strings used to be
spelled out again here, because the engine may not import an experiment and they lived in
``experiments/sheet/paper_focus.py``; they are now in :mod:`scripts.core.present`, which
is a sibling of this file, so the labels below are *built* from them and the duplication
is gone.
"""
from __future__ import annotations

from .present import QUAL_DET, QUAL_NOISY, PATHB_NAME

#: display names and a stable colour per arm, used by both figures and the LaTeX doc
ARM_LABELS = {
    # the deterministic family.  The parenthetical is the metric form, which is what
    # separates these three from each other; "deterministic" is what separates all three
    # from the noisy pair below, and both are needed in a table that prints all five.
    "ffm": f"{PATHB_NAME}, {QUAL_DET} (Randers)",
    "ffm_riem": f"{PATHB_NAME}, {QUAL_DET} (Riemannian)",
    "ffm_eucl": f"{PATHB_NAME}, {QUAL_DET} (Euclidean)",
    "ffm_drift": f"{PATHB_NAME}, {QUAL_DET} (drift only)",
    "cfm": "OT-CFM",
    "mfm_land": "MFM (LAND)",
    "curly": "Curly-FM",
    "pathb": f"{PATHB_NAME}, {QUAL_NOISY}",
    "pathb_iso": f"{PATHB_NAME}, {QUAL_NOISY} ($G\\equiv I$)",
    "pathb_const": f"{PATHB_NAME}, {QUAL_NOISY} ($M_t\\equiv\\bar M$)",
}
ARM_COLOURS = {
    "ffm": "#d62728",
    "ffm_riem": "#ff7f0e",
    "ffm_eucl": "#8c564b",
    "ffm_drift": "#e377c2",
    "cfm": "#1f77b4",
    "mfm_land": "#2ca02c",
    "curly": "#17becf",
    "pathb": "#9467bd",
    "pathb_iso": "#7f7f7f",
    "pathb_const": "#bcbd22",
}
