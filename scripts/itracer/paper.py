import pandas as pd

from scripts import records as rec

NAME = "itracer_bench"
POINT = "tuned"
DIMS = (20, 50)
SEEDS = tuple(range(5))
ARMS = ("cfm", "mfm_land", "curly", "ffm_expr", "ffm_velo", "ffm", "ffm_lin_velo")
LABELS = {"cfm": "OT-CFM", "mfm_land": "MFM / LAND", "curly": "Curly-FM", "ffm_expr": "FFM, expression",
          "ffm_velo": "FFM + velocity", "ffm": "FFM + lineage", "ffm_lin_velo": "FFM + velocity + lineage"}
KEYS = (("mid W2", "interpolation"), ("end W2", "endpoint"))


def _parts(dim, arm, seed):
    return POINT, f"d{dim}", f"{arm}_s{seed}.json"


def expected():
    """The Table 2 records: seven arms at d = 20 and 50, seeds 0-4, tuned point."""
    return [rec.path(NAME, *_parts(d, a, s)) for d in DIMS for a in ARMS for s in SEEDS]


def ensure(cfg=None):
    """Train only the missing records with the benchmark driver; nothing when all exist."""
    def build(missing):
        from .bench import run_all
        run_all(dims=DIMS, arms=ARMS, seeds=SEEDS, points=(POINT,), cfg=cfg)
    return rec.ensure(expected(), build)


def _cell(dim, arm, key):
    return rec.pm([rec.load_one(NAME, *_parts(dim, arm, s))[key]["W2"] for s in SEEDS], 2)


def _floor(dim, key):
    return f"{rec.load_one(NAME, *_parts(dim, ARMS[0], SEEDS[0]))['floors'][key]['W2']:.2f}"


def mid_end():
    """Midpoint and endpoint W2 per method and dimension, with the sampling floors."""
    cols = [(d, lab, key) for d in DIMS for lab, key in KEYS]
    rows = [{"Method": LABELS[a], **{(f"d = {d}", lab): _cell(d, a, key) for d, lab, key in cols}} for a in ARMS]
    rows.append({"Method": "sampling floor", **{(f"d = {d}", lab): _floor(d, key) for d, lab, key in cols}})
    df = rec.frame(rows, index="Method")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def table2():
    """Table 2, iTracer rows: state interpolation, midpoint W2."""
    rows = [{"Method": LABELS[a], **{f"d = {d}": _cell(d, a, "interpolation") for d in DIMS}} for a in ARMS]
    return rec.frame(rows, index="Method")


def figure_dataset():
    """The lineage-recorded cells on their UMAP, coloured by diffusion-pseudotime tertile."""
    import matplotlib.pyplot as plt

    from ffm.core.itracer_data import N_THIRDS, load_cloud

    from .figures import THIRD_COLOUR, _bare, third_label
    cloud = load_cloud()
    U = cloud.umap
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for b in range(N_THIRDS):
        m = cloud.third == b
        ax.scatter(U[m, 0], U[m, 1], s=4.4, alpha=0.85, linewidths=0, color=THIRD_COLOUR[b],
                   label=third_label(cloud, b))
    ax.set_title(f"iTracer R2 hindbrain, {len(U)} lineage-recorded cells\n"
                 f"tertiles of the NPC $\\to$ neuron pseudotime at "
                 f"$s = {cloud.edges[0]:.3f},\\ {cloud.edges[1]:.3f}$", fontsize=11)
    ax.legend(loc="best", frameon=False, fontsize=8, markerscale=2.4)
    _bare(ax)
    fig.tight_layout()
    stem = rec.figure_path("itracer_umap")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.show()
