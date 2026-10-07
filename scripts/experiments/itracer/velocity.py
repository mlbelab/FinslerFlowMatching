"""RNA velocity for the iTracer R2 hindbrain, quantified from the public FASTQs.

He et al. (2022) never published spliced/unspliced counts: their Mendeley and ArrayExpress
releases carry one count matrix, and neither their Methods nor their code mentions velocyto,
kallisto or scVelo.  The raw reads *are* public, though -- ENA study PRJEB47712, four 10x
lanes for the two R2 microdissection reactions -- so velocity is recoverable by requantifying
them.  This module is that requantification, cut into stages that a scheduler can run
independently:

======================  ====================================  ===================
verb                    what it does                          cost
======================  ====================================  ===================
``preflight``           every precondition, and nothing else  ~1 min, 1 cpu
``fetch --index 0..3``  one run's R1+R2, md5-verified         ~1 h, 66 GB total
``ref``                 GENCODE GRCh38 + STAR genome index    ~1 h, 64 GB ram
``joincheck``           raw R1 barcodes vs the metadata       ~5 min, 1 cpu
``count --index 0..1``  STARsolo ``Velocyto`` per reaction    ~2 h, 128 GB ram
``assemble``            scVelo on the 2901 candidate cells    ~10 min
======================  ====================================  ===================

**Every stage is a gate.**  The expensive ones sit behind cheap ones that can refute the
whole plan: ``preflight`` HEADs all ten URLs and reparses the metadata before a byte is
downloaded, and ``joincheck`` reads barcodes out of one raw R1 -- no index, no alignment --
to prove the FASTQs and the published cell IDs share a barcode space *before* the two
STARsolo jobs start.  A stage that cannot prove its precondition raises; the sbatch chain is
wired ``afterok``, so a raise cancels everything downstream instead of burning a node on it.

The whitelist STARsolo is given is the published barcode list itself, one file per reaction,
matched ``Exact``.  That is deliberate: cell calling was already done by the authors, this
pipeline is only recovering the splice state of cells that are already in the benchmark, and
an exact match against a 1.5 k-barcode whitelist cannot invent a cell or pull a neighbouring
droplet's reads into one.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np

from scripts.core.itracer_data import (CACHE_DIR, VELO_H5AD, cloud_cell_ids, fit_velocity,
                                       normalize_for_velocity)

#: the scratch tree the FASTQs, the genome and the STAR index live in.  ~100 GB, none of it
#: in the repository and none of it needed once ``assemble`` has written its h5ad.
WORK = os.environ.get("ITRACER_VELO_WORK",
                      os.path.join(os.path.expanduser("~"), "scratch", "itracer_velocity"))

_ENA = "https://ftp.sra.ebi.ac.uk/vol1/run/ERR681/{acc}/{stem}_{read}_001.fastq.gz"

#: the four 10x runs of ENA study PRJEB47712 that are R2, with the sizes and checksums the
#: ENA portal reports.  I1 is not listed because STARsolo never reads it.
RUNS = (
    dict(acc="ERR6812893", reaction="Region2.1", stem="S2_1_3_10x_S7_L001",
         size=dict(R1=4501615486, R2=10757449844),
         md5=dict(R1="186e6e6701f6c66655db17c6041c798d",
                  R2="a752b90f66e5179bb1bbb04e5130db83")),
    dict(acc="ERR6812894", reaction="Region2.1", stem="S2_1_3_10x_S7_L002",
         size=dict(R1=4723808382, R2=11262684337),
         md5=dict(R1="2a438b300f0a7fe80fa441256dfd7bc2",
                  R2="3f2c1aeb4f39730c0e328e71f496e4fa")),
    dict(acc="ERR6812899", reaction="Region2.2", stem="S2_2_4_10x_S10_L001",
         size=dict(R1=4974597527, R2=11953032941),
         md5=dict(R1="4637c4721934f495983e726099442995",
                  R2="f3994fdee6ec7f37f2795f0b6a514f04")),
    dict(acc="ERR6812900", reaction="Region2.2", stem="S2_2_4_10x_S10_L002",
         size=dict(R1=5228871234, R2=12537239098),
         md5=dict(R1="e929dfc1e6e2563548f40c6fcc19e076",
                  R2="f49a028a8905fc08761b29e4393ea4ca")),
)

#: the two microdissection reactions, in the order ``count --index`` uses.
REACTIONS = ("Region2.1", "Region2.2")

_GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44"
REF_URL = {
    "genome.fa.gz": f"{_GENCODE}/GRCh38.primary_assembly.genome.fa.gz",
    "genes.gtf.gz": f"{_GENCODE}/gencode.v44.primary_assembly.annotation.gtf.gz",
}

#: Chromium cell-barcode length.  The UMI length separates v2 from v3 and is *measured* by
#: :func:`cmd_joincheck` off the raw R1 rather than assumed.
CB_LEN = 16

#: ``joincheck`` reads this many R1 records per reaction.  At ~1.5 k cells a lane this is
#: some thousands of reads per cell -- far past what is needed to see every barcode once.
JOIN_READS = 4_000_000

#: the gate.  ``seen`` is the fraction of published barcodes observed at all; ``share`` is
#: the fraction of sampled reads carrying one.  Random 16-mers would give 0 and 0.
JOIN_MIN_SEEN, JOIN_MIN_SHARE = 0.90, 0.02

#: free bytes ``preflight`` insists on under :data:`WORK`: 66 GB of FASTQ, ~33 GB of STAR
#: index, the uncompressed genome, and room to be wrong about all three.
NEED_BYTES = 250 * 1024 ** 3


def _p(*a) -> str:
    return os.path.join(WORK, *a)


def _say(msg: str) -> None:
    print(f"[velocity] {msg}", flush=True)


def _die(msg: str):
    raise SystemExit(f"[velocity] FAIL: {msg}")


def _run(cmd: list, **kw) -> None:
    _say("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# --------------------------------------------------------------------------- #
#  The published barcodes
# --------------------------------------------------------------------------- #
def published_barcodes(cache_dir: str = CACHE_DIR) -> dict:
    """``{reaction: array of 16-mers}`` for the two R2 reactions, from ``meta.tsv.gz``.

    Every cell ID in the release reads ``<reaction>_<barcode>`` with no lane suffix, so the
    join key against a raw R1 is the ID's tail, unmodified.
    """
    path = os.path.join(cache_dir, "meta.tsv.gz")
    if not os.path.exists(path):
        _die(f"{path} missing; run `python -m scripts.experiments.itracer.bench --prebuild`")
    out = {r: [] for r in REACTIONS}
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            cid = line.split("\t", 1)[0]
            reaction, _, bc = cid.partition("_")
            if reaction in out:
                out[reaction].append(bc)
    for r, v in out.items():
        if not v:
            _die(f"no cells for {r} in meta.tsv.gz")
        bad = [b for b in v[:50] if len(b) != CB_LEN or set(b) - set("ACGT")]
        if bad:
            _die(f"{r} barcodes are not bare {CB_LEN}-mers, e.g. {bad[:3]}")
    return {r: np.asarray(v) for r, v in out.items()}


# --------------------------------------------------------------------------- #
#  Stage 0 -- preflight
# --------------------------------------------------------------------------- #
def _head(url: str) -> tuple:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, int(r.headers.get("Content-Length") or 0)
    except Exception as e:  # noqa: BLE001 -- any failure here is the same failure
        return None, str(e)


def cmd_preflight(args) -> None:
    """Refute the plan in a minute or admit it cannot.  Nothing here downloads."""
    fails = []

    os.makedirs(WORK, exist_ok=True)
    free = shutil.disk_usage(WORK).free
    _say(f"{WORK}: {free / 1024 ** 3:.0f} GB free (need {NEED_BYTES / 1024 ** 3:.0f})")
    if free < NEED_BYTES:
        fails.append(f"only {free / 1024 ** 3:.0f} GB free under {WORK}")

    star = shutil.which("STAR")
    _say(f"STAR: {star or 'MISSING'}")
    if star is None:
        fails.append("STAR not on PATH -- the sbatch scripts `module load STAR/2.7.11b-GCC-13.3.0`")
    else:
        v = subprocess.run([star, "--version"], capture_output=True, text=True)
        _say(f"STAR --version -> {v.stdout.strip() or v.stderr.strip()}")

    try:
        import scvelo
        import scanpy
        _say(f"scvelo {scvelo.__version__}, scanpy {scanpy.__version__}")
    except Exception as e:  # noqa: BLE001
        fails.append(f"assemble's imports are broken: {e}")

    # the ten URLs, HEADed.  A fastq whose advertised length disagrees with the ENA portal's
    # submitted_bytes means the accession moved and the checksums below are stale.
    for run in RUNS:
        for read in ("R1", "R2"):
            url = _ENA.format(acc=run["acc"], stem=run["stem"], read=read)
            status, got = _head(url)
            want = run["size"][read]
            ok = status == 200 and got == want
            _say(f"{'ok ' if ok else 'BAD'} {run['acc']}/{read} {status} {got} (want {want})")
            if not ok:
                fails.append(f"{url}: {status}, {got} bytes, expected {want}")
    for name, url in REF_URL.items():
        status, got = _head(url)
        ok = status == 200 and isinstance(got, int) and got > 0
        _say(f"{'ok ' if ok else 'BAD'} {name} {status} {got}")
        if not ok:
            fails.append(f"{url}: {status} {got}")

    bcs = published_barcodes()
    for r in REACTIONS:
        _say(f"{r}: {len(bcs[r])} published barcodes")

    ids = cloud_cell_ids()
    known = {f"{r}_{b}" for r in REACTIONS for b in bcs[r]}
    missing = [c for c in ids if c not in known]
    _say(f"cloud: {len(ids)} candidate cells, {len(missing)} not resolvable to an R2 barcode")
    if missing:
        fails.append(f"{len(missing)} cloud cells have no R2 barcode, e.g. {missing[:3]}")

    if fails:
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        _die(f"{len(fails)} precondition(s) failed; the chain will not run")
    _say("preflight clean")


# --------------------------------------------------------------------------- #
#  Stage 1 -- fetch
# --------------------------------------------------------------------------- #
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(args) -> None:
    """One run's R1 and R2, resumable, verified by size *and* md5 before ``.done``."""
    run = RUNS[args.index]
    out = _p("fastq", run["acc"])
    os.makedirs(out, exist_ok=True)
    for read in ("R1", "R2"):
        dst = os.path.join(out, f"{run['stem']}_{read}_001.fastq.gz")
        done = dst + ".done"
        if os.path.exists(done) and os.path.getsize(dst) == run["size"][read]:
            _say(f"{os.path.basename(dst)} already verified")
            continue
        url = _ENA.format(acc=run["acc"], stem=run["stem"], read=read)
        _run(["curl", "--fail", "--location", "--continue-at", "-",
              "--retry", "5", "--retry-delay", "20", "--speed-time", "300",
              "--speed-limit", "10000", "-o", dst, url])
        got = os.path.getsize(dst)
        if got != run["size"][read]:
            _die(f"{dst}: {got} bytes, ENA says {run['size'][read]}")
        digest = _md5(dst)
        if digest != run["md5"][read]:
            _die(f"{dst}: md5 {digest}, ENA says {run['md5'][read]}")
        open(done, "w").close()
        _say(f"{os.path.basename(dst)} ok ({got / 1024 ** 3:.1f} GB, md5 matched)")


def fastqs_for(reaction: str) -> dict:
    """``{'R1': [...], 'R2': [...]}`` -- both lanes of one reaction, in lane order.

    Raises if any file is missing or unverified, so a ``count`` job started on a partial
    download dies in its first second rather than three hours in.
    """
    out = {"R1": [], "R2": []}
    for run in RUNS:
        if run["reaction"] != reaction:
            continue
        for read in ("R1", "R2"):
            path = _p("fastq", run["acc"], f"{run['stem']}_{read}_001.fastq.gz")
            if not os.path.exists(path + ".done"):
                _die(f"{path} not fetched (no .done); stage `fetch` did not complete")
            if os.path.getsize(path) != run["size"][read]:
                _die(f"{path} changed size since it was verified")
            out[read].append(path)
    if not out["R1"]:
        _die(f"no runs for reaction {reaction}")
    return out


# --------------------------------------------------------------------------- #
#  Stage 2 -- the reference
# --------------------------------------------------------------------------- #
GENOME_DIR = _p("star_GRCh38")


def cmd_ref(args) -> None:
    """GENCODE v44 primary assembly, then ``STAR --runMode genomeGenerate``."""
    if shutil.which("STAR") is None:
        _die("STAR not on PATH")
    ref = _p("ref")
    os.makedirs(ref, exist_ok=True)
    plain = {}
    for name, url in REF_URL.items():
        gz, flat = os.path.join(ref, name), os.path.join(ref, name[:-3])
        plain[name] = flat
        if os.path.exists(flat) and os.path.getsize(flat) > 0:
            _say(f"{name[:-3]} present")
            continue
        if not os.path.exists(gz):
            _run(["curl", "--fail", "--location", "--retry", "5", "-o", gz, url])
        _run(["gzip", "-t", gz])
        with gzip.open(gz, "rb") as fi, open(flat + ".part", "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 24)
        os.replace(flat + ".part", flat)

    if os.path.exists(os.path.join(GENOME_DIR, "SA")):
        _say("STAR index present")
        return
    os.makedirs(GENOME_DIR, exist_ok=True)
    _run(["STAR", "--runMode", "genomeGenerate",
          "--runThreadN", str(args.threads),
          "--genomeDir", GENOME_DIR,
          "--genomeFastaFiles", plain["genome.fa.gz"],
          "--sjdbGTFfile", plain["genes.gtf.gz"],
          "--sjdbOverhang", "100",
          "--outTmpDir", _p("star_tmp_ref")])
    if not os.path.exists(os.path.join(GENOME_DIR, "SA")):
        _die("genomeGenerate produced no SA")
    _say(f"index built in {GENOME_DIR}")


# --------------------------------------------------------------------------- #
#  Stage 2.5 -- the join gate
# --------------------------------------------------------------------------- #
def _sample_r1(path: str, n_reads: int) -> tuple:
    """``(Counter of CB_LEN-mers, R1 read length, reads read)`` off the head of an R1."""
    from collections import Counter

    counts, length, seen = Counter(), None, 0
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i % 4 != 1:
                continue
            seq = line.rstrip("\n")
            if length is None:
                length = len(seq)
            counts[seq[:CB_LEN]] += 1
            seen += 1
            if seen >= n_reads:
                break
    return counts, length, seen


def cmd_joincheck(args) -> None:
    """Do the raw reads and the published cell IDs share a barcode space?

    This is the cheap refutation of the whole pipeline.  It touches one R1 per reaction,
    parses nothing but sequence lines, and needs neither the genome nor the index -- so it
    can run the moment the first download lands and kill the two 128 GB STARsolo jobs before
    they are scheduled.  It also *measures* the UMI length from the R1 read length instead of
    assuming a chemistry, and writes it where :func:`cmd_count` will read it.
    """
    bcs = published_barcodes()
    report, umi_lens = {}, set()
    for reaction in REACTIONS:
        r1 = fastqs_for(reaction)["R1"][0]
        counts, length, seen = _sample_r1(r1, args.reads)
        want = set(bcs[reaction].tolist())
        hit = want & set(counts)
        share = sum(counts[b] for b in hit) / max(seen, 1)
        rc = str.maketrans("ACGT", "TGCA")
        rc_hit = {b.translate(rc)[::-1] for b in want} & set(counts)
        umi = length - CB_LEN
        umi_lens.add(umi)
        report[reaction] = dict(reads=seen, r1_len=length, umi_len=umi,
                                published=len(want), seen=len(hit),
                                seen_frac=len(hit) / len(want), read_share=share,
                                revcomp_seen=len(rc_hit))
        _say(f"{reaction}: {seen} reads, R1 {length} bp -> UMI {umi} bp; "
             f"{len(hit)}/{len(want)} published barcodes seen ({len(hit) / len(want):.1%}), "
             f"{share:.1%} of reads; revcomp control {len(rc_hit)}/{len(want)}")

    bad = [r for r, d in report.items()
           if d["seen_frac"] < JOIN_MIN_SEEN or d["read_share"] < JOIN_MIN_SHARE]
    if bad:
        _die(f"barcode join fails for {bad}: the FASTQs and the published IDs do not agree, "
             f"so requantifying them cannot produce velocity for these cells")
    if len(umi_lens) != 1:
        _die(f"the two reactions disagree on UMI length: {sorted(umi_lens)}")
    umi = umi_lens.pop()
    if umi not in (10, 12):
        _die(f"UMI length {umi} is neither 10x v2 (10) nor v3 (12)")
    os.makedirs(WORK, exist_ok=True)
    with open(_p("chem.json"), "w") as f:
        json.dump(dict(cb_len=CB_LEN, umi_len=umi, report=report), f, indent=2)
    _say(f"join ok; chemistry 10x {'v3' if umi == 12 else 'v2'} written to chem.json")


# --------------------------------------------------------------------------- #
#  Stage 3 -- STARsolo
# --------------------------------------------------------------------------- #
def solo_dir(reaction: str) -> str:
    return _p("solo", reaction)


def cmd_count(args) -> None:
    """``Gene`` + ``Velocyto`` counts for one reaction, both lanes in one pass."""
    reaction = REACTIONS[args.index]
    fq = fastqs_for(reaction)                      # raises unless both lanes are verified
    if not os.path.exists(os.path.join(GENOME_DIR, "SA")):
        _die(f"{GENOME_DIR} holds no STAR index; stage `ref` did not complete")
    if not os.path.exists(_p("chem.json")):
        _die("chem.json missing; stage `joincheck` did not complete")
    chem = json.load(open(_p("chem.json")))

    out = solo_dir(reaction)
    os.makedirs(out, exist_ok=True)
    wl = os.path.join(out, "whitelist.txt")
    bcs = published_barcodes()[reaction]
    with open(wl, "w") as f:
        f.write("\n".join(bcs.tolist()) + "\n")
    _say(f"{reaction}: whitelist of {len(bcs)} published barcodes, matched Exact")

    tmp = _p(f"star_tmp_{reaction}")
    shutil.rmtree(tmp, ignore_errors=True)
    _run(["STAR",
          "--runThreadN", str(args.threads),
          "--genomeDir", GENOME_DIR,
          "--readFilesIn", ",".join(fq["R2"]), ",".join(fq["R1"]),
          "--readFilesCommand", "zcat",
          "--soloType", "CB_UMI_Simple",
          "--soloCBwhitelist", wl,
          "--soloCBlen", str(chem["cb_len"]),
          "--soloUMIstart", str(chem["cb_len"] + 1),
          "--soloUMIlen", str(chem["umi_len"]),
          "--soloBarcodeReadLength", "0",
          "--soloCBmatchWLtype", "Exact",
          "--soloStrand", "Forward",
          "--soloFeatures", "Gene", "Velocyto",
          "--soloCellFilter", "None",
          "--soloMultiMappers", "Unique",
          "--outSAMtype", "None",
          "--outFileNamePrefix", out + "/",
          "--outTmpDir", tmp])
    mtx = os.path.join(out, "Solo.out", "Velocyto", "raw", "spliced.mtx")
    if not os.path.exists(mtx):
        _die(f"STARsolo wrote no {mtx}")
    _say(f"{reaction}: Velocyto matrices in {os.path.dirname(mtx)}")


# --------------------------------------------------------------------------- #
#  Stage 4 -- assemble
# --------------------------------------------------------------------------- #
def _read_velocyto(reaction: str):
    """``AnnData`` of one reaction's raw Velocyto output, cells named as the release does."""
    import anndata as ad
    import pandas as pd
    from scipy.io import mmread

    raw = os.path.join(solo_dir(reaction), "Solo.out", "Velocyto", "raw")
    if not os.path.isdir(raw):
        _die(f"{raw} missing; stage `count --index {REACTIONS.index(reaction)}` did not run")
    var = pd.read_csv(os.path.join(raw, "features.tsv"), sep="\t", header=None)
    obs = pd.read_csv(os.path.join(raw, "barcodes.tsv"), header=None)[0].to_numpy()
    # STARsolo writes genes x cells with the three layers as three value columns.
    m = mmread(os.path.join(raw, "spliced.mtx")).T.tocsr()
    u = mmread(os.path.join(raw, "unspliced.mtx")).T.tocsr()
    A = ad.AnnData(X=m.copy())
    A.layers["spliced"], A.layers["unspliced"] = m, u
    A.obs_names = [f"{reaction}_{b}" for b in obs]
    # a bare list, not the frame's column: pandas would carry the integer column label
    # through as the index *name*, which AnnData rejects.
    A.var_names = var[1].astype(str).str.upper().tolist()
    A.var_names_make_unique()
    return A


def cmd_assemble(args) -> None:
    """Join both reactions onto the benchmark's cells and run scVelo."""
    import anndata as ad
    import scvelo as scv

    parts = [_read_velocyto(r) for r in REACTIONS]
    A = ad.concat(parts, join="inner")
    _say(f"{A.n_obs} quantified cells x {A.n_vars} genes")

    # every benchmark cell, or none.  The whitelist STARsolo was given is the published
    # barcode list, of which the cloud is a subset, so a missing cell means the two sides
    # disagree about what a cell ID is -- and a velocity field defined on part of the cloud
    # is worse than no velocity field, because every kernel built from it would silently be
    # on a different cell set than the kernels it is compared against.
    ids = cloud_cell_ids()
    have = set(A.obs_names)
    missing = [c for c in ids if c not in have]
    _say(f"join: {len(ids) - len(missing)}/{len(ids)} benchmark cells recovered")
    if len(missing) == len(ids):
        _die("the join is empty -- no quantified barcode matches a benchmark cell ID")
    if missing:
        _die(f"{len(missing)} benchmark cells were not quantified, e.g. {missing[:3]}; "
             f"refusing a partial cloud")
    A = A[ids].copy()                                   # cloud row order, exactly

    s = float(A.layers["spliced"].sum())
    u = float(A.layers["unspliced"].sum())
    _say(f"unspliced fraction {u / max(s + u, 1.0):.1%} (10x typically 0.15-0.30)")

    # both halves come from scripts.core.itracer_data, which is also what a per-split refit
    # calls: the artifact written here and the velocity a training set steers by must be the
    # same pipeline, or the cached graph below would describe a different estimate.
    normalize_for_velocity(A)
    fit_velocity(A)
    scv.tl.velocity_graph(A, n_jobs=args.threads)

    # a velocity field that is all zero or all NaN writes out perfectly happily and only
    # fails much later, inside a kernel, as an unexplained tie.  Refuse it here instead.
    V = np.asarray(A.layers["velocity"])
    fit = int(np.isfinite(A.var["velocity_gamma"].to_numpy()).sum())
    moving = int((np.nanstd(V, axis=0) > 0).sum())
    _say(f"velocity: gamma fitted for {fit}/{A.n_vars} genes, {moving} with non-zero field; "
         f"graph {A.uns['velocity_graph'].nnz} edges")
    if fit < A.n_vars // 2 or moving < A.n_vars // 2:
        _die("the velocity fit is mostly empty; refusing to write a degenerate field")

    out = os.path.join(CACHE_DIR, VELO_H5AD)
    A.write_h5ad(out)
    _say(f"wrote {out} ({A.n_obs} cells, {A.n_vars} genes, layers "
         f"{sorted(A.layers)}, obsp {sorted(A.obsp)})")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("preflight").set_defaults(fn=cmd_preflight)

    f = sub.add_parser("fetch")
    f.add_argument("--index", type=int, required=True, choices=range(len(RUNS)))
    f.set_defaults(fn=cmd_fetch)

    r = sub.add_parser("ref")
    r.add_argument("--threads", type=int, default=16)
    r.set_defaults(fn=cmd_ref)

    j = sub.add_parser("joincheck")
    j.add_argument("--reads", type=int, default=JOIN_READS)
    j.set_defaults(fn=cmd_joincheck)

    c = sub.add_parser("count")
    c.add_argument("--index", type=int, required=True, choices=range(len(REACTIONS)))
    c.add_argument("--threads", type=int, default=16)
    c.set_defaults(fn=cmd_count)

    a = sub.add_parser("assemble")
    a.add_argument("--threads", type=int, default=8)
    a.set_defaults(fn=cmd_assemble)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
