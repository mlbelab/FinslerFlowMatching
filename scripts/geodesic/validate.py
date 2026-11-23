from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import ot
import pandas as pd
import torch

from ffm.core.paths import experiment_root
from ffm.method import Coupler, GeoPathNet, Run, TrueFW, interpolant

NAME = "geodesic_validation"
CASES = ("randers", "sphere", "wind")
DIMS = (2, 3, 5, 10, 20, 50, 100)
SEEDS = (0, 1, 2)
KAPPA, C_EFF, LAM = 100.0, 0.9, 1.0
RADIAL, OMEGA, CAP, ARC = 10.0, 0.5, 0.35, np.pi / 2
GRID_DIM = 20
GRID = ((1.0, 0.9), (10.0, 0.9), (1000.0, 0.9), (100.0, 0.0), (100.0, 0.5), (100.0, 0.99))
ITERS, BUDGET = 10000, 2500
PROBES = tuple(sorted(set(np.round(np.logspace(1, 4, 31)).astype(int).tolist()) | {BUDGET}))
N_TRAIN, N_EVAL, N_PROBE, OT_PTS = 2000, 256, 64, 200
NODES = (8, 32, 256)
T_FINE = 65
CHUNK = 4096


class Analytic(TrueFW):

    def __init__(self, d: int, cfg: Run, C, m, lam: float = LAM):
        self.cfg, self.d, self.device, self.dtype = cfg, d, cfg.device, cfg.dtype
        self.C, self.m, self.lam, self.scale = C, m, float(lam), 1.0

    def _fields(self, x):
        return torch.linalg.inv(self.C(x)), self.m(x)


def rotate(x, phi):
    c, s = torch.cos(phi), torch.sin(phi)
    y = x.clone()
    y[..., 0] = c * x[..., 0] - s * x[..., 1]
    y[..., 1] = s * x[..., 0] + c * x[..., 1]
    return y


class Case:

    def __init__(self, case: str, d: int, cfg: Run, kappa: float = KAPPA, c_eff: float = C_EFF,
                 seed: int = 0):
        self.case, self.d, self.cfg = case, d, cfg
        self.omega = OMEGA if case == "wind" else 0.0
        kw = cfg.torch_kw
        self.I = torch.eye(d, **kw)
        J = torch.zeros(d, d, **kw)
        J[1, 0], J[0, 1] = 1.0, -1.0
        self.J = J
        if case == "randers":
            rng = np.random.default_rng(1000 + seed)
            Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
            C0 = (Q * np.logspace(0.0, np.log10(kappa), d)) @ Q.T
            u = rng.standard_normal(d)
            mu = LAM * c_eff / np.sqrt(1.0 - c_eff ** 2)
            m0 = u * mu / np.sqrt(u @ np.linalg.solve(C0, u))
            C0_t, m0_t = cfg.tensor(C0), cfg.tensor(m0)
            self.metric = Analytic(d, cfg, lambda x: C0_t.expand(len(x), d, d),
                                   lambda x: m0_t.expand(len(x), d))
        else:
            self.metric = Analytic(d, cfg, self._sphere_C, self._wind)

    def _wind(self, x):
        return self.omega * x @ self.J.T

    def _sphere_C(self, x):
        r2 = (x * x).sum(1)
        n = x / r2.sqrt()[:, None]
        nn = n[:, :, None] * n[:, None, :]
        W = self._wind(x)
        return (r2[:, None, None] * (nn / RADIAL ** 2 + self.I - nn)
                - W[:, :, None] * W[:, None, :])

    def zermelo(self, x, v):
        r2 = (x * x).sum(1)
        n = x / r2.sqrt()[:, None]

        def h(a, b):
            return ((a * b).sum(1) + (RADIAL ** 2 - 1.0) * (n * a).sum(1) * (n * b).sum(1)) / r2
        W = self._wind(x)
        s = 1.0 - h(W, W)
        return (torch.sqrt(h(v, W) ** 2 + s * h(v, v)) - h(v, W)) / s

    def sample(self, n: int, side: int, gen: torch.Generator):
        kw = self.cfg.torch_kw
        if self.case == "randers":
            z = torch.randn(n, self.d, generator=gen).to(**kw) * (0.5 / np.sqrt(self.d))
            z[:, 0] += 0.5 if side else -0.5
            return z
        c = torch.zeros(self.d, **kw)
        c[0], c[1] = (np.cos(ARC), np.sin(ARC)) if side else (1.0, 0.0)
        u = torch.randn(n, self.d, generator=gen).to(**kw)
        u = u - (u @ c)[:, None] * c[None]
        u = u / u.norm(dim=1, keepdim=True)
        psi = CAP * torch.rand(n, 1, generator=gen).to(**kw)
        return torch.cos(psi) * c[None] + torch.sin(psi) * u

    def length(self, x0, x1):
        if self.case == "randers":
            return torch.cat([self.metric.F(x0[s:s + CHUNK], (x1 - x0)[s:s + CHUNK])
                              for s in range(0, len(x0), CHUNK)])
        a, b = x0.double(), x1.double()
        if self.omega == 0.0:
            return torch.arccos((a * b).sum(1).clamp(-1.0, 1.0)).to(x0.dtype)
        lo = torch.zeros(len(a), dtype=a.dtype, device=a.device)
        hi = torch.full_like(lo, np.pi / (1.0 - self.omega) + 1.0)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            g = torch.arccos((a * rotate(b, -self.omega * mid)).sum(1).clamp(-1.0, 1.0)) - mid
            lo, hi = torch.where(g > 0, mid, lo), torch.where(g > 0, hi, mid)
        return (0.5 * (lo + hi)).to(x0.dtype)

    def exact(self, x0, x1, t, L):
        if self.case == "randers":
            return ((1.0 - t)[:, None, None] * x0[None] + t[:, None, None] * x1[None],
                    (x1 - x0)[None].expand(len(t), *x0.shape))
        y = rotate(x1, -self.omega * L)
        u = y - (x0 * y).sum(1, keepdim=True) * x0
        u = u / u.norm(dim=1, keepdim=True)
        s = t[:, None, None] * L[None, :, None]
        sig = torch.cos(s) * x0[None] + torch.sin(s) * u[None]
        dsig = -torch.sin(s) * x0[None] + torch.cos(s) * u[None]
        phi = self.omega * s[..., 0]
        x = rotate(sig, phi)
        return x, L[None, :, None] * (rotate(dsig, phi) + self.omega * x @ self.J.T)


def midpoints(K: int, cfg: Run):
    return (torch.arange(K, **cfg.torch_kw) + 0.5) / K


def path_at(phi, x0, x1, tk: float, cfg: Run):
    tt = torch.full((len(x0), 1), tk, **cfg.torch_kw)
    with torch.enable_grad():
        x, dx = interpolant(phi, tt, x0, x1, create_graph=False)
    return x.detach(), dx.detach()


def pair_cost(phi, metric, x0, x1, K: int, cfg: Run):
    out = []
    for s in range(0, len(x0), CHUNK):
        a, b = x0[s:s + CHUNK], x1[s:s + CHUNK]
        acc = torch.zeros(len(a), **cfg.torch_kw)
        for tk in midpoints(K, cfg).tolist():
            acc = acc + metric.F2(*path_at(phi, a, b, tk, cfg))
        out.append(acc / K)
    return torch.cat(out)


def exact_cost(case: Case, x0, x1, L, K: int, cfg: Run):
    acc = torch.zeros(len(x0), **cfg.torch_kw)
    for tk in midpoints(K, cfg):
        x, dx = case.exact(x0, x1, tk[None], L)
        acc = acc + case.metric.F2(x[0], dx[0])
    return acc / K


def _q(v, p):
    return float(np.quantile(np.asarray(v, dtype=np.float64), p))


def evaluate(case: Case, phi, ev: dict, cfg: Run) -> dict:
    m, (x0, x1), E = case.metric, ev["pairs"], ev["E"]
    out = {}
    for K in NODES:
        rel = ((pair_cost(phi, m, x0, x1, K, cfg) - E) / E).cpu().numpy()
        out[f"rel_K{K}"] = float(np.median(np.abs(rel)))
        out[f"rel_K{K}_mean"] = float(rel.mean())
        out[f"rel_K{K}_p90"] = _q(np.abs(rel), 0.9)
        if K == NODES[-1]:
            out["below_exact"] = float((rel < -1e-3).mean())
    xh, dxh = map(torch.stack, zip(*[path_at(phi, x0, x1, tk, cfg)
                                     for tk in torch.linspace(0.0, 1.0, T_FINE).tolist()]))
    chord = (x1 - x0).norm(dim=1)
    dev = ((xh - ev["path"]).norm(dim=2).max(0).values / chord).cpu().numpy()
    curve = (torch.cdist(xh.transpose(0, 1), ev["dense"].transpose(0, 1)).min(2).values.max(1).values
             / chord).cpu().numpy()
    speed = torch.stack([m.F(xh[k], dxh[k]) for k in range(T_FINE)])
    out.update({"path_dev": float(np.median(dev)), "path_dev_p90": _q(dev, 0.9),
                "curve_dev": float(np.median(curve)), "curve_dev_p90": _q(curve, 0.9),
                "speed_cv": float((speed.std(0) / speed.mean(0)).median())})
    a, b = ev["ot"]
    n0, n1 = len(a), len(b)
    A = a[:, None].expand(n0, n1, case.d).reshape(-1, case.d)
    B = b[None].expand(n0, n1, case.d).reshape(-1, case.d)
    Ch = pair_cost(phi, m, A, B, cfg.ot_k, cfg).reshape(n0, n1).double().cpu().numpy()
    Cs = ev["C_ot"]
    ph, ps = [ot.sinkhorn(np.full(n0, 1 / n0), np.full(n1, 1 / n1), C,
                          float(cfg.ot_blur_frac * np.median(C) + 1e-12),
                          method="sinkhorn_log", numItermax=cfg.ot_n_iter) for C in (Ch, Cs)]
    out.update({"plan_tv": float(0.5 * np.abs(ph - ps).sum()),
                "plan_subopt": float((ph * Cs).sum() / (ps * Cs).sum() - 1.0),
                "ot_cost_rel": float((ph * Ch).sum() / (ps * Cs).sum() - 1.0)})
    return out


def eval_set(case: Case, cfg: Run, seed: int) -> dict:
    kw = cfg.torch_kw
    gen = torch.Generator().manual_seed(7919 + seed)
    x0, x1 = case.sample(N_EVAL, 0, gen), case.sample(N_EVAL, 1, gen)
    L = case.length(x0, x1)
    E = L ** 2
    path, _ = case.exact(x0, x1, torch.linspace(0.0, 1.0, T_FINE, **kw), L)
    dense, _ = case.exact(x0, x1, torch.linspace(0.0, 1.0, 1025, **kw), L)
    a, b = case.sample(OT_PTS, 0, gen), case.sample(OT_PTS, 1, gen)
    A = a[:, None].expand(OT_PTS, OT_PTS, case.d).reshape(-1, case.d)
    B = b[None].expand(OT_PTS, OT_PTS, case.d).reshape(-1, case.d)
    C_ot = (case.length(A, B) ** 2).reshape(OT_PTS, OT_PTS).double().cpu().numpy()
    t = midpoints(NODES[-1], cfg)
    chord = torch.stack([case.metric.F2((1.0 - tk) * x0 + tk * x1, x1 - x0) for tk in t]).mean(0)
    checks = {"exact_energy": float(((exact_cost(case, x0, x1, L, NODES[-1], cfg) - E) / E)
                                    .abs().max())}
    if case.case != "randers":
        xv = torch.randn(512, case.d, generator=gen).to(**kw)
        xv = xv / xv.norm(dim=1, keepdim=True) * (0.5 + torch.rand(512, 1, generator=gen).to(**kw))
        vv = torch.randn(512, case.d, generator=gen).to(**kw)
        z = case.zermelo(xv, vv)
        checks["zermelo"] = float(((case.metric.F(xv, vv) - z).abs() / z).max())
    return {"pairs": (x0, x1), "E": E, "path": path, "dense": dense, "ot": (a, b), "C_ot": C_ot,
            "probe": (x0[:N_PROBE], x1[:N_PROBE], E[:N_PROBE]), "checks": checks,
            "chord_excess": float(((chord - E) / E).median())}


def train(case: Case, seed: int, cfg: Run, ev: dict):
    gen = torch.Generator().manual_seed(seed)
    X0, X1 = case.sample(N_TRAIN, 0, gen), case.sample(N_TRAIN, 1, gen)
    m = case.metric
    torch.manual_seed(seed)
    phi = GeoPathNet(case.d, **cfg.net_kw).to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(phi.parameters(), lr=cfg.geo_lr)
    coupler = Coupler(X0, X1, seed=seed)
    p0, p1, pE = ev["probe"]
    iters, budget = (200, 100) if cfg.smoke else (ITERS, BUDGET)
    curve, evals, t0 = [], {}, time.perf_counter()

    def probe(it, loss):
        rel = ((pair_cost(phi, m, p0, p1, 32, cfg) - pE) / pE).abs()
        curve.append([it, loss, float(rel.median())])

    probe(0, float("nan"))
    for it in range(1, iters + 1):
        x0, x1 = coupler.sample(cfg.batch)
        t = torch.rand(cfg.batch, 1, **cfg.torch_kw)
        x_t, x_dot = interpolant(phi, t, x0, x1, create_graph=True)
        loss = m.F2(x_t, x_dot).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(phi.parameters(), cfg.grad_clip)
        opt.step()
        if it in PROBES or it == iters:
            probe(it, float(loss))
        if it in (budget, iters):
            evals[str(it)] = {**evaluate(case, phi, ev, cfg), "seconds": time.perf_counter() - t0}
    return curve, evals


def tag(case: str, d: int, seed: int, kappa: float = KAPPA, c_eff: float = C_EFF) -> str:
    extra = f"_k{kappa:g}_c{c_eff:g}" if case == "randers" else ""
    return f"{case}{extra}_d{d}_s{seed}"


def tasks() -> list:
    out = [(c, d, s, KAPPA, C_EFF) for c in CASES for d in DIMS for s in SEEDS]
    return out + [("randers", GRID_DIM, s, k, c) for k, c in GRID for s in SEEDS]


def root_dir(smoke: bool = False) -> str:
    return experiment_root(NAME, smoke)


def run_task(i: int, cfg: Run, force: bool = False) -> None:
    case_name, d, seed, kappa, c_eff = tasks()[i]
    path = os.path.join(root_dir(cfg.smoke), "records",
                        tag(case_name, d, seed, kappa, c_eff) + ".json")
    if os.path.exists(path) and not force:
        return
    case = Case(case_name, d, cfg, kappa, c_eff, seed)
    ev = eval_set(case, cfg, seed)
    curve, evals = train(case, seed, cfg, ev)
    rec = {"case": case_name, "dim": d, "seed": seed, "kappa": kappa, "c_eff": c_eff,
           "radial": RADIAL, "omega": case.omega, "iters": ITERS, "budget_iters": BUDGET,
           "geo_lr": cfg.geo_lr, "batch": cfg.batch, "width": cfg.net_width,
           "depth": cfg.net_depth, "ot_k": cfg.ot_k, "chord_excess": ev["chord_excess"],
           "checks": ev["checks"], "eval": evals, "curve": curve,
           "gpu": torch.cuda.get_device_name(0) if cfg.device.type == "cuda" else "cpu"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh)
    os.replace(path + ".tmp", path)
    e = evals[min(evals, key=int)]
    print(f"[geodesic] {tag(case_name, d, seed, kappa, c_eff)} rel8 {e['rel_K8']:.2e} "
          f"rel256 {e['rel_K256']:.2e} dev {e['path_dev']:.2e} {e['seconds']:.0f}s", flush=True)


def records(smoke: bool = False) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(root_dir(smoke), "records", "*.json"))):
        with open(p) as fh:
            out.append(json.load(fh))
    return out


COLS = ("rel_K8", "rel_K32", "rel_K256", "rel_K8_mean", "below_exact", "path_dev", "curve_dev",
        "speed_cv", "plan_tv", "plan_subopt", "seconds")


def frames(smoke: bool = False) -> dict:
    recs = records(smoke)
    if not recs:
        return {}
    rows, curves = [], []
    for r in recs:
        key = {k: r[k] for k in ("case", "dim", "seed", "kappa", "c_eff")}
        for it, e in r["eval"].items():
            rows.append({**key, "iters": int(it), "chord_excess": r["chord_excess"],
                         **{k: e[k] for k in COLS},
                         **{f"check_{k}": v for k, v in r["checks"].items()}})
        curves += [{**key, "iter": it, "loss": loss, "rel": rel} for it, loss, rel in r["curve"]]
    df, cv = pd.DataFrame(rows), pd.DataFrame(curves)
    main_rows = lambda f: f[((f.kappa == KAPPA) & (f.c_eff == C_EFF)) | (f.case != "randers")]
    m, c = main_rows(df), main_rows(cv)
    keys = ["case", "dim", "iters"]
    checks = [k for k in df.columns if k.startswith("check_")]
    summary = m.groupby(keys)[list(COLS) + ["chord_excess"]].mean().join(
        m.groupby(keys)[["rel_K8", "path_dev"]].std(ddof=0).add_suffix(" sd")).join(
        m.groupby(keys)[checks].max())
    grid = df[(df.case == "randers") & (df.dim == GRID_DIM)].groupby(
        ["kappa", "c_eff", "iters"])[list(COLS)].mean()
    med = c.groupby(["case", "dim", "iter"])["rel"].median()
    curve = med.unstack("iter")
    reach = med.groupby(["case", "dim"]).apply(
        lambda s: pd.Series({f"iters to {p:g}": s.index.get_level_values("iter")[
            np.argmax(s.values < p)] if (s.values < p).any() else np.nan
            for p in (1e-1, 1e-2, 1e-3)}))
    return {"summary": summary, "grid": grid, "curve": curve, "reach": reach}


CASE_LABELS = {"randers": "constant Randers", "sphere": "sphere", "wind": "sphere + wind"}
CASE_COLORS = {"randers": "#2a78d6", "sphere": "#eb6834", "wind": "#1baf7a"}


def figure(out: str, smoke: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = frames(smoke)
    s, cv = f["summary"], f["curve"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for case in CASES:
        for it, ls in ((BUDGET, "-"), (ITERS, "--")):
            g = s.xs((case, it), level=("case", "iters"))
            kw = dict(color=CASE_COLORS[case], ls=ls, lw=2, marker="o", ms=4)
            ax[0].plot(g.index, g["rel_K8"], label=f"{CASE_LABELS[case]}, {it} it.", **kw)
            ax[1].plot(g.index, g["path_dev"], **kw)
        for d, ls in ((2, ":"), (20, "-"), (100, "--")):
            c = cv.loc[(case, d)]
            c = c[c.index > 0]
            ax[2].plot(c.index, c.values, color=CASE_COLORS[case], ls=ls, lw=1.5)
    for a_, t, y in zip(ax, ("relative cost error (8 nodes)", "path deviation / chord",
                             "error vs iteration\nd = 2 dotted, 20 solid, 100 dashed"),
                        ("|c - c*| / c*", "max_t |x - x*| / |x1 - x0|", "|c - c*| / c*")):
        a_.set_yscale("log")
        a_.set_title(t, fontsize=10)
        a_.set_ylabel(y, fontsize=9)
        a_.grid(True, which="major", color="#e6e6e3", lw=0.6)
        a_.spines[["top", "right"]].set_visible(False)
    for a_ in ax[:2]:
        a_.set_xscale("log")
        a_.set_xticks(DIMS)
        a_.set_xticklabels([str(d) for d in DIMS])
        a_.set_xlabel("dimension d", fontsize=9)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("iteration", fontsize=9)
    ax[2].axvline(BUDGET, color="#8a8984", lw=0.8)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None)
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="geodesic.validate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", type=int, required=True)
    r.add_argument("--force", action="store_true")
    sub.add_parser("count")
    sub.add_parser("report")
    sub.add_parser("figure")
    a = ap.parse_args(argv)
    if a.cmd == "figure":
        from ffm.core.paths import PUBLIC_ROOT
        figure(os.path.join(PUBLIC_ROOT, "notebooks", "out", "geodesic_validation"))
    elif a.cmd == "count":
        print(len(tasks()))
    elif a.cmd == "run":
        run_task(a.task, Run.from_env(), a.force)
    else:
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            for k, v in frames(Run.from_env().smoke).items():
                print(k, v.round(5).to_string(), sep="\n")


if __name__ == "__main__":
    main()
