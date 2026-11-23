from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ffm.core.models import _MLP, timestep_embedding
from ffm.core.paths import experiment_root
from ffm.method import GeoPathNet, VelocityNet, integrate
from ffm.method.config import Run
from ffm.method.nets import Coupler

from . import problem as prob
from .bench import load_tuned
from .tune import warm_up
from .variants import MatchedFactory, matched_path_a

KEYS = ("sheet", "pancreas20")
VARIANTS = {"sheet_lam1": ("sheet", {"lam_mult": 1.0}, ("ffm", "wlf_finsler"))}
METHODS = ("ffm", "wlf_finsler", "wlf_euclid", "gsbm_finsler", "gsbm_euclid")
SEEDS = (0, 1, 2, 3, 4)
HERE = os.path.dirname(os.path.abspath(__file__))
WIDTH_FILES = {"sheet": (os.path.join(HERE, os.pardir, "sheet", "tuned.json"), None),
               "pancreas20": (os.path.join(HERE, os.pardir, "pancreas", "tuned.json"), "20")}
WLF_ITERS = 4000
GSBM_ROUNDS = (("fwd", "indep"), ("bwd", "fwd"), ("fwd", "bwd"))
GSBM_PAIRS = 1024
GSBM_CHUNK = 256
GSBM_SOC_ITERS = 200
GSBM_KNOTS = 8
GSBM_STD_KNOTS = 4
GSBM_GRID = 16
GSBM_MC = 4
GSBM_EPS = 1e-3
GSBM_LR_MEAN = 0.1
GSBM_LR_STD = 0.05
DIR_TIMES = (0.25, 0.5, 0.75)
LABELS = {"ffm": "FFM (geodesics + Finsler Sinkhorn)", "wlf_finsler": "WLF, Finsler H",
          "wlf_euclid": "WLF, Euclidean H", "gsbm_finsler": "GSBM, Finsler L",
          "gsbm_euclid": "GSBM, Euclidean L"}


def root_dir(smoke: bool = False) -> str:
    return os.path.join(experiment_root("matched_bench", smoke), "hamiltonian")


def rec_path(root: str, key: str, method: str, seed: int) -> str:
    return os.path.join(root, f"{key}_{method}_s{seed}.json")


def width_of(key: str) -> float:
    path, sub = WIDTH_FILES[key]
    with open(path) as fh:
        t = json.load(fh)
    return float((t[sub] if sub else t)["pathb"]["width_mult"])


class Finsler:
    def __init__(self, cost):
        self.c = cost

    def fields(self, x):
        k = self.c._kernel(x)
        C = self.c._smooth(k, self.c.D_pts) + self.c.rho * self.c.I[None]
        return C, self.c._smooth(k, self.c.b0_pts) + self.c.eps_b

    def L(self, x, v):
        return 0.5 * self.c.F2(x, v)

    def H(self, x, p):
        C, b = self.fields(x)
        l2 = self.c.lam ** 2
        pb = (p * b).sum(1)
        q = torch.einsum("bi,bij,bj->b", p, C, p) / l2 + pb ** 2 / l2 ** 2
        return 0.5 * ((torch.sqrt(q.clamp(min=1e-12)) + pb / l2) / self.c.scale) ** 2


class Euclid(Finsler):
    def L(self, x, v):
        return 0.5 * (v ** 2).sum(1)

    def H(self, x, p):
        return 0.5 * (p ** 2).sum(1)


def dual_check(lag, X, n: int = 64, seed: int = 0) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = X[torch.randperm(len(X), generator=g)[:n].to(X.device)]
    p = torch.randn(n, X.shape[1], generator=g).to(X.device, X.dtype).requires_grad_(True)
    h = lag.H(x, p)
    v = torch.autograd.grad(h.sum(), p)[0].detach().requires_grad_(True)
    L = lag.L(x, v)
    gv = torch.autograd.grad(L.sum(), v)[0]
    with torch.no_grad():
        fy = ((p * v).sum(1) - L - h).abs() / h.abs().clamp(min=1e-12)
        st = (gv - p).norm(dim=1) / p.norm(dim=1)
    return {"fenchel_young": float(fy.max()), "stationarity": float(st.max())}


class Potential(nn.Module):
    def __init__(self, d: int, t_emb_dim: int = 32, width: int = 64, depth: int = 4):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.mlp = _MLP(t_emb_dim + d, 1, width=width, depth=depth)

    def forward(self, t, x):
        return self.mlp(torch.cat([timestep_embedding(t, self.t_emb_dim), x], dim=1))[:, 0]


def _wlf_objective(lag, s_net, phi, x0, x1, t):
    xt = (1.0 - t) * x0 + t * x1 + t * (1.0 - t) * phi(t, x0, x1)
    tt = t.detach().requires_grad_(True)
    gt, gx = torch.autograd.grad(s_net(tt, xt).sum(), (tt, xt), create_graph=True)
    return (s_net(torch.ones_like(t), x1).mean() - s_net(torch.zeros_like(t), x0).mean()
            - (gt[:, 0] + lag.H(xt, gx)).mean())


def train_wlf(lag, X0, X1, seed: int, cfg: Run):
    iters = 200 if cfg.smoke else WLF_ITERS
    torch.manual_seed(seed)
    d = X0.shape[1]
    s_net = Potential(d, **cfg.net_kw).to(cfg.device, cfg.dtype)
    phi = GeoPathNet(d, **cfg.net_kw).to(cfg.device, cfg.dtype)
    opts = ((torch.optim.Adam(s_net.parameters(), lr=cfg.cfm_lr), -1.0, s_net),
            (torch.optim.Adam(phi.parameters(), lr=cfg.geo_lr), 1.0, phi))
    coupler, hist = Coupler(X0, X1, seed=seed), []
    for it in range(iters):
        for opt, sign, net in opts:
            x0, x1 = coupler.sample(cfg.batch)
            t = torch.rand(cfg.batch, 1, **cfg.torch_kw)
            J = _wlf_objective(lag, s_net, phi, x0, x1, t)
            opt.zero_grad(set_to_none=True)
            (sign * J).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
        if it % 100 == 0 or it == iters - 1:
            hist.append(float(J))
    return s_net, hist


def wlf_velocity(lag, s_net):
    def v(t, x):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            p = torch.autograd.grad(s_net(t, x).sum(), x)[0].detach().requires_grad_(True)
            return torch.autograd.grad(lag.H(x.detach(), p).sum(), p)[0].detach()
    return v


def _basis(t, K: int):
    k = torch.arange(1, K + 1, device=t.device, dtype=t.dtype) * math.pi
    return torch.sin(t[..., None] * k), k * torch.cos(t[..., None] * k)


def _gauss(x0, x1, c, e, t, sigma: float):
    B, Bd = _basis(t, c.shape[1])
    S, Sd = _basis(t, e.shape[1])
    m = (1.0 - t)[..., None] * x0[:, None] + t[..., None] * x1[:, None] \
        + torch.einsum("pgk,pkd->pgd", B, c)
    md = (x1 - x0)[:, None] + torch.einsum("pgk,pkd->pgd", Bd, c)
    s = math.log(math.e - 1.0) + torch.einsum("pgk,pk->pg", S, e)
    sd = torch.einsum("pgk,pk->pg", Sd, e)
    r = torch.sqrt(t * (1.0 - t))
    g = sigma * r * torch.nn.functional.softplus(s)
    gd = sigma * ((1.0 - 2.0 * t) / (2.0 * r) * torch.nn.functional.softplus(s)
                  + r * torch.sigmoid(s) * sd)
    return m, md, g, gd


def _drifts(m, md, g, gd, x, sigma: float):
    af = (gd - sigma ** 2 / (2.0 * g)) / g
    ab = (-gd - sigma ** 2 / (2.0 * g)) / g
    return md + af[..., None] * (x - m), -md + ab[..., None] * (x - m)


def _grid(n: int, like):
    return torch.linspace(GSBM_EPS, 1.0 - GSBM_EPS, n, device=like.device, dtype=like.dtype)


def cond_soc(lag, x0, x1, init_c, sigma: float, direction: str, scale: float, iters: int):
    c = init_c.clone().requires_grad_(True)
    e = torch.zeros(len(x0), GSBM_STD_KNOTS, device=x0.device, dtype=x0.dtype,
                    requires_grad=True)
    opt = torch.optim.Adam([{"params": [c], "lr": GSBM_LR_MEAN * scale},
                            {"params": [e], "lr": GSBM_LR_STD}])
    t = _grid(GSBM_GRID, x0)[None].expand(len(x0), GSBM_GRID)
    hist = []
    for it in range(iters):
        m, md, g, gd = _gauss(x0, x1, c, e, t, sigma)
        x = m[:, :, None] + g[:, :, None, None] * torch.randn(
            *m.shape[:2], GSBM_MC, m.shape[2], device=m.device, dtype=m.dtype)
        uf, ub = _drifts(m[:, :, None], md[:, :, None], g[:, :, None], gd[:, :, None], x, sigma)
        v = uf if direction == "fwd" else -ub
        loss = lag.L(x.reshape(-1, x.shape[-1]), v.reshape(-1, v.shape[-1])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append(float(loss))
    return c.detach(), e.detach(), hist


def _project(x0, x1, traj, K: int):
    T = traj.shape[0] - 1
    t = torch.linspace(0.0, 1.0, T + 1, device=x0.device, dtype=x0.dtype)[1:-1]
    base = (1.0 - t)[:, None, None] * x0[None] + t[:, None, None] * x1[None]
    dev = (traj[1:-1] - base).permute(1, 0, 2)
    Phi = _basis(t[None], K)[0][0]
    return torch.einsum("kg,pgd->pkd", torch.linalg.pinv(Phi), dev)


def simulate(net, x, sigma: float, cfg: Run, direction: str, seed: int):
    g = torch.Generator(device=x.device).manual_seed(seed)
    n, dt = cfg.n_steps, 1.0 / cfg.n_steps
    traj = [x.clone()]
    with torch.no_grad():
        for k in range(n):
            tk = k * dt if direction == "fwd" else 1.0 - k * dt
            t = torch.full((len(x), 1), tk, device=x.device, dtype=x.dtype)
            noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
            x = x + net(t, x) * dt + sigma * math.sqrt(dt) * noise
            traj.append(x.clone())
    return torch.stack(traj)


def _match(net, opt, x0, x1, c, e, sigma: float, direction: str, cfg: Run):
    hist = []
    for it in range(cfg.cfm_iters):
        i = torch.randint(len(x0), (cfg.batch,), device=x0.device)
        t = (GSBM_EPS + (1.0 - 2 * GSBM_EPS) * torch.rand(cfg.batch, 1, **cfg.torch_kw))
        m, md, g, gd = _gauss(x0[i], x1[i], c[i], e[i], t, sigma)
        x = m[:, 0] + g[:, 0, None] * torch.randn_like(m[:, 0])
        uf, ub = _drifts(m[:, 0], md[:, 0], g[:, 0], gd[:, 0], x, sigma)
        loss = ((net(t, x) - (uf if direction == "fwd" else ub)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        if it % 250 == 0 or it == cfg.cfm_iters - 1:
            hist.append(float(loss))
    return hist


def train_gsbm(lag, X0, X1, sigma: float, seed: int, cfg: Run, scale: float):
    n_pairs, soc_iters = (256, 20) if cfg.smoke else (GSBM_PAIRS, GSBM_SOC_ITERS)
    torch.manual_seed(seed)
    d = X0.shape[1]
    nets = {k: VelocityNet(d, **cfg.net_kw).to(cfg.device, cfg.dtype) for k in ("fwd", "bwd")}
    opts = {k: torch.optim.Adam(n.parameters(), lr=cfg.cfm_lr) for k, n in nets.items()}
    hist = []
    for r, (direction, coupling) in enumerate(GSBM_ROUNDS):
        i0 = torch.randint(len(X0), (n_pairs,), device=X0.device)
        i1 = torch.randint(len(X1), (n_pairs,), device=X1.device)
        x0, x1 = X0[i0], X1[i1]
        init = torch.zeros(n_pairs, GSBM_KNOTS, d, device=X0.device, dtype=X0.dtype)
        if coupling == "fwd":
            tr = simulate(nets["fwd"], x0, sigma, cfg, "fwd", seed + 1000 * r)
            x1 = tr[-1]
            init = _project(x0, x1, tr, GSBM_KNOTS)
        elif coupling == "bwd":
            tr = simulate(nets["bwd"], x1, sigma, cfg, "bwd", seed + 1000 * r)
            x0 = tr[-1]
            init = _project(x0, x1, tr.flip(0), GSBM_KNOTS)
        cs, es, soc = [], [], []
        for a in range(0, n_pairs, GSBM_CHUNK):
            c, e, h = cond_soc(lag, x0[a:a + GSBM_CHUNK], x1[a:a + GSBM_CHUNK],
                               init[a:a + GSBM_CHUNK], sigma, direction, scale, soc_iters)
            cs.append(c)
            es.append(e)
            soc.append(h)
        c, e = torch.cat(cs), torch.cat(es)
        mh = _match(nets[direction], opts[direction], x0, x1, c, e, sigma, direction, cfg)
        hist.append({"round": r, "direction": direction, "coupling": coupling,
                     "soc": np.mean(soc, axis=0).tolist(), "match": mh})
    return nets["fwd"], hist


def _peak_reset(cfg: Run) -> None:
    if cfg.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(cfg.device)


def _peak(cfg: Run) -> float | None:
    return (torch.cuda.max_memory_allocated(cfg.device) / 2 ** 20
            if cfg.device.type == "cuda" else None)


def path_metrics(pb, traj, fin: Finsler, cfg: Run, seed: int, windows: int = 10) -> dict:
    out = dict(pb.test(traj, seed=seed))
    T = torch.as_tensor(traj, **cfg.torch_kw)
    n = T.shape[0] - 1
    w = n // windows
    act, cos = [], []
    with torch.no_grad():
        for a in range(0, n - w + 1, w):
            x, dx = T[a + w // 2], T[a + w] - T[a]
            act.append(float(fin.c.F2(x, dx * n / w).mean()))
            cos.append(float(torch.nn.functional.cosine_similarity(dx, fin.fields(x)[1],
                                                                   dim=1).mean()))
    out.update({"action": float(np.mean(act)), "drift_cos": float(np.mean(cos))})
    return out


def run(key: str, method: str, seed: int, cfg: Run, root: str, force: bool = False) -> dict:
    path = rec_path(root, key, method, seed)
    if os.path.exists(path) and not force:
        print(f"[ham] skip {key} {method} s{seed}")
        with open(path) as fh:
            return json.load(fh)
    base, over = (VARIANTS[key][0], VARIANTS[key][1]) if key in VARIANTS else (key, {})
    pb = prob.build(base, "bench", cfg)
    factory = MatchedFactory(pb.X, pb.D_pts, pb.b0_pts, cfg, pb.X0_t, pb.X1_t)
    point = {**load_tuned()[base]["specific"]["ffm_full"], **over}
    cost = factory.at("ffm_full", point)
    warm_up(factory, pb.X0_t, pb.X1_t, cfg)
    fin = Finsler(cost)
    lag = Euclid(cost) if method.endswith("euclid") else fin
    sigma = 2.0 * width_of(base) * float(cost.eps_kernel)
    extra = {}
    _peak_reset(cfg)
    t0 = time.perf_counter()
    if method == "ffm":
        out = matched_path_a(cost, pb.X0_t, pb.X1_t, seed, cfg)["specific"]
        traj = out["traj"]
        extra = {"t_ot": out["t_ot"], "t_phase3": out["t_phase3"]}
    elif method.startswith("wlf"):
        s_net, hist = train_wlf(lag, pb.X0_t, pb.X1_t, seed, cfg)
        train_s = time.perf_counter() - t0
        vel = wlf_velocity(lag, s_net)
        traj = integrate(vel, pb.X0_t, cfg).cpu().numpy().astype(np.float64)
        extra = {"objective": hist, "train_s": train_s}
    else:
        net, hist = train_gsbm(lag, pb.X0_t, pb.X1_t, sigma, seed, cfg, float(cost.eps_kernel))
        train_s = time.perf_counter() - t0
        traj = simulate(net, pb.X0_t, sigma, cfg, "fwd", seed).cpu().numpy().astype(np.float64)
        extra = {"rounds": hist, "train_s": train_s, "sigma": sigma}
    seconds = out["total_s"] if method == "ffm" else time.perf_counter() - t0
    rec = {"key": key, "method": method, "seed": seed, "point": point,
           "rho": float(cost.rho), "lam": float(cost.lam), "scale": float(cost.scale),
           "seconds": seconds, "peak_mb": _peak(cfg), "device": str(cfg.device),
           **path_metrics(pb, traj, fin, cfg, seed), **extra}
    if method == "wlf_finsler" and seed == SEEDS[0]:
        rec["dual_check"] = dual_check(fin, pb.X0_t)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(path + ".tmp", path)
    print(f"[ham] {key} {method:13s} s{seed}  mid W2 {rec['W2_mid']:.4f}  end W2 "
          f"{rec['W2_end']:.4f}  action {rec['action']:.4f}  cos(dx,m) {rec['drift_cos']:.3f}  "
          f"({seconds:.0f}s) {rec.get('dual_check', '')}", flush=True)
    return rec


def tasks() -> list[tuple[str, str, int]]:
    return ([(k, m, s) for k in KEYS for m in METHODS for s in SEEDS]
            + [(k, m, s) for k, (_, _, ms) in VARIANTS.items() for m in ms for s in SEEDS])


def frames(root: str | None = None, smoke: bool = False) -> dict:
    root = root_dir(smoke) if root is None else root
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "*.json")))]
    df = pd.DataFrame([{"key": r["key"], "method": LABELS[r["method"]], "seed": r["seed"],
                        **{k: r.get(k) for k in ("W2_mid", "W2_end", "action", "drift_cos",
                                                 "seconds", "peak_mb")}} for r in rows])
    if not len(df):
        return {"summary": df, "dual": pd.DataFrame()}
    pm = lambda v: (f"{np.mean(v):.4f} ± {np.std(v):.4f}" if len(v) > 1 else f"{np.mean(v):.4f}")
    cols = ["W2_mid", "W2_end", "action", "drift_cos", "seconds", "peak_mb"]
    summary = df.groupby(["key", "method"])[cols].agg(
        lambda v: pm(pd.to_numeric(v, errors="coerce").dropna().to_numpy()))
    summary["n"] = df.groupby(["key", "method"]).size()
    dual = pd.DataFrame([{"key": r["key"], **r["dual_check"]} for r in rows if "dual_check" in r])
    return {"summary": summary, "dual": dual}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="hamiltonian")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--force", action="store_true")
    sub.add_parser("count")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    cfg = Run.from_env()
    if a.cmd == "count":
        print(len(tasks()))
    elif a.cmd == "run":
        key, method, seed = tasks()[a.index]
        run(key, method, seed, cfg, root_dir(cfg.smoke), a.force)
    else:
        f = frames(smoke=cfg.smoke)
        print(f["summary"].to_string())
        print(f["dual"].to_string())


if __name__ == "__main__":
    main()
