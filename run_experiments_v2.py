#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments_v2.py  --  complete, resumable experiment suite (Google Colab ready)
for the paper "Hybrid Grey Wolf and Genetic Algorithm for Energy-Efficient Resource
Allocation in NOMA Fog Computing-Based IoT Systems".

The core model, optimisers, seeds and parameters are IDENTICAL to run_experiments.py
(v1), so E1-E8 reproduce the numbers you already have (deterministic for equal seeds).
v2 adds what the reviewers asked for and what v1 could not provide:
  * E1 now stores convergence curves; E5 stores population diversity (exported as numbers)
  * E9  budget sensitivity (2k-50k evaluations): does GWOGA help at small/large budgets?
  * E10 long-run reference (10x budget): how far from "converged" is the 10^4 budget?
  * E11 several fog nodes (M = 1, 2, 4)
  * E12 larger scale (N = 60, 100 with proportionally scaled blocks/nodes)
  * E13 clean sequential timing + hardware / library versions
  * energy breakdown (transmission vs local computation), Holm-corrected tests
  * a redrawn system-model figure (image1.png)

HOW TO RUN IN GOOGLE COLAB (one cell each)
    !pip -q install numpy scipy matplotlib
    # upload run_experiments_v2.py, then:
    !python run_experiments_v2.py --only new --workers 2
    # (needs ~1-2 h on a free Colab; the run is RESUMABLE: if the runtime disconnects,
    #  start the same command again and finished experiments are skipped)
    # download colab_results_bundle.zip from the file browser and send it back.
Use --only all to recompute E1-E8 as well (about double the time), --quick for a smoke test.

OUTPUT (in the working directory)
    results/summary.txt, results/output_v2.csv, results/curves_E1.csv,
    results/stats_tests.csv, results/timing.json, results/raw_*.json
    images/media/image1.png, image2..image10.png (as before) + image11..image14.png
    colab_results_bundle.zip   <- the only file you need to send back
Figure map: image1 system model | image2 convergence | image3 energy vs tasks |
image4 energy vs fog capacity | image5 delay vs tasks | image6 population + run time |
image7 ablation + diversity | image8 NOMA vs OMA | image9 weight trade-off |
image10 p_c/p_m/mu sensitivity | image11 budget sensitivity | image12 long-run gap |
image13 several fog nodes | image14 larger scale (cost and run time).
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon, t as student_t
except Exception:  # scipy is optional
    wilcoxon = None
    student_t = None

# ----------------------------------------------------------------------------
# Default configuration (every value appears in Table I of the paper)
# ----------------------------------------------------------------------------
DEFAULT = dict(
    N=30,            # IoT devices (= tasks)
    M=2,             # fog nodes
    F=20e9,          # computing capacity of each fog node [cycles/s]
    Rb=8,            # orthogonal resource blocks (sub-bands)
    Umax=3,          # max NOMA users per block (1 => OMA)
    B=10e6,          # system bandwidth [Hz]
    Pmax=0.3,        # max transmit power [W]
    Lmax=1.0,        # delay constraint [s]
    kappa=1e-28,     # switched capacitance coefficient
    rho_lo=300, rho_hi=700,      # CPU cycles per bit
    S_lo=0.1e6, S_hi=1e6,        # task size [bit]
    fl_lo=0.7e9, fl_hi=1e9,      # device CPU [cycles/s]
    N0_dBm=-174,     # noise PSD [dBm/Hz]
    g0_dB=-30,       # path loss at 1 m [dB]
    alpha=3.5,       # path-loss exponent
    d_lo=50, d_hi=300,           # device-fog distance [m]
    gamma_th=1.0,    # minimum SINR for successful SIC decoding (0 dB)
    wE=0.5, wL=0.5,  # objective weights
    pen=10.0,        # penalty coefficient for constraint violations
    K=100,           # population size (all population-based algorithms)
    budget=10000,    # number of TRUE fitness evaluations per run
    pc=0.8, pm=0.1, mu=0.02, sigma_m=0.1,   # GA parameters
    rho_c=0.5,       # GWOGA offspring ratio (children per iteration / K)
    tournament=3,
)

MAIN_ALGOS = ["RS", "GA", "GWO", "PSO", "DE", "WOA", "GWOGA"]
REF_ALGOS = ["EQ", "LOC"]
LABEL = {"RS": "Random search", "GA": "GA", "GWO": "GWO", "PSO": "PSO",
         "DE": "DE", "WOA": "WOA", "GWOGA": "GWOGA (proposed)",
         "EQ": "Equal allocation", "LOC": "All local",
         "GWO+X": "GWO+crossover", "GWO+M": "GWO+mutation"}
STYLE = {"RS": ("x", ":", "0.55"), "GA": ("s", "--", "tab:green"),
         "GWO": ("o", "--", "tab:red"), "PSO": ("^", "-.", "tab:purple"),
         "DE": ("v", "-.", "tab:orange"), "WOA": ("D", ":", "tab:brown"),
         "GWOGA": ("*", "-", "tab:blue"), "EQ": ("+", ":", "0.3"),
         "LOC": ("", "-", "0.0"), "GWO+X": ("P", "--", "tab:cyan"),
         "GWO+M": ("h", "--", "tab:olive")}


# ----------------------------------------------------------------------------
# Problem instance and vectorised fitness (batch of candidate solutions)
# ----------------------------------------------------------------------------
class Problem:
    """One random scenario. A candidate is a vector in [0,1]^(4N):
       genes [0:N]   offloading target  (0 -> local, 1..M -> fog node)
       genes [N:2N]  resource block
       genes [2N:3N] transmit power  p_i = Pmax*(0.01+0.99*x)
       genes [3N:4N] computing share f_i = F*(0.01+0.99*x)
    Repair: (i) in every block only the Umax strongest-channel offloaders keep
    offloading, the others execute locally; (ii) if the computing shares of a
    node exceed F they are scaled down proportionally."""

    def __init__(self, cfg, seed):
        r = np.random.default_rng(10_000 + seed)
        self.cfg = cfg
        N = cfg["N"]
        self.N, self.D = N, 4 * N
        self.S = r.uniform(cfg["S_lo"], cfg["S_hi"], N)
        self.c = self.S * r.uniform(cfg["rho_lo"], cfg["rho_hi"], N)
        self.floc = r.uniform(cfg["fl_lo"], cfg["fl_hi"], N)
        d = r.uniform(cfg["d_lo"], cfg["d_hi"], N)
        fade = r.exponential(1.0, N)                 # |xi|^2, Rayleigh fading
        self.g = 10 ** (cfg["g0_dB"] / 10) * d ** (-cfg["alpha"]) * fade
        self.Bblk = cfg["B"] / cfg["Rb"]
        self.sig2 = 10 ** ((cfg["N0_dBm"] - 30) / 10) * self.Bblk
        self.Lloc = self.c / self.floc
        self.Eloc = cfg["kappa"] * self.c * self.floc ** 2

    def _eval(self, X):
        cfg, N, M = self.cfg, self.N, self.cfg["M"]
        Rb, Umax = cfg["Rb"], cfg["Umax"]
        node = np.minimum((X[:, :N] * (M + 1)).astype(int), M)
        blk = np.minimum((X[:, N:2 * N] * Rb).astype(int), Rb - 1)
        p = cfg["Pmax"] * (0.01 + 0.99 * X[:, 2 * N:3 * N])
        f = cfg["F"] * (0.01 + 0.99 * X[:, 3 * N:])
        off = node > 0
        same = blk[:, :, None] == blk[:, None, :]
        pair = off[:, :, None] & off[:, None, :] & same
        stronger = self.g[None, None, :] > self.g[None, :, None]   # g_j > g_i
        rank = (pair & stronger).sum(axis=2)
        off = off & (rank < Umax)                                   # repair (i)
        pair = off[:, :, None] & off[:, None, :] & same
        rp = self.g[None, :] * p                                    # received power
        later = rp[:, None, :] < rp[:, :, None]                     # decoded after i
        I = ((pair & later) * rp[:, None, :]).sum(axis=2)
        gam = rp / (I + self.sig2)
        rate = self.Bblk * np.log2(1.0 + gam)
        for m in range(1, M + 1):                                   # repair (ii)
            msk = off & (node == m)
            tot = (f * msk).sum(axis=1)
            sc = np.minimum(1.0, cfg["F"] / np.maximum(tot, 1e-9))
            f = np.where(msk, f * sc[:, None], f)
        L = np.where(off, self.S / rate + self.c / f, self.Lloc[None, :])
        E = np.where(off, p * self.S / rate, self.Eloc[None, :])
        viol = np.maximum(0.0, L / cfg["Lmax"] - 1.0) + \
            np.where(off, np.maximum(0.0, 1.0 - gam / cfg["gamma_th"]), 0.0)
        Jc = (cfg["wE"] * E / self.Eloc[None, :] +
              cfg["wL"] * L / self.Lloc[None, :]).mean(axis=1)
        return Jc, viol, E, L, off

    def cost(self, X):
        Jc, viol, _, _, _ = self._eval(X)
        return Jc + self.cfg["pen"] * viol.mean(axis=1)

    def detail(self, x):
        Jc, viol, E, L, off = self._eval(x[None, :])
        o = off[0]
        return dict(Jc=float(Jc[0]), E_mJ=float(E[0].mean() * 1e3),
                    E_tx_mJ=float(E[0][o].sum() / self.N * 1e3),
                    E_loc_mJ=float(E[0][~o].sum() / self.N * 1e3),
                    L_ms=float(L[0].mean() * 1e3), Lmax_ms=float(L[0].max() * 1e3),
                    feasible=bool(viol[0].max() <= 1e-12),
                    off=float(off[0].mean()))


class Evaluator:
    """Counts TRUE fitness evaluations and keeps the best-so-far history."""

    def __init__(self, prob, budget):
        self.p, self.budget, self.fe = prob, budget, 0
        self.best, self.bestx = np.inf, None
        self.hfe, self.hbest, self.div = [], [], []

    @property
    def done(self):
        return self.fe >= self.budget

    def __call__(self, X):
        X = np.atleast_2d(X)
        n = len(X)
        out = np.full(n, np.inf)
        rem = self.budget - self.fe
        if rem <= 0:
            return out
        m = min(n, rem)
        f = self.p.cost(X[:m])
        out[:m] = f
        run = np.minimum.accumulate(np.concatenate(([self.best], f)))[1:]
        self.hfe.extend(range(self.fe + 1, self.fe + m + 1))
        self.hbest.extend(run.tolist())
        j = int(np.argmin(f))
        if f[j] < self.best:
            self.best, self.bestx = float(f[j]), X[j].copy()
        self.fe += m
        return out


# ----------------------------------------------------------------------------
# Optimisers (all use the same K, the same budget and the same evaluator)
# ----------------------------------------------------------------------------
def tournament(f, n, k, rng):
    idx = rng.integers(len(f), size=(n, k))
    return idx[np.arange(n), np.argmin(f[idx], axis=1)]


def make_children(X, f, n, cfg, rng, use_x=True, use_m=True):
    """GA operators: tournament selection, arithmetic crossover (prob. pc),
    Gaussian mutation (individual prob. pm, per-gene rate mu)."""
    D = X.shape[1]
    p1 = X[tournament(f, n, cfg["tournament"], rng)]
    p2 = X[tournament(f, n, cfg["tournament"], rng)]
    child = p1.copy()
    if use_x:
        lam = rng.random((n, 1))
        cx = rng.random(n) < cfg["pc"]
        child = np.where(cx[:, None], lam * p1 + (1 - lam) * p2, child)
    if use_m:
        mi = rng.random(n) < cfg["pm"]
        mg = rng.random((n, D)) < cfg["mu"]
        child = child + (mi[:, None] & mg) * rng.normal(0, cfg["sigma_m"], (n, D))
    return np.clip(child, 0.0, 1.0)


def update_leaders(lead, X, f):
    Xs = np.vstack([lead[0], X])
    fs = np.concatenate([lead[1], f])
    idx = np.argsort(fs)[:3]
    return Xs[idx], fs[idx]


def gwo_move(X, lead, theta, rng):
    new = np.zeros_like(X)
    for l in range(3):
        A = 2 * theta * rng.random(X.shape) - theta
        C = 2 * rng.random(X.shape)
        new += lead[l] - A * np.abs(C * lead[l] - X)
    return np.clip(new / 3.0, 0.0, 1.0)


def algo_random(ev, rng, cfg):
    while not ev.done:
        ev(rng.random((cfg["K"], ev.p.D)))


def algo_ga(ev, rng, cfg):
    K, D = cfg["K"], ev.p.D
    X = rng.random((K, D)); f = ev(X)
    while not ev.done:
        el = np.argsort(f)[:2]
        ch = make_children(X, f, K - 2, cfg, rng)
        fc = ev(ch)
        X = np.vstack([X[el], ch]); f = np.concatenate([f[el], fc])
        ev.div.append((ev.fe, float(X.std(0).mean())))


def algo_gwo_family(ev, rng, cfg, ga=False, use_x=True, use_m=True):
    """ga=False: plain GWO.  ga=True: GWOGA (GWO move, then GA offspring that
    replace the worst individuals when better).  use_x/use_m switch the
    crossover / mutation operator on or off (ablation)."""
    K, D = cfg["K"], ev.p.D
    n_off = int(math.ceil(cfg["rho_c"] * K)) if ga else 0
    iters = max(1, (ev.budget - K) // (K + n_off))
    X = rng.random((K, D)); f = ev(X)
    lead = (np.zeros((3, D)), np.full(3, np.inf))
    lead = update_leaders(lead, X, f)
    t = 0
    while not ev.done:
        theta = max(0.0, 2.0 * (1.0 - t / iters)); t += 1
        X = gwo_move(X, lead[0], theta, rng)
        f = ev(X)
        lead = update_leaders(lead, X, f)
        if ga and not ev.done:
            ch = make_children(X, f, n_off, cfg, rng, use_x, use_m)
            fc = ev(ch)
            worst = np.argsort(-f)[:n_off]
            bestc = np.argsort(fc)[:len(worst)]
            better = fc[bestc] < f[worst]
            X[worst[better]] = ch[bestc[better]]
            f[worst[better]] = fc[bestc[better]]
            lead = update_leaders(lead, ch, fc)
        ev.div.append((ev.fe, float(X.std(0).mean())))


def algo_pso(ev, rng, cfg):
    K, D = cfg["K"], ev.p.D
    X = rng.random((K, D)); V = np.zeros((K, D)); f = ev(X)
    pb, pf = X.copy(), f.copy(); gb = pb[np.argmin(pf)]
    while not ev.done:
        V = 0.7 * V + 1.5 * rng.random((K, D)) * (pb - X) + \
            1.5 * rng.random((K, D)) * (gb - X)
        V = np.clip(V, -0.2, 0.2); X = np.clip(X + V, 0, 1)
        f = ev(X)
        imp = f < pf
        pb[imp], pf[imp] = X[imp], f[imp]
        gb = pb[np.argmin(pf)]
        ev.div.append((ev.fe, float(X.std(0).mean())))


def algo_de(ev, rng, cfg):
    K, D = cfg["K"], ev.p.D
    X = rng.random((K, D)); f = ev(X)
    while not ev.done:
        r = np.argsort(rng.random((K, K)), axis=1)[:, :3]
        V = np.clip(X[r[:, 0]] + 0.5 * (X[r[:, 1]] - X[r[:, 2]]), 0, 1)
        mask = rng.random((K, D)) < 0.9
        mask[np.arange(K), rng.integers(D, size=K)] = True
        U = np.where(mask, V, X)
        fu = ev(U)
        b = fu < f
        X[b], f[b] = U[b], fu[b]
        ev.div.append((ev.fe, float(X.std(0).mean())))


def algo_woa(ev, rng, cfg):
    K, D = cfg["K"], ev.p.D
    iters = max(1, (ev.budget - K) // K)
    X = rng.random((K, D)); f = ev(X)
    bx, bf = X[np.argmin(f)].copy(), f.min()
    t = 0
    while not ev.done:
        a = max(0.0, 2.0 * (1.0 - t / iters)); t += 1
        A = 2 * a * rng.random((K, 1)) - a
        C = 2 * rng.random((K, 1))
        p = rng.random((K, 1)); l = rng.uniform(-1, 1, (K, 1))
        rand = X[rng.integers(K, size=K)]
        ref = np.where(np.abs(A) < 1, bx[None, :], rand)
        enc = ref - A * np.abs(C * ref - X)
        spi = np.abs(bx[None, :] - X) * np.exp(l) * np.cos(2 * np.pi * l) + bx[None, :]
        X = np.clip(np.where(p < 0.5, enc, spi), 0, 1)
        f = ev(X)
        j = int(np.argmin(f))
        if f[j] < bf:
            bf, bx = f[j], X[j].copy()
        ev.div.append((ev.fe, float(X.std(0).mean())))


def reference_solution(ev, kind):
    """kind='LOC': everybody computes locally.  kind='EQ': everybody offloads,
    round-robin fog node and resource block, power Pmax/2, equal CPU shares."""
    cfg, N, M, Rb = ev.p.cfg, ev.p.N, ev.p.cfg["M"], ev.p.cfg["Rb"]
    x = np.zeros(4 * N)
    i = np.arange(N)
    if kind == "EQ":
        x[:N] = ((i % M) + 1 + 0.5) / (M + 1)
        x[N:2 * N] = ((i % Rb) + 0.5) / Rb
        x[2 * N:3 * N] = (0.5 - 0.01) / 0.99
        x[3 * N:] = min(1.0, max(0.0, (1.0 / math.ceil(N / M) - 0.01) / 0.99))
    ev(x[None, :])


ALGO = {
    "RS": lambda ev, r, c: algo_random(ev, r, c),
    "GA": lambda ev, r, c: algo_ga(ev, r, c),
    "GWO": lambda ev, r, c: algo_gwo_family(ev, r, c, ga=False),
    "GWOGA": lambda ev, r, c: algo_gwo_family(ev, r, c, ga=True),
    "GWO+X": lambda ev, r, c: algo_gwo_family(ev, r, c, ga=True, use_x=True, use_m=False),
    "GWO+M": lambda ev, r, c: algo_gwo_family(ev, r, c, ga=True, use_x=False, use_m=True),
    "PSO": lambda ev, r, c: algo_pso(ev, r, c),
    "DE": lambda ev, r, c: algo_de(ev, r, c),
    "WOA": lambda ev, r, c: algo_woa(ev, r, c),
    "EQ": lambda ev, r, c: reference_solution(ev, "EQ"),
    "LOC": lambda ev, r, c: reference_solution(ev, "LOC"),
}
ALGO_ID = {k: i for i, k in enumerate(ALGO)}
GRID = 100


def run_one(job):
    exp, label, algo, seed, cfg = job
    prob = Problem(cfg, seed)
    budget = cfg["budget"] if algo not in REF_ALGOS else 1
    ev = Evaluator(prob, budget)
    rng = np.random.default_rng(1_000_003 * seed + 97 * ALGO_ID[algo] + 13)
    t0 = time.perf_counter()
    ALGO[algo](ev, rng, cfg)
    dt = time.perf_counter() - t0
    d = prob.detail(ev.bestx)
    d.update(exp=exp, label=label, algo=algo, seed=seed, J=ev.best, time=dt)
    grid = np.linspace(cfg["budget"] / GRID, cfg["budget"], GRID)
    hfe, hb = np.array(ev.hfe), np.array(ev.hbest)
    if algo in REF_ALGOS:
        d["curve"] = [float(ev.best)] * GRID
    else:
        pos = np.searchsorted(hfe, grid, side="right") - 1
        d["curve"] = hb[np.clip(pos, 0, len(hb) - 1)].tolist()
    if ev.div:
        dfe, dv = zip(*ev.div)
        d["div"] = np.interp(grid, dfe, dv).tolist()
    else:
        d["div"] = None
    return d


# ----------------------------------------------------------------------------
# Experiment definitions
# ----------------------------------------------------------------------------
def build_jobs(args):
    base = dict(DEFAULT)
    base["budget"] = args.budget
    seeds = range(args.seeds)
    q = args.quick
    tasks = [10, 30, 50] if q else [10, 20, 30, 40, 50]
    caps = [10e9, 30e9, 50e9] if q else [10e9, 20e9, 30e9, 40e9, 50e9]
    pops = [50, 200] if q else [50, 100, 200, 400, 600]
    wEs = [0.1, 0.5, 0.9] if q else [0.1, 0.3, 0.5, 0.7, 0.9]
    pcs = [0.4, 0.8] if q else [0.4, 0.6, 0.8, 1.0]
    pms = [0.05, 0.2] if q else [0.05, 0.1, 0.2, 0.4]
    mus = [0.02, 0.1] if q else [0.005, 0.01, 0.02, 0.05, 0.1]
    jobs = []

    def add(exp, label, algos, seeds_=None, **ov):
        cfg = dict(base); cfg.update(ov)
        for a in algos:
            for s in (seeds_ if seeds_ is not None else seeds):
                jobs.append((exp, label, a, s, cfg))

    only = args.only
    if only == "new":
        only = "E1,E5,E9,E10,E11,E12"
    elif only == "all":
        only = ""
    want = set(only.split(",")) if only else None
    on = lambda e: (want is None) or (e in want)
    if on("E1"):
        add("E1", "base", MAIN_ALGOS + REF_ALGOS)
    if on("E2"):
        for n in tasks:
            add("E2", f"N={n}", MAIN_ALGOS + REF_ALGOS, N=n)
    if on("E3"):
        for F in caps:
            add("E3", f"F={F/1e9:g}", MAIN_ALGOS + REF_ALGOS, F=F)
    if on("E4"):
        for K in pops:
            add("E4", f"K={K}", ["GA", "GWO", "GWOGA"], K=K)
    if on("E5"):
        add("E5", "base", ["GWO", "GWO+X", "GWO+M", "GWOGA"])
    if on("E6"):
        for n in tasks:
            add("E6", f"NOMA|N={n}", ["GWOGA"], N=n, Umax=3)
            add("E6", f"OMA|N={n}", ["GWOGA"], N=n, Umax=1)
    if on("E7"):
        for w in wEs:
            add("E7", f"wE={w}", ["GA", "GWO", "GWOGA"], wE=w, wL=1 - w)
    if on("E8"):
        for pc in pcs:
            for pm in pms:
                add("E8", f"pc={pc}|pm={pm}", ["GWOGA"], pc=pc, pm=pm)
        for mu in mus:
            add("E8", f"mu={mu}", ["GWOGA"], mu=mu)
    four = ["GA", "GWO", "PSO", "GWOGA"]
    budgets = [1000, 2000] if q else [2000, 5000, 10000, 20000, 50000]
    if on("E9"):
        for b in budgets:
            add("E9", f"Q={b}", four, budget=b)
    if on("E10"):
        add("E10", "Q=100000" if not q else "Q=4000", ["GWO", "PSO", "GWOGA"],
            seeds_=list(range(args.long_seeds)), budget=100000 if not q else 4000)
    if on("E11"):
        for m in ([1, 2] if q else [1, 2, 4]):
            add("E11", f"M={m}", four + REF_ALGOS, M=m)
    if on("E12"):
        for n in ([40] if q else [60, 100]):
            add("E12", f"N={n}", four + REF_ALGOS, seeds_=list(range(args.scale_seeds)),
                N=n, Rb=int(round(8 * n / 30)), M=max(2, int(round(2 * n / 30))))
    return jobs


# ----------------------------------------------------------------------------
# Aggregation, statistics, plotting
# ----------------------------------------------------------------------------
def tcrit(n):
    if n < 2:
        return 0.0
    return float(student_t.ppf(0.975, n - 1)) if student_t else 1.96


def group(results):
    g = {}
    for r in results:
        g.setdefault((r["exp"], r["label"], r["algo"]), []).append(r)
    for k in g:
        g[k].sort(key=lambda r: r["seed"])
    return g


def stat(rs, key):
    v = np.array([r[key] for r in rs], float)
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else 0.0
    return v.mean(), sd, tcrit(n) * sd / math.sqrt(n) if n > 1 else 0.0


def pval(g, exp, label, a, b, key="J"):
    if wilcoxon is None:
        return float("nan")
    x = np.array([r[key] for r in g[(exp, label, a)]])
    y = np.array([r[key] for r in g[(exp, label, b)]])
    if len(x) != len(y) or np.allclose(x, y):
        return float("nan")
    try:
        return float(wilcoxon(x, y).pvalue)
    except Exception:
        return float("nan")


def setup_mpl():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.1,
                         "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 150, "savefig.dpi": 300})


def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name + ".png"))
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    plt.close(fig)


def x_plot(ax, xs, g, exp, labels, algos, key, scale=1.0):
    for a in algos:
        m = np.array([stat(g[(exp, l, a)], key)[0] for l in labels]) * scale
        ci = np.array([stat(g[(exp, l, a)], key)[2] for l in labels]) * scale
        mk, ls, col = STYLE[a]
        ax.errorbar(xs, m, yerr=ci, marker=mk or None, ls=ls, color=col,
                    capsize=2, label=LABEL[a])


def make_figures(g, outdir, args):
    setup_mpl()
    W, H = 3.5, 2.7
    algos = MAIN_ALGOS + REF_ALGOS
    # --- image2: convergence (E1)
    if ("E1", "base", "GWOGA") in g:
        fig, ax = plt.subplots(figsize=(W, H))
        xs = np.linspace(args.budget / GRID, args.budget, GRID)
        for a in MAIN_ALGOS:
            C = np.array([r["curve"] for r in g[("E1", "base", a)]])
            m = C.mean(0); ci = tcrit(len(C)) * C.std(0, ddof=1) / math.sqrt(len(C)) if len(C) > 1 else 0 * m
            mk, ls, col = STYLE[a]
            ax.plot(xs, m, ls=ls, color=col, label=LABEL[a], marker=mk or None, markevery=10)
            ax.fill_between(xs, m - ci, m + ci, color=col, alpha=0.15, lw=0)
        for a in REF_ALGOS:
            ax.axhline(stat(g[("E1", "base", a)], "J")[0], color=STYLE[a][2], ls=STYLE[a][1], lw=0.9, label=LABEL[a])
        ax.set_xlabel("Number of fitness evaluations (NFE)")
        ax.set_ylabel("Best-so-far normalized cost $J$")
        ax.legend(ncol=2)
        save(fig, outdir, "image2")
    # --- image3 / image5 (E2)
    labs = sorted({k[1] for k in g if k[0] == "E2"}, key=lambda s: int(s.split("=")[1]))
    if labs:
        xs = [int(l.split("=")[1]) for l in labs]
        fig, ax = plt.subplots(figsize=(W, H))
        x_plot(ax, xs, g, "E2", labs, algos, "E_mJ")
        ax.set_xlabel("Number of tasks $N$"); ax.set_ylabel("Mean energy per task (mJ)")
        ax.legend(ncol=2); save(fig, outdir, "image3")
        fig, ax = plt.subplots(figsize=(W, H))
        x_plot(ax, xs, g, "E2", labs, algos, "L_ms")
        ax.axhline(DEFAULT["Lmax"] * 1e3, color="k", ls=(0, (1, 3)), lw=0.8)
        ax.set_xlabel("Number of tasks $N$"); ax.set_ylabel("Mean delay per task (ms)")
        ax.legend(ncol=2); save(fig, outdir, "image5")
    # --- image4 (E3)
    labs = sorted({k[1] for k in g if k[0] == "E3"}, key=lambda s: float(s.split("=")[1]))
    if labs:
        xs = [float(l.split("=")[1]) for l in labs]
        fig, ax = plt.subplots(figsize=(W, H))
        x_plot(ax, xs, g, "E3", labs, algos, "E_mJ")
        ax.set_xlabel("Fog-node computing capacity (GHz)"); ax.set_ylabel("Mean energy per task (mJ)")
        ax.legend(ncol=2); save(fig, outdir, "image4")
    # --- image6 (E4)
    labs = sorted({k[1] for k in g if k[0] == "E4"}, key=lambda s: int(s.split("=")[1]))
    if labs:
        xs = [int(l.split("=")[1]) for l in labs]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.2), sharex=True)
        x_plot(a1, xs, g, "E4", labs, ["GA", "GWO", "GWOGA"], "J")
        a1.set_ylabel("Final normalized cost $J$"); a1.legend()
        x_plot(a2, xs, g, "E4", labs, ["GA", "GWO", "GWOGA"], "time")
        a2.set_ylabel("Execution time (s)"); a2.set_xlabel("Population size $K$ (fixed NFE budget)")
        save(fig, outdir, "image6")
    # --- image7 (E5)
    if ("E5", "base", "GWOGA") in g:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.4), sharex=True)
        xs = np.linspace(args.budget / GRID, args.budget, GRID)
        for a in ["GWO", "GWO+X", "GWO+M", "GWOGA"]:
            mk, ls, col = STYLE[a]
            C = np.array([r["curve"] for r in g[("E5", "base", a)]]).mean(0)
            Dv = np.array([r["div"] for r in g[("E5", "base", a)]]).mean(0)
            a1.plot(xs, C, ls=ls, color=col, label=LABEL[a], marker=mk or None, markevery=10)
            a2.plot(xs, Dv, ls=ls, color=col, label=LABEL[a], marker=mk or None, markevery=10)
        a1.set_ylabel("Best-so-far cost $J$"); a1.legend()
        a2.set_ylabel("Population diversity"); a2.set_xlabel("Number of fitness evaluations (NFE)")
        save(fig, outdir, "image7")
    # --- image8 (E6)
    labs = sorted({k[1].split("|")[1] for k in g if k[0] == "E6"}, key=lambda s: int(s.split("=")[1]))
    if labs:
        xs = [int(l.split("=")[1]) for l in labs]
        fig, ax = plt.subplots(figsize=(W, H))
        for kind, mk, ls, col in [("NOMA", "*", "-", "tab:blue"), ("OMA", "o", "--", "tab:red")]:
            m = [stat(g[("E6", f"{kind}|{l}", "GWOGA")], "J")[0] for l in labs]
            ci = [stat(g[("E6", f"{kind}|{l}", "GWOGA")], "J")[2] for l in labs]
            ax.errorbar(xs, m, yerr=ci, marker=mk, ls=ls, color=col, capsize=2, label=f"GWOGA, {kind}")
        if any(k[0] == "E2" for k in g):
            l2 = [f"N={n}" for n in xs]
            if all(("E2", l, "EQ") in g for l in l2):
                for a in REF_ALGOS:
                    ax.plot(xs, [stat(g[("E2", l, a)], "J")[0] for l in l2], color=STYLE[a][2],
                            ls=STYLE[a][1], marker=STYLE[a][0] or None, label=LABEL[a])
        ax.set_xlabel("Number of tasks $N$"); ax.set_ylabel("Final normalized cost $J$")
        ax.legend(); save(fig, outdir, "image8")
    # --- image9 (E7)
    labs = sorted({k[1] for k in g if k[0] == "E7"}, key=lambda s: float(s.split("=")[1]))
    if labs:
        fig, ax = plt.subplots(figsize=(W, H))
        for a in ["GA", "GWO", "GWOGA"]:
            ex = [stat(g[("E7", l, a)], "E_mJ")[0] for l in labs]
            lx = [stat(g[("E7", l, a)], "L_ms")[0] for l in labs]
            mk, ls, col = STYLE[a]
            ax.plot(lx, ex, marker=mk or None, ls=ls, color=col, label=LABEL[a])
        a0 = "GWOGA"
        for l in labs:
            ax.annotate(f"$w_E$={l.split('=')[1]}", (stat(g[("E7", l, a0)], "L_ms")[0],
                        stat(g[("E7", l, a0)], "E_mJ")[0]), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel("Mean delay per task (ms)"); ax.set_ylabel("Mean energy per task (mJ)")
        ax.legend(); save(fig, outdir, "image9")
    # --- image10 (E8)
    grid_labs = [k[1] for k in g if k[0] == "E8" and k[1].startswith("pc=")]
    mu_labs = sorted({k[1] for k in g if k[0] == "E8" and k[1].startswith("mu=")},
                     key=lambda s: float(s.split("=")[1]))
    if grid_labs:
        pcs = sorted({float(l.split("|")[0].split("=")[1]) for l in grid_labs})
        pms = sorted({float(l.split("|")[1].split("=")[1]) for l in grid_labs})
        Z = np.array([[stat(g[("E8", f"pc={pc}|pm={pm}", "GWOGA")], "J")[0] for pm in pms] for pc in pcs])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.4))
        im = a1.imshow(Z, cmap="viridis", aspect="auto", origin="lower")
        a1.set_xticks(range(len(pms))); a1.set_xticklabels(pms)
        a1.set_yticks(range(len(pcs))); a1.set_yticklabels(pcs)
        a1.set_xlabel("Mutation probability $p_m$"); a1.set_ylabel("Crossover probability $p_c$")
        a1.grid(False)
        for i in range(len(pcs)):
            for j in range(len(pms)):
                a1.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", color="w", fontsize=6)
        fig.colorbar(im, ax=a1, label="Final $J$")
        if mu_labs:
            xs = [float(l.split("=")[1]) for l in mu_labs]
            m = [stat(g[("E8", l, "GWOGA")], "J")[0] for l in mu_labs]
            ci = [stat(g[("E8", l, "GWOGA")], "J")[2] for l in mu_labs]
            a2.errorbar(xs, m, yerr=ci, marker="*", color="tab:blue", capsize=2)
            a2.set_xscale("log"); a2.set_xlabel("Per-gene mutation rate $\\mu$")
            a2.set_ylabel("Final $J$")
        save(fig, outdir, "image10")


def write_summary(g, results, args, resdir):
    L = []
    P = L.append
    n_seeds = args.seeds
    P(f"# SUMMARY  (seeds={n_seeds}, budget={args.budget} fitness evaluations, quick={args.quick})")
    P("# Values: mean +- SD over independent runs. J = normalized cost (all-local = 1).")
    P("# E in mJ per task, L in ms per task, Lmax = worst per-task delay (ms), feas = share of runs with no violation.")
    P("")

    def row(exp, lab, a):
        rs = g[(exp, lab, a)]
        Jm, Js, _ = stat(rs, "J"); Em, Es, _ = stat(rs, "E_mJ"); Lm, Ls, _ = stat(rs, "L_ms")
        Xm, _, _ = stat(rs, "Lmax_ms"); Tm, _, _ = stat(rs, "time"); Om, _, _ = stat(rs, "off")
        fe = np.mean([r["feasible"] for r in rs])
        return (f"{LABEL[a]:<22} J={Jm:.4f}+-{Js:.4f}  E={Em:.2f}+-{Es:.2f}  L={Lm:.1f}+-{Ls:.1f}  "
                f"Lmax={Xm:.0f}  feas={fe:.2f}  offload={Om:.2f}  time={Tm:.2f}s")

    tex = []
    if ("E1", "base", "GWOGA") in g:
        P("## E1  Default scenario (final values at the end of the budget)")
        for a in MAIN_ALGOS + REF_ALGOS:
            P(row("E1", "base", a))
            if a in MAIN_ALGOS:
                Jm, Js, _ = stat(g[("E1", "base", a)], "J")
                Em, Es, _ = stat(g[("E1", "base", a)], "E_mJ")
                Lm, Ls, _ = stat(g[("E1", "base", a)], "L_ms")
                Tm, Ts, _ = stat(g[("E1", "base", a)], "time")
                p = pval(g, "E1", "base", "GWOGA", a) if a != "GWOGA" else float("nan")
                tex.append(f"{LABEL[a]} & ${Jm:.3f}\\pm{Js:.3f}$ & ${Em:.1f}\\pm{Es:.1f}$ & "
                           f"${Lm:.0f}\\pm{Ls:.0f}$ & ${Tm:.2f}$ & "
                           f"{'--' if a == 'GWOGA' else f'{p:.1e}'}")
        P("Wilcoxon signed-rank p-values, GWOGA vs other (final J, paired by seed):")
        for a in MAIN_ALGOS[:-1]:
            P(f"  GWOGA vs {a}: p={pval(g, 'E1', 'base', 'GWOGA', a):.3g}")
        P("")
    for exp, title, key in [("E2", "E2  Number of tasks N", "N"), ("E3", "E3  Fog capacity F (GHz)", "F")]:
        labs = sorted({k[1] for k in g if k[0] == exp},
                      key=lambda s: float(s.split("=")[1]))
        if labs:
            P(f"## {title}")
            for l in labs:
                P(f"-- {l}")
                for a in MAIN_ALGOS + REF_ALGOS:
                    P("   " + row(exp, l, a))
                P("   p-values GWOGA vs: " + ", ".join(
                    f"{a}={pval(g, exp, l, 'GWOGA', a):.2g}" for a in MAIN_ALGOS[:-1]))
            P("")
    labs = sorted({k[1] for k in g if k[0] == "E4"}, key=lambda s: int(s.split("=")[1]))
    if labs:
        P("## E4  Population size K (same evaluation budget)")
        for l in labs:
            P(f"-- {l}")
            for a in ["GA", "GWO", "GWOGA"]:
                P("   " + row("E4", l, a))
        P("")
    if ("E5", "base", "GWOGA") in g:
        P("## E5  Ablation (final diversity = mean per-gene std of population)")
        for a in ["GWO", "GWO+X", "GWO+M", "GWOGA"]:
            dv = np.mean([r["div"][-1] for r in g[("E5", "base", a)]])
            P(row("E5", "base", a) + f"  final_diversity={dv:.4f}")
        P("  p-values GWOGA vs: " + ", ".join(
            f"{a}={pval(g, 'E5', 'base', 'GWOGA', a):.2g}" for a in ["GWO", "GWO+X", "GWO+M"]))
        P("")
    labs = sorted({k[1].split("|")[1] for k in g if k[0] == "E6"}, key=lambda s: int(s.split("=")[1]))
    if labs:
        P("## E6  NOMA (Umax=3) vs OMA (Umax=1), GWOGA")
        for l in labs:
            P(f"-- {l}")
            for kind in ["NOMA", "OMA"]:
                rs = g[("E6", f"{kind}|{l}", "GWOGA")]
                P(f"   {kind:<5} " + row("E6", f"{kind}|{l}", "GWOGA").split("  ", 1)[1])
            p = pval(g, "E6", f"NOMA|{l}", "GWOGA", "GWOGA") if False else float("nan")
            if wilcoxon is not None:
                x = np.array([r["J"] for r in g[("E6", f"NOMA|{l}", "GWOGA")]])
                y = np.array([r["J"] for r in g[("E6", f"OMA|{l}", "GWOGA")]])
                try:
                    p = float(wilcoxon(x, y).pvalue)
                except Exception:
                    pass
            P(f"   Wilcoxon NOMA vs OMA: p={p:.3g}")
        P("")
    labs = sorted({k[1] for k in g if k[0] == "E7"}, key=lambda s: float(s.split("=")[1]))
    if labs:
        P("## E7  Weight sensitivity (wL = 1 - wE)")
        for l in labs:
            P(f"-- {l}")
            for a in ["GA", "GWO", "GWOGA"]:
                P("   " + row("E7", l, a))
        P("")
    labs = [k[1] for k in g if k[0] == "E8"]
    if labs:
        P("## E8  GWOGA parameter sensitivity (final J)")
        for l in sorted(set(labs)):
            Jm, Js, _ = stat(g[("E8", l, "GWOGA")], "J")
            P(f"   {l:<18} J={Jm:.4f}+-{Js:.4f}")
        P("")
    # overall feasibility
    P("## Constraint feasibility over all runs: %.3f" % np.mean([r["feasible"] for r in results]))
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    if tex:
        with open(os.path.join(resdir, "table_main.tex"), "w", encoding="utf-8") as f:
            f.write("% columns: Algorithm & J & Energy (mJ) & Delay (ms) & Time (s) & Wilcoxon p vs GWOGA (last row break is added by the tex)\n")
            f.write(" \\\\\n".join(tex) + "\n")   # rows separated by \\ ; the tex adds the last one
    print("\n".join(L))


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# v2 additions: hardware info, timing, statistics, extra figures, system figure
# ----------------------------------------------------------------------------
def hardware_info():
    import platform
    info = {"python": sys.version.split()[0], "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count()}
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu_model"] = [l.split(":", 1)[1].strip() for l in f if l.startswith("model name")][0]
    except Exception:
        info["cpu_model"] = platform.processor()
    try:
        with open("/proc/meminfo") as f:
            info["ram_gb"] = round(int(f.readline().split()[1]) / 1e6, 1)
    except Exception:
        pass
    return info


def run_timing(n_seeds, budget, algos):
    """Sequential, single-process timing (no parallel interference). The order of the
    algorithms is shuffled for every seed to avoid a systematic order effect."""
    import gc
    cfg = dict(DEFAULT); cfg["budget"] = budget
    rows = []
    rng0 = np.random.default_rng(7)
    prob = Problem(cfg, 0); ev = Evaluator(prob, min(budget, 2000))           # warm-up
    ALGO["GWOGA"](ev, np.random.default_rng(0), cfg)
    for s in range(n_seeds):
        for a in rng0.permutation(algos):
            prob = Problem(cfg, s); ev = Evaluator(prob, budget)
            rng = np.random.default_rng(1_000_003 * s + 97 * ALGO_ID[a] + 13)
            gc.collect()
            t0, c0 = time.perf_counter(), time.process_time()
            ALGO[a](ev, rng, cfg)
            rows.append(dict(algo=str(a), seed=s, wall=time.perf_counter() - t0,
                             cpu=time.process_time() - c0, J=float(ev.best)))
    return rows


def holm_adjust(pdict):
    srt = sorted(pdict.items(), key=lambda t: t[1])
    m, run, out = len(srt), 0.0, {}
    for i, (k, p) in enumerate(srt):
        run = max(run, (m - i) * p)
        out[k] = min(1.0, run)
    return out


def stats_tests(g, resdir):
    """Paired Wilcoxon (GWOGA vs each other metaheuristic) with Holm correction
    within every scenario of E1, E2, E3."""
    rows = []
    if wilcoxon is None:
        return rows
    others = ["RS", "GA", "DE", "WOA", "PSO", "GWO"]
    for exp in ["E1", "E2", "E3"]:
        for lab in sorted({k[1] for k in g if k[0] == exp}):
            if (exp, lab, "GWOGA") not in g:
                continue
            raw, diff = {}, {}
            xs = np.array([r["J"] for r in g[(exp, lab, "GWOGA")]])
            for a in others:
                ys = np.array([r["J"] for r in g[(exp, lab, a)]])
                try:
                    raw[a] = float(wilcoxon(xs, ys).pvalue)
                except Exception:
                    raw[a] = 1.0
                diff[a] = float((xs - ys).mean())
            adj = holm_adjust(raw)
            for a in others:
                rows.append(dict(exp=exp, label=lab, vs=a, mean_diff_J=diff[a],
                                 p_raw=raw[a], p_holm=adj[a],
                                 better="GWOGA" if diff[a] < 0 else "other"))
    import csv
    with open(os.path.join(resdir, "stats_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def flat_csv(results, resdir):
    import csv
    cols = ["exp", "label", "algo", "seed", "J", "Jc", "E_mJ", "E_tx_mJ", "E_loc_mJ", "L_ms",
            "Lmax_ms", "off", "feasible", "time", "final_div"]
    with open(os.path.join(resdir, "output_v2.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            row = dict(r); row["final_div"] = r["div"][-1] if r.get("div") else ""
            w.writerow([row.get(c, "") for c in cols])


def curves_csv(g, resdir, budget):
    import csv
    if ("E1", "base", "GWOGA") not in g or "curve" not in g[("E1", "base", "GWOGA")][0]:
        return
    xs = np.linspace(budget / GRID, budget, GRID)
    with open(os.path.join(resdir, "curves_E1.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["algo", "nfe", "J_mean", "J_sd"])
        for a in MAIN_ALGOS + REF_ALGOS:
            C = np.array([r["curve"] for r in g[("E1", "base", a)]])
            for x, m, sd in zip(xs, C.mean(0), C.std(0, ddof=1) if len(C) > 1 else 0 * C.mean(0)):
                w.writerow([a, int(x), m, sd])


def extra_figures(g, outdir, args):
    setup_mpl()
    W, H = 3.5, 2.7
    four = ["GA", "GWO", "PSO", "GWOGA"]
    # image11: budget sensitivity
    labs = sorted({k[1] for k in g if k[0] == "E9"}, key=lambda s: int(s.split("=")[1]))
    if labs:
        xs = [int(l.split("=")[1]) for l in labs]
        fig, ax = plt.subplots(figsize=(W, H))
        x_plot(ax, xs, g, "E9", labs, four, "J")
        ax.set_xscale("log"); ax.set_xlabel("Evaluation budget $Q$"); ax.set_ylabel("Final normalized cost $J$")
        ax.legend(); save(fig, outdir, "image11")
    # image12: long run
    l10 = [k for k in g if k[0] == "E10"]
    if l10 and ("E1", "base", "GWOGA") in g:
        lab = l10[0][1]
        fig, ax = plt.subplots(figsize=(W, H))
        seeds = sorted({r["seed"] for r in g[l10[0]]})
        for j, a in enumerate(["GWO", "PSO", "GWOGA"]):
            short = [r["J"] for r in g[("E1", "base", a)] if r["seed"] in seeds]
            long_ = [r["J"] for r in g[("E10", lab, a)]]
            ax.errorbar([j - 0.12], [np.mean(short)], yerr=[np.std(short, ddof=1)], fmt="o", color="0.4", capsize=2,
                        label="budget $10^4$" if j == 0 else None)
            ax.errorbar([j + 0.12], [np.mean(long_)], yerr=[np.std(long_, ddof=1)], fmt="s", color="tab:blue", capsize=2,
                        label=f"long run ({lab.split('=')[1]})" if j == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(["GWO", "PSO", "GWOGA"])
        ax.set_ylabel("Final normalized cost $J$"); ax.legend(); save(fig, outdir, "image12")
    # image13: several fog nodes
    labs = sorted({k[1] for k in g if k[0] == "E11"}, key=lambda s: int(s.split("=")[1]))
    if labs:
        xs = [int(l.split("=")[1]) for l in labs]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.2), sharex=True)
        x_plot(a1, xs, g, "E11", labs, four + REF_ALGOS, "J"); a1.set_ylabel("Final normalized cost $J$"); a1.legend(ncol=2)
        x_plot(a2, xs, g, "E11", labs, four, "E_mJ"); a2.set_ylabel("Mean energy per task (mJ)")
        a2.set_xlabel("Number of fog nodes $M$"); a2.set_xticks(xs); save(fig, outdir, "image13")
    # image14: scale
    labs = sorted({k[1] for k in g if k[0] == "E12"}, key=lambda s: int(s.split("=")[1]))
    if labs:
        xs = [int(l.split("=")[1]) for l in labs]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.2), sharex=True)
        x_plot(a1, xs, g, "E12", labs, four + REF_ALGOS, "J"); a1.set_ylabel("Final normalized cost $J$"); a1.legend(ncol=2)
        x_plot(a2, xs, g, "E12", labs, four, "time"); a2.set_ylabel("Execution time (s)")
        a2.set_xlabel("Number of tasks $N$ (scaled blocks and nodes)"); a2.set_xticks(xs); save(fig, outdir, "image14")


def draw_system_figure(outdir):
    """image1.png: three-layer model with NOMA clusters in resource blocks."""
    from matplotlib.patches import FancyBboxPatch, Circle
    setup_mpl()
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.grid(False)
    def box(x, y, w, h, text, fc="white", fs=6.5, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fc, ec="0.2", lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    box(2.2, 8.6, 5.6, 1.1, "Cloud data center\n(outside the optimization)", fc="0.93", ls="--")
    box(0.5, 5.6, 4.0, 1.5, "Fog node 1\ncapacity $F_1$", fc="0.85")
    box(5.5, 5.6, 4.0, 1.5, "Fog node 2\ncapacity $F_2$", fc="0.85")
    ax.annotate("", xy=(5, 8.55), xytext=(2.5, 7.15), arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8))
    ax.annotate("", xy=(5, 8.55), xytext=(7.5, 7.15), arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8))
    rng = np.random.default_rng(3)
    blocks = [(0.3, "Block 1"), (3.5, "Block 2"), (6.7, "Block 3")]
    for bx, name in blocks:
        ax.add_patch(FancyBboxPatch((bx, 2.5), 3.0, 1.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="none", ec="tab:blue", lw=1.0, ls="-"))
        ax.text(bx + 1.5, 4.65, name + " (NOMA cluster)", ha="center", fontsize=5.8, color="tab:blue")
        for k in range(3):
            cx, cy = bx + 0.55 + k * 0.95, 3.3
            ax.add_patch(Circle((cx, cy), 0.28, fc="white", ec="0.2", lw=0.9))
            ax.text(cx, cy, "UE", ha="center", va="center", fontsize=4.5)
            ax.text(cx, 2.75, ["$p_1$", "$p_2$", "$p_3$"][k], ha="center", fontsize=5)
    for (x0, x1) in [(1.6, 2.5), (4.8, 2.5), (8.0, 7.5)]:
        ax.annotate("", xy=(x1, 5.55), xytext=(x0, 4.45), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))
    ax.text(5, 1.75, "Devices in one block share it (NOMA); the fog node applies SIC in\n"
                     "decreasing received power $q_i=g_ip_i$. Devices may also run locally.",
            ha="center", va="center", fontsize=5.6)
    box(1.2, 0.2, 2.9, 0.9, "Local execution ($a_i=0$)", fc="0.93", fs=5.8)
    box(5.2, 0.2, 3.6, 0.9, "Offloading ($a_i>0$): uplink", fc="0.93", fs=5.8)
    fig.savefig(os.path.join(outdir, "image1.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "image1.pdf"), bbox_inches="tight")
    plt.close(fig)


def write_extra_summary(g, results, timing, hw, stats, resdir):
    L, P = [], None
    P = L.append
    P("\n" + "=" * 70)
    P("# v2 ADDITIONS")
    P("## Hardware / software"); P(json.dumps(hw, indent=1))
    def row(exp, lab, a):
        rs = g[(exp, lab, a)]
        Jm, Js, _ = stat(rs, "J"); Em, Es, _ = stat(rs, "E_mJ"); Lm, Ls, _ = stat(rs, "L_ms")
        Xm = stat(rs, "Lmax_ms")[0]; Tm = stat(rs, "time")[0]; Om = stat(rs, "off")[0]
        fe = np.mean([r["feasible"] for r in rs])
        Et = stat(rs, "E_tx_mJ")[0] if "E_tx_mJ" in rs[0] else float("nan")
        El = stat(rs, "E_loc_mJ")[0] if "E_loc_mJ" in rs[0] else float("nan")
        return (f"{LABEL[a]:<22} J={Jm:.4f}+-{Js:.4f}  E={Em:.2f}(tx {Et:.2f} + local {El:.2f})  "
                f"L={Lm:.1f}+-{Ls:.1f}  Lmax={Xm:.0f}  feas={fe:.2f}  offload={Om:.2f}  time={Tm:.2f}s")
    if ("E1", "base", "GWOGA") in g and "E_tx_mJ" in g[("E1", "base", "GWOGA")][0]:
        P("\n## E1 energy breakdown (default scenario)")
        for a in MAIN_ALGOS + REF_ALGOS:
            P("   " + row("E1", "base", a))
    if ("E5", "base", "GWOGA") in g and g[("E5", "base", "GWOGA")][0].get("div"):
        P("\n## E5 final population diversity (mean per-gene std)")
        for a in ["GWO", "GWO+X", "GWO+M", "GWOGA"]:
            dv = np.array([r["div"][-1] for r in g[("E5", "base", a)]])
            P(f"   {LABEL[a]:<18} {dv.mean():.5f} +- {dv.std(ddof=1):.5f}")
        for a in ["GWO", "GWOGA"]:
            C = np.array([r["div"] for r in g[("E5", "base", a)]]).mean(0)
            P(f"   diversity at 10/50/100% of budget, {a}: {C[9]:.4f} / {C[49]:.4f} / {C[-1]:.4f}")
    if ("E1", "base", "GWOGA") in g and "curve" in g[("E1", "base", "GWOGA")][0]:
        P("\n## E1 convergence: mean best-so-far J at 5/10/25/50/100% of the budget")
        for a in MAIN_ALGOS:
            C = np.array([r["curve"] for r in g[("E1", "base", a)]]).mean(0)
            P(f"   {LABEL[a]:<22} " + "  ".join(f"{C[i]:.4f}" for i in [4, 9, 24, 49, 99]))
        P("   evaluations needed to reach J <= 0.75 (mean best-so-far curve):")
        for a in MAIN_ALGOS:
            C = np.array([r["curve"] for r in g[("E1", "base", a)]]).mean(0)
            hit = np.where(C <= 0.75)[0]
            P(f"   {LABEL[a]:<22} " + (f"{(hit[0]+1)}% of budget" if len(hit) else "not reached"))
    for exp, title in [("E9", "Budget sensitivity"), ("E10", "Long run"), ("E11", "Several fog nodes"),
                       ("E12", "Larger scale")]:
        labs = sorted({k[1] for k in g if k[0] == exp})
        if labs:
            P(f"\n## {exp} {title}")
            for l in labs:
                P(f"-- {l}")
                for a in ["GA", "GWO", "PSO", "GWOGA", "EQ", "LOC"]:
                    if (exp, l, a) in g:
                        P("   " + row(exp, l, a))
                if wilcoxon is not None and (exp, l, "GWOGA") in g and (exp, l, "GWO") in g:
                    P(f"   Wilcoxon GWOGA vs GWO: p={pval(g, exp, l, 'GWOGA', 'GWO'):.3g}; "
                      f"vs PSO: p={pval(g, exp, l, 'GWOGA', 'PSO'):.3g}" if (exp, l, "PSO") in g else "")
    if ("E10" in {k[0] for k in g}) and ("E1", "base", "GWOGA") in g:
        lab = [k[1] for k in g if k[0] == "E10"][0]
        P("\n## Gap between the 10^4-evaluation result and the long-run result (same seeds)")
        for a in ["GWO", "PSO", "GWOGA"]:
            seeds = sorted({r["seed"] for r in g[("E10", lab, a)]})
            sh = np.mean([r["J"] for r in g[("E1", "base", a)] if r["seed"] in seeds])
            lg = np.mean([r["J"] for r in g[("E10", lab, a)]])
            P(f"   {a:<6} J(10^4)={sh:.4f}  J(long)={lg:.4f}  improvement={100*(sh-lg)/sh:.2f}%")
    if timing:
        P("\n## E13 sequential timing (seconds per run, default scenario)")
        for a in sorted({t['algo'] for t in timing}):
            w = np.array([t["wall"] for t in timing if t["algo"] == a]); c = np.array([t["cpu"] for t in timing if t["algo"] == a])
            P(f"   {LABEL[a]:<22} wall {w.mean():.3f} +- {w.std(ddof=1):.3f}   cpu {c.mean():.3f}")
    if stats:
        P("\n## Holm-corrected paired Wilcoxon tests (GWOGA vs other; negative diff = GWOGA lower J)")
        for r in stats:
            P(f"   {r['exp']} {r['label']:<8} vs {r['vs']:<5} diff={r['mean_diff_J']:+.4f}  p={r['p_raw']:.3g}  holm={r['p_holm']:.3g}")
    txt = "\n".join(L)
    with open(os.path.join(resdir, "summary.txt"), "a", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


def main():
    import shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--long-seeds", dest="long_seeds", type=int, default=10)
    ap.add_argument("--scale-seeds", dest="scale_seeds", type=int, default=10)
    ap.add_argument("--timing-seeds", dest="timing_seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=str, default="new",
                    help="'new' (E1,E5,E9-E12 + timing), 'all', or a comma list such as E1,E9")
    ap.add_argument("--force", action="store_true", help="ignore saved raw_*.json files")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--outdir", type=str, default="images/media")
    ap.add_argument("--resdir", type=str, default="results")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.budget = min(args.seeds, 3), min(args.budget, 2000)
        args.long_seeds = args.scale_seeds = args.timing_seeds = 2
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resdir, exist_ok=True)
    hw = hardware_info()
    jobs = build_jobs(args)
    exps = []
    for j in jobs:
        if j[0] not in exps:
            exps.append(j[0])
    print(f"{len(jobs)} runs in experiments {exps} on {args.workers} worker(s)", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for e in exps:
            fn = os.path.join(args.resdir, f"raw_{e}.json")
            if os.path.exists(fn) and not args.force:
                with open(fn) as f:
                    results.extend(json.load(f))
                print(f"[{e}] loaded saved results ({fn})", flush=True)
                continue
            ej = [j for j in jobs if j[0] == e]
            part = []
            for i, r in enumerate(ex.map(run_one, ej, chunksize=2), 1):
                part.append(r)
                if i % max(1, len(ej) // 10) == 0:
                    print(f"[{e}] {i}/{len(ej)} ({time.time()-t0:.0f}s)", flush=True)
            with open(fn, "w") as f:
                json.dump(part, f)
            results.extend(part)
    timing = []
    tfn = os.path.join(args.resdir, "timing.json")
    if not args.no_timing and (args.only in ("new", "all") or "E13" in args.only):
        if os.path.exists(tfn) and not args.force:
            timing = json.load(open(tfn))["rows"]
        else:
            print("[E13] sequential timing ...", flush=True)
            timing = run_timing(args.timing_seeds, args.budget, MAIN_ALGOS)
            json.dump({"hardware": hw, "rows": timing}, open(tfn, "w"))
    g = group(results)
    # also merge previously saved experiments (so that figures/statistics use everything available)
    for fn in sorted(os.listdir(args.resdir)):
        if fn.startswith("raw_") and fn.endswith(".json"):
            e = fn[4:-5]
            if e not in exps:
                results.extend(json.load(open(os.path.join(args.resdir, fn))))
    g = group(results)
    flat_csv(results, args.resdir)
    curves_csv(g, args.resdir, args.budget)
    make_figures(g, args.outdir, args)
    extra_figures(g, args.outdir, args)
    draw_system_figure(args.outdir)
    write_summary(g, results, args, args.resdir)
    stats = stats_tests(g, args.resdir)
    write_extra_summary(g, results, timing, hw, stats, args.resdir)
    # bundle everything that has to be sent back
    bundle = "colab_results_bundle"
    tmp = "_bundle_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(args.resdir, os.path.join(tmp, "results"),
                    ignore=shutil.ignore_patterns("raw_*.json"))
    shutil.copytree(args.outdir, os.path.join(tmp, "images_media"))
    shutil.copy(os.path.join(args.resdir, "raw_E1.json"), os.path.join(tmp, "results")) if os.path.exists(os.path.join(args.resdir, "raw_E1.json")) else None
    shutil.make_archive(bundle, "zip", tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nFinished in {time.time()-t0:.0f}s. Send {bundle}.zip back "
          f"(it contains summary.txt, CSV files and all figures).")


if __name__ == "__main__":
    main()
