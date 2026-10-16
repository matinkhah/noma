#!/usr/bin/env python3
"""
run_experiments19.py -- reference implementation for noma19.tex

Usage examples
  python run_experiments19.py --exp all --runs 30 --workers 8
  python run_experiments19.py --exp E1,E2 --runs 30
  python run_experiments19.py --exp E1 --runs 3 --quick --workers 1      # smoke test

Experiments
  E1 default scenario (all methods, constructive, seeded variants, pairwise tests)
  E2 budget sweep Q = 2e3 ... 1e5, Holm families, TOST equivalence (margin 0.01)
  E3 factorial ablation + hybrid sweeps (rho, p_m, mu) + convergence/diversity curves
  E4 NOMA vs OMA (equal-capacity OMA control with 3R blocks), fixed and scaled (R,M),
     and imperfect-SIC sweep beta
  E5 scalability N = 60, 100 (fixed and linearly scaled budget; feasible-only J)
  E6 population-size sweep K = 10 ... 400
  E7 best-known reference (long seeded restarts) and optimality-gap of each method
  E8 fog-node sweep M = 1, 2, 4
All outputs go to ./results/*.csv ; figures to ./results/*.pdf
Instance seed depends only on (experiment, run, N); algorithm seed additionally on the
algorithm name, so all algorithms see identical instances (paired design).
"""
import argparse, copy, math, os, time, zlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
from scipy.spatial.distance import pdist

P = dict(Btot=10e6, Pmax=0.3, Lmax=1.0, gth=1.0, N0=10 ** (-174 / 10) * 1e-3,
         kappa=1e-28, lam=10.0, Fk=20e9)
OUT = "results"


# ----------------------------------------------------------------------------- model
class Inst:
    def __init__(self, N, rng):
        self.N = N
        self.f = rng.uniform(0.7e9, 1e9, N)
        self.S = rng.uniform(0.1e6, 1e6, N)
        self.c = rng.uniform(300, 700, N)
        d = rng.uniform(50, 300, N)
        self.g = 1e-3 * d ** -3.5
        self.C = self.S * self.c
        self.Eloc = P['kappa'] * self.f ** 2 * self.C
        self.Lloc = self.C / self.f
        self.D = 4 * N
        self.cfg(2, 8, 3)

    def cfg(self, M, R, Umax, beta=0.0, wE=0.5):
        self.M, self.R, self.Umax, self.beta, self.wE = M, R, Umax, beta, wE
        self.W = P['Btot'] / R
        self.sig = P['N0'] * self.W
        return self


def with_cfg(I, **kw):
    J = copy.copy(I)
    return J.cfg(**kw)


def scaled_RM(N):
    return max(1, math.ceil(2 * N / 30)), max(1, math.ceil(8 * N / 30))


def sinr_sorted(qs, bs, R, sig, beta):
    """qs sorted by (block, -q). Uplink SIC: user i sees weaker users (+ beta * cancelled stronger ones)."""
    tot = np.bincount(bs, weights=qs, minlength=R)
    cs = np.cumsum(qs)
    start = np.concatenate(([0.0], np.cumsum(tot)))[:-1]
    incl = cs - start[bs]
    return qs / (np.maximum(tot[bs] - incl, 0) + beta * (incl - qs) + sig)


def decode(x, I):
    N, M, R = I.N, I.M, I.R
    a = np.minimum((x[:N] * (M + 1)).astype(int), M)
    b = np.minimum((x[N:2 * N] * R).astype(int), R - 1)
    p = P['Pmax'] * (0.01 + 0.99 * x[2 * N:3 * N])
    f = P['Fk'] * (0.01 + 0.99 * x[3 * N:])
    return a, b, p, f


def encode(a, b, p, f, I):
    N, M, R = I.N, I.M, I.R
    x = np.zeros(4 * N)
    x[:N] = (a + 0.5) / (M + 1)
    x[N:2 * N] = (b + 0.5) / R
    x[2 * N:3 * N] = np.clip((p / P['Pmax'] - 0.01) / 0.99, 0, 1)
    x[3 * N:] = np.clip((f / P['Fk'] - 0.01) / 0.99, 0, 1)
    return x


def evaluate_dec(I, a, b, p, f, full=False):
    """J = mean_i [ wE*E_i/Eloc_i + wL*L_i/Lloc_i + penalty_i ]  (all-local gives exactly 1).
    Repair: per block only the Umax strongest received-power offloaders stay offloaded;
    node compute shares above Fk are scaled proportionally; fog-side energy is not counted."""
    N = I.N
    a = a.copy()
    L = I.Lloc.copy()
    E = I.Eloc.copy()
    pen = np.zeros(N)
    idx = np.nonzero(a > 0)[0]
    if idx.size:
        q = I.g[idx] * p[idx]
        bb = b[idx]
        o = np.lexsort((-q, bb))
        io, qs, bs = idx[o], q[o], bb[o]
        first = np.searchsorted(bs, bs, side='left')
        keep = (np.arange(len(bs)) - first) < I.Umax
        a[io[~keep]] = 0
        io, qs, bs = io[keep], qs[keep], bs[keep]
        gam = sinr_sorted(qs, bs, I.R, I.sig, I.beta)
        r = I.W * np.log2(1 + gam)
        ttx = I.S[io] / r
        ks = a[io] - 1
        fi = f[io].copy()
        ssum = np.bincount(ks, weights=fi, minlength=I.M)
        fi *= np.minimum(1.0, P['Fk'] / np.maximum(ssum[ks], 1e-9))
        Lo = ttx + I.C[io] / fi
        L[io] = Lo
        E[io] = p[io] * ttx
        pen[io] = P['lam'] * (np.maximum(0, Lo - P['Lmax']) / P['Lmax']
                              + np.maximum(0, 1 - gam / P['gth']))
    En, Ln = E / I.Eloc, L / I.Lloc
    Ji = I.wE * En + (1 - I.wE) * Ln
    J = float((Ji + pen).mean())
    if not full:
        return J
    off = a > 0
    return dict(J=J, J_unpen=float(Ji.mean()), E_abs=float(E.mean() * 1e3), L_abs=float(L.mean() * 1e3),
                En=float(En.mean()), Ln=float(Ln.mean()), Etx=float(E[off].sum() / N * 1e3),
                off_frac=float(off.mean()), feasible=bool(pen.max() <= 1e-12))


class Problem:
    def __init__(self, I, Q, track=False):
        self.I, self.Q, self.D, self.track = I, Q, I.D, track
        self.n, self.bf, self.bx, self.curve, self.div = 0, np.inf, None, [], []

    def __call__(self, x):
        x = np.clip(x, 0, 1)
        v = evaluate_dec(self.I, *decode(x, self.I))
        self.n += 1
        if v < self.bf:
            self.bf, self.bx = v, x.copy()
        if self.n % 100 == 0:
            self.curve.append(self.bf)
        return v

    def gen(self, pop):
        if self.track:
            self.div.append(float(pdist(pop).mean()))


# ----------------------------------------------------------------------------- constructive & references
def constructive(I):
    """Strongest-channel-first. Devices in decreasing g. Node = largest residual capacity;
    share = min(residual, Fk/ceil(N/M)); power = Pmax/2. Blocks tried by increasing occupancy;
    device is admitted if all users of the block remain SIC-decodable (SINR>=gth), all delays
    <= Lmax and every user's own normalised cost stays < 1; otherwise it executes locally."""
    N, M, R = I.N, I.M, I.R
    a = np.zeros(N, int); b = np.zeros(N, int)
    p = np.full(N, P['Pmax'] / 2); f = np.zeros(N)
    occ = [[] for _ in range(R)]
    Fres = np.full(M, P['Fk'])
    share = P['Fk'] / math.ceil(N / M)
    for i in np.argsort(-I.g):
        k = int(np.argmax(Fres))
        fi = min(Fres[k], share)
        if fi <= 0:
            continue
        for bl in np.argsort([len(o) for o in occ], kind='stable'):
            if len(occ[bl]) >= I.Umax:
                continue
            users = np.array(occ[bl] + [i])
            q = I.g[users] * p[users]
            o = np.argsort(-q)
            us, qs = users[o], q[o]
            gam = sinr_sorted(qs, np.zeros(len(qs), int), 1, I.sig, I.beta)
            if np.any(gam < P['gth']):
                continue
            r = I.W * np.log2(1 + gam)
            fs = np.array([f[u] if u != i else fi for u in us])
            Lu = I.S[us] / r + I.C[us] / fs
            if np.any(Lu > P['Lmax']):
                continue
            Eu = p[us] * I.S[us] / r
            if np.any(I.wE * Eu / I.Eloc[us] + (1 - I.wE) * Lu / I.Lloc[us] >= 1):
                continue
            occ[bl].append(i); a[i] = k + 1; b[i] = bl; f[i] = fi; Fres[k] -= fi
            break
    return a, b, p, f


def equal_alloc(I):
    N, M, R = I.N, I.M, I.R
    a = (np.arange(N) % M) + 1
    b = np.arange(N) % R
    return a, b, np.full(N, P['Pmax']), np.full(N, P['Fk'] / math.ceil(N / M))


# ----------------------------------------------------------------------------- optimisers
def _init(prob, K, rng, seed_x):
    pop = rng.random((K, prob.D))
    if seed_x is not None:
        pop[0] = seed_x
    fit = np.full(K, np.inf)
    for i in range(min(K, prob.Q - prob.n)):
        fit[i] = prob(pop[i])
    return pop, fit


def _tourn(pop, fit, n, rng):
    i = rng.integers(0, len(pop), (n, 2))
    return pop[np.where(fit[i[:, 0]] < fit[i[:, 1]], i[:, 0], i[:, 1])]


def rs(prob, K, Q, rng, seed_x=None, **_):
    while prob.n < Q:
        prob(rng.random(prob.D))


def ga(prob, K, Q, rng, seed_x=None, pc=0.8, pm=0.1, mu=0.02, **_):
    pop, fit = _init(prob, K, rng, seed_x)
    while prob.n < Q:
        prob.gen(pop)
        n = min(K - 1, Q - prob.n)
        el = int(np.argmin(fit))
        p1, p2 = _tourn(pop, fit, n, rng), _tourn(pop, fit, n, rng)
        al = rng.random((n, 1)); cx = rng.random((n, 1)) < pc
        ch = np.where(cx, al * p1 + (1 - al) * p2, p1)
        ch = np.clip(ch + (rng.random(ch.shape) < pm) * rng.normal(0, mu, ch.shape), 0, 1)
        cf = np.array([prob(c) for c in ch])
        pop = np.vstack([pop[el], ch]); fit = np.concatenate([[fit[el]], cf])


def gwo_ga(prob, K, Q, rng, seed_x=None, rho=0.0, use_x=True, use_m=True,
           pc=0.8, pm=0.1, mu=0.02, **_):
    """rho = 0 -> plain GWO. rho > 0: per generation (1-rho)K wolves make GWO moves
    (unconditional replacement) and rho*K GA offspring (tournament, arithmetic crossover,
    Gaussian mutation) replace the worst members if better. All evaluations count against Q."""
    D = prob.D
    pop, fit = _init(prob, K, rng, seed_x)
    T = max(1, (Q - prob.n) // K); t = 0
    while prob.n < Q:
        prob.gen(pop)
        a = max(0.0, 2 * (1 - t / T)); t += 1
        leaders = pop[np.argsort(fit)[:3]].copy()
        nA = int(round(rho * K)); nG = min(K - nA, Q - prob.n)
        sel = rng.permutation(K)[:nG]
        Xs = []
        for Ld in leaders:
            A = 2 * a * rng.random((nG, D)) - a
            C = 2 * rng.random((nG, D))
            Xs.append(Ld - A * np.abs(C * Ld - pop[sel]))
        new = np.clip(sum(Xs) / 3, 0, 1)
        for k, i in enumerate(sel):
            pop[i] = new[k]; fit[i] = prob(new[k])
        nA = min(nA, Q - prob.n)
        if nA > 0:
            ch = _tourn(pop, fit, nA, rng)
            if use_x:
                p2 = _tourn(pop, fit, nA, rng)
                al = rng.random((nA, 1)); cx = rng.random((nA, 1)) < pc
                ch = np.where(cx, al * ch + (1 - al) * p2, ch)
            if use_m:
                ch = ch + (rng.random(ch.shape) < pm) * rng.normal(0, mu, ch.shape)
            ch = np.clip(ch, 0, 1)
            for k, i in enumerate(np.argsort(fit)[::-1][:nA]):
                v = prob(ch[k])
                if v < fit[i]:
                    pop[i] = ch[k]; fit[i] = v


def pso(prob, K, Q, rng, seed_x=None, w=0.72, c1=1.49, c2=1.49, **_):
    pop, fit = _init(prob, K, rng, seed_x)
    v = rng.uniform(-0.1, 0.1, pop.shape); pb, pf = pop.copy(), fit.copy()
    g = pb[np.argmin(pf)].copy()
    while prob.n < Q:
        prob.gen(pop)
        v = np.clip(w * v + c1 * rng.random(pop.shape) * (pb - pop)
                    + c2 * rng.random(pop.shape) * (g - pop), -0.2, 0.2)
        pop = np.clip(pop + v, 0, 1)
        for i in range(K):
            if prob.n >= Q:
                break
            fit[i] = prob(pop[i])
            if fit[i] < pf[i]:
                pb[i], pf[i] = pop[i], fit[i]
        g = pb[np.argmin(pf)].copy()


def de(prob, K, Q, rng, seed_x=None, F=0.5, CR=0.9, **_):
    D = prob.D
    pop, fit = _init(prob, K, rng, seed_x)
    while prob.n < Q:
        prob.gen(pop)
        for i in range(K):
            if prob.n >= Q:
                break
            r = rng.choice([j for j in range(K) if j != i], 3, replace=False)
            mut = pop[r[0]] + F * (pop[r[1]] - pop[r[2]])
            cr = rng.random(D) < CR; cr[rng.integers(D)] = True
            tr = np.clip(np.where(cr, mut, pop[i]), 0, 1)
            v = prob(tr)
            if v <= fit[i]:
                pop[i], fit[i] = tr, v


def woa(prob, K, Q, rng, seed_x=None, b=1.0, **_):
    pop, fit = _init(prob, K, rng, seed_x)
    best = prob.bx.copy()
    T = max(1, (Q - prob.n) // K); t = 0
    while prob.n < Q:
        prob.gen(pop)
        a = 2 * (1 - t / T); t += 1
        for i in range(K):
            if prob.n >= Q:
                break
            A = 2 * a * rng.random() - a; C = 2 * rng.random()
            if rng.random() < 0.5:
                ref = best if abs(A) < 1 else pop[rng.integers(K)]
                new = ref - A * np.abs(C * ref - pop[i])
            else:
                l = rng.uniform(-1, 1)
                new = np.abs(best - pop[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            new = np.clip(new, 0, 1)
            pop[i], fit[i] = new, prob(new)
        best = prob.bx.copy()


BASE = {'RS': (rs, {}), 'GA': (ga, {}), 'GWO': (gwo_ga, dict(rho=0.0)), 'PSO': (pso, {}),
        'DE': (de, {}), 'WOA': (woa, {}), 'HYB': (gwo_ga, dict(rho=0.5)),
        'GWO+X': (gwo_ga, dict(rho=0.5, use_m=False)),
        'GWO+M': (gwo_ga, dict(rho=0.5, use_x=False))}


def parse(name):
    seeded = name.startswith('S-')
    n = name[2:] if seeded else name
    kw = {}
    if ':' in n:
        n, rest = n.split(':')
        for kv in rest.split(','):
            k, v = kv.split('='); kw[k] = float(v)
    fn, base = BASE[n]
    return fn, {**base, **kw}, seeded


# ----------------------------------------------------------------------------- jobs
def run_alg(name, I, Q, K, expid, run, salt=0, track=False):
    t0 = time.perf_counter()
    res = {}
    if name in ('CONS', 'EQUAL', 'LOCAL'):
        dec = {'CONS': constructive, 'EQUAL': equal_alloc,
               'LOCAL': lambda J: (np.zeros(J.N, int), np.zeros(J.N, int),
                                   np.full(J.N, P['Pmax']), np.zeros(J.N))}[name](I)
        m = evaluate_dec(I, *dec, full=True)
    else:
        fn, kw, seeded = parse(name)
        rng = np.random.default_rng(np.random.SeedSequence([expid, run, zlib.crc32(name.encode()), salt]))
        sx = encode(*constructive(I), I) if seeded else None
        prob = Problem(I, Q, track)
        fn(prob, K, Q, rng, seed_x=sx, **kw)
        m = evaluate_dec(I, *decode(prob.bx, I), full=True)
        m['J'] = prob.bf
        if track:
            res = dict(curve=prob.curve, div=prob.div)
    m['time'] = time.perf_counter() - t0
    return {**m, **res}


def job(t):
    rng = np.random.default_rng(np.random.SeedSequence([t['expid'], t['run'], t['N']]))
    I = Inst(t['N'], rng)
    I.cfg(t['M'], t['R'], t['Umax'], t.get('beta', 0.0), t.get('wE', 0.5))
    row = run_alg(t['alg'], I, t['Q'], t['K'], t['expid'], t['run'], t.get('salt', 0), t.get('track', False))
    tags = {k: v for k, v in t.items() if k not in ('expid',)}
    return {**tags, **row}


def T(expid, run, alg, N=30, Q=10000, K=100, cfg=None, **extra):
    M, R = 2, 8
    d = dict(expid=expid, run=run, alg=alg, N=N, Q=Q, K=K, M=M, R=R, Umax=3)
    d.update(cfg or {})
    d.update(extra)
    return d


def run_tasks(tasks, workers):
    if workers <= 1:
        return pd.DataFrame([job(t) for t in tasks])
    with ProcessPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(job, tasks, chunksize=4)))


# ----------------------------------------------------------------------------- statistics
def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(ps)):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def paired(df, a, b, val='J', key='alg', idx=('run',), margin=0.01):
    x = df[df[key] == a].set_index(list(idx))[val]
    y = df[df[key] == b].set_index(list(idx))[val]
    x, y = x.align(y, join='inner')
    d = (x - y).values
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
    tc = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    p1 = stats.wilcoxon(d + margin, alternative='greater').pvalue
    p2 = stats.wilcoxon(d - margin, alternative='less').pvalue
    return dict(a=a, b=b, n=n, mean_diff=d.mean(), ci_lo=d.mean() - tc * se, ci_hi=d.mean() + tc * se,
                p_raw=p, p_tost=max(p1, p2), margin=margin,
                dz=d.mean() / d.std(ddof=1) if n > 1 and d.std(ddof=1) > 0 else np.nan)


def pair_family(df, pairs, groupcols=(), **kw):
    rows = []
    groups = [((), df)] if not groupcols else list(df.groupby(list(groupcols)))
    for g, sub in groups:
        fam = [paired(sub, a, b, idx=('run',), **kw) for a, b in pairs]
        adj = holm([r['p_raw'] for r in fam])
        for r, ad in zip(fam, adj):
            r['p_holm'] = ad
            for c, v in zip(groupcols, g if isinstance(g, tuple) else (g,)):
                r[c] = v
            rows.append(r)
    return pd.DataFrame(rows)


def summ(df, by):
    d = df.copy()
    d['J_feas'] = d['J'].where(d['feasible'])
    agg = d.groupby(by).agg(J=('J', 'mean'), J_sd=('J', 'std'), J_feas=('J_feas', 'mean'),
                            feas=('feasible', 'mean'), E_mJ=('E_abs', 'mean'), L_ms=('L_abs', 'mean'),
                            En=('En', 'mean'), Ln=('Ln', 'mean'), Etx_mJ=('Etx', 'mean'),
                            off=('off_frac', 'mean'), time=('time', 'mean'), n=('J', 'size'))
    return agg.reset_index()


def save(df, name, drop=('curve', 'div')):
    os.makedirs(OUT, exist_ok=True)
    df.drop(columns=[c for c in drop if c in df.columns]).to_csv(f"{OUT}/{name}.csv", index=False)


def show(df, title):
    print(f"\n=== {title} ==="); print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


# ----------------------------------------------------------------------------- experiments
def E1(A):
    algs = ['RS', 'GA', 'GWO', 'PSO', 'DE', 'WOA', 'HYB', 'S-GWO', 'S-HYB', 'CONS', 'EQUAL', 'LOCAL']
    df = run_tasks([T(1, r, a, Q=A.q(1e4)) for r in range(A.runs) for a in algs], A.workers)
    save(df, 'E1_runs'); s = summ(df, ['alg']); save(s, 'E1_summary'); show(s, 'E1 default scenario')
    pairs = [('HYB', 'GWO'), ('PSO', 'GWO'), ('PSO', 'HYB'), ('GA', 'GWO'), ('S-GWO', 'GWO'),
             ('S-GWO', 'CONS'), ('CONS', 'RS'), ('CONS', 'WOA'), ('CONS', 'GA'), ('CONS', 'DE'),
             ('CONS', 'GWO'), ('CONS', 'PSO'), ('CONS', 'HYB')]
    pf = pair_family(df, pairs); save(pf, 'E1_pairs'); show(pf, 'E1 paired tests (Holm over the family)')
    J = s.set_index('alg')['J']
    best = J[['GA', 'GWO', 'PSO', 'DE', 'HYB']].min()
    print(f"\nShare of best-found reduction recovered by CONS: {(1 - J['CONS']) / (1 - best):.3f}")
    d = df.copy(); d['chk'] = 0.5 * d['En'] + 0.5 * d['Ln']
    print("J vs 0.5*En+0.5*Ln (penalty-free check):",
          d.groupby('alg').apply(lambda g: (g['J_unpen'] - g['chk']).abs().max()).max())


def E2(A):
    Qs = [2000, 5000, 10000, 20000, 50000, 100000]
    algs = ['GA', 'GWO', 'PSO', 'HYB', 'S-GWO']
    df = run_tasks([T(2, r, a, Q=A.q(Q), Qlab=Q) for r in range(A.runs) for Q in Qs for a in algs], A.workers)
    save(df, 'E2_runs'); s = summ(df, ['Qlab', 'alg']); save(s, 'E2_summary'); show(s, 'E2 budget sweep')
    pairs = [('HYB', 'GWO'), ('HYB', 'PSO'), ('PSO', 'GWO'), ('GA', 'GWO')]
    pf = pair_family(df, pairs, ['Qlab']); save(pf, 'E2_pairs')
    show(pf[['Qlab', 'a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost', 'dz']],
         'E2 pairwise (p_tost<0.05 => equivalent within +-0.01 J)')


def E3(A):
    base = ['GWO', 'GWO+X', 'GWO+M', 'HYB']
    sweep = ['HYB:rho=0.1', 'HYB:rho=0.25', 'HYB:mu=0.005', 'HYB:mu=0.05', 'HYB:pm=0.02',
             'GWO+M:mu=0.005', 'GWO+M:rho=0.1', 'S-GWO', 'S-HYB']
    tasks = [T(3, r, a, Q=A.q(1e4), track=True) for r in range(A.runs) for a in base]
    tasks += [T(3, r, a, Q=A.q(1e4)) for r in range(A.runs) for a in sweep]
    df = run_tasks(tasks, A.workers)
    s = summ(df, ['alg']); save(df, 'E3_runs'); save(s, 'E3_summary'); show(s, 'E3 ablation and hybrid sweeps')
    df['div_final'] = df['div'].apply(lambda v: v[-1] if isinstance(v, list) and v else np.nan)
    print("\nFinal diversity (mean pairwise distance):")
    print(df[df['alg'].isin(base)].groupby('alg')['div_final'].agg(['mean', 'std']))
    algs = [a for a in base + sweep if a != 'GWO']
    pf = pair_family(df, [(a, 'GWO') for a in algs]); save(pf, 'E3_pairs')
    show(pf[['a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost']], 'E3 vs GWO')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
        for a in base:
            sub = df[(df['alg'] == a)]
            c = np.mean([x for x in sub['curve']], axis=0)
            ax[0].plot(np.arange(1, len(c) + 1) * 100, c, label=a)
            dv = [x for x in sub['div'] if len(x)]
            L = min(len(x) for x in dv); ax[1].plot(np.mean([x[:L] for x in dv], axis=0), label=a)
        ax[0].set_xlabel('evaluations'); ax[0].set_ylabel('best-so-far J'); ax[0].legend(fontsize=6)
        ax[1].set_xlabel('generation'); ax[1].set_ylabel('population diversity')
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_convergence.pdf")
    except Exception as e:
        print("figure skipped:", e)


def E4(A):
    Ns = [10, 20, 30, 40, 50, 60]
    tasks = []
    for regime in ('fixed', 'scaled'):
        for N in Ns:
            M, R = (2, 8) if regime == 'fixed' else scaled_RM(N)
            Q = A.q(1e4 * N / 30)
            for name, cfg in (('NOMA', dict(M=M, R=R, Umax=3)), ('OMA', dict(M=M, R=R, Umax=1)),
                              ('OMA3R', dict(M=M, R=3 * R, Umax=1))):
                for alg in ('GWO', 'CONS'):
                    tasks += [T(4, r, alg, N=N, Q=Q, cfg=cfg, scheme=name, regime=regime, beta=0.0)
                              for r in range(A.runs)]
    for beta in (0.05, 0.1, 0.2):
        tasks += [T(4, r, 'GWO', N=30, Q=A.q(1e4), cfg=dict(M=2, R=8, Umax=3), scheme='NOMA',
                    regime='fixed', beta=beta) for r in range(A.runs)]
    df = run_tasks(tasks, A.workers); save(df, 'E4_runs')
    s = summ(df, ['regime', 'N', 'alg', 'scheme', 'beta']); save(s, 'E4_summary'); show(s, 'E4 NOMA vs OMA')
    rows = []
    for (reg, N, alg), g in df[df['beta'] == 0.0].groupby(['regime', 'N', 'alg']):
        mm = g.groupby(['scheme']).agg(J=('J', 'mean'), E=('E_abs', 'mean'))
        for ref in ('OMA', 'OMA3R'):
            pe = paired(g.assign(alg=g['scheme']), 'NOMA', ref, idx=('run',))
            rows.append(dict(regime=reg, N=N, alg=alg, ref=ref, J_NOMA=mm.loc['NOMA', 'J'], J_ref=mm.loc[ref, 'J'],
                             dJ_pct=100 * (mm.loc[ref, 'J'] - mm.loc['NOMA', 'J']) / mm.loc[ref, 'J'],
                             dE_pct=100 * (mm.loc[ref, 'E'] - mm.loc['NOMA', 'E']) / mm.loc[ref, 'E'],
                             p_raw=pe['p_raw']))
    d = pd.DataFrame(rows); d['p_holm'] = holm(d['p_raw']); save(d, 'E4_gap'); show(d, 'E4 NOMA gain (positive = NOMA better)')
    sb = df[(df['N'] == 30) & (df['regime'] == 'fixed') & (df['alg'] == 'GWO')]
    print("\nImperfect SIC (beta) at N=30 (OMA unaffected):")
    print(sb.groupby(['scheme', 'beta'])['J'].mean())
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for reg, ls in (('fixed', '--'), ('scaled', '-')):
            for sch in ('NOMA', 'OMA', 'OMA3R'):
                sub = s[(s.regime == reg) & (s.alg == 'GWO') & (s.scheme == sch) & (s.beta == 0.0)]
                ax.plot(sub['N'], sub['J'], ls, marker='o', ms=3, label=f"{sch} ({reg})")
        ax.set_xlabel('N'); ax.set_ylabel('J'); ax.legend(fontsize=5); fig.tight_layout()
        fig.savefig(f"{OUT}/fig_noma_oma.pdf")
    except Exception as e:
        print("figure skipped:", e)


def E5(A):
    tasks = []
    for N in (60, 100):
        M, R = scaled_RM(N)
        for mode, Q in (('fixed', 1e4), ('scaled', 1e4 * N / 30)):
            for alg in ('GA', 'GWO', 'PSO', 'HYB', 'S-GWO', 'CONS'):
                tasks += [T(5, r, alg, N=N, Q=A.q(Q), cfg=dict(M=M, R=R, Umax=3), budget=mode)
                          for r in range(A.runs)]
    df = run_tasks(tasks, A.workers); save(df, 'E5_runs')
    s = summ(df, ['N', 'budget', 'alg']); save(s, 'E5_summary'); show(s, 'E5 scalability (J_feas = feasible runs only)')


def E6(A):
    Ks = [10, 20, 30, 50, 100, 200, 400]
    tasks = [T(6, r, a, K=K, Q=A.q(1e4)) for r in range(A.runs) for K in Ks for a in ('GA', 'GWO', 'PSO', 'DE', 'HYB')]
    df = run_tasks(tasks, A.workers); save(df, 'E6_runs')
    s = summ(df, ['K', 'alg']); save(s, 'E6_summary'); show(s, 'E6 population-size sweep')
    pf = pair_family(df, [('HYB', 'GWO'), ('GA', 'GWO')], ['K']); save(pf, 'E6_pairs')
    show(pf[['K', 'a', 'b', 'mean_diff', 'p_raw', 'p_holm', 'p_tost']], 'E6 tests')


def E7(A):
    tasks = []
    for N in (8, 10, 30):
        M, R = scaled_RM(N) if N != 30 else (2, 8)
        cfg = dict(M=M, R=R, Umax=3)
        for r in range(A.runs):
            for k in range(A.restarts):
                tasks.append(T(7, r, 'S-GWO', N=N, Q=A.q(A.refQ), cfg=cfg, salt=k + 1, kind='ref'))
            for a in ('RS', 'GA', 'GWO', 'PSO', 'DE', 'HYB', 'S-GWO', 'CONS'):
                tasks.append(T(7, r, a, N=N, Q=A.q(1e4), cfg=cfg, kind='alg'))
    df = run_tasks(tasks, A.workers); save(df, 'E7_runs')
    bk = df.groupby(['N', 'run'])['J'].min().rename('J_bk').reset_index()
    a = df[df['kind'] == 'alg'].merge(bk, on=['N', 'run'])
    a['gap_pct'] = 100 * (a['J'] - a['J_bk']) / a['J_bk']
    s = a.groupby(['N', 'alg']).agg(J=('J', 'mean'), J_bk=('J_bk', 'mean'), gap_pct=('gap_pct', 'mean')).reset_index()
    save(s, 'E7_gap'); show(s, 'E7 gap to best-known (best-known is an upper bound on the optimum, not a bound)')


def E8(A):
    tasks = [T(8, r, a, cfg=dict(M=M, R=8, Umax=3), Mlab=M, Q=A.q(1e4))
             for r in range(A.runs) for M in (1, 2, 4) for a in ('GA', 'GWO', 'PSO', 'HYB', 'CONS')]
    df = run_tasks(tasks, A.workers); save(df, 'E8_runs')
    s = summ(df, ['Mlab', 'alg']); save(s, 'E8_summary'); show(s, 'E8 fog-node sweep')
    pf = pair_family(df, [('HYB', 'GWO'), ('PSO', 'GWO'), ('GA', 'GWO')], ['Mlab']); save(pf, 'E8_pairs')
    show(pf[['Mlab', 'a', 'b', 'mean_diff', 'p_raw', 'p_holm']], 'E8 tests')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='all'); ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--quick', action='store_true', help='budgets /5 for smoke tests')
    ap.add_argument('--restarts', type=int, default=10); ap.add_argument('--refQ', type=float, default=2e5)
    A = ap.parse_args()
    A.q = (lambda q: max(200, int(q / 5))) if A.quick else (lambda q: int(q))
    os.makedirs(OUT, exist_ok=True)
    ex = {'E1': E1, 'E2': E2, 'E3': E3, 'E4': E4, 'E5': E5, 'E6': E6, 'E7': E7, 'E8': E8}
    for e in (ex if A.exp == 'all' else A.exp.split(',')):
        t0 = time.time(); ex[e](A); print(f"[{e}] done in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
