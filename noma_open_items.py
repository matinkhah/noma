#!/usr/bin/env python3
"""
=============================================================================
 noma_open_items.py -- greedy-GWO reruns and the remaining open items
=============================================================================
Builds on noma_e9_e12.py (must sit in the same directory): same model, same optimisers, same instance
streams, same statistics (paired Wilcoxon, Holm, TOST).  The greedy GWO is `G-GWO` in that file (a moved wolf
replaces its parent only if it is not worse); `S-G-GWO` is G-GWO with the constructive solution injected.

A. RERUNS WITH GREEDY GWO  (experiment ids and instance streams as in run_experiments_v3.py)
   E2  budget sweep Q = 2e3..1e5        + G-GWO, S-G-GWO; pairs against G-GWO; change of J with the budget
   E5  scalability N = 60, 100          + G-GWO, S-G-GWO; feasibility and share of runs beating all-local
   E6  population size K = 10..400      + G-GWO; best K per method
   E7  best-known reference             restarts are S-G-GWO (greedy) instead of S-GWO; gap of every method
   E8  fog nodes M = 1, 2, 4            + G-GWO
   Legacy algorithms (GWO, S-GWO, ...) are rerun with the original seeds, so their rows reproduce the paper
   (checked against the paper's tables: EXACT / OK / CHECK).

B. OPEN ITEMS (new experiment ids -- rename freely)
   E13 tuned hybrid vs PSO at PSO's best K:  K (and mu) selected on disjoint instances, evaluated on the E6 instances
   E14 ablation of the constructive rule with searched power (power fixed / searched at admission / + coordinate
       refinement on the true objective); one component changed at a time; N = 30 and N = 100
   E15 imperfect SIC beta beyond 0.05, with and without Rayleigh fading; NOMA vs OMA vs OMA3R; break-even beta
   E16 partial offloading (fraction phi_i in [0,1]) under the system-level cost at xi >= 0.5 (omega = 1):
       binary vs partial vs partial warm-started from the binary solution, same total budget

USAGE
  python noma_open_items.py --exp E2 --runs 30 --workers 8
  python noma_open_items.py --exp GREEDY --workers 8        # E2,E5,E6,E7,E8
  python noma_open_items.py --exp OPEN   --workers 8        # E13..E16
  python noma_open_items.py --exp all    --workers 8
  python noma_open_items.py --exp all --runs 2 --quick --restarts 2 --refQ 2000 --workers 1     # smoke test
Flags: --reuse (reuse CSVs), --out DIR (default results_open), --ref-algs S-G-GWO[,S-GWO] (E7 restarts),
       --restarts 10 --refQ 2e5, --ks, --mus, --budgets (E13), --betas, --beta-N (E15), --cons-N (E14), --xis (E16)
Requirements: numpy, scipy, pandas.   Output: ./results_open/*.csv
=============================================================================
"""
import argparse, math, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import noma_e9_e12 as B

OUT = "results_open"
PG = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]) * B.P['Pmax']      # power grid of the searched-power variants


# ============================================================================= generic helpers
def run_custom(fn, tasks, A, name):
    """Like noma_e9_e12.get_runs but with an own job function."""
    path = f"{B.OUT}/{name}_runs.csv"
    if A.reuse and os.path.exists(path):
        print(f"[{name}] reusing {path}")
        return pd.read_csv(path)
    n = len(tasks); rows = []; t0 = time.time(); step = max(1, n // 20)
    print(f"[{name}] {n} jobs, workers={A.workers}", flush=True)
    if A.workers <= 1:
        for i, t in enumerate(tasks):
            rows.append(fn(t))
            if (i + 1) % step == 0:
                print(f"[{name}] {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(A.workers) as ex:
            futs = [ex.submit(fn, t) for t in tasks]
            for i, f in enumerate(as_completed(futs)):
                rows.append(f.result())
                if (i + 1) % step == 0:
                    print(f"[{name}] {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    return df


def legacy_check(A, table, df, ref_table, key_cols, algs, dec=3, feas_col=None):
    """Compare the legacy (non-greedy) rows of a rerun with the paper's table: same seeds -> EXACT."""
    if A.quick:
        return
    for key, per_alg in ref_table.items():
        cond = key if isinstance(key, tuple) else (key,)
        sub = df
        for c, v in zip(key_cols, cond):
            sub = sub[sub[c] == v]
        for a in algs:
            if a in per_alg and len(sub[sub.alg == a]):
                ref = per_alg[a]
                B.chk(table, f"{key} {a}", sub[sub.alg == a]['J'], ref[0], ref[1], dec)


def deterioration(df, algs, col, pairs, name):
    det = []
    for a in algs:
        for q0, q1 in pairs:
            x = df[(df.alg == a) & (df[col] == q1)].set_index('run')['J']
            y = df[(df.alg == a) & (df[col] == q0)].set_index('run')['J']
            x, y = x.align(y, join='inner'); d = (x - y).values; n = len(d)
            se = d.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
            tc = B.stats.t.ppf(.975, n - 1) if n > 1 else np.nan
            det.append(dict(alg=a, **{f'from_{col}': q0, f'to_{col}': q1}, dJ=d.mean(),
                            ci_lo=d.mean() - tc * se, ci_hi=d.mean() + tc * se, p_raw=B.wil(d)))
    dd = pd.DataFrame(det); dd['p_holm'] = B.holm(dd['p_raw'])
    dd['verdict'] = np.where((dd.dJ > 0.005) & (dd.p_holm < 0.05), 'DETERIORATES',
                             np.where((dd.dJ < -0.005) & (dd.p_holm < 0.05), 'improves', 'no significant change'))
    B.save(dd, name); B.show(dd, f'{name}: change of J when the budget grows (positive = worse; Holm over all rows)')


# ============================================================================= A. greedy reruns
def E2(A):
    Qs = [2000, 5000, 10000, 20000, 50000, 100000]
    algs = ['GA', 'GWO', 'G-GWO', 'PSO', 'HYB', 'S-GWO', 'S-G-GWO']
    df = B.get_runs(A, 'E2', [B.T(2, r, a, Q=A.q(Q), Qlab=Q) for r in range(A.runs) for Q in Qs for a in algs])
    legacy_check(A, 'Table Q (rerun)', df, B.REF['TQ'], ['Qlab'], ['GA', 'GWO', 'PSO', 'HYB', 'S-GWO'])
    s = B.summ(df, ['Qlab', 'alg']); B.save(s, 'E2_summary')
    B.show(s, 'E2 budget sweep with greedy GWO', ['Qlab', 'alg', 'J', 'J_sd', 'feas', 'n'])
    pairs = [('G-GWO', 'GWO'), ('HYB', 'G-GWO'), ('PSO', 'G-GWO'), ('GA', 'G-GWO'), ('HYB', 'PSO'), ('S-G-GWO', 'G-GWO')]
    pf = B.family(df, pairs, ['Qlab']); B.save(pf, 'E2_pairs')
    B.show(pf, 'E2 paired differences (first minus second; Holm over 6 per budget; p_tost<0.05 => equivalent within 0.01 J)',
           ['Qlab', 'a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost'])
    deterioration(df, ['GWO', 'G-GWO', 'S-GWO', 'S-G-GWO', 'HYB', 'PSO'], 'Qlab', [(10000, 50000), (10000, 100000)], 'E2_deterioration')


def E5(A):
    algs = ['GA', 'GWO', 'G-GWO', 'PSO', 'HYB', 'S-GWO', 'S-G-GWO', 'CONS']; tasks = []
    for N in (60, 100):
        M, R = B.scaled_RM(N)
        for mode, Q in (('fixed', 1e4), ('scaled', 1e4 * N / 30)):
            for a in algs:
                tasks += [B.T(5, r, a, N=N, Q=A.q(Q), cfg=dict(M=M, R=R, Umax=3), budget=mode) for r in range(A.runs)]
    df = B.get_runs(A, 'E5', tasks)
    if not A.quick:
        for (N, mode), ref in B.REF['TS'].items():
            for a in ['GA', 'GWO', 'PSO', 'HYB', 'S-GWO', 'CONS']:
                g = df[(df.N == N) & (df.budget == mode) & (df.alg == a)]
                if len(g):
                    B.chk('Table S (rerun)', f'N={N} {mode} {a} J', g['J'], ref[a][0], ref[a][1], 3)
    df['beat_local'] = df['J'] < 1 - 1e-12
    s = B.summ(df, ['N', 'budget', 'alg']); s = s.merge(df.groupby(['N', 'budget', 'alg'])['beat_local'].mean().reset_index(), on=['N', 'budget', 'alg'])
    B.save(s, 'E5_summary'); B.show(s, 'E5 scalability with greedy GWO (J_feas = feasible runs only; beat_local = share of runs with J<1)',
                                    ['N', 'budget', 'alg', 'J', 'J_sd', 'J_feas', 'feas', 'beat_local', 'n'])
    pairs = [('G-GWO', 'GWO'), ('HYB', 'G-GWO'), ('PSO', 'G-GWO'), ('GA', 'G-GWO'), ('S-G-GWO', 'G-GWO'), ('CONS', 'G-GWO')]
    pf = B.family(df, pairs, ['N', 'budget']); B.save(pf, 'E5_pairs')
    B.show(pf, 'E5 paired differences (first minus second; Holm over 6 per cell)', ['N', 'budget', 'a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm'])


def E6(A):
    Ks = [10, 20, 30, 50, 100, 200, 400]; algs = ['GA', 'GWO', 'G-GWO', 'PSO', 'DE', 'HYB']
    df = B.get_runs(A, 'E6', [B.T(6, r, a, K=K, Q=A.q(1e4)) for r in range(A.runs) for K in Ks for a in algs])
    legacy_check(A, 'Table K (rerun)', df, B.REF['TK'], ['K'], ['DE', 'GA', 'GWO', 'HYB', 'PSO'])
    s = B.summ(df, ['K', 'alg']); B.save(s, 'E6_summary'); B.show(s, 'E6 population-size sweep with greedy GWO', ['K', 'alg', 'J', 'J_sd', 'feas', 'n'])
    pairs = [('G-GWO', 'GWO'), ('HYB', 'G-GWO'), ('PSO', 'G-GWO'), ('GA', 'G-GWO'), ('DE', 'G-GWO'), ('HYB', 'PSO')]
    pf = B.family(df, pairs, ['K']); B.save(pf, 'E6_pairs')
    B.show(pf, 'E6 tests (Holm over 6 per K)', ['K', 'a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost'])
    best = s.loc[s.groupby('alg')['J'].idxmin(), ['alg', 'K', 'J']].reset_index(drop=True); B.save(best, 'E6_bestK')
    B.show(best, 'E6 best K per method (descriptive: K chosen on the same instances)')
    print(f"Spread of the per-method best J over {', '.join(best.alg)}: {best.J.max() - best.J.min():.4f}")


def E7(A):
    refalgs = A.ref_algs.split(','); ALG = ['RS', 'GA', 'GWO', 'G-GWO', 'PSO', 'DE', 'HYB', 'S-GWO', 'S-G-GWO', 'CONS']; tasks = []
    for N in (8, 10, 30):
        M, R = B.scaled_RM(N) if N != 30 else (2, 8)
        cfg = dict(M=M, R=R, Umax=3)
        for r in range(A.runs):
            for ra in refalgs:
                tasks += [B.T(7, r, ra, N=N, Q=A.q(A.refQ), cfg=cfg, salt=k + 1, kind='ref', refalg=ra) for k in range(A.restarts)]
            tasks += [B.T(7, r, a, N=N, Q=A.q(1e4), cfg=cfg, kind='alg', refalg='') for a in ALG]
    df = B.get_runs(A, 'E7', tasks)
    bk = df.groupby(['N', 'run'])['J'].min().rename('J_bk').reset_index()          # as in the paper: restarts and compared runs
    rm = df[df.kind == 'ref'].groupby(['N', 'run', 'refalg'])['J'].min().unstack('refalg').add_prefix('J_ref_').reset_index()
    a = df[df.kind == 'alg'].merge(bk, on=['N', 'run']); a['gap_pct'] = 100 * (a['J'] - a['J_bk']) / a['J_bk']
    s = a.groupby(['N', 'alg']).agg(J=('J', 'mean'), J_bk=('J_bk', 'mean'), gap_pct=('gap_pct', 'mean'), gap_sd=('gap_pct', 'std')).reset_index()
    B.save(s, 'E7_gap'); B.show(s, f'E7 gap to best-known (reference restarts: {", ".join(refalgs)}; J_bk is an upper bound on the optimum)')
    rr = df[df.kind == 'ref'].groupby(['N', 'refalg']).agg(J_restart_mean=('J', 'mean'), J_restart_sd=('J', 'std')).reset_index()
    rr = rr.merge(rm.groupby('N')[[c for c in rm.columns if c.startswith('J_ref_')]].mean().reset_index(), on='N')
    rr = rr.merge(bk.groupby('N')['J_bk'].mean().reset_index(), on='N'); B.save(rr, 'E7_reference')
    B.show(rr, 'E7 quality of the reference restarts (J_ref_<alg> = mean over instances of the best restart of that type)')
    if len(refalgs) > 1:
        for x in refalgs[1:]:
            d = (rm[f'J_ref_{x}'] - rm[f'J_ref_{refalgs[0]}']).groupby(rm['N']).mean()
            print(f"Mean best-restart J of {x} minus {refalgs[0]} by N:", d.round(5).to_dict())


def E8(A):
    algs = ['GA', 'GWO', 'G-GWO', 'PSO', 'HYB', 'CONS']
    df = B.get_runs(A, 'E8', [B.T(8, r, a, cfg=dict(M=M, R=8, Umax=3), Mlab=M, Q=A.q(1e4)) for r in range(A.runs) for M in (1, 2, 4) for a in algs])
    legacy_check(A, 'Table M (rerun)', df, B.REF['TM'], ['Mlab'], ['CONS', 'GA', 'GWO', 'HYB', 'PSO'])
    s = B.summ(df, ['Mlab', 'alg']); B.save(s, 'E8_summary'); B.show(s, 'E8 fog-node sweep with greedy GWO', ['Mlab', 'alg', 'J', 'J_sd', 'feas', 'n'])
    pairs = [('G-GWO', 'GWO'), ('HYB', 'G-GWO'), ('PSO', 'G-GWO'), ('GA', 'G-GWO'), ('HYB', 'PSO')]
    pf = B.family(df, pairs, ['Mlab']); B.save(pf, 'E8_pairs')
    B.show(pf, 'E8 tests (Holm over 5 per M)', ['Mlab', 'a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost'])


# ============================================================================= E13 tuned hybrid vs PSO at its best K
def E13(A):
    Ks = [int(k) for k in A.ks.split(',')]; mus = [float(m) for m in A.mus.split(',')]; OFF = 1000; allp = []
    for Qraw in [float(q) for q in A.budgets.split(',')]:
        Q = int(Qraw); Qe = A.q(Q)
        sel = []
        for r in range(A.runs):
            for K in Ks:
                for al in ('PSO', 'G-GWO'):
                    sel.append(B.T(131, OFF + r, al, K=K, Q=Qe, Qlab=Q, inst_exp=13, fam=al, mu=np.nan))
                for m in mus:
                    sel.append(B.T(131, OFF + r, f'HYB:mu={m}', K=K, Q=Qe, Qlab=Q, inst_exp=13, fam='HYB', mu=m))
        ds = B.get_runs(A, f'E13_select_Q{Q}', sel)
        sg = ds.groupby(['fam', 'K', 'mu'], dropna=False)['J'].agg(['mean', 'std']).reset_index(); B.save(sg, f'E13_selection_Q{Q}')
        B.show(sg, f'E13 selection instances (disjoint seeds), Q={Q}: mean J per configuration')
        bp = int(sg[sg.fam == 'PSO'].sort_values('mean').iloc[0]['K']); bg = int(sg[sg.fam == 'G-GWO'].sort_values('mean').iloc[0]['K'])
        h = sg[sg.fam == 'HYB'].sort_values('mean').iloc[0]; bh, bm = int(h['K']), float(h['mu'])
        print(f"Q={Q}: selected PSO K*={bp}, G-GWO K*={bg}, hybrid K*={bh}, mu*={bm}")
        cfgs = [('PSO[K*]', 'PSO', bp), ('G-GWO[K*]', 'G-GWO', bg), ('HYB[K*,mu*]', f'HYB:mu={bm}', bh),
                ('HYB[mu*,K=PSO K*]', f'HYB:mu={bm}', bp), ('PSO[K=100]', 'PSO', 100), ('GWO[K=100]', 'GWO', 100),
                ('G-GWO[K=100]', 'G-GWO', 100), ('HYB[default]', 'HYB', 100), ('HYB[mu*,K=100]', f'HYB:mu={bm}', 100)]
        dt = B.get_runs(A, f'E13_test_Q{Q}', [B.T(132, r, nm, K=K, Q=Qe, Qlab=Q, inst_exp=6, lab=lab) for r in range(A.runs) for lab, nm, K in cfgs])
        s = dt.groupby('lab')['J'].agg(['mean', 'std', 'count']).reset_index(); B.save(s, f'E13_test_summary_Q{Q}')
        B.show(s, f'E13 test instances (E6 stream), Q={Q}')
        pairs = [('HYB[K*,mu*]', 'PSO[K*]'), ('HYB[mu*,K=PSO K*]', 'PSO[K*]'), ('HYB[K*,mu*]', 'G-GWO[K*]'), ('PSO[K*]', 'G-GWO[K*]'),
                 ('HYB[K*,mu*]', 'HYB[default]'), ('PSO[K*]', 'PSO[K=100]'), ('HYB[mu*,K=100]', 'PSO[K=100]'), ('HYB[mu*,K=100]', 'PSO[K*]')]
        pf = B.family(dt, pairs, key='lab'); pf['Qlab'] = Q; allp.append(pf)
        B.show(pf, f'E13 paired differences (first minus second; Holm over {len(pairs)}), Q={Q}', ['a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost'])
        r0 = pf.iloc[0]
        v = ("tuned hybrid SIGNIFICANTLY BETTER than PSO at its best K" if (r0.mean_diff < 0 and r0.p_holm < 0.05) else
             "tuned hybrid SIGNIFICANTLY WORSE than PSO at its best K" if (r0.mean_diff > 0 and r0.p_holm < 0.05) else "no significant difference to PSO at its best K")
        print(f"Q={Q}: {v} (diff {r0.mean_diff:+.4f}, Holm p={r0.p_holm:.3g}, TOST p={r0.p_tost:.3g})")
    B.save(pd.concat(allp), 'E13_pairs')


# ============================================================================= E14 constructive rule with searched power
def _block_eval(I, users, pw, fs, admit):
    """Admission test of one block (users with powers pw and shares fs). Returns (ok, total gain = sum(1 - normalised cost))."""
    q = I.g[users] * pw; o = np.argsort(-q); us, qs, ps, fss = users[o], q[o], pw[o], fs[o]
    gam = B.sinr_sorted(qs, np.zeros(len(qs), int), 1, I.sig, I.beta)
    if admit != 'no_sic' and np.any(gam < B.P['gth']):
        return False, 0.0
    r = I.W * np.log2(1 + gam)
    Lu = I.S[us] / r + I.C[us] / fss + I.eps * I.S[us] / I.rdl[us]
    if np.any(Lu > B.P['Lmax']):
        return False, 0.0
    Eu = ps * I.S[us] / r + I.omega * I.kf * fss ** 2 * I.C[us]
    cost = I.wE * Eu / I.Eloc[us] + (1 - I.wE) * Lu / I.Lloc[us]
    if admit != 'no_cost' and np.any(cost >= 1):
        return False, 0.0
    return True, float((1 - cost).sum())


def cons2(I, order='gain', node='maxres', share='equal', block='least', admit='full', power='fixed', rng=None):
    """Parametrised constructive rule; the defaults (power='fixed') reproduce noma_e9_e12.constructive exactly.
    order  gain | gain_asc | load | random        device order (strongest channel first = paper)
    node   maxres | rr                             fog node (largest residual capacity = paper)
    share  equal | demand                          compute share (F/ceil(N/M) = paper; demand scales it with the task load)
    block  least | first | best                    block choice (least occupied = paper; best = largest gain increment)
    admit  full | no_cost | no_sic                 admission test (SIC + delay + cost<1 = paper)
    power  fixed | grid                            Pmax/2 = paper; grid = power chosen from PG at admission (max block gain)"""
    N, M, R = I.N, I.M, I.R; Pm = B.P['Pmax']
    a = np.zeros(N, int); b = np.zeros(N, int); p = np.full(N, Pm / 2); f = np.zeros(N)
    occ = [[] for _ in range(R)]; Fres = np.full(M, B.P['Fk']); eq = B.P['Fk'] / math.ceil(N / M); gold = np.zeros(R)
    seq = {'gain': lambda: np.argsort(-I.g), 'gain_asc': lambda: np.argsort(I.g), 'load': lambda: np.argsort(-I.Lloc),
           'random': lambda: (rng if rng is not None else np.random.default_rng(0)).permutation(N)}[order]()
    cand = [Pm / 2] if power == 'fixed' else list(PG); rr = 0; Cbar = I.C.mean()
    for i in seq:
        if node == 'maxres':
            k = int(np.argmax(Fres))
        else:
            k = rr % M; rr += 1
        fi = min(Fres[k], eq if share == 'equal' else float(np.clip(eq * I.C[i] / Cbar, 0.25 * eq, 2 * eq)))
        if fi <= 0:
            continue
        blocks = np.arange(R) if block == 'first' else np.argsort([len(o) for o in occ], kind='stable')
        best = None
        for bl in blocks:
            if len(occ[bl]) >= I.Umax:
                continue
            users = np.array(occ[bl] + [i]); fs = np.array([f[u] if u != i else fi for u in users]); bb = None
            for pi in cand:
                pw = np.array([p[u] if u != i else pi for u in users])
                ok, g = _block_eval(I, users, pw, fs, admit)
                if ok and (bb is None or g > bb[0]):
                    bb = (g, pi)
            if bb is None:
                continue
            inc = bb[0] - gold[bl]
            if block == 'best':
                if best is None or inc > best[0]:
                    best = (inc, bl, bb[1], bb[0])
            else:
                best = (inc, bl, bb[1], bb[0]); break
        if best is None:
            continue
        _, bl, pi, gnew = best
        occ[bl].append(i); a[i] = k + 1; b[i] = bl; f[i] = fi; p[i] = pi; Fres[k] -= fi; gold[bl] = gnew
    return a, b, p, f


def refine_power(I, a, b, p, f, sweeps=2):
    """Coordinate search of each offloader's power over PG on the TRUE objective (repair and penalty included)."""
    p = p.copy(); cur = B.evaluate_dec(I, a, b, p, f); off = np.nonzero(a > 0)[0]
    for _ in range(sweeps):
        improved = False
        for i in off[np.argsort(-I.g[off])]:
            keep = p[i]; bv, bp = cur, keep
            for pv in PG:
                if pv == keep:
                    continue
                p[i] = pv; v = B.evaluate_dec(I, a, b, p, f)
                if v < bv - 1e-12:
                    bv, bp = v, pv
            p[i] = bp
            if bp != keep:
                improved = True; cur = bv
        if not improved:
            break
    return p


VARIANTS = {'base': {}, 'order=gain_asc': dict(order='gain_asc'), 'order=random': dict(order='random'), 'order=load': dict(order='load'),
            'node=rr': dict(node='rr'), 'share=demand': dict(share='demand'), 'block=first': dict(block='first'),
            'block=best': dict(block='best'), 'admit=no_cost': dict(admit='no_cost'), 'admit=no_sic': dict(admit='no_sic')}
POWERS = ('fixed', 'search', 'search+refine')


def build_cons(I, variant, power, rng):
    dec = cons2(I, power='fixed' if power == 'fixed' else 'grid', rng=rng, **VARIANTS[variant])
    if power == 'search+refine':
        dec = (dec[0], dec[1], refine_power(I, *dec), dec[3])
    return dec


def job_cons(t):
    rng0 = np.random.default_rng(np.random.SeedSequence([t['inst_exp'], t['run'], t['N']]))
    I = B.Inst(t['N'], rng0); I.cfg(t['M'], t['R'], t['Umax'])
    t0 = time.perf_counter()
    if t['kind'] == 'cons':
        rng = np.random.default_rng(np.random.SeedSequence([t['expid'], t['run'], zlib.crc32(b'order')]))
        m = B.evaluate_dec(I, *build_cons(I, t['variant'], t['power'], rng), full=True)
    elif t['kind'] == 'opt':
        m = B.run_alg(t['alg'], I, t['Q'], t['K'], t['expid'], t['run'])
    else:                                                    # G-GWO seeded with the searched-power constructive solution
        fn, kw, _ = B.parse('G-GWO')
        rng = np.random.default_rng(np.random.SeedSequence([t['expid'], t['run'], zlib.crc32(b'seeded')]))
        prob = B.Problem(I, t['Q'])
        fn(prob, t['K'], t['Q'], rng, seed_x=B.encode(*build_cons(I, 'base', 'search+refine', rng), I), **kw)
        m = B.evaluate_dec(I, *B.decode(prob.bx, I), full=True); m['J'] = prob.bf
    m['time'] = time.perf_counter() - t0
    return {**{k: v for k, v in t.items() if k != 'expid'}, **m}


def E14(A):
    tasks = []
    for N in [int(x) for x in A.cons_N.split(',')]:
        M, R = B.scaled_RM(N); ie = 1 if N == 30 else 5; Q = A.q(1e4)
        base = dict(expid=14, N=N, M=M, R=R, Umax=3, Q=Q, K=100, inst_exp=ie)
        for r in range(A.runs):
            for pw in POWERS:
                for v in VARIANTS:
                    tasks.append(dict(base, run=r, kind='cons', variant=v, power=pw, lab=f'{pw}|{v}'))
            tasks.append(dict(base, run=r, kind='opt', alg='G-GWO', lab='G-GWO'))
            tasks.append(dict(base, run=r, kind='seeded', alg='S-G-GWO[search+refine]', lab='S-G-GWO[search+refine]'))
    df = run_custom(job_cons, tasks, A, 'E14')
    s = df.groupby(['N', 'lab']).agg(J=('J', 'mean'), J_sd=('J', 'std'), feas=('feasible', 'mean'), off=('off_frac', 'mean'),
                                     E_mJ=('E_abs', 'mean'), L_ms=('L_abs', 'mean')).reset_index()
    for N, g in s.groupby('N'):
        jg = float(g[g.lab == 'G-GWO']['J'].iloc[0]); s.loc[g.index, 'recovered'] = (1 - g['J']) / (1 - jg)
    B.save(s, 'E14_summary')
    for N, g in s.groupby('N'):
        c = g[g.lab.str.contains(r'\|')].copy(); c['power'] = c.lab.str.split('|').str[0]; c['variant'] = c.lab.str.split('|').str[1]
        print(f"\n=== E14 N={N}: mean J (feasible share in brackets); G-GWO reference J = {float(g[g.lab == 'G-GWO']['J'].iloc[0]):.4f} ===")
        pv = c.pivot(index='variant', columns='power', values='J')[list(POWERS)].reindex(list(VARIANTS))
        pf_ = c.pivot(index='variant', columns='power', values='feas')[list(POWERS)].reindex(list(VARIANTS))
        print(pd.concat({'J': pv, 'feas': pf_}, axis=1).to_string(float_format=lambda v: f'{v:.4f}'))
        print(g[g.lab.isin(['G-GWO', 'S-G-GWO[search+refine]'])][['lab', 'J', 'J_sd', 'feas', 'recovered']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    same = [(f'{pw}|{v}', f'{pw}|base') for pw in POWERS for v in VARIANTS if v != 'base']
    cross = [(f'{pw}|base', 'fixed|base') for pw in POWERS[1:]] + [(f'{pw}|{v}', f'fixed|{v}') for pw in POWERS[1:] for v in VARIANTS if v != 'base']
    ref = [('search+refine|base', 'G-GWO'), ('fixed|base', 'G-GWO'), ('S-G-GWO[search+refine]', 'G-GWO')]
    out = []
    for nm, pairs in (('component change, same power mode', same), ('searched power vs fixed, same component', cross), ('vs G-GWO', ref)):
        pf = B.family(df, pairs, ['N'], key='lab'); pf['family'] = nm; out.append(pf)
    po = pd.concat(out); B.save(po, 'E14_pairs')
    shown = po[(po.family == 'vs G-GWO') | (po.p_holm < 0.05)]
    B.show(shown, f'E14 paired differences (first minus second; Holm within each family and N); {len(shown)} of {len(po)} rows shown '
                  '(all G-GWO comparisons and every row with Holm p<0.05; full table in E14_pairs.csv)',
           ['N', 'family', 'a', 'b', 'mean_diff', 'p_raw', 'p_holm'])


# ============================================================================= E15 imperfect SIC and fading
def E15(A):
    betas = [float(x) for x in A.betas.split(',')]; Ns = [int(x) for x in A.beta_N.split(',')]; tasks = []
    for fd in (False, True):
        for N in Ns:
            M, R = B.scaled_RM(N); Q = A.q(1e4 * N / 30)
            for alg in ('G-GWO', 'CONS'):
                for sch, cfg in (('OMA', dict(M=M, R=R, Umax=1)), ('OMA3R', dict(M=M, R=3 * R, Umax=1))):
                    tasks += [B.T(15, r, alg, N=N, Q=Q, cfg=cfg, scheme=sch, beta=0.0, fading=fd, inst_exp=4) for r in range(A.runs)]
                for be in betas:
                    tasks += [B.T(15, r, alg, N=N, Q=Q, cfg=dict(M=M, R=R, Umax=3), scheme='NOMA', beta=be, fading=fd, inst_exp=4) for r in range(A.runs)]
    df = B.get_runs(A, 'E15', tasks); df['beta'] = df['beta'].astype(float)
    s = B.summ(df, ['fading', 'N', 'alg', 'scheme', 'beta']); B.save(s, 'E15_summary')
    rows = []
    for (fd, N, alg), g in df.groupby(['fading', 'N', 'alg']):
        refs = {sc: g[g.scheme == sc].set_index('run')['J'] for sc in ('OMA', 'OMA3R')}; sub = []
        for be in betas:
            x = g[(g.scheme == 'NOMA') & np.isclose(g.beta, be)].set_index('run')['J']
            for sc, y in refs.items():
                xx, yy = x.align(y, join='inner'); d = (yy - xx).values
                sub.append(dict(fading=fd, N=N, alg=alg, beta=be, ref=sc, J_NOMA=xx.mean(), J_ref=yy.mean(),
                                dJ_pct=100 * (yy.mean() - xx.mean()) / yy.mean(), p_raw=B.wil(d), n=len(d)))
        sd = pd.DataFrame(sub); sd['p_holm'] = B.holm(sd['p_raw']); rows.append(sd)
    gap = pd.concat(rows, ignore_index=True); B.save(gap, 'E15_gap')
    for (fd, N, alg), g in gap.groupby(['fading', 'N', 'alg']):
        print(f"\n=== E15 fading={fd} N={N} {alg}: NOMA gain in % of J over the orthogonal reference (positive = NOMA better; * = Holm p<0.05) ===")
        pv = g.pivot(index='beta', columns='ref', values='dJ_pct'); sg = g.pivot(index='beta', columns='ref', values='p_holm')
        print(pd.DataFrame({f'{c}': [f"{pv.loc[b, c]:+.2f}{'*' if sg.loc[b, c] < 0.05 else ' '}" for b in pv.index] for c in pv.columns}, index=pv.index).to_string())
    be_rows = []
    for (fd, N, alg, ref), g in gap.groupby(['fading', 'N', 'alg', 'ref']):
        g = g.sort_values('beta'); b, d = g['beta'].values, g['dJ_pct'].values
        if d[0] <= 0:
            bs = 0.0
        else:
            bs = np.inf
            for j in range(1, len(b)):
                if d[j] <= 0:
                    bs = b[j - 1] + (b[j] - b[j - 1]) * d[j - 1] / (d[j - 1] - d[j]); break
        be_rows.append(dict(fading=fd, N=N, alg=alg, ref=ref, gain_at_beta0=d[0], break_even_beta=bs))
    be = pd.DataFrame(be_rows); B.save(be, 'E15_break_even')
    B.show(be, f'E15 break-even beta (linear interpolation; inf = NOMA still better at beta={max(betas):g})')


# ============================================================================= E16 partial offloading
def decode_p(x, I):
    N = I.N
    a, b, p, f = B.decode(x[:4 * N], I); phi = x[4 * N:]
    return a, b, p, f, np.where(phi < 0.02, 0.0, np.where(phi > 0.98, 1.0, phi))


def evaluate_partial(I, a, b, p, f, phi, full=False):
    """Partial offloading: a fraction phi_i of the task (bits and cycles) is offloaded, the rest runs locally in parallel.
       L_i = max((1-phi) C/f_loc, phi S/r + phi C/f_fog + eps phi S/r_dl);  E_dev = (1-phi) E_loc + p phi S/r;
       E_fog = kf f^2 phi C.  Same repair, SIC and delay penalty as the binary model; phi=1 gives evaluate_dec exactly
       (the transmission is assumed to last the whole interference window of the block, as in the binary model)."""
    N = I.N; a = a.copy(); ph = np.where(a > 0, phi, 0.0); a = np.where(ph > 0, a, 0)
    L = I.Lloc.copy(); Edev = I.Eloc.copy(); Efog = np.zeros(N); Etx = np.zeros(N); pen = np.zeros(N)
    idx = np.nonzero(a > 0)[0]
    if idx.size:
        q = I.g[idx] * p[idx]; bb = b[idx]; o = np.lexsort((-q, bb)); io, qs, bs = idx[o], q[o], bb[o]
        first = np.searchsorted(bs, bs, side='left'); keep = (np.arange(len(bs)) - first) < I.Umax
        a[io[~keep]] = 0; ph[io[~keep]] = 0.0; io, qs, bs = io[keep], qs[keep], bs[keep]
        gam = B.sinr_sorted(qs, bs, I.R, I.sig, I.beta); r = I.W * np.log2(1 + gam); po = ph[io]
        ttx = po * I.S[io] / r; ks = a[io] - 1; fi = f[io].copy()
        ssum = np.bincount(ks, weights=fi, minlength=I.M); fi *= np.minimum(1.0, B.P['Fk'] / np.maximum(ssum[ks], 1e-9))
        Lo = np.maximum((1 - po) * I.C[io] / I.f[io], ttx + po * I.C[io] / fi + I.eps * po * I.S[io] / I.rdl[io])
        L[io] = Lo; Etx[io] = p[io] * ttx; Edev[io] = (1 - po) * I.Eloc[io] + Etx[io]; Efog[io] = I.kf * fi ** 2 * po * I.C[io]
        pen[io] = B.P['lam'] * (np.maximum(0, Lo - B.P['Lmax']) / B.P['Lmax'] + np.maximum(0, 1 - gam / B.P['gth']))
    En = (Edev + I.omega * Efog) / I.Eloc; Ln = L / I.Lloc; Ji = I.wE * En + (1 - I.wE) * Ln; J = float((Ji + pen).mean())
    if not full:
        return J
    off = a > 0; pp = ph[off]
    return dict(J=J, J_unpen=float(Ji.mean()), E_abs=float(Edev.mean() * 1e3), Efog_abs=float(Efog.mean() * 1e3), L_abs=float(L.mean() * 1e3),
                En=float((Edev / I.Eloc).mean()), Ln=float(Ln.mean()), Etx=float(Etx.sum() / N * 1e3), off_frac=float(off.mean()),
                feasible=bool(pen.max() <= 1e-12), Eloc_abs=float(I.Eloc.mean() * 1e3),
                mean_phi=float(pp.mean()) if pp.size else np.nan, frac_partial=float(((pp > 0) & (pp < 1)).mean()) if pp.size else 0.0)


class PProblem:
    def __init__(self, I, Q):
        self.I, self.Q, self.D = I, Q, 5 * I.N
        self.n, self.bf, self.bx = 0, np.inf, None

    def __call__(self, x):
        x = np.clip(x, 0, 1); v = evaluate_partial(self.I, *decode_p(x, self.I)); self.n += 1
        if v < self.bf:
            self.bf, self.bx = v, x.copy()
        return v


def job_part(t):
    rng0 = np.random.default_rng(np.random.SeedSequence([t['inst_exp'], t['run'], t['N']]))
    I = B.Inst(t['N'], rng0); I.cfg(t['M'], t['R'], t['Umax'], 0.0, 0.5, t['omega'], t['eps'], t['xi'])
    alg, arm, Q, K = t['alg'], t['arm'], t['Q'], t['K']; fn, kw, _ = B.parse(alg); t0 = time.perf_counter()
    sr = lambda salt: np.random.default_rng(np.random.SeedSequence([t['expid'], t['run'], zlib.crc32(alg.encode()), salt]))
    if arm == 'binary':
        prob = B.Problem(I, Q); fn(prob, K, Q, sr(0), **kw)
        m = B.evaluate_dec(I, *B.decode(prob.bx, I), full=True); m['J'] = prob.bf
        m['mean_phi'] = 1.0 if m['off_frac'] > 0 else np.nan; m['frac_partial'] = 0.0
    elif arm == 'partial':
        prob = PProblem(I, Q); fn(prob, K, Q, sr(1), **kw)
        m = evaluate_partial(I, *decode_p(prob.bx, I), full=True); m['J'] = prob.bf
    else:                                                   # warm: Q/2 binary, then Q/2 partial seeded with that solution
        Q1 = Q // 2; Q2 = Q - Q1
        pb = B.Problem(I, Q1); fn(pb, K, Q1, sr(2), **kw); a = B.decode(pb.bx, I)[0]
        prob = PProblem(I, Q2); fn(prob, K, Q2, sr(3), seed_x=np.concatenate([pb.bx, (a > 0).astype(float)]), **kw)
        m = evaluate_partial(I, *decode_p(prob.bx, I), full=True); m['J'] = prob.bf
    m['time'] = time.perf_counter() - t0
    return {**{k: v for k, v in t.items() if k != 'expid'}, **m}


def E16(A):
    xis = [float(x) for x in A.xis.split(',')]; algs = ['G-GWO', 'PSO']; Q = A.q(1e4)
    tasks = [dict(expid=16, run=r, N=30, M=2, R=8, Umax=3, Q=Q, K=100, inst_exp=1, omega=1, eps=A.eps16, xi=x, alg=a, arm=arm)
             for r in range(A.runs) for x in xis for a in algs for arm in ('binary', 'partial', 'warm')]
    df = run_custom(job_part, tasks, A, 'E16')
    s = B.summ(df, ['xi', 'alg', 'arm'])
    ex = df.groupby(['xi', 'alg', 'arm']).agg(mean_phi=('mean_phi', 'mean'), frac_partial=('frac_partial', 'mean')).reset_index()
    s = s.merge(ex, on=['xi', 'alg', 'arm']); s['beat_local'] = df.assign(b=df.J < 1 - 1e-12).groupby(['xi', 'alg', 'arm'])['b'].mean().values
    B.save(s, 'E16_summary')
    B.show(s, f'E16 system-level cost J_sys (omega=1, eps={A.eps16:g}); off = offloaded share, mean_phi = mean offloaded fraction, '
              'frac_partial = share of offloaders with 0<phi<1; sav_dev/sav_sys vs all-local',
           ['xi', 'alg', 'arm', 'J', 'J_sd', 'off', 'mean_phi', 'frac_partial', 'sav_dev', 'sav_sys', 'beat_local', 'feas', 'n'])
    pf = B.family(df, [('partial', 'binary'), ('warm', 'binary'), ('warm', 'partial')], ['xi', 'alg'], key='arm'); B.save(pf, 'E16_pairs')
    B.show(pf, 'E16 paired differences (first minus second; negative = first better; Holm over 3 per xi and method)',
           ['xi', 'alg', 'a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost'])
    for (x, a), g in pf.groupby(['xi', 'alg']):
        w = g[(g.a == 'warm') & (g.b == 'binary')].iloc[0]
        print(f"xi={x:g} {a}: warm-started partial minus binary = {w.mean_diff:+.4f} (Holm p={w.p_holm:.3g}) -> "
              + ("partial offloading helps" if (w.mean_diff < 0 and w.p_holm < 0.05) else
                 "partial offloading hurts" if (w.mean_diff > 0 and w.p_holm < 0.05) else "no significant effect"))


# ============================================================================= self test
def selftest():
    rng = np.random.default_rng(0)
    for it in range(30):                                      # partial model nests the binary one (phi = 1)
        I = B.Inst(30, np.random.default_rng(it), fading=bool(it % 2))
        I.cfg(2, 8, 3, beta=0.05 * (it % 3), omega=1, eps=0.1 * (it % 2), xi=[0.25, 0.5, 1.0][it % 3])
        x = rng.random(120); a, b, p, f = B.decode(x, I)
        v1 = B.evaluate_dec(I, a, b, p, f); v2 = evaluate_partial(I, a, b, p, f, (a > 0).astype(float))
        assert abs(v1 - v2) < 1e-9, (it, v1, v2)
        v3 = evaluate_partial(I, a, b, p, f, np.zeros(30)); assert abs(v3 - 1.0) < 1e-12      # phi = 0 -> all local
    for it in range(10):                                      # cons2 default == paper's constructive rule
        I = B.Inst(30, np.random.default_rng(it)); I.cfg(2, 8, 3, beta=0.05 * (it % 3))
        d1, d2 = B.constructive(I), cons2(I)
        assert all(np.array_equal(u, v) for u, v in zip(d1, d2)), it
    print("selftest ok: partial model reduces to the binary model; cons2 reproduces the constructive rule")


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description="Greedy-GWO reruns and open items for the NOMA study")
    ap.add_argument('--exp', default='all', help='comma list of E2,E5,E6,E7,E8,E13,E14,E15,E16 | GREEDY | OPEN | all')
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--quick', action='store_true', help='divide budgets by 5 (smoke test)')
    ap.add_argument('--reuse', action='store_true'); ap.add_argument('--out', default=OUT)
    ap.add_argument('--restarts', type=int, default=10); ap.add_argument('--refQ', type=float, default=2e5)
    ap.add_argument('--ref-algs', default='S-G-GWO', help='E7 reference restarts, e.g. S-G-GWO,S-GWO to compare with the old reference')
    ap.add_argument('--ks', default='10,20,30,50,100,200'); ap.add_argument('--mus', default='0.02,0.05,0.1,0.2')
    ap.add_argument('--budgets', default='1e4', help='E13 budgets, e.g. 1e4,1e5')
    ap.add_argument('--betas', default='0,0.05,0.1,0.15,0.2,0.3,0.5,1.0'); ap.add_argument('--beta-N', default='30,50')
    ap.add_argument('--cons-N', default='30,100'); ap.add_argument('--xis', default='0.5,1.0,2.0'); ap.add_argument('--eps16', type=float, default=0.0)
    A = ap.parse_args()
    B.OUT = A.out; os.makedirs(A.out, exist_ok=True)
    A.q = (lambda q: max(200, int(q / 5))) if A.quick else (lambda q: int(q))
    selftest()
    ex = {'E2': E2, 'E5': E5, 'E6': E6, 'E7': E7, 'E8': E8, 'E13': E13, 'E14': E14, 'E15': E15, 'E16': E16}
    sel = []
    for tok in A.exp.split(','):
        sel += {'GREEDY': ['E2', 'E5', 'E6', 'E7', 'E8'], 'OPEN': ['E13', 'E14', 'E15', 'E16'],
                'all': ['E14', 'E16', 'E15', 'E13', 'E8', 'E6', 'E5', 'E2', 'E7']}.get(tok, [tok])
    for e in sel:
        t0 = time.time(); ex[e](A); print(f"[{e}] done in {time.time() - t0:.0f}s", flush=True)
    if B.VAL:
        v = pd.DataFrame(B.VAL); B.save(v, 'validation_rerun')
        print(f"\nLegacy rows vs the paper's tables: {int((v.status == 'EXACT').sum())} exact, {int((v.status == 'OK').sum())} consistent, "
              f"{int((v.status == 'CHECK').sum())} to check (details in validation_rerun.csv)")


if __name__ == '__main__':
    main()
