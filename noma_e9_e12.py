#!/usr/bin/env python3
"""
=============================================================================
 noma_e9_e12.py -- self-contained script that produces the numbers still missing
 from the paper (experiments E9-E12) and validates the model against every
 table of the paper (V1-V8).
=============================================================================
Contents
  * model      uplink NOMA fog allocation with SIC; extensions: Rayleigh block fading,
               fog-side energy (J_sys), result-return delay
  * optimisers RS, GA, GWO, G-GWO (greedy), F-GWO (fixed schedule), PSO, DE, WOA, GWO-GA hybrid,
               constructive-seeded variants, constructive rule (CONS), equal allocation, all-local
  * E9  fading (default scenario with and without fading; NOMA vs OMA vs OMA3R)
  * E10 system-level cost J_sys, omega in {0,1}, epsilon in {0,0.1}, fog-energy sensitivity xi
  * E11 GWO diagnostics: G-GWO and F-GWO vs GWO, hybrid, PSO at Q = 1e4, 5e4, 1e5
  * E12 hybrid mutation step mu in {0.02,0.05,0.1,0.2}; mu selected on disjoint instances
  * V1..V8 re-run the baseline experiments (omega = eps = 0, no fading) with the ORIGINAL seeds and
           compare every number in Tables II, P, Q, QP, K, KP, M, MP, A, N, S, R with the paper:
           status EXACT (equal up to table rounding), OK (|z| <= 3 against the table's own sampling error)
           or CHECK.

Fixed choices for the two parameters left open in the paper (both documented, both overridable)
  kappa_f   fog energy constant  kappa_f = xi * kappa * (f_dev / f_ref)^2,  f_dev = 0.85 GHz (mean device clock),
            f_ref = F_k / ceil(N/M) (default compute share). xi = fog energy per cycle at the reference share,
            relative to the device energy per cycle at f_dev. Default xi = 0.25 (fog hardware assumed 4x more
            energy-efficient per cycle than a battery device); E10 reports xi in {0.1, 0.25, 0.5, 1}.
            With default parameters kappa_f = 1.02e-29.
  r_dl      downlink rate of device i on a dedicated orthogonal resource of width B_tot/N at fog power P_max:
            r_dl_i = (B_tot/N) * log2(1 + P_max * g_i / (N0 * B_tot/N)).   (g_i includes fading.)

USAGE
  python noma_e9_e12.py --exp E9,E10,E11,E12 --runs 30 --workers 8       # numbers for the paper
  python noma_e9_e12.py --exp V1 --workers 8                              # quickest sanity check (Table II, Table P)
  python noma_e9_e12.py --exp VAL --workers 8                             # all validations except the heaviest cells
  python noma_e9_e12.py --exp VAL --vheavy --workers 8                    # + Q = 5e4, 1e5 (V2) and E7 gaps (V7)
  python noma_e9_e12.py --exp all --workers 8                             # NEW (E9-E12) + VAL
  python noma_e9_e12.py --exp E9 --runs 2 --quick --workers 1             # smoke test
Other flags: --reuse (reuse CSVs in the output directory), --xi, --out DIR, --vruns N (validation runs, 30 = exact repro).
Requirements: numpy, scipy, pandas.   Output: ./results_ext/*.csv, validation.csv, paper_values.json
=============================================================================
"""
import argparse, copy, json, math, os, re, sys, time, zlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats

OUT = "results_ext"
DEFAULT_XI = 0.25
P = dict(Btot=10e6, Pmax=0.3, Lmax=1.0, gth=1.0, N0=10 ** (-174 / 10) * 1e-3,
         kappa=1e-28, lam=10.0, Fk=20e9)
F_DEV = 0.85e9

REF = {'T2': {'CONS': {'E': (16.6, 1.7), 'J': (0.89, 0.01), 'L': (314.0, 29.0)},
        'DE': {'E': (12.5, 1.4), 'J': (0.744, 0.013), 'L': (282.0, 26.0)},
        'EQUAL': {'E': (51.3, 8.2), 'J': (2.07, 0.34), 'L': (393.0, 40.0)},
        'GA': {'E': (15.5, 2.1), 'J': (0.813, 0.037), 'L': (273.0, 27.0)},
        'GWO': {'E': (11.2, 1.9), 'J': (0.696, 0.022), 'L': (269.0, 24.0)},
        'HYB': {'E': (11.4, 1.6), 'J': (0.696, 0.02), 'L': (265.0, 22.0)},
        'LOCAL': {'E': (20.3, 2.0), 'J': (1.0, 0.0), 'L': (334.0, 30.0)},
        'PSO': {'E': (10.9, 1.9), 'J': (0.693, 0.023), 'L': (274.0, 26.0)},
        'RS': {'E': (15.2, 1.7), 'J': (0.827, 0.018), 'L': (286.0, 31.0)},
        'S-GWO': {'E': (11.1, 1.6), 'J': (0.692, 0.017), 'L': (268.0, 23.0)},
        'S-HYB': {'E': (11.5, 1.5), 'J': (0.695, 0.026), 'L': (261.0, 24.0)},
        'WOA': {'E': (15.3, 1.7), 'J': (0.813, 0.021), 'L': (285.0, 26.0)}},
 'TA': {'GWO': (0.6958, 0.0178, None),
        'GWO+M': (0.7139, 0.02, 0.0181),
        'GWO+M:mu=0.005': (0.7208, 0.0241, 0.0249),
        'GWO+M:rho=0.1': (0.6981, 0.0234, 0.0023),
        'GWO+X': (0.697, 0.0244, 0.0011),
        'HYB': (0.6915, 0.0211, -0.0043),
        'HYB:mu=0.005': (0.6976, 0.0184, 0.0018),
        'HYB:mu=0.05': (0.6815, 0.016, -0.0144),
        'HYB:pm=0.02': (0.6943, 0.0172, -0.0016),
        'HYB:rho=0.1': (0.6956, 0.0197, -0.0003),
        'HYB:rho=0.25': (0.6892, 0.0215, -0.0066),
        'S-GWO': (0.6982, 0.0228, 0.0024),
        'S-HYB': (0.6919, 0.0171, -0.0039)},
 'TK': {10: {'DE': (0.769, 0.029), 'GA': (0.691, 0.024), 'GWO': (0.816, 0.015), 'HYB': (0.746, 0.028), 'PSO': (0.689, 0.024)},
        20: {'DE': (0.689, 0.02), 'GA': (0.719, 0.032), 'GWO': (0.78, 0.016), 'HYB': (0.707, 0.024), 'PSO': (0.678, 0.021)},
        30: {'DE': (0.703, 0.022), 'GA': (0.743, 0.029), 'GWO': (0.748, 0.016), 'HYB': (0.707, 0.021), 'PSO': (0.677, 0.022)},
        50: {'DE': (0.721, 0.022), 'GA': (0.769, 0.037), 'GWO': (0.724, 0.022), 'HYB': (0.697, 0.024), 'PSO': (0.685, 0.023)},
        100: {'DE': (0.745, 0.019), 'GA': (0.816, 0.04), 'GWO': (0.695, 0.025), 'HYB': (0.691, 0.02), 'PSO': (0.701, 0.024)},
        200: {'DE': (0.764, 0.018), 'GA': (0.838, 0.042), 'GWO': (0.686, 0.019), 'HYB': (0.696, 0.021), 'PSO': (0.72, 0.025)},
        400: {'DE': (0.791, 0.017), 'GA': (0.848, 0.028), 'GWO': (0.692, 0.019), 'HYB': (0.707, 0.022), 'PSO': (0.736, 0.022)}},
 'TKP': {10: [(-0.069, 4e-09), (-0.124, 4e-09)],
         20: [(-0.073, 4e-09), (-0.06, 4e-09)],
         30: [(-0.04, 4e-09), (-0.005, 0.54)],
         50: [(-0.027, 6e-07), (0.045, 6e-07)],
         100: [(-0.003, 0.67), (0.121, 4e-09)],
         200: [(0.01, 0.055), (0.152, 4e-09)],
         400: [(0.015, 6.000000000000001e-05), (0.157, 4e-09)]},
 'TM': {1: {'CONS': (0.97, 0.005), 'GA': (0.844, 0.017), 'GWO': (0.783, 0.012), 'HYB': (0.776, 0.012), 'PSO': (0.774, 0.01)},
        2: {'CONS': (0.887, 0.012), 'GA': (0.813, 0.041), 'GWO': (0.697, 0.02), 'HYB': (0.697, 0.021), 'PSO': (0.704, 0.026)},
        4: {'CONS': (0.83, 0.02), 'GA': (0.8, 0.052), 'GWO': (0.623, 0.022), 'HYB': (0.612, 0.021), 'PSO': (0.653, 0.041)}},
 'TMP': {1: [(-0.006, 0.047), (-0.008, 0.0007), (0.062, 1e-08)],
         2: [(0.0, 0.95), (0.007, 0.2), (0.117, 6.000000000000001e-09)],
         4: [(-0.011, 0.013), (0.03, 0.0001), (0.178, 6.000000000000001e-09)]},
 'TN': {('fixed', 10): (0.337, 0.4, 0.542, 15.8, 37.9, 2.9, 5.4, 5.1),
        ('fixed', 20): (0.574, 0.68, 0.647, 15.7, 11.3, 7.7, 11.7, 7.5),
        ('fixed', 30): (0.693, 0.783, 0.738, 11.5, 6.1, 11.2, 15.1, 10.9),
        ('fixed', 40): (0.763, 0.836, 0.793, 8.7, 3.8, 12.8, 15.9, 12.6),
        ('fixed', 50): (0.804, 0.869, 0.829, 7.4, 3.1, 14.1, 17.1, 14.2),
        ('fixed', 60): (0.838, 0.89, 0.857, 5.9, 2.2, 14.8, 16.7, 14.5),
        ('scaled', 10): (0.423, 0.733, 0.468, 42.3, 9.7, 3.7, 13.6, 4.8),
        ('scaled', 20): (0.552, 0.743, 0.611, 25.7, 9.6, 6.9, 13.8, 7.4),
        ('scaled', 30): (0.693, 0.783, 0.738, 11.5, 6.1, 11.2, 15.1, 10.9),
        ('scaled', 40): (0.739, 0.786, 0.79, 6.1, 6.5, 12.2, 14.5, 11.9),
        ('scaled', 50): (0.789, 0.798, 0.84, 1.2, 6.2, 14.2, 14.8, 14.0),
        ('scaled', 60): (0.833, 0.82, 0.881, -1.6, 5.4, 14.7, 14.4, 14.6)},
 'TP': {'CONS': (0.1945, 0.1871, 0.2019, 1e-07),
        'DE': (0.0485, 0.0412, 0.0558, 1e-07),
        'GA': (0.1171, 0.1037, 0.1305, 1e-07),
        'HYB': (0.0005, -0.0075, 0.0085, 1.0),
        'PSO': (-0.0021, -0.0105, 0.0064, 1.0),
        'RS': (0.1314, 0.1229, 0.1398, 1e-07),
        'S-GWO': (-0.0034, -0.0097, 0.0028, 1.0),
        'S-HYB': (-0.0002, -0.0088, 0.0084, 1.0),
        'WOA': (0.118, 0.1075, 0.1284, 1e-07)},
 'TQ': {2000: {'GA': (0.864, 0.041), 'GWO': (0.733, 0.018), 'HYB': (0.748, 0.021), 'PSO': (0.784, 0.033), 'S-GWO': (0.738, 0.018)},
        5000: {'GA': (0.842, 0.046), 'GWO': (0.703, 0.019), 'HYB': (0.717, 0.016), 'PSO': (0.735, 0.028), 'S-GWO': (0.709, 0.019)},
        10000: {'GA': (0.82, 0.044), 'GWO': (0.697, 0.022), 'HYB': (0.693, 0.016), 'PSO': (0.697, 0.024), 'S-GWO': (0.699, 0.02)},
        20000: {'GA': (0.786, 0.05), 'GWO': (0.7, 0.019), 'HYB': (0.687, 0.017), 'PSO': (0.672, 0.022), 'S-GWO': (0.704, 0.019)},
        50000: {'GA': (0.728, 0.051), 'GWO': (0.718, 0.016), 'HYB': (0.676, 0.019), 'PSO': (0.662, 0.025), 'S-GWO': (0.719, 0.015)},
        100000: {'GA': (0.694, 0.04), 'GWO': (0.725, 0.014), 'HYB': (0.672, 0.021), 'PSO': (0.661, 0.024), 'S-GWO': (0.731, 0.015)}},
 'TQP': {2000: [(0.015, 0.001), (-0.037, 1e-05), (0.052, 1e-07), (0.131, 7.000000000000001e-09)],
         5000: [(0.014, 0.0002), (-0.018, 0.002), (0.032, 2e-05), (0.139, 7.000000000000001e-09)],
         10000: [(-0.004, 1.0), (-0.003, 1.0), (-0.001, 1.0), (0.123, 7.000000000000001e-09)],
         20000: [(-0.013, 0.002), (0.016, 0.00030000000000000003), (-0.029, 8e-07), (0.086, 7.000000000000001e-09)],
         50000: [(-0.042, 7.000000000000001e-09), (0.014, 0.011), (-0.056, 7.000000000000001e-09), (0.01, 0.72)],
         100000: [(-0.054, 7.000000000000001e-09), (0.011, 0.008), (-0.065, 7.000000000000001e-09), (-0.032, 0.0005)]},
 'TR': {8: (0.309, {'CONS': 60.8, 'DE': 10.4, 'GA': 41.7, 'GWO': 7.0, 'HYB': 6.8, 'PSO': 6.3, 'RS': 45.8, 'S-GWO': 9.7}),
        10: (0.389, {'CONS': 51.0, 'DE': 9.1, 'GA': 29.9, 'GWO': 5.1, 'HYB': 5.0, 'PSO': 5.1, 'RS': 36.9, 'S-GWO': 4.0}),
        30: (0.677, {'CONS': 30.6, 'DE': 9.4, 'GA': 19.7, 'GWO': 2.3, 'HYB': 1.4, 'PSO': 3.4, 'RS': 22.3, 'S-GWO': 2.4})},
 'TS': {(60, 'fixed'): {'CONS': (0.939, 0.006, 1.0),
                        'GA': (1.133, 0.048, 0.87),
                        'GWO': (0.835, 0.015, 1.0),
                        'HYB': (0.854, 0.046, 1.0),
                        'PSO': (1.069, 0.076, 0.9),
                        'S-GWO': (0.837, 0.012, 1.0)},
        (60, 'scaled'): {'CONS': (0.939, 0.006, 1.0),
                         'GA': (1.076, 0.032, 0.93),
                         'GWO': (0.83, 0.012, 1.0),
                         'HYB': (0.816, 0.022, 1.0),
                         'PSO': (0.966, 0.048, 1.0),
                         'S-GWO': (0.834, 0.011, 1.0)},
        (100, 'fixed'): {'CONS': (0.96, 0.005, 1.0),
                         'GA': (1.532, 0.135, 0.07),
                         'GWO': (0.922, 0.01, 1.0),
                         'HYB': (1.459, 0.165, 0.1),
                         'PSO': (1.949, 0.149, 0.0),
                         'S-GWO': (0.922, 0.009, 1.0)},
        (100, 'scaled'): {'CONS': (0.96, 0.005, 1.0),
                          'GA': (1.341, 0.092, 0.17),
                          'GWO': (0.921, 0.009, 1.0),
                          'HYB': (1.119, 0.104, 0.77),
                          'PSO': (1.442, 0.098, 0.3),
                          'S-GWO': (0.921, 0.01, 1.0)}}}



# ============================================================================= model
class Inst:
    """One random instance. The fading draw (if any) is made AFTER the baseline draws, so the baseline
    part of an instance is identical with and without fading."""
    def __init__(self, N, rng, fading=False):
        self.N = N
        self.f = rng.uniform(0.7e9, 1e9, N)
        self.S = rng.uniform(0.1e6, 1e6, N)
        self.c = rng.uniform(300, 700, N)
        d = rng.uniform(50, 300, N)
        self.g = 1e-3 * d ** -3.5
        if fading:
            self.g = self.g * rng.exponential(1.0, N)
        self.C = self.S * self.c
        self.Eloc = P['kappa'] * self.f ** 2 * self.C
        self.Lloc = self.C / self.f
        self.D = 4 * N
        self.cfg(2, 8, 3)

    def cfg(self, M, R, Umax, beta=0.0, wE=0.5, omega=0.0, eps=0.0, xi=DEFAULT_XI):
        self.M, self.R, self.Umax, self.beta, self.wE = M, R, Umax, beta, wE
        self.omega, self.eps, self.xi = omega, eps, xi
        self.W = P['Btot'] / R
        self.sig = P['N0'] * self.W
        f_ref = P['Fk'] / math.ceil(self.N / M)
        self.kf = xi * P['kappa'] * (F_DEV / f_ref) ** 2
        bw = P['Btot'] / self.N
        self.rdl = bw * np.log2(1 + P['Pmax'] * self.g / (P['N0'] * bw))
        return self


def scaled_RM(N):
    return max(1, math.ceil(2 * N / 30)), max(1, math.ceil(8 * N / 30))


def sinr_sorted(qs, bs, R, sig, beta):
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
    """J_sys = mean_i [ wE*(E_i + omega*E_fog_i)/Eloc_i + wL*L_i/Lloc_i + pen_i ];  omega=eps=0 gives the baseline J.
    L_i = S/r + C/f_fog + eps*S/r_dl. Repair: per block only the Umax strongest received-power offloaders stay
    offloaded; node shares above capacity are scaled proportionally."""
    N = I.N
    a = a.copy()
    L = I.Lloc.copy(); Edev = I.Eloc.copy(); Efog = np.zeros(N); pen = np.zeros(N)
    idx = np.nonzero(a > 0)[0]
    if idx.size:
        q = I.g[idx] * p[idx]; bb = b[idx]
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
        Lo = ttx + I.C[io] / fi + I.eps * I.S[io] / I.rdl[io]
        L[io] = Lo
        Edev[io] = p[io] * ttx
        Efog[io] = I.kf * fi ** 2 * I.C[io]
        pen[io] = P['lam'] * (np.maximum(0, Lo - P['Lmax']) / P['Lmax'] + np.maximum(0, 1 - gam / P['gth']))
    En = (Edev + I.omega * Efog) / I.Eloc
    Ln = L / I.Lloc
    Ji = I.wE * En + (1 - I.wE) * Ln
    J = float((Ji + pen).mean())
    if not full:
        return J
    off = a > 0
    return dict(J=J, J_unpen=float(Ji.mean()), E_abs=float(Edev.mean() * 1e3), Efog_abs=float(Efog.mean() * 1e3),
                L_abs=float(L.mean() * 1e3), En=float((Edev / I.Eloc).mean()), Ln=float(Ln.mean()),
                Etx=float(Edev[off].sum() / N * 1e3), off_frac=float(off.mean()), feasible=bool(pen.max() <= 1e-12),
                Eloc_abs=float(I.Eloc.mean() * 1e3))


class Problem:
    def __init__(self, I, Q):
        self.I, self.Q, self.D = I, Q, I.D
        self.n, self.bf, self.bx = 0, np.inf, None

    def __call__(self, x):
        x = np.clip(x, 0, 1)
        v = evaluate_dec(self.I, *decode(x, self.I))
        self.n += 1
        if v < self.bf:
            self.bf, self.bx = v, x.copy()
        return v


# ============================================================================= constructive and reference rules
def constructive(I):
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
            Lu = I.S[us] / r + I.C[us] / fs + I.eps * I.S[us] / I.rdl[us]
            if np.any(Lu > P['Lmax']):
                continue
            Eu = p[us] * I.S[us] / r + I.omega * I.kf * fs ** 2 * I.C[us]
            if np.any(I.wE * Eu / I.Eloc[us] + (1 - I.wE) * Lu / I.Lloc[us] >= 1):
                continue
            occ[bl].append(i); a[i] = k + 1; b[i] = bl; f[i] = fi; Fres[k] -= fi
            break
    return a, b, p, f


def equal_alloc(I):
    N, M, R = I.N, I.M, I.R
    return ((np.arange(N) % M) + 1, np.arange(N) % R, np.full(N, P['Pmax']), np.full(N, P['Fk'] / math.ceil(N / M)))


# ============================================================================= optimisers
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
        n = min(K - 1, Q - prob.n)
        el = int(np.argmin(fit))
        p1, p2 = _tourn(pop, fit, n, rng), _tourn(pop, fit, n, rng)
        al = rng.random((n, 1)); cx = rng.random((n, 1)) < pc
        ch = np.where(cx, al * p1 + (1 - al) * p2, p1)
        ch = np.clip(ch + (rng.random(ch.shape) < pm) * rng.normal(0, mu, ch.shape), 0, 1)
        cf = np.array([prob(c) for c in ch])
        pop = np.vstack([pop[el], ch]); fit = np.concatenate([[fit[el]], cf])


def gwo_ga(prob, K, Q, rng, seed_x=None, rho=0.0, use_x=True, use_m=True, pc=0.8, pm=0.1, mu=0.02,
           greedy=False, sched=None, **_):
    """rho=0: plain GWO. greedy=True (G-GWO): a moved wolf replaces its parent only if not worse.
    sched=100 (F-GWO): a decreases linearly over `sched` generations, then stays 0 (default: over Q/K generations)."""
    D = prob.D
    pop, fit = _init(prob, K, rng, seed_x)
    T = int(sched) if sched else max(1, (Q - prob.n) // K)
    t = 0
    while prob.n < Q:
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
            v = prob(new[k])
            if (not greedy) or v <= fit[i]:
                pop[i] = new[k]; fit[i] = v
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
        v = np.clip(w * v + c1 * rng.random(pop.shape) * (pb - pop) + c2 * rng.random(pop.shape) * (g - pop), -0.2, 0.2)
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


BASE = {'RS': (rs, {}), 'GA': (ga, {}), 'GWO': (gwo_ga, dict(rho=0.0)), 'PSO': (pso, {}), 'DE': (de, {}),
        'WOA': (woa, {}), 'HYB': (gwo_ga, dict(rho=0.5)),
        'GWO+X': (gwo_ga, dict(rho=0.5, use_m=False)), 'GWO+M': (gwo_ga, dict(rho=0.5, use_x=False)),
        'G-GWO': (gwo_ga, dict(rho=0.0, greedy=True)), 'F-GWO': (gwo_ga, dict(rho=0.0, sched=100))}


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


# ============================================================================= jobs
def run_alg(name, I, Q, K, expid, run, salt=0):
    t0 = time.perf_counter()
    if name in ('CONS', 'EQUAL', 'LOCAL'):
        dec = {'CONS': constructive, 'EQUAL': equal_alloc,
               'LOCAL': lambda J: (np.zeros(J.N, int), np.zeros(J.N, int), np.full(J.N, P['Pmax']), np.zeros(J.N))}[name](I)
        m = evaluate_dec(I, *dec, full=True)
    else:
        fn, kw, seeded = parse(name)
        rng = np.random.default_rng(np.random.SeedSequence([expid, run, zlib.crc32(name.encode()), salt]))
        sx = encode(*constructive(I), I) if seeded else None
        prob = Problem(I, Q)
        fn(prob, K, Q, rng, seed_x=sx, **kw)
        m = evaluate_dec(I, *decode(prob.bx, I), full=True)
        m['J'] = prob.bf
    m['time'] = time.perf_counter() - t0
    return m


def job(t):
    inst_exp = t.get('inst_exp', t['expid'])
    inst_run = t.get('inst_run', t['run'])
    rng = np.random.default_rng(np.random.SeedSequence([inst_exp, inst_run, t['N']]))
    I = Inst(t['N'], rng, fading=t.get('fading', False))
    I.cfg(t['M'], t['R'], t['Umax'], t.get('beta', 0.0), t.get('wE', 0.5), t.get('omega', 0.0), t.get('eps', 0.0),
          t.get('xi', DEFAULT_XI))
    row = run_alg(t['alg'], I, t['Q'], t['K'], t['expid'], t['run'], t.get('salt', 0))
    return {**{k: v for k, v in t.items() if k != 'expid'}, **row}


def T(expid, run, alg, N=30, Q=10000, K=100, cfg=None, **extra):
    d = dict(expid=expid, run=run, alg=alg, N=N, Q=Q, K=K, M=2, R=8, Umax=3)
    d.update(cfg or {}); d.update(extra)
    return d


def run_tasks(tasks, workers, label=''):
    n = len(tasks); rows = []; t0 = time.time(); step = max(1, n // 20)
    print(f"[{label}] {n} jobs, workers={workers}", flush=True)
    if workers <= 1:
        for i, t in enumerate(tasks):
            rows.append(job(t))
            if (i + 1) % step == 0:
                print(f"[{label}] {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(job, t) for t in tasks]
            for i, f in enumerate(as_completed(futs)):
                rows.append(f.result())
                if (i + 1) % step == 0:
                    print(f"[{label}] {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def get_runs(A, name, tasks):
    path = f"{OUT}/{name}_runs.csv"
    if A.reuse and os.path.exists(path):
        print(f"[{name}] reusing {path}")
        return pd.read_csv(path)
    df = run_tasks(tasks, A.workers, name)
    df.to_csv(path, index=False)
    return df


# ============================================================================= statistics
def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(ps)):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def wil(d):
    return stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0


def paired(df, a, b, val='J', key='alg', idx='run', margin=0.01):
    x = df[df[key] == a].set_index(idx)[val]; y = df[df[key] == b].set_index(idx)[val]
    x, y = x.align(y, join='inner'); d = (x - y).values; n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
    tc = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    p1 = stats.wilcoxon(d + margin, alternative='greater').pvalue
    p2 = stats.wilcoxon(d - margin, alternative='less').pvalue
    return dict(a=a, b=b, n=n, mean_diff=d.mean(), se=se, ci_lo=d.mean() - tc * se, ci_hi=d.mean() + tc * se,
                p_raw=wil(d), p_tost=max(p1, p2))


def family(df, pairs, groupcols=(), **kw):
    rows = []
    groups = [((), df)] if not groupcols else list(df.groupby(list(groupcols)))
    for g, sub in groups:
        fam = [paired(sub, a, b, **kw) for a, b in pairs]
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
    d['sav_dev'] = 1 - d['E_abs'] / d['Eloc_abs']
    d['sav_sys'] = 1 - (d['E_abs'] + d['Efog_abs']) / d['Eloc_abs']
    g = d.groupby(by).agg(J=('J', 'mean'), J_sd=('J', 'std'), J_feas=('J_feas', 'mean'), feas=('feasible', 'mean'),
                          E_mJ=('E_abs', 'mean'), Efog_mJ=('Efog_abs', 'mean'), L_ms=('L_abs', 'mean'),
                          En=('En', 'mean'), Ln=('Ln', 'mean'), Etx_mJ=('Etx', 'mean'), off=('off_frac', 'mean'),
                          sav_dev=('sav_dev', 'mean'), sav_sys=('sav_sys', 'mean'), time=('time', 'mean'), n=('J', 'size'))
    return g.reset_index()


def friedman(df, algs, val='J'):
    M = df[df.alg.isin(algs)].pivot(index='run', columns='alg', values=val)[algs]
    st = stats.friedmanchisquare(*[M[c] for c in algs])
    return st.statistic, st.pvalue, M.rank(axis=1).mean()


def pm(m, s, d=3):
    return f"\\({m:.{d}f}\\pm{s:.{d}f}\\)"


def save(df, name):
    df.to_csv(f"{OUT}/{name}.csv", index=False)


def show(df, title, cols=None):
    print(f"\n=== {title} ==="); print((df if cols is None else df[cols]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


PV = {}   # paper values collected for paper_values.json


# ============================================================================= E9 fading
def E9(A):
    Q = A.q(1e4)
    algs = ['RS', 'GA', 'GWO', 'PSO', 'DE', 'WOA', 'HYB', 'S-GWO', 'S-HYB', 'CONS', 'EQUAL', 'LOCAL']
    df = get_runs(A, 'E9a', [T(91, r, a, Q=Q, inst_exp=1, fading=fd) for r in range(A.runs) for fd in (False, True) for a in algs])
    s = summ(df, ['fading', 'alg']); save(s, 'E9a_summary'); show(s, 'E9a default scenario, no fading vs Rayleigh fading',
        ['fading', 'alg', 'J', 'J_sd', 'feas', 'E_mJ', 'L_ms', 'Etx_mJ', 'off', 'sav_dev', 'n'])
    f = df[df['fading']].copy()
    pairs = [('HYB', 'GWO'), ('PSO', 'GWO'), ('S-GWO', 'GWO'), ('S-HYB', 'GWO'), ('DE', 'GWO'), ('GA', 'GWO'), ('WOA', 'GWO'), ('RS', 'GWO'), ('CONS', 'GWO')]
    pf = family(f, pairs)
    cand = [0, 1, 2, 3]; pf['tost_holm'] = np.nan
    pf.loc[cand, 'tost_holm'] = holm(pf.loc[cand, 'p_tost'])
    save(pf, 'E9a_pairs'); show(pf, 'E9a paired differences vs GWO under fading (Holm over 9; TOST Holm over 4 candidates)',
        ['a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost', 'tost_holm'])
    ten = ['RS', 'GA', 'GWO', 'PSO', 'DE', 'WOA', 'HYB', 'S-GWO', 'S-HYB', 'CONS']
    for fd in (False, True):
        chi, p, rk = friedman(df[df['fading'] == fd], ten)
        print(f"\nFriedman (fading={fd}): chi2={chi:.1f} p={p:.2e}; mean ranks:", ', '.join(f"{k} {v:.2f}" for k, v in rk.sort_values().items()))
    eff = []
    for a in ['GWO', 'PSO', 'HYB', 'GA', 'CONS']:
        r = paired(df[df.alg == a].assign(alg=lambda d: d['fading'].map({True: 'fad', False: 'base'})), 'fad', 'base')
        eff.append(dict(alg=a, fading_minus_base=r['mean_diff'], ci_lo=r['ci_lo'], ci_hi=r['ci_hi'], p=r['p_raw']))
    show(pd.DataFrame(eff), 'E9a effect of fading on J (paired by instance)')
    sm = s.set_index(['fading', 'alg'])
    print("\n--- LaTeX rows for Table E9 (baseline column = values printed in the paper) ---")
    basetxt = {'GWO': '0.696\\pm0.022', 'PSO': '0.693\\pm0.023', 'HYB': '0.696\\pm0.020', 'GA': '0.813\\pm0.037', 'CONS': '0.890\\pm0.010'}
    names = {'GWO': 'GWO', 'PSO': 'PSO', 'HYB': 'Hybrid', 'GA': 'GA', 'CONS': 'Constructive'}
    for a in ['GWO', 'PSO', 'HYB', 'GA', 'CONS']:
        print(f"{names[a]} & \\({basetxt[a]}\\) & {pm(sm.loc[(True, a), 'J'], sm.loc[(True, a), 'J_sd'])} \\\\")
    PV['E9a'] = {a: dict(J_base=float(sm.loc[(False, a), 'J']), J_fad=float(sm.loc[(True, a), 'J']), sd_fad=float(sm.loc[(True, a), 'J_sd'])) for a in ten}
    # ---- E9b NOMA vs OMA vs OMA3R under fading
    Ns = [10, 20, 30, 40, 50, 60]; tasks = []
    for regime in ('fixed', 'scaled'):
        for N in Ns:
            M, R = (2, 8) if regime == 'fixed' else scaled_RM(N)
            Qn = A.q(1e4 * N / 30)
            for sch, cfg in (('NOMA', dict(M=M, R=R, Umax=3)), ('OMA', dict(M=M, R=R, Umax=1)), ('OMA3R', dict(M=M, R=3 * R, Umax=1))):
                for alg in ('GWO', 'CONS'):
                    tasks += [T(92, r, alg, N=N, Q=Qn, cfg=cfg, scheme=sch, regime=regime, inst_exp=4, fading=True) for r in range(A.runs)]
    db = get_runs(A, 'E9b', tasks)
    sb = summ(db, ['regime', 'N', 'alg', 'scheme']); save(sb, 'E9b_summary')
    rows = []
    for (reg, N, alg), g in db.groupby(['regime', 'N', 'alg']):
        mm = g.groupby('scheme').agg(J=('J', 'mean'), E=('E_abs', 'mean'), L=('L_abs', 'mean'))
        for ref in ('OMA', 'OMA3R'):
            pe = paired(g.assign(alg=g['scheme']), 'NOMA', ref)
            rows.append(dict(regime=reg, N=N, alg=alg, ref=ref, J_NOMA=mm.loc['NOMA', 'J'], J_ref=mm.loc[ref, 'J'],
                             dJ_pct=100 * (mm.loc[ref, 'J'] - mm.loc['NOMA', 'J']) / mm.loc[ref, 'J'],
                             dE_pct=100 * (mm.loc[ref, 'E'] - mm.loc['NOMA', 'E']) / mm.loc[ref, 'E'],
                             dL_pct=100 * (mm.loc[ref, 'L'] - mm.loc['NOMA', 'L']) / mm.loc[ref, 'L'], p_raw=pe['p_raw']))
    gap = pd.DataFrame(rows); gap['p_holm'] = holm(gap['p_raw']); save(gap, 'E9b_gap')
    show(gap[gap.alg == 'GWO'], 'E9b NOMA gain under fading, GWO (positive = NOMA better; Holm over all rows)')
    show(gap[gap.alg == 'CONS'], 'E9b NOMA gain under fading, constructive rule')
    gg = gap[(gap.alg == 'GWO') & (gap.ref == 'OMA') & (gap.regime == 'scaled')]
    neg = gg[gg.dJ_pct < 0]['N'].tolist()
    print("\nScaled-resource gain over OMA is negative at N =", neg if neg else "none (no sign reversal)", "(paper, no fading: N=60)")
    c30 = sb.set_index(['regime', 'N', 'alg', 'scheme']).loc[('fixed', 30, 'GWO')]
    print("--- LaTeX rows for Table E9 (N=30, fixed resources, GWO) ---")
    for sch, nm, bt in (('OMA', 'GWO, OMA', '0.783\\) (E4)'), ('OMA3R', 'GWO, OMA3R', '0.738\\) (E4)')):
        print(f"{nm} & \\({bt} & {pm(c30.loc[sch, 'J'], c30.loc[sch, 'J_sd'])} \\\\")
    PV['E9b'] = gap[gap.alg == 'GWO'][['regime', 'N', 'ref', 'dJ_pct', 'dE_pct', 'dL_pct', 'p_holm']].to_dict('records')


# ============================================================================= E10 system-level cost
def E10(A):
    Q = A.q(1e4); algs = ['GWO', 'PSO', 'HYB', 'GA', 'CONS']
    sets = [(0, 0.0), (1, 0.0), (0, 0.1), (1, 0.1)]
    tasks = [T(10, r, a, Q=Q, inst_exp=1, omega=w, eps=e, xi=A.xi) for r in range(A.runs) for (w, e) in sets for a in algs]
    xis = [x for x in (0.1, 0.25, 0.5, 1.0) if abs(x - A.xi) > 1e-12]
    tasks += [T(10, r, a, Q=Q, inst_exp=1, omega=1, eps=0.0, xi=x) for r in range(A.runs) for x in xis for a in ('GWO', 'PSO')]
    df = get_runs(A, 'E10', tasks)
    s = summ(df, ['omega', 'eps', 'xi', 'alg']); save(s, 'E10_summary')
    show(s, f'E10 J_sys (xi={A.xi} unless stated); sav_dev = device-energy saving, sav_sys = device+fog energy saving vs all-local',
         ['omega', 'eps', 'xi', 'alg', 'J', 'J_sd', 'off', 'E_mJ', 'Efog_mJ', 'sav_dev', 'sav_sys', 'L_ms', 'feas'])
    base = df[np.isclose(df.xi, A.xi)]
    rk = []
    for (w, e) in sets:
        sub = base[(base.omega == w) & (base.eps == e)]
        chi, p, r = friedman(sub, algs)
        pr = paired(sub, 'HYB', 'GWO'); pp = paired(sub, 'PSO', 'GWO')
        rk.append(dict(omega=w, eps=e, ranks=', '.join(f"{k} {v:.2f}" for k, v in r.sort_values().items()), friedman_p=p,
                       HYB_minus_GWO=pr['mean_diff'], HYB_tost=pr['p_tost'], PSO_minus_GWO=pp['mean_diff'], PSO_p=pp['p_raw']))
    show(pd.DataFrame(rk), 'E10 ranking and key paired tests per setting')
    sm = s[np.isclose(s.xi, A.xi)].set_index(['omega', 'eps', 'alg'])
    print(f"\n--- LaTeX rows for Table E10 (xi={A.xi}; last column = device-energy saving / system-energy saving, GWO) ---")
    for (w, e) in sets:
        g, p_ = sm.loc[(w, e, 'GWO')], sm.loc[(w, e, 'PSO')]
        print(f"({w}, {e:g}) & {g['J']:.3f} & {p_['J']:.3f} & {g['off']:.2f} & {100 * g['sav_dev']:.1f}\\,\\% / {100 * g['sav_sys']:.1f}\\,\\% \\\\")
    sx = s[(s.omega == 1) & (s.eps == 0)].set_index(['xi', 'alg'])
    print("\nFog-energy sensitivity (omega=1, eps=0): xi, GWO J_sys, PSO J_sys, GWO offloaded, GWO device saving, GWO system saving")
    for x in sorted(sx.index.get_level_values(0).unique()):
        g, p_ = sx.loc[(x, 'GWO')], sx.loc[(x, 'PSO')]
        print(f"  xi={x:<5g} {g['J']:.3f} {p_['J']:.3f} {g['off']:.2f} {100 * g['sav_dev']:.1f}% {100 * g['sav_sys']:.1f}%")
    PV['E10'] = s.to_dict('records')


# ============================================================================= E11 GWO diagnostics
def E11(A):
    Qs = [1e4, 5e4, 1e5]; algs = ['GWO', 'G-GWO', 'F-GWO', 'HYB', 'PSO']
    df = get_runs(A, 'E11', [T(11, r, a, Q=A.q(Q), Qlab=Q, inst_exp=2) for r in range(A.runs) for Q in Qs for a in algs])
    s = summ(df, ['Qlab', 'alg']); save(s, 'E11_summary'); show(s, 'E11 GWO diagnostics (E2 instances)', ['Qlab', 'alg', 'J', 'J_sd', 'feas', 'n'])
    pairs = [('G-GWO', 'GWO'), ('F-GWO', 'GWO'), ('G-GWO', 'HYB'), ('G-GWO', 'PSO'), ('F-GWO', 'HYB'), ('F-GWO', 'PSO'), ('HYB', 'GWO'), ('HYB', 'PSO')]
    pf = family(df, pairs, ['Qlab']); save(pf, 'E11_pairs')
    show(pf, 'E11 paired differences (first minus second; Holm over 8 per budget)', ['Qlab', 'a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost'])
    # deterioration with budget, paired by instance
    det = []
    for a in algs:
        for q0, q1 in ((1e4, 5e4), (1e4, 1e5)):
            x = df[(df.alg == a) & (df.Qlab == q1)].set_index('run')['J']; y = df[(df.alg == a) & (df.Qlab == q0)].set_index('run')['J']
            x, y = x.align(y, join='inner'); d = (x - y).values; n = len(d); se = d.std(ddof=1) / math.sqrt(n); tc = stats.t.ppf(.975, n - 1)
            det.append(dict(alg=a, from_Q=q0, to_Q=q1, dJ=d.mean(), ci_lo=d.mean() - tc * se, ci_hi=d.mean() + tc * se, p_raw=wil(d)))
    dd = pd.DataFrame(det); dd['p_holm'] = holm(dd['p_raw']); save(dd, 'E11_deterioration')
    dd['verdict'] = np.where((dd.dJ > 0.005) & (dd.p_holm < 0.05), 'DETERIORATES', np.where((dd.dJ < -0.005) & (dd.p_holm < 0.05), 'improves', 'no significant change'))
    show(dd, 'E11 change of J when the budget grows (positive = worse; Holm over all rows)')
    sm = s.set_index(['Qlab', 'alg'])
    print("\n--- LaTeX rows for Table E11 ---")
    for Q, lab in ((1e4, '\\(10^{4}\\)'), (5e4, '\\(5\\times10^{4}\\)'), (1e5, '\\(10^{5}\\)')):
        print(f"{lab} & " + " & ".join(f"{sm.loc[(Q, a), 'J']:.3f}" for a in algs) + " \\\\")
    rem = dd[(dd.to_Q == 1e5) & dd.alg.isin(['G-GWO', 'F-GWO'])]
    print("\nVerdict: GWO deteriorates:", dd[(dd.alg == 'GWO') & (dd.to_Q == 1e5)]['verdict'].iloc[0],
          "| G-GWO:", rem[rem.alg == 'G-GWO']['verdict'].iloc[0], "| F-GWO:", rem[rem.alg == 'F-GWO']['verdict'].iloc[0])
    PV['E11'] = dict(summary=s.to_dict('records'), deterioration=dd.to_dict('records'), pairs=pf.to_dict('records'))


# ============================================================================= E12 mutation step
def E12(A):
    mus = [0.02, 0.05, 0.1, 0.2]; Qs = [1e4, 1e5]; OFF = 1000
    sel = get_runs(A, 'E12_select', [T(121, OFF + r, f'HYB:mu={m}', Q=A.q(Q), Qlab=Q, mu=m, inst_exp=12) for r in range(A.runs) for Q in Qs for m in mus])
    ss = sel.groupby(['Qlab', 'mu'])['J'].agg(['mean', 'std']).reset_index(); save(ss, 'E12_selection')
    show(ss, 'E12 selection instances (disjoint seeds): hybrid J by mutation step')
    best = {Q: float(ss[ss.Qlab == Q].sort_values('mean').iloc[0]['mu']) for Q in Qs}
    print("Selected mu per budget:", best)
    tasks = []
    for r in range(A.runs):
        for Q in Qs:
            for m in mus:
                tasks.append(T(122, r, f'HYB:mu={m}', Q=A.q(Q), Qlab=Q, mu=m, inst_exp=2))
            for a in ('PSO', 'G-GWO'):
                tasks.append(T(122, r, a, Q=A.q(Q), Qlab=Q, mu=np.nan, inst_exp=2))
    df = get_runs(A, 'E12_test', tasks)
    s = df.groupby(['Qlab', 'alg'])['J'].agg(['mean', 'std', 'count']).reset_index(); save(s, 'E12_test_summary')
    show(s, 'E12 test instances (E2 instance stream)')
    out = []
    for Q in Qs:
        sub = df[df.Qlab == Q]; tn = f"HYB:mu={best[Q]}"
        pf = family(sub, [(tn, 'PSO'), (tn, 'G-GWO')]); pf['Qlab'] = Q; pf['tuned'] = True; out.append(pf)
        pa = family(sub, [(f'HYB:mu={m}', 'PSO') for m in mus]); pa['Qlab'] = Q; pa['tuned'] = False; out.append(pa)
    po = pd.concat(out); save(po, 'E12_pairs')
    show(po, 'E12 paired differences (first minus second; Holm within each family)', ['Qlab', 'tuned', 'a', 'b', 'mean_diff', 'ci_lo', 'ci_hi', 'p_raw', 'p_holm', 'p_tost'])
    sm = s.set_index(['Qlab', 'alg'])
    print("\n--- LaTeX rows for Table E12 ---")
    for Q, lab, base in ((1e4, '\\(10^{4}\\)', '0.696 (E1)'), (1e5, '\\(10^{5}\\)', '0.672 (E2)')):
        cells = [f"{sm.loc[(Q, f'HYB:mu={m}'), 'mean']:.3f}" + ('$^{*}$' if m == best[Q] else '') for m in mus[1:]]
        print(f"{lab} & {base} & " + " & ".join(cells) + " \\\\")
    print("($^{*}$ = value of mu selected on the disjoint selection instances)")
    for Q in Qs:
        t = po[(po.Qlab == Q) & po.tuned]
        pso = t[t.b == 'PSO'].iloc[0]; gg = t[t.b == 'G-GWO'].iloc[0]
        verdict = ("tuned hybrid SIGNIFICANTLY BETTER than PSO" if (pso.mean_diff < 0 and pso.p_holm < 0.05) else
                   "tuned hybrid SIGNIFICANTLY WORSE than PSO" if (pso.mean_diff > 0 and pso.p_holm < 0.05) else "no significant difference to PSO")
        print(f"Q={Q:g}: mu*={best[Q]}; vs PSO {pso.mean_diff:+.4f} (Holm p={pso.p_holm:.3g}) -> {verdict}; vs G-GWO {gg.mean_diff:+.4f} (Holm p={gg.p_holm:.3g})")
    PV['E12'] = dict(selected=best, summary=s.to_dict('records'), pairs=po.to_dict('records'))


# ============================================================================= validation against the paper's tables
VAL = []


def _status(m, ref, z, tol):
    return 'EXACT' if abs(m - ref) <= tol else ('OK' if abs(z) <= 3 else 'CHECK')


def chk(table, label, vals, ref, ref_sd, dec, n_ref=30):
    vals = np.asarray(vals, float); n = len(vals); m = vals.mean(); sd = vals.std(ddof=1) if n > 1 else 0.0
    se = math.sqrt(((ref_sd if ref_sd else sd) ** 2) / n_ref + sd ** 2 / n)
    z = (m - ref) / se if se > 0 else 0.0
    VAL.append(dict(table=table, cell=label, new=m, paper=ref, z=z, status=_status(m, ref, z, 0.6 * 10 ** -dec)))


def chk_val(table, label, m, ref, dec, se=None):
    z = (m - ref) / se if se else 0.0
    VAL.append(dict(table=table, cell=label, new=m, paper=ref, z=z, status=_status(m, ref, z, 0.6 * 10 ** -dec)))


def chk_diff(table, label, x, y, ref, dec, ref_p=None, new_p=None):
    x, y = x.align(y, join='inner'); d = (x - y).values; n = len(d); se = d.std(ddof=1) / math.sqrt(n)
    z = (d.mean() - ref) / (math.sqrt(2) * se) if se > 0 else 0.0
    st = _status(d.mean(), ref, z, 0.6 * 10 ** -dec)
    if ref_p is not None and new_p is not None and (ref_p < 0.05) != (new_p < 0.05) and st != 'EXACT':
        st = 'CHECK'
    VAL.append(dict(table=table, cell=label, new=d.mean(), paper=ref, z=z, status=st))


def ser(df, **cond):
    m = np.ones(len(df), bool)
    for k, v in cond.items():
        m &= (df[k] == v).values
    return df[m]


def V1(A):
    algs = ['RS', 'GA', 'GWO', 'PSO', 'DE', 'WOA', 'HYB', 'S-GWO', 'S-HYB', 'CONS', 'EQUAL', 'LOCAL']
    df = get_runs(A, 'V1', [T(1, r, a) for r in range(A.vruns) for a in algs])
    for a, d in REF['T2'].items():
        g = ser(df, alg=a)
        if a == 'LOCAL':
            chk_val('Table II', 'LOCAL J', g['J'].mean(), 1.0, 3)
        else:
            chk('Table II', f'{a} J', g['J'], *d['J'], 3)
        chk('Table II', f'{a} energy', g['E_abs'], *d['E'], 1); chk('Table II', f'{a} delay', g['L_abs'], *d['L'], 0)
    base = df[df.alg == 'GWO'].set_index('run')['J']
    labs = ['HYB', 'PSO', 'S-GWO', 'S-HYB', 'DE', 'GA', 'WOA', 'RS', 'CONS']
    fam = family(df, [(a, 'GWO') for a in labs])
    for _, r in fam.iterrows():
        ref = REF['TP'][r['a']]
        chk_diff('Table P', f"{r['a']} - GWO", df[df.alg == r['a']].set_index('run')['J'], base, ref[0], 4, ref[3], r['p_holm'])


def V2(A):
    Qs = [2000, 5000, 10000, 20000] + ([50000, 100000] if A.vheavy else [])
    algs = ['GA', 'GWO', 'PSO', 'HYB', 'S-GWO']
    df = get_runs(A, 'V2', [T(2, r, a, Q=Q, Qlab=Q) for r in range(A.vruns) for Q in Qs for a in algs])
    for Q in Qs:
        for a in algs:
            chk('Table Q', f'Q={Q} {a}', ser(df, Qlab=Q, alg=a)['J'], *REF['TQ'][Q][a], 3)
    pf = family(df, [('HYB', 'GWO'), ('HYB', 'PSO'), ('PSO', 'GWO'), ('GA', 'GWO')], ['Qlab'])
    for _, r in pf.iterrows():
        col = [('HYB', 'GWO'), ('HYB', 'PSO'), ('PSO', 'GWO'), ('GA', 'GWO')].index((r['a'], r['b']))
        sub = df[df.Qlab == r['Qlab']]; ref = REF['TQP'][int(r['Qlab'])][col]
        chk_diff('Table QP', f"Q={int(r['Qlab'])} {r['a']}-{r['b']}", sub[sub.alg == r['a']].set_index('run')['J'],
                 sub[sub.alg == r['b']].set_index('run')['J'], ref[0], 3, ref[1], r['p_holm'])


def V3(A):
    names = list(REF['TA'].keys())
    df = get_runs(A, 'V3', [T(3, r, a) for r in range(A.vruns) for a in names])
    base = df[df.alg == 'GWO'].set_index('run')['J']
    for a in names:
        g = df[df.alg == a]
        chk('Table A', f'{a} J', g['J'], REF['TA'][a][0], REF['TA'][a][1], 4)
        if a != 'GWO':
            chk_diff('Table A', f'{a} - GWO', g.set_index('run')['J'], base, REF['TA'][a][2], 4)


def _regime_tasks(vr):
    tasks = []
    for regime in ('fixed', 'scaled'):
        for N in [10, 20, 30, 40, 50, 60]:
            if regime == 'scaled' and N == 30:
                continue                      # identical to fixed N=30 (same configuration and seeds)
            M, R = (2, 8) if regime == 'fixed' else scaled_RM(N)
            for sch, cfg in (('NOMA', dict(M=M, R=R, Umax=3)), ('OMA', dict(M=M, R=R, Umax=1)), ('OMA3R', dict(M=M, R=3 * R, Umax=1))):
                tasks += [T(4, r, 'GWO', N=N, Q=int(1e4 * N / 30), cfg=cfg, scheme=sch, regime=regime) for r in range(vr)]
    return tasks


def V4(A):
    df = get_runs(A, 'V4', _regime_tasks(A.vruns))
    for (reg, N), ref in REF['TN'].items():
        g = df[(df.regime == ('fixed' if (reg == 'scaled' and N == 30) else reg)) & (df.N == N)]
        mm = {s: g[g.scheme == s].set_index('run') for s in ('NOMA', 'OMA', 'OMA3R')}
        J = {s: mm[s]['J'].mean() for s in mm}; E = {s: mm[s]['E_abs'].mean() for s in mm}
        sd = {s: mm[s]['J'].std(ddof=1) for s in mm}; n = len(mm['NOMA'])
        for s, i in (('NOMA', 0), ('OMA', 1), ('OMA3R', 2)):
            chk_val('Table N', f'{reg} N={N} J {s}', J[s], ref[i], 3, se=sd[s] * math.sqrt(1 / 30 + 1 / n))
            chk_val('Table N', f'{reg} N={N} E {s}', E[s], ref[5 + i], 1, se=mm[s]['E_abs'].std(ddof=1) * math.sqrt(1 / 30 + 1 / n))
        chk_val('Table N', f'{reg} N={N} dJ vs OMA %', 100 * (J['OMA'] - J['NOMA']) / J['OMA'], ref[3], 1, se=100 * sd['NOMA'] / J['OMA'] * math.sqrt(1 / n + 1 / 30))
        chk_val('Table N', f'{reg} N={N} dJ vs OMA3R %', 100 * (J['OMA3R'] - J['NOMA']) / J['OMA3R'], ref[4], 1, se=100 * sd['NOMA'] / J['OMA3R'] * math.sqrt(1 / n + 1 / 30))


def V5(A):
    algs = ['GA', 'GWO', 'PSO', 'HYB', 'S-GWO', 'CONS']; tasks = []
    for N in (60, 100):
        M, R = scaled_RM(N)
        for mode, Q in (('fixed', 1e4), ('scaled', 1e4 * N / 30)):
            for a in algs:
                tasks += [T(5, r, a, N=N, Q=int(Q), cfg=dict(M=M, R=R, Umax=3), budget=mode) for r in range(A.vruns)]
    df = get_runs(A, 'V5', tasks)
    for (N, mode), ref in REF['TS'].items():
        for a in algs:
            g = df[(df.N == N) & (df.budget == mode) & (df.alg == a)]
            chk('Table S', f'N={N} {mode} {a} J', g['J'], ref[a][0], ref[a][1], 3)
            chk_val('Table S', f'N={N} {mode} {a} feasible', g['feasible'].mean(), ref[a][2], 2, se=0.1)


def V6(A):
    Ks = [10, 20, 30, 50, 100, 200, 400]; algs = ['DE', 'GA', 'GWO', 'HYB', 'PSO']
    df = get_runs(A, 'V6', [T(6, r, a, K=K) for r in range(A.vruns) for K in Ks for a in algs])
    for K in Ks:
        for a in algs:
            chk('Table K', f'K={K} {a}', ser(df, K=K, alg=a)['J'], *REF['TK'][K][a], 3)
    pf = family(df, [('HYB', 'GWO'), ('GA', 'GWO')], ['K'])
    for _, r in pf.iterrows():
        col = [('HYB', 'GWO'), ('GA', 'GWO')].index((r['a'], r['b'])); sub = df[df.K == r['K']]; ref = REF['TKP'][int(r['K'])][col]
        chk_diff('Table KP', f"K={int(r['K'])} {r['a']}-{r['b']}", sub[sub.alg == r['a']].set_index('run')['J'],
                 sub[sub.alg == r['b']].set_index('run')['J'], ref[0], 3, ref[1], r['p_holm'])


def V7(A):
    tasks = []
    for N in (8, 10, 30):
        M, R = scaled_RM(N) if N != 30 else (2, 8)
        cfg = dict(M=M, R=R, Umax=3)
        for r in range(A.vruns):
            tasks += [T(7, r, 'S-GWO', N=N, Q=int(A.refQ), cfg=cfg, salt=k + 1, kind='ref') for k in range(A.restarts)]
            tasks += [T(7, r, a, N=N, Q=10000, cfg=cfg, kind='alg') for a in ('RS', 'GA', 'GWO', 'PSO', 'DE', 'HYB', 'S-GWO', 'CONS')]
    df = get_runs(A, 'V7', tasks)
    if A.restarts != 10 or A.refQ != 2e5:
        print(f"NOTE: V7 uses {A.restarts} restarts x {A.refQ:g} evaluations (paper: 10 x 2e5); weaker references give smaller gaps.")
    bk = df.groupby(['N', 'run'])['J'].min().rename('J_bk').reset_index()
    a = df[df.kind == 'alg'].merge(bk, on=['N', 'run']); a['gap'] = 100 * (a['J'] - a['J_bk']) / a['J_bk']
    for N in (8, 10, 30):
        chk('Table R', f'N={N} J_bk', bk[bk.N == N]['J_bk'], REF['TR'][N][0], None, 3)
        for alg, ref in REF['TR'][N][1].items():
            chk('Table R', f'N={N} {alg} gap%', a[(a.N == N) & (a.alg == alg)]['gap'], ref, None, 1)


def V8(A):
    algs = ['GA', 'GWO', 'PSO', 'HYB', 'CONS']
    df = get_runs(A, 'V8', [T(8, r, a, cfg=dict(M=M, R=8, Umax=3), Mlab=M) for r in range(A.vruns) for M in (1, 2, 4) for a in algs])
    for M in (1, 2, 4):
        for a in algs:
            chk('Table M', f'M={M} {a}', ser(df, Mlab=M, alg=a)['J'], *REF['TM'][M][a], 3)
    cmp_ = [('HYB', 'GWO'), ('PSO', 'GWO'), ('GA', 'GWO')]
    pf = family(df, cmp_, ['Mlab'])
    for _, r in pf.iterrows():
        col = cmp_.index((r['a'], r['b'])); sub = df[df.Mlab == r['Mlab']]; ref = REF['TMP'][int(r['Mlab'])][col]
        chk_diff('Table MP', f"M={int(r['Mlab'])} {r['a']}-{r['b']}", sub[sub.alg == r['a']].set_index('run')['J'],
                 sub[sub.alg == r['b']].set_index('run')['J'], ref[0], 3, ref[1], r['p_holm'])


def report_validation():
    if not VAL:
        return
    v = pd.DataFrame(VAL); save(v, 'validation')
    print("\n" + "=" * 78 + "\nVALIDATION AGAINST THE PAPER'S TABLES\n" + "=" * 78)
    print("EXACT = equal up to table rounding (same seeds, same code);  OK = |z|<=3 vs sampling error;  CHECK = investigate")
    summ_ = v.groupby('table')['status'].value_counts().unstack(fill_value=0)
    for c in ('EXACT', 'OK', 'CHECK'):
        if c not in summ_:
            summ_[c] = 0
    summ_['cells'] = summ_[['EXACT', 'OK', 'CHECK']].sum(axis=1)
    print(summ_[['cells', 'EXACT', 'OK', 'CHECK']].to_string())
    bad = v[v.status == 'CHECK']
    if len(bad):
        print("\nCells to inspect:"); print(bad.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nTOTAL {len(v)} cells: {int((v.status == 'EXACT').sum())} exact, {int((v.status == 'OK').sum())} statistically consistent, "
          f"{int((v.status == 'CHECK').sum())} to check.")
    if (v.status == 'CHECK').sum() == 0:
        print("=> The extended model with omega=0, eps=0 and no fading reproduces the paper's tables.")


# ============================================================================= main
def main():
    global OUT
    ap = argparse.ArgumentParser(description="Numbers for E9-E12 and validation of the paper's tables")
    ap.add_argument('--exp', default='NEW', help='comma list of E9,E10,E11,E12,V1..V8 | NEW (E9-E12) | VAL | all')
    ap.add_argument('--runs', type=int, default=30); ap.add_argument('--vruns', type=int, default=30)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--quick', action='store_true', help='divide the budgets of E9-E12 by 5 (smoke test)')
    ap.add_argument('--reuse', action='store_true'); ap.add_argument('--vheavy', action='store_true')
    ap.add_argument('--xi', type=float, default=DEFAULT_XI)
    ap.add_argument('--restarts', type=int, default=10); ap.add_argument('--refQ', type=float, default=2e5)
    ap.add_argument('--out', default=OUT)
    A = ap.parse_args()
    OUT = A.out; os.makedirs(OUT, exist_ok=True)
    A.q = (lambda q: max(200, int(q / 5))) if A.quick else (lambda q: int(q))
    ex = {'E9': E9, 'E10': E10, 'E11': E11, 'E12': E12, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'V7': V7, 'V8': V8}
    sel = []
    for tok in A.exp.split(','):
        sel += {'NEW': ['E9', 'E10', 'E11', 'E12'], 'VAL': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V8'] + (['V7'] if A.vheavy else []),
                'all': ['V1', 'E9', 'E10', 'E11', 'E12', 'V2', 'V3', 'V4', 'V5', 'V6', 'V8'] + (['V7'] if A.vheavy else [])}.get(tok, [tok])
    for e in sel:
        t0 = time.time(); ex[e](A); print(f"[{e}] done in {time.time() - t0:.0f}s", flush=True)
    report_validation()
    if PV:
        json.dump(PV, open(f"{OUT}/paper_values.json", 'w'), indent=1, default=float)
        print(f"\nAll numbers for the paper are in {OUT}/*.csv and {OUT}/paper_values.json")


if __name__ == '__main__':
    main()
