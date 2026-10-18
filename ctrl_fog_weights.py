"""Referee controls (items 8, 9): (A) fixed equal fog share instead of searched shares; (B) objective-weight sweep.
Uses the paper's own model/optimiser code (noma_e9_e12.py), E10 settings: N=30, M=2, R=8, U=3, Q=1e4, K=100, GWO, xi=0.25.
Instances = E1 instances (inst_exp=1), as in E10."""
import sys, math, os, csv, time
import numpy as np
import noma_e9_e12 as m

OUT = '/home/claude/ctrl_results.csv'
FIELDS = ['arm','wE','omega','run','J','E_abs','Efog_abs','L_abs','Eloc_abs','off_frac','feasible']
orig_decode = m.decode

def eq_decode(x, I):
    a, b, p, f = orig_decode(x, I)
    return a, b, p, np.full(I.N, m.P['Fk'] / math.ceil(I.N / I.M))

def one(arm, wE, omega, r, Q=10000, K=100):
    rng = np.random.default_rng(np.random.SeedSequence([1, r, 30]))
    I = m.Inst(30, rng); I.cfg(2, 8, 3, 0.0, wE, omega, 0.0, m.DEFAULT_XI)
    m.decode = eq_decode if arm == 'equalshare' else orig_decode
    row = m.run_alg('GWO', I, Q, K, 10, r)
    return dict(arm=arm, wE=wE, omega=omega, run=r, **{k: row[k] for k in FIELDS[4:]})

if __name__ != "__main__": raise SystemExit
done = set()
if os.path.exists(OUT):
    for d in csv.DictReader(open(OUT)): done.add((d['arm'], float(d['wE']), float(d['omega']), int(d['run'])))
plan = [('equalshare', 0.5, 0), ('equalshare', 0.5, 1), ('searched', 0.5, 0), ('searched', 0.5, 1),
        ('searched', 0.25, 0), ('searched', 0.25, 1), ('searched', 0.75, 0), ('searched', 0.75, 1)]
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
new = not os.path.exists(OUT)
with open(OUT, 'a', newline='') as fh:
    w = csv.DictWriter(fh, FIELDS)
    if new: w.writeheader()
    for arm, wE, om in plan:
        for r in range(runs):
            if (arm, wE, float(om), r) in done: continue
            w.writerow(one(arm, wE, om, r)); fh.flush()
        print('finished', arm, wE, om, time.strftime('%H:%M:%S'), flush=True)
