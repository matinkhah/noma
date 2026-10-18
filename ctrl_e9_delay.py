"""Item 22: energy and delay of NOMA vs OMA / OMA3R under Rayleigh fading (E9b settings, fixed resources, GWO)."""
import sys, os, numpy as np, pandas as pd, noma_e9_e12 as m
OUT='/home/claude/e9_delay_runs.csv'
Ns=[int(a) for a in sys.argv[1:]] or [10,20,30,40,50,60]
rows=[]
for N in Ns:
    Qn=int(round(1e4*N/30))
    for sch,cfg in (('NOMA',dict(M=2,R=8,Umax=3)),('OMA',dict(M=2,R=8,Umax=1)),('OMA3R',dict(M=2,R=24,Umax=1))):
        for r in range(30):
            t=m.T(92,r,'GWO',N=N,Q=Qn,cfg=cfg,scheme=sch,regime='fixed',inst_exp=4,fading=True)
            o=m.job(t); rows.append(dict(N=N,scheme=sch,run=r,J=o['J'],E=o['E_abs'],L=o['L_abs']))
    print('N',N,'done',flush=True)
df=pd.DataFrame(rows)
df.to_csv(OUT,mode='a',header=not os.path.exists(OUT),index=False)
