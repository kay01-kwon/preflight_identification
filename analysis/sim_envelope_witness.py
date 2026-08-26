"""The envelope-witness bound per run: free-fit RMS vs rho_bar K C2 sqrt(B/x).

Nominal (onset-shifted as needed) is a family member, so the free
minimiser's residual is bounded by the e_omega envelope RMS at the
design cap -- no calibration, no latency channel.
"""
import contextlib, io, sys, csv
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset

G=9.81; Z=0.272; R_PHI=1/7; PHI=np.deg2rad(5.0); NOISE=2.45e-4
J_CAD={'x':0.051085,'y':0.050564}
CASES={'S1':3.066,'S2':3.066,'S3':3.066,'S4':3.066,'S5':3.066,'S6':3.066,
       'S7':3.066,'S8':3.066,'S9':3.066,'S11':3.066,'S13':3.220}
OFF={'S1':{'Mx':0.0,'My':6.0},'S2':{'Mx':10.0,'My':0.0},
     'S3':{'Mx':5.0,'My':10.0},'S4':{'Mx':20.0,'My':20.0},
     'S5':{'Mx':20.0,'My':20.0},'S6':{'Mx':20.0,'My':20.0},
     'S7':{'Mx':20.0,'My':20.0},'S8':{'Mx':25.0,'My':25.0},
     'S9':{'Mx':32.0,'My':32.0},'S11':{'Mx':14.0,'My':38.0},
     'S13':{'Mx':25.0,'My':25.0}}
L={'Mx':0.110,'My':0.140}
rows=[]
for case, mass in CASES.items():
    W=mass*G
    for simax,axis in (('Mx','x'),('My','y')):
        arm=L[simax]+OFF[case][simax]*1e-3
        jp=J_CAD[axis]+mass*(Z**2+L[simax]**2)
        c2=float(np.sqrt(W*Z/jp)); k=1.0/(W*Z)
        rb=R_PHI*0.5*W*arm*PHI**2
        with contextlib.redirect_stdout(io.StringIO()):
            bags=load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
        for bag in bags:
            md=cvp.commanded_ramp_rate(bag.name)
            if md is None: continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    crit, pw = cvp.extract_piecewise(bag, axis, model='cosh',
                                                     cosh_c2=None,
                                                     ramp_gain=None)
            except Exception:
                continue
            x=brentq(lambda v: np.sinh(v)-v-PHI*W*Z*c2/md, 1e-3, 40)
            B=0.25*np.sinh(2*x)-0.5*x
            cap=rb*k*c2*np.sqrt(B/x)+NOISE
            rows.append(dict(case=case,ax=simax,rate=md,
                             res=np.degrees(float(pw['rmse'])),
                             cap=np.degrees(cap)))
    print('done',case,flush=True)
with open('docs/sim_env_witness_runs.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader()
    w.writerows(rows)
g=defaultdict(list)
for r in rows: g[r['rate']].append(r)
print(f"\n{'rate':>6}{'n':>4}{'cap':>8}{'rmse':>8}{'inside':>8}")
ti=tt=0
for rate in sorted(g):
    v=g[rate]
    ins=sum(r['res']<=r['cap'] for r in v); ti+=ins; tt+=len(v)
    print(f"{rate:>6.2f}{len(v):>4}{np.mean([r['cap'] for r in v]):>8.3f}"
          f"{np.mean([r['res'] for r in v]):>8.3f}{ins:>5}/{len(v)}")
print(f"\n{ti}/{tt} inside; worst usage "
      f"{max(r['res']/r['cap'] for r in rows):.2f}")
out=[r for r in rows if r['res']>r['cap']]
for r in sorted(out,key=lambda r:-r['res']/r['cap'])[:8]:
    print(f"  over: {r['case']}/{r['ax']} rate {r['rate']:.2f}  "
          f"res {r['res']:.3f} vs cap {r['cap']:.3f}")
