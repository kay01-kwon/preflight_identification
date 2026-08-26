"""Eq (107)+(108) on the simulation: kernel-free bound vs the FREE fit.

GE channel absent: (107) reduces to rho_dot_phi,max / (12 W z).
rho_dot_phi,max = W l_arm phi_max om_max with om_max the true end-rate
anchor; Delta_pre from (108) with dt_c = beta/(C1 C2),
beta = rho_bar/(J_P C2). All closed-form at the design tilt cap.
Measured: the post-onset RMSE of the full free nonlinear fit.
"""
import contextlib, io, sys
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset

G=9.81; Z=0.272; R_PHI=1/7; PHI=np.deg2rad(10.0)   # design tilt cap
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

g=defaultdict(list)
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
            md_c=cvp.commanded_ramp_rate(bag.name)
            if md_c is None: continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    crit, pw = cvp.extract_piecewise(bag, axis,
                                                     model='cosh',
                                                     cosh_c2=None,
                                                     ramp_gain=None)
            except Exception:
                continue
            meas=float(pw['rmse'])
            # a priori bound at the design cap, per rate
            target=PHI*W*Z*c2/md_c
            x=brentq(lambda v: np.sinh(v)-v-target, 1e-3, 40)
            om=k*md_c*(np.cosh(x)-1)+rb*np.sinh(x)/(jp*c2)
            rd=W*arm*PHI*om
            b107=rd/(12*W*Z)
            beta=rb/(jp*c2)
            c1=k*md_c
            dtc=beta/(c1*c2)
            te=x/c2
            a=c2*dtc
            d2pre=(c1**2/te)*(1.5*dtc+np.sinh(2*a)/(2*c2)-2*np.sinh(a)/c2)
            cap=b107+np.sqrt(max(d2pre,0.0))
            g[md_c].append((np.degrees(meas),np.degrees(cap)))
    print('done',case,flush=True)

print(f"\n{'rate':>6}{'n':>4}{'cap(107+108)':>13}{'free rmse':>10}"
      f"{'inside':>8}")
ti=tt=0
for rate in sorted(g):
    v=np.array(g[rate])
    ins=int((v[:,0]<=v[:,1]).sum()); ti+=ins; tt+=len(v)
    print(f"{rate:>6.2f}{len(v):>4}{v[:,1].mean():>13.3f}{v[:,0].mean():>10.3f}"
          f"{ins:>5}/{len(v)}")
print(f"\n{ti}/{tt} inside; worst usage "
      f"{max(a/b for r in g.values() for a,b in r):.2f}")
