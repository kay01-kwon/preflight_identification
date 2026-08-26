"""The e_omega envelope by ramp rate, against the measured deviation.

Per run: measured e_omega RMS (deviation from the pinned-theory
nominal, post-onset), the envelope RMS at the run's REALISED tilt,
and the a priori envelope at the 5-deg design cap.
"""
import contextlib, io, sys, csv
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset

G=9.81; Z=0.272; R_PHI=1/7; PHI_D=np.deg2rad(5.0)
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

def env_rms(rb,k,c2,x):
    B=0.25*np.sinh(2*x)-0.5*x
    return rb*k*c2*np.sqrt(B/x)

rows=[]
for case, mass in CASES.items():
    W=mass*G
    for simax,axis in (('Mx','x'),('My','y')):
        arm=L[simax]+OFF[case][simax]*1e-3
        jp=J_CAD[axis]+mass*(Z**2+L[simax]**2)
        c2=float(np.sqrt(W*Z/jp)); k=1.0/(W*Z)
        with contextlib.redirect_stdout(io.StringIO()):
            bags=load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
        for bag in bags:
            md=cvp.commanded_ramp_rate(bag.name)
            if md is None: continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig=cvp.prepare_signals(bag,axis)
            i0,i1=cvp.detect_excitation_window(
                sig['moment'],moment_cap=cvp.MOMENT_CAP.get(axis))
            w=slice(i0,i1+1)
            t,om,mom=sig['t'][w],sig['omega'][w],sig['moment'][w]
            if len(t)<24: continue
            mds=float(np.polyfit(t,mom,1)[0])
            pw=cvp.cosh_onset_fit(t,om,np.zeros_like(t),onset_guess=None,
                                  c2_fixed=c2,moment_floor=0.0,
                                  ramp_gain=k,ramp_rate=mds)
            j=pw['onset_idx']
            if j<12 or len(om)-j<12: continue
            tau=t[j:]-t[j]
            e=om[j:]-pw['omega_pred'][j:]          # measured e_omega
            oc=om[j:]-float(pw['c'])
            phi_r=float(abs(np.trapezoid(oc,tau))) # realised tilt
            x_r=float(c2*tau[-1])
            rb_r=R_PHI*0.5*W*arm*phi_r**2
            # a priori at the design cap for this rate
            x_d=brentq(lambda v: np.sinh(v)-v-PHI_D*W*Z*c2/md, 1e-3, 40)
            rb_d=R_PHI*0.5*W*arm*PHI_D**2
            rows.append(dict(rate=md,
                e=np.degrees(float(np.sqrt(np.mean(e**2)))),
                env_r=np.degrees(env_rms(rb_r,k,c2,x_r)),
                env_d=np.degrees(env_rms(rb_d,k,c2,x_d))))
    print('done',case,flush=True)

with open('docs/sim_envelope_rate.csv','w',newline='') as fh:
    wtr=csv.DictWriter(fh,fieldnames=list(rows[0]))
    wtr.writeheader(); wtr.writerows(rows)
g=defaultdict(list)
for r in rows: g[r['rate']].append(r)
print(f"\n{'rate':>6}{'n':>4}{'env@design':>11}{'env@real':>10}"
      f"{'meas e_w':>10}{'e/env_r':>9}")
for rate in sorted(g):
    v=g[rate]
    f=lambda kk: np.mean([x[kk] for x in v])
    print(f"{rate:>6.2f}{len(v):>4}{f('env_d'):>11.3f}{f('env_r'):>10.3f}"
          f"{f('e'):>10.3f}{np.mean([x['e']/x['env_r'] for x in v]):>9.2f}")
a=np.array([[r['e'],r['env_r']] for r in rows])
print(f"\nshape: corr(measured e_w, envelope@realised) = "
      f"{np.corrcoef(a[:,0],a[:,1])[0,1]:.2f} over {len(a)} runs")
