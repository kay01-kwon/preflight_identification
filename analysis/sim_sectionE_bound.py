"""Section VI-E validated on the simulation, in its own terms.

Per run: the kernel-free cap -- shaped model term (M2 on the arm
channel; the GE channel is zero in the simulator) plus the exact
pre-onset segment, plus the method-blind noise anchor N_n = 3 x the
campaign median SG high-band -- against the measured free-fit
residual. Constants are the design box only: phi = 5 deg, z = 0.272.
"""
import contextlib, io, sys, csv
from collections import defaultdict
import numpy as np
from scipy.signal import savgol_filter
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset
from failing_runs import split as fsplit

G=9.81; Z=0.272; R_PHI=1/7; PHI=np.deg2rad(5.0); SAFETY=1.05; FC=5.0
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

def M2f(x, n=2001):
    u=np.linspace(-x,0.0,n)
    v=np.exp(2*u)-np.sinh(u+x)/np.sinh(x)-np.exp(-2*x)*np.sinh(-u)/np.sinh(x)
    return float(np.abs(v).max()/3.0)

rows=[]; sg_anchor=[]
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
                    crit, pw = cvp.extract_piecewise(bag, axis,
                                                     model='cosh',
                                                     cosh_c2=None,
                                                     ramp_gain=None)
                    sig=cvp.prepare_signals(bag,axis)
            except Exception:
                continue
            i0,i1=cvp.detect_excitation_window(
                sig['moment'],moment_cap=cvp.MOMENT_CAP.get(axis))
            t=sig['t'][i0:i1+1]; om=sig['omega'][i0:i1+1]
            mom=sig['moment'][i0:i1+1]
            if len(t)<24: continue
            dt=float(np.median(np.diff(t)))
            mds=abs(float(np.polyfit(t,mom,1)[0]))
            j=crit.onset_idx-i0
            if j<8 or len(om)-j<8: continue
            T=float(t[-1]-t[j])
            x=min(c2*T, 30.0)
            om_max=k*mds*(np.cosh(x)-1)+rb*np.sinh(x)/(jp*c2)
            rd2=SAFETY*(W*arm*PHI)*om_max
            de=rd2*M2f(x)*k                     # /Wz = *k
            c1=k*mds
            beta=rb/(jp*c2)
            dtc=np.arctanh(min(beta/c1,0.99))/c2
            a=min(c2*dtc,30.0)
            I=1.5*dtc+np.sinh(2*a)/(4*c2)-2*np.sinh(a)/c2
            dpre=c1*np.sqrt(max(I,0.0)/T)
            # SG hi-band anchor, method-blind, over the full window
            n=len(om)
            wlen=max(int(round(2.0/(FC*dt)))|1,7)
            wlen=min(wlen, n-1 if (n-1)%2 else n-2)
            hi=fsplit(om-savgol_filter(om,wlen,3),dt)[1]
            sg_anchor.append(float(np.sqrt(np.mean(hi**2))))
            rows.append(dict(case=case,ax=simax,rate=md,
                             de=np.degrees(de+dpre),
                             res=np.degrees(float(pw['rmse']))))
    print('done',case,flush=True)

N_n=3.0*np.degrees(np.median(sg_anchor))
print(f'\nN_n = 3 x median SG hi-band = {N_n:.3f} deg/s  '
      f'({len(sg_anchor)} runs)')
g=defaultdict(list)
for r in rows:
    r['cap']=r['de']+N_n
    g[r['rate']].append(r)
print(f"{'rate':>6}{'n':>4}{'model':>8}{'cap':>8}{'rmse':>8}{'inside':>8}")
ti=tt=0
for rate in sorted(g):
    v=g[rate]
    ins=sum(r['res']<=r['cap'] for r in v); ti+=ins; tt+=len(v)
    print(f"{rate:>6.2f}{len(v):>4}"
          f"{np.mean([r['de'] for r in v]):>8.3f}"
          f"{np.mean([r['cap'] for r in v]):>8.3f}"
          f"{np.mean([r['res'] for r in v]):>8.3f}{ins:>5}/{len(v)}")
print(f"\n{ti}/{tt} inside; worst usage "
      f"{max(r['res']/r['cap'] for r in rows):.2f}")
with open('docs/sim_sectionE_runs.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader()
    w.writerows(rows)
