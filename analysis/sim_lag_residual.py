"""Does a 10 ms estimator lag account for the excitation residual?

Per run: the pinned-theory-fit residual (as in the bound test) against
the prediction dt * RMS(omega_dot) with dt = 10 ms measured from the
free-flight IMU/odom comparison.
"""
import contextlib, io, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset

G=9.81; Z=0.272; DT_LAG=0.010
J_CAD={'x':0.051085,'y':0.050564}
CASES={'S1':3.066,'S2':3.066,'S3':3.066,'S4':3.066,'S5':3.066,'S6':3.066,
       'S7':3.066,'S8':3.066,'S9':3.066,'S11':3.066,'S13':3.220}
L={'Mx':0.110,'My':0.140}
g=defaultdict(list)
for case, mass in CASES.items():
    W=mass*G
    for simax,axis in (('Mx','x'),('My','y')):
        jp=J_CAD[axis]+mass*(Z**2+L[simax]**2)
        c2=float(np.sqrt(W*Z/jp)); k=1.0/(W*Z)
        with contextlib.redirect_stdout(io.StringIO()):
            bags=load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
        for bag in bags:
            rate=cvp.commanded_ramp_rate(bag.name)
            if rate is None: continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig=cvp.prepare_signals(bag,axis)
            i0,i1=cvp.detect_excitation_window(
                sig['moment'],moment_cap=cvp.MOMENT_CAP.get(axis))
            w=slice(i0,i1+1)
            t,om,mom=sig['t'][w],sig['omega'][w],sig['moment'][w]
            if len(t)<24: continue
            md=float(np.polyfit(t,mom,1)[0])
            pw=cvp.cosh_onset_fit(t,om,np.zeros_like(t),onset_guess=None,
                                  c2_fixed=c2,moment_floor=0.0,
                                  ramp_gain=k,ramp_rate=md)
            j=pw['onset_idx']
            if j<12 or len(om)-j<12: continue
            tau=t[j:]-t[j]
            r=om[j:]-pw['omega_pred'][j:]
            wdot=np.gradient(om[j:],tau)
            g[rate].append((np.degrees(np.sqrt(np.mean(r**2))),
                            np.degrees(DT_LAG*np.sqrt(np.mean(wdot**2)))))
    print('done',case,flush=True)
print(f"\n{'rate':>6}{'n':>4}{'measured':>10}{'10ms*wdot':>11}{'ratio':>7}")
for rate in sorted(g):
    v=np.array(g[rate])
    print(f"{rate:>6.2f}{len(v):>4}{v[:,0].mean():>10.3f}"
          f"{v[:,1].mean():>11.3f}{np.mean(v[:,0]/v[:,1]):>7.2f}")
a=np.concatenate([np.array(g[r]) for r in g])
print(f"\ncorrelation(measured, predicted) over {len(a)} runs: "
      f"{np.corrcoef(a[:,0],a[:,1])[0,1]:.2f}")
