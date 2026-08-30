"""Hardware envelope-witness cap with the kappa_b noise channel.

Per run: free-fit residual RMS, its >5 Hz band (disturbance by
construction -- the family cannot produce it), extended to the full
band by the campaign constant kappa_b = 1.31 measured outside the
windows (tight_rms_bound). cap = envelope(design 10 deg, GE incl.)
+ n_hi sqrt(1 + kappa_b^2).
"""
import contextlib, io, sys, csv
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from pathlib import Path

KB = 1.31
rows=[]
for d in sorted(Path('DataSet/exp').glob('case_*/M[xy]')):
    axis='x' if d.name=='Mx' else 'y'
    with contextlib.redirect_stdout(io.StringIO()):
        bags=load_excitation_dataset(d)
    for bag in bags:
        md=cvp.commanded_ramp_rate(bag.name)
        if md is None: continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                crit, pw = cvp.extract_piecewise(bag, axis, model='cosh',
                                                 cosh_c2=None, ramp_gain=None,
                                                 free_seed=False)
                sig = cvp.prepare_signals(bag, axis)
            i0,i1=cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            om = sig['omega'][i0:i1+1]
            pred = pw['omega_pred']
            n = min(len(om), len(pred))
            r = om[:n] - pred[:n]
            dt = float(np.median(np.diff(sig['t'][i0:i0+n])))
            rr = r - r.mean()
            F = np.fft.rfft(rr)
            f = np.fft.rfftfreq(len(rr), d=dt)
            F[f <= 5.0] = 0.0
            hi = np.fft.irfft(F, n=len(rr))
            rows.append(dict(case=d.parent.name, ax=d.name, rate=md,
                             res=float(np.degrees(np.sqrt(np.mean(r**2)))),
                             nhi=float(np.degrees(np.sqrt(np.mean(hi**2))))))
        except Exception as e:
            print('fail', d, bag.name, e, flush=True)
    print('done', d, flush=True)
with open('docs/hw_env_noise_runs.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader()
    w.writerows(rows)
print(len(rows), 'runs with hi-band noise')
