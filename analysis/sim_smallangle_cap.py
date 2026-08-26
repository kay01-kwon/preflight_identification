"""Small-angle residual cap on the simulation campaign.

Same per-run test as analysis/fit_quality_bound.py: pin the calibrated
(C2, K), fit the cosh onset, and set the post-onset residual against
RMS(r) <= rho_bar K C2 sqrt(B(x)/x). rho_bar keeps only the
small-angle channel (R_PHI/7 of 0.5 W arm phi^2) -- the simulator has
no ground effect, so the GE channel would only loosen the cap. The
tilt is each run's own realised phi_end, the variant that hardware
FAILS (10/140 inside).
"""
import contextlib, io, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset

G = 9.81
R_PHI = 1.0 / 7.0
CASES = {'S1':(-6.0,0.0,3.066),'S2':(0.0,10.0,3.066),'S3':(10.0,-5.0,3.066),
         'S4':(20.0,20.0,3.066),'S5':(20.0,-20.0,3.066),
         'S6':(-20.0,20.0,3.066),'S7':(-20.0,-20.0,3.066),
         'S8':(25.0,25.0,3.066),'S9':(32.0,32.0,3.066),
         'S11':(38.0,14.0,3.066),'S13':(25.0,25.0,3.220)}
L = {'Mx': 0.110, 'My': 0.140}          # sim contact lines (verified)

def rms_cap(rb, k, c2, x):
    b = 0.25 * np.sinh(2 * x) - 0.5 * x
    return rb * k * c2 * np.sqrt(b / x)

rows = []
for case in sorted(CASES, key=lambda c: int(c[1:])):
    tx, ty, mass = CASES[case]
    W = mass * G
    for simax, axis in (('Mx','x'), ('My','y')):
        off = abs(ty if simax == 'Mx' else tx) * 1e-3
        arm = L[simax] + off
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
            c2, k = cvp.estimate_rig_constants(bags, axis)
        for bag in bags:
            rate = cvp.commanded_ramp_rate(bag.name)
            if rate is None:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig = cvp.prepare_signals(bag, axis)
            i0, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            w = slice(i0, i1 + 1)
            t, om, mom = sig['t'][w], sig['omega'][w], sig['moment'][w]
            if len(t) < 24:
                continue
            md = float(np.polyfit(t, mom, 1)[0])
            pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t),
                                    onset_guess=None, c2_fixed=c2,
                                    moment_floor=0.0, ramp_gain=k,
                                    ramp_rate=md)
            j = pw['onset_idx']
            if j < 12 or len(om) - j < 12:
                continue
            tau = t[j:] - t[j]
            r = om[j:] - pw['omega_pred'][j:]
            oc = om[j:] - float(pw['c'])
            phi_end = float(abs(np.trapezoid(oc, tau)))
            q = sig['omega'][:i0]
            noise = float(np.std(q)) if q.size > 50 else 0.0
            x = float(c2 * tau[-1])
            rb = R_PHI * 0.5 * W * arm * phi_end ** 2
            rows.append(dict(case=case, ax=simax, rate=rate, x=x,
                             phi=np.degrees(phi_end),
                             cap=np.degrees(rms_cap(rb, k, c2, x)),
                             res=np.degrees(float(np.sqrt(np.mean(r**2)))),
                             noise=np.degrees(noise)))
    print(f'done {case}', flush=True)

g = defaultdict(list)
for r in rows:
    g[r['rate']].append(r)
print(f"\n{'rate':>6}{'n':>4}{'phi[deg]':>9}{'cap[deg/s]':>11}"
      f"{'resid':>8}{'noise':>8}{'inside':>8}")
tot_in = tot = 0
for rate in sorted(g):
    v = g[rate]
    cap = np.array([r['cap'] for r in v])
    res = np.array([r['res'] for r in v])
    nz = np.array([r['noise'] for r in v])
    ins = int(np.sum(res <= cap + nz))
    tot_in += ins; tot += len(v)
    print(f"{rate:>6.2f}{len(v):>4}{np.mean([r['phi'] for r in v]):>9.2f}"
          f"{cap.mean():>11.3f}{res.mean():>8.3f}{nz.mean():>8.3f}"
          f"{ins:>5}/{len(v)}")
print(f"\n{tot_in}/{tot} runs inside the realised-tilt small-angle cap "
      f"(hardware: 10/140)")
