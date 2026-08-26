"""Reported bound with its own noise recipe: Phi + hi-band * sqrt(1+kb^2).

The >5 Hz content of the post-onset residual is disturbance by
construction (the cosh family cannot produce it); the quiet-window
shape ratio kb extends it to the full band, exactly as on hardware.
"""
import contextlib, io, sys, csv
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset

G = 9.81; R_PHI = 1/7; PHI_BOX = np.deg2rad(5.0); Z = 0.272
J_CAD = {'x': 0.051085, 'y': 0.050564}
CASES = {'S1':(-6.0,0.0,3.066),'S2':(0.0,10.0,3.066),'S3':(10.0,-5.0,3.066),
         'S4':(20.0,20.0,3.066),'S5':(20.0,-20.0,3.066),
         'S6':(-20.0,20.0,3.066),'S7':(-20.0,-20.0,3.066),
         'S8':(25.0,25.0,3.066),'S9':(32.0,32.0,3.066),
         'S11':(38.0,14.0,3.066),'S13':(25.0,25.0,3.220)}
L = {'Mx': 0.110, 'My': 0.140}

def split(v, dt, fc=5.0):
    vv = v - v.mean()
    F = np.fft.rfft(vv)
    f = np.fft.rfftfreq(len(vv), d=dt)
    Fl = F.copy(); Fl[f > fc] = 0.0
    lo = np.fft.irfft(Fl, n=len(vv))
    return (float(np.sqrt(np.mean(lo**2))),
            float(np.sqrt(np.mean((vv - lo)**2))))

def phi_term(tau, c2, k, rb):
    jp = 1.0/(k*c2**2); N = len(tau)
    u = np.cosh(np.clip(c2*tau, 0, 30)) - 1.0
    ut = u - u.mean(); su2 = float(ut@ut)
    ds = np.gradient(tau)
    T = tau[:,None] - tau[None,:]
    Km = np.where(T >= 0, np.cosh(np.clip(c2*np.maximum(T,0),0,30))/jp,
                  0.0) * ds[None,:]
    R = Km - Km.mean(axis=0)[None,:] - ut[:,None]*((ut@Km)/su2)[None,:]
    cn = np.sqrt((R**2).sum(axis=0))
    return min(rb*(cn/np.sqrt(N)).sum(),
               rb*np.sqrt(np.sum(np.abs(R).sum(axis=1)**2)/N))

rows = []; kqs = []
for case in sorted(CASES, key=lambda c: int(c[1:])):
    tx, ty, mass = CASES[case]
    W = mass*G
    for simax, axis in (('Mx','x'), ('My','y')):
        off = abs(ty if simax=='Mx' else tx)*1e-3
        arm = L[simax] + off
        jp = J_CAD[axis] + mass*(Z**2 + L[simax]**2)
        c2 = float(np.sqrt(W*Z/jp)); k = 1.0/(W*Z)
        rb = R_PHI*0.5*W*arm*PHI_BOX**2
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
        for bag in bags:
            if cvp.commanded_ramp_rate(bag.name) is None: continue
            with contextlib.redirect_stdout(io.StringIO()):
                sig = cvp.prepare_signals(bag, axis)
            i0, i1 = cvp.detect_excitation_window(
                sig['moment'], moment_cap=cvp.MOMENT_CAP.get(axis))
            w = slice(i0, i1+1)
            t, om, mom = sig['t'][w], sig['omega'][w], sig['moment'][w]
            if len(t) < 24: continue
            md = float(np.polyfit(t, mom, 1)[0])
            pw = cvp.cosh_onset_fit(t, om, np.zeros_like(t),
                                    onset_guess=None, c2_fixed=c2,
                                    moment_floor=0.0, ramp_gain=k,
                                    ramp_rate=md)
            j = pw['onset_idx']
            if j < 12 or len(om)-j < 12: continue
            tau = t[j:]-t[j]; seg = om[j:]
            dt = float(np.median(np.diff(tau)))
            u = np.cosh(np.clip(c2*tau,0,30))-1.0
            A = np.vstack([np.ones_like(tau), u]).T
            coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
            rf = seg - A@coef
            lo, hi = split(rf, dt)
            q = sig['omega'][:i0]
            if q.size >= 60:
                ql, qh = split(q, dt)
                if qh > 0: kqs.append(ql/qh)
            rows.append(dict(case=case, ax=simax,
                             rate=cvp.commanded_ramp_rate(bag.name),
                             phi=phi_term(tau, c2, k, rb),
                             res=float(np.sqrt(np.mean(rf**2))),
                             hi=hi))
    print(f'done {case}', flush=True)

kb = max(kqs) if kqs else 1.31
print(f'\nquiet-window shape ratio: median {np.median(kqs):.2f}, '
      f'kb = max = {kb:.2f}  ({len(kqs)} windows)')
g = defaultdict(list)
for r in rows:
    r['cap'] = r['phi'] + r['hi']*np.sqrt(1+kb**2)
    g[r['rate']].append(r)
d2 = np.rad2deg
print(f"{'rate':>6}{'n':>4}{'Phi':>8}{'noise':>8}{'cap':>8}"
      f"{'resid':>8}{'used':>7}{'inside':>8}")
ti = tt = 0
for rate in sorted(g):
    v = g[rate]
    ins = int(sum(r['res'] <= r['cap'] for r in v)); ti += ins; tt += len(v)
    print(f"{rate:>6.2f}{len(v):>4}"
          f"{d2(np.mean([r['phi'] for r in v])):>8.3f}"
          f"{d2(np.mean([r['hi']*np.sqrt(1+kb**2) for r in v])):>8.3f}"
          f"{d2(np.mean([r['cap'] for r in v])):>8.3f}"
          f"{d2(np.mean([r['res'] for r in v])):>8.3f}"
          f"{np.mean([r['res']/r['cap'] for r in v]):>7.2f}{ins:>5}/{len(v)}")
print(f"\n{ti}/{tt} inside; worst usage "
      f"{max(r['res']/r['cap'] for r in rows):.2f}")
with open('docs/sim_smallangle_runs.csv', 'w', newline='') as fh:
    out = [dict(case=r['case'], ax=r['ax'], rate=r['rate'],
                cap=d2(r['cap']), res=d2(r['res'])) for r in rows]
    wtr = csv.DictWriter(fh, fieldnames=list(out[0]))
    wtr.writeheader(); wtr.writerows(out)
