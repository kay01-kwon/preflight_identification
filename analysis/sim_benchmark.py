"""Detector benchmark on the simulation campaign, hardware-style.

Same estimators as analysis/nls_comparison.py, same aggregation; the
truth is the injected offset instead of the load cell. The reported
cosh variant here is the wide-box score calibration -- the sim has no
frozen PNLS stage-2 constants, and the wide box is what every sim
number so far was produced with.
"""
import contextlib, csv, io, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import critical_value_getter_piecewise as cvp
from utils.extractor import load_packed_dataset
from analysis.pelt_crosscheck import classic_onset_index, _window, CLASSIC

G = 9.81
Z_CAD = 0.261
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}
CASES = {'S1':(-6.0,0.0,3.066),'S2':(0.0,10.0,3.066),'S3':(10.0,-5.0,3.066),
         'S4':(20.0,20.0,3.066),'S5':(20.0,-20.0,3.066),
         'S6':(-20.0,20.0,3.066),'S7':(-20.0,-20.0,3.066),
         'S8':(25.0,25.0,3.066),'S9':(32.0,32.0,3.066),
         'S11':(38.0,14.0,3.066),'S13':(25.0,25.0,3.220)}
SIGN = {'Mx': +1.0, 'My': -1.0}
METHODS = ['cosh', 'cosh_cad', 'nls'] + CLASSIC

rows = []
for case in sorted(CASES, key=lambda c: int(c[1:])):
    tx, ty, mass = CASES[case]
    for simax, axis in (('Mx','x'), ('My','y')):
        truth = ty if simax == 'Mx' else tx
        j_cad = J_CAD[axis] + mass * (Z_CAD**2 + LP[axis]**2)
        c2_cad = float(np.sqrt(mass*G*Z_CAD/j_cad))
        k_cad = 1.0/(mass*G*Z_CAD)
        with contextlib.redirect_stdout(io.StringIO()):
            bags = load_packed_dataset(f'SimDataSet/R3/{case}/{simax}')
            crits_a,_ = cvp.extract_piecewise_batch(bags, axis)
            crits_c,_ = cvp.extract_piecewise_batch(bags, axis,
                                                    cosh_c2=c2_cad,
                                                    ramp_gain=k_cad)
        gated = {c.bag_name for c in crits_a}
        by_bag = {b.name: b for b in bags}
        res = {'cosh': {c.bag_name: c.onset_moment for c in crits_a},
               'cosh_cad': {c.bag_name: c.onset_moment for c in crits_c}}
        res.update({m: {} for m in METHODS if m not in res})
        for name in sorted(gated):
            bag = by_bag[name]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    crit,_ = cvp.extract_piecewise(bag, axis, model='cosh',
                                                   cosh_c2=None,
                                                   ramp_gain=None)
                res['nls'][name] = crit.onset_moment
            except Exception as e:
                print(f'  nls failed on {case}/{simax}/{name}: {e}',
                      flush=True)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    base,i0,i1,win,guess,direction = _window(bag, axis)
                for m in CLASSIC:
                    j = classic_onset_index(m, base.omega[win], guess,
                                            direction)
                    res[m][name] = float(base.moment[win][j])
            except Exception as e:
                print(f'  classic failed on {case}/{simax}/{name}: {e}',
                      flush=True)
        for name in sorted(gated):
            row = dict(case=case, axis=simax, bag=name,
                       truth_mm=truth, mass=mass,
                       dir='pos' if name.startswith('pos') else 'neg')
            for m in METHODS:
                row[f'mcrit_{m}'] = (f'{res[m][name]:.4f}'
                                     if name in res[m] else '')
            rows.append(row)
        print(f'done {case}/{simax}', flush=True)

with open('docs/sim_benchmark_runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# aggregate: per method, offsets vs truth over the 22 case-axis cells
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    for m in METHODS:
        if r[f'mcrit_{m}']:
            agg[(r['case'], r['axis'])][m][r['dir']].append(
                float(r[f'mcrit_{m}']))

summary = []
for (case, simax) in sorted(agg):
    tx, ty, mass = CASES[case]
    truth = ty if simax == 'Mx' else tx
    W = mass * G
    for m in METHODS:
        g = agg[(case, simax)][m]
        if not g['pos'] or not g['neg']:
            continue
        p, n = np.array(g['pos']), np.array(g['neg'])
        mff = 0.5*(p.mean()+n.mean())
        off = SIGN[simax]*1e3*mff/W
        summary.append(dict(case=case, axis=simax, method=m,
                            cv_pos=100*p.std(ddof=1)/abs(p.mean()),
                            cv_neg=100*n.std(ddof=1)/abs(n.mean()),
                            offset_mm=off, truth_mm=truth,
                            err_mm=off-truth))
with open('docs/sim_benchmark_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)

print(f"\n{'method':<12}{'RMS [mm]':>9}{'med |e|':>9}{'max |e|':>9}"
      f"{'med CV%':>9}{'max CV%':>9}")
for m in METHODS:
    e = np.array([abs(s['err_mm']) for s in summary if s['method']==m])
    cv = np.array([v for s in summary if s['method']==m
                   for v in (s['cv_pos'], s['cv_neg'])])
    print(f'{m:<12}{np.sqrt((e**2).mean()):>9.3f}{np.median(e):>9.3f}'
          f'{e.max():>9.3f}{np.median(cv):>9.2f}{cv.max():>9.1f}')
print('\nwritten docs/sim_benchmark_{runs,summary}.csv')
