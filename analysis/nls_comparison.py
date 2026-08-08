#!/usr/bin/env python3
"""Estimator benchmark: dataset-calibrated COSH vs per-run alternatives.

All estimators see identical inputs — same signals, excitation window,
allocator caps and onset-free gates (the gated run set is method-independent
because the gates use the moment trace only).  They differ only in how the
onset is located per run:

  cosh         (reported) (C2, K) from the two-stage calibration: the
               per-run PNLS medians locate the pair, the ramp-invariance
               score refines it in a neighbourhood of that centre.
               Amplitude pinned (C1 = K Mdot), baseline = median, onset
               swept exhaustively.
  cosh_wide    the same score minimised over the wide box
               [3,8] x [0.05,0.70] instead, with no PNLS centre.
  cosh_cad     (C2, K) derived from the CAD geometry -- z_CoM = 0.256 m,
               Table 5 inertias, parallel-axis theorem -- nothing fitted.
  nls          per-run scipy least_squares (TRF) fit of (C1, C2, C) with
               C2 in [3, 8] rad/s, onset swept locally around the quadratic
               seed (the pipeline's historical mode).
  pelt_normal  Binary Segmentation change-point, Gaussian cost (ruptures).
  pelt_rbf     Binary Segmentation change-point, RBF-kernel cost.
  cusum        tabular CUSUM sequential detector (allowance k = 2 sigma).

Reported per case/axis and method: directional means and CVs of M_crit,
the pivot-free offset M_ff = (mean+ + mean-)/2, and the CoM offset
lambda = +/- M_ff / W against the load-cell ground truth (Table 7).

Usage: PYTHONPATH=<stubs> python analysis/nls_comparison.py [outdir]
"""
import contextlib
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import critical_value_getter_piecewise as cvp
from utils.extractor import load_excitation_dataset
from analysis.pelt_crosscheck import classic_onset_index, _window, CLASSIC
from analysis.pnls_constants import PNLS_CONSTANTS

ROOT = Path(__file__).resolve().parents[1] / 'DataSet' / 'exp'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
METHODS = ['cosh', 'cosh_wide', 'cosh_cad', 'nls'] + CLASSIC   # cosh = reported

# CAD geometry (manuscript Table 5 and the landing-gear datum), used by the
# cosh_cad variant: J_P = J_CAD + m (z_CoM^2 + l_p^2) fixes both constants.
Z_CAD = 0.256
J_CAD = {'x': 0.051085, 'y': 0.050564}
LP = {'x': 0.140, 'y': 0.110}

G = 9.81
MASS_KG = {'case_01': 3.066, 'case_02': 3.220, 'case_03': 3.220,
           'case_04': 3.220, 'case_05': 3.220}   # manuscript Table 7
# Load-cell ground truth [mm] (manuscript Table 7).  A roll excitation (Mx)
# senses the y-offset with M_ff,x = +W*y_off; a pitch excitation (My) senses
# the x-offset with M_ff,y = -W*x_off.
TRUTH_MM = {('case_01', 'Mx'): -2.90,  ('case_01', 'My'): -11.45,
            ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
            ('case_03', 'Mx'): -5.26,  ('case_03', 'My'): 3.14,
            ('case_04', 'Mx'): 6.67,   ('case_04', 'My'): 2.40,
            ('case_05', 'Mx'): 10.91,  ('case_05', 'My'): -10.89}
SIGN = {'Mx': +1.0, 'My': -1.0}

rows = []
for d in sorted(ROOT.glob('case_*/M[xy]')):
    axis = 'x' if d.name == 'Mx' else 'y'
    key0 = (d.parent.name, d.name)
    mass = MASS_KG[key0[0]]
    c2_pn, k_pn = PNLS_CONSTANTS[key0]
    j_cad = J_CAD[axis] + mass * (Z_CAD ** 2 + LP[axis] ** 2)
    c2_cad = float(np.sqrt(mass * G * Z_CAD / j_cad))
    k_cad = 1.0 / (mass * G * Z_CAD)
    with contextlib.redirect_stdout(io.StringIO()):
        bags = load_excitation_dataset(d)
        crits_a, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2_pn,
                                                 ramp_gain=k_pn)   # reported
        crits_w, _ = cvp.extract_piecewise_batch(bags, axis)       # wide box
        crits_c, _ = cvp.extract_piecewise_batch(bags, axis, cosh_c2=c2_cad,
                                                 ramp_gain=k_cad)  # CAD
    gated = {c.bag_name for c in crits_a}
    by_bag = {b.name: b for b in bags}
    res = {'cosh': {c.bag_name: c.onset_moment for c in crits_a},
           'cosh_wide': {c.bag_name: c.onset_moment for c in crits_w},
           'cosh_cad': {c.bag_name: c.onset_moment for c in crits_c}}
    res.update({m: {} for m in METHODS if m not in res})
    for name in sorted(gated):
        bag = by_bag[name]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                crit, _ = cvp.extract_piecewise(
                    bag, axis, model='cosh', cosh_c2=None, ramp_gain=None)
            res['nls'][name] = crit.onset_moment
        except Exception as e:
            print(f"  nls failed on {name}: {e}", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                base, i0, i1, win, guess, direction = _window(bag, axis)
            for m in CLASSIC:
                j = classic_onset_index(m, base.omega[win], guess, direction)
                res[m][name] = float(base.moment[win][j])
        except Exception as e:
            print(f"  classic failed on {name}: {e}", flush=True)
    for name in sorted(gated):
        row = dict(case=d.parent.name, axis=d.name, bag=name,
                   rate=cvp.commanded_ramp_rate(name),
                   dir='pos' if name.startswith('pos') else 'neg')
        for m in METHODS:
            row[f'mcrit_{m}'] = (f"{res[m][name]:.4f}"
                                 if name in res[m] else '')
        rows.append(row)
    print(f"done {d.parent.name}/{d.name}", flush=True)

with open(OUT / 'nls_comparison_runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    for m in METHODS:
        if r[f'mcrit_{m}']:
            agg[(r['case'], r['axis'])][m][r['dir']].append(
                float(r[f'mcrit_{m}']))

print(f"\n{'case':8} {'ax':3} {'method':12} {'mean_neg':>9} {'mean_pos':>9} "
      f"{'CV_neg':>7} {'CV_pos':>7} {'M_ff':>8} {'offset_mm':>9} "
      f"{'truth_mm':>8} {'err_mm':>7}")
summary = []
for key in sorted(agg):
    for m in METHODS:
        g = agg[key][m]
        if 'pos' not in g or 'neg' not in g:
            continue
        mn, mp = np.mean(g['neg']), np.mean(g['pos'])
        cvn = np.std(g['neg'], ddof=1) / abs(mn)
        cvp_ = np.std(g['pos'], ddof=1) / abs(mp)
        mff = 0.5 * (mp + mn)
        off = SIGN[key[1]] * 1e3 * mff / (MASS_KG[key[0]] * G)
        truth = TRUTH_MM.get(key)
        err = off - truth if truth is not None else None
        summary.append(dict(case=key[0], axis=key[1], method=m,
                            mean_neg=f"{mn:.4f}", mean_pos=f"{mp:.4f}",
                            cv_neg=f"{cvn:.4f}", cv_pos=f"{cvp_:.4f}",
                            M_ff=f"{mff:+.4f}", offset_mm=f"{off:+.3f}",
                            truth_mm=(f"{truth:+.2f}" if truth is not None
                                      else ''),
                            err_mm=(f"{err:+.3f}" if err is not None else '')))
        print(f"{key[0]:8} {key[1]:3} {m:12} {mn:>9.4f} {mp:>9.4f} "
              f"{cvn:>7.4f} {cvp_:>7.4f} {mff:>+8.4f} {off:>+9.3f} "
              f"{truth:>+8.2f} {err:>+7.3f}")

with open(OUT / 'nls_comparison_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

print()
for m in METHODS:
    e = np.array([float(s['err_mm']) for s in summary
                  if s['method'] == m and s['err_mm']])
    cvs = np.array([float(s[c]) for s in summary if s['method'] == m
                    for c in ('cv_neg', 'cv_pos')])
    lag = np.array([1e2 * (abs(float(r[f'mcrit_{m}']))
                           - abs(float(r['mcrit_cosh'])))
                    / abs(float(r['mcrit_cosh']))
                    for r in rows if r[f'mcrit_{m}'] and r['mcrit_cosh']])
    print(f"{m:12} |CoM err| median {np.median(np.abs(e)):.2f} / RMS "
          f"{np.sqrt(np.mean(e**2)):.2f} / max {np.max(np.abs(e)):.2f} mm ; "
          f"median CV {np.median(cvs)*100:.1f}% worst {np.max(cvs)*100:.1f}% ; "
          f"mean |M| lag vs cosh {np.mean(lag):+.1f}%")
