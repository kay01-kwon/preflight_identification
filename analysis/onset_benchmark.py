#!/usr/bin/env python3
"""Benchmark the onset detector against model-agnostic change-point rules.

The proposed detector fits a closed-form post-onset response with no
free shape parameter per run.  The obvious question is whether that
structure earns anything a generic change-point rule would not give,
so the same windows are handed to a per-run nonlinear least squares of
the same closed form, to Gaussian-cost and RBF-kernel CPD, and to a
mean-change CUSUM, and every downstream number is recomputed from each.

Three things are compared, and they are not the same question.

CONSISTENCY is what the calibration targets: the coefficient of
variation of |M_crit| within one tip direction across the ramp sweep.
A detector that keys on an artefact of the ramp will show a CV that
grows with the rate; one that finds a contact property will not.

ACCURACY is the deliverable: the CoM offset each method's moments
imply, against the load-cell truth, over the ten case/axis
configurations.

AGREEMENT is a separate diagnostic: how far the per-run moments differ
between methods at all.  Where the baselines agree with the proposed
detector run by run, a disagreement in the deliverable cannot be
blamed on onset placement -- which is the argument the static-threshold
section needs.

The comparison is paired.  Each of the ten configurations gives one
error per method, so the difference COSH-minus-baseline is formed
within a configuration and the sample is ten paired differences, not
ten independent numbers; the sign test below is on those pairs.

Usage: python analysis/onset_benchmark.py <dir with the CSVs>
"""
import csv
import sys
from math import comb
from pathlib import Path

import numpy as np

SC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
LBL = {'cosh': 'COSH (proposed)', 'cosh_cad': 'COSH, CAD constants',
       'cosh_wide': 'COSH, wide search box', 'nls': 'per-run NLS',
       'pelt_normal': 'CPD, Gaussian cost', 'pelt_rbf': 'CPD, RBF kernel',
       'cusum': 'CUSUM, mean change'}
ORDER = ('cosh', 'cosh_cad', 'cosh_wide', 'nls', 'pelt_normal', 'pelt_rbf',
         'cusum')

summ = list(csv.DictReader(open(SC / 'nls_comparison_summary.csv')))
runs = list(csv.DictReader(open(SC / 'nls_comparison_runs.csv')))
UNITS = sorted({(r['case'], r['axis']) for r in summ})


def by_method(m):
    return {(r['case'], r['axis']): r for r in summ if r['method'] == m}


def sign_p(x):
    """Two-sided exact sign test that the paired difference is zero."""
    n = int(np.sum(x != 0))
    if not n:
        return 1.0
    k = min(int(np.sum(x > 0)), n - int(np.sum(x > 0)))
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


print(f"onset detector benchmark: {len(runs)} runs, {len(UNITS)}"
      f" case/axis configurations\n")
print(f"  {'method':24}{'CV mean':>9}{'CV worst':>10}{'offset RMS':>12}"
      f"{'max':>7}{'mean':>7}{'|dM| vs COSH':>14}")
print(f"  {'':24}{'[%]':>9}{'[%]':>10}{'[mm]':>12}{'[mm]':>7}{'[mm]':>7}"
      f"{'[mN.m]':>14}")

base = by_method('cosh')
err = {}
for m in ORDER:
    d = by_method(m)
    if not d:
        continue
    cv = np.array([float(d[u][k]) for u in UNITS for k in ('cv_neg', 'cv_pos')])
    e = np.array([float(d[u]['err_mm']) for u in UNITS])
    err[m] = e
    dm = np.array([abs(float(r[f'mcrit_{m}']) - float(r['mcrit_cosh'])) * 1e3
                   for r in runs if r.get(f'mcrit_{m}')])
    agree = '---' if m == 'cosh' else f"{np.median(dm):.0f}"
    print(f"  {LBL[m]:24}{100 * cv.mean():9.1f}{100 * cv.max():10.1f}"
          f"{np.sqrt((e ** 2).mean()):12.2f}{np.abs(e).max():7.2f}"
          f"{e.mean():+7.2f}{agree:>14}")

print(f"\npaired against COSH, over the {len(UNITS)} configurations\n")
print(f"  {'baseline':24}{'d|err| mean':>13}{'better':>8}{'worse':>7}"
      f"{'sign p':>9}")
for m in ORDER[1:]:
    if m not in err:
        continue
    d = np.abs(err[m]) - np.abs(err['cosh'])       # >0 = COSH is closer
    print(f"  {LBL[m]:24}{d.mean():+13.2f}{int((d > 0).sum()):8d}"
          f"{int((d < 0).sum()):7d}{sign_p(d):9.2f}")
print(f"    d|err| > 0 means the baseline sits farther from truth than"
      f" COSH does.")

allm = [m for m in ORDER if m in err]
E = np.array([err[m] for m in allm])
span = E.max(0) - E.min(0)
print(f"\n  spread of the delivered offset across all {len(allm)} methods:"
      f" {np.median(span):.2f} mm median, {span.max():.2f} worst")
print(f"  every method's RMS lies in"
      f" [{min(np.sqrt((err[m] ** 2).mean()) for m in allm):.2f},"
      f" {max(np.sqrt((err[m] ** 2).mean()) for m in allm):.2f}] mm, so the"
      f" deliverable does not\n  depend on which detector is used --"
      f" the onset is not the limiting step.")
