#!/usr/bin/env python3
"""Accuracy and repeatability of the identified offset, with intervals.

The identification is currently reported as an RMS and a worst case.
That answers "how big is the error" but not "how well is it known",
which is what a reviewer asking for statistical credibility is asking.
Two different quantities are involved and they need separating.

ACCURACY is a per-configuration property: each case/axis gives one
independent comparison against the load-cell truth, so the sample is
ten, and the statistics that belong to it are the mean error with a
confidence interval and a sign test on the bias.

REPEATABILITY is a within-configuration property: the offset can be
formed from every pairing of a positive-tip run with a negative-tip
run, giving n_+ * n_- estimates per case/axis from the same hardware
state.  Their spread is what a repeat of the procedure would return,
and it is NOT the accuracy -- a configuration can be repeatable and
biased at once, which is exactly what happens here.

Reporting both, and their ratio, says where the error lives: if the
between-configuration spread far exceeds the within, the error is
systematic per configuration and no number of repeats will reduce it.

Usage: python analysis/offset_accuracy_stats.py <dir with the CSVs>
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

SC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
G = 9.81
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
TRUTH = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
         ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
         ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
         ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
         ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
SGN = {'Mx': +1.0, 'My': -1.0}            # S_off = +W y_off / -W x_off
COMP = {'Mx': 'y_off', 'My': 'x_off'}
RNG = np.random.RandomState(20240810)     # fixed: the CI must be stable

runs = list(csv.DictReader(open(SC / 'mcrit_per_run.csv')))
CASES = sorted({r['case'] for r in runs})


def moments(case, axn, dirn):
    return np.array([float(r['M_ident']) for r in runs
                     if r['case'] == case and r['axis'] == axn
                     and r['dir'] == dirn])


def signrank_p(x):
    """Two-sided exact sign test on the median being zero (n <= 25)."""
    from math import comb
    n = int(np.sum(x != 0))
    k = int(np.sum(x > 0))
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


print("identified CoM offset: accuracy and repeatability  [mm]\n")
print(f"  {'case':9}{'comp':7}{'ident':>8}{'truth':>8}{'error':>8}"
      f"{'pairs':>7}{'repeat sd':>11}{'err/sd':>8}{'range':>16}")
err, rep = [], []
for case in CASES:
    for axn in ('Mx', 'My'):
        W = MASS[case] * G
        mp, mn = moments(case, axn, 'pos'), moments(case, axn, 'neg')
        # every positive run against every negative run: the same
        # estimator the pipeline forms, over the runs of one state
        pair = np.array([SGN[axn] * 0.5 * (a + b) / W * 1e3
                         for a, b in itertools.product(mp, mn)])
        ident = SGN[axn] * 0.5 * (mp.mean() + mn.mean()) / W * 1e3
        tru = TRUTH[(case, axn)]
        err.append(ident - tru)
        rep.append(pair.std(ddof=1))
        print(f"  {case:9}{COMP[axn]:7}{ident:8.2f}{tru:8.2f}"
              f"{ident - tru:8.2f}{len(pair):7d}{pair.std(ddof=1):11.2f}"
              f"{abs(ident - tru) / pair.std(ddof=1):8.1f}"
              f"   [{pair.min():6.2f},{pair.max():6.2f}]")

e, r = np.array(err), np.array(rep)
n = len(e)
se = e.std(ddof=1) / np.sqrt(n)
# t(0.975, 9) = 2.262; kept explicit so the script needs no scipy
tcrit = 2.262
boot = np.array([np.sqrt((RNG.choice(e, n) ** 2).mean())
                 for _ in range(20000)])

print(f"\naccuracy, over the {n} independent case/axis configurations\n")
print(f"  RMS error            {np.sqrt((e ** 2).mean()):.2f} mm"
      f"   (95% CI {np.percentile(boot, 2.5):.2f} to"
      f" {np.percentile(boot, 97.5):.2f}, 20k bootstrap)")
print(f"  mean error           {e.mean():+.2f} mm"
      f"   (95% CI {e.mean() - tcrit * se:+.2f} to"
      f" {e.mean() + tcrit * se:+.2f})")
print(f"  worst configuration  {np.abs(e).max():.2f} mm")
print(f"  sign test on the bias: {int((e < 0).sum())}/{n} negative,"
      f" p = {signrank_p(e):.2f}")
print(f"    -> the negative mean is the point estimate, but ten"
      f" configurations do\n       not establish it: the interval"
      f" includes zero.")

z = np.abs(e) / r
print(f"\n  the per-configuration biases are a different question, and"
      f" they ARE\n  established: measured against each configuration's"
      f" own repeat scatter,\n  {int((z > 3).sum())} of {n} exceed"
      f" 3 sigma (worst {z.max():.1f}), so cases that sit\n  off truth"
      f" do so far beyond what repeating them would move.")

print(f"\nrepeatability, within a configuration\n")
print(f"  run-pair sd          {np.median(r):.2f} mm median,"
      f" {r.max():.2f} worst")
print(f"  between-configuration sd of the error   {e.std(ddof=1):.2f} mm")
print(f"  ratio                {e.std(ddof=1) / np.median(r):.1f}x")
print(f"    -> the error is systematic per configuration, not run"
      f" scatter;\n       repeating a configuration cannot reduce it,"
      f" and the reported\n       accuracy is therefore limited by the"
      f" ground-contact geometry\n       rather than by the number of"
      f" excitation runs.")
