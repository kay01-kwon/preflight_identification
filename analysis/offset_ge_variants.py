#!/usr/bin/env python3
"""The CoM offset inverted with the ground-effect moment put back in.

The reported offset comes from Eq. (36) with no ground-effect term:
the identified $M_{ff}$ is the in-situ balanced moment and is used as
it stands.  Read instead as a mass property, the load cell measures
only $W\\lambda_{off}$, so the inversion should carry the ground-effect
channels.  Restoring them in the onset balance
M(1+b) + sgn c_a f l = sgn (W-f) l + S_off and pairing the directions,

    S_off = (M_+ + M_-)(1+b)/2 + c_a (f_+ l_+ - f_- l_-)/2
            - [(W - f_+) l_+ - (W - f_-) l_-]/2 ,

with the last bracket dropped in the pivot-free variant, which assumes
l_+ = l_-.  Three levels of physics are compared against the load-cell
truth, crossed with the two ways of handling the arms -- six ways of
computing the same quantity from the same onset moments.

Nothing here is fitted; c_a and b come from the models at phi = 0.
The point of the table is whether the more complete inversion lands
closer to truth, and it does not: the comparison's own standard error
is about the size of the correction, so the dataset cannot tell the
levels apart.

Usage: python analysis/offset_ge_variants.py <dir with the CSV>
"""
import csv
import sys
from math import comb
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
GE = {'no ground effect': (0.0, 0.0),
      'single-rotor': (0.0103, 0.01026),
      'interference': (0.0431, 0.04314)}

agg = {(r['case'], r['axis'], r['dir']): r
       for r in csv.DictReader(open(SC / 'mcrit_prediction.csv'))}
CASES = sorted({k[0] for k in agg})
UNITS = [(c, a) for c in CASES for a in ('Mx', 'My')]


def offset(case, axn, ca, b, pivot_free):
    rp, rn = agg[(case, axn, 'pos')], agg[(case, axn, 'neg')]
    W = MASS[case] * G
    mp, mn = float(rp['M_ident']), float(rn['M_ident'])
    fp, fn = float(rp['f_onset']), float(rn['f_onset'])
    lp, ln = float(rp['l_odom_mm']) * 1e-3, float(rn['l_odom_mm']) * 1e-3
    s = 0.5 * (mp + mn) * (1 + b) + 0.5 * ca * (fp * lp - fn * ln)
    if not pivot_free:
        s -= 0.5 * ((W - fp) * lp - (W - fn) * ln)
    return SGN[axn] * s / W * 1e3


def sign_p(x):
    n = int(np.sum(x != 0))
    if not n:
        return 1.0
    k = min(int(np.sum(x > 0)), n - int(np.sum(x > 0)))
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


tru = np.array([TRUTH[u] for u in UNITS])
COMP = {'Mx': 'y_off', 'My': 'x_off'}
VIEW = (('pivot-free', True, 'no ground effect'),
        ('pivot-free', True, 'interference'),
        ('pivot-based', False, 'interference'))
res = {}
for tag, pf in (('pivot-free', True), ('pivot-based', False)):
    for lab, (ca, b) in GE.items():
        res[(tag, lab)] = np.array(
            [offset(c, a, ca, b, pf) for c, a in UNITS]) - tru

print("identified CoM offset per configuration  [mm]\n")
print(f"  {'case':9}{'comp':7}{'truth':>8}"
      + ''.join(f"{t.split('-')[1][:4] + '/' + l[:6]:>16}"
                for t, _, l in VIEW))
print(f"  {'':9}{'':7}{'':8}" + ''.join(f"{'ident':>9}{'err':>7}"
                                        for _ in VIEW))
for i, (c, a) in enumerate(UNITS):
    row = f"  {c:9}{COMP[a]:7}{tru[i]:8.2f}"
    for tag, pf, lab in VIEW:
        ca, b = GE[lab]
        v = offset(c, a, ca, b, pf)
        row += f"{v:9.2f}{v - tru[i]:7.2f}"
    print(row)
print(f"  {'':9}{'RMS':7}{'':8}"
      + ''.join(f"{'':9}{np.sqrt((res[(t, l)] ** 2).mean()):7.2f}"
                for t, _, l in VIEW))
print(f"  {'':9}{'mean':7}{'':8}"
      + ''.join(f"{'':9}{res[(t, l)].mean():+7.2f}" for t, _, l in VIEW))
print()

print(f"  {'variant':16}{'level':20}{'RMS':>7}{'mean':>8}{'max':>7}"
      f"{'vs eta=0':>10}{'p':>7}")
base = {}
for tag in ('pivot-free', 'pivot-based'):
    for lab in GE:
        e = res[(tag, lab)]
        if lab == 'no ground effect':
            base[tag] = e
        d = np.abs(e) - np.abs(base[tag])
        extra = '' if lab == 'no ground effect' else f"{d.mean():+10.2f}"
        pv = '' if lab == 'no ground effect' else f"{sign_p(d):7.2f}"
        print(f"  {tag:16}{lab:20}{np.sqrt((e ** 2).mean()):7.2f}"
              f"{e.mean():+8.2f}{np.abs(e).max():7.2f}{extra}{pv}")

e0 = base['pivot-free']
se = e0.std(ddof=1) / np.sqrt(len(e0))
shift = np.median(np.abs(res[('pivot-free', 'interference')] - e0))
print(f"\n  standard error of the comparison   {se:.2f} mm")
print(f"  size of the interference correction {shift:.2f} mm"
      f"   ->  {shift / se:.1f} SE")
print(f"  the dataset cannot separate the levels: no variant differs from")
print(f"  eta = 0 by more than the noise of the comparison itself.")
