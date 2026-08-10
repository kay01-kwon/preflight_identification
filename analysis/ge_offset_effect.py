#!/usr/bin/env python3
"""What the static ground effect does to the delivered CoM offset.

The error budget propagates the ground-effect moment as a perturbation
of the response SHAPE.  That is not the only way it can reach the
answer.  It also enters the static balance the offset is read from,
and there the arithmetic is elementary and exact.

With the pivot-moment decomposition Delta M_GE = sgn a + b M,
a = c_a f l, the onset balance is

    M (1 + b) + sgn a = sgn (W - f) l + S_off ,

so pairing the two tip directions and solving for the offset term,

    S_off = (1 + b) M_ff + (a_+ - a_-)/2 - [(W-f_+)l_+ - (W-f_-)l_-]/2
            \_________/   \___________/   \___________________________/
             moment ch.     thrust ch.       contact-arm asymmetry

with M_ff = (M_+ + M_-)/2.  Two things follow, and they pull in
opposite directions as evidence.

The thrust channel is antisymmetric and cancels: what survives is the
direction asymmetry of f l, and it is small.  But the
moment-proportional channel does NOT cancel -- it multiplies M_ff, so
it scales the delivered offset by (1 + b) whatever the geometry.  With
the reported interference coefficients that is 4.31%, which on these
configurations is a few tenths of a millimetre: not negligible beside
a 1.64 mm validation RMS, and a bias rather than scatter.

So the honest question is whether applying it improves the agreement
with the load-cell truth.  It does not.  The corrected offsets are
farther from truth than the uncorrected ones, by an amount inside the
scatter -- which is the same degeneracy the static section reports,
seen from the deliverable's side: something else of comparable size
acts the other way, and this dataset cannot separate them.  The
correction is therefore reported as a bound on the channel, not
applied.

Usage: PYTHONPATH=<stubs> python analysis/ge_offset_effect.py <dir>
       where <dir> holds mcrit_prediction.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

SC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
G = 9.81
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
OFF = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
       ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
       ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
       ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
       ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
SGN = {'Mx': +1.0, 'My': -1.0}            # S_off = +W y_off / -W x_off
GE = {'single': (0.0103, 0.01026), 'interf': (0.0431, 0.04314)}

agg = {(r['case'], r['axis'], r['dir']): r
       for r in csv.DictReader(open(SC / 'mcrit_prediction.csv'))}
CASES = sorted({k[0] for k in agg})

print("the static ground-effect channels at the deliverable  [mm]\n")
print(f"  {'case':9}{'axis':5}{'ident':>8}{'truth':>8}{'err':>7}"
      f"{'moment ch.':>12}{'thrust ch.':>12}{'corrected':>11}{'err':>7}")
err0, err1 = [], []
for c in CASES:
    for axn in ('Mx', 'My'):
        rp, rn = agg[(c, axn, 'pos')], agg[(c, axn, 'neg')]
        W = MASS[c] * G
        mff = 0.5 * (float(rp['M_ident']) + float(rn['M_ident']))
        ca, b = GE['interf']
        ap = ca * float(rp['f_onset']) * float(rp['l_odom_mm']) * 1e-3
        an = ca * float(rn['f_onset']) * float(rn['l_odom_mm']) * 1e-3
        mom = b * mff                                   # scales with M_ff
        thr = 0.5 * (ap - an)                           # survives asymmetry
        ident = SGN[axn] * mff / W * 1e3
        corr = SGN[axn] * (mff + mom + thr) / W * 1e3
        tru = OFF[(c, axn)]
        err0.append(ident - tru)
        err1.append(corr - tru)
        print(f"  {c:9}{axn:5}{ident:8.2f}{tru:8.2f}{ident - tru:7.2f}"
              f"{SGN[axn] * mom / W * 1e3:12.3f}"
              f"{SGN[axn] * thr / W * 1e3:12.3f}{corr:11.2f}"
              f"{corr - tru:7.2f}")

e0, e1 = np.array(err0), np.array(err1)
print(f"\n  {'':14}{'RMS':>8}{'mean':>8}{'max':>8}   [mm]")
print(f"  {'as delivered':14}{np.sqrt((e0 ** 2).mean()):8.2f}"
      f"{e0.mean():8.2f}{np.abs(e0).max():8.2f}")
print(f"  {'GE-corrected':14}{np.sqrt((e1 ** 2).mean()):8.2f}"
      f"{e1.mean():8.2f}{np.abs(e1).max():8.2f}")
for name, (ca, b) in GE.items():
    sh = np.array([abs(b * 0.5 * (float(agg[(c, a, 'pos')]['M_ident'])
                                  + float(agg[(c, a, 'neg')]['M_ident'])))
                   / (MASS[c] * G) * 1e3
                   for c in CASES for a in ('Mx', 'My')])
    print(f"\n  {name:8} moment channel b = {100 * b:.3f}%  ->  offset shift"
          f" {np.median(sh):.3f} mm median, {sh.max():.3f} max")
ca = GE['interf'][0]
thr = []
for c in CASES:
    for a in ('Mx', 'My'):
        rp, rn = agg[(c, a, 'pos')], agg[(c, a, 'neg')]
        d = 0.5 * ca * (float(rp['f_onset']) * float(rp['l_odom_mm'])
                        - float(rn['f_onset']) * float(rn['l_odom_mm']))
        thr.append(abs(d * 1e-3) / (MASS[c] * G) * 1e3)
print(f"\n  the thrust channel is antisymmetric and cancels at the pairing;"
      f"\n  what its direction asymmetry leaves is {np.median(thr):.3f} mm"
      f" median, {max(thr):.3f} max.")
