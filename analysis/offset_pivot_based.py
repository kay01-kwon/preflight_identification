#!/usr/bin/env python3
"""The pivot-based CoM offset, per direction and per configuration.

Eq. (35) estimates the offset from each tip direction separately, using
that direction's own contact arm, and Eq. (34) supplies the weight the
inversion needs from the moments themselves:

    W_Mx = f_crit + (M_x+ - M_x-) / (l_r + l_l)
    y_off,+ = M_x+/W - (1 - f/W) l_r ,   y_off,- = M_x-/W + (1 - f/W) l_l

so the route needs no independent weight measurement at all.  Averaging
the two directions recovers the pivot-free combination corrected for
the arm asymmetry, which is the quantity compared against the load
cell here.

Three things are printed, because each can fail on its own:

  1. the weight Eq. (34) returns, against mg.  It is not used as a
     weight anywhere else in the pipeline, so a disagreement here is a
     free diagnostic on the arms and the onset thrust.
  2. the two directional estimates and their spread.  Eq. (35) gives
     the same offset twice; how far apart they land is the internal
     consistency of the contact model, before any comparison to truth.
  3. the average against the load-cell truth, with and without the
     ground-effect channels.

Usage: python analysis/offset_pivot_based.py <dir with the CSV> [out.csv]
"""
import csv
import sys
from pathlib import Path

import numpy as np

SC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
G = 9.81
MASS = {'case_01': 3.066, **{f'case_0{i}': 3.220 for i in range(2, 6)}}
TRUTH = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
         ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
         ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
         ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
         ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
SGN = {'Mx': +1.0, 'My': -1.0}
COMP = {'Mx': 'y_off', 'My': 'x_off'}
GE = {'none': (0.0, 0.0), 'interf': (0.0431, 0.04314)}

agg = {(r['case'], r['axis'], r['dir']): r
       for r in csv.DictReader(open(SC / 'mcrit_prediction.csv'))}
CASES = sorted({k[0] for k in agg})
UNITS = [(c, a) for c in CASES for a in ('Mx', 'My')]


def parts(case, axn):
    rp, rn = agg[(case, axn, 'pos')], agg[(case, axn, 'neg')]
    return (float(rp['M_ident']), float(rn['M_ident']),
            float(rp['f_onset']), float(rn['f_onset']),
            float(rp['l_odom_mm']) * 1e-3, float(rn['l_odom_mm']) * 1e-3)


def weight_eq34(case, axn, ca=0.0, b=0.0):
    """Eq. (34) with the ground-effect channels restored.

    With M(1+b) + sgn c_a f l = sgn (W - f) l + S_off, differencing the
    two directions removes S_off and leaves
    W = (1 + c_a) f + (1 + b) (M_+ - M_-) / (l_+ + l_-).
    """
    mp, mn, fp, fn, lp, ln = parts(case, axn)
    return (1 + ca) * 0.5 * (fp + fn) + (1 + b) * (mp - mn) / (lp + ln)


def directional(case, axn, ca, b, W):
    """Eq. (35) per direction, with the ground-effect channels."""
    mp, mn, fp, fn, lp, ln = parts(case, axn)
    op = ((1 + b) * mp + ca * fp * lp) / W - (1 - fp / W) * lp
    on = ((1 + b) * mn - ca * fn * ln) / W + (1 - fn / W) * ln
    return SGN[axn] * op * 1e3, SGN[axn] * on * 1e3


print("1. the weight Eq. (34) returns, against mg\n")
print(f"  {'case':9}{'axis':5}{'mg':>8}{'no GE':>9}{'err':>7}"
      f"{'interference':>15}{'err':>7}")
for c, a in UNITS:
    W = MASS[c] * G
    w0 = weight_eq34(c, a)
    w1 = weight_eq34(c, a, *GE['interf'])
    print(f"  {c:9}{a:5}{W:8.2f}{w0:9.2f}{w0 - W:7.2f}"
          f"{w1:15.2f}{w1 - W:7.2f}")
d0 = np.array([weight_eq34(c, a) - MASS[c] * G for c, a in UNITS])
d1 = np.array([weight_eq34(c, a, *GE['interf']) - MASS[c] * G
               for c, a in UNITS])
print(f"  {'':14}{'mean':>8}{'':9}{d0.mean():7.2f}{'':15}{d1.mean():7.2f}"
      f"   [N]")
print(f"  {'':14}{'':8}{'':9}{100 * d0.mean() / 31.2:6.1f}%{'':15}"
      f"{100 * d1.mean() / 31.2:6.1f}%")

rows = []
for lab, (ca, b) in GE.items():
    print(f"\n2/3. Eq. (35) per direction and averaged --- ground effect:"
          f" {lab}\n")
    print(f"  {'case':9}{'comp':7}{'from +':>9}{'from -':>9}{'spread':>9}"
          f"{'average':>10}{'truth':>8}{'error':>8}")
    err = []
    for c, a in UNITS:
        W = MASS[c] * G
        op, on = directional(c, a, ca, b, W)
        av, t = 0.5 * (op + on), TRUTH[(c, a)]
        err.append(av - t)
        print(f"  {c:9}{COMP[a]:7}{op:9.2f}{on:9.2f}{op - on:9.2f}"
              f"{av:10.2f}{t:8.2f}{av - t:8.2f}")
        rows.append(dict(case=c, axis=a, component=COMP[a], ge=lab,
                         W_eq34=f"{weight_eq34(c, a, ca, b):.3f}",
                         off_pos=f"{op:+.3f}", off_neg=f"{on:+.3f}",
                         off_avg=f"{av:+.3f}", truth=f"{t:+.2f}",
                         error=f"{av - t:+.3f}"))
    e = np.array(err)
    sp = np.array([abs(np.subtract(*directional(c, a, ca, b, MASS[c] * G)))
                   for c, a in UNITS])
    print(f"  {'':9}{'':7}{'':9}{'':9}{np.median(sp):9.2f}"
          f"{'RMS':>10}{'':8}{np.sqrt((e ** 2).mean()):8.2f}")
    print(f"  {'':9}{'':7}{'':9}{'':9}{'(median)':>9}{'mean':>10}{'':8}"
          f"{e.mean():+8.2f}")

if OUT:
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {OUT}  ({len(rows)} rows)")
