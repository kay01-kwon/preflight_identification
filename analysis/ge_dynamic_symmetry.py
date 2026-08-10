#!/usr/bin/env python3
"""The dynamic ground-effect check, split the way the method uses it.

Read per direction, the dynamic inversion looks hopeless: the level at
the onset runs from -0.67x to 2.64x of the parameter-free model across
the four axis x tip-direction classes.  That spread is real -- it is in
the raw per-sample values, not in any fit -- but it is almost entirely
ANTISYMMETRIC in the tip direction.  On My the pos and neg groups sit
at +346 and -366 mN.m, at +237 and -259, and so on, summing to within
90 mN.m of zero in four of the five cases.

Expressed as a length the antisymmetric half is +7.7 +- 2.2 mm on My
and -1.4 +- 1.9 mm on Mx: constant across cases, hence a property of
the rig's geometry about each contact line rather than of the vehicle.
It is NOT the assumed CoM offset -- regressed against OFF_MM the slope
is 0.14, not 1.

This matters because the deliverable is the pivot-free average

    M_ff = sign * 0.5 * (M_pos + M_neg),    offset = M_ff / W

so the antisymmetric part never reaches the identified offset.  Only
the symmetric half does, and there the check passes: residual median
-11 mN.m, RMS 55, max 111 over the ten case/axis groups, against a
model of 139-184 mN.m.  The static check on the same groups gives +41,
73 and 129.

Ruled out along the way, each with its number, so they are not retried:

  pivot arm            run-to-run mocap scatter is 1.5 mm (median over
                       groups, 7.3 worst); substituting the group
                       median for every run removes 2% of the spread
  moment at the onset  within +5 to +40 mN.m of the COSH-identified
                       threshold, so the two checks read the same m
  fitting method       per-run medians, pooled-within-class and ANCOVA
                       agree on the class ratios to two decimals
  the derivative       J_P omega_dot is identically zero at the first
                       sample (the polynomial is onset-anchored), and
                       the balance is already off by 300-520 mN.m there

Usage:
  HD_DERIV=polyk:6 HD_GAIN=0.890 HD_DUMP=hd.npz \
      python analysis/heave_damping.py
  python analysis/ge_dynamic_symmetry.py hd.npz [mcrit_prediction.csv]
"""
import csv
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('hd.npz')
CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else None

# the assumed CoM offsets the balance is built with, and their axis sign
OFF_MM = {('case_01', 'Mx'): -2.90, ('case_01', 'My'): -11.45,
          ('case_02', 'Mx'): -14.29, ('case_02', 'My'): -9.90,
          ('case_03', 'Mx'): -5.26, ('case_03', 'My'): 3.14,
          ('case_04', 'Mx'): 6.67, ('case_04', 'My'): 2.40,
          ('case_05', 'Mx'): 10.91, ('case_05', 'My'): -10.89}
OFF_SIGN = {'Mx': +1.0, 'My': -1.0}
W = 31.59                       # N
BAND = 0.4                      # deg of excursion counted as "the onset"

d = np.load(SRC)
for k in ('bag', 'tip'):
    if k not in d:
        sys.exit(f"{SRC} predates the tip-direction dump. Re-run "
                 f"heave_damping.py with HD_DUMP set.")
rid, phi = d['rid'], d['phi']
res, mod = d['resid'], d['model']            # both already mN.m
grp = np.array([f"{c}/{a}/{t}"
                for c, a, t in zip(d['case'], d['axis'], d['tip'])])
CASES = sorted(set(d['case']))


def onset(group, y):
    """median of y over the onset band of one case/axis/direction group"""
    m = (grp[rid] == group) & (phi >= 0) & (phi < BAND)
    return np.median(y[m]) if m.sum() else np.nan


print(f"{SRC.name}: {len(np.unique(rid))} runs, onset band 0-{BAND} deg\n")
print("residual (inversion - model) at the onset, by tip direction "
      "[mN.m]\n")
print(f"  {'case/axis':14}{'pos':>8}{'neg':>8}{'anti/2':>9}{'-> mm':>8}"
      f"{'OFF_MM':>9}{'sym':>8}{'model':>8}")
anti_mm, off_mm, sym, mods = [], [], [], []
for case in CASES:
    for axn in ('Mx', 'My'):
        k = f'{case}/{axn}'
        p, n = onset(k + '/pos', res), onset(k + '/neg', res)
        a2 = 0.5 * (p - n)
        s = 0.5 * (p + n)
        mo = 0.5 * (onset(k + '/pos', mod) + onset(k + '/neg', mod))
        off = OFF_MM[(case, axn)] * OFF_SIGN[axn]
        anti_mm.append(a2 * 1e-3 / W * 1e3)
        off_mm.append(off)
        sym.append(s)
        mods.append(mo)
        print(f"  {k:14}{p:8.0f}{n:8.0f}{a2:9.0f}{anti_mm[-1]:8.1f}"
              f"{off:9.2f}{s:8.0f}{mo:8.0f}")

anti_mm = np.array(anti_mm)
off_mm = np.array(off_mm)
sym = np.array(sym)
mods = np.array(mods)

print("\nthe antisymmetric half -- removed by the pivot-free average")
for k, axn in ((0, 'Mx'), (1, 'My')):
    v = anti_mm[k::2]
    print(f"  {axn}: {np.mean(v):+6.2f} +- {np.std(v):.2f} mm"
          f"   (constant across cases -> a property of the rig)")
print(f"  against the assumed CoM offset: corr "
      f"{np.corrcoef(off_mm, anti_mm)[0, 1]:+.3f}, slope "
      f"{np.polyfit(off_mm, anti_mm, 1)[0]:+.3f}"
      f"  -- 1.0 would mean it IS the offset")

print("\nthe symmetric half -- the only part that reaches the result")
print(f"  median {np.median(sym):+7.1f}   RMS {np.sqrt(np.mean(sym**2)):6.1f}"
      f"   max |{np.max(np.abs(sym)):.0f}|   against a model of "
      f"{mods.min():.0f}-{mods.max():.0f} mN.m")

if CSV and CSV.exists():
    S = {}
    for r in csv.DictReader(open(CSV)):
        S.setdefault(f"{r['case']}/{r['axis']}", []).append(
            abs(float(r['M_pred_interf'])) - abs(float(r['M_ident'])))
    st = np.array([np.mean(S[f'{c}/{a}']) * 1e3
                   for c in CASES for a in ('Mx', 'My')])
    print(f"  the static check on the same groups, direction-averaged:")
    print(f"  median {np.median(st):+7.1f}   RMS {np.sqrt(np.mean(st**2)):6.1f}"
          f"   max |{np.max(np.abs(st)):.0f}|")
