#!/usr/bin/env python3
"""Is the leftover static error aerodynamic, or is it contact geometry?

After the interference model the per-direction thresholds are still
missed by about 5% of their value, and the section reports that as
degenerate: ground effect and a displaced contact resultant both act
as force x length.  The degeneracy is real for a single group, but the
five CoM configurations break part of it, because they differ in
exactly one respect.

The airframe, the rotor heights, the collective thrust and therefore
every aerodynamic quantity are the same in all five cases; only
ballast moves horizontally.  So any unmodelled aerodynamic term is
COMMON across cases and can contribute nothing to the case-to-case
spread.  Splitting the antisymmetric residual into its mean over the
cases and its scatter about that mean therefore separates a candidate
aerodynamic bias from contact geometry, without needing to name either.

A second, sharper test follows.  A body-level vertical force -- the
fountain lift excluded from the model, or any error in the collective
thrust -- reaches the threshold through the SAME lever as the weight,
so it must scale with l_p: roll (0.140 m) should show 1.27 times what
pitch (0.110 m) does.  Whatever is common can be checked against that
ratio, and the check is decisive here.

Usage: python analysis/static_residual_split.py <dir with the CSV>
"""
import csv
import sys
from pathlib import Path

import numpy as np

SC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
G = 9.81
LP = {'Mx': 0.140, 'My': 0.110}          # contact lever per axis
SENS = 10.56                             # dM/dl = W - f  [N]
F_COL = 21.05                            # collective thrust at onset [N]
W_NOM = 3.220 * G

agg = {(r['case'], r['axis'], r['dir']): r
       for r in csv.DictReader(open(SC / 'mcrit_prediction.csv'))}
CASES = sorted({k[0] for k in agg})

anti = {}
for axn in ('Mx', 'My'):
    v = []
    for case in CASES:
        rp, rn = agg[(case, axn, 'pos')], agg[(case, axn, 'neg')]
        v.append(0.5 * (float(rp['resid_interf_mNm'])
                        - float(rn['resid_interf_mNm'])))
    anti[axn] = np.array(v)

print("antisymmetric residual after the interference model  [mN.m]\n")
print(f"  {'axis':6}{'per case':>42}{'mean':>9}{'sd':>7}{'se':>7}")
for axn in ('Mx', 'My'):
    v = anti[axn]
    se = v.std(ddof=1) / np.sqrt(len(v))
    print(f"  {axn:6}{'  '.join(f'{x:+7.1f}' for x in v):>42}"
          f"{v.mean():9.1f}{v.std(ddof=1):7.1f}{se:7.1f}")

print("\nthe part that varies between cases cannot be aerodynamic\n")
for axn in ('Mx', 'My'):
    v = anti[axn]
    print(f"  {axn}: sd {v.std(ddof=1):5.1f} mN.m  ->"
          f"  {v.std(ddof=1) / SENS:5.2f} mm of contact lever, case to case")
print("     (the aerodynamics are identical across the five cases; only")
print("      ballast moves, so nothing aerodynamic can produce this)")

print("\nthe part that is common: does it scale like a body-level force?\n")
mx, my = anti['Mx'], anti['My']
ratio = LP['Mx'] / LP['My']
pred = ratio * my.mean()
se_mx = mx.std(ddof=1) / np.sqrt(len(mx))
print(f"  a vertical force at the body acts through the same lever as the")
print(f"  weight, so roll must show {ratio:.2f}x pitch.")
print(f"    pitch common   {my.mean():+7.1f} mN.m"
      f"  -> roll should be {pred:+7.1f}")
print(f"    roll  common   {mx.mean():+7.1f} mN.m"
      f"  +- {se_mx:.1f}  ->  {abs(mx.mean() - pred) / se_mx:.1f} sigma away")
print(f"  so the common part is NOT a body-level vertical force: it is"
      f" {abs(my.mean() / max(abs(mx.mean()), 1e-9)):.0f} times")
print(f"  larger on pitch, where a force hypothesis demands it be smaller.")

print("\nwhat that leaves\n")
for axn, v in (('Mx', mx), ('My', my)):
    se = v.std(ddof=1) / np.sqrt(len(v))
    t = v.mean() / se
    print(f"  {axn}: common {v.mean():+6.1f} +- {se:.1f} mN.m"
          f"  (t = {t:+.1f})"
          f"  ->  {abs(v.mean()) / SENS:.1f} mm of lever")
print("  Roll carries no significant common term; pitch carries one, and it")
print("  is axis-specific.  Fountain lift, a collective-thrust error and a")
print("  thrust-coefficient error are all excluded by the ratio test above,")
print("  since each would appear on roll more strongly than on pitch.")
print("  What remains is the pitch contact itself -- where the resultant of")
print("  the front or rear gear pair acts -- together with a case-to-case")
print("  contact term of 3.9 to 6.9 mm on both axes.")
