"""Calibrated (C2, K) against the CAD reference.

The manuscript defines C2 = sqrt(W z_CoM / J_P) and K = 1/(W z_CoM), so the
calibrated numbers are not free knobs and the comparison has to be reported.

Reference: the CAD model gives z_CoM = 0.261 m and the Table 5 CoM
inertias, so the parallel-axis theorem fixes J_P and with it both
constants -- nothing is fitted.  Also printed is the parallel-axis FLOOR
m (z_CoM^2 + l_p^2), the value J_P would take if the CoM inertia were
zero; an identified J_P below that floor is not physically attainable.
The per-run NLS medians are carried along for comparison.

Reproduces Table tab:phys of docs/cosh_methodology.tex.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_calibration import MASS_KG, G, J_CAD, LP

Z_CAD = 0.261
M_REF = 3.220
W_REF = M_REF * G
JP_FLOOR = {a: M_REF * (Z_CAD ** 2 + LP[a] ** 2) for a in 'xy'}
JP_CAD = {a: J_CAD[a] + JP_FLOOR[a] for a in 'xy'}
C2_CAD = {a: np.sqrt(W_REF * Z_CAD / JP_CAD[a]) for a in 'xy'}
JP_NLS = {'x': 0.257, 'y': 0.225}          # per-run NLS medians, 70 runs/axis
C2_NLS_MEDIAN = 4.765

CAL = {('case_01', 'Mx'): (8.000, 0.0600), ('case_01', 'My'): (3.875, 0.4900),
       ('case_02', 'Mx'): (6.125, 0.1100), ('case_02', 'My'): (4.500, 0.5200),
       ('case_03', 'Mx'): (3.500, 0.3000), ('case_03', 'My'): (3.875, 0.3600),
       ('case_04', 'Mx'): (3.750, 0.3000), ('case_04', 'My'): (5.625, 0.2000),
       ('case_05', 'Mx'): (6.875, 0.0900), ('case_05', 'My'): (6.625, 0.1800)}
NAME = {'x': 'roll', 'y': 'pitch'}

print(f"CAD reference: z_CoM = {Z_CAD} m, W z_CoM = {W_REF * Z_CAD:.3f} N.m, "
      f"K = {1 / (W_REF * Z_CAD):.4f}")
for a in 'xy':
    print(f"  {NAME[a]:<6} J_CAD={J_CAD[a]:.6f}  floor={JP_FLOOR[a]:.4f}  "
          f"J_P={JP_CAD[a]:.4f} kg.m^2  ->  C2={C2_CAD[a]:.3f} rad/s")

hdr = (f"\n{'case':<9}{'ax':<4}{'C2_cal':>8}{'C2_CAD':>8}{'ratio':>7}"
       f"{'1/K_cal':>9}{'1/K_CAD':>9}{'ratio':>7}"
       f"{'J_cal':>8}{'J_CAD':>8}{'ratio':>7}{'vs floor':>10}")
print(hdr)
print('-' * (len(hdr) - 1))
rows = {'x': [], 'y': []}
for key in sorted(CAL):
    ax = 'x' if key[1] == 'Mx' else 'y'
    c2, k = CAL[key]
    w = MASS_KG[key[0]] * G
    j_cal = 1.0 / (k * c2 ** 2)
    rows[ax].append((c2, 1 / k, j_cal))
    print(f"{key[0]:<9}{key[1]:<4}{c2:8.3f}{C2_CAD[ax]:8.3f}"
          f"{c2 / C2_CAD[ax]:7.2f}"
          f"{1 / k:9.2f}{W_REF * Z_CAD:9.2f}{1 / (k * W_REF * Z_CAD):7.2f}"
          f"{j_cal:8.3f}{JP_CAD[ax]:8.3f}{j_cal / JP_CAD[ax]:7.2f}"
          f"{100 * (j_cal / JP_FLOOR[ax] - 1):+9.1f}%")

print()
for ax in 'xy':
    a = np.array(rows[ax])
    refs = (C2_CAD[ax], W_REF * Z_CAD, JP_CAD[ax])
    for j, lab in enumerate(('C2 [rad/s]', '1/K = W z_CoM [N.m]',
                             'J_P = 1/(K C2^2)')):
        v = a[:, j]
        print(f"  {NAME[ax]:<6}{lab:<21} mean {v.mean():7.3f}  "
              f"CV {100 * v.std(ddof=1) / v.mean():5.1f}%  "
              f"spread {v.max() / v.min():5.2f}x  vs CAD {refs[j]:7.3f}  "
              f"({100 * (v.mean() / refs[j] - 1):+6.1f}%)")
    print(f"  {NAME[ax]:<6}{'J_P vs the floor':<21} "
          f"{100 * (a[:, 2].mean() / JP_FLOOR[ax] - 1):+6.1f}%  "
          f"(floor {JP_FLOOR[ax]:.3f}; below it means not attainable)")
    print(f"  {NAME[ax]:<6}{'J_P from per-run NLS':<21} {JP_NLS[ax]:7.3f}  "
          f"vs CAD {100 * (JP_NLS[ax] / JP_CAD[ax] - 1):+6.1f}%  "
          f"vs floor {100 * (JP_NLS[ax] / JP_FLOOR[ax] - 1):+6.1f}%")
    print(f"  {NAME[ax]:<6}{'C2 from per-run NLS':<21} "
          f"{C2_NLS_MEDIAN:7.3f}  vs CAD "
          f"{100 * (C2_NLS_MEDIAN / C2_CAD[ax] - 1):+6.1f}%")
    print()
