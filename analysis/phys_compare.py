"""Calibrated (C2, K) against the physically derived pair.

The manuscript defines C2 = sqrt(W z_CoM / J_P) and K = 1/(W z_CoM), so the
calibrated numbers are not free knobs and the comparison has to be reported.
Physical reference: J_P from the per-run NLS medians, z_CoM from CAD +
parallel axis on that same J_P.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_calibration import MASS_KG, G, J_CAD, LP

JP_NLS = {'x': 0.257, 'y': 0.225}
Z_PHYS = {a: np.sqrt((JP_NLS[a] - J_CAD[a]) / 3.220 - LP[a] ** 2) for a in 'xy'}

CAL = {('case_01','Mx'):(8.000,0.0600), ('case_01','My'):(3.875,0.4900),
       ('case_02','Mx'):(6.125,0.1100), ('case_02','My'):(4.500,0.5200),
       ('case_03','Mx'):(3.500,0.3000), ('case_03','My'):(3.875,0.3600),
       ('case_04','Mx'):(3.750,0.3000), ('case_04','My'):(5.625,0.2000),
       ('case_05','Mx'):(6.875,0.0900), ('case_05','My'):(6.625,0.1800)}

print("physical reference (NLS J_P -> CAD parallel axis -> z_CoM):")
for a in 'xy':
    print(f"  {a}: J_P={JP_NLS[a]:.3f} kg.m^2, z_CoM={Z_PHYS[a]:.3f} m")
z_phys = float(np.mean(list(Z_PHYS.values())))
print(f"  adopted z_CoM = {z_phys:.3f} m (two axes agree to "
      f"{100*abs(Z_PHYS['x']-Z_PHYS['y'])/z_phys:.1f}%)\n")

hdr = (f"{'case':<9}{'ax':<4}" + f"{'C2_cal':>8}{'C2_phys':>9}{'ratio':>7}"
       + f"{'1/K_cal':>9}{'1/K_phys':>9}{'ratio':>7}"
       + f"{'z_cal':>7}{'J_cal':>7}{'J_phys':>7}{'ratio':>7}")
print(hdr); print('-' * len(hdr))
rows = {'x': [], 'y': []}
for key in sorted(CAL):
    ax = 'x' if key[1] == 'Mx' else 'y'
    c2, k = CAL[key]
    w = MASS_KG[key[0]] * G
    c2p, kp = np.sqrt(w * z_phys / JP_NLS[ax]), 1.0 / (w * z_phys)
    j_cal, z_cal = 1.0 / (k * c2 ** 2), 1.0 / (k * w)
    rows[ax].append((c2, 1 / k, z_cal, j_cal))
    print(f"{key[0]:<9}{key[1]:<4}{c2:8.3f}{c2p:9.3f}{c2/c2p:7.2f}"
          f"{1/k:9.2f}{1/kp:9.2f}{kp/k:7.2f}"
          f"{z_cal:7.3f}{j_cal:7.3f}{JP_NLS[ax]:7.3f}{j_cal/JP_NLS[ax]:7.2f}")

print()
for ax, name in (('x', 'roll'), ('y', 'pitch')):
    a = np.array(rows[ax])
    for j, lab in enumerate(('C2', '1/K [N.m]', 'z_cal [m]', 'J_P [kg.m^2]')):
        v = a[:, j]
        ref = {0: np.sqrt(31.59*z_phys/JP_NLS[ax]), 1: 31.59*z_phys,
               2: z_phys, 3: JP_NLS[ax]}[j]
        print(f"  {name:<6}{lab:<14} mean {v.mean():7.3f}  CV {100*v.std(ddof=1)/v.mean():5.1f}%"
              f"  spread {v.max()/v.min():5.2f}x  vs phys {ref:7.3f}"
              f"  ({100*(v.mean()/ref-1):+6.1f}%)")
    print()

# ---------------------------------------------------------------------
# Sensitivity of the physical reference to the CAD inertia it consumes.
# Inverting J_P = J_CAD + m (z_CoM^2 + l_p^2) for z_CoM shows how much a
# CAD error would matter -- and that the roll/pitch agreement, which is
# really a test of the landing-gear geometry l_p, survives J_CAD = 0.
print("sensitivity of the derived z_CoM to the CAD inertia:")
print(f"{'J_CAD factor':>14}{'z_roll':>9}{'z_pitch':>9}{'mean':>8}{'2-axis gap':>12}")
for f, lab in ((0.0, '0 (omitted)'), (0.5, 'x0.5'), (1.0, 'x1 (Table 5)'),
               (1.5, 'x1.5'), (2.0, 'x2')):
    zz = {}
    for a in 'xy':
        v = (JP_NLS[a] - f * J_CAD[a]) / 3.220 - LP[a] ** 2
        zz[a] = np.sqrt(v) if v > 0 else float('nan')
    mu = 0.5 * (zz['x'] + zz['y'])
    print(f"{lab:>14}{zz['x']:9.3f}{zz['y']:9.3f}{mu:8.3f}"
          f"{100 * abs(zz['x'] - zz['y']) / mu:11.1f}%")
