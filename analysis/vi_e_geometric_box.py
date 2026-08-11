#!/usr/bin/env python3
"""Sec. VI-E evaluated over the geometric operating box of Sec. VI-D.

VI-D fixes an admissible box from geometry rather than from the fits:
the unloaded weight W, the CoM height z_CoM in [0.2, 0.3] m, the planar
offset box, the measured threshold magnitudes, and the parallel-axis
identity (82) for the pivot inertia.  This script carries that box
through the deviation bound and reports what the gyro channel can do
inside it -- so every number in VI-E rests on the same admissible set
as VI-C and VI-D, with nothing borrowed from the calibration.

Three things are worth stating about how the inertia enters.

  1. C2 needs an inertia LOWER bound, and (82) supplies it.  Better,
     the mass cancels outright:

         C2^2 = W z_CoM / J_P <= g z / (z^2 + a^2),   a = l_p + p_off,

     so the ceiling on C2 is pure geometry.  Over z in [0.2, 0.3] it is
     attained at the low end; the unconstrained maximum sqrt(g/2a) at
     z = a sits below that interval.

  2. dM_win needs an inertia UPPER bound, which (82) does NOT give:
     tau_end = (6 phi_max J_P / Mdot)^(1/3) grows with J_P.  Retaining
     the term (82) discards closes it,

         J_P <= J_CoM^CAD + m (z_max^2 + a^2),

     and a second, inertia-free route is available as a cross-check:
     the excitation window is truncated at MOMENT_CAP, so
     dM_win <= cap - |M_crit|_min.  The tighter of the two is used.

  3. Nothing else in the section needs J_P at all.  The kernel factors
     of (90) close on Wz and x, and the reported bound (97) is free of
     both.

Usage: python analysis/vi_e_geometric_box.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pnls_constants import PNLS_CONSTANTS as CONST

G = 9.80665
W = 30.08                          # N, unloaded (Sec. VI-D)
MASS = W / G
PHI_MAX = np.deg2rad(10.0)         # tilt cap of the excitation design
BETA_M = 0.0345                    # rad^-1, bilinear moment-channel slope
J_COM_CAD = 0.0537                 # kg m^2, both axes
Z_RANGE = (0.20, 0.30)             # m
RATES = (0.10, 0.45, 1.20)         # N m/s, Sec. V-B
AXES = {
    'roll  (Mx)': dict(arm=0.140 + 0.020, m_lo=0.7, m_hi=2.1, cap=2.37),
    'pitch (My)': dict(arm=0.110 + 0.020, m_lo=0.4, m_hi=1.7, cap=2.74),
}


def lam(u):
    return np.sinh(u) - u


def r_phi(x, n=20001):
    u = np.linspace(0.0, x, n)
    return float(np.trapz(lam(u) ** 2, u) / x / lam(x) ** 2)


def r_ge(x, n=20001):
    u = np.linspace(0.0, x, n)
    return float(np.trapz(u * lam(u), u) / x / (x * lam(x)))


def psi(x):
    """x sinh x / (cosh x - 1) = x coth(x/2), the rate normaliser."""
    return x / np.tanh(0.5 * x)


print(f"W = {W} N (unloaded), m = {MASS:.3f} kg, phi_max = 10 deg, "
      f"beta_M = {BETA_M} /rad\n")

print("1. eq (82) two ways.  For C2 the mass cancels; for dM_win it does not\n")
print(f"  {'axis':12}{'z':>6}{'Wz':>8}{'J_P lo':>8}{'J_P hi':>8}"
      f"{'C2 <=':>8}{'C2 >=':>8}{'Wz C2 <=':>10}")
box = {}
for name, p in AXES.items():
    a = p['arm']
    rows = []
    for z in Z_RANGE:
        wz = W * z
        jp_lo = MASS * (z ** 2 + a ** 2)
        jp_hi = J_COM_CAD + jp_lo
        rows.append(dict(z=z, wz=wz, jp_lo=jp_lo, jp_hi=jp_hi,
                         c2_hi=np.sqrt(wz / jp_lo),
                         c2_lo=np.sqrt(wz / jp_hi)))
        r = rows[-1]
        print(f"  {name:12}{z:6.2f}{wz:8.3f}{jp_lo:8.4f}{jp_hi:8.4f}"
              f"{r['c2_hi']:8.3f}{r['c2_lo']:8.3f}{wz * r['c2_hi']:10.2f}")
    box[name] = rows
print(f"\n  unconstrained ceiling sqrt(g/2a) at z = a: "
      + ',  '.join(f"{n.split()[0]} {np.sqrt(G / (2 * p['arm'])):.3f}"
                   for n, p in AXES.items()) + " /s")

print("\n2. the window: two independent ceilings on dM_win\n")
print(f"  {'axis':12}{'Mdot':>6}{'x_bar':>7}{'tau_end<=':>10}"
      f"{'dM tilt':>9}{'dM cap':>8}{'dM_win<=':>10}   [s], [N.m]")
win = {}
for name, p in AXES.items():
    wzc2 = max(r['wz'] * r['c2_hi'] for r in box[name])
    jp_hi = max(r['jp_hi'] for r in box[name])
    dm_cap = p['cap'] - p['m_lo']
    for md in RATES:
        xb = (6 * PHI_MAX * wzc2 / md) ** (1 / 3)
        te = (6 * PHI_MAX * jp_hi / md) ** (1 / 3)
        dm_tilt = md * te
        dm = min(dm_tilt, dm_cap)
        win[(name, md)] = (xb, dm)
        print(f"  {name:12}{md:6.2f}{xb:7.2f}{te:10.3f}"
              f"{dm_tilt:9.3f}{dm_cap:8.2f}{dm:10.3f}")

print("\n3. the two channels and their Chebyshev average\n")
print(f"  {'axis':12}{'Mdot':>6}{'rho_phi':>9}{'rho_GE':>8}{'rho_max':>9}"
      f"{'R_phi':>8}{'R_GE':>7}{'rho_bar':>9}   [mN.m]")
chan = {}
for name, p in AXES.items():
    rp = 0.5 * W * p['arm'] * PHI_MAX ** 2
    for md in RATES:
        xb, dm = win[(name, md)]
        rg = BETA_M * dm * PHI_MAX
        rb = r_phi(xb) * rp + r_ge(xb) * rg
        chan[(name, md)] = (rp, rg, rb)
        print(f"  {name:12}{md:6.2f}{1e3 * rp:9.2f}{1e3 * rg:8.2f}"
              f"{1e3 * (rp + rg):9.2f}{r_phi(xb):8.4f}{r_ge(xb):7.4f}"
              f"{1e3 * rb:9.2f}")

print("\n4. what the gyro channel can do inside the box\n")
print(f"  {'axis':12}{'Mdot':>6}{'w_nom':>9}{'|e_w|':>9}{'rel':>8}"
      f"{'|e_w|':>10}{'|dM_crit|':>11}{'|d lam_off|':>12}")
print(f"  {'':12}{'[N.m/s]':>6}{'[rad/s]':>9}{'[rad/s]':>9}{'[%]':>8}"
      f"{'[deg/s]':>10}{'[mN.m]':>11}{'[mm]':>12}")
for name, p in AXES.items():
    r0 = max(box[name], key=lambda r: r['wz'] * r['c2_hi'])
    wz, c2 = r0['wz'], r0['c2_hi']
    for md in RATES:
        rp, rg, _ = chan[(name, md)]
        x = brentq(lambda t: lam(t) - PHI_MAX * wz * c2 / md, 1e-9, 40.0)
        w_nom = (md / wz) * (np.cosh(x) - 1.0)
        rel = psi(x) * (r_phi(x) * rp + r_ge(x) * rg) / (md * x / c2)
        dmc = rp / 7 + rg / 5
        print(f"  {name:12}{md:6.2f}{w_nom:9.3f}{rel * w_nom:9.4f}"
              f"{100 * rel:8.2f}{np.rad2deg(rel * w_nom):10.3f}"
              f"{1e3 * dmc:11.2f}{1e3 * dmc / W:12.3f}")
print(f"\n  the x-free form (97) is rate-independent; taking the largest"
      f" rho_GE per axis,")
for name, p in AXES.items():
    rp, rg, _ = chan[(name, max(RATES))]
    print(f"    {name}:  |dM_crit| <= {1e3 * rp:.2f}/7 + {1e3 * rg:.2f}/5"
          f" = {1e3 * (rp / 7 + rg / 5):.2f} mN.m"
          f"  ->  {1e3 * (rp / 7 + rg / 5) / W:.3f} mm")

print("\n5. cross-check: does the box contain the identified constants?\n")
print(f"  {'case':9}{'axis':5}{'C2 fit':>8}{'C2 ceiling (82)':>18}   verdict")
for (case, axis), (c2, _) in sorted(CONST.items()):
    name = 'roll  (Mx)' if axis == 'Mx' else 'pitch (My)'
    ceil = max(r['c2_hi'] for r in box[name])
    print(f"  {case:9}{axis:5}{c2:8.3f}{ceil:18.3f}   "
          f"{'ok' if c2 <= ceil else 'EXCEEDS'}")
bad = [(c, a, v[0]) for (c, a), v in CONST.items()
       if v[0] > max(r['c2_hi'] for r in
                     box['roll  (Mx)' if a == 'Mx' else 'pitch (My)'])]
if bad:
    print(f"\n  {len(bad)} fit(s) above the (82) ceiling: "
          + ', '.join(f"{c}/{a} at {v:.3f}" for c, a, v in bad))
    print(f"  The margin is set by p_off: dropping it (a = l_p) raises the")
    for name, p in AXES.items():
        print(f"    {name} ceiling from "
              f"{max(r['c2_hi'] for r in box[name]):.3f} to "
              f"{np.sqrt(W * Z_RANGE[0] / (MASS * (Z_RANGE[0] ** 2 + (p['arm'] - 0.020) ** 2))):.3f} /s")
    print(f"  so either p_off is excluded from the arm of (82), or the")
    print(f"  contact is not a rigid pivot for that run.  Worth a footnote.")
