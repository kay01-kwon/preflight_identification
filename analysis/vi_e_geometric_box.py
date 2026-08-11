#!/usr/bin/env python3
"""Sec. VI-E evaluated over the geometric operating box of Sec. VI-D.

VI-D fixes an admissible box from geometry rather than from the fits:
the weight W, the CoM height z_CoM in [0.2, 0.3] m, the planar
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

A note on consistency.  At the tilt cap x is DETERMINED by the geometry
and the ramp rate through Lambda(x) = phi_max Wz C2 / Mdot, and so is
dM_win = Mdot x / C2.  The bounds (92)-(93) exist only to avoid solving
that transcendental equation; they must not be substituted into the
denominator of the relative form, which would flatter it.  Sections 3
and 4 below therefore solve for x once and use the window it implies,
with (93) reported alongside as the shortcut and used only where it
appears on the conservative side -- in rho_GE for the x-free (97).

The weight enters from both ends of its admissible range.  W_max is
unfavourable for Wz, rho_phi and J_P; W_min for omega_nom and for the
conversion of a threshold error into an offset error.  Each quantity is
taken at whichever end is unfavourable for it.

Usage: python analysis/vi_e_geometric_box.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pnls_constants import PNLS_CONSTANTS as CONST

G = 9.80665
W_RANGE = (30.08, 31.59)           # N, unloaded to fully ballasted
W = max(W_RANGE)                   # conservative for Wz, rho_phi, J_P
W_MIN = min(W_RANGE)               # conservative for omega_nom and d lambda
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


print(f"W in {W_RANGE} N (unloaded .. ballasted), m_max = {MASS:.3f} kg,")
print(f"phi_max = 10 deg, beta_M = {BETA_M} /rad.  Each quantity is taken at")
print(f"whichever end of the weight range is unfavourable for it: W_max for")
print(f"Wz, rho_phi and J_P, W_min for omega_nom and the offset conversion.\n")

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

print("\n2. the window at the tilt cap, and the (92)-(93) shortcuts\n")
print(f"  {'axis':12}{'Mdot':>6}{'x exact':>9}{'x_bar':>7}{'tau_end':>9}"
      f"{'dM_win':>8}{'(93) bnd':>9}{'cap bnd':>9}   [s], [N.m]")
win = {}
for name, p in AXES.items():
    r0 = max(box[name], key=lambda r: r['wz'] * r['c2_hi'])
    wz, c2 = r0['wz'], r0['c2_hi']
    wzc2 = wz * c2
    jp_hi = max(r['jp_hi'] for r in box[name])
    dm_cap = p['cap'] - p['m_lo']
    for md in RATES:
        xb = (6 * PHI_MAX * wzc2 / md) ** (1 / 3)
        x = brentq(lambda t: lam(t) - PHI_MAX * wzc2 / md, 1e-9, 40.0)
        te = x / c2
        dm = md * te
        dm93 = md * (6 * PHI_MAX * jp_hi / md) ** (1 / 3)
        win[(name, md)] = dict(x=x, xb=xb, te=te, dm=dm, wz=wz, c2=c2,
                               dm93=min(dm93, dm_cap))
        print(f"  {name:12}{md:6.2f}{x:9.2f}{xb:7.2f}{te:9.3f}"
              f"{dm:8.3f}{dm93:9.3f}{dm_cap:9.2f}")

print("\n3. the two channels and their Chebyshev average\n")
print(f"  {'axis':12}{'Mdot':>6}{'rho_phi':>9}{'rho_GE':>8}{'rho_max':>9}"
      f"{'R_phi':>8}{'R_GE':>7}{'rho_bar':>9}   [mN.m]")
chan = {}
for name, p in AXES.items():
    rp = 0.5 * W * p['arm'] * PHI_MAX ** 2
    for md in RATES:
        w = win[(name, md)]
        rg = BETA_M * w['dm'] * PHI_MAX
        rb = r_phi(w['x']) * rp + r_ge(w['x']) * rg
        chan[(name, md)] = (rp, rg, rb)
        print(f"  {name:12}{md:6.2f}{1e3 * rp:9.2f}{1e3 * rg:8.2f}"
              f"{1e3 * (rp + rg):9.2f}{r_phi(w['x']):8.4f}{r_ge(w['x']):7.4f}"
              f"{1e3 * rb:9.2f}")

print("\n4. what the gyro channel can do inside the box\n")
print(f"  {'axis':12}{'Mdot':>6}{'w_nom':>9}{'|e_w|':>9}{'env':>7}{'rel':>7}"
      f"{'|e_w|':>9}{'|dM_crit|':>11}{'|d lam_off|':>12}")
print(f"  {'':12}{'[N.m/s]':>6}{'[rad/s]':>9}{'[rad/s]':>9}{'[%]':>7}{'[%]':>7}"
      f"{'[deg/s]':>9}{'[mN.m]':>11}{'[mm]':>12}")
for name, p in AXES.items():
    for md in RATES:
        w = win[(name, md)]
        rp, rg, rb = chan[(name, md)]
        x = w['x']
        w_nom = (md / (W_MIN * w['wz'] / W)) * (np.cosh(x) - 1.0)
        rel = psi(x) * rb / w['dm']
        env = psi(x) * (rp + rg) / w['dm']
        rg93 = BETA_M * w['dm93'] * PHI_MAX
        dmc = rp / 7 + rg93 / 5
        print(f"  {name:12}{md:6.2f}{w_nom:9.3f}{rel * w_nom:9.4f}"
              f"{100 * env:7.0f}{100 * rel:7.2f}"
              f"{np.rad2deg(rel * w_nom):9.3f}{1e3 * dmc:11.2f}"
              f"{1e3 * dmc / W_MIN:12.3f}")
print(f"\n  the x-free form (97) is rate-independent up to rho_GE;"
      f" at the fastest ramp,")
for name, p in AXES.items():
    w = win[(name, max(RATES))]
    rp = 0.5 * W * p['arm'] * PHI_MAX ** 2
    rg93 = BETA_M * w['dm93'] * PHI_MAX
    print(f"    {name}:  |dM_crit| <= {1e3 * rp:.2f}/7 + {1e3 * rg93:.2f}/5"
          f" = {1e3 * (rp / 7 + rg93 / 5):.2f} mN.m"
          f"  ->  {1e3 * (rp / 7 + rg93 / 5) / W_MIN:.3f} mm"
          f"   (gravity share {100 * (rp / 7) / (rp / 7 + rg93 / 5):.0f}%)")

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
              f"{np.sqrt(G * Z_RANGE[0] / (Z_RANGE[0] ** 2 + (p['arm'] - 0.020) ** 2)):.3f} /s")
    print(f"  so either p_off is excluded from the arm of (82), or the")
    print(f"  contact is not a rigid pivot for that run.  Worth a footnote.")
