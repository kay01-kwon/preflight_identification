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


def rho_phi_exact(a, z):
    """The gravity remainder at the tilt cap, without a Taylor bound.

    The span holds {1, tau, dphi}, so what survives of
    G(phi) = -W a cos phi + W z sin phi is its second-order remainder
    about phi = 0, which is elementary and exact:

        rho_phi = G(phi) - G(0) - G'(0) phi
                = W [ a (1 - cos phi) + z (sin phi - phi) ] .

    The second term is negative, so the customary 1/2 W a phi^2 is
    loose by 8-14% over the box.
    """
    return W * (a * (1 - np.cos(PHI_MAX)) + z * (np.sin(PHI_MAX) - PHI_MAX))


_KC = {}


def span_constants(x, n=4001):
    """(||Pw||_1, K_phi, K_GE) with P removing span{1, u, Lambda}.

    The reduction (98) is dM_crit = -<rho, w>, and rho is already the
    out-of-span part, so dM_crit = -<P rho, w> = -<P rho, P w> because
    an orthogonal projection is self-adjoint and idempotent.  Hence

        |dM_crit| <= ||P rho||_inf * ||P w||_1 ,

    which uses no monotonicity at all -- only that the estimator
    absorbs its own span, which the reduction already assumes.  The
    weight w turns out to lie mostly IN the span, so ||P w||_1 is small
    and the product beats the Chebyshev constants over the whole box.
    """
    key = round(x, 4)
    if key in _KC:
        return _KC[key]
    u = np.linspace(0.0, x, n)
    A = np.stack([np.ones_like(u), u, lam(u)], 1)

    def proj_off(y):
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ c

    # int_s^x sinh(v) cosh(v-s) dv in closed form, via
    # sinh(v) cosh(v-s) = [sinh(2v-s) + sinh(s)]/2:
    w = 0.25 * (np.cosh(2 * x - u) - np.cosh(u)) + 0.5 * np.sinh(u) * (x - u)
    w /= np.trapz(w, u)
    n1 = float(np.trapz(np.abs(proj_off(w)), u))
    out = [float(np.max(np.abs(proj_off(f(u)))) / f(u)[-1]) * n1
           for f in (lambda v: lam(v) ** 2, lambda v: v * lam(v))]
    _KC[key] = (n1, out[0], out[1])
    return _KC[key]


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

print("\n5. the sharp bound: use the absorption instead of monotonicity\n")
print(f"  {'axis':12}{'Mdot':>6}{'z*':>6}{'x':>6}{'rho_phi':>9}{'rho_GE':>8}"
      f"{'K_phi':>8}{'K_GE':>7}{'R_phi':>8}{'R_GE':>7}"
      f"{'(99)':>8}{'(97)':>8}{'  [mN.m]'}")
sharp = {}
for name, p in AXES.items():
    a = p['arm']
    for md in RATES:
        rec = None
        for z in np.linspace(*Z_RANGE, 11):
            wz, c2 = W * z, np.sqrt(G * z / (z ** 2 + a ** 2))
            x = brentq(lambda t: lam(t) - PHI_MAX * wz * c2 / md, 1e-9, 40.0)
            rp, rg = rho_phi_exact(a, z), BETA_M * (md * x / c2) * PHI_MAX
            _, kp, kg = span_constants(x)
            v = (min(kp, r_phi(x)) * rp + min(kg, r_ge(x)) * rg)
            if rec is None or v > rec[0]:
                rec = (v, z, x, rp, rg, kp, kg)
        v, z, x, rp, rg, kp, kg = rec
        xfree = max(rho_phi_exact(a, zz) for zz in Z_RANGE) / 7 + rg / 5
        sharp[(name, md)] = (v, xfree)
        print(f"  {name:12}{md:6.2f}{z:6.2f}{x:6.2f}{1e3 * rp:9.2f}"
              f"{1e3 * rg:8.2f}{kp:8.4f}{kg:7.4f}{r_phi(x):8.4f}{r_ge(x):7.4f}"
              f"{1e3 * v:8.2f}{1e3 * xfree:8.2f}")
print()
for name in AXES:
    v = max(sharp[(name, md)][0] for md in RATES)
    f = max(sharp[(name, md)][1] for md in RATES)
    print(f"  {name}:  (99) {1e3 * v:5.2f} mN.m -> {1e3 * v / W_MIN:.3f} mm"
          f"     (97) {1e3 * f:5.2f} -> {1e3 * f / W_MIN:.3f} mm"
          f"     ratio {f / v:.1f}x")
print(f"\n  K rises with x and crosses R near x = 5.5, so the sharp form wins")
print(f"  everywhere in the box (x <= 5.21) but would not for a slower ramp.")

print("\n6. cross-check: does the box contain the identified constants?\n")
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
