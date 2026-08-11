#!/usr/bin/env python3
"""Every constant quoted in Sec. VI-E, recomputed from the calibration.

The section bounds the onset deviation without solving for the window
length and without the pivot inertia.  Four claims carry the argument
and each is checked here:

  1. the pivot inertia is never assumed.  estimate_rig_constants
     returns the pair (C2, K); W z_CoM = 1/K is one of them and
     J_P = 1/(K C2^2) is the derived one.  The identified J_P is
     printed beside the rigid parallel-axis value to show why it is
     not interpreted as an inertia.

  2. Eq. (92), x <= x_bar = (6 phi_max W z_CoM C2 / Mdot)^(1/3).  The
     bound follows from sinh x - x >= x^3/6, whose retained term is
     the inertia-only rise phi = Mdot tau^3 / (6 J_P): the slowest
     admissible ascent, hence the latest arrival at the tilt cap.  The
     worst case is therefore the SLOWEST ramp, which is easy to get
     backwards since every other quantity worsens with the fastest.

  3. Eq. (94), the shared bracket.  sinh x = Lambda(x) + x with
     Lambda known as an equality from the tilt cap, so only the linear
     term needs x_bar.  Substituting x_bar into sinh instead costs an
     order of magnitude; the ratio is printed.

  4. Eqs. (95)-(97).  The estimator fits the RATE, so the relative
     form must normalise on omega_nom = C1(cosh x - 1) and not on the
     excursion; error_budget.py does exactly that (its nom_end is
     K Mdot (cosh x - 1), and what it propagates is dw, not dphi).
     The resulting factor is

         Psi(x) = x sinh x / (cosh x - 1) = x coth(x/2) >= 2,

     the same expression the exponent-perturbation note in
     error_budget.py already quotes.  Against the Chebyshev time
     averages R_phi <= 1/7 and R_GE <= 1/5 its products rise
     monotonically from 2/7 to 1/2 and from 2/5 to 1.  Normalising on
     phi instead would give x(cosh x - 1)/(sinh x - x) and the lower
     limits 3/7 and 3/5; both are printed so the two cannot be
     confused again.  The upper limits -- the only ones that reach
     the boxed result -- are 1/2 and 1 either way.

     The envelope form alone (rho_bar = rho_max) is evaluated too,
     since it is the version that fails: it returns a relative
     deviation larger than the signal it bounds at the slowest ramp,
     which is why (97) and not (90) is the reported result.

Nothing here is fitted.  beta_M comes from the LPV channel fit, the
arms from the geometry, and (C2, K) from pnls_constants.

Usage: python analysis/vi_e_bound_constants.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pnls_constants import PNLS_CONSTANTS as CONST

PHI_MAX = np.deg2rad(10.0)         # tilt cap of the excitation design
BETA_M = 0.0345                    # rad^-1, LPV moment channel slope
W = 31.59                          # N
Z_COM = 0.30                       # m, deliberately above the CAD value
ARM = dict(Mx=0.160, My=0.130)     # l_p + lambda_off
J_CAD, MASS = 0.0537, 3.22
RATES = (0.10, 0.45, 1.20)


def lam(u):
    """Lambda(u) = sinh u - u, the excursion shape."""
    return np.sinh(u) - u


def r_phi(x, n=8001):
    """Chebyshev time average of the gravity remainder, rho ~ Lambda^2."""
    u = np.linspace(0.0, x, n)
    return float(np.trapz(lam(u) ** 2, u) / x / lam(x) ** 2)


def r_ge(x, n=8001):
    """Same for the bilinear channel, rho ~ u Lambda(u)."""
    u = np.linspace(0.0, x, n)
    return float(np.trapz(u * lam(u), u) / x / (x * lam(x)))


def psi(x):
    """Rate normaliser, |e_w|/w_nom <= Psi rho_bar / dM_win.

    x sinh x / (cosh x - 1) = x coth(x/2), via cosh x - 1 = 2 sinh^2(x/2)
    and sinh x = 2 sinh(x/2) cosh(x/2).
    """
    return x / np.tanh(0.5 * x)


def psi_phi(x):
    """The excursion normaliser, kept only to contrast with psi()."""
    return x * (np.cosh(x) - 1.0) / lam(x)


def solve_x(l_target):
    return brentq(lambda t: lam(t) - l_target, 1e-9, 40.0)


print("1. the calibration identifies (C2, K); J_P is derived\n")
print(f"  {'case':9}{'axis':5}{'C2 [1/s]':>10}{'K':>8}{'Wz=1/K':>9}"
      f"{'J_P':>8}   {'x_bar at Mdot = 0.10 / 0.45 / 1.20':>34}")
xb_all, rge_all, jp_all = [], [], []
for (case, axis), (c2, k) in sorted(CONST.items()):
    wz = 1.0 / k
    jp = wz / c2 ** 2
    jp_all.append(jp)
    xb = [(6 * PHI_MAX * wz * c2 / m) ** (1 / 3) for m in RATES]
    xb_all += xb
    rge_all.append(BETA_M * PHI_MAX
                   * (6 * PHI_MAX * wz * RATES[-1] ** 2 / c2 ** 2) ** (1 / 3))
    print(f"  {case:9}{axis:5}{c2:10.3f}{k:8.4f}{wz:9.2f}{jp:8.3f}"
          + ''.join(f"{v:11.2f}" for v in xb))
d = np.hypot(max(ARM.values()), Z_COM)
print(f"\n  J_P identified  [{min(jp_all):.3f}, {max(jp_all):.3f}] kg.m^2")
print(f"  J_P parallel axis, J_CoM + m d^2 with d = {d:.3f} m: "
      f"{J_CAD + MASS * d ** 2:.3f} kg.m^2 -- the identified values fall")
print(f"  below it, so C2 is used as a fitted time constant, not an inertia.")

print("\n2. the a priori bounds of Eqs. (92) and (93)\n")
print(f"  x_bar        <= {max(xb_all):.2f}   at the SLOWEST ramp"
      f"  (Lambda ~ 1/Mdot)")
print(f"  rho_GE,max   <= {1e3 * max(rge_all):.2f} mN.m  at the fastest"
      f"  (opposite corner of the box)")
for ax, a in sorted(ARM.items()):
    print(f"  rho_phi,max  <= {0.5e3 * W * a * PHI_MAX ** 2:.1f} mN.m  "
          f"({ax}, cos <= 1 and the -Wz sin term dropped)")
full = 0.5 * (W * ARM['Mx'] * np.cos(PHI_MAX)
              - W * Z_COM * np.sin(PHI_MAX)) * PHI_MAX ** 2
print(f"  rho_phi,max   = {1e3 * full:.1f} mN.m  with that term retained "
      f"at the cap")

print("\n3. Eq. (94): bound Lambda by the cap, not by sinh(x_bar)\n")
print(f"  {'x_bar':>7}{'sinh(x_bar)':>13}{'Lambda + x_bar':>16}{'ratio':>8}")
for xb in (max(xb_all), 5.04, 3.16):
    print(f"  {xb:7.2f}{np.sinh(xb):13.1f}{xb ** 3 / 6 + xb:16.1f}"
          f"{np.sinh(xb) / (xb ** 3 / 6 + xb):8.1f}x")

print("\n4. Eqs. (95)-(97): the products are bounded uniformly in x\n")
chk = [0.5, 2.0, 5.0, 9.0]
print(f"  Psi(x) == x coth(x/2): "
      f"{np.allclose([psi(x) for x in chk], [x * np.sinh(x) / (np.cosh(x) - 1) for x in chk])}"
      f";  Psi >= 2 everywhere: "
      f"{bool(np.all([psi(x) >= 2 for x in np.linspace(0.02, 30, 500)]))}\n")
print(f"  {'x':>7}{'R_phi':>9}{'R_GE':>8}{'Psi_w':>9}{'PwR_phi':>10}"
      f"{'PwR_GE':>9}   |{'Psi_phi':>9}{'PpR_phi':>10}{'PpR_GE':>9}")
for x in (0.01, 1.0, 1.79, 2.67, 5.20, 7.23, 20.0):
    print(f"  {x:7.2f}{r_phi(x):9.4f}{r_ge(x):8.4f}{psi(x):9.3f}"
          f"{psi(x) * r_phi(x):10.4f}{psi(x) * r_ge(x):9.4f}   |"
          f"{psi_phi(x):9.3f}{psi_phi(x) * r_phi(x):10.4f}"
          f"{psi_phi(x) * r_ge(x):9.4f}")
grid = np.linspace(0.05, 12.0, 400)
pa = np.array([psi(x) * r_phi(x) for x in grid])
pb = np.array([psi(x) * r_ge(x) for x in grid])
print(f"\n  rate-normalised (the one used): {2 / 7:.4f} = 2/7 -> 1/2 and"
      f" {2 / 5:.4f} = 2/5 -> 1")
print(f"  excursion-normalised (not used): {3 / 7:.4f} = 3/7 -> 1/2 and"
      f" {3 / 5:.4f} = 3/5 -> 1")
print(f"  monotone: {bool(np.all(np.diff(pa) > 0))}"
      f" / {bool(np.all(np.diff(pb) > 0))};  the upper limits 1/2 and 1"
      f" are shared, so (97) is the same either way.")

print("\n5. what the envelope form alone would return\n")
worst = dict(env=0.0, cheb=0.0, xfree=0.0)
for (case, axis), (c2, k) in CONST.items():
    wz, rp = 1.0 / k, 0.5 * W * ARM[axis] * PHI_MAX ** 2
    for m in RATES:
        x = solve_x(PHI_MAX * wz * c2 / m)
        dm = m * x / c2
        rge = BETA_M * PHI_MAX * dm
        worst['env'] = max(worst['env'], psi(x) * (rp + rge) / dm)
        worst['cheb'] = max(worst['cheb'],
                            psi(x) * (r_phi(x) * rp + r_ge(x) * rge) / dm)
        worst['xfree'] = max(worst['xfree'], (0.5 * rp + rge) / dm)
print(f"  worst |e_omega| / omega_nom over 10 configurations x 3 rates:")
print(f"    Eq. (90) with rho_bar = rho_max      {100 * worst['env']:8.0f} %"
      f"   <- larger than the signal it bounds")
print(f"    Eq. (96) with the averages of (95)   {100 * worst['cheb']:8.1f} %")
print(f"    Eq. (97), x-free                     {100 * worst['xfree']:8.1f} %")
rp = 0.5 * W * ARM['Mx'] * PHI_MAX ** 2
dmc = rp / 7 + max(rge_all) / 5
print(f"\n  a priori threshold bound at the cap: |dM_crit| <= "
      f"{1e3 * dmc:.1f} mN.m -> |d lambda_off| <= {1e3 * dmc / W:.2f} mm")
