#!/usr/bin/env python3
"""Numerical check of docs/minimizer_absorption.tex, claim by claim.

The derivation is short but every step is an identity, so each one can
be checked to machine precision on a synthetic rho.  Constants must obey
Wz = J_P C2^2 throughout: getting that wrong is what made an earlier
check of the last claim appear to fail by a factor of two.

Usage: python analysis/absorption_check.py
"""
import sys

import numpy as np

C2, J_P, C1 = 5.299, 0.1813, 0.2635
WZ = J_P * C2 ** 2                      # the identity, not an independent value
MDOT = C1 * WZ


def main():
    tau = np.linspace(0, 0.40, 8001)
    rho = 0.02 * np.exp(-((tau - 0.05) / 0.03) ** 2)   # early and localised
    cum = lambda f: np.array([np.trapz(f[:i + 1], tau[:i + 1])
                              for i in range(len(tau))])
    P, Q = cum(np.cosh(C2 * tau) * rho), cum(np.sinh(C2 * tau) * rho)

    direct = np.array([np.trapz(np.cosh(C2 * (tau[i] - tau[:i + 1]))
                                * rho[:i + 1], tau[:i + 1]) / J_P
                       for i in range(len(tau))])
    split = (np.cosh(C2 * tau) * P - np.sinh(C2 * tau) * Q) / J_P
    ok = []

    def check(name, err, tol):
        ok.append(err < tol)
        print(f"  {'PASS' if err < tol else 'FAIL'}  {name:52}{err:.2e}")

    check("(9)  kernel split equals the Duhamel integral",
          float(np.max(np.abs(split - direct))), 1e-12)

    a, b = C1 + P[-1] / J_P, -Q[-1] / J_P
    R, psi = np.sqrt(a ** 2 - b ** 2), np.arctanh(b / a)
    lhs = a * np.cosh(C2 * tau) + b * np.sinh(C2 * tau)
    check("(13) a cosh + b sinh = R cosh(. + psi)",
          float(np.max(np.abs(lhs - R * np.cosh(C2 * tau + psi)))), 1e-12)

    y = C1 * (np.cosh(C2 * tau) - 1) + direct
    m = tau > 0.20                                   # after rho has died
    check("(15) nominal + e_omega is a shifted, rescaled cosh",
          float(np.max(np.abs(y[m] - (R * np.cosh(C2 * tau[m] + psi) - C1)))),
          1e-12)

    d_id = -psi / C2
    d_row = np.arcsinh(Q[-1] / (J_P * C1)) / C2
    check("(20) vs (21) the onset from either route [s]",
          abs(d_id - d_row), 3e-5)

    # (22) is an identity at first order in delta and only there: the
    # sinh row gives delta through arcsinh, so Mdot*delta carries a
    # -z^3/6 correction that C2 Q does not.  Both halves are checked.
    check("(22) Mdot Q/(J_P C1 C2) = C2 Q, exactly",
          abs(MDOT * Q[-1] / (J_P * C1 * C2) - C2 * Q[-1]), 1e-15)
    z = Q[-1] / (J_P * C1)
    check("(22) the arcsinh correction is the cube term",
          abs((MDOT * d_row - C2 * Q[-1]) + MDOT * z ** 3 / (6 * C2)), 1e-11)

    check("(19) residual cosh coefficient is P/J_P to O(Q^2)",
          abs((a - C1 * np.cosh(C2 * d_row)) - P[-1] / J_P), 1e-5)

    check("Case A: amplitude R absorbs exactly the cosh half",
          abs((R - C1) - (a - C1 * np.cosh(C2 * d_row))), 1e-5)

    print(f"\n  third-order term dropped by (22): "
          f"{1e3 * MDOT * z ** 3 / (6 * C2):.2e} mN.m against"
          f" {1e3 * C2 * Q[-1]:.4f}")
    print(f"  for scale on this synthetic: delta = {1e3 * d_row:.3f} ms,"
          f"  dM_crit = {1e3 * C2 * Q[-1]:.4f} mN.m,"
          f"  P/(J_P C1) = {100 * P[-1] / (J_P * C1):.2f}%")
    print("  rho was placed EARLY, so most of it went to the amplitude and"
          "\n  little to the onset -- consequence (ii) of the derivation.")
    ok.append(short_route())
    return 0 if all(ok) else 1




def short_route():
    """(D.13a)-(D.13c): the pointwise bound on e_omega, pushed through the
    projection, gives exactly rho_bar.  E and chi are both proportional to
    sinh(C2 tau), so the integral cancels and nothing has to be evaluated;
    this checks that the cancellation is real over a range of constants."""
    from scipy.integrate import quad
    print("\n  short route: Mdot <E,|chi|>/||chi||^2 against rho_bar\n")
    print(f"  {'C2':>7}{'J_P':>8}{'C1':>7}{'tau_end':>9}{'bound':>11}"
          f"{'rho_bar':>10}{'ratio':>10}")
    worst = 0.0
    for c2, j_p, c1 in ((5.299, 0.1813, 0.2635), (5.046, 0.3723, 0.1266),
                        (4.100, 0.1900, 0.3100), (7.500, 0.0900, 0.4000)):
        wz, rb = j_p * c2 ** 2, 0.010
        mdot = c1 * wz
        for te in (0.30, 0.55, 0.80, 1.05):
            num = quad(lambda t: (rb * np.sinh(c2 * t) / (j_p * c2))
                       * c1 * c2 * np.sinh(c2 * t), 0, te)[0]
            den = quad(lambda t: (c1 * c2 * np.sinh(c2 * t)) ** 2, 0, te)[0]
            r = mdot * num / den / rb
            worst = max(worst, abs(r - 1.0))
            print(f"  {c2:7.3f}{j_p:8.4f}{c1:7.3f}{te:9.2f}"
                  f"{1e3 * mdot * num / den:11.4f}{1e3 * rb:10.4f}{r:10.6f}")
    print(f"\n  worst departure from 1: {worst:.2e}")
    return worst < 1e-9


if __name__ == '__main__':
    sys.exit(main())
