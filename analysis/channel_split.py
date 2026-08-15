#!/usr/bin/env python3
"""Where does e_omega actually go?  Four directions, two of them frozen.

The question this answers is not whether e_omega is absorbed but by
WHICH channel, and how much the answer changes because two of the four
are held fixed during the onset sweep.

The model has four parameters and therefore four tangent directions at
the nominal,

    u   = dg/dC1     = cosh(C2 tau) - 1          amplitude
    v   = dg/dC2     = C1 tau sinh(C2 tau)       exponent
    chi = dg/ddelta  = -C1 C2 sinh(C2 tau)       onset
    1                                            baseline

The tempting thing is to project e_omega onto all four and read off who
took what.  That does not work, and the reason it does not work IS the
answer: the four directions are nearly dependent, with a normalised Gram
condition number of 2e4 to 3e5 over the operating windows.  The split is
not identifiable, so its coefficients carry no meaning and are not
reported.

What IS well posed is the question one step back.  Ask how much of each
frozen direction the surviving pair can reproduce:

    angle(u, span{chi,1})   how much of an amplitude change looks, to
                            the onset sweep, like a shift plus a level
    angle(v, span{chi,1})   the same for an exponent change

Both come out at a few degrees, so the answer is essentially all of it.
That settles the question in the direction the intuition points, and
harder: freezing C1 and C2 costs almost nothing in FIT quality, because
the shift can mimic what they would have done -- and it costs a great
deal in ANSWER quality, because the mimicry is an onset displacement and
the onset is the answer.  The conversion rates are the closed forms of
(D.15), A/(C2 B) per unit of relative amplitude error and D/(C2 B) per
unit of relative exponent error.

Then, within the deployed sweep, how the absorbed part divides between
shift and level.  That projection is well conditioned (cond 4 to 7) and
is reported with the two directions orthogonalised, level first, so the
shares add up to the absorbed fraction instead of overlapping.

All of this is the tangent-space picture, exact to first order in the
deviation and stated as such; the estimator itself is not linear.

Usage: python analysis/channel_split.py
"""
import sys

import numpy as np
from scipy.optimize import brentq

C2, J_P = 5.046, 0.4260
WZ = J_P * C2 ** 2
W, ARM, ZCOM, BETA = 31.59, 0.160, 0.30, -0.03446
PHI_MAX = np.deg2rad(10.0)
N = 4001


def window(mdot):
    rhs = PHI_MAX * WZ * C2 / mdot
    return brentq(lambda x: np.sinh(x) - x - rhs, 1e-6, 20.0)


def build(mdot):
    """Nominal, forcing and deviation on the run's own window."""
    x = window(mdot)
    tau = np.linspace(0.0, x / C2, N)
    c1 = mdot / WZ
    phi = c1 * (np.sinh(C2 * tau) / C2 - tau)
    g2 = W * ARM * np.cos(phi) - W * ZCOM * np.sin(phi)
    rho = 0.5 * g2 * phi ** 2 - BETA * mdot * tau * np.abs(phi) * np.sign(mdot)
    e = np.zeros_like(tau)
    for i in range(1, len(tau)):
        k = np.cosh(np.clip(C2 * (tau[i] - tau[:i + 1]), 0, 40))
        e[i] = np.trapz(k * rho[:i + 1], tau[:i + 1]) / J_P
    dirs = dict(
        u=np.cosh(C2 * tau) - 1.0,
        v=c1 * tau * np.sinh(C2 * tau),
        chi=-c1 * C2 * np.sinh(C2 * tau),
        one=np.ones_like(tau))
    return dict(mdot=mdot, x=x, tau=tau, c1=c1, e=e, d=dirs)


def solve(r, names):
    """Least squares of e on the named directions; returns coefs and energy."""
    A = np.column_stack([r['d'][n] for n in names])
    w = np.gradient(r['tau'])
    w[0] *= 0.5
    w[-1] *= 0.5
    s = np.sqrt(w)
    co, *_ = np.linalg.lstsq(A * s[:, None], r['e'] * s, rcond=None)
    fit = A @ co
    en = float(np.sum(fit ** 2 * w)) / float(np.sum(r['e'] ** 2 * w))
    G = (A * w[:, None]).T @ A
    dg = np.sqrt(np.diag(G))
    cond = float(np.linalg.cond(G / np.outer(dg, dg)))
    return dict(zip(names, co)), en, cond


def angle_to_span(r, target, names):
    """Angle between one direction and the span of the others, in degrees."""
    w = np.gradient(r['tau'])
    w[0] *= 0.5
    w[-1] *= 0.5
    s = np.sqrt(w)
    A = np.column_stack([r['d'][n] for n in names])
    t = r['d'][target]
    co, *_ = np.linalg.lstsq(A * s[:, None], t * s, rcond=None)
    res = t - A @ co
    return np.rad2deg(np.arcsin(np.sqrt(
        max(np.sum(res ** 2 * w) / np.sum(t ** 2 * w), 0.0))))


def main():
    rates = (0.10, 0.45, 1.20)
    runs = [build(m) for m in rates]
    A = lambda x: 0.25 * np.cosh(2 * x) - np.cosh(x) + 0.75
    B = lambda x: 0.25 * np.sinh(2 * x) - 0.5 * x
    D = lambda x: 0.25 * x * np.sinh(2 * x) - 0.125 * np.cosh(2 * x) \
        + 0.125 - 0.25 * x ** 2

    print("\n  (1) how much of each FROZEN direction the surviving pair"
          " can reproduce\n")
    print(f"  {'Mdot':>6}{'x':>7}{'angle(u)':>10}{'angle(v)':>10}"
          f"{'u repro':>10}{'v repro':>10}{'A/(C2 B)':>11}{'D/(C2 B)':>11}")
    print(f"  {'N m/s':>6}{'':7}{'deg':>10}{'deg':>10}{'%':>10}{'%':>10}"
          f"{'ms per 1%':>11}{'ms per 1%':>11}")
    for r in runs:
        x = r['x']
        au = angle_to_span(r, 'u', ['chi', 'one'])
        av = angle_to_span(r, 'v', ['chi', 'one'])
        print(f"  {r['mdot']:6.2f}{x:7.3f}{au:10.3f}{av:10.3f}"
              f"{100*np.cos(np.deg2rad(au)):10.3f}"
              f"{100*np.cos(np.deg2rad(av)):10.3f}"
              f"{1e3*0.01*A(x)/(C2*B(x)):11.3f}"
              f"{1e3*0.01*D(x)/(C2*B(x)):11.3f}")
    print("\n  Both frozen directions are reproducible from {chi, 1} to")
    print("  better than 99.4%.  Freezing C1 and C2 therefore costs almost")
    print("  nothing in FIT quality -- the shift mimics what they would have")
    print("  done -- and costs a great deal in ANSWER quality, because the")
    print("  mimicry IS an onset displacement.  The last two columns are the")
    print("  exchange rate: how many ms of onset a 1% error in each buys.")

    print(f"\n  (2) the four-way split is NOT identifiable\n")
    print(f"  {'Mdot':>6}{'cond{u,v,chi,1}':>18}{'cond{chi,1}':>14}")
    for r in runs:
        _, _, c4 = solve(r, ['u', 'v', 'chi', 'one'])
        _, _, c2 = solve(r, ['chi', 'one'])
        print(f"  {r['mdot']:6.2f}{c4:18.0f}{c2:14.1f}")
    print("\n  So asking 'how much of e_omega went to the amplitude' has no")
    print("  stable answer, and that near-degeneracy is the mechanism rather")
    print("  than a numerical nuisance.  The pinned pair, by contrast, is")
    print("  well conditioned and can be read directly.")

    print(f"\n  (3) what the deployed sweep absorbs, and how it divides it\n")
    print(f"  {'Mdot':>6}{'absorbed':>11}{'residual':>11}{'baseline':>11}"
          f"{'onset':>10}{'onset /':>10}")
    print(f"  {'N m/s':>6}{'% of e^2':>11}{'% of e^2':>11}{'% of e^2':>11}"
          f"{'% of e^2':>10}{'baseline':>10}")
    for r in runs:
        w = np.gradient(r['tau'])
        w[0] *= 0.5
        w[-1] *= 0.5
        ip = lambda a, b: float(np.sum(a * b * w))
        e = r['e']
        tot = ip(e, e)
        # level first, then the onset direction orthogonalised against it,
        # so the two shares add to the absorbed fraction
        q1 = r['d']['one'] / np.sqrt(ip(r['d']['one'], r['d']['one']))
        q2 = r['d']['chi'] - ip(r['d']['chi'], q1) * q1
        q2 /= np.sqrt(ip(q2, q2))
        sb, sc = ip(e, q1) ** 2 / tot, ip(e, q2) ** 2 / tot
        print(f"  {r['mdot']:6.2f}{100*(sb+sc):11.2f}{100*(1-sb-sc):11.2f}"
              f"{100*sb:11.2f}{100*sc:10.2f}{sc/sb:10.1f}")
    print(f"\n  The sweep takes 86 to 91% of e_omega, and of that the onset")
    print(f"  takes 3 to 4 times what the level does.  The intuition that")
    print(f"  the shift carries most of the adjustment is correct, and the")
    print(f"  reason is (1): it is the only direction left that can imitate")
    print(f"  the two that were frozen.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
