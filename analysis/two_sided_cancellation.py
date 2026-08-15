#!/usr/bin/env python3
"""Does the two-sided difference kill the absorption bias too?

The offset is read from the half-sum, M_off = (M_pos + M_neg)/2, which
is built to cancel whatever is common to the two directions.  The
question is which parts of the deviation forcing are common and which
are not, because only the second kind reaches the answer.

Set up honestly, the two runs are NOT mirror images.  They tip over
different edges: the positive run about +l_p with restoring arm
A_+ = l_p - a, the negative about -l_p with A_- = l_p + a.  That
asymmetry IS the measurement.  About its pivot each run obeys

    J_P phi'' = M(t) - W A cos(phi) + W z sin(phi),

exactly, and the estimator's model is the linearisation

    J_P phi'' = Mdot tau + W z phi,      tau = t - t_c,  M(t_c) = W A,

so the forcing the model does not carry is

    rho = W A (1 - cos phi) + W z (sin phi - phi)
        ~= (1/2) W A phi^2 - (1/6) W z phi^3.

In mirrored coordinates both runs obey the same equation, so in signed
phi the A-term carries sign(Mdot) and the Wz term does not.  The A-term
is therefore ANTISYMMETRIC and cancels in the half-sum -- except that
its coefficient differs between the runs, A_+ - A_- = -2a, and what
survives is proportional to the very offset being measured:

    relative error on the offset  ~  (1/2) <phi^2>_w      (order only)

a SCALE error rather than an additive bias.  Measured, that term is
about a third of what survives; the rest comes from the two windows not
being the same length, since A_+ != A_- puts the tilt cap at different
times and the Wz channel then cancels imperfectly too.  Both are tiny.

The channel that does NOT cancel is the ground effect, and only if its
parity is even.  rho_GE = beta M phi flips with neither M nor phi alone
but with both, so whether it survives turns on a modelling question the
form does not settle.  Both parities are reported; the difference is
between 1 um and 78 um of offset, so it is worth settling.

Nothing is linearised in the measurement: the trajectory comes from the
exact ODE, the windows are where each run actually reaches the tilt cap
(and they differ, because A_+ != A_-), and the fit is the piecewise
onset sweep with C1 and C2 pinned.

Usage: python analysis/two_sided_cancellation.py
"""
import sys

import numpy as np
from scipy.integrate import solve_ivp

W, LP, ZCOM = 31.59, 0.160, 0.30
J_P = 0.4260
WZ = W * ZCOM                          # the destabilising gain
C2 = np.sqrt(WZ / J_P)                 # the identity, not an independent value
PHI_MAX = np.deg2rad(10.0)
BETA = 0.03446
A_TRUE = 0.0020                        # 2 mm of offset, the thing measured
PRE = 0.15


def tipover(mdot, arm, beta=0.0, ge_odd=True, n=4001):
    """The exact run, from the onset to the tilt cap.

    beta drives the ground-effect channel; ge_odd chooses its parity
    under the direction flip, which is not settled by the model form
    and decides on its own whether that channel cancels.
    """
    tc = W * arm / abs(mdot)            # true onset: |applied| = W A

    sg = np.sign(mdot)

    def f(t, s):
        # In mirrored coordinates the negative run obeys the SAME equation
        # with A_- in place of A_+, so in signed phi the restoring-arm
        # nonlinearity carries sign(Mdot) while the Wz sin(phi) term does
        # not.  Getting that wrong makes the two biases share a sign and
        # the cancellation disappear.
        phi, om = s
        ge = (sg if ge_odd else 1.0) * beta * abs(mdot * (tc + t)) * abs(phi)
        return [om, (mdot * t + sg * W * arm * (1.0 - np.cos(phi))
                     + WZ * np.sin(phi) + ge) / J_P]

    def cap(t, s):
        return abs(s[0]) - PHI_MAX
    cap.terminal, cap.direction = True, 1
    sol = solve_ivp(f, (0.0, 30.0), [0.0, 0.0], events=cap,
                    rtol=1e-10, atol=1e-12, dense_output=True)
    te = float(sol.t_events[0][0])
    t = np.linspace(-PRE, te, n)
    tau = np.clip(t, 0.0, None)
    om = np.where(t < 0, 0.0, sol.sol(tau)[1])
    phi = np.where(t < 0, 0.0, sol.sol(tau)[0])
    return dict(t=t, om=om, phi=phi, te=te, tc=tc, arm=arm, mdot=mdot)


def fit(run, mdot, c2=C2, dmax=0.25, nd=5001):
    """The deployed estimator: C1 and C2 pinned, onset swept, c closed form."""
    t, y = run['t'], run['om']
    c1 = mdot / WZ
    w = np.gradient(t)
    w[0] *= 0.5
    w[-1] *= 0.5
    best = (np.inf, 0.0)
    for d in np.linspace(-dmax, dmax, nd):
        sh = np.where(t < d, 0.0,
                      c1 * (np.cosh(np.clip(c2 * (t - d), 0, 40)) - 1.0))
        c = float(np.sum((y - sh) * w) / w.sum())
        v = float(np.sum((y - sh - c) ** 2 * w))
        if v < best[0]:
            best = (v, d)
    return best[1]                      # onset offset from the TRUE onset


def one(mdot_mag, beta=0.0, ge_odd=True, a=A_TRUE):
    """Both directions, and the offset the method would report."""
    out = {}
    for sgn, arm in ((+1.0, LP - a), (-1.0, LP + a)):
        r = tipover(sgn * mdot_mag, arm, beta, ge_odd)
        d = fit(r, sgn * mdot_mag)
        out['+' if sgn > 0 else '-'] = dict(
            run=r, d=d,
            m_true=sgn * W * arm,
            m_hat=sgn * W * arm + sgn * mdot_mag * d)
    p, m = out['+'], out['-']
    out['off_true'] = 0.5 * (p['m_true'] + m['m_true'])
    out['off_hat'] = 0.5 * (p['m_hat'] + m['m_hat'])
    return out


def phi2_weighted(run):
    """<phi^2> against the normalised onset weight w of (107)."""
    m = run['t'] >= 0
    tau, phi = run['t'][m], run['phi'][m]
    te = tau[-1]
    num = np.array([np.trapz(np.sinh(C2 * tau[tau >= s])
                             * np.cosh(C2 * (tau[tau >= s] - s)),
                             tau[tau >= s]) for s in tau])
    wgt = num / np.trapz(num, tau)
    return float(np.trapz(phi ** 2 * wgt, tau))


def main():
    print(f"\n  W = {W} N, l_p = {LP} m, z = {ZCOM} m, J_P = {J_P},"
          f" C2 = {C2:.4f}")
    print(f"  true offset a = {1e3*A_TRUE:.3f} mm,"
          f" so M_off = {1e3*W*A_TRUE:.3f} mN.m\n")

    print("  gravity channels only (beta = 0)\n")
    print(f"  {'Mdot':>6}{'dir':>5}{'M_true':>10}{'onset err':>11}"
          f"{'M_hat err':>11}{'  |  '}{'a_hat':>8}{'err':>9}{'rel':>8}")
    print(f"  {'N m/s':>6}{'':5}{'N m':>10}{'ms':>11}{'mN.m':>11}"
          f"{'  |  '}{'mm':>8}{'um':>9}{'%':>8}")
    rows = []
    for md in (0.10, 0.45, 1.20):
        o = one(md)
        for k in ('+', '-'):
            v = o[k]
            print(f"  {md if k=='+' else '':6}{k:>5}{v['m_true']:10.4f}"
                  f"{1e3*v['d']:11.3f}"
                  f"{1e3*(v['m_hat']-v['m_true']):11.3f}", end='')
            if k == '+':
                print()
            else:
                a_hat = -o['off_hat'] / W
                print(f"{'  |  '}{1e3*a_hat:8.4f}"
                      f"{1e6*(a_hat-A_TRUE):9.2f}"
                      f"{100*(a_hat/A_TRUE-1):8.3f}")
        rows.append((md, o))

    print(f"\n  the prediction: relative error = (1/2) <phi^2>_w\n")
    print(f"  {'Mdot':>6}{'<phi^2>_w':>12}{'(1/2)<phi^2>':>14}"
          f"{'measured rel':>14}{'ratio':>8}")
    for md, o in rows:
        p2 = phi2_weighted(o['+']['run'])
        a_hat = -o['off_hat'] / W
        rel = a_hat / A_TRUE - 1
        print(f"  {md:6.2f}{p2:12.6f}{0.5*p2:14.6f}{rel:14.6f}"
              f"{rel/(0.5*p2):8.3f}")

    print(f"\n  per-direction bias against what survives the half-sum\n")
    print(f"  {'Mdot':>6}{'bias +':>10}{'bias -':>10}{'half-sum':>11}"
          f"{'cancelled':>12}")
    print(f"  {'':6}{'mN.m':>10}{'mN.m':>10}{'mN.m':>11}{'':12}")
    for md, o in rows:
        bp = 1e3 * (o['+']['m_hat'] - o['+']['m_true'])
        bm = 1e3 * (o['-']['m_hat'] - o['-']['m_true'])
        hs = 0.5 * (bp + bm)
        print(f"  {md:6.2f}{bp:10.3f}{bm:10.3f}{hs:11.4f}"
              f"{1 - abs(hs)/max(abs(bp), abs(bm)):11.4%}")

    print(f"\n  the ground-effect channel, both parities\n")
    print(f"  {'Mdot':>6}{'GE parity':>12}{'half-sum bias':>16}"
          f"{'as offset':>12}{'cancels':>10}")
    print(f"  {'':6}{'':12}{'mN.m':>16}{'um':>12}")
    for md in (0.45, 1.20):
        base = one(md)['off_hat']
        for odd in (True, False):
            o = one(md, beta=BETA, ge_odd=odd)
            dh = 1e3 * (o['off_hat'] - base)
            print(f"  {md if odd else '':6}{'odd' if odd else 'even':>12}"
                  f"{dh:16.4f}{1e6*abs(dh)/1e3/W:12.2f}"
                  f"{'yes' if abs(dh) < 0.05 else 'NO':>10}")

    print(f"\n  what breaks it: the two directions not matched\n")
    print(f"  {'mismatch':>10}{'a_hat [mm]':>13}{'err [um]':>11}"
          f"{'rel err %':>12}")
    for mm in (0.0, 0.01, 0.02, 0.05):
        o = {}
        for sgn, arm, sc in ((+1.0, LP - A_TRUE, 1.0 + mm),
                             (-1.0, LP + A_TRUE, 1.0)):
            md = sgn * 0.45 * sc
            r = tipover(md, arm)
            d = fit(r, md)
            o['+' if sgn > 0 else '-'] = (sgn * W * arm,
                                          sgn * W * arm + md * d)
        off = 0.5 * (o['+'][1] + o['-'][1])
        a_hat = -off / W
        print(f"  {100*mm:9.0f}%{1e3*a_hat:13.4f}{1e6*(a_hat-A_TRUE):11.2f}"
              f"{100*(a_hat/A_TRUE-1):12.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
