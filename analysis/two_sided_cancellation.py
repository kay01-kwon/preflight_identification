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

    relative error on the offset  ~  -(1/2) <phi^2>_w

a SCALE error rather than an additive bias, and an under-estimate.  The
magnitude is confirmed to within 23% at every ramp rate; the sign above
is what the measurement gives and the rest of the discrepancy is the two
windows not being the same length, since A_+ != A_- puts the tilt cap at
different times and the Wz channel then cancels imperfectly too.

The ground-effect channel is the bilinear term (iv) of (79),

    rho_GE(tau, phi) = beta_M Mdot tau phi,     beta_M < 0,

and its parity is settled by where beta_M comes from rather than by the
form, which on its face is even.  From (37)-(38) the effective hub
height is h_i = -(l_y,i + l_p) sin(phi) + h cos(phi), so

    beta_M  =  d/dphi [ sum_i (l_y,i + l_p) eta_i^GE b_i^dagger ]
            =  -c sum_i (l_y,i + l_p)^2 b_i^dagger .

Under the direction flip the pivot moves to the other edge, l_p -> -l_p,
the rotor labels mirror, l_y,i -> -l_y,i, and the allocation basis
reverses, b^dagger -> -b^dagger.  The squared bracket is unchanged and
b^dagger is not, so BETA_M FLIPS SIGN.  Then

    rho_GE  ->  (-beta_M)(-Mdot) tau (-phi)  =  -rho_GE :   ODD.

Two checks that this is the right reading.  k_0^GE = beta_f f l_p +
beta_M M_crit stays EVEN under the same flip, as it must -- it is quoted
in the paper as a single negative constant per axis, [-0.218, -0.169]
N m for roll -- and k_0^GE phi is then a restoring term in both
directions, which is what "the ground effect only reduces the net
destabilising gradient" means.  And a roll moment is odd under a mirror
about the x-z plane on general grounds.

So the ground-effect channel cancels too, and the even case below is a
counterfactual kept only to show what the parity is worth: 78 um of
offset if it were even against 0.7 um as it is.

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
BETA = -0.03446                        # beta_M < 0: the GE term is restoring
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


def fit(run, mdot, c2=C2, k_scale=1.0, dmax=0.25, nd=5001):
    """The deployed estimator: C1 and C2 pinned, onset swept, c closed form."""
    t, y = run['t'], run['om']
    c1 = k_scale * mdot / WZ
    w = np.gradient(t)
    w[0] *= 0.5
    w[-1] *= 0.5
    def cost(d):
        sh = np.where(t < d, 0.0,
                      c1 * (np.cosh(np.clip(c2 * (t - d), 0, 40)) - 1.0))
        c = float(np.sum((y - sh) * w) / w.sum())
        return float(np.sum((y - sh - c) ** 2 * w))

    ds = np.linspace(-dmax, dmax, nd)
    vs = np.array([cost(d) for d in ds])
    j = int(np.argmin(vs))
    # Sub-grid refinement.  Without it the residual bias measured below
    # sits at half a grid step and is quantisation, not physics: at
    # Mdot = 1.20 the two directions land on adjacent grid points and the
    # half-sum reads out the step, whatever the truth is.  The cost is
    # smooth and locally quadratic in d, so a three-point vertex is
    # enough; the parabola is refit twice on a shrinking bracket.
    lo, hi = ds[max(j - 1, 0)], ds[min(j + 1, nd - 1)]
    for _ in range(3):
        g = np.linspace(lo, hi, 21)
        v = np.array([cost(x) for x in g])
        k = int(np.argmin(v))
        if 0 < k < 20:
            a0, a1, a2 = v[k - 1], v[k], v[k + 1]
            den = a0 - 2 * a1 + a2
            step = 0.5 * (a0 - a2) / den * (g[1] - g[0]) if den > 0 else 0.0
            centre = g[k] + step
        else:
            centre = g[k]
        half = (g[1] - g[0]) * 1.5
        lo, hi = centre - half, centre + half
    return float(centre)                # onset offset from the TRUE onset


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


def ab(x):
    """A(x)/B(x): how strongly an amplitude error leaks into the onset."""
    a = 0.25 * np.cosh(2 * x) - np.cosh(x) + 0.75
    b = 0.25 * np.sinh(2 * x) - 0.5 * x
    return a / b


def miscalibrated(mdot_mag, eps_k=0.0, eps_c2=0.0, a=A_TRUE):
    """Both directions carrying the SAME calibration error.

    PNLS_CONSTANTS is keyed by (case, axis) with no direction split, so
    whatever rho does to C2 and K is common to the two runs.  That is the
    case worth measuring: a common error is exactly what the half-sum is
    built to reject, and the question is only how much survives.
    """
    out = {}
    for sgn, arm in ((+1.0, LP - a), (-1.0, LP + a)):
        md = sgn * mdot_mag
        r = tipover(md, arm, BETA, True)
        d = fit(r, md, c2=C2 * (1 + eps_c2), k_scale=1 + eps_k, nd=1201)
        out['+' if sgn > 0 else '-'] = dict(
            run=r, d=d, m_true=sgn * W * arm,
            m_hat=sgn * W * arm + md * d)
    p, m = out['+'], out['-']
    out['off_hat'] = 0.5 * (p['m_hat'] + m['m_hat'])
    out['a_hat'] = -out['off_hat'] / W
    return out


def amplitude_budget():
    """What the amplitude and exponent channels cost the OFFSET."""
    print("\n\n  the amplitude channel through the half-sum\n")
    md = 0.45
    x = C2 * (np.abs(one(md)['+']['run']['te']))
    print(f"  Mdot = {md} N m/s, x = C2 tau_end = {x:.3f},"
          f" A/B = {ab(x):.4f}")
    print(f"  per-direction prediction: eps * (Mdot/C2) * A/B"
          f" = eps * {1e3*md/C2*ab(x):.1f} mN.m\n")
    print(f"  {'eps_K':>7}{'bias +':>10}{'bias -':>10}{'half-sum':>11}"
          f"{'offset err':>12}{'cancelled':>11}")
    print(f"  {'%':>7}{'mN.m':>10}{'mN.m':>10}{'mN.m':>11}{'um':>12}")
    base = miscalibrated(md)['a_hat']
    for e in (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10):
        o = miscalibrated(md, eps_k=e)
        bp = 1e3 * (o['+']['m_hat'] - o['+']['m_true'])
        bm = 1e3 * (o['-']['m_hat'] - o['-']['m_true'])
        hs = 0.5 * (bp + bm)
        print(f"  {100*e:7.0f}{bp:10.3f}{bm:10.3f}{hs:11.4f}"
              f"{1e6*(o['a_hat']-base):12.3f}"
              f"{1 - abs(hs)/max(abs(bp), abs(bm)):10.3%}")

    print(f"\n  the exponent channel\n")
    print(f"  {'eps_C2':>7}{'bias +':>10}{'bias -':>10}{'half-sum':>11}"
          f"{'offset err':>12}{'cancelled':>11}")
    print(f"  {'%':>7}{'mN.m':>10}{'mN.m':>10}{'mN.m':>11}{'um':>12}")
    for e in (-0.05, -0.02, 0.02, 0.05):
        o = miscalibrated(md, eps_c2=e)
        bp = 1e3 * (o['+']['m_hat'] - o['+']['m_true'])
        bm = 1e3 * (o['-']['m_hat'] - o['-']['m_true'])
        hs = 0.5 * (bp + bm)
        print(f"  {100*e:7.0f}{bp:10.3f}{bm:10.3f}{hs:11.4f}"
              f"{1e6*(o['a_hat']-base):12.3f}"
              f"{1 - abs(hs)/max(abs(bp), abs(bm)):10.3%}")

    # The half-sum is immune, but (34a)-(34b) read the weight from the
    # DIFFERENCE, which takes 2x the per-direction bias, and W then sits
    # in the denominator of (35a)-(35d).  That is where these channels
    # actually reach the offset.
    print(f"\n  what does NOT cancel: (34) reads W from the DIFFERENCE\n")
    print(f"  W_Mx = f_crit + (M_x,+ - M_x,-)/(l_r + l_l), so the bias adds")
    print(f"  instead of cancelling:  dW = 2 eps (Mdot/C2)(A/B)/(l_r + l_l)\n")
    span = 2 * LP                       # l_r + l_l; replace with the measured
    print(f"  taking l_r + l_l = {span:.3f} m and W = 30.08 N\n")
    print(f"  {'channel':>10}{'per 1%':>11}{'dW per 1%':>12}{'dW/W':>10}"
          f"{'at 5%':>10}")
    print(f"  {'':10}{'mN.m':>11}{'N':>12}{'%':>10}{'%':>10}")
    for name, kw in (('amplitude', dict(eps_k=0.01)),
                     ('exponent', dict(eps_c2=0.01))):
        o = miscalibrated(md, **kw)
        bp = o['+']['m_hat'] - o['+']['m_true']
        bm = o['-']['m_hat'] - o['-']['m_true']
        dw = (bp - bm) / span
        print(f"  {name:>10}{1e3*bp:11.3f}{dw:12.5f}{100*dw/30.08:10.4f}"
              f"{500*dw/30.08:10.3f}")

    print(f"\n  the realised case: what rho does to the PNLS calibration")
    print(f"  (C2 +1.4%, K -5.2%, measured in analysis/absorption_picture.py)\n")
    print(f"  {'Mdot':>6}{'bias +':>10}{'bias -':>10}{'half-sum':>11}"
          f"{'offset err':>12}")
    print(f"  {'N m/s':>6}{'mN.m':>10}{'mN.m':>10}{'mN.m':>11}{'um':>12}")
    for m0 in (0.10, 0.45, 1.20):
        b0 = miscalibrated(m0)['a_hat']
        o = miscalibrated(m0, eps_k=-0.052, eps_c2=0.014)
        bp = 1e3 * (o['+']['m_hat'] - o['+']['m_true'])
        bm = 1e3 * (o['-']['m_hat'] - o['-']['m_true'])
        print(f"  {m0:6.2f}{bp:10.3f}{bm:10.3f}{0.5*(bp+bm):11.4f}"
              f"{1e6*(o['a_hat']-b0):12.3f}")


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

    print(f"\n  the ground-effect channel; ODD is the physical one\n")
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
                  f"{('yes' if abs(dh) < 0.05 else 'NO') + ('' if odd else ' (n/a)'):>10}")

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
    amplitude_budget()
    return 0


if __name__ == '__main__':
    sys.exit(main())
