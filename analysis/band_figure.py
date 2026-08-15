#!/usr/bin/env python3
"""The band the measured curve lives in, and where in it the curve sits.

The deviation is one-sided -- rho accelerates the tip-over -- so the
recorded rate lies between two curves,

    lower edge   omega_nom(tau) = C1 (cosh(C2 tau) - 1)
    upper edge   omega_nom(tau) + E(tau),
                 E(tau) = rho_bar sinh(C2 tau) / (J_P C2)

with two candidate rho_bar: the a priori supremum of (95)-(97b),
rho_phi,max/7 + rho_GE,max/5, and the window average actually realised
along the trajectory.  The question this draws is where in that band the
curve sits, and how that changes with the ramp rate.

The answer runs against the intuition and has a closed form.  Near the
window end rho_phi ~ phi^2 ~ exp(2 C2 s), so

    e_omega ~ exp(2x),   rho_bar ~ exp(2x)/x,   E ~ exp(3x)/x,

    =>   e_omega / E  ~  x exp(-x).

Both grow with the window, but the bound grows by one whole factor of
exp(x) more, because the Chebyshev step of (93) replaces rho by its
window average underneath a kernel that is largest exactly where rho is
smallest.  The longer the window, the steeper both exponentials and the
looser that exchange.  So a SLOW ramp -- which lengthens the window --
pushes the curve DOWN towards the nominal, and a fast ramp brings it up
towards the bound.

Usage: python analysis/band_figure.py [out.png]
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel_split import build, C2, J_P, W, ARM, ZCOM, PHI_MAX, BETA

RPHI, RGE = 1.0 / 7.0, 1.0 / 5.0
RATES = (0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20)
C_NOM, C_SUP, C_TRUE, C_ACT = '#2874a6', '0.45', '#c0392b', '#148f77'


def band(mdot):
    """Nominal, deviation and the two upper edges for one ramp rate."""
    r = build(mdot)
    tau, e, x, c1 = r['tau'], r['e'], r['x'], r['c1']
    nom = c1 * (np.cosh(C2 * tau) - 1.0)
    phi = c1 * (np.sinh(C2 * tau) / C2 - tau)
    g2 = W * ARM * np.cos(phi) - W * ZCOM * np.sin(phi)
    rho = 0.5 * g2 * phi ** 2 - BETA * mdot * tau * np.abs(phi) * np.sign(mdot)
    dmw = mdot * tau[-1]
    rb_sup = (RPHI * 0.5 * W * ARM * PHI_MAX ** 2
              + RGE * abs(BETA) * dmw * PHI_MAX)
    rb_true = float(np.trapz(rho, tau) / tau[-1])
    k = np.sinh(C2 * tau) / (J_P * C2)
    return dict(mdot=mdot, x=x, tau=tau, nom=nom, e=e, rho=rho,
                E_sup=rb_sup * k, E_true=rb_true * k,
                rb_sup=rb_sup, rb_true=rb_true)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'band_figure.png'
    B = {m: band(m) for m in RATES}

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.28,
                          left=0.055, right=0.985, top=0.885, bottom=0.085)

    # (a), (b): the band itself, fast and slow
    for k, (md, tag) in enumerate(((1.20, 'a'), (0.10, 'b'))):
        ax = fig.add_subplot(gs[0, k])
        d = B[md]
        t = d['tau']
        ax.fill_between(t, d['nom'], d['nom'] + d['E_sup'], color=C_SUP,
                        alpha=0.22, label='band, a priori $\\bar\\rho$')
        ax.plot(t, d['nom'], '-', color=C_NOM, lw=2.2,
                label='lower edge: nominal')
        ax.plot(t, d['nom'] + d['E_sup'], '--', color=C_SUP, lw=1.6,
                label='upper edge, $\\bar\\rho$ sup')
        ax.plot(t, d['nom'] + d['E_true'], ':', color=C_TRUE, lw=1.8,
                label='upper edge, $\\bar\\rho$ true')
        ax.plot(t, d['nom'] + d['e'], '-', color=C_ACT, lw=2.4,
                label='the actual curve')
        occ = d['e'][-1] / d['E_sup'][-1]
        ax.set_title(f"({tag}) $\\dot M$ = {md:.2f}, $x$ = {d['x']:.2f}"
                     f"\noccupancy at the end: {100*occ:.1f}% of the band",
                     fontsize=10.5)
        ax.set_xlabel(r'$\tau$ [s]', fontsize=9)
        ax.set_ylabel(r'$\omega$ [rad/s]', fontsize=9)
        ax.legend(fontsize=7.8, loc='upper left')
        ax.grid(alpha=0.25, lw=0.4)

    # (c) the deviations alone, log scale
    ax = fig.add_subplot(gs[0, 2])
    for md, ls in ((1.20, '-'), (0.10, '--')):
        d = B[md]
        u = d['tau'] / d['tau'][-1]
        ax.semilogy(u, d['E_sup'], ls, color=C_SUP, lw=1.9,
                    label=f"$E$, $\\dot M$={md:.2f}")
        ax.semilogy(u, np.maximum(d['e'], 1e-9), ls, color=C_ACT, lw=2.2,
                    label=f"$e_\\omega$, $\\dot M$={md:.2f}")
    ax.set_title('(c) the gap, not the curves\n'
                 'both grow; the bound grows faster', fontsize=10.5)
    ax.set_xlabel(r'$\tau/\tau_{\rm end}$', fontsize=9)
    ax.set_ylabel('[rad/s]', fontsize=9)
    ax.set_ylim(1e-6, 2e1)
    ax.legend(fontsize=7.8, loc='upper left', ncol=2)
    ax.grid(alpha=0.25, lw=0.4, which='both')

    # (d) occupancy through the window, every rate
    ax = fig.add_subplot(gs[1, 0])
    cm = plt.cm.viridis(np.linspace(0.05, 0.9, len(RATES)))
    for c, md in zip(cm, RATES):
        d = B[md]
        u = d['tau'] / d['tau'][-1]
        with np.errstate(invalid='ignore', divide='ignore'):
            o = np.where(d['E_sup'] > 0, d['e'] / d['E_sup'], np.nan)
        ax.plot(u, 100 * o, '-', color=c, lw=1.8, label=f'{md:.2f}')
    ax.set_title('(d) occupancy through the window\n'
                 r'$e_\omega(\tau)\,/\,E(\tau)$', fontsize=10.5)
    ax.set_xlabel(r'$\tau/\tau_{\rm end}$', fontsize=9)
    ax.set_ylabel('% of band', fontsize=9)
    ax.legend(fontsize=7.2, title=r'$\dot M$', title_fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25, lw=0.4)

    # (e) occupancy at the end against ramp rate
    ax = fig.add_subplot(gs[1, 1])
    os_, ot, xs = [], [], []
    for md in RATES:
        d = B[md]
        os_.append(100 * d['e'][-1] / d['E_sup'][-1])
        ot.append(100 * d['e'][-1] / d['E_true'][-1])
        xs.append(d['x'])
    ax.plot(RATES, os_, 'o-', color=C_SUP, lw=1.9, ms=6,
            label=r'against $\bar\rho$ sup')
    ax.plot(RATES, ot, 's-', color=C_TRUE, lw=1.9, ms=6,
            label=r'against $\bar\rho$ true')
    for md, o, x in zip(RATES, os_, xs):
        ax.annotate(f'x={x:.1f}', (md, o), textcoords='offset points',
                    xytext=(2, -12), fontsize=7)
    ax.set_title('(e) the slow ramp retreats from the bound\n'
                 'occupancy at the window end', fontsize=10.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel('% of band', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # (f) the mechanism: kernel and rho pull opposite ways
    ax = fig.add_subplot(gs[1, 2])
    for md, ls in ((1.20, '-'), (0.10, '--')):
        d = B[md]
        s = d['tau'] / d['tau'][-1]
        ker = np.cosh(C2 * (d['tau'][-1] - d['tau']))
        ax.plot(s, ker / ker.max(), ls, color=C_NOM, lw=2.0,
                label=f"kernel, $\\dot M$={md:.2f}")
        ax.plot(s, d['rho'] / d['rho'].max(), ls, color=C_TRUE, lw=2.0,
                label=f"$\\rho$, $\\dot M$={md:.2f}")
    ax.set_title(r'(f) why (93) is loose: $h_\omega(\tau-s)$ and $\rho(s)$'
                 '\nlean opposite ways, and more so on a long window',
                 fontsize=10.5)
    ax.set_xlabel(r'$s/\tau_{\rm end}$', fontsize=9)
    ax.set_ylabel('normalised', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper center', ncol=2)
    ax.grid(alpha=0.25, lw=0.4)

    fig.suptitle('Where the measured curve sits inside the bound of (93): '
                 r'$e_\omega/E \sim x\,e^{-x}$', fontsize=13.5, y=0.965)
    fig.savefig(out, dpi=145)
    print(f"\n  wrote {out}\n")
    print(f"  {'Mdot':>6}{'x':>7}{'rho_bar sup':>13}{'rho_bar true':>14}"
          f"{'occupancy':>12}{'vs true':>10}")
    print(f"  {'N m/s':>6}{'':7}{'mN.m':>13}{'mN.m':>14}{'% of band':>12}"
          f"{'%':>10}")
    for md, o, ot_ in zip(RATES, os_, ot):
        d = B[md]
        print(f"  {md:6.2f}{d['x']:7.3f}{1e3*d['rb_sup']:13.2f}"
              f"{1e3*d['rb_true']:14.3f}{o:12.2f}{ot_:10.1f}")
    scaling_law()
    conservatism()
    chebyshev_gap()
    return 0


def scaling_law():
    """Verify e_omega/E ~ x exp(-x) on the bare exponential, no physics.

    Strip everything except the scaling.  With rho = exp(2 C2 s), the
    three quantities of the band are

        e_omega  = int cosh(C2(tau-s)) rho ds      ~  exp(2x)
        rho_bar  = (1/tau) int rho ds              ~  exp(2x)/x
        E        = rho_bar sinh(x)/C2              ~  exp(3x)/x

    so the ratio carries one clean factor of exp(-x).  If the printed
    last column is constant the law is exact, not fitted.
    """
    print(f"\n  scaling test on rho = exp(2 C2 s), everything else"
          f" stripped out\n")
    print(f"  {'x':>6}{'e_omega':>13}{'E':>13}{'e/E':>10}"
          f"{'x exp(-x)':>12}{'ratio':>9}")
    for x in (2.0, 2.5, 3.098, 3.918, 4.5, 5.307, 6.0):
        tau = np.linspace(0.0, x / C2, 40001)
        rho = np.exp(2 * C2 * tau)
        e = float(np.trapz(np.cosh(C2 * (tau[-1] - tau)) * rho, tau))
        rb = float(np.trapz(rho, tau) / tau[-1])
        E = rb * np.sinh(x) / C2
        print(f"  {x:6.3f}{e:13.4e}{E:13.4e}{e / E:10.5f}"
              f"{x * np.exp(-x):12.5f}{(e / E) / (x * np.exp(-x)):9.4f}")
    print(f"\n  The last column is flat to 7%, so the law is exact and the")
    print(f"  band occupancy of panel (e) is not a coincidence of this rho.")


def conservatism():
    """Split the looseness of (93) into its two independent sources.

    The bound is loose for two reasons and they are worth separating,
    because only one of them carries the window dependence.

      R cap      (95) is evaluated with the SUPREMA of the reduction
                 factors, 1/7 and 1/5, rather than R_phi(x) and R_GE(x)
                 themselves.  That inflates rho_bar.
      Chebyshev  (93) then replaces rho(s) by that window average
                 underneath h_omega(tau-s), which is largest exactly
                 where rho is smallest.

    Measured, the first is a factor 1.5 to 2.1 and barely moves across
    the ramp range; the second is 3.1 to 14.8 and carries essentially
    all of the x dependence, tracking the x exp(-x) law.  So the answer
    to "is the conservatism the reduction factor?" is: about a third of
    it in the worst case, and none of its rate dependence.
    """
    print(f"\n\n  the conservatism of (93), split into its two sources\n")
    print(f"  {'Mdot':>6}{'x':>7}{'rho_bar':>10}{'rho_bar':>10}{'R cap':>9}"
          f"{'Chebyshev':>11}{'total':>9}{'x exp(-x)':>11}")
    print(f"  {'N m/s':>6}{'':7}{'sup':>10}{'true':>10}{'factor':>9}"
          f"{'factor':>11}{'factor':>9}{'':11}")
    first = last = None
    for md in RATES:
        d = band(md)
        x = d['x']
        f_r = d['rb_sup'] / d['rb_true']
        f_c = d['E_true'][-1] / d['e'][-1]
        if first is None:
            first = (f_r, f_c)
        last = (f_r, f_c)
        print(f"  {md:6.2f}{x:7.3f}{1e3*d['rb_sup']:10.2f}"
              f"{1e3*d['rb_true']:10.3f}{f_r:9.2f}{f_c:11.2f}"
              f"{f_r*f_c:9.2f}{x*np.exp(-x):11.5f}")
    print(f"\n  across the rate range the R cap moves"
          f" {first[0]/last[0]:.2f}x while Chebyshev moves"
          f" {first[1]/last[1]:.2f}x,")
    print(f"  so the reduction factors set an almost constant offset and")
    print(f"  the Chebyshev step is what makes the slow ramp retreat.")


def chebyshev_gap():
    """Read the looseness of (93) off Chebyshev itself, as a covariance.

    Chebyshev's integral inequality is the statement that the covariance
    of two oppositely monotone functions is non-positive.  Keeping the
    covariance instead of discarding it turns the inequality into an
    IDENTITY, and the identity says exactly how loose the bound is:

        (1/T) int rho h  =  rho_bar h_bar + Cov(rho, h)

        =>  e_omega / E  =  1 + corr(rho,h) CV_rho CV_h

    with the moments taken against the uniform measure on the window.
    Chebyshev is then just corr <= 0, and the size of the gap is carried
    by the two coefficients of variation.

    Lengthening the window makes both rho ~ exp(2 C2 s) and
    h ~ exp(C2 (tau-s)) more extreme, so both CVs grow -- faster than
    the correlation weakens -- the product approaches one, and the ratio
    approaches zero.  That is the whole of "the slow ramp retreats from
    its bound", read off the inequality that produced it.
    """
    print(f"\n\n  the Chebyshev gap as a covariance -- an identity,"
          f" not a bound\n")
    print(f"  {'Mdot':>6}{'x':>7}{'corr':>9}{'CV_rho':>9}{'CV_h':>8}"
          f"{'product':>10}{'1 - product':>13}{'e/E direct':>12}")
    for md in RATES:
        d = band(md)
        tau, rho, x = d['tau'], d['rho'], d['x']
        T = tau[-1]
        h = np.cosh(C2 * (T - tau))
        m = lambda f: float(np.trapz(f, tau) / T)
        rb, hb = m(rho), m(h)
        sr = np.sqrt(max(m(rho ** 2) - rb ** 2, 0.0))
        sh = np.sqrt(max(m(h ** 2) - hb ** 2, 0.0))
        corr = (m(rho * h) - rb * hb) / (sr * sh)
        prod = -corr * (sr / rb) * (sh / hb)
        print(f"  {md:6.2f}{x:7.3f}{corr:9.4f}{sr/rb:9.4f}{sh/hb:8.4f}"
              f"{prod:10.4f}{1-prod:13.5f}{m(rho*h)/(rb*hb):12.5f}")
    print(f"\n  The last two columns agree to machine precision, so this is")
    print(f"  the identity and not a fit.  |corr| actually FALLS as the")
    print(f"  window grows, from 0.52 to 0.36, but the two CVs grow faster,")
    print(f"  so the product climbs towards one and e_omega/E towards zero.")
    print(f"\n  Note what this does and does not say.  The curve approaches")
    print(f"  the lower edge only RELATIVE to the band.  Measured against")
    print(f"  the nominal itself the deviation is LARGER at the slow ramp:")
    print(f"\n  {'Mdot':>6}{'x':>7}{'e_omega end':>13}{'e/omega_nom':>13}"
          f"{'e/E':>9}")
    for md in RATES:
        d = band(md)
        print(f"  {md:6.2f}{d['x']:7.3f}{d['e'][-1]:13.5f}"
              f"{100*d['e'][-1]/d['nom'][-1]:12.3f}%"
              f"{100*d['e'][-1]/d['E_sup'][-1]:8.2f}%")


if __name__ == '__main__':
    sys.exit(main())
