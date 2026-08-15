#!/usr/bin/env python3
"""The band checked against the EXACT nonlinear tip-over, not a linearisation.

Everything drawn so far used the first-order construction: a linear
nominal plus the Duhamel integral of rho evaluated along that nominal.
That is what the derivation uses, so it is what the derivation should be
checked against -- but it is not what a vehicle does.  This integrates
the exact equation instead,

    J_P phi'' = M(t) - W A cos(phi) + W z sin(phi),   M(t) = Mdot t + W A,

from the onset to the tilt cap, and compares the resulting rate with the
nominal cosh and with the a priori upper edge of (93).  Nothing is
expanded; phi is whatever the ODE gives.

Constants are made self-consistent for this purpose: Wz = W z_CoM from
the geometry and C2 = sqrt(Wz/J_P), so the identity that (93) rests on
holds exactly.  (The band scripts elsewhere use the PNLS C2 with Wz
derived from it, which is the same identity read the other way.)

The small-tau scalings are worth confirming while the exact solution is
in hand, because the whole envelope argument rests on them:

    omega_nom ~ tau^2,   phi ~ tau^3,
    rho_phi ~ phi^2 ~ tau^6,   rho_GE ~ Mdot tau phi ~ tau^4  (restoring),
    e_omega = int cosh(.) rho ~ tau^7,

while the upper edge uses the ENDPOINT rho_bar and so grows only as
sinh(C2 tau) ~ tau.  That mismatch, tau^7 against tau^1, is why the
bound is enormously loose near the onset and why the occupancy curves
start at zero.

Usage: python analysis/nonlinear_band.py [out.png]
"""
import sys

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W, LP, ZCOM, A_OFF = 31.59, 0.160, 0.30, 0.0020
J_P = 0.4260
WZ = W * ZCOM                          # the destabilising gain, from geometry
C2 = np.sqrt(WZ / J_P)                 # the identity, not an independent value
PHI_MAX = np.deg2rad(10.0)
BETA_M = -0.03446                      # restoring
RPHI, RGE = 1.0 / 7.0, 1.0 / 5.0
ARM = LP - A_OFF                       # the + direction restoring arm
C_NOM, C_SUP, C_TRUE, C_ACT = '#2874a6', '0.45', '#c0392b', '#148f77'


def exact(mdot, n=6001, beta=BETA_M, use_eff=True):
    """Integrate the tip-over exactly, from the onset to the tilt cap.

    The ground-effect moment beta_M M(tau) phi must be SPLIT before any
    of it is called a remainder.  Writing M = M_crit + Mdot tau,

        beta_M M phi = beta_M M_crit phi  +  beta_M Mdot tau phi
                       \____ k_0^GE phi ___/   \___ bilinear ___/

    and (79) groups the first with gravity as (Wz + k_0^GE) phi -- it is
    a stiffness, not a forcing.  Only the bilinear term is rho_GE.
    Putting the whole thing in the forcing, as a first attempt at this
    script did, makes rho negative over most of the window (the linear
    piece is O(phi) against the gravity remainder's O(phi^2)) and turns
    the deviation the wrong way.
    """
    tc = W * ARM / mdot
    k0 = beta * (W * ARM)               # beta_M M_crit, the GE stiffness
    wz_eff = WZ + k0
    c2_eff = np.sqrt(wz_eff / J_P) if use_eff else C2
    c1_eff = mdot / (J_P * c2_eff ** 2)

    def f(t, s):
        phi, om = s
        return [om, (mdot * t + W * ARM * (1.0 - np.cos(phi))
                     + WZ * np.sin(phi)
                     + beta * (mdot * (tc + t)) * phi) / J_P]

    def cap(t, s):
        return s[0] - PHI_MAX
    cap.terminal, cap.direction = True, 1
    sol = solve_ivp(f, (0.0, 60.0), [0.0, 0.0], events=cap,
                    rtol=1e-11, atol=1e-13, dense_output=True)
    te = float(sol.t_events[0][0])
    tau = np.linspace(0.0, te, n)
    phi, om = sol.sol(tau)
    nom = c1_eff * (np.cosh(c2_eff * tau) - 1.0)
    # rho is what the exact right-hand side carries and the LINEARISED
    # one, stiffness k_0^GE included, does not
    r_phi = W * ARM * (1.0 - np.cos(phi)) + WZ * (np.sin(phi) - phi)
    r_ge = beta * mdot * tau * phi
    rho = r_phi + r_ge
    dmw = mdot * te
    rb_sup = (RPHI * 0.5 * W * LP * PHI_MAX ** 2
              + RGE * abs(beta) * dmw * PHI_MAX)
    rb_true = float(np.trapz(rho, tau) / te)
    k = np.sinh(c2_eff * tau) / (J_P * c2_eff)
    return dict(mdot=mdot, tau=tau, te=te, x=c2_eff * te, phi=phi, om=om,
                c2=c2_eff, k0=k0,
                nom=nom, e=om - nom, rho=rho, r_phi=r_phi, r_ge=r_ge,
                E_sup=rb_sup * k, E_true=rb_true * k,
                rb_sup=rb_sup, rb_true=rb_true, c1=c1_eff)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'nonlinear_band.png'
    S, F = exact(0.10), exact(1.20)
    print(f"\n  exact nonlinear tip-over.  W z_CoM = {WZ:.3f} N m,"
          f" k_0^GE = {S['k0']:.4f} N m/rad")
    print(f"  so Wz_eff = {WZ + S['k0']:.3f} and C2_eff = {S['c2']:.4f}"
          f" against C2 = {C2:.4f}, a shift of"
          f" {100*(S['c2']/C2 - 1):+.2f}%")
    print(f"  arm = {ARM:.4f} m, tilt cap {np.rad2deg(PHI_MAX):.0f} deg\n")
    print(f"  {'Mdot':>6}{'tau_end':>9}{'x':>7}{'phi_end':>9}{'omega_end':>11}"
          f"{'e_end':>10}{'E_sup end':>11}{'occupancy':>11}")
    print(f"  {'N m/s':>6}{'s':>9}{'':7}{'deg':>9}{'rad/s':>11}{'rad/s':>10}"
          f"{'rad/s':>11}{'%':>11}")
    for d in (S, F):
        print(f"  {d['mdot']:6.2f}{d['te']:9.4f}{d['x']:7.3f}"
              f"{np.rad2deg(d['phi'][-1]):9.3f}{d['om'][-1]:11.5f}"
              f"{d['e'][-1]:10.5f}{d['E_sup'][-1]:11.5f}"
              f"{100*d['e'][-1]/d['E_sup'][-1]:11.2f}")

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.055, right=0.985, top=0.885, bottom=0.085)

    for k, (d, tag) in enumerate(((F, 'a'), (S, 'b'))):
        ax = fig.add_subplot(gs[0, k])
        t = d['tau']
        ax.fill_between(t, d['nom'], d['nom'] + d['E_sup'], color=C_SUP,
                        alpha=0.20)
        ax.plot(t, d['nom'], '-', color=C_NOM, lw=2.2, label='nominal cosh')
        ax.plot(t, d['nom'] + d['E_sup'], '--', color=C_SUP, lw=1.7,
                label=r'upper edge, $\bar\rho$ sup')
        ax.plot(t, d['nom'] + d['E_true'], ':', color=C_TRUE, lw=1.8,
                label=r'upper edge, $\bar\rho$ true')
        ax.plot(t, d['om'], '-', color=C_ACT, lw=2.4,
                label='exact nonlinear')
        ax.set_title(f"({tag}) $\\dot M$ = {d['mdot']:.2f}, "
                     f"$x$ = {d['x']:.2f}, "
                     f"{100*d['e'][-1]/d['E_sup'][-1]:.1f}% of the band",
                     fontsize=10.5)
        ax.set_xlabel(r'$\tau$ [s]', fontsize=9)
        ax.set_ylabel(r'$\omega$ [rad/s]', fontsize=9)
        ax.legend(fontsize=7.8, loc='upper left')
        ax.grid(alpha=0.25, lw=0.4)

    # (c) the deviation on log-log, signed, against tau^5 and tau^7
    ax = fig.add_subplot(gs[0, 2])
    for d, ls, nm in ((F, '-', '1.20'), (S, '--', '0.10')):
        t, e = d['tau'][1:], d['e'][1:]
        neg, pos = e < 0, e > 0
        ax.loglog(t[neg], -e[neg], ls, color=C_B, lw=2.1,
                  label=f'$-e_\\omega$ (GE wins), {nm}')
        ax.loglog(t[pos], e[pos], ls, color=C_ACT, lw=2.1,
                  label=f'$+e_\\omega$ (gravity wins), {nm}')
        ax.loglog(t, d['E_sup'][1:], ls, color=C_SUP, lw=1.5,
                  label=f'$E$, {nm}')
        j = int(np.argmax(np.diff(np.sign(e)) != 0)) + 1
        ax.plot(t[j], abs(e[j]), 'kv', ms=6)
    t0 = F['tau'][1:]
    ax.loglog(t0, 2e1 * t0 ** 5, ':', color='k', lw=1.2, label=r'$\tau^5$')
    ax.loglog(t0, 3e2 * t0 ** 7, '-.', color='k', lw=1.2, label=r'$\tau^7$')
    ax.set_ylim(1e-11, 1e1)
    ax.set_title(r'(c) $\rho_{GE}\sim\tau^4$ leads, so $e_\omega\sim-\tau^5$'
                 '\nfirst; gravity $\\tau^6$ takes over at the marker',
                 fontsize=10.5)
    ax.set_xlabel(r'$\tau$ [s]', fontsize=9)
    ax.set_ylabel('[rad/s]', fontsize=9)
    ax.legend(fontsize=6.4, ncol=2, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4, which='both')

    # (d) the two rho channels, with their small-tau powers
    ax = fig.add_subplot(gs[1, 0])
    for d, ls, nm in ((F, '-', '1.20'), (S, '--', '0.10')):
        m = d['tau'] > 0
        ax.loglog(d['tau'][m], np.abs(d['r_phi'][m]) * 1e3, ls, color=C_B
                  if False else C_TRUE, lw=2.1,
                  label=f'$|\\rho_\\varphi|$, {nm}')
        ax.loglog(d['tau'][m], np.abs(d['r_ge'][m]) * 1e3, ls, color=C_NOM,
                  lw=1.7, label=f'$|\\rho_{{GE}}|$, {nm}')
    ax.loglog(t0, 2e5 * t0 ** 6, '-.', color='k', lw=1.1, label=r'$\tau^6$')
    ax.loglog(t0, 1e2 * t0 ** 4, ':', color='k', lw=1.1, label=r'$\tau^4$')
    ax.set_ylim(1e-8, 1e3)
    ax.set_title(r'(d) $\rho_\varphi\sim\tau^6$, $\rho_{GE}\sim\tau^4$'
                 '\nand the GE channel subtracts', fontsize=10.5)
    ax.set_xlabel(r'$\tau$ [s]', fontsize=9)
    ax.set_ylabel('[mN.m]', fontsize=9)
    ax.legend(fontsize=7.0, ncol=2, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4, which='both')

    # (e) exact against the first-order construction
    ax = fig.add_subplot(gs[1, 1])
    for d, ls, nm in ((F, '-', '1.20'), (S, '--', '0.10')):
        lin = np.zeros_like(d['tau'])
        for i in range(1, len(d['tau'])):
            ker = np.cosh(np.clip(d['c2'] * (d['tau'][i] - d['tau'][:i+1]),
                                  0, 40))
            lin[i] = np.trapz(ker * d['rho'][:i + 1],
                              d['tau'][:i + 1]) / J_P
        # mask the sign change, where the ratio is 0/0
        m = (d['tau'] > 0.15 * d['te']) & (np.abs(d['e']) > 1e-3 * np.abs(
            d['e']).max())
        ax.plot(d['tau'][m] / d['te'], 100 * (lin[m] / d['e'][m] - 1), ls,
                color=C_ACT, lw=2.1, label=f'$\\dot M$={nm}')
    ax.axhline(0, color='k', lw=1.0, ls=':')
    ax.set_title('(e) first-order Duhamel against the exact\n'
                 'the linearisation the derivation uses', fontsize=10.5)
    ax.set_xlabel(r'$\tau/\tau_{\rm end}$', fontsize=9)
    ax.set_ylabel('linearised / exact $-1$ [%]', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # (f) occupancy through the window
    ax = fig.add_subplot(gs[1, 2])
    for d, ls, nm in ((F, '-', '1.20'), (S, '--', '0.10')):
        m = d['tau'] > 0
        ax.plot(d['tau'][m] / d['te'], 100 * d['e'][m] / d['E_sup'][m], ls,
                color=C_SUP, lw=2.1, label=f'vs sup, {nm}')
        ax.plot(d['tau'][m] / d['te'], 100 * d['e'][m] / d['E_true'][m], ls,
                color=C_TRUE, lw=2.1, label=f'vs true, {nm}')
    ax.set_title('(f) occupancy, exact dynamics\n'
                 'the fast ramp fills more of the band', fontsize=10.5)
    ax.set_xlabel(r'$\tau/\tau_{\rm end}$', fontsize=9)
    ax.set_ylabel('% of band', fontsize=9)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25, lw=0.4)

    fig.suptitle('The band against the exact nonlinear tip-over, '
                 r'$\dot M = 0.10$ and $1.20$ N m/s', fontsize=13.5, y=0.965)
    fig.savefig(out, dpi=145)
    print(f"\n  the sign change: rho_GE ~ tau^4 leads rho_phi ~ tau^6, so")
    print(f"  e_omega starts NEGATIVE and crosses over\n")
    print(f"  {'Mdot':>6}{'cross at':>10}{'as % of':>10}{'peak neg':>11}"
          f"{'end pos':>10}{'ratio':>9}")
    print(f"  {'N m/s':>6}{'s':>10}{'window':>10}{'urad/s':>11}"
          f"{'mrad/s':>10}{'|neg|/end':>9}")
    for d in (S, F):
        e, t = d['e'], d['tau']
        # the LAST sign change is the meaningful crossover; the first is
        # at the numerical floor where both channels are ~1e-12
        k = np.nonzero(np.diff(np.sign(e[1:])) != 0)[0]
        j = int(k[-1]) + 1 if len(k) else 0
        print(f"  {d['mdot']:6.2f}{t[j]:10.4f}{100*t[j]/d['te']:9.1f}%"
              f"{1e6*e.min():11.4f}{1e3*e[-1]:10.4f}"
              f"{1e6*abs(e.min())/(1e3*e[-1])/1e3:9.2e}")
    print(f"\n  So the ground-effect channel does lead -- it is tau^4"
          f" against tau^6 --")
    print(f"  and e_omega really does start negative, but the excursion"
          f" peaks five")
    print(f"  orders below the final value.  It is a feature of the"
          f" derivation, not")
    print(f"  a budget item.  The first-order Duhamel the derivation uses"
          f" is exact")
    print(f"  to better than 0.01% away from the crossing.")
    print(f"\n  wrote {out}")
    return 0


C_B = '#c0392b'

if __name__ == '__main__':
    sys.exit(main())
