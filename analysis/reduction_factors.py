#!/usr/bin/env python3
"""Do the reduction factors explain the shape of the residual bound?

The envelope of (93) uses rho_bar <= rho_phi,max/7 + rho_GE,max/5, and
the two constants are stand-ins for the window averages of the two
channels.  Since the measured residual is flat across the ramp rates
while RMS(E) falls sixfold, it is worth asking whether the constants are
what makes the difference -- whether computing them properly, per rate,
would flatten the bound.

It would not, and the reason is visible before any arithmetic: the true
reduction factors are themselves nearly flat.  Measured on the exact
nonlinear tip-over they run 0.094 to 0.124 for the gravity channel and
0.146 to 0.178 for the ground-effect one, against the 1/7 = 0.143 and
1/5 = 0.200 in use.  Both nominal values do bound their channel, with
15 to 52% of margin, but a factor that barely moves cannot produce a
shape.

So replacing them lowers RMS(E) by 1.3 to 1.7 and leaves the slope
almost untouched: the bound still falls 4.7-fold across the rates where
it fell 6.3-fold before, while the true RMS(e_omega) falls only 1.43.

The shape is in the propagation, not the reduction.  RMS(E) carries
sqrt(B(x)/x) with x = C2 tau_end, which grows with the window, and the
window is longest at the slowest ramp.  The true deviation has no such
factor: e_omega is the Duhamel integral of a rho set by the tilt
excursion, and every run covers the same excursion because every window
ends at the same tilt cap.  What differs is only how fast it is
traversed.

Usage: python analysis/reduction_factors.py [out.png]
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nonlinear_band import exact, J_P, RPHI, RGE

RATES = (0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20)
C_NOM, C_TRUE, C_E, C_PHI, C_GE = '#c0392b', '#148f77', '#2874a6', \
    '#2874a6', '#e08214'


def collect():
    out = []
    for m in RATES:
        s = exact(m, n=20001)
        tau, c2, rb = s['tau'], s['c2'], s['rb_sup']
        T = float(tau[-1])
        rms = lambda v: float(np.sqrt(np.trapz(v ** 2, tau) / T))
        rp, rg = np.abs(s['r_phi']), np.abs(s['r_ge'])
        fp = float(np.trapz(rp, tau) / T / rp.max())
        fg = float(np.trapz(rg, tau) / T / rg.max())
        rb_true = fp * rp.max() + fg * rg.max()
        k = np.sinh(c2 * tau) / (J_P * c2)
        out.append(dict(mdot=m, x=float(s['x']), fp=fp, fg=fg,
                        E_nom=rms(rb * k), E_true=rms(rb_true * k),
                        e=rms(s['e'])))
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'reduction_factors.png'
    r = collect()
    d2 = np.rad2deg
    md = [d['mdot'] for d in r]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.9))
    fig.subplots_adjust(left=0.068, right=0.985, top=0.845, bottom=0.135,
                        wspace=0.26)

    # ---- (a) the reduction factors themselves ----------------------
    a1.axhline(RPHI, color=C_PHI, ls='--', lw=1.6,
               label=r'nominal $1/7$, gravity')
    a1.axhline(RGE, color=C_GE, ls='--', lw=1.6,
               label=r'nominal $1/5$, ground effect')
    a1.plot(md, [d['fp'] for d in r], 'o-', color=C_PHI, lw=2.2, ms=7,
            label=r'true $\langle\rho_\varphi\rangle/\rho_{\varphi,\max}$')
    a1.plot(md, [d['fg'] for d in r], 's-', color=C_GE, lw=2.2, ms=7,
            label=r'true $\langle\rho_{GE}\rangle/\rho_{GE,\max}$')
    a1.set_ylim(0, 0.24)
    a1.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9.5)
    a1.set_ylabel('window average / maximum', fontsize=9.5)
    a1.set_title('(a) the reduction factors are flat\n'
                 'so they cannot produce a shape', fontsize=11)
    a1.legend(fontsize=8.5, loc='lower right')
    a1.grid(alpha=0.25, lw=0.4)

    # ---- (b) what that does to the bound ---------------------------
    en = np.array([d2(d['E_nom']) for d in r])
    et = np.array([d2(d['E_true']) for d in r])
    ee = np.array([d2(d['e']) for d in r])
    a2.plot(md, en, 'o-', color=C_NOM, lw=2.2, ms=7,
            label=f'$\\mathrm{{RMS}}(E)$, $1/7$ and $1/5$'
                  f'   ($\\times${en[0]/en[-1]:.1f})')
    a2.plot(md, et, 's-', color=C_E, lw=2.2, ms=7,
            label=f'$\\mathrm{{RMS}}(E)$, true factors'
                  f'   ($\\times${et[0]/et[-1]:.1f})')
    a2.plot(md, ee, '^-', color=C_TRUE, lw=2.4, ms=8,
            label=f'true $\\mathrm{{RMS}}(e_\\omega)$'
                  f'   ($\\times${ee[0]/ee[-1]:.2f})')
    a2.set_yscale('log')
    a2.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9.5)
    a2.set_ylabel(r'RMS [$^\circ$/s]', fontsize=9.5)
    a2.set_title('(b) correcting them lowers the bound but not its slope\n'
                 'only the truth is flat', fontsize=11)
    a2.set_ylim(0.10, 12.0)
    a2.legend(fontsize=8.5, loc='upper right')
    a2.grid(alpha=0.25, lw=0.4, which='both')

    fig.suptitle('The reduction factors are not what makes the bound fall',
                 fontsize=13, y=0.955)
    fig.savefig(out, dpi=150)

    print(f"\n  wrote {out}\n")
    print(f"  {'Mdot':>6}{'x':>6}{'factor phi':>12}{'factor GE':>11}"
          f"{'E, 1/7+1/5':>13}{'E, true':>10}{'true e':>10}")
    print(f"  {'N m/s':>6}{'':6}{'(1/7=.143)':>12}{'(1/5=.200)':>11}"
          f"{'deg/s':>13}{'deg/s':>10}{'deg/s':>10}")
    for d in r:
        print(f"  {d['mdot']:6.2f}{d['x']:6.2f}{d['fp']:12.4f}{d['fg']:11.4f}"
              f"{d2(d['E_nom']):13.3f}{d2(d['E_true']):10.3f}"
              f"{d2(d['e']):10.4f}")
    print(f"\n  across the rate range: the bound falls"
          f" {en[0]/en[-1]:.1f}x with the nominal factors and"
          f" {et[0]/et[-1]:.1f}x with the true ones,")
    print(f"  while the true deviation falls {ee[0]/ee[-1]:.2f}x."
          f"  Correcting the factors buys a level")
    print(f"  ({en[0]/et[0]:.1f}x at the slowest rate,"
          f" {en[-1]/et[-1]:.1f}x at the fastest), not a slope.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
