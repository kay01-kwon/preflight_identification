#!/usr/bin/env python3
"""Why the bound depends on the ramp rate when the data does not.

The measured residual is flat across a twelvefold change in ramp rate --
Pearson r = +0.09 against Mdot, p = 0.30 -- while RMS(E) falls 6.3-fold.
That reads as a disagreement between the data and the theory, and it is
not one.  Three measurements settle it.

  the deviation the theory is about is ALSO flat.  Integrating the exact
  nonlinear tip-over, RMS(e_omega) falls 1.43-fold across the rates,
  0.236 to 0.165 deg/s, against the bound's 6.26.  Theory and data agree
  on the residual; it is the bound that moves.

  the reduction factors are not what moves it.  The true window-average
  ratios are 0.094-0.124 for gravity and 0.146-0.178 for ground effect,
  against the 1/7 = 0.143 and 1/5 = 0.200 in use.  Both nominal values
  bound their channel with 15-52% of margin, so the choice is sound, but
  a factor that barely moves cannot produce a shape: replacing them
  lowers RMS(E) by 1.3-1.7x and turns 6.26-fold into 4.7-fold.

  what moves it is the propagation.  RMS(E) = rho_bar K C2 sqrt(B(x)/x)
  with x = C2 tau_end, and sqrt(B(x)/x) alone falls 6.69-fold across the
  rate range -- the whole of the 6.26.  rho_bar itself barely moves,
  11.13 to 11.90 mN.m, a rise of 7%.

So the bound's rate dependence is conservatism, not a prediction about
the residual, and the conservatism has a closed form.  E/e_omega falls
34.1 to 7.8, a factor 4.37, against the 4.98 that the x e^-x law of
(VIII.1) predicts.  Replacing a rising rho by the constant rho_bar under
a kernel whose dynamic range is e^x costs a factor that grows with the
window, and the window is longest at the slowest ramp.

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
from rms_check import measure

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
        B = 0.25 * np.sinh(2 * s['x']) - 0.5 * s['x']
        out.append(dict(mdot=m, x=float(s['x']), fp=fp, fg=fg,
                        E_nom=rms(rb * k), E_true=rms(rb_true * k),
                        e=rms(s['e']), sqB=float(np.sqrt(B / s['x'])),
                        law=float(np.exp(s['x']) / s['x'])))
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'reduction_factors.png'
    r = collect()
    d2 = np.rad2deg
    md = [d['mdot'] for d in r]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '.failing_cache.pkl'), 'rb') as fh:
        import pickle
        rows = measure(pickle.load(fh))
    meas = [float(np.mean([q['rms_min'] for q in rows
                           if abs(q['rate'] - m) < 1e-6])) for m in md]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.2, 4.9))
    fig.subplots_adjust(left=0.056, right=0.99, top=0.825, bottom=0.135,
                        wspace=0.27)

    # ---- (a) everything normalised: what is flat and what is not ---
    en = np.array([d2(d['E_nom']) for d in r])
    ee = np.array([d2(d['e']) for d in r])
    mm = np.array(meas)
    a1.plot(md, en / en[0], 'o-', color=C_NOM, lw=2.2, ms=7,
            label=f'bound $\\mathrm{{RMS}}(E)$   ($\\times${en[0]/en[-1]:.1f})')
    a1.plot(md, ee / ee[0], '^-', color=C_TRUE, lw=2.4, ms=8,
            label=f'theory $\\mathrm{{RMS}}(e_\\omega)$'
                  f'   ($\\times${ee[0]/ee[-1]:.2f})')
    a1.plot(md, mm / mm[0], 's--', color='0.25', lw=2.0, ms=6.5,
            label=f'measured residual   ($\\times${mm[0]/mm[-1]:.2f})')
    a1.axhline(1.0, color='k', lw=0.9, ls=':')
    a1.set_yscale('log')
    a1.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9.5)
    a1.set_ylabel(r'relative to $\dot M = 0.10$', fontsize=9.5)
    a1.set_title('(a) the data and the theory agree\n'
                 'it is the bound that moves', fontsize=11)
    a1.legend(fontsize=8.5, loc='lower left')
    a1.grid(alpha=0.25, lw=0.4, which='both')

    # ---- (b) it is not the reduction factors -----------------------
    a2.axhline(RPHI, color=C_PHI, ls='--', lw=1.6,
               label=r'nominal $1/7$, gravity')
    a2.axhline(RGE, color=C_GE, ls='--', lw=1.6,
               label=r'nominal $1/5$, ground effect')
    a2.plot(md, [d['fp'] for d in r], 'o-', color=C_PHI, lw=2.2, ms=7,
            label=r'true $\langle\rho_\varphi\rangle/\rho_{\varphi,\max}$')
    a2.plot(md, [d['fg'] for d in r], 's-', color=C_GE, lw=2.2, ms=7,
            label=r'true $\langle\rho_{GE}\rangle/\rho_{GE,\max}$')
    a2.set_ylim(0, 0.24)
    a2.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9.5)
    a2.set_ylabel('window average / maximum', fontsize=9.5)
    a2.set_title('(b) not the reduction factors\n'
                 'they are flat, and $1/7$, $1/5$ bound them', fontsize=11)
    a2.legend(fontsize=8.5, loc='lower right')
    a2.grid(alpha=0.25, lw=0.4)

    # ---- (c) it is the propagation, and it has a closed form -------
    sq = np.array([d['sqB'] for d in r])
    law = np.array([d['law'] for d in r])
    a3.plot(md, sq / sq[0], 'o-', color=C_NOM, lw=2.2, ms=7,
            label=r'$\sqrt{B(x)/x}$, the propagation')
    a3.plot(md, (en / ee) / (en / ee)[0], 'd-', color='#7b3294', lw=2.2,
            ms=7, label=r'measured conservatism $E/e_\omega$')
    a3.plot(md, law / law[0], ':', color='k', lw=2.0,
            label=r'$e^{x}/x$, the $xe^{-x}$ law of (VIII.1)')
    a3.set_yscale('log')
    a3.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9.5)
    a3.set_ylabel(r'relative to $\dot M = 0.10$', fontsize=9.5)
    a3.set_title('(c) it is the propagation\n'
                 'and the conservatism follows its own law', fontsize=11)
    a3.legend(fontsize=8.5, loc='lower left')
    a3.grid(alpha=0.25, lw=0.4, which='both')

    fig.suptitle('The bound falls with the ramp rate; the deviation and the '
                 'data do not', fontsize=13, y=0.95)
    fig.savefig(out, dpi=150)

    print(f"\n  wrote {out}\n")
    print(f"  {'Mdot':>6}{'x':>6}{'factor phi':>12}{'factor GE':>11}"
          f"{'RMS(E)':>9}{'RMS(e)':>9}{'measured':>10}{'E/e':>7}")
    print(f"  {'N m/s':>6}{'':6}{'(1/7=.143)':>12}{'(1/5=.200)':>11}"
          f"{'deg/s':>9}{'deg/s':>9}{'deg/s':>10}{'':7}")
    for d, q in zip(r, meas):
        print(f"  {d['mdot']:6.2f}{d['x']:6.2f}{d['fp']:12.4f}{d['fg']:11.4f}"
              f"{d2(d['E_nom']):9.3f}{d2(d['e']):9.4f}{q:10.3f}"
              f"{d2(d['E_nom'])/d2(d['e']):7.1f}")
    print(f"\n  across the rate range")
    print(f"    bound RMS(E)          falls {en[0]/en[-1]:.2f}x")
    print(f"    of which sqrt(B(x)/x)       {sq[0]/sq[-1]:.2f}x   <- all of it")
    print(f"    theory RMS(e_omega)   falls {ee[0]/ee[-1]:.2f}x")
    print(f"    measured residual     falls {mm[0]/mm[-1]:.2f}x")
    print(f"    conservatism E/e      falls {(en/ee)[0]/(en/ee)[-1]:.2f}x,"
          f" against {law[0]/law[-1]:.2f} from the x e^-x law")
    print(f"\n  So the data and the theory agree that the residual does not")
    print(f"  depend on the ramp rate.  What depends on it is the bound, and")
    print(f"  its dependence is the known conservatism of replacing a rising")
    print(f"  rho by a constant under a kernel of dynamic range e^x.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
