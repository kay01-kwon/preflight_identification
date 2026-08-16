#!/usr/bin/env python3
"""(VIII.3) in RMS, checked on the campaign, and drawn.

The statement is one line.  Writing w for the part of the record the
fitted family cannot represent, the minimiser was choosing among members
of that family and so cannot be farther from the data than the best of
them:

    RMS(r) = min_g RMS(y - g) <= RMS(w) + RMS(n),

and the nominal being one member gives RMS(w) <= RMS(e) <= RMS(E).

Two things are worth checking, and the second is the reason this file
exists rather than a line in the text.

  Does it hold?  For the minimiser -- which is what the inequality is
  about -- yes, on all 140 runs, with the FULL residual and no frequency
  split anywhere.  The 5 Hz split that Sec. VIII used to apply to the
  residual is now only a device for estimating the noise: above 5 Hz the
  fitted curve is smooth on the scale of the window and can produce
  nothing, so what is there is disturbance by construction.

  How tight is it?  The ratio runs 0.15 at the slowest ramp to 0.53 at
  the fastest, and what moves is the CAP, not the residual.  The
  minimiser leaves 0.88 to 1.03 deg/s at every rate -- flat across a
  twelvefold change in Mdot -- while RMS(E) falls from 5.5 to 1.1.  That
  is the x e^-x behaviour of (VIII.1) seen from the residual side: the
  bound is loosest where the window is longest.  The deployed
  residual does rise, 1.2 to 2.3 deg/s, but that is its amplitude error
  growing with C1 = K Mdot, not the physics.

The deployed fit is more constrained than the minimiser: it takes
C1 = K Mdot from calibration, so the nominal is not among its options
and the amplitude error survives.  Without that term it holds on 110 of
140; with it, on all 140.

Usage: python analysis/rms_check.py [out.png]
"""
import collections
import os
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_quality_bound import rho_bar
from failing_runs import split, amplitude_best

HERE = os.path.dirname(os.path.abspath(__file__))
PHI_BOX = np.deg2rad(10.0)
C_MIN, C_DEP, C_AMP, C_E, C_N = '#148f77', '#c0392b', '#e08214', '#2874a6', '0.55'


def measure(rows):
    """Per run: the cap, its parts, and the measured residual."""
    d2 = np.rad2deg
    for d in rows:
        tau, c2, k = d['tau'], d['c2'], d['k']
        jp = 1.0 / (k * c2 ** 2)
        rb, _, _ = rho_bar(d['axis'], PHI_BOX, d['dm_win'])
        E = rb * np.sinh(np.clip(c2 * tau, 0, 30)) / (jp * c2)
        u = np.cosh(np.clip(c2 * tau, 0, 30)) - 1.0
        w = np.gradient(tau)
        w[0] *= 0.5
        w[-1] *= 0.5
        T = float(w.sum())
        rms = lambda v: float(np.sqrt(np.sum(v ** 2 * w) / T))
        c1a, rf = amplitude_best(tau, d['om'], c2)
        _, hif = split(rf, d['dt'])
        d['rms_E'] = d2(rms(E))
        d['rms_n'] = d2(rms(hif))
        d['rms_amp'] = d2(abs(abs(c1a) - k * d['md_full']) * rms(u))
        d['rms_dep'] = d2(rms(d['r']))
        d['rms_min'] = d2(rms(rf))
        d['cap'] = d['rms_E'] + d['rms_n']
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'rms_check.png'
    with open(os.path.join(HERE, '.failing_cache.pkl'), 'rb') as fh:
        rows = measure(pickle.load(fh))
    rates = sorted({d['rate'] for d in rows})
    cmap = plt.get_cmap('viridis')
    col = {r: cmap(i / max(len(rates) - 1, 1)) for i, r in enumerate(rates)}
    n = len(rows)

    fig = plt.figure(figsize=(15.0, 4.9))
    gs = fig.add_gridspec(1, 3, wspace=0.26, left=0.052, right=0.99,
                          top=0.80, bottom=0.135)

    # ---- (a) measured against the cap ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    lo, hi = 0.5, 12.0
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.4, zorder=1,
            label='equality: measured $=$ cap')
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color='0.92', zorder=0)
    for d in rows:
        ax.plot(d['cap'], d['rms_min'], 'o', color=col[d['rate']], ms=4.6,
                mew=0, alpha=0.85, zorder=3)
        ax.plot(d['cap'] + d['rms_amp'], d['rms_dep'], '^',
                color=col[d['rate']], ms=4.6, mew=0, alpha=0.45, zorder=2)
    ax.plot([], [], 'o', color='0.3', ms=6, label='minimiser: 140/140')
    ax.plot([], [], '^', color='0.3', ms=6, alpha=0.5,
            label='deployed $+\\,|\\Delta C_1|$ term: 140/140')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r'cap $\mathrm{RMS}(E)+\mathrm{RMS}(n)$ [$^\circ$/s]',
                  fontsize=9)
    ax.set_ylabel(r'measured $\mathrm{RMS}(r)$ [$^\circ$/s]', fontsize=9)
    ax.set_title('(a) every run below the line\n'
                 'grey = the region the bound forbids', fontsize=10.5)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25, lw=0.4, which='both')
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(rates[0], rates[-1]))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.046)
    cb.set_label(r'$\dot M$ [N m/s]', fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # ---- (b) what the cap is made of, per rate ---------------------
    ax = fig.add_subplot(gs[0, 1])
    med = lambda kk, rt: np.median([d[kk] for d in rows if d['rate'] == rt])
    i = np.arange(len(rates))
    e = [med('rms_E', r) for r in rates]
    nn = [med('rms_n', r) for r in rates]
    am = [med('rms_amp', r) for r in rates]
    ax.bar(i, e, 0.62, color=C_E, label=r'$\mathrm{RMS}(E)$, the model')
    ax.bar(i, nn, 0.62, bottom=e, color=C_N,
           label=r'$\mathrm{RMS}(n)$, in-window')
    ax.bar(i, am, 0.62, bottom=np.array(e) + nn, color=C_AMP, alpha=0.85,
           label=r'$|\Delta C_1|\,\mathrm{RMS}(u)$, deployed only')
    ax.plot(i, [med('rms_min', r) for r in rates], 'o-', color=C_MIN,
            lw=2.0, ms=7, label='measured, minimiser')
    ax.plot(i, [med('rms_dep', r) for r in rates], 's--', color=C_DEP,
            lw=2.0, ms=6.5, label='measured, deployed')
    ax.set_xticks(i)
    ax.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=8.5)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel(r'RMS [$^\circ$/s]', fontsize=9)
    ax.set_title("(b) the cap falls; the minimiser's residual does not\n"
                 r'$\mathrm{RMS}(E)$ $5.5\to1.1$, measured'
                 r' $0.88\to1.03^\circ$/s', fontsize=10.5)
    ax.legend(fontsize=7.4, loc='upper right')
    ax.grid(alpha=0.25, lw=0.4, axis='y')

    # ---- (c) how tight, and what each term buys --------------------
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.RandomState(0)
    series = (('minimiser', lambda d: d['rms_min'] / d['cap'], C_MIN, 'o'),
              ('deployed, no $\\Delta C_1$ term',
               lambda d: d['rms_dep'] / d['cap'], C_DEP, 's'),
              ('deployed, with it',
               lambda d: d['rms_dep'] / (d['cap'] + d['rms_amp']),
               C_AMP, '^'))
    for k, (lab, f, cc, mk) in enumerate(series):
        x = np.array([d['rate'] for d in rows])
        x = x * (1.0 + 0.045 * (k - 1) + 0.02 * rng.uniform(-1, 1, len(x)))
        v = np.array([f(d) for d in rows])
        ax.plot(x, v, mk, color=cc, ms=4.0, mew=0, alpha=0.55)
        ax.plot(rates, [np.median([f(d) for d in rows if d['rate'] == r])
                        for r in rates], '-', color=cc, lw=2.2,
                label=f"{lab}: {int(sum(1 for d in rows if f(d) <= 1))}/{n}")
    ax.axhline(1.0, color='k', lw=1.6, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=9)
    ax.set_ylabel('measured / cap', fontsize=9)
    ax.set_title('(c) how much of the cap is used\n'
                 'tightest at the fast end, never reached', fontsize=10.5)
    ax.legend(fontsize=7.8, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4, which='both')

    fig.suptitle(r'(VIII.3) in RMS: $\mathrm{RMS}(r)\leq\mathrm{RMS}(E)'
                 r'+\mathrm{RMS}(n)$, on the 140-run campaign',
                 fontsize=13, y=0.955)
    fig.savefig(out, dpi=145)

    print(f"\n  wrote {out}\n")
    print(f"  {'Mdot':>6}{'RMS(E)':>9}{'RMS(n)':>9}{'cap':>8}"
          f"{'minimiser':>11}{'deployed':>10}{'|dC1| term':>12}"
          f"{'inside, min':>13}{'dep+term':>10}")
    print(f"  {'N m/s':>6}{'deg/s':>9}{'deg/s':>9}{'deg/s':>8}"
          f"{'deg/s':>11}{'deg/s':>10}{'deg/s':>12}{'':13}{'':10}")
    tm = td = 0
    for r in rates:
        v = [d for d in rows if d['rate'] == r]
        im = sum(1 for d in v if d['rms_min'] <= d['cap'])
        idd = sum(1 for d in v if d['rms_dep'] <= d['cap'] + d['rms_amp'])
        tm += im
        td += idd
        print(f"  {r:6.2f}{med('rms_E', r):9.3f}{med('rms_n', r):9.3f}"
              f"{med('rms_E', r) + med('rms_n', r):8.3f}"
              f"{med('rms_min', r):11.3f}{med('rms_dep', r):10.3f}"
              f"{med('rms_amp', r):12.3f}{im:10d}/{len(v)}"
              f"{idd:7d}/{len(v)}")
    print(f"\n  minimiser {tm}/{n} inside, deployed with the amplitude term"
          f" {td}/{n};")
    print(f"  deployed without it"
          f" {sum(1 for d in rows if d['rms_dep'] <= d['cap'])}/{n}.")
    print(f"  The full residual is used throughout -- no frequency split")
    print(f"  enters the statement, only the noise estimate.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
