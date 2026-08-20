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

  How tight is it?  The ratio of means runs 0.17 at the slowest ramp to
  0.53 at the fastest, and what moves is the CAP, not the residual.  The
  minimiser leaves 0.93 to 1.08 deg/s at every rate -- flat across a
  twelvefold change in Mdot, and the 95% intervals overlap throughout --
  while the cap falls from 5.78 to 1.98.  That
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
from scipy.stats import t as student

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_quality_bound import rho_bar
from failing_runs import split, amplitude_best

HERE = os.path.dirname(os.path.abspath(__file__))
PHI_BOX = np.deg2rad(5.0)
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
    n = len(rows)
    def stat(kk, rt):
        """Mean over the runs at that rate, with a 95% t interval."""
        v = np.array([d[kk] for d in rows if d['rate'] == rt])
        return v.mean(), student.ppf(0.975, len(v) - 1) * v.std(ddof=1) \
            / np.sqrt(len(v))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.125)
    i = np.arange(len(rates))
    e = np.array([stat('rms_E', r)[0] for r in rates])
    nn = np.array([stat('rms_n', r)[0] for r in rates])
    cap = np.array([stat('cap', r)[0] for r in rates])
    cap_ci = np.array([stat('cap', r)[1] for r in rates])
    mm = np.array([stat('rms_min', r)[0] for r in rates])
    mm_ci = np.array([stat('rms_min', r)[1] for r in rates])

    ax.bar(i - 0.20, e, 0.36, color=C_E,
           label=r'$\mathrm{RMS}(E)$, the modelling bound')
    ax.bar(i - 0.20, nn, 0.36, bottom=e, color=C_N,
           label=r'$\mathrm{RMS}(n)$, the in-window disturbance')
    ax.errorbar(i - 0.20, cap, yerr=cap_ci, fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, zorder=5)
    ax.bar(i + 0.20, mm, 0.36, color=C_MIN,
           label=r'measured $\mathrm{RMS}(r)$')
    ax.errorbar(i + 0.20, mm, yerr=mm_ci, fmt='none', ecolor='0.15',
                elinewidth=1.3, capsize=4, zorder=5)

    for k in range(len(rates)):
        ax.text(k + 0.20, mm[k] + mm_ci[k] + 0.16,
                f'{mm[k] / cap[k]:.2f}', ha='center', fontsize=9,
                color=C_MIN, weight='bold')
    ax.text(0.985, 0.545, 'fraction of the bound used',
            transform=ax.transAxes, ha='right', fontsize=9, color=C_MIN)

    ax.errorbar([], [], yerr=[], fmt='none', ecolor='0.15', elinewidth=1.3,
                capsize=4, label='mean of 20 runs, 95% CI')
    ax.set_xticks(i)
    ax.set_xticklabels([f'{r:.2f}' for r in rates], fontsize=9)
    ax.set_xlabel(r'$\dot M$ [N m/s]', fontsize=10)
    ax.set_ylabel(r'RMS [$^\circ$/s]', fontsize=10)
    ax.set_title(r'$\mathrm{RMS}(r)\leq\mathrm{RMS}(E)+\mathrm{RMS}(n)$'
                 f' on all {n} runs\n'
                 'the bound falls fivefold across the rates;'
                 ' the residual does not move', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.25, lw=0.4, axis='y')
    ax.set_ylim(0, max(cap + cap_ci) * 1.16)
    fig.savefig(out, dpi=150)

    print(f"\n  wrote {out}\n")
    print(f"  means over the 20 runs at each rate, 95% t intervals\n")
    print(f"  {'Mdot':>6}{'RMS(E)':>9}{'RMS(n)':>9}{'cap':>17}"
          f"{'minimiser':>18}{'deployed':>10}{'used':>7}"
          f"{'inside, min':>13}{'dep+term':>10}")
    print(f"  {'N m/s':>6}{'deg/s':>9}{'deg/s':>9}{'deg/s':>17}"
          f"{'deg/s':>18}{'deg/s':>10}{'':7}{'':13}{'':10}")
    tm = td = 0
    for r in rates:
        v = [d for d in rows if d['rate'] == r]
        im = sum(1 for d in v if d['rms_min'] <= d['cap'])
        idd = sum(1 for d in v if d['rms_dep'] <= d['cap'] + d['rms_amp'])
        tm += im
        td += idd
        c, cci = stat('cap', r)
        m, mci = stat('rms_min', r)
        print(f"  {r:6.2f}{stat('rms_E', r)[0]:9.3f}{stat('rms_n', r)[0]:9.3f}"
              f"{c:10.3f} +/-{cci:5.3f}{m:11.3f} +/-{mci:5.3f}"
              f"{stat('rms_dep', r)[0]:10.3f}{m / c:7.2f}"
              f"{im:10d}/{len(v)}{idd:7d}/{len(v)}")
    print(f"\n  minimiser {tm}/{n} inside, deployed with the amplitude term"
          f" {td}/{n};")
    print(f"  deployed without it"
          f" {sum(1 for d in rows if d['rms_dep'] <= d['cap'])}/{n}.")
    print(f"  The full residual is used throughout -- no frequency split")
    print(f"  enters the statement, only the noise estimate.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
