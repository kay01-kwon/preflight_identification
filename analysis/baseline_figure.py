#!/usr/bin/env python3
"""The three runs the pre-onset level moves most, data and both fits.

analysis/baseline_zero.py shows that pinning C = 0 moves 39 of 140
thresholds, always earlier and always low, and that which runs move is
predicted by |C|.  A table cannot say WHY, because the two candidate
explanations differ only in the shape of the pre-onset segment:

    a compliance rate      the vehicle rotates slowly as the load
                           shifts, at a rate set by Mdot, so the
                           pre-onset segment is FLAT and non-zero
    a late onset           the tip-over has already begun and C is
                           absorbing its first samples, so the segment
                           RISES into the onset

So this plots them.  The top row is the whole excitation window for
each run -- the measured rate, the fit as shipped with its baseline and
its onset, and the fit with C pinned to zero with its own onset.  The
bottom row magnifies the segment C is formed on, which is where the
question lives: at full scale the two fits are indistinguishable and
the interesting part is a couple of degrees per second.

Reads .baseline_cache.pkl, written by analysis/baseline_zero.py.

Usage: python analysis/baseline_figure.py [out.png]
"""
import os
import pickle
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
W_MIN = 30.08
C_DATA, C_FIT, C_ZERO, C_BASE = '0.25', '#2874a6', '#c0392b', '#148f77'


def trend(t, y):
    """Least-squares slope with its t statistic."""
    sl, ic = np.polyfit(t, y, 1)
    r = y - (sl * t + ic)
    se = r.std(ddof=2) / np.sqrt(np.sum((t - t.mean()) ** 2))
    return sl, ic, sl / se if se > 0 else np.nan


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'baseline_figure.png'
    with open(os.path.join(HERE, '.baseline_cache.pkl'), 'rb') as fh:
        rows = pickle.load(fh)
    d2 = np.rad2deg

    diff = [d for d in rows if d['zero']['j'] != d['fit']['j']]
    diff.sort(key=lambda z: -abs(z['zero']['mcrit'] - z['fit']['mcrit']))
    show = diff[:3]

    fig = plt.figure(figsize=(15.0, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.22,
                          left=0.055, right=0.99, top=0.845, bottom=0.075)

    for k, d in enumerate(show):
        s = float(d['sign']) or 1.0
        t, om = d['t'], d2(s * d['om'])
        a, b = d['fit'], d['zero']
        jj = a['j']
        sh = 1e3 * (abs(b['mcrit']) - abs(a['mcrit']))
        name = (d['case'].replace('case_', 'case ') + '/' + d['axis'] +
                f", $\\dot M = {d['rate']:.2f}$, "
                f"{'+' if d['sign'] > 0 else '$-$'}")

        # ---- top: the whole window --------------------------------
        ax = fig.add_subplot(gs[0, k])
        ax.axvspan(t[0], t[jj], color='0.90', zorder=0)
        ax.plot(t, om, '.', color=C_DATA, ms=3.6, label='measured', zorder=3)
        ax.plot(t, d2(s * a['pred']), '-', color=C_FIT, lw=2.0, zorder=4,
                label=f"fitted, $C = {d2(s * a['c']):+.2f}$")
        ax.plot(t, d2(s * b['pred']), '--', color=C_ZERO, lw=1.8, zorder=4,
                label=r'fitted, $C \equiv 0$')
        ax.axvline(t[jj], color=C_FIT, lw=1.1, ls=':', zorder=2)
        ax.axvline(t[b['j']], color=C_ZERO, lw=1.1, ls=':', zorder=2)
        ax.set_title(f"{name}\nonset ${b['j'] - jj:+d}$ samples,"
                     f" $M_{{\\rm crit}}\\;{sh:+.1f}$ mN$\\,$m"
                     f" (${sh / W_MIN:+.2f}$ mm)", fontsize=10)
        ax.set_xlabel('time in the excitation window [s]', fontsize=9)
        ax.set_ylabel(r'$\omega$ toward the fall [$^\circ$/s]', fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=8, loc='upper left')

        # ---- bottom: the segment C is formed on -------------------
        ax = fig.add_subplot(gs[1, k])
        ax.axhline(0.0, color=C_ZERO, lw=1.8, ls='--',
                   label=r'$C \equiv 0$ assumes this')
        ax.plot(t[:jj], om[:jj], '.', color=C_DATA, ms=4.0,
                label='measured')
        ax.axhline(d2(s * a['c']), color=C_BASE, lw=2.0,
                   label=f"fitted $C = {d2(s * a['c']):+.2f}$")
        if jj > 8:
            sl, ic, tt = trend(t[:jj], om[:jj])
            ax.plot(t[:jj], sl * t[:jj] + ic, '-', color=C_FIT, lw=1.6,
                    label=f"trend ${sl:+.2f}^\\circ$/s$^2$, $t = {tt:.1f}$")
        ax.axvline(t[jj], color=C_FIT, lw=1.1, ls=':')
        ax.set_title(f'the segment $C$ is formed on, {jj} samples',
                     fontsize=10)
        ax.set_xlabel('time in the excitation window [s]', fontsize=9)
        ax.set_ylabel(r'$\omega$ [$^\circ$/s]', fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=7.8, loc='upper left')

    fig.suptitle('Pinning the pre-onset level to zero: the three runs it '
                 'moves most\ntop, the whole window;  bottom, the segment '
                 '$C$ is formed on', fontsize=13, y=0.965)
    fig.savefig(out, dpi=145)

    print(f"\n  wrote {out}\n")
    print(f"  {'config':>10}{'Mdot':>7}{'C':>9}{'pre-onset slope':>18}"
          f"{'t':>7}{'samples':>9}{'shift':>9}")
    print(f"  {'':10}{'N m/s':>7}{'deg/s':>9}{'deg/s^2':>18}{'':7}{'':9}"
          f"{'mN.m':>9}")
    for d in show:
        a, b = d['fit'], d['zero']
        s = float(d['sign']) or 1.0
        jj = a['j']
        sl, _, tt = trend(d['t'][:jj], d2(s * d['om'][:jj]))
        print(f"  {d['case'].replace('case_', '') + '/' + d['axis']:>10}"
              f"{d['rate']:7.2f}{d2(s * a['c']):9.3f}{sl:18.3f}{tt:7.1f}"
              f"{jj:9d}{1e3 * (abs(b['mcrit']) - abs(a['mcrit'])):9.2f}")

    # the same question over all 39, not just the three drawn
    sl_all, t_all, flat = [], [], 0
    for d in diff:
        s = float(d['sign']) or 1.0
        jj = d['fit']['j']
        if jj < 12:
            continue
        sl, _, tt = trend(d['t'][:jj], d2(s * d['om'][:jj]))
        sl_all.append(sl)
        t_all.append(tt)
        flat += abs(tt) < 2
    sl_all, t_all = np.array(sl_all), np.array(t_all)
    print(f"\n  over all {len(sl_all)} runs that moved:")
    print(f"    pre-onset slope median {np.median(sl_all):+.3f} deg/s^2,"
          f" positive in {int((sl_all > 0).sum())}")
    print(f"    |t| < 2 (no resolvable trend) in {flat}, |t| >= 2 in"
          f" {len(t_all) - flat}")
    tail_test(rows)
    return 0


def tail_test(rows):
    """Is the segment flat, or does it rise into the onset?

    A linear trend is the wrong statistic for this: the segment carries
    a visible 10-20 Hz oscillation that inflates the naive t, and a rise
    concentrated in the last few samples is not linear anyway.  Compare
    the last fifth of the segment against the first four fifths instead.
    """
    d2 = np.rad2deg
    print(f"\n  the last fifth of the segment against the first four"
          f" fifths\n")
    print(f"  {'group':>22}{'n':>5}{'first 80%':>11}{'last 20%':>11}"
          f"{'rise':>9}{'x level':>10}{'rise > 2 se':>13}")
    print(f"  {'':22}{'':5}{'deg/s':>11}{'deg/s':>11}{'deg/s':>9}{'':10}"
          f"{'':13}")
    for tag, sel in (('moved when C -> 0',
                      lambda d: d['zero']['j'] != d['fit']['j']),
                     ('did not move',
                      lambda d: d['zero']['j'] == d['fit']['j'])):
        a = []
        for d in rows:
            if not sel(d):
                continue
            s = float(d['sign']) or 1.0
            jj = d['fit']['j']
            if jj < 25:
                continue
            y = d2(s * d['om'][:jj])
            m = int(0.8 * jj)
            lo, hi = y[:m], y[m:]
            se = lo.std(ddof=1) / np.sqrt(len(hi))
            a.append((lo.mean(), hi.mean(), hi.mean() - lo.mean(),
                      (hi.mean() - lo.mean()) / max(abs(lo.mean()), 1e-6),
                      (hi.mean() - lo.mean()) / se if se > 0 else 0.0))
        a = np.array(a)
        print(f"  {tag:>22}{len(a):5d}{np.median(a[:, 0]):11.3f}"
              f"{np.median(a[:, 1]):11.3f}{np.median(a[:, 2]):9.3f}"
              f"{np.median(a[:, 3]):10.2f}{int((a[:, 4] > 2).sum()):8d}"
              f"/{len(a)}")
    print(f"\n  The segment is neither zero nor flat: in both groups the")
    print(f"  last fifth is about double the first four fifths, and the")
    print(f"  rise clears two standard errors in three quarters of the runs")
    print(f"  that move.  So C, a constant, is a one-parameter summary of a")
    print(f"  rising signal, which is why fitting it moves the onset.")
    print(f"\n  It does NOT settle what the rise is.  A late onset predicts")
    print(f"  one -- the tip-over has begun and C absorbs its first samples")
    print(f"  -- but so does genuine pre-onset compliance, because the")
    print(f"  restoring stiffness collapses as the threshold is approached,")
    print(f"  so any load-driven rotation accelerates into it.  Separating")
    print(f"  them needs the SHAPE of the rise, not its presence.")


if __name__ == '__main__':
    sys.exit(main())
